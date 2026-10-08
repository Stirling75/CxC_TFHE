#!/usr/bin/env python3
"""Public-schedule research; no ciphertext evaluation or latency prediction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "source_snapshot"))
import cbs_variance_estimator as est

CONFIG = json.loads((ROOT / "source_snapshot/parameters.json").read_text())
CONTRACTS = {4: "chunk4x4-tree-digits", 8: "chunk8x8-direct-digits"}
WIDTHS = (16, 32, 64, 128, 256)
THREADS = (1, 2, 4, 8, 16, 32, 64)
BASE_PARTITION = est.chunk_terms_by_bound


def profile_for(bits):
    common = CONFIG["hybrid_common"]
    item = CONFIG[f"hybrid_{bits}x{bits}"]
    updates = {}
    for source, key, prefix in (
        (common, "normalizer_pbs", "pbs"),
        (common, "normalizer_key_switch", "ks"),
        (common, "automorphism", "auto"),
        (common, "scheme_switch", "ss"),
        (item, "selector_lift_pbs", "cbs_lift_pbs"),
        (item, "selector_lift_key_switch", "cbs_lift_ks"),
        (item, "circuit_bootstrap", "cbs"),
    ):
        updates[f"{prefix}_base_log"], updates[f"{prefix}_level"] = source[key]
    return replace(est.profiles()[common["tfhe_profile"]], **updates)


def primitive_for(bits):
    profile = profile_for(bits)
    primitive = est.primitive_vars(profile, 4, 2, "linear", "sage", "revhomtrace", "refined-cbs")
    return profile, est.align_source_split(primitive, CONTRACTS[bits])


def input_tail(group, profile, primitive):
    variance = est.chunk_input_var_source_aware(group) + primitive.normalizer_ks_var
    return est.pbs_input_log2_pfail(
        profile.lwe_n, est.Q, profile.poly_n, 0, est.delta_from_bits(4), variance,
        "gaussian", centered_binary_ms=True,
    )


def group_union(groups, profile, primitive):
    return est.log2_sum_exp(input_tail(g, profile, primitive) for g in groups if len(g) >= 3)


def source_balanced_partition(terms, cap, profile, primitive):
    """Reassign equal-type slots; preserve bounds, refreshed flags, and residuals.

    Only adopt the candidate if its modeled input-event union decreases.
    This is a deterministic heuristic, not a globally optimal partition.
    """
    original = BASE_PARTITION(terms, cap, True)
    active = [i for i, g in enumerate(original) if len(g) >= 3]
    if len(active) < 2 or not any(not t.refreshed for i in active for t in original[i]):
        return original
    pools = {}
    for i in active:
        for t in original[i]:
            pools.setdefault((t.bound, t.refreshed), []).append(t)
    candidate = [[] if i in active else list(g) for i, g in enumerate(original)]
    # Round-robin slot filling avoids concentrating all early choices in one bin.
    for slot in range(max(len(original[i]) for i in active)):
        for i in active:
            if slot >= len(original[i]):
                continue
            reference = original[i][slot]
            pool = pools[(reference.bound, reference.refreshed)]
            j = min(range(len(pool)), key=lambda j: (
                est.chunk_input_var_source_aware(candidate[i] + [pool[j]]), j
            ))
            candidate[i].append(pool.pop(j))
    assert all(not pool for pool in pools.values())
    if group_union(candidate, profile, primitive) < group_union(original, profile, primitive) - 1e-12:
        return candidate
    return original


def simulate(columns, digits, profile, primitive, cap=15, policy="baseline"):
    original = est.chunk_terms_by_bound
    if policy == "source-balanced":
        est.chunk_terms_by_bound = lambda ts, c, token_aware=False: source_balanced_partition(ts, c, profile, primitive)
    try:
        return est.simulate_normalization(
            columns, digits, primitive, profile, "height2", "refresh-digit", 4,
            4, 0, 2, 2, cap, normalizer_centered_ms=True,
            token_aware_normalizer=True, output_row_bits=4,
            lift_centered_ms=True, lift_box_centered=True, correlation_model="source-aware",
        )
    finally:
        est.chunk_terms_by_bound = original


def public_trace(columns, digits, profile, primitive, cap=15, policy="baseline"):
    columns = [list(c) for c in columns[:digits]]
    waves = []
    all_tails = []
    while any(len(c) > 2 for c in columns):
        before = sum(map(len, columns))
        next_columns = [[] for _ in range(digits)]
        jobs = []
        for q, current in enumerate(columns):
            if len(current) <= 2:
                next_columns[q].extend(current)
                continue
            groups = (source_balanced_partition(current, cap, profile, primitive)
                      if policy == "source-balanced" else BASE_PARTITION(current, cap, True))
            assert sorted(map(id, current)) == sorted(id(t) for g in groups for t in g)
            assert all(sum(t.bound for t in g) <= cap for g in groups)
            for group in groups:
                if len(group) < 3:
                    next_columns[q].extend(group)
                else:
                    jobs.append((q, group))
        tails = [input_tail(g, profile, primitive) for _, g in jobs]
        variances = [est.chunk_input_var_source_aware(g) for _, g in jobs]
        all_tails.extend(tails)
        for q, group in jobs:
            bound = sum(t.bound for t in group)
            digit = est.Term(3, primitive.normalizer_pbs_var, "norm-digit", True)
            carry = est.Term(bound // 4, primitive.normalizer_pbs_var, "norm-carry", True)
            if all(hasattr(t, "plain") for t in group):
                total = sum(t.plain for t in group)
                digit.plain, carry.plain = total % 4, total // 4
            next_columns[q].append(digit)
            if carry.bound > 0 and q + 1 < digits:
                next_columns[q + 1].append(carry)
        after = sum(map(len, next_columns))
        assert after < before
        waves.append({
            "wave": len(waves), "jobs": len(jobs), "terms_before": before,
            "terms_after": after, "max_height": max(map(len, columns)),
            "max_input_log2_var": math.log2(max(variances)),
            "wave_input_log2_pfail": est.log2_sum_exp(tails),
            "product_containing_jobs": sum(any(not t.refreshed for t in g) for _, g in jobs),
        })
        columns = next_columns
    return waves, columns, all_tails


def analyze(bits, width, cap=15, identified=False, policy="baseline", columns=None, lift_factor=1):
    profile, primitive = primitive_for(bits)
    if columns is None:
        columns = est.product_terms(width // 2, primitive, contract=CONTRACTS[bits], identify_operands=identified)
    stats = simulate(columns, width // 2, profile, primitive, cap, policy)
    trace, rows, tails = public_trace(columns, width // 2, profile, primitive, cap, policy)
    jobs = sum(w["jobs"] for w in trace)
    has_add = any(len(c) > 1 for c in rows)
    refresh = sum(not t.refreshed for c in rows for t in c) if has_add else 0
    seq, par, seq_calls, par_calls = est.tfhe_final_add_event_multipliers(width // 2)
    add_calls = max(seq_calls, par_calls) if has_add else 0
    assert stats.pbs == 2 * jobs + refresh + add_calls
    assert stats.rounds == len(trace) + int(refresh > 0) + int(has_add)
    union = est.log2_sum_exp([
        stats.lift_union_log2_pfail + math.log2(lift_factor),
        stats.normalizer_union_log2_pfail, stats.final_union_log2_pfail,
    ])
    row = {
        "chunk_bits": bits, "width": width, "cap": cap,
        "operands": "identified" if identified else "distinct", "policy": policy,
        "initial_terms": sum(len(c) for c in columns[:width // 2]),
        "waves": len(trace), "reduction_jobs": jobs, "reduction_pbs": 2 * jobs,
        "refresh_counter": refresh, "final_add_pbs_counter": add_calls,
        "modeled_total_pbs": stats.pbs, "final_terms": sum(map(len, rows)),
        "log2_pfail": union, "meets_target": union < -128,
        "reduction_log2_pfail": est.log2_sum_exp(tails),
        "max_chunk_log2_var": stats.max_chunk_input_log2_var,
        "lut_threshold_log2": stats.normalizer_threshold_log2,
    }
    for threads in THREADS:
        slots = sum(math.ceil(w["jobs"] / threads) for w in trace)
        row[f"pbs_slots_t{threads}"] = 2 * slots
        row[f"job_utilization_t{threads}"] = jobs / (threads * slots) if slots else 0
    return row, trace


def geometry(width, bits, b=2, capacity=2048):
    assert width % bits == 0 and bits % b == 0
    h, r = width // bits, bits // b
    output_digits = 2 * r
    required = (1 << (2 * bits)) * output_digits
    banks = 1 << max(0, ((required + capacity - 1) // capacity - 1).bit_length())
    high = banks.bit_length() - 1
    assert high <= 2 * bits
    gates = banks - 1 + 2 * bits - high
    lookups = h * (h + 1) // 2
    return dict(chunk_bits=bits, width=width, lookups=lookups, retained_terms=r * h * h,
                banks=banks, cmux_per_lookup=gates, cmux_total=gates * lookups,
                dense_table_body_bytes=banks * capacity * 8,
                gaussian_scope="current-profile" if bits in CONTRACTS else "not-evaluated")


def fused_columns(bits, width, count, identified=False):
    _, primitive = primitive_for(bits)
    columns = [[] for _ in range(width // 2)]
    for index in range(count):
        product = est.product_terms(width // 2, primitive, contract=CONTRACTS[bits], identify_operands=identified)
        for q in range(width // 2):
            for term in product[q]:
                term.shared = {(index,) + key: amplitude for key, amplitude in term.shared.items()}
                columns[q].append(term)
    return columns


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def separate_failure(bits, width, cap, identified, count):
    """Sufficient Gaussian union for m complete products and m-1 clean additions."""
    profile, primitive = primitive_for(bits)
    single, _ = analyze(bits, width, cap, identified)
    sequential, parallel, _, _ = est.tfhe_final_add_event_multipliers(width // 2)
    def add_union(multipliers):
        return est.log2_sum_exp(
            est.pbs_input_log2_pfail(profile.lwe_n, est.Q, profile.poly_n, 0,
                est.delta_from_bits(4), multiplier * primitive.normalizer_pbs_var + primitive.normalizer_ks_var,
                "gaussian", centered_binary_ms=True) for multiplier in multipliers
        )
    decode_bound = math.log2(width // 2) + est.centered_decode_log2_pfail(4, primitive.normalizer_pbs_var)
    add_bound = est.log2_sum_exp([max(add_union(sequential), add_union(parallel)), decode_bound])
    total = est.log2_sum_exp([
        single["log2_pfail"] + math.log2(count),
        add_bound + math.log2(count - 1) if count > 1 else -math.inf,
    ])
    return total, count * single["modeled_total_pbs"] + (count - 1) * single["final_add_pbs_counter"]


def fusion_capacity_search(output):
    rows = []
    for identified in (False, True):
        for count in (2, 4, 8):
            columns = fused_columns(8, 256, count, identified)
            for cap in range(9, 16):
                row, _ = analyze(8, 256, cap, identified, columns=columns, lift_factor=count)
                separate_bound, separate_calls = separate_failure(8, 256, cap, identified, count)
                row.update(products=count, separate_log2_pfail=separate_bound,
                           separate_meets_target=separate_bound < -128,
                           separate_then_add_pbs=separate_calls)
                rows.append(row)
            print(f"fusion capacity: m={count}, identified={identified}", flush=True)
    save_csv(output / "fusion_capacity_search.csv", rows)


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    capacity_rows, wave_rows, balance_rows, fusion_rows = [], [], [], []
    checks = []
    baseline = {}
    for bits in CONTRACTS:
        for identified in (False, True):
            for width in WIDTHS:
                for cap in range(9, 16):
                    row, trace = analyze(bits, width, cap, identified)
                    capacity_rows.append(row)
                    if cap == 15:
                        baseline[bits, width, identified] = row
                        key = "union_log2_p_fail_by_width" + ("_squaring" if identified else "")
                        expected = CONFIG[f"hybrid_{bits}x{bits}"][key][str(width)]
                        assert abs(row["log2_pfail"] - expected) < 1e-6
                        checks.append(f"recorded bound matched: {bits}x{bits}, W={width}, identified={identified}")
                        for wave in trace:
                            wave_rows.append({"chunk_bits": bits, "width": width, "identified": identified, **wave})
                print(f"capacity sweep: {bits}x{bits}, W={width}, identified={identified}", flush=True)
                candidate, trace = analyze(bits, width, 15, identified, "source-balanced")
                reference = baseline[bits, width, identified]
                for key in ("waves", "reduction_jobs", "refresh_counter", "final_add_pbs_counter", "modeled_total_pbs"):
                    assert candidate[key] == reference[key]
                assert candidate["log2_pfail"] <= reference["log2_pfail"] + 1e-10
                candidate["baseline_log2_pfail"] = reference["log2_pfail"]
                candidate["bound_gain_bits"] = reference["log2_pfail"] - candidate["log2_pfail"]
                balance_rows.append(candidate)
    for bits in CONTRACTS:
        for width in (64, 256):
            for identified in (False, True):
                single = baseline[bits, width, identified]
                for count in (1, 2, 4, 8):
                    columns = fused_columns(bits, width, count, identified)
                    row, _ = analyze(bits, width, 15, identified, columns=columns, lift_factor=count)
                    row["products"] = count
                    row["separate_then_add_pbs"] = (
                        count * single["modeled_total_pbs"]
                        + (count - 1) * single["final_add_pbs_counter"]
                    )
                    row["pbs_change_percent"] = 100 * (row["modeled_total_pbs"] / row["separate_then_add_pbs"] - 1)
                    fusion_rows.append(row)
                    print(f"fused: {bits}x{bits}, W={width}, m={count}, identified={identified}", flush=True)
    for name, rows in (("capacity_sweep", capacity_rows), ("wave_trace", wave_rows),
                       ("source_balancing", balance_rows), ("sum_of_products", fusion_rows)):
        save_csv(output / f"{name}.csv", rows)
    save_csv(output / "lookup_geometry.csv", [geometry(256, bits) for bits in (2, 4, 8, 16)])
    fusion_capacity_search(output)
    sources = {}
    for path in (ROOT / "source_snapshot").iterdir():
        if path.is_file():
            sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "verification.json").write_text(json.dumps({
        "source_sha256": sources, "bound_reproductions": checks,
        "capacity_rows": len(capacity_rows), "balancing_rows": len(balance_rows),
        "fusion_rows": len(fusion_rows), "fusion_capacity_rows": 42,
        "bounds_are_gaussian_model_estimates": True,
        "no_ciphertext_or_latency_benchmark": True,
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    run(args.output)
