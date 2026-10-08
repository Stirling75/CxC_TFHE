#!/usr/bin/env python3
"""Algebra and analytic Gaussian checks, without ciphertext evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from array import array
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import baseline_study as base

ROOT = Path(__file__).resolve().parent
est = base.est
A = (2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, -1, -2)
FD = (-1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 1)
FK = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0)


def bits_msb(value, count=8):
    return tuple((value >> (count - 1 - i)) & 1 for i in range(count))


def decode_byte(index):
    bits = bits_msb(index)
    return sum(((bits[i] ^ bits[i + 4]) + 2 * bits[i + 4]) << (2 * i)
               for i in range(4))


def table_bank():
    bank = []
    for x in range(256):
        row = array("B")
        for y in range(256):
            product = decode_byte(x) * decode_byte(y)
            row.extend((product >> (2 * t)) & 3 for t in range(8))
        bank.append(row)
    return bank


def select_bank(bank, bits):
    for bit in bits:
        half = len(bank) // 2
        bank = bank[half:] if bit else bank[:half]
    return bank[0]


def anti(poly, index):
    turns, offset = divmod(index, len(poly))
    return (-1 if turns % 2 else 1) * poly[offset]


def rotate(poly, degree):
    return tuple(anti(poly, i + degree) for i in range(len(poly)))


def interleaved_lut(N):
    assert N % 64 == 0
    w, g = N // 16, N // 64
    accumulator = [None] * N
    for s in range(16):
        for center, value in ((w * s, s % 4), (w * s + w // 2, s // 4)):
            for e in range(-g, g):
                turns, idx = divmod(center + e, N)
                assert accumulator[idx] is None
                accumulator[idx] = (-1 if turns % 2 else 1) * value
    assert all(x is not None for x in accumulator)
    return accumulator


def ordinary_lut(values, N):
    assert N % (2 * len(values)) == 0
    w = N // len(values)
    accumulator = [None] * N
    for s, value in enumerate(values):
        for e in range(-w // 2, w // 2):
            turns, idx = divmod(w * s + e, N)
            assert accumulator[idx] is None
            accumulator[idx] = (-1 if turns % 2 else 1) * value
    return accumulator


def convolution(left, right):
    assert len(left) == len(right)
    n = len(left)
    out = [0] * n
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            out[(i + j) % n] += (-1 if i + j >= n else 1) * a * b
    return tuple(out)


def filtered_sample(poly, filt, degree):
    w = len(poly) // len(filt)
    return sum(c * anti(poly, degree - i * w) for i, c in enumerate(filt))


def cached_columns(width, identified):
    profile, primitive = base.primitive_for(8)
    d, h = width // 2, width // 8
    original = est.product_terms(d, primitive, contract=base.CONTRACTS[8],
                                 identify_operands=identified)
    columns = [[] for _ in original]
    positions = [0] * len(original)
    for u in range(h):
        for v in range(h - u):
            for t in range(8):
                q = 4 * (u + v) + t
                term = original[q][positions[q]]
                positions[q] += 1
                shared = dict(term.shared)
                shared[("cached-gadget", u)] = math.sqrt(8 * primitive.cmux_gadget_var)
                columns[q].append(replace(term,
                    private_var=8 * primitive.cmux_gadget_var, shared=shared))
    assert positions == list(map(len, original))
    return columns, profile, primitive


def cache_screens():
    rows = []
    for identified in (False, True):
        for width in base.WIDTHS:
            columns, p, v = cached_columns(width, identified)
            original = est.product_terms(width // 2, v, contract=base.CONTRACTS[8],
                                         identify_operands=identified)
            reference = base.simulate(original, width // 2, p, v)
            for policy in ("baseline", "source-balanced"):
                result = base.simulate(columns, width // 2, p, v, policy=policy)
                assert result.pbs == reference.pbs
                rows.append({"width": width, "squaring": identified, "partition": policy,
                    "baseline_log2_failure": reference.union_log2_pfail,
                    "cached_log2_failure": result.union_log2_pfail,
                    "modeled_pbs": result.pbs})
    return rows


@dataclass
class NoiseTerm:
    bound: int
    private: float
    shared: dict
    kind: str


def group_variance(terms):
    shared = {}
    for term in terms:
        for key, amplitude in term.shared.items():
            shared[key] = shared.get(key, 0.0) + amplitude
    return sum(t.private for t in terms) + sum(a * a for a in shared.values())


def tail(variance, profile, primitive, half_margin=False):
    return est.pbs_input_log2_pfail(profile.lwe_n, est.Q, profile.poly_n, 0,
        est.Q / (64 if half_margin else 32), variance + primitive.normalizer_ks_var,
        "gaussian", centered_binary_ms=True)


def final_add_union(d, profile, primitive):
    seq, par, ns, np = est.tfhe_final_add_event_multipliers(d)
    return max(est.log2_sum_exp(tail(m * primitive.normalizer_pbs_var, profile, primitive)
                               for m in path) for path in (seq, par)), max(ns, np)


def filter_screen(bits, width, identified, policy, cached=False):
    """Two-output variants with explicit output noise and a final-row adapter.

    Purely public simulation. Baseline partition shape is unchanged. Every
    surviving filtered output is individually identity-refreshed before the
    existing final adder, so its nominal-input model is not reused silently.
    """
    p, v = base.primitive_for(bits)
    filter_levels = 2 if policy.endswith("-12x2") else 1
    filter_variance = v.normalizer_pbs_var
    if filter_levels == 2:
        filter_variance = est.get_var_pbs(p.poly_n, p.glwe_k, p.lwe_n, est.Q,
            p.glwe_var, 12, 2) + est.get_var_fft_pbs(p.poly_n, p.glwe_k, p.lwe_n, 12, 2)
    d = width // 2
    if cached:
        assert bits == 8
        source, _, _ = cached_columns(width, identified)
    else:
        source = est.product_terms(d, v, contract=base.CONTRACTS[bits], identify_operands=identified)
    columns = [[NoiseTerm(t.bound, t.private_var, dict(t.shared), "product") for t in col]
               for col in source[:d]]
    wave, jobs_total, br_count, filtered_jobs, top_jobs, eligible_api = 0, 0, 0, 0, 0, 0
    tails = []
    raw_max_units = 0.0
    while any(len(col) > 2 for col in columns):
        output = [[] for _ in range(d)]
        jobs = []
        for q, col in enumerate(columns):
            if len(col) <= 2:
                output[q].extend(col)
                continue
            proxies = [est.Term(t.bound, group_variance([t]), str(i),
                                refreshed=t.kind != "product", private_var=t.private,
                                shared=dict(t.shared)) for i, t in enumerate(col)]
            groups = (base.source_balanced_partition(proxies, 15, p, v) if cached
                      else est.chunk_terms_by_bound(proxies, 15, True))
            for group in groups:
                members = [col[int(t.kind)] for t in group]
                if len(group) < 3:
                    output[q].extend(members)
                else:
                    jobs.append((q, members))
        for q, group in jobs:
            batch = (wave, jobs_total)
            jobs_total += 1
            bound = sum(t.bound for t in group)
            is_late = all(t.kind != "product" for t in group)
            use_filter = policy.startswith("filtered-all") or (policy == "filtered-late" and is_late)
            use_filter = use_filter and q < d - 1 and bound >= 4
            raw_var = group_variance(group)
            if is_late:
                raw_max_units = max(raw_max_units, raw_var / v.normalizer_pbs_var)
            batch_ids = [key for term in group for key in term.shared if key[0] == "filtered"]
            assert len(batch_ids) == len(set(batch_ids)), "unrefreshed siblings reconverged"
            tails.append(tail(raw_var, p, v))
            top_jobs += q == d - 1
            eligible_api += bound <= 7 and q < d - 1 and bound >= 4
            if use_filter:
                filtered_jobs += 1
                br_count += 1
                src = ("filtered", batch)
                digit = NoiseTerm(3, 0.0, {src: 6 * math.sqrt(filter_variance)}, "filtered")
                carry = NoiseTerm(bound // 4, 0.0, {src: math.sqrt(filter_variance)}, "filtered")
            else:
                br_count += 1 + (q < d - 1 and bound >= 4)
                digit = NoiseTerm(3, v.normalizer_pbs_var, {}, "ordinary")
                carry = NoiseTerm(bound // 4, v.normalizer_pbs_var, {}, "ordinary")
            output[q].append(digit)
            if q < d - 1 and carry.bound:
                output[q + 1].append(carry)
        assert sum(map(len, output)) < sum(map(len, columns))
        columns = output
        wave += 1
    adapter = 0
    if any(len(c) > 1 for c in columns):
        for col in columns:
            for term in col:
                if term.kind in ("product", "filtered"):
                    tails.append(tail(group_variance([term]), p, v))
                    adapter += 1
        add_union, add_calls = final_add_union(d, p, v)
        tails.append(add_union)
        decode = est.centered_decode_log2_pfail(4, v.normalizer_pbs_var)
        tails.append(decode + math.log2(d))
    else:
        add_calls = 0
        for col in columns:
            if not col:
                continue
            output_variance = group_variance(col)
            if policy != "ordinary" and col[0].kind in ("product", "filtered"):
                tails.append(tail(output_variance, p, v))
                adapter += 1
                output_variance = v.normalizer_pbs_var
            tails.append(est.centered_decode_log2_pfail(4, output_variance))
    reference = base.simulate(source, d, p, v)
    total = est.log2_sum_exp(tails + [reference.lift_union_log2_pfail])
    return {"chunk_bits": bits, "width": width, "squaring": identified, "policy": policy,
        "cached_product": cached,
        "reduction_jobs": jobs_total, "waves": wave, "filtered_jobs": filtered_jobs,
        "reduction_br": br_count, "top_column_jobs": top_jobs, "api_manylut_eligible_jobs": eligible_api,
        "adapter_pbs": adapter, "modeled_final_add_pbs": add_calls,
        "modeled_total_pbs": br_count + adapter + add_calls,
        "filtered_variance_ordinary_units": filter_variance / v.normalizer_pbs_var,
        "modeled_br_level_units": br_count + adapter + add_calls + filtered_jobs * (filter_levels - 1),
        "max_refreshed_group_variance_pbs_units": raw_max_units, "log2_failure": total}


def write_csv(name, rows):
    with (ROOT / "results" / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    (ROOT / "results").mkdir(exist_ok=True)
    counts = []
    for width in base.WIDTHS:
        h = width // 8
        H = h * (h + 1) // 2
        counts.append({"width": width, "chunk_pairs": H, "existing_cmux": 263 * H,
                       "reused_cmux": 255 * h + 8 * H})
    write_csv("cmux_counts.csv", counts)
    write_csv("cache_screens.csv", cache_screens())
    filters = [filter_screen(b, w, ident, policy) for b in (4, 8) for w in base.WIDTHS
               for ident in (False, True) for policy in ("ordinary", "filtered-late", "filtered-all", "filtered-all-12x2")]
    write_csv("filter_screens.csv", filters)
    combined = [filter_screen(8, w, ident, policy, cached=True) for w in base.WIDTHS
                for ident in (False, True) for policy in ("ordinary", "filtered-all-12x2")]
    write_csv("combined_screens.csv", combined)
    p, v = base.primitive_for(8)
    ms_rows = []
    for units in (0, 1, 3, 5, 15):
        variance = units * v.normalizer_pbs_var + (v.normalizer_ks_var if units else 0)
        ms_rows.append({"input_pbs_variance_units": units, "includes_ks": units != 0,
            "ordinary_log2_failure": est.pbs_input_log2_pfail(p.lwe_n, est.Q, p.poly_n, 0,
                est.Q / 32, variance, "gaussian", centered_binary_ms=True),
            "interleaved_log2_failure": est.pbs_input_log2_pfail(p.lwe_n, est.Q, p.poly_n, 0,
                est.Q / 64, variance, "gaussian", centered_binary_ms=True)})
    write_csv("interleaved_margin_screen.csv", ms_rows)
    metadata = {"analytic_gaussian_only": True, "empirical_floor": False,
        "source_sha256": {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                           for f in (ROOT / "source_snapshot").glob("*")},
        "primitive_log2_variances": {str(bits): {"gadget": math.log2(base.primitive_for(bits)[1].cmux_gadget_var),
            "selector_dependent": math.log2(base.primitive_for(bits)[1].cmux_key_var),
            "pbs": math.log2(base.primitive_for(bits)[1].normalizer_pbs_var),
            "ks": math.log2(base.primitive_for(bits)[1].normalizer_ks_var)} for bits in (4, 8)}}
    (ROOT / "results" / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("W256 screens")
    for row in filters + combined:
        if row["width"] == 256 and row["squaring"]:
            print(row)


if __name__ == "__main__":
    main()
