#!/usr/bin/env python3
"""Conditional Gaussian sensitivity on the actual fixed heterogeneous-ring plan.

Not a parameter certificate: external-product joint moments and the inherited
fresh-PBS/final-add models remain assumptions. No fitted floor or tail multiplier.
"""
import argparse
import csv
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "model"))
import baseline_study as base
import fused_model
import make_plan
import reassess_selectors

e = base.est


# Normalizer LUT cell.  TFHE-rs `generate_programmable_bootstrap_glwe_lut` (N=2048,
# message*carry=16) gives every value a box of 128 indices and rotates the table by
# half a box, so after the modulus switch to 2N=4096 (index step D=q/4096=2^52) a
# correct blind-rotation index lies in [-64, 63] around the message centre.  The
# body is rounded to the nearest index, so the phase perturbation must stay in
# [-64.5 D, +63.5 D).
NORMALIZER_MS_MODULUS = 4096
CELL_UPPER_STEPS = 63.5
CELL_LOWER_STEPS = 64.5


def normalizer_input_log2_pfail(lwe_n, variance):
    """Gaussian log2 Pr[index leaves the asymmetric normalizer LUT cell].

    `variance` is the pre-modulus-switch PBS-input variance (including key switch).
    The centered binary modulus-switch variance proxy and its worst-case mean are
    those of `pbs_input_log2_pfail`; the mean bound is removed from both sides.
    """
    additive, bias = e.centered_binary_ms_decision_noise(lwe_n, e.Q, NORMALIZER_MS_MODULUS)
    total = variance + additive
    if total <= 0.0:
        return float("-inf")
    step = e.Q / NORMALIZER_MS_MODULUS
    scale = math.sqrt(2.0 * total)
    upper = e.log2_erfc((CELL_UPPER_STEPS * step - bias) / scale) - 1.0
    lower = e.log2_erfc((CELL_LOWER_STEPS * step - bias) / scale) - 1.0
    return e.log2_sum_exp([upper, lower])


def ss_rounding_norm_bound(n, k):
    """||S_j||_2^2 + sum_i ||S_j*S_i||_2^2 <= N + k*N^3.

    Binary secrets and negacyclic convolution; variance use additionally assumes
    conditionally independent centered rounding coordinates. This is not a claim
    that the actual GLev decomposition error is independent of its row errors.
    """
    if n < 1 or k < 1:
        raise ValueError("positive ring dimensions required")
    return n + k * n**3


def primitive(n, auto, ss, lift_ks=(5, 3), *, chunk_bits=8, cbs=None):
    if chunk_bits not in (4, 8) or n not in (1024, 2048):
        raise ValueError("unsupported chunk width or CBS ring")
    normalizer = base.profile_for(chunk_bits)
    cbs = cbs if cbs is not None else (normalizer.cbs_base_log, 7)
    profile = replace(normalizer, poly_n=n, glwe_k=2048 // n,
        cbs_base_log=cbs[0], cbs_level=cbs[1],
        auto_base_log=auto[0], auto_level=auto[1], ss_base_log=ss[0], ss_level=ss[1],
        cbs_lift_ks_base_log=lift_ks[0], cbs_lift_ks_level=lift_ks[1])
    old = e.primitive_vars(profile, 4, 2, "linear", "sage", "revhomtrace", "refined-cbs")
    _, norm = base.primitive_for(chunk_bits)
    k, q = profile.glwe_k, e.Q
    auto_crypto = e.get_var_glwe_ks(n, k, q, profile.glwe_var, *auto)
    auto_fft = e.get_var_fft_glwe_ks(n, k, *auto, 2.0**40)
    # RevHomTrace (Lee-Yoon, TCHES 2026(1), Theorem 4): the trace adds at most
    # 4 log N V_MS + log N V_Auto per coefficient. Scheme switching multiplies
    # the trace output by a binary key polynomial (N/2 per coefficient, Refined
    # CBS composition). Compared with measured selector rows in
    # paper-scripts/noise_table.py.
    trace_var = (4 * math.log2(n) * e.get_var_modswitch_1bit(n, k)
                 + math.log2(n) * (auto_crypto + auto_fft))
    pbs = e.get_var_pbs(n, k, profile.lwe_n, q, profile.glwe_var,
        profile.cbs_lift_pbs_base_log, profile.cbs_lift_pbs_level)
    pbs += e.get_var_fft_pbs(n, k, profile.lwe_n,
        profile.cbs_lift_pbs_base_log, profile.cbs_lift_pbs_level)
    b = 2.0**ss[0]
    rounding = (q * q / b**(2 * ss[1]) - 1) / 12
    ss_round = rounding * ss_rounding_norm_bound(n, k)
    ss_key = (k + 1) * ss[1] * n * ((b * b + 2) / 12) * old.input_glwe_var
    ss_fft = e.get_var_fft_ext_prod(n, k, q, *ss)
    selector_var = pbs + (n / 2) * trace_var + ss_round + ss_key + ss_fft
    ext = e.get_var_ext_prod(n, k, q, selector_var, profile.cbs_base_log, profile.cbs_level)
    ext += e.get_var_fft_ext_prod(n, k, q, profile.cbs_base_log, profile.cbs_level)
    lift_input = 64 * norm.normalizer_pbs_var + e.get_var_lwe_ks(2048, q, profile.lwe_var, *lift_ks)
    updated = replace(old, cbs_lift_var=selector_var, cmux_ext_var=ext,
        cmux_key_var=ext - old.cmux_gadget_var,
        product_chunk2x2_var=8 * ext, product_chunk4x4_split_var=16 * ext,
        normalizer_pbs_var=norm.normalizer_pbs_var, normalizer_ks_var=norm.normalizer_ks_var,
        lift_input_var=lift_input)
    return profile, updated


def product_columns(plan, prim):
    width = plan["width"]
    if plan.get("public_scalar") is not None:
        return fused_model.public_columns(plan["public_scalar"], plan["public_groups"], prim)
    if plan["mode"] == "fused":
        return fused_model.columns(width, int(plan["env"]["FUSED_PRODUCTS_PER_GROUP"]), prim,
                                    shared_prefix=True)
    if plan["mode"] not in ("baseline", "cached"):
        raise ValueError("unsupported product mode")
    columns = e.product_terms(width // 2, prim, contract=base.CONTRACTS[plan["chunk_bits"]])
    if plan["mode"] == "cached":
        if plan["chunk_bits"] != 8:
            raise ValueError("cached products require eight-bit chunks")
        positions = [0] * len(columns)
        # The cached prefix has eight shared gadget errors; the suffix has eight private ones.
        for u in range(width // 8):
            for v in range(width // 8 - u):
                for t in range(8):
                    q = 4 * (u + v) + t
                    term = columns[q][positions[q]]
                    positions[q] += 1
                    term.private_var = 8 * prim.cmux_gadget_var
                    term.shared[("cached-gadget", u)] = math.sqrt(8 * prim.cmux_gadget_var)
        if positions != list(map(len, columns)):
            raise ValueError("cached product ordering mismatch")
    return columns[:width // 2]


def final_addition_pbs(digits):
    """PBS calls of the TFHE-rs 1.6.1 clean radix addition, per carry-propagation path."""
    _, _, sequential, parallel = e.tfhe_final_add_event_multipliers(digits)
    return {"sequential": sequential, "parallel": parallel}


def identical_operands(columns):
    """A=B: one ciphertext feeds both sides, so the selector sources of Y_j and X_j coincide."""
    out = []
    for column in columns:
        merged = []
        for t in column:
            t = replace(t, shared=None if t.shared is None else dict(t.shared))
            if t.shared:
                shared = {}
                for key, amp in t.shared.items():
                    key = ("X", key[1]) if key[0] == "Y" else key
                    shared[key] = shared.get(key, 0.0) + amp
                t.shared = shared
            merged.append(t)
        out.append(merged)
    return out


# Variance of the multi-value digit in units of V_PBS: 8 for uncorrelated extracted
# coefficients (squared L2 norm of the filter), 36 for the L1 envelope.
MVB_DIGIT_FACTOR = [8.0]


def linear_digit(group, vpbs, carry):
    """digit = sum - 4*carry (or the sum itself without a carry), not refreshed: the group's
    sources with amplitudes added, its private parts, and 16 V_PBS from the carry PBS."""
    private, shared = 0.0, {}
    for t in group:
        if t.refreshed:
            private += t.var
            continue
        private += t.private_var
        for key, amp in t.shared.items():
            shared[key] = shared.get(key, 0.0) + amp
    if carry:
        private += 16 * vpbs
    var = private + sum(a * a for a in shared.values())
    return e.Term(3, var, "linear-digit", False, private_var=private, shared=shared)


def replay(plan, prim, columns, decide=None):
    """Replay the public reduction waves. Returns (final columns, jobs per wave), a job being
    (q, group, carry, linear). With `decide(wave, q, group, carry, last, digit) -> bool` the
    linear flags are chosen; otherwise they are read from the plan ("linear", default False)."""
    digits = plan["width"] // 2
    vpbs = prim.normalizer_pbs_var
    if plan.get("mvb_digit_factor"):
        MVB_DIGIT_FACTOR[0] = plan["mvb_digit_factor"]
    jobs_per_wave = []
    for w, wave in enumerate(plan["waves"]):
        following = [[] for _ in range(digits)]
        jobs = []
        for q, record in enumerate(wave):
            current = columns[q]
            if decide is None:
                assert [(t.bound, t.refreshed) for t in current] == [
                    (t["bound"], t["refreshed"]) for t in record["terms"]]
            used = [i for indices in record["groups"] for i in indices]
            if sorted(used) != list(range(len(current))):
                raise ValueError("partition loses or duplicates a term")
            flags = record.get("linear") or [False] * len(record["groups"])
            mvb_flags = record.get("mvb") or [False] * len(record["groups"])
            for gi, indices in enumerate(record["groups"]):
                group = [current[i] for i in indices]
                if len(group) < 3:
                    following[q].extend(group)
                else:
                    jobs.append([q, group, gi, flags[gi], mvb_flags[gi]])
        # Term counts do not depend on the linear choice: the wave whose outputs
        # reach the final addition is known before choosing.
        counts = [len(c) for c in following]
        for q, group, _, _, _ in jobs:
            counts[q] += 1
            if sum(t.bound for t in group) // 4 and q + 1 < digits:
                counts[q + 1] += 1
        last = all(c <= 2 for c in counts)
        out = []
        for job in jobs:
            q, group, gi, linear, mvb = job
            bound = sum(t.bound for t in group)
            if not 0 < bound <= 15:
                raise ValueError("inadmissible normalizer group")
            carry = bound // 4 > 0 and q + 1 < digits
            if mvb:
                # One multi-value bootstrapping returns digit and carry; the digit is a
                # signed combination of five extracted coefficients of one blind rotation.
                if not carry:
                    raise ValueError("a multi-value group needs a retained carry")
                following[q].append(e.Term(3, MVB_DIGIT_FACTOR[0] * vpbs, "mvb-digit", True))
                following[q + 1].append(e.Term(bound // 4, vpbs, "carry", True))
                out.append((q, group, carry, False, gi, True))
                continue
            digit = linear_digit(group, vpbs, carry) if (carry or bound <= 3) else None
            if decide is not None:
                linear = digit is not None and bool(decide(w, q, group, carry, last, digit))
            if linear and digit is None:
                raise ValueError("a linear digit needs a retained carry or a bound below four")
            following[q].append(digit if linear else e.Term(3, vpbs, "normalizer", True))
            if carry:
                following[q + 1].append(e.Term(bound // 4, vpbs, "carry", True))
            out.append((q, group, carry, linear, gi, False))
        jobs_per_wave.append(out)
        columns = following
    return columns, jobs_per_wave


def assign_linear(plan, prim, cap_mid, cap_last):
    """Choose linear digits for both operand cases (AB and A=B) and rewrite the plan's term
    metadata. A digit is linear if its variance is at most cap*V_PBS in both cases, with
    cap_last for the wave that feeds the final addition; the final addition then reads the
    unrefreshed digits directly (`final_direct`)."""
    vpbs = prim.normalizer_pbs_var
    base = product_columns(plan, prim)
    decisions = {}
    def decide_ab(w, q, group, carry, last, digit):
        ok = digit.var <= (cap_last if last else cap_mid) * vpbs
        decisions[(w, len(decisions))] = ok
        return ok
    replay(plan, prim, base, decide_ab)
    # Same decisions must also hold for A=B: replay A=B with the AB flags removed where
    # its digit exceeds the cap (the replay order of jobs is identical).
    order = iter(sorted(decisions.items(), key=lambda kv: kv[0][1]))
    def decide_aa(w, q, group, carry, last, digit):
        _, ok = next(order)
        return ok and digit.var <= (cap_last if last else cap_mid) * vpbs
    _, jobs = replay(plan, prim, identical_operands(base), decide_aa)
    for w, wave_jobs in enumerate(jobs):
        for q, _, _, linear, gi, _ in wave_jobs:
            record = plan["waves"][w][q]
            record.setdefault("linear", [False] * len(record["groups"]))[gi] = linear
    # Rewrite term metadata of later waves and of the final row.
    columns = product_columns(plan, prim)
    digits = plan["width"] // 2
    for w, wave in enumerate(plan["waves"]):
        for q, record in enumerate(wave):
            record["terms"] = [{**meta, "refreshed": t.refreshed}
                               for meta, t in zip(record["terms"], columns[q])]
        columns, _ = replay({**plan, "waves": [wave]}, prim, columns)
    plan["final_columns"] = [[{**meta, "refreshed": t.refreshed}
                              for meta, t in zip(col_meta, col)]
                             for col_meta, col in zip(plan["final_columns"], columns)]
    plan["final_direct"] = True
    plan["linear_caps"] = [cap_mid, cap_last]
    return plan


def assign_mvb(plan, prim, factor):
    """Evaluate every reduction group with a retained carry by one multi-value
    bootstrapping (digit variance factor*V_PBS), keeping the linear-digit choice
    of the remaining groups, and rewrite the term metadata."""
    MVB_DIGIT_FACTOR[0] = factor
    digits = plan["width"] // 2
    columns = product_columns(plan, prim)
    for wave in plan["waves"]:
        for q, record in enumerate(wave):
            record["terms"] = [{**meta, "refreshed": t.refreshed} for meta, t in zip(record["terms"], columns[q])]
            mvb = []
            lin = record.get("linear") or [False] * len(record["groups"])
            for gi, indices in enumerate(record["groups"]):
                bound = sum(columns[q][i].bound for i in indices)
                flag = len(indices) >= 3 and bound // 4 > 0 and q + 1 < digits
                mvb.append(flag)
                if flag:
                    lin[gi] = False
            record["mvb"], record["linear"] = mvb, lin
        columns, _ = replay({**plan, "waves": [wave]}, prim, columns)
    plan["final_columns"] = [[{**meta, "refreshed": t.refreshed} for meta, t in zip(col_meta, col)]
                             for col_meta, col in zip(plan["final_columns"], columns)]
    plan["mvb_digit_factor"] = factor
    return plan


def model_variances(plan, prim, identical=False):
    """Pre-key-switch model variances in executor order: reduction inputs per wave and the
    block sums of the final addition (for the noise probe)."""
    columns = product_columns(plan, prim)
    if identical:
        columns = identical_operands(columns)
    final, jobs = replay(plan, prim, columns)
    waves = [[e.chunk_input_var_source_aware(g) for _, g, _, _, _, _ in wave] for wave in jobs]
    blocks = [e.chunk_input_var_source_aware(c) if c else 0.0 for c in final]
    return {"waves": waves, "final_blocks": blocks}


def analyze(plan, profile, prim, identical=False):
    width, digits = plan["width"], plan["width"] // 2
    capacity = int(plan["env"]["FUSED_PRODUCTS_PER_GROUP"])
    columns = product_columns(plan, prim)
    if identical:
        columns = identical_operands(columns)
    normalizer = base.profile_for(plan["chunk_bits"])
    if e.delta_from_bits(4) != 128 * e.Q / NORMALIZER_MS_MODULUS:
        raise ValueError("normalizer cell model expects 16 LUT boxes at N=2048")
    def input_tail(variance):
        return normalizer_input_log2_pfail(normalizer.lwe_n, variance + prim.normalizer_ks_var)
    reduction_tails = []
    reduction_pbs = 0
    columns, jobs = replay(plan, prim, columns)
    for wave in jobs:
        for q, group, carry, linear, _, mvb in wave:
            events = 1 if mvb else int(carry) if linear else 1 + int(carry)
            reduction_tails.extend([input_tail(e.chunk_input_var_source_aware(group))] * events)
            reduction_pbs += events
    assert [[(t.bound, t.refreshed) for t in col] for col in columns] == [
        [(t["bound"], t["refreshed"]) for t in col] for col in plan["final_columns"]]
    refresh_tails, addition, final_addition = [], float("-inf"), None
    if any(len(c) > 1 for c in columns):
        final_addition = final_addition_pbs(digits)
        seq, par, _, _ = e.tfhe_final_add_event_multipliers(digits)
        vpbs = prim.normalizer_pbs_var
        seq, par = [m * vpbs for m in seq], [m * vpbs for m in par]
        if plan.get("final_direct"):
            # Unrefreshed digits enter the addition directly: its first-layer inputs are
            # the block sums with their own variances; later layers read its PBS outputs.
            block = [e.chunk_input_var_source_aware(c) if c else 0.0 for c in columns]
            seq = [block[0]] + [b + vpbs for b in block[1:]]
            par = block + par[digits:]
        else:
            refresh_tails = [input_tail(e.term_var_source_aware(t)) for col in columns for t in col if not t.refreshed]
        addition = max(e.log2_sum_exp(input_tail(x) for x in seq),
                       e.log2_sum_exp(input_tail(x) for x in par))
        outputs = [prim.normalizer_pbs_var] * digits
    else:
        outputs = [e.chunk_input_var_source_aware(c) for c in columns]
    decoding = e.log2_sum_exp(e.centered_decode_log2_pfail(4, v, "gaussian") for v in outputs)
    # One lift per encrypted digit: 2d for two encrypted operands, d with a public one.
    lifts = width // 2 if (plan.get("public_scalar") is not None or identical) else width
    # The packed tables are read at offsets of up to cbs_level - 1 rotation
    # indices (of 2N / 2^theta, theta = 2), which shortens one side of the
    # q/8 selector margin; both sides are shortened here, conservatively.
    offset = (profile.cbs_level - 1) * e.Q * 4 / (2 * profile.poly_n)
    lift = math.log2(lifts) + e.pbs_input_log2_pfail(profile.lwe_n, e.Q, profile.poly_n, 2,
        2.0**62 - 2 * offset, prim.lift_input_var, "gaussian", centered_binary_ms=True)
    split = reassess_selectors.split_high_rounding(profile, width, 40)["union_log2"]
    reduction = e.log2_sum_exp(reduction_tails)
    refresh = e.log2_sum_exp(refresh_tails)
    union = e.log2_sum_exp([lift, reduction, refresh, addition, decoding, split])
    return {"width": width, "N_CBS": profile.poly_n, "k_CBS": profile.glwe_k,
        "N_normalizer": 2048, "k_normalizer": 1, "group_size": capacity,
        "auto": f"{profile.auto_base_log}x{profile.auto_level}",
        "ss": f"{profile.ss_base_log}x{profile.ss_level}",
        "lift_ks": f"{profile.cbs_lift_ks_base_log}x{profile.cbs_lift_ks_level}",
        "selector_log2_variance": math.log2(prim.cbs_lift_var),
        "lift_log2": lift, "reduction_log2": reduction, "refresh_log2": refresh,
        "addition_log2": addition, "decoding_log2": decoding, "split_high_log2": split,
        "conditional_union_log2": union, "conditional_screen_pass": union < -128,
        "approved": False, "reduction_pbs": reduction_pbs, "row_refresh": len(refresh_tails),
        "final_addition_pbs": final_addition,
        "normalizer_cell": [-CELL_LOWER_STEPS, CELL_UPPER_STEPS]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--widths", type=int, nargs="+", default=[256])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for n, capacity, lift_ks in ((2048, 32, (7, 2)), (1024, 28, (5, 3))):
        plans = {w: make_plan.derive(w, "fused", capacity=capacity, cbs_level=7,
                    kernel="reuse-cache", auto_fft="split40") for w in args.widths}
        for auto in ((9, 5), (8, 6), (7, 7), (6, 8), (5, 9), (5, 10), (4, 12), (3, 16)):
            for ss in ((13, 3), (13, 4)):
                profile, prim = primitive(n, auto, ss, lift_ks)
                for width, plan in plans.items():
                    row = analyze(plan, profile, prim)
                    records.append(row)
                    if row["conditional_screen_pass"]:
                        print(n, width, row["auto"], row["ss"], row["conditional_union_log2"])
        for width, plan in plans.items():
            (args.output / f"plan-N{n}-W{width}.json").write_text(json.dumps(plan, indent=2) + "\n")
    with (args.output / "conditional-screen.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (args.output / "scope.json").write_text(json.dumps({
        "approved": False, "gaussian_only": True, "empirical_floor": False,
        "normalizer_ring_is_unchanged": True, "partition": "actual inherited public plan, not retuned by this screen",
        "remaining_assumptions": [
            "Primitive Gaussian and FFT marginal estimates",
            "Conditional iid scheme-switch rounding coordinates",
            "Inherited external-product second moments: missing joint D_i D_j U_i U_j control",
            "Source amplitudes of the selected-path product-sum kernel",
            "Fresh-PBS and final-addition variance model"
        ],
        "warning": "Passing this conditional screen is not an established whole-multiplier bound or permission to claim target-128.",
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), *sorted((ROOT / "model").glob("*.py")),
                      ROOT / "model/source_snapshot/cbs_variance_estimator.py"]},
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
