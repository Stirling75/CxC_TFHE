#!/usr/bin/env python3
"""Conditional Gaussian event screen for the independent revised-ST evaluator.

Refined primitive variances and decomposition/error factorization are modelling
assumptions, not derived independence of ciphertext-dependent decompositions.
Source reuse and within-selector trace covariance are screened explicitly.
No empirical variance floor or multiplicative safety factor is applied.
"""
import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "ring-variant-multiplier/model/source_snapshot"))
sys.path.insert(0, str(ROOT.parent / "shokri-revised-comparison"))
import cbs_variance_estimator as primitive
from reference import make_plan
from st_gaussian import log2_gaussian_outside, rounding_moments, key_error_variance

Q = 2**64
DELTA = 2**52
MODEL_ID = "st-cc2-conditional-envelope-20261001-v3"


def validate_parameters(p):
    if p["polynomial_size"] not in (1024, 2048) or p["glwe_dimension"] not in (1, 2):
        raise ValueError("unsupported ring shape")
    if not 0 < p["lwe_dimension"] < p["polynomial_size"] * p["glwe_dimension"]:
        raise ValueError("unsupported key dimensions")
    for key in ("lwe_sigma", "glwe_sigma"):
        if not math.isfinite(p[key]) or not 0 < p[key] < 1:
            raise ValueError("invalid noise standard deviation")
    for key in ("pbs", "ks", "auto", "ss", "cbs"):
        base, levels = p[key]
        if not all(isinstance(x, int) and x > 0 for x in (base, levels)) or base * levels >= 64:
            raise ValueError("invalid decomposition")
    if p["cbs"][1] != 4 or p.get("terminal_lut_count_log", 2) not in (1, 2):
        raise ValueError("unsupported terminal CBS packing")
    if not isinstance(p.get("cc2_big_key", False), bool):
        raise ValueError("cc2_big_key must be boolean")


def merge_sources(terms):
    out = {}
    for weight, sources in terms:
        for key, amplitude in sources.items():
            out[key] = out.get(key, 0.0) + abs(weight) * amplitude
    return out


def source_variance(sources):
    return math.fsum(x*x for x in sources.values())


def variances(p):
    n, k, small = p["polynomial_size"], p["glwe_dimension"], p["lwe_dimension"]
    glwe = p["glwe_sigma"]**2
    vpbs = primitive.get_var_pbs(n, k, small, Q, glwe, *p["pbs"])
    vpbs += primitive.get_var_fft_pbs(n, k, small, *p["pbs"])
    mean, rounding = rounding_moments(k*n, math.prod(p["ks"]))
    vks = rounding + key_error_variance(k*n, *p["ks"], p["lwe_sigma"])
    auto_crypto = primitive.get_var_glwe_ks(n, k, Q, glwe, *p["auto"])
    auto_fft = primitive.get_var_fft_glwe_ks(n, k, *p["auto"],
        2**40 if p["split_trace_fft"] else Q)
    # RevHomTrace (Lee-Yoon, TCHES 2026(1), Theorem 4) per coefficient; scheme
    # switching multiplies it by a binary key polynomial (N/2 per coefficient).
    trace_var = (4*math.log2(n)*primitive.get_var_modswitch_1bit(n, k)
                 + math.log2(n)*(auto_crypto + auto_fft))
    trace_sigma = math.sqrt(n/2*trace_var)
    base, levels = p["ss"]
    b = 2**base
    # Conditional iid rounding coordinates, with binary-secret norm bounds:
    # ||S_j||_2^2 + sum_i ||S_j*S_i||_2^2 <= N + k*N^3.
    ss_rounding = (Q*Q/b**(2*levels)-1)/12 * (n + k*n**3)
    ss_key = (k+1)*levels*n*(b*b+2)/12 * glwe*Q*Q
    ss_fft = primitive.get_var_fft_ext_prod(n, k, Q, *p["ss"])
    ss_sigma = math.sqrt(ss_rounding) + math.sqrt(ss_key) + math.sqrt(ss_fft)
    row = vpbs + trace_sigma**2 + ss_rounding + ss_key + ss_fft
    cmux = primitive.get_var_ext_prod(n, k, Q, row, *p["cbs"])
    cmux += primitive.get_var_fft_ext_prod(n, k, Q, *p["cbs"])
    return dict(pbs=vpbs, ks=vks, ks_mean=mean, trace_sigma=trace_sigma,
                ss_rounding=ss_rounding, ss_key=ss_key, ss_fft=ss_fft, ss_sigma=ss_sigma,
                selector_row_envelope=row, cmux=cmux, closed=vpbs+vks)


def split_high_rounding(p, conversions):
    if not p["split_trace_fft"]:
        return None
    n, k = p["polynomial_size"], p["glwe_dimension"]
    base, levels = p["auto"]
    single = primitive.get_var_fft_glwe_ks(n, 1, base, 1, 2**24)
    variance = (k*levels)**2 * single
    count = conversions * p["cbs"][1] * int(math.log2(n)) * (k+1) * n
    return {"stage": "split_high_integer_rounding", "variance_before_ms": variance,
            "ms_variance": 0.0, "mean_bound": 0.0, "margin": 0.5,
            "log2_p_fail": log2_gaussian_outside(0.5, 0.0, variance), "multiplicity": count}


def expected_runtime_counts(p, plan):
    """Runtime counters predicted from the Python plan, independently of Rust.

    The evaluator reads only the low CBS bit of the top column W-1: its high
    bit would feed column W, which is outside the lower-W output.
    Key switches: one per compressor, one per terminal column sum, and two per
    emission block (before the PBS and the closing KS). Input ciphertexts are
    already small-key, so lifts perform no KS.
    """
    width, n = plan.width, p["polynomial_size"]
    theta = p.get("terminal_lut_count_log", 2)
    compressors = sum(len(layer.jobs) for layer in plan.layers)
    reads = sum(1 + int(len(plan.final[c]) > 1 and c < width - 1) for c in range(8, width))
    terminal_cmux = sum(4 if c > 8 and len(plan.final[c-1]) > 1 else 2 for c in range(8, width))
    product_cmux = len(plan.products) * (65536 // n - 1 + int(math.log2(n)))
    return {"blind_rotations": width + compressors + (4 >> theta) * reads + width // 2,
            "key_switches": compressors + (width - 8)
                + (width // 2 + width if p.get("cc2_big_key", False) else width),
            "ggsw_conversions": 2 * width + reads,
            "terminal_binary_cbs": reads, "compressors": compressors,
            "product_cmux": product_cmux, "terminal_cmux": terminal_cmux,
            "cmux": product_cmux + terminal_cmux}


def event(p, stage, variance, theta=0, spacing=2**61, mean_bound=0.0, **metadata):
    n, small = p["polynomial_size"], p["lwe_dimension"]
    ms = 0.0
    margin = spacing/2
    if stage != "output_decode":
        ms, bias = primitive.centered_binary_ms_decision_noise(small, Q, 2*n/2**theta)
        # Reserve the offset of every extracted lane in a packed accumulator.
        margin -= (2**theta-1) * Q/(2*n)
        mean_bound += bias
    logp = log2_gaussian_outside(margin, mean_bound, variance+ms)
    return {"stage": stage, "variance_before_ms": variance, "ms_variance": ms,
            "margin": margin, "mean_bound": mean_bound, "log2_p_fail": logp, **metadata}


def screen(p, width):
    validate_parameters(p)
    if width not in (16, 32, 64, 128, 256):
        raise ValueError("unsupported width")
    plan = make_plan(width)
    v = variances(p)
    events = []
    big = p.get("cc2_big_key", False)
    # Section 3.1 allows twice the closed PBS-output variance at the input.
    # Small-key CC2 carries the closing KS through the 2^10 lift; big-key CC2
    # is lifted first and key-switched just before the blind rotation.
    for index in range(width):
        if big:
            events.append(event(p, "input_grouped_lift", 2**20 * 2*v["pbs"] + v["ks"],
                theta=2, spacing=Q/4, mean_bound=abs(v["ks_mean"]), index=index))
        else:
            events.append(event(p, "input_grouped_lift", 2**20 * 2*v["closed"],
                theta=2, spacing=Q/4, mean_bound=2**10 * 2*abs(v["ks_mean"]), index=index))
    amp = math.sqrt(v["cmux"])
    values = {}
    for index, product in enumerate(plan.products):
        # A selected scalar VP path has 16 selectors. Each pair is derived
        # from one grouped refresh, and is shared across incident products.
        values[index] = {(side, 4*chunk+block): 2*amp
            for side, chunk in (("x", product.left), ("y", product.right))
            for block in range(4)}
    for wave, layer in enumerate(plan.layers):
        for index, job in enumerate(layer.jobs):
            sources = merge_sources((1, values[i]) for i in job.inputs)
            events.append(event(p, "compressor", source_variance(sources)+v["ks"], theta=1,
                mean_bound=abs(v["ks_mean"]), wave=wave, column=job.column, index=index))
            refreshed = {("pbs", wave, index): math.sqrt(v["pbs"])}
            values[job.parity] = refreshed
            if job.carry is not None:
                values[job.carry] = refreshed.copy()
    for column in range(8, width):
        inputs = merge_sources((1, values[i]) for i in plan.final[column])
        # No high-bit read for the top column (it would only feed column W).
        for read in range(1 + int(len(plan.final[column]) > 1 and column < width - 1)):
            theta = p.get("terminal_lut_count_log", 2)
            for batch in range(4 // 2**theta):
                events.append(event(p, "terminal_binary_lift", source_variance(inputs)+v["ks"],
                    theta=theta, mean_bound=abs(v["ks_mean"]), column=column, read=read, batch=batch))
    # There are two CMux uses per low selector and two per preceding high
    # selector. Sum their amplitudes within each selector. U2=U0+C-A cancels
    # A exactly, so it does not introduce another independent state error.
    terminal_sources = {}
    for column in range(8, width):
        terminal_sources[("lo", column)] = 2*amp
        if column > 8 and len(plan.final[column-1]) > 1:
            terminal_sources[("hi", column-1)] = 2*amp
    for block in range(width//2):
        if block < 4:
            sources = merge_sources([(1, values[i]) for i in plan.final[2*block]]
                + [(2, values[i]) for i in plan.final[2*block+1]])
        else:
            sources = terminal_sources
        events.append(event(p, "emission_pbs", source_variance(sources)+v["ks"],
            mean_bound=abs(v["ks_mean"]), block=block))
        events.append(event(p, "output_decode", v["pbs"] if big else v["closed"], spacing=DELTA,
            mean_bound=0.0 if big else abs(v["ks_mean"]), block=block))
    counts = Counter(e["stage"] for e in events)
    runtime = expected_runtime_counts(p, plan)
    reference = plan.counts()
    assert runtime["compressors"] == reference["dadda_pbs_manylut"] == counts["compressor"]
    # The paper-level reference counts the unused top-column high read.
    assert runtime["terminal_binary_cbs"] == reference["terminal_binary_cbs"] - int(len(plan.final[-1]) > 1)
    terminal_multiplier = 4 // 2**p.get("terminal_lut_count_log", 2)
    assert counts["terminal_binary_lift"] == terminal_multiplier * runtime["terminal_binary_cbs"]
    assert runtime["blind_rotations"] == sum(counts[k] for k in
        ("input_grouped_lift", "compressor", "terminal_binary_lift", "emission_pbs"))
    assert len(events) == runtime["blind_rotations"] + width//2
    split = split_high_rounding(p, runtime["ggsw_conversions"])
    if split is not None:
        events.append(split)
        counts[split["stage"]] = split["multiplicity"]
    logs = [e["log2_p_fail"] + math.log2(e.get("multiplicity", 1)) for e in events]
    pivot = max(logs)
    union = pivot + math.log2(math.fsum(2**(x-pivot) for x in logs))
    worst = max(events, key=lambda e: e["log2_p_fail"])
    return {"width": width, "conditional_log2_union": union,
            "conditional_gaussian_screen_pass": union < -128,
            "dominant_event_stage": worst["stage"], "event_count": sum(counts.values()),
            "event_record_count": len(events),
            "stage_counts": dict(counts), "expected_runtime_counts": runtime,
            "variances": v, "events": events,
            "whole_multiplier_approved": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--widths", type=int, nargs="+", default=[16,32,64,128,256])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    p = json.loads(args.parameters.read_text())
    if any(w not in (16,32,64,128,256) for w in args.widths):
        parser.error("supported unsegmented widths are 16 through 256")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for width in args.widths:
        result = screen(p, width)
        events = result.pop("events")
        (args.output / f"events-w{width}.json").write_text(json.dumps(events, indent=2)+"\n")
        rows.append(result)
        print(width, result["conditional_log2_union"], result["dominant_event_stage"], flush=True)
    (args.output / "analysis.json").write_text(json.dumps({"parameters": p, "rows": rows,
        "model_id": MODEL_ID,
        "model_scope": __doc__, "inputs": "twice closed PBS variance, public scale 2^10",
        "assumptions": ["Gaussian event errors", "Refined primitive variance recurrences",
            "decomposition/error factorization in the external-product recurrence",
            "conditionally iid centered scheme-switch rounding coordinates",
            "separate source labels add in variance; reused/grouped source amplitudes add",
            "valid primitive FFT marginals; trace errors amplitude-enveloped; Split40 integer-rounding events unioned",
            "centered-MS uniform-residue heuristic, retaining the fixed-key bias envelope"],
        "open_issue": "Ciphertext-dependent decomposition and selector-error joint moments are not derived here.",
        "whole_multiplier_approved": False,
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), args.parameters, Path(primitive.__file__),
                         ROOT / "st_gaussian.py",
                         ROOT.parent / "shokri-revised-comparison/reference.py")}}, indent=2)+"\n")


if __name__ == "__main__":
    main()
