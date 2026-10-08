#!/usr/bin/env python3
"""Recompute the candidate's conditional noise and conventional security screens."""
import argparse
import json
import math
from pathlib import Path

import analyze_noise as noise

HERE = Path(__file__).resolve().parent
SMALL = HERE / "results/security-search.json"
RING = HERE.parents[1] / "results/st-retune-security.json"
ESTIMATOR_COMMIT = "6019056011d10d7e9c30a0d5da2d2f729fbc2eec"
ATTACKS = {"arora-gb", "bkw", "usvp", "bdd", "bdd_hybrid", "bdd_mitm_hybrid",
           "dual", "dual_hybrid"}
SECURITY_SHA256 = {
    SMALL: "97884b3d503e11eff62700aaeda66dff66f4e5231d4d47121dd0eedbc9d91a1c",
    RING: "d9767a461147be60022684fe4bd48aa2397250918287876c98eecac3fc8b36ae",
}


def security_row(report, dimension, sigma):
    if report["estimator_commit"] != ESTIMATOR_COMMIT:
        raise ValueError("unexpected lattice-estimator version")
    rows = [r for r in report["rows"] if r["n"] == dimension and r["sigma"] == sigma]
    if len(rows) != 1:
        raise ValueError("missing or ambiguous exact-parameter security row")
    row = rows[0]
    if not (row["full_estimator"] and row["sample_model"] == "unlimited"
            and row["secret_distribution"] == "binary"
            and row["error_distribution"] == "discrete Gaussian"):
        raise ValueError("security model mismatch")
    if set(row["attacks_rop_log2"]) != ATTACKS:
        raise ValueError("incomplete attack set")
    costs = [float(v) for v in row["attacks_rop_log2"].values()]
    if any(math.isnan(v) for v in costs):
        raise ValueError("invalid attack estimate")
    minimum = min(costs)
    if not math.isfinite(minimum) or minimum < 128:
        raise ValueError("classical security screen below 128 bits")
    if abs(minimum-row["minimum_rop_log2"]) > 1e-9:
        raise ValueError("recorded security minimum mismatch")
    return row | {"attacks_rop_log2": {
        name: value if math.isfinite(value) else "infinity"
        for name, value in zip(row["attacks_rop_log2"], costs)}}


def verify(parameters):
    for path, expected in SECURITY_SHA256.items():
        if noise.digest(path) != expected:
            raise ValueError(f"security report hash mismatch: {path}")
    for kind in ("pbs", "encoding_pbs", "ks", "packing", "relin"):
        base, levels = parameters[kind]
        if not (isinstance(base, int) and isinstance(levels, int)
                and base > 0 and levels > 0 and base*levels < 64):
            raise ValueError("unsupported decomposition")
    # This verifier intentionally covers only the tested k=1, full-width layout.
    if not (parameters["input_lut_domain"] == 4
            and parameters["normalizer_lut_domain"] == 8 and parameters["degree_aware"]
            and parameters["polynomial_size"] == 2048):
        raise ValueError("unsupported construction; derive and test it separately")
    security = {
        "small_lwe": security_row(json.loads(SMALL.read_text()), parameters["n"],
                                  parameters["lwe_sigma"]),
        "glwe_lwe_proxy": security_row(json.loads(RING.read_text()),
                                       parameters["polynomial_size"], parameters["glwe_sigma"])}
    rows = []
    for width in (8, 16, 32, 64, 128, 256):
        row = noise.screen(parameters, noise.make_plan(width, True, 8), True, False)
        if not math.isfinite(row["raw_log2_union"]) or row["raw_log2_union"] >= -128:
            raise ValueError(f"conditional whole-operation screen failed at W={width}")
        rows.append({k: row[k] for k in ("width", "events", "raw_log2_union",
                    "families", "worst_event", "conditional_model_target_pass")})
    return {"parameters": parameters, "security": security, "rows": rows,
            "security_report_sha256": {path.name: digest for path, digest in SECURITY_SHA256.items()},
            "conditional_preflight_pass": True, "whole_multiplier_approved": False,
            "scope": "general XY; clean radix-4 input/output; no coefficient padding",
            "limitations": ["Gaussian and variance-additive primitive heuristic",
                "Layout-aware CLOT C.1 coefficient-sum adaptation, not the dense envelope",
                "Existing FFT model, not a certified tail bound for this encoding BSK",
                "GLWE security uses a conventional LWE proxy",
                "Secret-dependent relinearization/evaluation-key security is not newly proved"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=Path,
                        default=HERE / "parameters/bfv-n800-range-aware.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a new output file")
    result = verify(json.loads(args.parameters.read_text()))
    sources = [args.parameters, SMALL, RING, Path(__file__), Path(noise.__file__),
               noise.SNAPSHOT / "cbs_variance_estimator.py"]
    result["sha256"] = {str(p.resolve()): noise.digest(p) for p in sources}
    with args.output.open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
    for row in result["rows"]:
        print(f"W={row['width']}: log2 conditional union={row['raw_log2_union']:.6f}")
    print("Conditional preflight passed; not an unconditional certificate.")


if __name__ == "__main__":
    main()
