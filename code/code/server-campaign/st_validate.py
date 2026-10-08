#!/usr/bin/env python3
"""Cross-check candidate parameters, model events, security, and raw pilot rows."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import statistics

import st_retune as st

ROOT = Path(__file__).resolve().parent
EVALUATOR = ROOT.parent / "ring-variant-multiplier"
EXPECTED_ATTACKS = {"arora-gb", "bkw", "usvp", "bdd", "bdd_hybrid", "bdd_mitm_hybrid", "dual", "dual_hybrid"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def attack_log2(value):
    """Estimator cost as a float. The retained report stores JSON Infinity
    (float inf after json.loads); st_security.sage.py writes "infinity"."""
    if isinstance(value, str):
        if value.strip().lower() in ("infinity", "+infinity", "inf", "+inf"):
            return math.inf
        raise ValueError(f"unexpected attack cost {value!r}")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"unexpected attack cost {value!r}")
    return float(value)


def check_security_row(row):
    attacks = {k: attack_log2(v) for k, v in row["attacks_rop_log2"].items()}
    assert set(attacks) == EXPECTED_ATTACKS
    assert not any(math.isnan(v) or v == -math.inf for v in attacks.values())
    finite = [v for v in attacks.values() if math.isfinite(v)]
    assert finite and min(finite) == row["minimum_rop_log2"]
    return {k: v if math.isfinite(v) else "infinity" for k, v in attacks.items()}


def check_counts(runtime, expected):
    """Runtime counters against the Python-derived plan counts (independent of Rust)."""
    for counter, key in (("blind_rotations", "blind_rotations"), ("key_switches", "key_switches"),
                         ("conversions", "ggsw_conversions"), ("cmux", "cmux")):
        assert runtime[counter] == expected[key], counter


def parameter_roundoff(actual, declared):
    # The runtime always reports cc2_big_key; older parameter files omit it (false).
    declared = {"cc2_big_key": False, **declared}
    actual = {"cc2_big_key": False, **actual}
    assert actual.keys() == declared.keys()
    differences = {}
    for key in declared:
        if key in ("lwe_sigma", "glwe_sigma"):
            # The Rust JSON parser may round a decimal one binary64 ULP away
            # from Python. Accept only this representation error, and expose it.
            assert abs(actual[key] - declared[key]) <= math.ulp(declared[key]), key
            if actual[key] != declared[key]:
                differences[key] = {"runtime": actual[key], "declared": declared[key]}
        else:
            assert actual[key] == declared[key], key
    return differences


def check_plan(directory, width):
    actual = json.loads((directory / "plan.json").read_text())
    reference = st.make_plan(width)
    assert actual["products"] == [asdict(p) for p in reference.products]
    assert actual["initial"] == list(map(list, reference.initial))
    assert actual["final_columns"] == list(map(list, reference.final))
    assert len(actual["layers"]) == len(reference.layers)
    for layer, expected in zip(actual["layers"], reference.layers):
        assert layer["target"] == expected.target
        assert layer["after"] == list(map(list, expected.after))
        assert layer["jobs"] == [{"inputs": list(j.inputs), "parity": j.parity, "carry": j.carry}
                                  for j in expected.jobs]
    return reference


def check_run(directory, models):
    metadata = json.loads((directory / "parameters.json").read_text())
    provenance = json.loads((directory / "provenance.json").read_text())
    params = metadata["parameter_set"]
    roundoff = parameter_roundoff(params, models[params["name"]]["parameters"])
    assert metadata["boundary_delta_log2"] == 52 and metadata["paper_CC2_boundary"]
    assert metadata["whole_multiplier_failure_bound"] is None
    assert provenance["exit_code"] == 0
    for name, expected in provenance["source_sha256"].items():
        assert sha(EVALUATOR / name) == expected, name
    assert sha(EVALUATOR / "Cargo.lock") == provenance["cargo_lock_sha256"]
    plan = check_plan(directory, metadata["width"])
    model = st.screen(params, metadata["width"])
    expected = model["expected_runtime_counts"]
    for key, python_key in (("total_blind_rotations", "blind_rotations"), ("key_switches", "key_switches"),
            ("ggsw_conversions", "ggsw_conversions"), ("terminal_binary_cbs", "terminal_binary_cbs"),
            ("product_cmux", "product_cmux"), ("terminal_cmux", "terminal_cmux")):
        assert metadata["counts"][key] == expected[python_key], key
    with (directory / "timings.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == metadata["repetitions"] + metadata["warmup"]
    assert [int(r["trial"]) for r in rows] == list(range(len(rows)))
    for index, row in enumerate(rows):
        assert (row["warmup"] == "true") == (index < metadata["warmup"])
        assert row["ok"] == "true" and int(row["output_errors"]) == 0 and int(row["post_pbs_errors"]) == 0
        for key in ("product_errors", "reduction_errors", "pre_emission_errors"):
            assert row[key] == "0" if metadata["inspect"] else row[key] == ""
        counts = json.loads(row["counts"])
        check_counts(counts, expected)
        assert counts["blind_rotations"] == metadata["counts"]["total_blind_rotations"]
        assert counts["conversions"] == metadata["counts"]["ggsw_conversions"]
        assert counts["cmux"] == metadata["counts"]["product_cmux"] + metadata["counts"]["terminal_cmux"]
        # Equality only checks CSV integrity (the evaluator writes phase_sum_s as
        # this same sum); the nesting check phase_sum <= total_s is informative.
        phase_sum = sum(float(row[k]) for k in
            ("lift_s", "product_s", "reduction_s", "terminal_s", "emission_s"))
        assert math.isclose(phase_sum, float(row["phase_sum_s"]), rel_tol=1e-12)
        assert phase_sum <= float(row["total_s"]) + 1e-9
    c = model["stage_counts"]
    assert counts["blind_rotations"] == sum(c.get(k, 0) for k in
        ("input_grouped_lift", "compressor", "terminal_binary_lift", "emission_pbs"))
    assert metadata["counts"]["product_cmux"] == len(plan.products) * (
        65536 // params["polynomial_size"] - 1 + params["polynomial_size"].bit_length() - 1)
    measured = [float(row["total_s"]) for row in rows if row["warmup"] == "false"]
    assert len(measured) == metadata["repetitions"]
    return {"path": str(directory), "preset": params["name"], "width": metadata["width"],
        "parameter_decimal_roundoff_at_most_one_ulp": roundoff,
        "threads": metadata["threads"], "inspect": metadata["inspect"], "pattern": metadata["pattern"],
        "correct_trials": len(rows), "measured_trials": len(measured),
        "mean_total_s": statistics.mean(measured),
        "sample_sd_s": statistics.stdev(measured) if len(measured) > 1 else None,
        "timing_usable_as_local_pilot": not metadata["inspect"],
        "timings_sha256": sha(directory / "timings.csv")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    models = {}
    for directory in args.models:
        model = json.loads((directory / "analysis.json").read_text())
        assert model["model_id"] == st.MODEL_ID and not model["whole_multiplier_approved"]
        for name, expected in model["source_sha256"].items():
            path = Path(name)
            assert sha(path if path.is_absolute() else ROOT / path) == expected, name
        for row in model["rows"]:
            checked = st.screen(model["parameters"], row["width"])
            assert checked["conditional_gaussian_screen_pass"]
            assert math.isclose(row["conditional_log2_union"], checked["conditional_log2_union"], abs_tol=1e-9)
            events = json.loads((directory / f"events-w{row['width']}.json").read_text())
            assert events == checked["events"]
        models[model["parameters"]["name"]] = model
    security_path = ROOT / "results/st-retune-security.json"
    security = json.loads(security_path.read_text())
    assert security["estimator_commit"] == "6019056011d10d7e9c30a0d5da2d2f729fbc2eec"
    assert len(security["rows"]) == 5
    for row in security["rows"]:
        assert row["full_estimator"] and row["sample_model"] == "unlimited"
        row["attacks_rop_log2"] = check_security_row(row)
    for model in models.values():
        p = model["parameters"]
        for n, sigma in ((p["lwe_dimension"], p["lwe_sigma"]),
            (p["polynomial_size"] * p["glwe_dimension"], p["glwe_sigma"])):
            candidates = [r for r in security["rows"] if r["n"] == n and r["sigma"] == sigma]
            assert len(candidates) == 1 and candidates[0]["minimum_rop_log2"] >= 128
    for root in args.runs:
        assert root.is_dir() and list(root.glob("w*")), root
    runs = [check_run(directory, models) for root in args.runs for directory in sorted(root.glob("w*"))]
    assert runs
    with args.output.open("x") as output:
        json.dump({"model_id": st.MODEL_ID, "whole_multiplier_approved": False,
            "security_raw_sha256": sha(security_path), "security": security,
            "reduction_cost_model": "MATZOV", "reduction_shape_model": "GSA", "runs": runs,
            "scope": "Internal research: raw-row, plan, parameter, count, hash, model, and security cross-check. Not a rare-failure certificate."},
            output, indent=2, allow_nan=False)
        output.write("\n")
    print(f"Validated {len(models)} parameter/model pairs and {len(runs)} run configurations.")
    print(f"All {sum(r['correct_trials'] for r in runs)} encrypted outputs correct; no failed rows omitted.")


if __name__ == "__main__":
    main()
