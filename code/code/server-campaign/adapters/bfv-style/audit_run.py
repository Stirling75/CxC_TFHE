#!/usr/bin/env python3
"""Validate a completed selected-candidate run and summarize measured stages."""
import argparse
import json
import math
from pathlib import Path
import statistics

import analyze_noise as noise

HERE = Path(__file__).resolve().parent
STAGES = {"input_rescale", "packing", "tensor", "relinearization", "extraction",
          "digit_decomposition", "normalization"}


def audit(directory, require_benchmark=False):
    manifest = json.loads((directory / "manifest.json").read_text())
    done = json.loads((directory / "completed.json").read_text())
    candidate = json.loads((HERE / "parameters/bfv-n800-range-aware.json").read_text())
    actual = manifest["parameters"]
    assert set(actual) == set(candidate), "parameter schema mismatch"
    for key, value in candidate.items():
        if key in ("lwe_sigma", "glwe_sigma"):
            assert abs(actual[key]-value) <= 2*math.ulp(value), (key, actual[key], value)
        else:
            assert actual[key] == value, (key, actual[key], value)
    assert not manifest["kernel_only"] and not manifest["plan_only"]
    assert not manifest["coefficient_padding"]
    benchmark = manifest.get("benchmark", False)
    if require_benchmark:
        assert benchmark, "continuous evaluator timing is required"
    if benchmark:
        assert manifest["timing_contract"] == "continuous-evaluator-wall-clock-v1"
        assert manifest["tfhe_version"] == "1.6.1"
        assert manifest["input_contract"] == "bootstrapped big-key radix-4, delta=2^59"
        assert manifest["output_contract"] == "same-key radix-4, delta=2^59, lower W bits (full mode only)"
    repetitions, warmup = manifest["repetitions"], manifest.get("warmup", 0)
    threads = manifest["host"]["evaluation_threads"]
    assert repetitions >= 1 and warmup >= 0 and threads >= 1
    expected_cases = len(manifest["plans"])*len(manifest["patterns"])*(repetitions+warmup)
    assert done["cases"] == expected_cases
    assert not done["whole_multiplier_approved"]
    assert not done["kernel_only"]
    assert done.get("threads", 1) == threads
    assert done.get("warmup_per_configuration", 0) == warmup
    summaries, cases, expected_files = [], [], set()
    for plan in manifest["plans"]:
        width = plan["width"]
        canonical = noise.make_plan(width, True, 8)
        for key, value in canonical.items():
            assert plan[key] == value, (key, plan[key], value)
        screen = noise.screen(actual, plan, True, False)
        assert screen["raw_log2_union"] < -128
        normalizer_tasks = sum(e["kind"] == "normalization" and e["label"][1] in ("low", "final")
                               for e in screen["event_trace"])
        expected_ks = plan["pbs_total"] if threads == 1 else (
            plan["pbs_input_rescale"]+plan["pbs_bit_extraction"]+normalizer_tasks)
        assert plan.get("ks_total", expected_ks) == expected_ks
        for pattern in manifest["patterns"]:
            measured = []
            totals = []
            for trial in range(repetitions+warmup):
                path = directory / f"w{width}-{pattern}-r{trial}.json"
                expected_files.add(path)
                row = json.loads(path.read_text())
                assert (row["width"], row["pattern"], row["repetition"]) == (width, pattern, trial)
                assert row.get("threads", 1) == threads
                assert row.get("warmup", False) == (trial < warmup)
                assert row["status"] == "full-path-correct"
                for field in ("native_input_a", "native_input_b", "rescaled_a", "rescaled_b",
                              "packed_a", "packed_b", "convolution", "output"):
                    assert row[field]["correct"] and not row[field]["errors"], field
                a, b = int(row["a_hex"], 16), int(row["b_hex"], 16)
                assert 0 <= a < 2**width and 0 <= b < 2**width
                expected = a*b % 2**width
                assert int(row["expected_hex"], 16) == expected
                if benchmark:
                    digits = row["output"]["decoded_digits"]
                    assert len(digits) == width//2 and all(type(d) is int and 0 <= d < 4 for d in digits)
                    assert sum(d * 4**i for i, d in enumerate(digits)) == expected
                assert row["counts"]["pbs"] == plan["pbs_total"]
                assert row["counts"]["ks"] == expected_ks
                assert row["counts"]["glwe_products"] == 1
                assert row["counts"]["packed_lwes"] == 2*plan["digits"]
                assert row["counts"]["extracted_coefficients"] == plan["digits"]
                if manifest.get("verify_serial", False):
                    check = row["serial_equivalence"]
                    assert check["ciphertexts_identical"] and check["same_keys_and_inputs"]
                    assert check["reference_counts"]["pbs"] == plan["pbs_total"]
                    assert check["reference_counts"]["ks"] == plan["pbs_total"]
                assert set(row["timings_seconds"]) == STAGES
                assert all(math.isfinite(v) and v >= 0 for v in row["timings_seconds"].values())
                total = row["total_seconds"] if benchmark else sum(row["timings_seconds"].values())
                assert math.isfinite(total) and total > 0
                assert total + 1e-9 >= sum(row["timings_seconds"].values()), "wall-clock shorter than phases"
                cases.append({"file": path.name, "sha256": noise.digest(path),
                              "warmup": trial < warmup})
                if trial >= warmup:
                    measured.append(row["timings_seconds"])
                    totals.append(total)
            summaries.append({"width": width, "pattern": pattern, "threads": threads,
                "repetitions": repetitions, "warmup": warmup,
                "seconds": totals, "mean_seconds": statistics.mean(totals),
                "sample_sd_seconds": statistics.stdev(totals) if repetitions > 1 else None,
                "mean_stage_seconds": {key: statistics.mean(row[key] for row in measured) for key in sorted(STAGES)},
                "conditional_union_log2": screen["raw_log2_union"],
                "pbs": plan["pbs_total"], "ks": expected_ks})
    assert set(directory.glob("w*-r*.json")) == expected_files
    return {"run": str(directory.resolve()), "validated_cases": len(cases), "cases": cases,
            "manifest_sha256": noise.digest(directory / "manifest.json"),
            "audit_source_sha256": noise.digest(Path(__file__)),
            "noise_model_sha256": noise.digest(Path(noise.__file__)),
            "primitive_model_sha256": noise.digest(Path(noise.primitive.__file__)),
            "serial_ciphertext_equivalence_checked": manifest.get("verify_serial", False),
            "summaries": summaries, "whole_multiplier_approved": False,
            "timing_contract": ("continuous-evaluator-wall-clock-v1" if benchmark else
                "sum of evaluation stages; excludes preparation, checks and serial reference")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run)
    with args.output.open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
    print(json.dumps(result["summaries"], indent=2))


if __name__ == "__main__":
    main()
