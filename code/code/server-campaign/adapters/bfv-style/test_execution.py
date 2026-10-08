import json
from pathlib import Path
import shutil
import tempfile
import unittest

import audit_run
import analyze_noise

HERE = Path(__file__).resolve().parent


class ExecutionAuditTests(unittest.TestCase):
    def setUp(self):
        source = HERE / "results/parallel-equivalence-small-20260918"
        if not (source / "completed.json").exists():
            self.skipTest("optional encrypted equivalence fixture absent")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "run"
        shutil.copytree(source, self.directory, ignore=shutil.ignore_patterns("source"))

    def alter(self, name, mutate):
        path = self.directory / name
        value = json.loads(path.read_text())
        mutate(value)
        path.write_text(json.dumps(value))

    def test_completed_equivalent_run(self):
        result = audit_run.audit(self.directory)
        self.assertEqual(result["validated_cases"], 6)
        self.assertTrue(result["serial_ciphertext_equivalence_checked"])
        self.assertFalse(result["whole_multiplier_approved"])
        self.assertTrue(all(row["ks"] == 28 for row in result["summaries"]))

    def test_rejects_incorrect_shared_ks_count(self):
        self.alter("w8-random-r0.json", lambda row: row["counts"].update(ks=30))
        with self.assertRaises(AssertionError):
            audit_run.audit(self.directory)

    def test_rejects_serial_ciphertext_mismatch(self):
        self.alter("w8-random-r0.json", lambda row: row["serial_equivalence"].update(ciphertexts_identical=False))
        with self.assertRaises(AssertionError):
            audit_run.audit(self.directory)

    def test_rejects_warmup_mislabel(self):
        self.alter("w8-random-r0.json", lambda row: row.update(warmup=True))
        with self.assertRaises(AssertionError):
            audit_run.audit(self.directory)

    def test_rejects_changed_parameters(self):
        self.alter("manifest.json", lambda row: row["parameters"].update(encoding_pbs=[2, 21]))
        with self.assertRaises(AssertionError):
            audit_run.audit(self.directory)


class MeasurementAuditTests(unittest.TestCase):
    def test_parallel_grid_excludes_warmups(self):
        directory = HERE / "results/parallel-t8-grid-20260918"
        if not (directory / "completed.json").exists():
            self.skipTest("optional measured grid not completed")
        result = audit_run.audit(directory)
        self.assertEqual(result["validated_cases"], 24)
        self.assertEqual(len(result["summaries"]), 6)
        self.assertEqual(sum(case["warmup"] for case in result["cases"]), 6)
        for summary in result["summaries"]:
            measured = [json.loads((directory / f"w{summary['width']}-random-r{i}.json").read_text())
                        for i in (1, 2, 3)]
            self.assertEqual(summary["seconds"], [sum(row["timings_seconds"].values()) for row in measured])
            self.assertEqual(summary["repetitions"], 3)


class CampaignTimingTests(unittest.TestCase):
    """Synthetic records test validation only; they are not performance results."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        params = json.loads((HERE / "parameters/bfv-n800-range-aware.json").read_text())
        plan = analyze_noise.make_plan(16, True, 8)
        manifest = {"parameters": params, "kernel_only": False, "plan_only": False,
                    "coefficient_padding": False, "repetitions": 5, "warmup": 1,
                    "host": {"evaluation_threads": 1}, "plans": [plan], "patterns": ["random"],
                    "benchmark": True, "timing_contract": "continuous-evaluator-wall-clock-v1",
                    "tfhe_version": "1.6.1", "input_contract": "bootstrapped big-key radix-4, delta=2^59",
                    "output_contract": "same-key radix-4, delta=2^59, lower W bits (full mode only)"}
        self.write("manifest.json", manifest)
        self.write("completed.json", {"cases": 6, "whole_multiplier_approved": False,
                                      "kernel_only": False, "threads": 1, "warmup_per_configuration": 1})
        for trial in range(6):
            row = {"width": 16, "pattern": "random", "repetition": trial, "threads": 1,
                   "warmup": trial == 0, "status": "full-path-correct", "a_hex": "0", "b_hex": "1",
                   "expected_hex": "0", "total_seconds": 10.0 if trial == 0 else float(trial),
                   "timings_seconds": dict.fromkeys(audit_run.STAGES, 0.1),
                   "counts": {"pbs": plan["pbs_total"], "ks": plan["pbs_total"],
                              "glwe_products": 1, "packed_lwes": 16, "extracted_coefficients": 8}}
            for key in ("native_input_a", "native_input_b", "rescaled_a", "rescaled_b",
                        "packed_a", "packed_b", "convolution", "output"):
                row[key] = {"correct": True, "errors": [], "decoded_digits": [0]*8}
            self.write(f"w16-random-r{trial}.json", row)

    def write(self, name, value):
        (self.directory / name).write_text(json.dumps(value))

    def alter(self, name, mutate):
        row = json.loads((self.directory / name).read_text())
        mutate(row)
        self.write(name, row)

    def test_uses_outer_timer_and_excludes_warmup(self):
        report = audit_run.audit(self.directory, require_benchmark=True)
        self.assertEqual(report["summaries"][0]["seconds"], [1., 2., 3., 4., 5.])
        self.assertEqual(report["summaries"][0]["mean_seconds"], 3.)

    def test_rejects_diagnostic_timing(self):
        self.alter("manifest.json", lambda r: r.update(benchmark=False))
        with self.assertRaisesRegex(AssertionError, "continuous"):
            audit_run.audit(self.directory, require_benchmark=True)

    def test_rejects_missing_trial(self):
        (self.directory / "w16-random-r5.json").unlink()
        with self.assertRaises(FileNotFoundError):
            audit_run.audit(self.directory, require_benchmark=True)

    def test_rejects_bad_wall_clock(self):
        for value in (float("nan"), float("inf"), 0., 0.5):
            self.alter("w16-random-r1.json", lambda r: r.update(total_seconds=value))
            with self.subTest(value=value), self.assertRaises(AssertionError):
                audit_run.audit(self.directory, require_benchmark=True)

    def test_rejects_wrong_decoded_product(self):
        self.alter("w16-random-r1.json", lambda r: r["output"].update(decoded_digits=[1]*8))
        with self.assertRaises(AssertionError):
            audit_run.audit(self.directory, require_benchmark=True)


if __name__ == "__main__":
    unittest.main()
