import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import analyze_noise as model
import verify_candidate


PARAMETERS = {"n": 1152, "polynomial_size": 2048,
              "lwe_sigma": 2**-26, "glwe_sigma": 9.25119974676756e-16,
              "pbs": [11, 3], "encoding_pbs": [3, 20], "ks": [3, 8],
              "packing": [4, 15], "relin": [4, 15]}


def plan(width, counts):
    d = width//2
    return {"width": width, "digits": d, "encoding_precision": (9*d).bit_length(),
            "pbs_input_rescale": counts[0], "pbs_bit_extraction": counts[1],
            "pbs_normalization": counts[2], "pbs_total": sum(counts)}


class NoiseTests(unittest.TestCase):
    def test_candidate_rejects_changed_security_report(self):
        p = json.loads((Path(__file__).parent / "parameters/bfv-n800-range-aware.json").read_text())
        with patch.object(model, "digest", return_value="wrong-hash"):
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_candidate.verify(p)

    def test_security_screen_requires_exact_parameters_and_all_attacks(self):
        report = json.loads(verify_candidate.SMALL.read_text())
        row = verify_candidate.security_row(report, 800, 2**-17)
        self.assertGreater(row["minimum_rop_log2"], 128)
        with self.assertRaises(ValueError):
            verify_candidate.security_row(report, 800, 2**-18)
        damaged = json.loads(verify_candidate.SMALL.read_text())
        del damaged["rows"][0]["attacks_rop_log2"]["dual_hybrid"]
        with self.assertRaises(ValueError):
            verify_candidate.security_row(damaged, 800, 2**-17)

    def test_candidate_preflight_is_conditional_and_rejects_old_normalizer(self):
        path = Path(__file__).parent / "parameters/bfv-n800-range-aware.json"
        p = json.loads(path.read_text())
        result = verify_candidate.verify(p)
        self.assertTrue(result["conditional_preflight_pass"])
        self.assertFalse(result["whole_multiplier_approved"])
        json.dumps(result, allow_nan=False)
        with self.assertRaises(ValueError):
            verify_candidate.verify(p | {"normalizer_lut_domain": 16})

    def test_range_aware_plan_and_scaled_input(self):
        p = PARAMETERS | {"input_lut_domain": 4, "normalizer_lut_domain": 8,
                          "degree_aware": True}
        plan = model.make_plan(256, True, 8)
        self.assertEqual(plan["pbs_normalization"], 1104)
        r = model.screen(p, model.make_plan(8, True, 8), True, False)
        first = r["event_trace"][0]
        self.assertEqual(first["spacing"], model.Q/8)
        self.assertEqual(first["variance_before_ks"], 16*r["ordinary_pbs"]["total"])
        for event in r["event_trace"]:
            if event["kind"] == "normalization":
                self.assertEqual(event["spacing"], model.Q/16)

    def test_retuned_model_all_widths(self):
        path = Path(__file__).parent / "parameters/bfv-n800-range-aware.json"
        p = json.loads(path.read_text())
        for width in (8, 16, 32, 64, 128, 256):
            r = model.screen(p, model.make_plan(width, True, 8), True, False)
            self.assertLess(r["raw_log2_union"], -128)
            self.assertFalse(r["whole_multiplier_approved"])

    def test_original_model_values_preserved(self):
        directory = Path(__file__).parent / "results/width-grid-20260918"
        previous = Path(__file__).parent / "results/noise-screen-20260918.json"
        if not previous.exists():
            self.skipTest("optional original model output absent")
        manifest = json.loads((directory / "manifest.json").read_text())
        rows = json.loads(previous.read_text())["rows"]
        for old in rows:
            plan = next(p for p in manifest["plans"] if p["width"] == old["width"])
            new = model.screen(manifest["parameters"], plan, old["pbs_fft_model_included"],
                               old["layout_model"] == "dense-envelope")
            self.assertAlmostEqual(new["raw_log2_union"], old["raw_log2_union"], places=7)

    def test_packing_filled_and_empty(self):
        p = model.packing_variances(2048, 8, 123.0, 1e-15, 4, 15)
        self.assertAlmostEqual(p["filled"]-p["empty"], 123+p["rounding"])
        twice = model.packing_variances(2048, 16, 123.0, 1e-15, 4, 15)
        self.assertEqual(twice["empty"], 2*p["empty"])

    def test_layout_reduces_to_theorem_one_for_dense_messages(self):
        for n in (8, 16):
            for col in range(n):
                a = model.tensor_terms(n, n, col, 2**54, 10., 2., 1e-15, (4, 15))
                b = model.tensor_terms(n, n, col, 2**54, 10., 2., 1e-15, (4, 15), True)
                self.assertEqual(a, b)

    def test_sparse_linear_and_quadratic_terms(self):
        n, d, col, fill, empty, delta = 8, 3, 1, 7., 2., 2**54
        t = model.tensor_terms(n, d, col, delta, fill, empty, 1e-15, (4, 15))
        # col=1 has two filled/filled pairs and one filled/empty message pair.
        self.assertEqual(t["message_times_error"], 18*(2*fill+empty))
        self.assertEqual(t["error_times_error"],
                         (2*fill**2+2*fill*empty+4*empty**2)/delta**2)

    def test_dense_envelope_dominates_layout(self):
        for col in range(4):
            a = model.tensor_terms(16, 4, col, 2**54, 10., 2., 1e-15, (4, 15))
            b = model.tensor_terms(16, 4, col, 2**54, 10., 2., 1e-15, (4, 15), True)
            for name in a:
                self.assertLessEqual(a[name], b[name])

    def test_source_count_and_union(self):
        r = model.screen(PARAMETERS, plan(8, (8, 15, 6)), True, False)
        self.assertEqual(r["events"], 33)
        self.assertAlmostEqual(r["raw_log2_union"], model.logsum(
            [v["raw_log2_union"] for v in r["families"].values()]))
        self.assertFalse(r["whole_multiplier_approved"])
        self.assertFalse(r["conditional_model_target_pass"])

    def test_w256_counts_and_fft_sensitivity(self):
        p = plan(256, (256, 1161, 508))
        a = model.screen(PARAMETERS, p, False, False)
        b = model.screen(PARAMETERS, p, True, False)
        self.assertEqual(b["events"], 2053)
        self.assertGreater(b["max_product_variance"], a["max_product_variance"])
        self.assertGreater(b["raw_log2_union"], a["raw_log2_union"])
        self.assertEqual(b["worst_event"]["kind"], "bit_extraction")
        self.assertFalse(b["conditional_model_target_pass"])

    def test_first_bit_uses_core_variance_with_scale_before_ks(self):
        p = plan(8, (8, 15, 6))
        r = model.screen(PARAMETERS, p, True, False)
        first = next(e for e in r["event_trace"] if e["kind"] == "bit_extraction")
        core = sum(model.tensor_terms(2048, 4, 0, model.Q/64,
                       r["packing"]["filled"], r["packing"]["empty"],
                       PARAMETERS["glwe_sigma"], PARAMETERS["relin"]).values())
        self.assertEqual(first["variance_before_ks"], 32**2*core)
        expected = model.primitive.pbs_input_log2_pfail(
            1152, model.Q, 2048, 0, model.Q/2, 32**2*core+r["ks_increment"])
        self.assertEqual(first["log2_gaussian"], expected)

    def test_repeated_bit_error_remains_in_residual(self):
        r = model.screen(PARAMETERS, plan(8, (8, 15, 6)), True, False)
        bits = [e for e in r["event_trace"]
                if e["kind"] == "bit_extraction" and e["label"][0] == 0]
        # For j=0,1, weight/base=1. At j=1 the shift halves but the previous
        # PBS output error remains in the residual, not a refreshed input.
        self.assertAlmostEqual(bits[1]["variance_before_ks"] / 16**2,
                               bits[0]["variance_before_ks"] / 32**2
                               + r["ordinary_pbs"]["total"])

    def test_saved_encrypted_trace_counts_when_available(self):
        directory = Path(__file__).parent / "results/width-grid-20260918"
        if not directory.exists():
            self.skipTest("optional local encrypted results absent")
        manifest = json.loads((directory / "manifest.json").read_text())
        for p in manifest["plans"]:
            r = model.screen(manifest["parameters"], p, True, False)
            observed = json.loads((directory / f"w{p['width']}-random-r0.json").read_text())
            self.assertEqual(observed["status"], "full-path-correct")
            self.assertEqual(r["events"]-p["digits"], observed["counts"]["pbs"])
            self.assertTrue(math.isfinite(r["raw_log2_union"]))

    def test_retuned_encrypted_schedule_matches_model_when_available(self):
        p = json.loads((Path(__file__).parent / "parameters/bfv-n800-range-aware.json").read_text())
        for name in ("retuned-grid-20260918", "retuned-extremes-20260918"):
            directory = Path(__file__).parent / "results" / name
            if not (directory / "completed.json").exists():
                self.skipTest("optional retuned encrypted grid not completed")
            manifest = json.loads((directory / "manifest.json").read_text())
            self.assertEqual(set(manifest["parameters"]), set(p))
            for key, value in p.items():
                if key in ("lwe_sigma", "glwe_sigma"):
                    # serde_json's decimal parse/serialize can differ by one
                    # binary64 ULP from Python's correctly rounded conversion.
                    self.assertLessEqual(abs(manifest["parameters"][key]-value), 2*math.ulp(value))
                else:
                    self.assertEqual(manifest["parameters"][key], value)
            self.assertEqual(manifest["host"]["evaluation_threads"], 1)
            completed = json.loads((directory / "completed.json").read_text())
            self.assertEqual(completed["cases"], len(manifest["plans"])*len(manifest["patterns"]))
            for plan in manifest["plans"]:
                expected = model.make_plan(plan["width"], True, 8)
                for key, value in expected.items():
                    self.assertEqual(plan[key], value)
                for pattern in manifest["patterns"]:
                    observed = json.loads((directory / f"w{plan['width']}-{pattern}-r0.json").read_text())
                    self.assertEqual(observed["status"], "full-path-correct")
                    self.assertEqual(plan["pbs_total"], observed["counts"]["pbs"])


if __name__ == "__main__":
    unittest.main()
