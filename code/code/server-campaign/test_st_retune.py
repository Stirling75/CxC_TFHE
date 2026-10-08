import copy
from fractions import Fraction
import json
import math
import random
import unittest

import st_retune as st
from st_gaussian import key_error_variance, signed_digit_second_moments
from st_validate import attack_log2, check_counts, check_security_row, parameter_roundoff

U64 = (1 << 64) - 1


def tfhe_rs_signed_digits(value, base_log, levels):
    """Bit-exact u64 transcription of tfhe 1.6.1 SignedDecomposer::decompose."""
    rep = base_log * levels
    res = value >> (64 - rep - 1)
    rounding_bit = res & 1
    res = ((res + 1) & U64) >> 1
    res &= U64 >> (64 - rep)
    balance = ((((res - 1) & U64) | (rounding_bit << (rep - 1))) & res) >> (rep - 1)
    state = (res - (balance << rep)) & U64
    digits = []
    for _ in range(levels):
        residue = state & ((1 << base_log) - 1)
        state = (state >> base_log) | (U64 ^ (U64 >> base_log) if state >> 63 else 0)
        carry = ((((residue - 1) & U64) | state) & residue) >> (base_log - 1)
        state = (state + carry) & U64
        digits.append(residue - (carry << base_log))
    return digits


class RetuningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = json.loads((st.ROOT / "parameters/st-cc2-n1152.json").read_text())

    def test_reused_source_amplitudes_add_before_squaring(self):
        combined = st.merge_sources([(1, {"a": 2.0}), (2, {"a": 3.0, "b": 4.0})])
        self.assertEqual(combined, {"a": 8.0, "b": 8.0})
        self.assertEqual(st.source_variance(combined), 128.0)
        self.assertEqual(st.merge_sources([(-2, {"a": 3.0})]), {"a": 6.0})

    def test_trace_envelope_retains_coefficient_covariance(self):
        v = st.variances(self.p)
        # RevHomTrace composition: independent variances of lift PBS, trace and scheme switch.
        self.assertAlmostEqual(v["selector_row_envelope"] / (
            v["pbs"] + v["trace_sigma"]**2 + v["ss_rounding"] + v["ss_key"] + v["ss_fft"]), 1.0)

    def test_output_uses_cc2_and_includes_closing_ks(self):
        result = st.screen(self.p, 16)
        v = result["variances"]
        events = [e for e in result["events"] if e["stage"] == "output_decode"]
        self.assertEqual(len(events), 8)
        for e in events:
            self.assertEqual(e["margin"], 2**51)
            self.assertEqual(e["ms_variance"], 0)
            self.assertEqual(e["variance_before_ms"], v["pbs"] + v["ks"])

    def test_input_contract_rescaled_closed_pbs_variance(self):
        r = st.screen(self.p, 16)
        e = r["events"][0]
        self.assertEqual(e["stage"], "input_grouped_lift")
        self.assertEqual(e["variance_before_ms"], 2**21 * r["variances"]["closed"])

    def test_two_lane_terminal_counts_and_margin(self):
        two = st.screen(self.p, 256)
        p4 = {**self.p, "terminal_lut_count_log": 2}
        four = st.screen(p4, 256)
        self.assertEqual(two["stage_counts"]["terminal_binary_lift"], 990)
        self.assertEqual(four["stage_counts"]["terminal_binary_lift"], 495)
        self.assertEqual(two["event_count"] - four["event_count"], 495)
        self.assertLess(two["conditional_log2_union"], -128)
        self.assertGreater(four["conditional_log2_union"], -128)
        self.assertFalse(two["whole_multiplier_approved"])

    def test_all_widths_pass_only_conditional_screen(self):
        for width in (16, 32, 64, 128, 256):
            result = st.screen(self.p, width)
            self.assertTrue(result["conditional_gaussian_screen_pass"])
            self.assertFalse(result["whole_multiplier_approved"])

    def test_full_event_union_recomputed_without_independence(self):
        r = st.screen(self.p, 16)
        logs = [e["log2_p_fail"] + math.log2(e.get("multiplicity", 1)) for e in r["events"]]
        pivot = max(logs)
        self.assertAlmostEqual(r["conditional_log2_union"],
            pivot + math.log2(math.fsum(2**(x-pivot) for x in logs)))

    def test_scheme_switch_rounding_changes_candidate_decision(self):
        # Under the RevHomTrace composition SS (13,3) still passes; (12,3) does not.
        weak = {**self.p, "ss": [12, 3]}
        self.assertGreater(st.screen(weak, 256)["conditional_log2_union"], -128)
        self.assertLess(st.screen(self.p, 256)["conditional_log2_union"], -128)

    def test_split_rounding_event_multiplicity(self):
        r = st.screen(self.p, 256)
        split = [e for e in r["events"] if e["stage"] == "split_high_integer_rounding"]
        self.assertEqual(len(split), 1)
        self.assertEqual(split[0]["multiplicity"], (512+495)*4*11*2*2048)
        self.assertEqual(r["event_count"], sum(r["stage_counts"].values()))

    def test_top_column_high_bit_is_not_read(self):
        for width in (16, 64, 256):
            r = st.screen(self.p, width)
            top = [e for e in r["events"] if e["stage"] == "terminal_binary_lift" and e["column"] == width - 1]
            self.assertEqual({e["read"] for e in top}, {0})
            self.assertEqual(len(top), 2)  # two-lane packing: two rotations per read

    def test_expected_runtime_counts_match_rust_plan_test_values(self):
        # Mirrors plan.rs counts_match_independent_python_reference.
        for width, compressors, reads in ((16, 0, 15), (64, 332, 111), (256, 7414, 495)):
            c = st.screen(self.p, width)["expected_runtime_counts"]
            self.assertEqual((c["compressors"], c["terminal_binary_cbs"]), (compressors, reads))
            self.assertEqual(c["key_switches"], compressors + 2*width - 8)
            self.assertEqual(c["ggsw_conversions"], 2*width + reads)
            self.assertEqual(c["blind_rotations"], width + compressors + 2*reads + width//2)
        c = st.screen(self.p, 16)["expected_runtime_counts"]
        self.assertEqual((c["product_cmux"], c["terminal_cmux"], c["cmux"]), (32*42, 30, 32*42+30))
        check_counts({"blind_rotations": 54, "key_switches": 24, "conversions": 47, "cmux": 1374}, c)
        with self.assertRaises(AssertionError):
            check_counts({"blind_rotations": 56, "key_switches": 24, "conversions": 48, "cmux": 1374}, c)

    def test_signed_digit_moments_exact_against_tfhe_rs_transcription(self):
        for base_log, levels in ((1, 1), (1, 4), (2, 3), (3, 2), (1, 8), (2, 5), (4, 3)):
            rep = base_log * levels
            total = [0] * levels
            for top in range(1 << (rep + 1)):  # every rounding class of a uniform u64
                for i, d in enumerate(tfhe_rs_signed_digits(top << (63 - rep), base_log, levels)):
                    total[i] += d * d
            exact = [Fraction(t, 1 << (rep + 1)) for t in total]
            model = signed_digit_second_moments(base_log, levels)
            for a, b in zip(exact, model):
                self.assertAlmostEqual(float(a), b, places=12)

    def test_signed_digit_moments_sampled_for_candidate_ks(self):
        rng = random.Random(20261001)
        for base_log, levels in ((1, 23), (3, 8)):
            n = 40000
            mean = sum(sum(d*d for d in tfhe_rs_signed_digits(rng.getrandbits(64), base_log, levels))
                       for _ in range(n)) / n
            self.assertAlmostEqual(mean / math.fsum(signed_digit_second_moments(base_log, levels)), 1, delta=0.02)
        self.assertAlmostEqual(math.fsum(signed_digit_second_moments(1, 23)) / 23, 0.33816425, places=7)
        self.assertAlmostEqual(math.fsum(signed_digit_second_moments(3, 8)), 42.419753074645996, places=9)
        # The first (least significant) level is the independent-uniform value.
        self.assertEqual(signed_digit_second_moments(3, 8)[0], 5.5)
        self.assertAlmostEqual(key_error_variance(1, 1, 23, 2.0**-64), 7.777777791023254)

    def test_security_report_accepts_float_and_string_infinity(self):
        row = {"attacks_rop_log2": {k: 140.0 for k in ("bkw", "usvp", "bdd", "bdd_hybrid",
               "bdd_mitm_hybrid", "dual", "dual_hybrid")}, "minimum_rop_log2": 130.0}
        row["attacks_rop_log2"]["dual"] = 130.0
        for infinity in (math.inf, "infinity", "Infinity"):
            row["attacks_rop_log2"]["arora-gb"] = infinity
            self.assertEqual(check_security_row(row)["arora-gb"], "infinity")
        self.assertEqual(attack_log2(json.loads("Infinity")), math.inf)
        with self.assertRaises(ValueError):
            attack_log2("nan-ish")
        row["attacks_rop_log2"]["arora-gb"] = math.nan
        with self.assertRaises(AssertionError):
            check_security_row(row)

    def test_retained_security_report_is_accepted(self):
        security = json.loads((st.ROOT / "results/st-retune-security-full-20260907.json").read_text())
        for row in security["rows"]:
            check_security_row(row)

    def test_screen_values_after_digit_moment_and_top_read_fixes(self):
        n1088 = json.loads((st.ROOT / "parameters/st-cc2-n1088.json").read_text())
        for p, w16 in ((self.p, -174.395), (n1088, -207.343)):
            for i, width in enumerate((16, 32, 64, 128, 256)):
                r = st.screen(p, width)
                self.assertAlmostEqual(r["conditional_log2_union"], w16 + i, delta=0.01)
                self.assertEqual(r["dominant_event_stage"], "input_grouped_lift")

    def test_invalid_parameters_rejected(self):
        for field, value in [("cbs", [5, 5]), ("ks", [0, 8]),
            ("lwe_sigma", float("nan")), ("lwe_dimension", 2048),
            ("terminal_lut_count_log", 0), ("polynomial_size", 512)]:
            p = copy.deepcopy(self.p)
            p[field] = value
            with self.assertRaises(ValueError):
                st.screen(p, 16)
        with self.assertRaises(ValueError):
            st.screen(self.p, 8)

    def test_parameter_check_allows_only_binary64_decimal_roundoff(self):
        actual = {**self.p, "glwe_sigma": math.nextafter(self.p["glwe_sigma"], 0.0)}
        self.assertIn("glwe_sigma", parameter_roundoff(actual, self.p))
        with self.assertRaises(AssertionError):
            parameter_roundoff({**self.p, "glwe_sigma": self.p["glwe_sigma"] * 1.000001}, self.p)
        with self.assertRaises(AssertionError):
            parameter_roundoff({**self.p, "ks": [3, 7]}, self.p)


if __name__ == "__main__":
    unittest.main()
