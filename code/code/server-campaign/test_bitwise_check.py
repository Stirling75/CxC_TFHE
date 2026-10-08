import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

from bitwise_check import bits_value, lut_counts, operands, validate_run


class BitwiseReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.metadata = {
            "tfhe_rs": "1.7.0", "whole_multiplier_approved": False,
            "parameters": {"message_modulus": 2, "carry_modulus": 2, "log2_p_fail": -128.186},
            "arguments": {"preset": "m1c1-gaussian", "methods": ["morshed", "trifan"],
                "widths": [2], "threads": [1], "patterns": ["max"], "warmup": 0,
                "repetitions": 1, "fresh_inputs": False, "seed": 20260908},
        }
        common = {"preset": "m1c1-gaussian", "width": 2, "threads": 1, "pattern": "max",
            "seed": 20260908, "trial": 0, "warmup": "false", "input_state": "bootstrapped", "input_pbs": 4,
            "total_s": 0.5, "x_hex": "3", "y_hex": "3", "ok": "true"}
        self.rows = [
            dict(common, method="morshed", output_width=4, logical_lut_calls=44,
                 trivial_lut_calls=2, real_pbs=42, library_pbs_count=44,
                 expected_hex="9", output_hex="9"),
            dict(common, method="trifan", output_width=2, logical_lut_calls=12,
                 trivial_lut_calls=0, real_pbs=12, library_pbs_count=12,
                 expected_hex="1", output_hex="1"),
        ]

    def write(self):
        (self.root / "parameters.json").write_text(json.dumps(self.metadata))
        (self.root / "preflight.json").write_text(json.dumps(
            {"encrypted_full_adder_triples": 8, "both_adders_correct": True}))
        with (self.root / "timings.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.rows[0])
            writer.writeheader()
            writer.writerows(self.rows)

    def test_valid_grid(self):
        self.write()
        result = validate_run(self.root)
        self.assertEqual(result["correct_products"], 2)
        self.assertFalse(result["whole_multiplier_approved"])
        self.assertIsNone(result["rows"][0]["sample_sd_s"])

    def test_pruned_count_and_contract(self):
        self.metadata["arguments"]["methods"][1] = "trifan-pruned"
        self.rows[1].update(method="trifan-pruned", logical_lut_calls=7, real_pbs=7,
                            library_pbs_count=7)
        self.write()
        self.assertEqual(validate_run(self.root)["correct_products"], 2)
        self.rows[1]["logical_lut_calls"] = 12
        self.write()
        with self.assertRaises(AssertionError):
            validate_run(self.root)

    def test_rejects_wrong_parameter_modulus(self):
        self.metadata["parameters"]["carry_modulus"] = 4
        self.write()
        with self.assertRaises(AssertionError):
            validate_run(self.root)

    def test_rejects_discarding_high_product(self):
        self.rows[0]["output_hex"] = "1"
        self.write()
        with self.assertRaises(AssertionError):
            validate_run(self.root)

    def test_rejects_missing_duplicate_or_extra_pbs(self):
        original = copy.deepcopy(self.rows)
        for variant in [original[:1], original + original[:1], copy.deepcopy(original)]:
            self.rows = variant
            if len(variant) == 2:
                self.rows[0]["real_pbs"] = 43
            self.write()
            with self.assertRaises(AssertionError):
                validate_run(self.root)

    def test_rejects_real_count_equal_to_logical(self):
        # The old library counter always equalled the logical count.
        self.rows[0].update(real_pbs=44, trivial_lut_calls=0)
        self.write()
        with self.assertRaises(AssertionError):
            validate_run(self.root)

    def test_rejects_shifted_real_trivial_split(self):
        self.rows[0].update(real_pbs=41, trivial_lut_calls=3)
        self.write()
        with self.assertRaises(AssertionError):
            validate_run(self.root)

    def test_rejects_library_counter_mismatch(self):
        self.rows[0]["library_pbs_count"] = 42
        self.write()
        with self.assertRaises(AssertionError):
            validate_run(self.root)

    def test_random_operands_are_rederived_from_seed(self):
        self.metadata["arguments"].update(patterns=["random"], seed=7, widths=[16],
                                          methods=["trifan"])
        self.rows = [dict(self.rows[1], pattern="random", seed=7, width=16, output_width=16,
                          input_pbs=32, logical_lut_calls=768, real_pbs=768,
                          library_pbs_count=768)]
        x, y = (bits_value(v) for v in operands(16, 7, 0, "random"))
        product = format(x * y % (1 << 16), "x")
        self.rows[0].update(x_hex=format(x, "x"), y_hex=format(y, "x"),
                            expected_hex=product, output_hex=product)
        self.write()
        self.assertEqual(validate_run(self.root)["correct_products"], 1)
        # A self-consistent row with substituted operands must be rejected.
        x ^= 1
        product = format(x * y % (1 << 16), "x")
        self.rows[0].update(x_hex=format(x, "x"), expected_hex=product, output_hex=product)
        self.write()
        with self.assertRaises(AssertionError):
            validate_run(self.root)
        self.rows[0]["seed"] = 8
        self.write()
        with self.assertRaises(AssertionError):
            validate_run(self.root)

    def test_operand_port_matches_pinned_rust_vectors(self):
        # Same vectors as inputs::tests::operand_derivation_is_pinned.
        for width, seed, trial, x_hex, y_hex in [
                (16, 20260908, 0, "83ff", "7d14"), (16, 20260908, 3, "77e6", "c147"),
                (7, 0, 0, "22", "77"),
                (256, 2**64 - 1, 5,
                 "8d345b771757ad0660c5ced50345e89f56dcf4c8350013b839a94aa5640e7451",
                 "625a903c20bcbf03b486291cec4b490ecbc76b4118866f2b4b66fd3b89f528b7")]:
            x, y = operands(width, seed, trial, "random")
            self.assertEqual((format(bits_value(x), "x"), format(bits_value(y), "x")),
                             (x_hex, y_hex))
        x, y = operands(4, 1, 0, "zero")
        self.assertEqual(bits_value(x), 0)
        self.assertEqual([bits_value(v) for v in operands(4, 1, 0, "alternating")], [0xa, 0x5])
        self.assertEqual([bits_value(v) for v in operands(4, 1, 0, "carry-chain")], [0xf, 0x3])

    def test_lut_split_matches_pinned_rust_model(self):
        # Same values as circuits::tests::real_pbs_split_is_pinned.
        self.assertEqual(lut_counts("morshed", 16, 1), (2816, 16, 2800))
        self.assertEqual(lut_counts("morshed", 16, 4), (3296, 539, 2757))
        self.assertEqual(lut_counts("trifan", 16, 4), (768, 0, 768))
        self.assertEqual(lut_counts("trifan-pruned", 16, 4), (392, 0, 392))
        for width in (1, 3, 8, 32):
            for threads in (1, 2, 4, 64):
                logical, trivial, real = lut_counts("morshed", width, threads)
                self.assertEqual(logical, width**2 + 10 * width * (width + min(width, threads) - 1))
                self.assertEqual(trivial + real, logical)
                self.assertGreater(real, 0)

    def test_rejects_missing_input_refresh(self):
        self.rows[0]["input_pbs"] = 0
        self.write()
        with self.assertRaises(AssertionError):
            validate_run(self.root)


if __name__ == "__main__":
    unittest.main()
