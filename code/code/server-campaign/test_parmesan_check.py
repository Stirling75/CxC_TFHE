import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

import parmesan_check
from parmesan_check import HEADER, decode, operands, splitmix, validate_run

# Rows recorded by the patched adapter (W=16, seed 5, one warmup + one trial).
ROWS = [
    "parmesan-native,16,4,0,true,50010,35530,5.542709859,33,true,1776855300,"
    "00+00000+00-+-+-+00+0+-00+0+0+-+0",
    "parmesan-native,16,4,1,false,3543,57344,5.520385534,33,true,203169792,"
    "0000000000000-+000-00+0000+-+0000",
]


class ParmesanCheckTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.meta = {"tfhe_rs": "0.5.4", "width": 16, "threads": 4, "seed": 5,
                     "repetitions": 1, "warmup": 1}
        self.rows = [dict(zip(HEADER, line.split(","))) for line in ROWS]

    def write(self):
        (self.root / "parameters.json").write_text(json.dumps(self.meta))
        with (self.root / "timings.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=HEADER, lineterminator="\n")
            writer.writeheader()
            writer.writerows(self.rows)

    def rejects(self, **changes):
        self.rows[1].update(changes)
        self.write()
        with self.assertRaises(AssertionError):
            validate_run(self.root)

    def test_recorded_run_is_valid(self):
        self.write()
        self.assertEqual(validate_run(self.root)["correct_products"], 2)

    def test_pinned_splitmix_vectors(self):
        # Same vectors as main.rs tests::operand_derivation_is_pinned.
        self.assertEqual(splitmix(0), 0xE220A8397B1DCDAF)
        self.assertEqual(splitmix(20260907), 0xAD2EA8A771202A78)
        self.assertEqual(splitmix(20260907 ^ 1), 0x2B7E1F89419061D4)
        self.assertEqual(operands(16, 5, 0), (50010, 35530))

    def test_decode(self):
        self.assertEqual(decode("-+"), 1)
        self.assertEqual(decode("0" * 64 + "+"), 1 << 64)
        for bad in ("", "0?", "02", "+1"):
            with self.assertRaises(AssertionError):
                decode(bad)

    def test_rejects_wrong_product_digits(self):
        self.rejects(output_digits_lsb_first="+" + self.rows[1]["output_digits_lsb_first"][1:])

    def test_rejects_self_consistent_substituted_operands(self):
        # x*y and output_value agree, but x is not the seed-derived operand.
        self.rejects(x="7086", y="28672")

    def test_rejects_out_of_range_operand(self):
        self.rejects(x=str(1 << 16))

    def test_rejects_wrong_length(self):
        digits = self.rows[1]["output_digits_lsb_first"] + "0"
        self.rejects(output_digits_lsb_first=digits, output_digits=str(len(digits)))

    def test_rejects_invalid_digit_and_count(self):
        self.rejects(output_digits_lsb_first=self.rows[1]["output_digits_lsb_first"][:-1] + "?")
        self.setUp()
        self.rejects(output_digits="32")

    def test_rejects_recorded_value_mismatch(self):
        self.rejects(output_value="203169793")

    def test_rejects_method_and_seed(self):
        self.rejects(method="parmesan")
        self.setUp()
        self.meta["seed"] = 6
        self.write()
        with self.assertRaises(AssertionError):
            validate_run(self.root)

    def test_rejects_missing_trial(self):
        self.rows = self.rows[:1]
        self.write()
        with self.assertRaises(AssertionError):
            validate_run(self.root)


if __name__ == "__main__":
    unittest.main()
