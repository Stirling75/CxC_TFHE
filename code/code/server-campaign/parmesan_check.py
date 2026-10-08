#!/usr/bin/env python3
"""Independent checks of PARMESAN adapter output (adapters/parmesan/main.rs)."""
import csv
import json
from pathlib import Path

MASK64 = (1 << 64) - 1
METHOD = "parmesan-native"
DIGIT = {"-": -1, "0": 0, "+": 1}
# Product length returned by PARMESAN mul_impl for W-bit operands.  It depends
# only on the public Karatsuba/schoolbook structure, never on operand values
# (observed: 33 for W=16 and 66 for W=32 with upstream a5254f8).
OUTPUT_DIGITS = {16: 33, 32: 66}
HEADER = ["method", "width", "rayon_threads", "trial", "warmup", "x", "y", "total_seconds",
          "output_digits", "ok", "output_value", "output_digits_lsb_first"]


def splitmix(x):
    """Port of main.rs::splitmix (SplitMix64 finalizer with increment)."""
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return x ^ (x >> 31)


def operands(width, seed, trial):
    mask = (1 << width) - 1
    return (splitmix(seed ^ ((2 * trial) & MASK64)) & mask,
            splitmix(seed ^ ((2 * trial + 1) & MASK64)) & mask)


def decode(text):
    """Value of an LSB-first signed-binary string; rejects digits outside {-1,0,1}."""
    if not text or any(c not in DIGIT for c in text):
        raise AssertionError("output digits outside {-1,0,1}")
    return sum(DIGIT[c] << i for i, c in enumerate(text))


def check_row(row, width, seed, threads=None):
    assert row["method"] == METHOD, "PARMESAN method mismatch"
    assert int(row["width"]) == width, "PARMESAN width mismatch"
    if threads is not None:
        assert int(row["rayon_threads"]) == threads, "PARMESAN thread mismatch"
    x, y = int(row["x"]), int(row["y"])
    assert 0 <= x < 1 << width and 0 <= y < 1 << width, "PARMESAN operand out of range"
    assert (x, y) == operands(width, seed, int(row["trial"])), \
        "PARMESAN operands differ from seed derivation"
    digits = row["output_digits_lsb_first"]
    value = decode(digits)
    assert len(digits) == int(row["output_digits"]), "PARMESAN digit count mismatch"
    expected_length = OUTPUT_DIGITS.get(width)
    assert expected_length is not None and len(digits) == expected_length, \
        "PARMESAN product length mismatch"
    assert value == x * y, "PARMESAN decoded digits differ from x*y"
    assert int(row["output_value"]) == value, "PARMESAN recorded value differs from digits"
    assert row["ok"] == "true"
    return value


def validate_run(directory):
    """Check a complete parmesan-benchmark output directory."""
    directory = Path(directory)
    meta = json.loads((directory / "parameters.json").read_text())
    assert meta["tfhe_rs"] == "0.5.4"
    width, seed, threads = meta["width"], meta["seed"], meta["threads"]
    warmup, repetitions = meta["warmup"], meta["repetitions"]
    with (directory / "timings.csv").open() as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == HEADER, "PARMESAN CSV header mismatch"
        rows = list(reader)
    assert [int(r["trial"]) for r in rows] == list(range(warmup + repetitions)), \
        "missing, duplicate or reordered PARMESAN trials"
    for row in rows:
        check_row(row, width, seed, threads)
        assert (row["warmup"] == "true") == (int(row["trial"]) < warmup)
    return {"correct_products": len(rows), "width": width, "output_digits": OUTPUT_DIGITS[width]}
