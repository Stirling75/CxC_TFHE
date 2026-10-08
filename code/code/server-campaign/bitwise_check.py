#!/usr/bin/env python3
"""Run and independently check the shortint comparison grid on the local host."""
import argparse
import csv
import functools
import hashlib
import itertools
import json
import platform
import statistics
import subprocess
import tarfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1


# --- Operand derivation: port of adapters/bitwise/src/inputs.rs -------------
# StdRng (rand 0.8.8) is ChaCha12Rng (rand_chacha 0.3.1).  seed_from_u64 is
# rand_core 0.6.4's PCG32 seed expansion; gen_range(0..2) for u8 samples a u32
# and, for range 2, returns its most significant bit (the zone never rejects).

def _pcg32_seed(state):
    words = []
    for _ in range(8):
        state = (state * 6364136223846793005 + 11634580027462260723) & MASK64
        xorshifted = (((state >> 18) ^ state) >> 27) & MASK32
        rot = state >> 59
        words.append(((xorshifted >> rot) | (xorshifted << (32 - rot))) & MASK32)
    return words


def _chacha12_words(key, count):
    """First `count` keystream u32 words of ChaCha12, counter 0, stream 0."""
    def rotl(v, n):
        return ((v << n) | (v >> (32 - n))) & MASK32

    def quarter(x, a, b, c, d):
        x[a] = (x[a] + x[b]) & MASK32; x[d] = rotl(x[d] ^ x[a], 16)
        x[c] = (x[c] + x[d]) & MASK32; x[b] = rotl(x[b] ^ x[c], 12)
        x[a] = (x[a] + x[b]) & MASK32; x[d] = rotl(x[d] ^ x[a], 8)
        x[c] = (x[c] + x[d]) & MASK32; x[b] = rotl(x[b] ^ x[c], 7)

    out = []
    block = 0
    while len(out) < count:
        state = [0x61707865, 0x3320646E, 0x79622D32, 0x6B206574, *key,
                 block & MASK32, block >> 32, 0, 0]
        x = list(state)
        for _ in range(6):
            quarter(x, 0, 4, 8, 12); quarter(x, 1, 5, 9, 13)
            quarter(x, 2, 6, 10, 14); quarter(x, 3, 7, 11, 15)
            quarter(x, 0, 5, 10, 15); quarter(x, 1, 6, 11, 12)
            quarter(x, 2, 7, 8, 13); quarter(x, 3, 4, 9, 14)
        out.extend((a + b) & MASK32 for a, b in zip(x, state))
        block += 1
    return out[:count]


def bits_value(bits):
    return sum(bit << j for j, bit in enumerate(bits))


def operands(width, seed, trial, pattern):
    """Return (x, y) as little-endian bit lists, exactly as inputs::operands."""
    state = (seed + trial + (width << 32)) & MASK64
    words = _chacha12_words(_pcg32_seed(state), 2 * width)
    x = [w >> 31 for w in words[:width]]
    y = [w >> 31 for w in words[width:]]
    if pattern == "random":
        pass
    elif pattern == "zero":
        x = [0] * width
    elif pattern == "max":
        x, y = [1] * width, [1] * width
    elif pattern == "alternating":
        x = [j % 2 for j in range(width)]
        y = [1 - b for b in x]
    elif pattern == "carry-chain":
        x, y = [1] * width, [0] * width
        y[0] = 1
        if width > 1:
            y[1] = 1
    else:
        raise ValueError(f"unknown pattern {pattern}")
    return x, y


# --- LUT-count model: triviality propagation of adapters/bitwise/src/circuits.rs
# A LUT input is trivial iff every ciphertext summed into it is trivial
# (Gates::zero() is a trivial encryption; encrypted operand bits never are).
# A LUT output is trivial iff its input is.  Triviality depends only on the
# public circuit structure, not on operand values.

class _Counter:
    def __init__(self):
        self.logical = self.real = 0

    def lut(self, *inputs, calls=1):
        trivial = all(inputs)
        self.logical += calls
        if not trivial:
            self.real += calls
        return trivial


def _morshed_adder(g, a, b, c):
    t1 = g.lut(a, c)
    t2 = g.lut(b, c)
    total = g.lut(a, t2)
    both = g.lut(t1, t2)
    return total, g.lut(c, both)


def _ripple_add(g, a, b):
    assert len(a) == len(b)
    carry, out = True, []
    for ai, bi in zip(a, b):
        total, carry = _morshed_adder(g, ai, bi, carry)
        out.append(total)
    return out


def _morshed(g, width, threads):
    groups = min(threads, width)
    sums = []
    for group in range(groups):
        acc = [True] * (2 * width)
        for i in range(group * width // groups, (group + 1) * width // groups):
            partial = [True] * (2 * width)
            for j in range(width):
                partial[i + j] = g.lut(False, False)
            acc = _ripple_add(g, acc, partial)
        sums.append(acc)
    while len(sums) > 1:
        sums = [_ripple_add(g, *sums[k:k + 2]) if k + 1 < len(sums) else sums[k]
                for k in range(0, len(sums), 2)]
    return sums[0]


def _trifan(g, width):
    row, carry = [True] * width, [True] * width
    for _ in range(width):
        partial = [g.lut(False, False) for _ in range(width)]
        reduced = [g.lut(row[j], partial[j], carry[j], calls=2) for j in range(width)]
        row, carry = reduced[1:] + [True], reduced


def _trifan_pruned(g, width):
    row, carry = [True] * width, [True] * width
    for _ in range(width):
        active = len(row)
        sums, carries = [], []
        for j in range(active):
            partial = g.lut(False, False)
            if j + 1 == active:
                sums.append(g.lut(row[j], partial, carry[j]))
            else:
                t = g.lut(row[j], partial, carry[j], calls=2)
                sums.append(t)
                carries.append(t)
        row, carry = sums[1:], carries


@functools.lru_cache(maxsize=None)
def lut_counts(method, width, threads):
    """(logical, trivial, real) LUT calls of one multiplication."""
    g = _Counter()
    if method == "morshed":
        _morshed(g, width, threads)
    elif method == "trifan":
        _trifan(g, width)
    elif method == "trifan-pruned":
        _trifan_pruned(g, width)
    else:
        raise ValueError(f"unknown circuit {method}")
    return g.logical, g.logical - g.real, g.real


def validate_run(directory):
    metadata = json.loads((directory / "parameters.json").read_text())
    args, params = metadata["arguments"], metadata["parameters"]
    assert metadata["tfhe_rs"] == "1.7.0"
    assert metadata["whole_multiplier_approved"] is False
    modulus = 4 if args["preset"] == "m2c2-gaussian-control" else 2
    assert params["message_modulus"] == params["carry_modulus"] == modulus
    assert params["log2_p_fail"] < -128
    preflight = json.loads((directory / "preflight.json").read_text())
    assert preflight == {"encrypted_full_adder_triples": 8, "both_adders_correct": True}
    with (directory / "timings.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    expected_grid = set(itertools.product(args["methods"], args["widths"],
        args["threads"], args["patterns"], range(args["warmup"] + args["repetitions"])))
    seen = set()
    groups = defaultdict(list)
    for row in rows:
        width, threads = int(row["width"]), int(row["threads"])
        key = (row["method"], width, threads, row["pattern"], int(row["trial"]))
        assert key not in seen
        seen.add(key)
        assert row["preset"] == args["preset"] and row["ok"] == "true"
        output_width = width * (2 if row["method"] == "morshed" else 1)
        assert int(row["output_width"]) == output_width
        x, y = int(row["x_hex"], 16), int(row["y_hex"], 16)
        assert 0 <= x < 1 << width and 0 <= y < 1 << width
        assert int(row["seed"]) == args["seed"]
        xb, yb = operands(width, args["seed"], int(row["trial"]), row["pattern"])
        assert (x, y) == (bits_value(xb), bits_value(yb)), "operands differ from seed derivation"
        expected = x * y % (1 << output_width)
        assert int(row["expected_hex"], 16) == int(row["output_hex"], 16) == expected
        logical = {"trifan": 3 * width**2,
                   "trifan-pruned": (3 * width**2 + width) // 2,
                   "morshed": width**2 + 10 * width * (width + min(width, threads) - 1)}[row["method"]]
        assert lut_counts(row["method"], width, threads)[0] == logical
        _, trivial, real = lut_counts(row["method"], width, threads)
        assert int(row["logical_lut_calls"]) == logical
        assert int(row["real_pbs"]) + int(row["trivial_lut_calls"]) == logical
        assert int(row["real_pbs"]) > 0
        assert int(row["real_pbs"]) == real and int(row["trivial_lut_calls"]) == trivial, \
            "real/trivial LUT split differs from the circuit structure"
        # TFHE-rs 1.7.0 pbs-stats counts trivial LUT evaluations as PBS.
        assert int(row["library_pbs_count"]) == logical
        assert row["input_state"] == ("fresh" if args["fresh_inputs"] else "bootstrapped")
        assert int(row["input_pbs"]) == (0 if args["fresh_inputs"] else 2 * width)
        assert (row["warmup"] == "true") == (int(row["trial"]) < args["warmup"])
        if row["warmup"] == "false":
            groups[key[:-1]].append(float(row["total_s"]))
    assert seen == expected_grid, "Missing or unexpected trials"
    summaries = []
    for (method, width, threads, pattern), values in groups.items():
        assert len(values) == args["repetitions"]
        assert all(v > 0 for v in values)
        summaries.append({"method": method, "preset": args["preset"], "width": width,
            "threads": threads, "pattern": pattern, "measured_trials": len(values),
            "mean_total_s": statistics.mean(values),
            "sample_sd_s": statistics.stdev(values) if len(values) > 1 else None})
    return {"correct_products": len(rows), "preflight_triples": 8,
            "whole_multiplier_approved": False, "rows": summaries}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/bitwise/release/bitwise-campaign")
    parser.add_argument("--presets", nargs="+", default=["m1c1-gaussian", "m1c1-tuniform"],
        choices=["m1c1-gaussian", "m1c1-tuniform", "m2c2-gaussian-control"])
    parser.add_argument("--widths", nargs="+", type=int, default=[8, 16])
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--patterns", nargs="+", default=["random", "max"],
        choices=["random", "max", "zero", "alternating", "carry-chain"])
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.binary = args.binary.resolve(strict=True)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    adapter = ROOT / "adapters/bitwise"
    sources = [adapter / "Cargo.toml", adapter / "Cargo.lock", *sorted((adapter / "src").glob("*.rs")), Path(__file__)]
    report = {"host": platform.platform(), "machine": platform.machine(),
        "kind": "local correctness checks, not a server comparison", "status": "running",
        "binary_sha256": sha256(args.binary),
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        "runs": []}
    archive = args.output / "tested-source.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for source in sources:
            bundle.add(source, arcname=str(source.relative_to(ROOT)))
    report["source_archive_sha256"] = sha256(archive)
    report_path = args.output / "summary.json"
    try:
        for preset in args.presets:
            command = [str(args.binary), "--preset", preset, "--allow-unverified-parameters",
                "--widths", ",".join(map(str, args.widths)),
                "--threads", ",".join(map(str, args.threads)),
                "--patterns", ",".join(args.patterns),
                "--repetitions", str(args.repetitions), "--warmup", str(args.warmup),
                "--output", str(args.output / preset)]
            report["current_command"] = command
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            with (args.output / f"{preset}.log").open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            checked = validate_run(args.output / preset)
            report["runs"].append(checked)
            print(json.dumps({"preset": preset, **checked}), flush=True)
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
