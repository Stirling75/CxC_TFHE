#!/usr/bin/env python3
"""Run the encrypted Shokri--Tsoutsos CxC benchmark."""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


BINARY = "shokri_tsoutsos_cxc"
PARAMETER = "SHOKRI_TSOUTSOS_TABLE1"


@dataclass(frozen=True)
class Case:
    width: int
    pattern: str
    trial: int
    seed: int
    x: str
    y: str


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def dense_operand(width: int) -> str:
    return "0x" + "f" * (width // 4)


def random_operand(width: int, rng: random.Random) -> str:
    return f"0x{rng.getrandbits(width):0{width // 4}x}"


def make_cases(
    widths: list[int], patterns: list[str], trials: int, seed: int
) -> list[Case]:
    rng = random.Random(seed)
    cases: list[Case] = []
    for width in widths:
        for pattern in patterns:
            for trial in range(1, trials + 1):
                if pattern == "dense":
                    x = y = dense_operand(width)
                elif pattern == "random":
                    x, y = random_operand(width, rng), random_operand(width, rng)
                else:
                    raise ValueError(f"unknown operand pattern: {pattern}")
                cases.append(Case(width, pattern, trial, seed, x, y))
    return cases


def command(repo: Path, case: Case, release: bool, features: str) -> list[str]:
    cmd = ["cargo", "run"]
    if release:
        cmd.append("--release")
    if features:
        cmd.extend(["--features", features])
    cmd.extend(
        [
            "--quiet",
            "--bin",
            BINARY,
            "--",
            str(case.width),
            case.x,
            case.y,
        ]
    )
    return cmd


def verify_parameter(repo: Path, release: bool, features: str) -> None:
    cmd = ["cargo", "run"]
    if release:
        cmd.append("--release")
    if features:
        cmd.extend(["--features", features])
    cmd.extend(["--quiet", "--bin", BINARY, "--", "--parameters"])
    completed = subprocess.run(
        cmd,
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    expected = f"parameter={PARAMETER}"
    if completed.returncode != 0 or expected not in completed.stdout:
        raise RuntimeError(
            f"parameter preflight failed; expected {expected!r}\n{completed.stdout}"
        )


def match_text(pattern: str, output: str) -> str:
    match = re.search(pattern, output)
    return match.group(1) if match else ""


def match_float(pattern: str, output: str) -> str:
    value = match_text(pattern, output)
    return f"{float(value):.3f}" if value else ""


def hex_field(name: str, output: str) -> str:
    return match_text(rf"{name}=0x([0-9a-fA-F]+)", output)


def elapsed_seconds(output: str) -> float | None:
    match = re.search(r"elapsed=([0-9.]+)(ns|us|µs|ms|s)", output)
    if not match:
        return None
    value = float(match.group(1))
    scale = {
        "s": 1.0,
        "ms": 1e-3,
        "us": 1e-6,
        "µs": 1e-6,
        "ns": 1e-9,
    }
    return value * scale[match.group(2)]


def summary_line(output: str) -> str:
    for line in output.splitlines():
        if "expected=0x" in line or "panicked" in line:
            return line.strip()
    return ""


def run_case(
    repo: Path,
    case: Case,
    release: bool,
    features: str,
    timeout_seconds: int,
) -> dict[str, str]:
    print(
        f"[run] width={case.width} pattern={case.pattern} trial={case.trial}",
        file=sys.stderr,
        flush=True,
    )
    try:
        completed = subprocess.run(
            command(repo, case, release, features),
            cwd=repo,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=timeout_seconds or None,
        )
        output = completed.stdout
        returncode = completed.returncode
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        output += f"\ncase timed out after {timeout_seconds} seconds\n"
        returncode = 124

    expected = hex_field("expected", output)
    got = hex_field("got", output)
    elapsed = elapsed_seconds(output)
    ok = returncode == 0 and bool(expected) and expected == got
    return {
        "width": str(case.width),
        "rayon_threads": os.environ.get("RAYON_NUM_THREADS", ""),
        "pattern": case.pattern,
        "trial": str(case.trial),
        "seed": str(case.seed),
        "parameter": PARAMETER,
        "ok": str(ok).lower(),
        "returncode": str(returncode),
        "elapsed_s": "" if elapsed is None else f"{elapsed:.9f}",
        "x": hex_field("x", output),
        "y": hex_field("y", output),
        "expected": expected,
        "got": got,
        "initial_lift_ms": match_float(
            r"initial CBS lift: [0-9]+ limbs, [0-9]+ 2-bit LWE lifts, ([0-9.]+) ms",
            output,
        ),
        "product_vp_outputs": match_text(
            r"product VP8x8: [0-9]+ byte-product groups, ([0-9]+) VP outputs",
            output,
        ),
        "product_vp_ms": match_float(
            r"product VP8x8: [0-9]+ byte-product groups, [0-9]+ VP outputs, ([0-9.]+) ms",
            output,
        ),
        "add16_calls": match_text(r"Add16CXC total: ([0-9]+) calls", output),
        "add16_ms": match_float(
            r"Add16CXC total: [0-9]+ calls, ([0-9.]+) ms", output
        ),
        "add16_lwe_lifts": match_text(
            r"Add16 CBS lift: [0-9]+ half-add lifts, ([0-9]+) LWE lifts", output
        ),
        "add16_lift_ms": match_float(
            r"Add16 CBS lift: [0-9]+ half-add lifts, [0-9]+ LWE lifts, ([0-9.]+) ms",
            output,
        ),
        "add16_vp_outputs": match_text(
            r"Add16 VP total: ([0-9]+) VP outputs", output
        ),
        "add16_vp_ms": match_float(
            r"Add16 VP total: [0-9]+ VP outputs, ([0-9.]+) ms", output
        ),
        "summary": summary_line(output),
    }


FIELDS = [
    "width",
    "rayon_threads",
    "pattern",
    "trial",
    "seed",
    "parameter",
    "ok",
    "returncode",
    "elapsed_s",
    "x",
    "y",
    "expected",
    "got",
    "initial_lift_ms",
    "product_vp_outputs",
    "product_vp_ms",
    "add16_calls",
    "add16_ms",
    "add16_lwe_lifts",
    "add16_lift_ms",
    "add16_vp_outputs",
    "add16_vp_ms",
    "summary",
]


def case_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row.get("parameter", ""),
        row.get("width", ""),
        row.get("pattern", ""),
        row.get("trial", ""),
        row.get("seed", ""),
        row.get("rayon_threads", ""),
    )


def expected_key(case: Case) -> tuple[str, ...]:
    return (
        PARAMETER,
        str(case.width),
        case.pattern,
        str(case.trial),
        str(case.seed),
        os.environ.get("RAYON_NUM_THREADS", ""),
    )


def successful_rows(path: Path) -> dict[tuple[str, ...], dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return {}
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            case_key(row): row
            for row in rows
            if row.get("ok", "").lower() == "true" and row.get("returncode") == "0"
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--widths", default="16,32,64,128,256")
    parser.add_argument("--patterns", default="random")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--features", default="multithread")
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.trials < 1:
        parser.error("--trials must be positive")
    if args.timeout < 0:
        parser.error("--timeout must be non-negative")

    repo = Path(args.repo).resolve()
    out = Path(args.out).resolve()
    widths = [int(value) for value in split_csv(args.widths)]
    cases = make_cases(widths, split_csv(args.patterns), args.trials, args.seed)
    if args.dry_run:
        for case in cases:
            print(" ".join(command(repo, case, not args.debug, args.features)))
        return 0

    verify_parameter(repo, not args.debug, args.features)
    previous = successful_rows(out) if args.resume else {}
    out.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in previous.values():
            writer.writerow({field: row.get(field, "") for field in FIELDS})
        handle.flush()

        for case in cases:
            key = expected_key(case)
            if key in previous:
                print(
                    f"[skip] width={case.width} pattern={case.pattern} trial={case.trial}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            row = run_case(
                repo, case, not args.debug, args.features, args.timeout
            )
            writer.writerow(row)
            handle.flush()
            failures += row["ok"] != "true"

    if failures:
        print(f"{failures} encrypted CxC trial(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
