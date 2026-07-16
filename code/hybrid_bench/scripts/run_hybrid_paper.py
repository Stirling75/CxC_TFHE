#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BENCH_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = BENCH_ROOT.parents[1]
PARAMETERS_PATH = ARTIFACT_ROOT / "config" / "parameters.json"
class RunnerError(RuntimeError):
    pass


@dataclass(frozen=True)
class PaperPreset:
    label: str
    profile: str
    product_contract: str
    target_bits: int
    expected_log2: str
    row_bits: int
    output_row_bits: int
    chunk_cap: int
    pbs: tuple[int, int]
    key_switch: tuple[int, int]
    lift_pbs: tuple[int, int]
    lift_key_switch: tuple[int, int]
    automorphism: tuple[int, int]
    scheme_switch: tuple[int, int]
    circuit_bootstrap: tuple[int, int]
    prefix_bits: int

    @property
    def pgen_label(self) -> str:
        if self.label == "hybrid-4x4":
            return "CBS-CMUX-CHUNK4X4-TREE-PAR"
        return "CBS-CMUX-8X8-DIRECT-PAR"

    @property
    def output_contract_label(self) -> str:
        if self.label == "hybrid-4x4":
            return "CHUNK4X4-TREE-DIGITS"
        return "CHUNK8X8-DIRECT-DIGITS"


def pair(value: Any, name: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise RunnerError(f"{name} must contain exactly two integers")
    try:
        result = (int(value[0]), int(value[1]))
    except (TypeError, ValueError) as exc:
        raise RunnerError(f"{name} must contain exactly two integers") from exc
    if result[0] <= 0 or result[1] <= 0:
        raise RunnerError(f"{name} values must be positive")
    return result


def require_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise RunnerError(f"unsupported {name}: expected {expected!r}, got {actual!r}")


def load_preset(label: str) -> PaperPreset:
    try:
        config = json.loads(PARAMETERS_PATH.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"cannot read {PARAMETERS_PATH}: {exc}") from exc

    common = config.get("hybrid_common")
    item_key = {"hybrid-4x4": "hybrid_4x4", "hybrid-8x8": "hybrid_8x8"}[label]
    item = config.get(item_key)
    if not isinstance(common, dict) or not isinstance(item, dict):
        raise RunnerError("config/parameters.json is missing the hybrid paper presets")

    require_equal(
        "centered normalizer modulus switch",
        common.get("centered_normalizer_modulus_switch"),
        True,
    )
    target_bits = int(common.get("target_failure_bits"))
    expected_log2 = str(item.get("estimated_worst_union_log2_p_fail"))
    preset = PaperPreset(
        label=label,
        profile=str(common.get("tfhe_profile")),
        product_contract=(
            "chunk4x4-tree-digits"
            if label == "hybrid-4x4"
            else "chunk8x8-direct-digits"
        ),
        target_bits=target_bits,
        expected_log2=expected_log2,
        row_bits=int(common.get("row_bits")),
        output_row_bits=int(common.get("output_row_bits")),
        chunk_cap=int(common.get("normalizer_capacity")),
        pbs=pair(common.get("normalizer_pbs"), "hybrid_common.normalizer_pbs"),
        key_switch=pair(
            common.get("normalizer_key_switch"), "hybrid_common.normalizer_key_switch"
        ),
        lift_pbs=pair(item.get("selector_lift_pbs"), f"{item_key}.selector_lift_pbs"),
        lift_key_switch=pair(
            item.get("selector_lift_key_switch"), f"{item_key}.selector_lift_key_switch"
        ),
        automorphism=pair(common.get("automorphism"), "hybrid_common.automorphism"),
        scheme_switch=pair(common.get("scheme_switch"), "hybrid_common.scheme_switch"),
        circuit_bootstrap=pair(
            common.get("circuit_bootstrap"), "hybrid_common.circuit_bootstrap"
        ),
        prefix_bits=int(item.get("lookup_prefix_bits", 8)),
    )

    if float(preset.expected_log2) > -preset.target_bits:
        raise RunnerError(
            f"{label} recorded failure estimate 2^{preset.expected_log2} "
            f"does not meet 2^-{preset.target_bits}"
        )
    return preset


def paper_environment(preset: PaperPreset) -> dict[str, str]:
    return {
        "MALLOC_ARENA_MAX": "2",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "CBS_PARAM_PROFILE": preset.profile,
        "CBS_FAILURE_PRESET": preset.label,
        "CBS_FAILURE_TARGET_BITS": str(preset.target_bits),
        "CBS_FAILURE_EXPECTED_WORST_LOG2": preset.expected_log2,
        "CBS_FAILURE_EXPECTED_ACCEPTANCE_LOG2": preset.expected_log2,
        "CBS_FAILURE_EXPECTED_UNION_LOG2": preset.expected_log2,
        "PBS_BASE_LOG": str(preset.pbs[0]),
        "PBS_LEVEL": str(preset.pbs[1]),
        "KS_BASE_LOG": str(preset.key_switch[0]),
        "KS_LEVEL": str(preset.key_switch[1]),
        "CBS_LIFT_PBS_BASE_LOG": str(preset.lift_pbs[0]),
        "CBS_LIFT_PBS_LEVEL": str(preset.lift_pbs[1]),
        "CBS_LIFT_KS_BASE_LOG": str(preset.lift_key_switch[0]),
        "CBS_LIFT_KS_LEVEL": str(preset.lift_key_switch[1]),
        "DIRECT_AUTO_BASE_LOG": str(preset.automorphism[0]),
        "DIRECT_AUTO_LEVEL": str(preset.automorphism[1]),
        "DIRECT_SS_BASE_LOG": str(preset.scheme_switch[0]),
        "DIRECT_SS_LEVEL": str(preset.scheme_switch[1]),
        "DIRECT_CBS_BASE_LOG": str(preset.circuit_bootstrap[0]),
        "DIRECT_CBS_LEVEL": str(preset.circuit_bootstrap[1]),
        "DIRECT_LOG_LUT_COUNT": "2",
        "CBS_CENTERED_MS": "0",
        "CBS_PRODUCT_MODE": "cmux-lut",
        "CBS_DIGIT_LIFT": "direct-revhomtrace",
        "CBS_CHUNK4X4_TREE_PRODUCT": "1" if preset.label == "hybrid-4x4" else "0",
        "CBS_CHUNK4X4_SPLIT_PRODUCT": "0",
        "CBS_CHUNK8X8_DIRECT_PRODUCT": "1" if preset.label == "hybrid-8x8" else "0",
        "CBS_CHUNK8X8_PREFIX_BITS": str(preset.prefix_bits),
        "CBS_CHUNK4X4_SPLIT_PREFIX_BITS": str(preset.prefix_bits),
        "CBS_PARALLEL_DIGIT_LIFT": "1",
        "CBS_PARALLEL_PRODUCT_LIFT": "1",
        "CMUX_ROW_BITS": str(preset.row_bits),
        "CBS_OUTPUT_ROW_BITS": str(preset.output_row_bits),
        "CBS_SHARED_LEFT": "0",
        "CBS_REUSE_LEFT_LIFTS": "0",
        "CBS_PARALLEL_CMUX_CELLS": "1",
        "CBS_NORMALIZER_CENTERED_MS": "1",
        "CBS_NORMALIZER_CHUNK_CAP": str(preset.chunk_cap),
    }


def binary_path() -> Path:
    target = Path(os.environ.get("CARGO_TARGET_DIR", "target"))
    if not target.is_absolute():
        target = BENCH_ROOT / target
    return target / "release" / "hybrid_cxc"


def parse_positive_csv(value: str, name: str, *, even: bool = False) -> list[int]:
    values: list[int] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            number = int(raw)
        except ValueError as exc:
            raise RunnerError(
                f"{name} must be a comma-separated list of positive integers"
            ) from exc
        if number <= 0 or (even and number % 2 != 0):
            qualifier = "positive even integers" if even else "positive integers"
            raise RunnerError(f"{name} must contain only {qualifier}")
        values.append(number)
    if not values:
        raise RunnerError(f"{name} must not be empty")
    return values


def zero_or_one(value: str) -> bool:
    if value not in ("0", "1"):
        raise argparse.ArgumentTypeError("expected 0 or 1")
    return value == "1"


def absolute(path: Path) -> Path:
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def expected_row_values(
    preset: PaperPreset, width: int, threads: int
) -> dict[str, str]:
    return {
        "width_bits": str(width),
        "product_count": "1",
        "digit_lift_schedule": "parallel-product-digits",
        "rayon_threads": str(threads),
        "normalizer_mode": "height2-parallel-add-token",
        "failure_preset": preset.label,
        "param_profile": preset.profile,
        "pbs_base_log": str(preset.pbs[0]),
        "pbs_level": str(preset.pbs[1]),
        "cbs_lift_pbs_base_log": str(preset.lift_pbs[0]),
        "cbs_lift_pbs_level": str(preset.lift_pbs[1]),
        "ks_base_log": str(preset.key_switch[0]),
        "ks_level": str(preset.key_switch[1]),
        "cbs_lift_ks_base_log": str(preset.lift_key_switch[0]),
        "cbs_lift_ks_level": str(preset.lift_key_switch[1]),
        "direct_cbs_base_log": str(preset.circuit_bootstrap[0]),
        "direct_cbs_level": str(preset.circuit_bootstrap[1]),
        "direct_auto_base_log": str(preset.automorphism[0]),
        "direct_auto_level": str(preset.automorphism[1]),
        "direct_ss_base_log": str(preset.scheme_switch[0]),
        "direct_ss_level": str(preset.scheme_switch[1]),
        "row_bits": str(preset.row_bits),
        "output_row_bits": str(preset.output_row_bits),
        "normalizer_chunk_cap": str(preset.chunk_cap),
        "normalizer_centered_ms": "1",
        "failure_target_bits": str(preset.target_bits),
        "failure_expected_worst_log2": preset.expected_log2,
        "failure_expected_acceptance_log2": preset.expected_log2,
        "failure_expected_union_log2": preset.expected_log2,
        "case_type": "cmux_lut_product",
        "pgen": preset.pgen_label,
        "nkernel": "K-LOWCARRY1",
        "output_contract": preset.output_contract_label,
        "reuse_model": "none",
        "ok": "true",
    }


def row_matches(
    row: dict[str, str], expected: dict[str, str], preset: PaperPreset
) -> bool:
    for key, value in expected.items():
        actual = (row.get(key) or "").lower() if key == "ok" else row.get(key)
        if actual != value:
            return False
    label = row.get("label", "")
    if preset.label == "hybrid-8x8" and not label.endswith(f"_p{preset.prefix_bits}"):
        return False
    return True


def rewrite_csv(
    path: Path, fieldnames: list[str], rows: list[dict[str, str]], drop: set[int]
) -> None:
    temporary = path.with_name(path.name + ".resume.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row for index, row in enumerate(rows) if index not in drop)
    os.replace(temporary, path)


def case_complete(
    output: Path,
    preset: PaperPreset,
    width: int,
    threads: int,
    repetitions: int,
) -> bool:
    if not output.is_file() or output.stat().st_size == 0:
        return False
    with output.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames:
        raise RunnerError(f"cannot resume malformed CSV without a header: {output}")

    expected = expected_row_values(preset, width, threads)
    matched_indices: list[int] = []
    unique_indices: list[int] = []
    labels: set[str] = set()
    unlabeled = 0
    for index, row in enumerate(rows):
        if not row_matches(row, expected, preset):
            continue
        matched_indices.append(index)
        label = row.get("label", "")
        if label:
            if label not in labels:
                labels.add(label)
                unique_indices.append(index)
        else:
            unlabeled += 1
            unique_indices.append(index)

    count = len(labels) + unlabeled
    if count >= repetitions:
        drop = set(matched_indices) - set(unique_indices)
        complete = True
    else:
        drop = set(matched_indices)
        complete = False
    if drop:
        rewrite_csv(output, fieldnames, rows, drop)
        print(
            f"resume cleanup: removed {len(drop)} duplicate/incomplete rows from {output}",
            file=sys.stderr,
        )
    return complete


def initialize_outputs(
    output: Path, failures: Path, log_dir: Path, resume: bool
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    failures.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    if not resume or not output.is_file() or output.stat().st_size == 0:
        output.write_text("")
    if not resume or not failures.is_file() or failures.stat().st_size == 0:
        with failures.open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(
                (
                    "status",
                    "threads",
                    "width",
                    "k",
                    "digit_lift",
                    "mode",
                    "elapsed_sec",
                    "log",
                )
            )


def append_failure(
    failures: Path,
    status: int,
    threads: int,
    width: int,
    elapsed_sec: int,
    log: Path,
) -> None:
    with failures.open("a", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (status, threads, width, 1, "product", "height2", elapsed_sec, log)
        )


def tail_log(path: Path, lines: int = 40) -> None:
    try:
        content = path.read_text(errors="replace").splitlines()
    except OSError:
        return
    for line in content[-lines:]:
        print(line, file=sys.stderr)


def describe(
    preset: PaperPreset,
    widths: list[int],
    threads: list[int],
    repetitions: int,
    timeout_sec: int,
    resume: bool,
    continue_on_error: bool,
    output: Path,
    failures: Path,
    log_dir: Path,
    stamp: str,
) -> dict[str, Any]:
    binary = binary_path()
    return {
        "preset": preset.label,
        "product_contract": preset.product_contract,
        "normalizer_schedule": "height2",
        "normalizer_kernel": "refresh-digit",
        "recorded_worst_union_log2_p_fail": preset.expected_log2,
        "widths": widths,
        "threads": threads,
        "repetitions": repetitions,
        "timeout_sec": timeout_sec,
        "resume": resume,
        "continue_on_error": continue_on_error,
        "output": str(output),
        "failures": str(failures),
        "log_dir": str(log_dir),
        "stamp": stamp,
        "binary": str(binary),
        "parameter_environment": paper_environment(preset),
        "cases": [
            {
                "width": width,
                "threads": thread_count,
                "argv": [str(binary), str(width), "1", str(repetitions)],
                "runtime_environment": {
                    "RAYON_NUM_THREADS": str(thread_count),
                    "CBS_TIMING_CSV": str(output),
                },
            }
            for width in widths
            for thread_count in threads
        ],
    }


def print_configuration(description: dict[str, Any], dry_run: bool) -> None:
    environment = description["parameter_environment"]
    print(
        f"{description['preset']}: "
        f"widths={','.join(str(value) for value in description['widths'])} "
        f"threads={','.join(str(value) for value in description['threads'])} "
        f"repetitions={description['repetitions']}"
    )
    print(
        f"  PBS=({environment['PBS_BASE_LOG']},{environment['PBS_LEVEL']}) "
        f"lift-PBS=({environment['CBS_LIFT_PBS_BASE_LOG']},"
        f"{environment['CBS_LIFT_PBS_LEVEL']}) "
        f"cap={environment['CBS_NORMALIZER_CHUNK_CAP']} "
        f"log2(p_fail)<={description['recorded_worst_union_log2_p_fail']}"
    )
    print(f"  output={description['output']}")
    if dry_run:
        print("  dry run")


def build_hybrid_binary() -> Path:
    command = ["cargo", "build", "--release", "--locked", "--bin", "hybrid_cxc"]
    print(f"build command: {shlex.join(command)}")
    completed = subprocess.run(
        command,
        cwd=BENCH_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout, file=sys.stderr, end="")
        raise RunnerError(f"hybrid binary build failed with status {completed.returncode}")
    binary = binary_path()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RunnerError(f"cargo build did not produce an executable at {binary}")
    print(f"binary ready: {binary}")
    return binary


def run_cases(
    preset: PaperPreset,
    description: dict[str, Any],
    output: Path,
    failures: Path,
    log_dir: Path,
) -> int:
    resume = bool(description["resume"])
    continue_on_error = bool(description["continue_on_error"])
    timeout_sec = int(description["timeout_sec"])
    repetitions = int(description["repetitions"])
    binary = build_hybrid_binary()
    initialize_outputs(output, failures, log_dir, resume)

    base_env = os.environ.copy()
    base_env.update(description["parameter_environment"])
    base_env.update(
        {
            "CBS_WIDTH_LIST": ",".join(str(value) for value in description["widths"]),
            "CBS_THREAD_LIST": ",".join(str(value) for value in description["threads"]),
            "CBS_K_LIST": "1",
            "CBS_TRIALS": str(repetitions),
            "CBS_NORMALIZER_MODES": "height2",
            "CBS_DIGIT_LIFT_SCHEDULES": "product",
            "CBS_ARTIFACT_LABEL": preset.label,
            "CBS_SWEEP_STAMP": str(description["stamp"]),
        }
    )

    for width in description["widths"]:
        for threads in description["threads"]:
            if resume and case_complete(output, preset, width, threads, repetitions):
                print(
                    f"skip complete threads={threads} width={width} k=1 digit_lift=product mode=height2"
                )
                continue

            log = log_dir / (
                f"cbs-{preset.label}-w{width}-t{threads}-k1-"
                "parallel-product-digits-height2-parallel-add-token.log"
            )
            command = [str(binary), str(width), "1", str(repetitions)]
            case_env = base_env.copy()
            case_env.update(
                {
                    "RAYON_NUM_THREADS": str(threads),
                    "CBS_TIMING_CSV": str(output),
                }
            )
            print(
                f"run threads={threads} width={width} k=1 digit_lift=product mode=height2 log={log}"
            )
            started = time.monotonic()
            status = 0
            try:
                with log.open("w") as handle:
                    try:
                        completed = subprocess.run(
                            command,
                            cwd=BENCH_ROOT,
                            env=case_env,
                            stdout=handle,
                            stderr=subprocess.STDOUT,
                            timeout=timeout_sec or None,
                            check=False,
                        )
                        status = completed.returncode
                    except subprocess.TimeoutExpired:
                        handle.write(f"\ncase timed out after {timeout_sec} seconds\n")
                        status = 124
                    except OSError as exc:
                        handle.write(f"\nfailed to execute {command[0]}: {exc}\n")
                        status = 127
            except KeyboardInterrupt:
                print("hybrid case interrupted; stopping the sweep", file=sys.stderr)
                return 130

            if status < 0:
                status = 128 + abs(status)
            if status in (130, 143):
                print(
                    f"hybrid case interrupted with status={status}; stopping the sweep",
                    file=sys.stderr,
                )
                return status
            elapsed_sec = int(time.monotonic() - started)
            if status != 0:
                append_failure(failures, status, threads, width, elapsed_sec, log)
                print(
                    f"case failed status={status} threads={threads} width={width} k=1 "
                    f"digit_lift=product mode=height2 elapsed={elapsed_sec}s log={log}",
                    file=sys.stderr,
                )
                tail_log(log)
                if not continue_on_error:
                    return status
            else:
                print(
                    f"case ok threads={threads} width={width} k=1 digit_lift=product "
                    f"mode=height2 elapsed={elapsed_sec}s log={log}"
                )
    print(f"done {output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the released hybrid-4x4 or hybrid-8x8 paper benchmark preset."
    )
    parser.add_argument("preset", choices=("hybrid-4x4", "hybrid-8x8"))
    parser.add_argument("--widths", default="16,32,64,128,256")
    parser.add_argument("--threads", default="1,2,4,8,16,32,64")
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--failures", type=Path)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--stamp")
    parser.add_argument("--resume", type=zero_or_one, default=True, metavar="0|1")
    parser.add_argument(
        "--continue-on-error", type=zero_or_one, default=False, metavar="0|1"
    )
    parser.add_argument("--timeout-sec", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--describe-json",
        action="store_true",
        help="emit the resolved dry-run contract as JSON (used by parameter verification)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repetitions <= 0:
        raise RunnerError("--repetitions must be a positive integer")
    if args.timeout_sec < 0:
        raise RunnerError("--timeout-sec must be zero or a positive integer")
    if args.describe_json and not args.dry_run:
        raise RunnerError("--describe-json requires --dry-run")

    widths = parse_positive_csv(args.widths, "--widths", even=True)
    threads = parse_positive_csv(args.threads, "--threads")
    preset = load_preset(args.preset)
    output = absolute(args.output or (BENCH_ROOT / "results" / f"{args.preset}.csv"))
    failures = absolute(args.failures or output.with_suffix(".failures.tsv"))
    log_dir = absolute(args.log_dir or (output.parent / "logs" / args.preset))
    stamp = args.stamp or f"paper-{args.preset}"
    description = describe(
        preset,
        widths,
        threads,
        args.repetitions,
        args.timeout_sec,
        args.resume,
        args.continue_on_error,
        output,
        failures,
        log_dir,
        stamp,
    )

    if args.describe_json:
        json.dump(description, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    print_configuration(description, args.dry_run)
    if args.dry_run:
        for case in description["cases"]:
            print(
                f"dry-run case width={case['width']} threads={case['threads']} "
                f"RAYON_NUM_THREADS={case['runtime_environment']['RAYON_NUM_THREADS']} "
                f"command={shlex.join(case['argv'])}"
            )
        print(
            f"dry-run: {len(description['cases'])} case(s); cryptographic benchmark skipped"
        )
        return 0
    return run_cases(preset, description, output, failures, log_dir)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunnerError as exc:
        print(f"hybrid paper runner failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except subprocess.CalledProcessError as exc:
        print(
            f"hybrid paper runner failed: command exited with status {exc.returncode}",
            file=sys.stderr,
        )
        raise SystemExit(exc.returncode or 1)
    except OSError as exc:
        print(f"hybrid paper runner failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
