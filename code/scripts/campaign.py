#!/usr/bin/env python3
"""Build, inspect, and run the CPU multiplication campaign."""
import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

from cases import BFV, CATALOG, CRATES, DEFAULT, ROOT, binary_for, crate_for, prepare, resolve
from results import check
from runtime import available_cpus, environment, execute, host, sha, source_id, source_record, write_json


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["list", "doctor", "fetch", "build", "test", "verify", "smoke", "bench"])
    parser.add_argument("--methods", nargs="+", choices=sorted(CATALOG), default=DEFAULT)
    parser.add_argument("--widths", nargs="+", type=int, choices=[16,32,64,128,256], default=[16])
    parser.add_argument("--threads", nargs="+", type=positive, default=[1])
    parser.add_argument("--cpu-ids", nargs="+", type=int,
                        help="ordered Linux CPU IDs; each case uses the first T IDs")
    parser.add_argument("--repetitions", type=positive, default=2,
                        help="measured products per case, excluding warm-up (default: 2)")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--jobs", type=positive, default=2, help="Cargo jobs, separate from benchmark threads")
    parser.add_argument("--build-root", type=Path, default=ROOT / "build")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--research", action="store_true", help="acknowledge unresolved whole-multiplier bounds")
    parser.add_argument("--allow-unpinned", action="store_true", help="Mac diagnostic only; Linux always enforces affinity")
    parser.add_argument("--allow-failing-screen", action="store_true",
                        help="measure cases whose packaged failure screen misses 2^-128, labelled in summary.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or not 0 <= args.seed < 2**64:
        parser.error("warmup must be nonnegative; seed must fit u64")
    for values in (args.methods, args.widths, args.threads):
        if len(values) != len(set(values)):
            parser.error("duplicate methods, widths, or thread counts are not allowed")
    args.build_root = args.build_root.resolve()
    if args.cpu_ids is not None:
        if (not args.cpu_ids or len(set(args.cpu_ids)) != len(args.cpu_ids) or
                not set(args.cpu_ids).issubset(available_cpus())):
            parser.error("CPU IDs must be distinct members of the available CPU set")
        if not hasattr(os, "sched_setaffinity"):
            parser.error("explicit CPU IDs require Linux affinity support")
    if args.action == "smoke":
        args.widths, args.threads, args.repetitions, args.warmup = [16], [1], 1, 0
    if args.action in ("bench", "smoke"):
        if not args.research and not args.dry_run:
            parser.error("no whole-multiplier-certified configuration is available; use --research for this campaign")
        if not args.output:
            parser.error("--output NEW_DIRECTORY is required")
        if max(args.threads) > min(64, len(available_cpus())):
            parser.error("requested threads exceed the available CPU budget (maximum 64)")
        if args.cpu_ids is not None and max(args.threads) > len(args.cpu_ids):
            parser.error("requested threads exceed the explicitly selected CPU budget")
        if not hasattr(os, "sched_setaffinity") and not (args.allow_unpinned or args.dry_run):
            parser.error("CPU affinity unavailable; --allow-unpinned is required for local diagnostics")
    return args


def cargo(args):
    if not shutil.which("cargo"):
        raise RuntimeError("cargo is required; install a stable Rust toolchain")
    records = source_record()
    if args.action == "test":
        directories = [("runner", ROOT / "scripts"),
                                ("analysis", ROOT / "code/server-campaign"),
                                ("plan", ROOT / "code/ring-variant-multiplier/model")]
        if any(crate_for(m) == "bfv" for m in args.methods):
            directories.append(("bfv", BFV))
        for name, directory in directories:
            command = [sys.executable, "-B", "-m", "unittest", "discover", "-s", str(directory), "-p", "test_*.py"]
            result = execute(command, ROOT, environment(), args.build_root / f"test-{name}.log")
            if result["exit_code"]:
                raise RuntimeError(f"Python tests failed: {args.build_root / ('test-' + name + '.log')}")
    for crate in sorted({crate_for(m) for m in args.methods}):
        target = args.build_root / crate
        target.mkdir(parents=True, exist_ok=True)
        previous = target / "build-record.json"
        previous_id = json.loads(previous.read_text()).get("source_id") if previous.exists() else None
        if args.action == "build" and previous_id != source_id(records):
            package = {"ring": "ring-variant-multiplier", "bitwise": "bitwise-campaign",
                       "parmesan": "parmesan-campaign", "bfv": "bfv-style-probe"}[crate]
            # Relocated archives can preserve mtimes older than cached objects.
            # Invalidate only this package, retaining downloaded dependency builds.
            clean = ["cargo", "clean", "--package", package, "--release", "--offline",
                     "--manifest-path", str(CRATES[crate]), "--target-dir", str(target)]
            cleaned = execute(clean, ROOT, environment(), target / "clean.log")
            if cleaned["exit_code"]:
                raise RuntimeError(f"Cargo clean failed: {target / 'clean.log'}")
        command = ["cargo", args.action, "--locked", "--offline", "--manifest-path", str(CRATES[crate])]
        if args.action != "fetch":
            command += ["--release", "--target-dir", str(target), "--jobs", str(args.jobs)]
            if args.action == "build":
                command += ["--bins"]
        print(f"{args.action}: {crate}", flush=True)
        result = execute(command, ROOT, environment(), target / f"{args.action}.log")
        write_json(target / f"{args.action}-result.json", result)
        if result["exit_code"]:
            raise RuntimeError(f"Cargo failed: {target / (args.action + '.log')}")
        if args.action == "build":
            if source_id(source_record()) != source_id(records):
                raise RuntimeError("source changed during build; rebuild before benchmarking")
            names = [m for m in CATALOG if crate_for(m) == crate]
            write_json(target / "build-record.json", {
                "source_id": source_id(records), "source_sha256": records, "host": host(),
                "rustc": subprocess.check_output(["rustc", "--version", "--verbose"], text=True),
                "rustflags": os.environ.get("RUSTFLAGS", ""),
                "encoded_rustflags": os.environ.get("CARGO_ENCODED_RUSTFLAGS", ""),
                "binaries": {binary_for(m, args.build_root).name: sha(binary_for(m, args.build_root)) for m in names}})


def verify_build(method, args, identity):
    target = args.build_root / crate_for(method)
    record = json.loads((target / "build-record.json").read_text())
    binary = binary_for(method, args.build_root)
    if record["source_id"] != identity or record["binaries"].get(binary.name) != sha(binary):
        raise RuntimeError("source or binary changed since build; run build again")


def run(args):
    # Check every packaged failure screen before creating output or starting timings.
    failing = [f"{m} W={w}" for m in args.methods for w in args.widths
               if w in CATALOG[m]["widths"] and resolve(m, w)["screen_failed"]]
    if failing and not args.allow_failing_screen:
        raise RuntimeError("failure screen misses 2^-128 for " + ", ".join(failing)
                           + "; rerun with --allow-failing-screen to measure them with that label")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    records = source_record()
    identity = source_id(records)
    write_json(output / "campaign.json", {
        "host": host(), "methods": args.methods, "widths": args.widths, "threads": args.threads,
        "repetitions": args.repetitions, "warmup": args.warmup, "seed": args.seed,
        "dry_run": args.dry_run, "research": args.research, "whole_multiplier_approved": False,
        "allow_failing_screen": args.allow_failing_screen,
        "source_id": identity, "source_sha256": records})
    if not args.dry_run:
        for method in args.methods:
            verify_build(method, args, identity)
    summaries, skipped = [], []
    for cell, (width, threads) in enumerate((w, t) for w in args.widths for t in args.threads):
        # Rotate method order across width/thread cells; do not run concurrent timings.
        offset = cell % len(args.methods)
        for method in args.methods[offset:] + args.methods[:offset]:
            if width not in CATALOG[method]["widths"]:
                skipped.append({"method": method, "width": width, "threads": threads, "reason": "unsupported width"})
                write_json(output / "skipped.json", skipped)
                continue
            case = output / f"{method}-w{width}-t{threads}"
            case.mkdir()
            resolved, command, overrides, cwd = prepare(method, width, threads, args, case)
            cpus = (args.cpu_ids or available_cpus())[:threads] if hasattr(os, "sched_setaffinity") else None
            record = {"command": command, "environment_overrides": overrides, "cpu_ids": cpus,
                      "thread_budget": threads, "affinity_enforced": cpus is not None,
                      "resolved": resolved, "whole_multiplier_approved": False}
            write_json(case / "run.json", record)
            estimate = resolved["whole_product_log2_estimate"]
            target = ("failure probability unverifiable" if math.isnan(estimate) else
                      "meets 2^-128" if resolved["failure_target_met"] else "MISSES 2^-128")
            print(f"{method}: W={width}, T={threads}, {CATALOG[method]['failure_status']}, "
                  f"estimate {resolved['whole_product_log2_estimate']:.1f} ({target})", flush=True)
            if args.dry_run:
                continue
            record["binary_sha256"] = sha(Path(command[0]))
            try:
                record.update(execute(command, cwd, environment(overrides), case / "run.log", cpus))
                if record["exit_code"]:
                    raise RuntimeError(f"benchmark failed: {case / 'run.log'}")
                summary = check(method, width, threads, args, case, resolved)
                for field in ("whole_product_log2_estimate", "failure_target_met", "failure_basis"):
                    summary[field] = resolved[field]
                summary["affinity_enforced"] = cpus is not None
                summary["peak_rss_bytes"] = record["peak_rss_bytes"]
                summaries.append(summary)
                record["validated"] = True
                with (output / "summary.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=summaries[0])
                    writer.writeheader()
                    writer.writerows(summaries)
                print(f"  mean={summary['mean_seconds']:.6f}s, all outputs correct", flush=True)
            except BaseException as error:
                record["validated"] = False
                record["error"] = str(error)
                raise
            finally:
                write_json(case / "run.json", record)
    write_json(output / "completed.json", {"dry_run": args.dry_run, "validated_cases": len(summaries),
               "skipped": skipped, "whole_multiplier_approved": False})
    print(f"Results: {output}", flush=True)


def main():
    if not __debug__:
        raise RuntimeError("Python optimization disables checker assertions; run without -O/PYTHONOPTIMIZE")
    args = parse()
    if args.action == "list":
        for name, item in CATALOG.items():
            print(f"{name:28} {item['output']:29} {item['failure_status']}")
        return
    if args.action == "doctor":
        print(json.dumps(host(), indent=2))
        for command in ("python3", "cargo", "rustc", "cc"):
            print(f"{command}: {shutil.which(command) or 'MISSING'}")
        return
    if args.action == "verify":
        rows = [resolve(m, w) for m in args.methods for w in args.widths if w in CATALOG[m]["widths"]]
        if args.output:
            args.output.mkdir(parents=True, exist_ok=False)
            write_json(args.output / "analysis.json", rows)
        for row in rows:
            a = row.get("analysis", {})
            value = a.get("conditional_union_log2", a.get("conditional_log2_union", "not derived"))
            print(f"{row['method']} W={row['width']}: screen {value}; whole-product estimate "
                  f"{row['whole_product_log2_estimate']:.2f}; target met: {row['failure_target_met']}")
        return
    args.build_root.mkdir(parents=True, exist_ok=True)
    with (args.build_root / "campaign.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"another campaign or build holds {args.build_root / 'campaign.lock'}") from None
        if args.action in ("build", "fetch", "test"):
            cargo(args)
        else:
            run(args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, AssertionError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
