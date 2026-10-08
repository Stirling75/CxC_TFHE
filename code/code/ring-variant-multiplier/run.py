#!/usr/bin/env python3
"""Research-only CBS ring-shape comparison with an unchanged radix normalizer."""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
# Stand-alone runner build directory (inside this crate); the campaign runner
# uses its own --build-root and never this directory.
TARGET = Path(os.environ.get("HYBRID_RUNNER_TARGET_DIR", ROOT / "target")).resolve()
sys.dont_write_bytecode = True
sys.path.insert(0,str(ROOT/"model"))
from make_plan import derive

def positive(value):
    n = int(value)
    if n < 1: raise argparse.ArgumentTypeError("must be positive")
    return n

def summarize(path):
    with path.open() as stream:
        measured = [r for r in csv.DictReader(stream) if r["warmup"]=="0"]
    if not measured: return
    samples = [float(r["total_ms"])/1000 for r in measured]
    print(f"{path.parent.name}: n={len(samples)} mean={statistics.mean(samples):.6f}s"
          f" sd={statistics.stdev(samples) if len(samples)>1 else 0:.6f}s",flush=True)

def _ceil_log2(value):
    return 0 if value <= 1 else (value - 1).bit_length()

def tfhe_add_uses_parallel(num_blocks, threads, full_modulus=16):
    """Mirror of TFHE-rs 1.6.1 `should_parallel_propagation_be_faster` (radix_parallel/add.rs)."""
    layer = lambda blocks: -(-blocks // threads)  # PARALLEL_LATENCY_PENALTY = 1
    latency = 3 * layer(num_blocks)
    grouping = full_modulus.bit_length() - 1
    carries = max(0, -(-num_blocks // grouping) - 1)
    sequential_depth = max(0, carries - 1) // (grouping - 1)
    hillis_steele_depth = 0 if carries == 0 else _ceil_log2(carries)
    if sequential_depth <= hillis_steele_depth:
        latency += sequential_depth * layer(grouping)
    else:
        space = 1
        for _ in range(_ceil_log2(num_blocks)):
            latency += layer(num_blocks - space)
            space *= 2
    return latency < num_blocks

def final_addition_pbs(digits):
    """(sequential, parallel) PBS calls of the clean radix addition, from the screen model."""
    sys.path.insert(0, str(ROOT / "model" / "source_snapshot"))
    import cbs_variance_estimator as estimator
    _, _, sequential, parallel = estimator.tfhe_final_add_event_multipliers(digits)
    return sequential, parallel

def normalization_counts(plan, threads):
    """Analytical normalizer events of the public plan: executed reduction PBS
    (digit plus retained carry), reduction key switches, row refreshes and the
    final radix addition.  Matches heterogeneous_screen.analyze event counts."""
    digits = plan["width"] // 2
    reduction = jobs = 0
    for wave in plan["waves"]:
        for q, record in enumerate(wave):
            linear = record.get("linear") or [False] * len(record["groups"])
            mvb = record.get("mvb") or [False] * len(record["groups"])
            for group, lin, multi in zip(record["groups"], linear, mvb):
                if len(group) < 3:
                    continue
                bound = sum(record["terms"][i]["bound"] for i in group)
                carry = int(bound // 4 > 0 and q + 1 < digits)
                if multi:
                    # One multi-value bootstrapping returns digit and carry.
                    jobs += 1
                    reduction += 1
                    continue
                # A linear digit (sum - 4*carry) runs only the carry PBS.
                jobs += int(carry or not lin)
                reduction += carry + int(not lin)
    final = plan["final_columns"]
    refresh = addition = 0
    if any(len(column) > 1 for column in final):
        refresh = 0 if plan.get("final_direct") else sum(
            not term["refreshed"] for column in final for term in column)
        sequential, parallel = final_addition_pbs(digits)
        addition = parallel if tfhe_add_uses_parallel(digits, threads) else sequential
    shared = plan["env"]["CBS_SHARED_PREPROCESSING"] == "1"
    return {"reduction_pbs": reduction, "jobs": jobs, "key_switches": jobs if shared else reduction,
            "row_refresh": refresh, "final_addition": addition}

def _milliseconds(row, key):
    value = float(row[key])
    if not math.isfinite(value) or value < 0:
        raise RuntimeError(f"invalid {key}: {row[key]}")
    return value

def validate_rows(path, plan, count, threads, analysis=None, seed=None, timed=True):
    """Check runtime rows against the public plan.

    `analysis` (heterogeneous_screen.analyze output) adds an exact check of the
    screen's normalizer event counts; `seed` checks operand provenance in
    parameters.json; `timed` rejects rows produced with audit callbacks.
    """
    with path.open() as stream: rows=list(csv.DictReader(stream))
    if len(rows)!=count: raise RuntimeError("incomplete trial results")
    counts = normalization_counts(plan, threads)
    if analysis is not None:
        if (analysis["reduction_pbs"], analysis["row_refresh"]) != (counts["reduction_pbs"], counts["row_refresh"]):
            raise RuntimeError("screen and plan normalizer event counts differ")
        final = analysis.get("final_addition_pbs")
        if final is not None and counts["final_addition"] not in (final["sequential"], final["parallel"]):
            raise RuntimeError("screen and plan final-addition PBS counts differ")
    if seed is not None:
        meta = json.loads((path.parent / "parameters.json").read_text())
        if meta.get("operand_pattern") != "random" or meta.get("random_operands_seed") != seed:
            raise RuntimeError("operand pattern/seed provenance mismatch")
    h=plan["width"]//8
    cmux=255*h+8*h*(h+1)//2 if plan["mode"]=="cached" else 263*h*(h+1)//2
    if plan["chunk_bits"]==4:
        h4=plan["width"]//4
        cmux=8*h4*(h4+1)//2
    if plan["mode"] == "fused":
        d=plan["width"]//2
        cmux=15*d*(d+1)//2
    direct=entries=hits=0
    if plan["mode"]=="fused" and plan["kernel"]!="tree":
        capacity=int(plan["env"]["FUSED_PRODUCTS_PER_GROUP"])
        if plan["kernel"]=="reuse-cache":
            offsets=[i for i in range(0,d,capacity) if i+capacity<d]
            entries=len(offsets)
            hits=sum(d-i-capacity+1 for i in offsets)
        saved=hits-entries
        cmux=6*d*(d+1)//2-3*saved
        direct=3*d*(d+1)//2-3*saved
    ks=counts["key_switches"]
    lifts=plan["width"]//2 if plan["squaring"] else plan["width"]
    expected={"ok":"true","width_bits":str(plan["width"]),"rayon_threads":str(threads),
              "cmux_count":str(cmux),"cbs_lifts":str(lifts),"reduction_key_switches":str(ks),
              "cached_product_table":str(int(plan["mode"]=="cached")),
              "squaring":str(int(plan["squaring"])),
              "normalization_reduction_pbs":str(counts["reduction_pbs"]),
              "normalization_final_pbs":str(counts["row_refresh"]+counts["final_addition"]),
              "normalization_pbs":str(counts["reduction_pbs"]+counts["row_refresh"]+counts["final_addition"]),
              "lift_centered_ms":plan["env"]["CBS_CENTERED_MS"],
              "fused_products_per_group":plan["env"]["FUSED_PRODUCTS_PER_GROUP"]}
    expected.update({"fused_kernel":plan["kernel"], "direct_external_products":str(direct),
        "external_product_equivalents":str(cmux+direct), "prefix_cache_entries":str(entries),
        "prefix_cache_hits":str(hits), "direct_auto_fft":plan["env"]["DIRECT_AUTO_FFT"]})
    n, k = map(int, plan["env"]["CBS_RING_SHAPE"].split("x"))
    expected.update({"cbs_polynomial_size": str(n), "cbs_glwe_dimension": str(k),
                     "normalizer_polynomial_size": "2048", "normalizer_glwe_dimension": "1",
                     "input_lwe_dimension": "866"})
    if "CBS_SECRET_LAYOUT" in plan["env"]:
        expected["cbs_secret_layout"] = plan["env"]["CBS_SECRET_LAYOUT"]
    for prefix in ("DIRECT_CBS", "DIRECT_SS", "DIRECT_AUTO", "CBS_LIFT_PBS", "CBS_LIFT_KS", "PBS", "KS"):
        for suffix in ("BASE_LOG", "LEVEL"):
            key=prefix+"_"+suffix
            expected[key.lower()]=plan["env"][key]
    if timed:
        expected["audit_callbacks"] = "0"
    for row in rows:
        for key,value in expected.items():
            if row[key]!=value: raise RuntimeError(f"unexpected {key}: {row[key]} != {value}")
        total = _milliseconds(row, "total_ms")
        if total <= 0: raise RuntimeError("non-positive total_ms")
        phases = _milliseconds(row, "product_generation_ms") + _milliseconds(row, "normalization_ms")
        if abs(total - phases) > 1e-5 + 1e-12 * total:
            raise RuntimeError(f"total_ms {total} differs from product+normalization {phases}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("fused","cached","baseline","all"),default="fused")
    parser.add_argument("--chunk-bits",choices=(4,8),type=int,default=8,
                        help="4x4 is supported in baseline mode; cached/fused modes use the 8x8 lift profile")
    parser.add_argument("--kernels",nargs="+",choices=("tree","reuse","reuse-cache"),default=["reuse-cache"])
    parser.add_argument("--group-size",type=int,choices=range(1,46),default=24)
    parser.add_argument("--ring", choices=("2048x1", "1024x2"), default="2048x1")
    parser.add_argument("--secret-layout", choices=("contiguous", "even-odd"), default="contiguous")
    parser.add_argument("--cbs-level",type=int,choices=(6,7,8),default=7)
    parser.add_argument("--lift-ks", nargs=2, type=positive, metavar=("BASE_LOG", "LEVEL"),
                        help="experimental selector-input key-switch decomposition")
    parser.add_argument("--auto", nargs=2, type=positive, metavar=("BASE_LOG", "LEVEL"))
    parser.add_argument("--ss", nargs=2, type=positive, metavar=("BASE_LOG", "LEVEL"))
    parser.add_argument("--conditional-screen", action="store_true",
                        help="record the incomplete Gaussian sensitivity screen, never a failure certificate")
    parser.add_argument("--auto-fft",choices=("vanilla", "split40"),default="vanilla",
                        help="automorphism key-switch FFT only; keeps the public schedule and diagnostic reference fixed")
    parser.add_argument("--widths",nargs="+",type=int,choices=(16,32,64,128,256),default=[16])
    parser.add_argument("--threads",nargs="+",type=positive,default=[1])
    parser.add_argument("--repetitions",type=positive,default=3)
    parser.add_argument("--warmup",type=int,default=1)
    parser.add_argument("--noise-audit",action="store_true",help="diagnostic build; not latency measurements")
    parser.add_argument("--step-audit",action="store_true",help="decompose selected-path rounding and selector/FFT errors (requires --noise-audit)")
    parser.add_argument("--exploratory",action="store_true",help="acknowledge the unresolved shared selector-variance model")
    parser.add_argument("--pattern",choices=("random","zero","max","alternating"),default="random")
    parser.add_argument("--seed",type=int,default=20260905)
    parser.add_argument("--key-id",type=int,default=0,help="diagnostic label, not a key generation seed")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--verify-pbs",action="store_true",
                        help="validation only: compare shared PBS ciphertexts to two separate calls")
    args = parser.parse_args()
    if args.secret_layout == "even-odd" and args.ring != "1024x2":
        parser.error("even-odd secret layout requires --ring 1024x2")
    for decomposition in (args.lift_ks, args.auto, args.ss):
        if decomposition and decomposition[0] * decomposition[1] >= 64:
            parser.error("TFHE-rs signed decomposition requires fewer than 64 bits")
    changed = args.ring != "2048x1" or args.lift_ks or args.auto or args.ss
    if changed and args.noise_audit:
        parser.error("noise-audit reference variances have not been ported to ring/decomposition overrides")
    if args.conditional_screen and (args.mode != "fused" or args.cbs_level != 7 or args.auto_fft != "split40"):
        parser.error("conditional screen currently requires fused, CBS level 7, and split40")
    n, _ = map(int, args.ring.split("x"))
    needed_digits = ((9 * args.group_size).bit_length() + 1) // 2
    if needed_digits * (9 * args.group_size + 1) > n:
        parser.error(f"group size {args.group_size} does not fit the CBS ring N={n}")
    if args.ring == "1024x2" and args.mode != "fused":
        parser.error("1024x2 currently requires --mode fused")
    if args.chunk_bits==4 and args.mode!="baseline":
        parser.error("--chunk-bits 4 requires --mode baseline")
    if args.step_audit and (not args.noise_audit or args.mode!="fused"):
        parser.error("--step-audit requires --noise-audit --mode fused")
    if not (args.exploratory or args.noise_audit or args.dry_run):
        parser.error("the shared selector-variance model is under audit; use --exploratory for research timings only")
    if args.warmup < 0: parser.error("warmup cannot be negative")
    if not 0<=args.seed<2**64: parser.error("seed must fit u64")
    output = args.output or ROOT/"results"/datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    modes = ("baseline","cached","fused") if args.mode=="all" else (args.mode,)
    build = ["cargo","build","--locked","--offline","--release","--bin","ring_variant_hybrid", "--target-dir", str(TARGET)]
    if args.noise_audit:
        build += ["--features", "noise-audit"]
    if not args.dry_run:
        build_env = dict(os.environ)
        build_env.pop("CARGO_TARGET_DIR",None)
        with (output/"build.log").open("w") as log:
            built = subprocess.run(build,cwd=ROOT,env=build_env,stdout=log,stderr=subprocess.STDOUT)
        if built.returncode: raise RuntimeError(f"build failed; inspect {output/'build.log'}")
    binary = TARGET/"release/ring_variant_hybrid"
    for width in args.widths:
        for mode,kernel in ((m,k) for m in modes for k in (args.kernels if m=="fused" else ["tree"])):
            plan = derive(width,mode,noise_audit=args.noise_audit,capacity=args.group_size,cbs_level=args.cbs_level,kernel=kernel,auto_fft=args.auto_fft,chunk_bits=args.chunk_bits)
            plan["env"]["CBS_RING_SHAPE"] = args.ring
            plan["env"]["CBS_SECRET_LAYOUT"] = args.secret_layout
            plan["cbs_ring"] = {"N": n, "k": 2048 // n}
            plan["normalizer_ring"] = {"N": 2048, "k": 1}
            for prefix, values in (("CBS_LIFT_KS", args.lift_ks), ("DIRECT_AUTO", args.auto), ("DIRECT_SS", args.ss)):
                if values:
                    plan["env"][prefix + "_BASE_LOG"] = str(values[0])
                    plan["env"][prefix + "_LEVEL"] = str(values[1])
            plan["reference_screen_applies_to_ring"] = not changed
            if not plan["reference_screen_applies_to_ring"]:
                plan["reference_screen_log2"] = None
                plan["audit_products"] = None
                plan["note"] = "Experimental ring/decomposition port. No whole-multiplier failure bound. Public bounds and source-balanced partition are inherited for a fixed-schedule comparison; diagnostic variances are not retuned."
            if args.conditional_screen:
                import heterogeneous_screen
                decomp = lambda prefix: tuple(int(plan["env"][prefix + suffix]) for suffix in ("_BASE_LOG", "_LEVEL"))
                p, v = heterogeneous_screen.primitive(n, decomp("DIRECT_AUTO"), decomp("DIRECT_SS"), decomp("CBS_LIFT_KS"))
                screen = heterogeneous_screen.analyze(plan, p, v)
                plan["conditional_screen"] = screen
                plan["conditional_screen_scope"] = "Unresolved external-product joint moments and fresh-PBS assumptions. Not a target-128 certificate."
                assert screen["approved"] is False and plan["log2_failure"] is None
                print(f"W{width} conditional sensitivity: {screen['conditional_union_log2']:.6f}; whole bound unresolved", flush=True)
            for threads in args.threads:
                mode_label = mode if args.chunk_bits==8 else f"{mode}4"
                case = output/f"{mode_label}-{kernel}-w{width}-t{threads}"
                case.mkdir()
                plan_path = case/"plan.json"
                plan_path.write_text(json.dumps(plan,indent=2)+"\n")
                bound = "" if plan["log2_failure"] is None else str(plan["log2_failure"])
                print(f"{case.name}: "+("failure bound UNRESOLVED" if not bound else "Gaussian model "+bound),flush=True)
                env = {k:v for k,v in os.environ.items()
                       if not k.startswith(("CBS_","DIRECT_","PBS_","KS_","CMUX_","CACHED_MULT_","FUSED_","CARGO_TARGET_DIR"))}
                env.update(plan["env"])
                env.update({"RAYON_NUM_THREADS":str(threads),"CACHED_MULT_PLAN":str(plan_path),
                    "CACHED_MULT_WARMUP":str(args.warmup),"CACHED_MULT_PATTERN":args.pattern,
                    "CACHED_MULT_VERIFY_PBS":str(int(args.verify_pbs)),
                    "CBS_RANDOM_OPERANDS_SEED":str(args.seed),"CBS_TIMING_CSV":str(case/"timings.csv"),
                    "CBS_FAILURE_TARGET_BITS":"128","CBS_FAILURE_EXPECTED_WORST_LOG2":bound})
                if args.noise_audit:
                    env.update({"CBS_NOISE_PROBE":"1","CBS_NOISE_PROBE_CSV":str(case/"probe.csv"),
                                "FUSED_COMPARE_PRODUCT":str(int(mode=="fused")),
                                "CBS_NOISE_PROBE_RUN_ID":case.name})
                if args.step_audit:
                    env["FUSED_STEP_AUDIT_CSV"] = str(case/"steps.csv")
                command = [str(binary),str(width),"1",str(args.warmup+args.repetitions)]
                record = {"command":command,"parameters":plan["env"],"threads":threads,
                    "width":width,"mode":mode,"squaring":False,"key_id":args.key_id,
                    "kernel":kernel,
                    "auto_fft":args.auto_fft,
                    "chunk_bits":args.chunk_bits,
                    "cbs_ring":args.ring,
                    "cbs_secret_layout": args.secret_layout,
                    "normalizer_ring":"2048x1",
                    "trials":args.warmup+args.repetitions,
                    "warmup":args.warmup,"repetitions":args.repetitions,"pattern":args.pattern,
                    "seed":args.seed,"input":"ordinary-PBS refreshed radix blocks",
                    "validation_only":args.verify_pbs or args.noise_audit,
                    "noise_audit":args.noise_audit,
                    "step_audit":args.step_audit,
                    "noise_screen_validated":False,
                    "binary_sha256":hashlib.sha256(binary.read_bytes()).hexdigest() if binary.exists() and not args.dry_run else None,
                    "plan_sha256":hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                    "source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted((ROOT/"src").rglob("*.rs"))},
                    "cargo_lock_sha256":hashlib.sha256((ROOT/"Cargo.lock").read_bytes()).hexdigest()}
                (case/"run.json").write_text(json.dumps(record,indent=2)+"\n")
                if args.dry_run: continue
                with (case/"run.log").open("w") as log:
                    result = subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f"run failed; inspect {case/'run.log'}")
                validate_rows(case/"timings.csv",plan,args.warmup+args.repetitions,threads,
                              analysis=plan.get("conditional_screen"),
                              seed=args.seed if args.pattern=="random" else None,
                              timed=not (args.verify_pbs or args.noise_audit))
                if args.verify_pbs: print(f"{case.name}: shared PBS ciphertext equality checks passed",flush=True)
                else: summarize(case/"timings.csv")
    print(f"Results: {output}",flush=True)

if __name__=="__main__":
    main()
