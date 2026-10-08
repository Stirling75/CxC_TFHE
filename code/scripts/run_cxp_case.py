"""Run one ciphertext-plaintext (CxP) product-sum case from a campaign run directory.
The environment of the ciphertext-ciphertext run is reused, CBS_CXP=1 selects the public
second operand, and CPUs 0..T-1 are pinned as in campaign.py.

With SCALAR_SEED, a public scalar is drawn from that seed and the scalar-aware plan
(groups of nonzero scalar digits, sum bounds 3*sum(b)) is derived and screened for it;
its derivation time and failure estimate are written to OUTPUT_PREFIX.plan.json.
Without it, the ciphertext-ciphertext plan and bounds are used unchanged.
Usage: run_cxp_case.py RUN_DIR WIDTH THREADS OUTPUT_PREFIX [SCALAR_SEED [GROUP_CAP]]"""
import json, os, random, subprocess, sys, time
from pathlib import Path
run_dir, width, threads = Path(sys.argv[1]).resolve(), sys.argv[2], int(sys.argv[3])
prefix = Path(sys.argv[4]).resolve()
root = Path(__file__).resolve().parents[1]
run = json.loads((run_dir / "run.json").read_text())
plan_path = run_dir / "plan.json"
if len(sys.argv) > 5:
    sys.path.insert(0, str(root / "scripts"))
    import cases
    rng = random.Random(int(sys.argv[5]))
    scalar = tuple(rng.randrange(4) for _ in range(int(width) // 2))
    cap = int(sys.argv[6]) if len(sys.argv) > 6 else 64
    started = time.perf_counter()
    resolved = cases.resolve(run["resolved"]["method"], int(width), scalar, cap)
    elapsed = time.perf_counter() - started
    plan_path = prefix.with_suffix(".plan.json")
    plan_path.write_text(json.dumps(resolved["plan"], indent=2) + "\n")
    a = resolved["analysis"]
    prefix.with_suffix(".screen.json").write_text(json.dumps({
        "scalar_seed": int(sys.argv[5]), "group_cap": cap, "plan_seconds": elapsed,
        "union_log2": a["conditional_union_log2"], "reduction_pbs": a["reduction_pbs"],
        "row_refresh": a["row_refresh"], "analysis": a}, indent=2) + "\n")
    if not a["conditional_union_log2"] < -128:
        sys.exit(f"scalar-aware plan misses the target: {a['conditional_union_log2']}")
env = {k: v for k, v in os.environ.items()
       if not k.startswith(("CBS_", "DIRECT_", "PBS_", "KS_", "CMUX_", "CACHED_MULT_", "FUSED_", "CARGO_TARGET_DIR"))}
env.update(run["environment_overrides"])
env.update({"RAYON_NUM_THREADS": str(threads), "CACHED_MULT_PLAN": str(plan_path),
            "CBS_TIMING_CSV": str(prefix.with_suffix(".csv")), "CACHED_MULT_WARMUP": "1", "CBS_CXP": "1"})
os.sched_setaffinity(0, list(range(threads)))
with open(prefix.with_suffix(".log"), "w") as log:
    r = subprocess.run([str(root / "build/ring/release/ring_variant_hybrid"), width, "1", "3"],
                       cwd=root / "code/ring-variant-multiplier", env=env, stdout=log, stderr=subprocess.STDOUT)
sys.exit(r.returncode)
