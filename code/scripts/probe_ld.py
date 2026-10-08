"""Noise-probe or timing run of hybrid METHOD at WIDTH on THREADS (cpus 0..T-1) with TRIALS (1 warmup) locally.
Usage: probe_ld.py METHOD WIDTH THREADS TRIALS OUTDIR [probe]"""
import json, os, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cases
method, W, T, trials, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5]).resolve()
probe = len(sys.argv) > 6
out.mkdir(parents=True, exist_ok=True)
r = cases.resolve(method, W)
plan = r["plan"]
(out / "plan.json").write_text(json.dumps(plan, indent=1))
(out / "model.json").write_text(json.dumps({"AB": r["analysis"], "AA": r.get("analysis_identical"),
    "variances": r.get("model_variances")}, indent=1, default=str))
env = {k: v for k, v in os.environ.items()
       if not k.startswith(("CBS_", "DIRECT_", "PBS_", "KS_", "CMUX_", "CACHED_MULT_", "FUSED_", "CARGO_TARGET_DIR"))}
env.update(plan["env"])
env.update(RAYON_NUM_THREADS=str(T), CACHED_MULT_PLAN=str(out / "plan.json"), CACHED_MULT_WARMUP="1",
           CACHED_MULT_PATTERN="random", CACHED_MULT_VERIFY_PBS="0", CBS_RANDOM_OPERANDS_SEED=str(W * 1000 + T),
           CBS_TIMING_CSV=str(out / "timings.csv"), CBS_FAILURE_TARGET_BITS="128", CBS_FAILURE_EXPECTED_WORST_LOG2="")
binary = ROOT / "build/ring/release/ring_variant_hybrid"
if probe:
    env.update(CBS_NOISE_PROBE="1", CBS_NOISE_PROBE_CSV=str(out / "probe.csv"), CBS_NOISE_PROBE_RUN_ID=out.name,
               FUSED_COMPARE_PRODUCT="0", CACHED_MULT_WARMUP="0")
    binary = ROOT / "build-audit/release/ring_variant_hybrid"
cpus = ",".join(str(i) for i in range(T))
with open(out / "stdout.log", "w") as log:
    rc = subprocess.run(["taskset", "-c", cpus, str(binary), str(W), "1", str(trials)],
                        cwd=ROOT / "code/ring-variant-multiplier", env=env, stdout=log, stderr=subprocess.STDOUT).returncode
print(method, W, T, "exit", rc)
