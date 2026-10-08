#!/usr/bin/env python3
"""Lattice-estimator checks for independently retuned ST candidates."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--estimator", type=Path, required=True)
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--full", action="store_true")
args = parser.parse_args()
if args.output.exists():
    parser.error("use a new output file")
sys.path.insert(0, str(args.estimator.resolve()))
from sage.all import log, oo, version
from estimator import LWE, ND
from estimator.conf import red_cost_model, red_shape_model
from estimator.lwe_parameters import LWEParameters

commit = subprocess.check_output(
    ["git", "-C", str(args.estimator), "rev-parse", "HEAD"], text=True).strip()
rows = []
for entry in json.loads(args.input.read_text()):
    p = LWEParameters(n=entry["n"], q=2**64, Xs=ND.Binary,
        Xe=ND.DiscreteGaussian(entry["sigma"] * 2**64), m=oo, tag=entry["name"])
    costs = LWE.estimate(p, catch_exceptions=False) if args.full else LWE.estimate.rough(p, catch_exceptions=False)
    attacks = {str(name): float(log(cost["rop"], 2)) for name, cost in costs.items()}
    expected = {"arora-gb", "bkw", "usvp", "bdd", "bdd_hybrid", "bdd_mitm_hybrid", "dual", "dual_hybrid"} if args.full else {"usvp", "dual_hybrid"}
    if set(attacks) != expected or any(math.isnan(v) for v in attacks.values()):
        raise RuntimeError(f"incomplete/invalid estimator result: {attacks}")
    finite = [v for v in attacks.values() if math.isfinite(v)]
    if not finite:
        raise RuntimeError("no finite attack estimate")
    row = {**entry, "attacks_rop_log2": {k: v if math.isfinite(v) else "infinity" for k, v in attacks.items()}, "minimum_rop_log2": min(finite),
           "full_estimator": args.full, "sample_model": "unlimited",
           "secret_distribution": "binary", "error_distribution": "discrete Gaussian"}
    rows.append(row)
    print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"estimator_commit": commit, "sage": version(),
        "reduction_cost_model": str(red_cost_model) if args.full else "ADPS16 (rough)",
        "reduction_shape_model": str(red_shape_model) if args.full else "gsa (rough)",
        "note": "GLWE rows are the kN-dimensional LWE proxy, not a ring-specific proof.",
        "rows": rows}, indent=2) + "\n")
