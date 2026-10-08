"""Recompute the whole-multiplication failure estimates of our configurations
(Tables 5 and 9, Table S3) with the model in ../code/scripts and write
../results/estimates-20261007/estimates.json. For each method and width it
stores the log2 estimate for separately prepared (AB) and identical (AA)
operands, the per-family terms of Table S3, and the restoration bootstrapping
counts, and writes out/data/failure-ours.csv (rounded, compared by `make check`).
It stops if a schedule differs from the plan.json of the measured run.
The estimates recorded in run.json of the measurement campaigns predate a
refinement of the model (shared first candidates, selector margin) and differ
from these by at most 0.1."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code" / "scripts"))
import cases

METHODS = ("hybrid-grouped-rev-ld", "hybrid-cached-rev-ld", "hybrid-grouped-rev-mvb",
           "hybrid-cached-rev-mvb", "hybrid-4x4-rev-ld")
WIDTHS = (16, 32, 64, 128, 256)
KEYS = ("selector_log2_variance", "lift_log2", "reduction_log2", "refresh_log2", "addition_log2",
        "decoding_log2", "split_high_log2", "conditional_union_log2")
RES = HERE.parent / "results"
SKIP = {"model_sha256", "planner_sha256"}

def load(f):
    o = json.loads(f.read_text())
    while isinstance(o, dict) and set(o) == {"same_as"}:
        f = (f.parent / o["same_as"]).resolve()
        o = json.loads(f.read_text())
    return o

def canon(plan):
    # a schedule up to the order of terms inside a group (the order of a sum)
    waves = [[(sorted(sorted(json.dumps(c["terms"][i], sort_keys=True) for i in g) for g in c["groups"]),
               json.dumps(c.get("linear"), sort_keys=True)) for c in wave] for wave in plan["waves"]]
    rest = {k: v for k, v in plan.items() if k not in SKIP | {"waves"}}
    return waves, json.dumps(rest, sort_keys=True)

def measured_plan(m, w):
    camp = {"hybrid-grouped-rev-mvb": f"mvb-20261005/mvb-w{w}-20261005",
            "hybrid-cached-rev-mvb": f"mvb-20261005/mvb-w{w}-20261005",
            "hybrid-4x4-rev-ld": "stats-20261004/stats-controls-20261004"}.get(
            m, f"stats-20261004/stats-main-w{w}-20261004")
    f = RES / camp / f"{m}-w{w}-t1" / "plan.json"
    return load(f) if f.exists() else None

out, rows = {}, ["method,width,AB,AA"]
for m in METHODS:
    for w in WIDTHS:
        r = cases.resolve(m, w)
        mp = measured_plan(m, w)
        if mp is not None and canon(mp) != canon(r["plan"]):
            sys.exit(f"schedule of {m} W={w} differs from the measured plan.json")
        ab, aa = r["analysis"], r["analysis_identical"]
        out[f"{m}|{w}"] = {
            "AB": {k: ab.get(k) for k in KEYS}, "AA": {k: aa.get(k) for k in KEYS},
            "reduction_pbs": ab["reduction_pbs"],
            "final_addition_pbs": ab["final_addition_pbs"],
        }
        rows.append(f"{m},{w},{ab['conditional_union_log2']:.1f},{aa['conditional_union_log2']:.1f}")
        print(m, w, "measured plan" if mp else "", f"AB={ab['conditional_union_log2']:.1f} AA={aa['conditional_union_log2']:.1f}", flush=True)
(HERE / "out" / "data").mkdir(parents=True, exist_ok=True)
(HERE / "out" / "data" / "failure-ours.csv").write_text("\n".join(rows) + "\n")
dst = RES / "estimates-20261007" / "estimates.json" if "--save" in sys.argv else HERE / "out" / "estimates.json"
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(out, indent=1) + "\n")
print("wrote", dst, "and out/data/failure-ours.csv")
