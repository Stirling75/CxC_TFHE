"""Recompute the failure estimates of our configurations, check their schedules
against the measured plan.json, and write out/data/failure-ours.csv and
out/estimates.json (--save: results/estimates/estimates.json). The estimates in
run.json and summary.csv of the timing runs predate the exact second moments of
the key-switching digits and are 1.2 to 1.5 bits higher than these."""
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
    camp = {"hybrid-grouped-rev-mvb": f"mvb/w{w}",
            "hybrid-cached-rev-mvb": f"mvb/w{w}",
            "hybrid-4x4-rev-ld": "controls"}.get(
            m, f"main/w{w}")
    f = RES / camp / f"{m}-w{w}-t1" / "plan.json"
    return load(f) if f.exists() else None

def resolve_plan(m, w, plan):
    import copy
    screen = sys.modules["heterogeneous_screen"]
    assign, mvb = screen.assign_linear, screen.assign_mvb
    screen.assign_linear = lambda *a, **k: copy.deepcopy(plan)
    screen.assign_mvb = lambda *a, **k: None
    try:
        cases.resolve.cache_clear()
        return cases.resolve(m, w)
    finally:
        screen.assign_linear, screen.assign_mvb = assign, mvb
        cases.resolve.cache_clear()

out, rows = {}, ["method,width,AB,AA"]
for m in METHODS:
    for w in WIDTHS:
        r = cases.resolve(m, w)
        mp = measured_plan(m, w)
        if mp is not None and canon(mp) != canon(r["plan"]):
            # The current scheduler would exchange equal-bound terms differently; the
            # estimate must describe the schedule that was measured, so evaluate that one.
            r = resolve_plan(m, w, mp)
            print(m, w, "evaluated on the measured plan (scheduler now differs)", flush=True)
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
dst = RES / "estimates" / "estimates.json" if "--save" in sys.argv else HERE / "out" / "estimates.json"
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(out, indent=1) + "\n")
print("wrote", dst, "and out/data/failure-ours.csv")
