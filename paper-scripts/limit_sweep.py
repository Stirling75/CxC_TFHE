"""Recompute the sweep of the linear-digit limit L (Figure 4, Table S4):
reduction bootstrappings and failure estimates of the restoration plan for
L = 0, 5, ..., 100, before the automatic lowering of L. Writes
../results/limit-sweep-20261006/sweep.txt (takes several minutes)."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code" / "scripts"))
import cases, heterogeneous_screen as screen

trials, state = [], {}
assign, analyze = screen.assign_linear, screen.analyze
def spy_assign(plan, prim, cap_mid, cap_last):
    state["cap"] = cap_mid
    return assign(plan, prim, cap_mid, cap_last)
def spy_analyze(plan, p, v, identical=False, **k):
    r = analyze(plan, p, v, identical=identical, **k)
    if "cap" in state:
        trials.append((state["cap"], identical, r["conditional_union_log2"], r.get("reduction_pbs")))
    return r
screen.assign_linear, screen.analyze = spy_assign, spy_analyze

out = []
for m in ("hybrid-grouped-rev-ld", "hybrid-cached-rev-ld"):
    for W in (128, 256):
        for cap in range(0, 101, 5):
            cases.CATALOG[m]["linear_digits"] = [cap, 1000]
            cases.resolve.cache_clear(); trials.clear(); state.clear()
            cases.resolve(m, W)
            t = [x for x in trials if x[0] == cap]
            ab = next(x for x in t if not x[1]); aa = next(x for x in t if x[1])
            out.append(f"{m[7:14]} W{W} cap={cap:5d} red_pbs={ab[3]:5d} AB={ab[2]:.1f} AA={aa[2]:.1f}")
            print(out[-1], flush=True)
(HERE.parent / "results" / "limit-sweep-20261006" / "sweep.txt").write_text("\n".join(out) + "\n")
