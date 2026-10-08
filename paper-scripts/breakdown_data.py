"""Export the phase-breakdown figure data from raw per-trial timings.

out/data/breakdown-wide.csv:    single-thread seconds per phase at W = 64, 128, 256
out/data/breakdown-threads.csv: percentage per phase at W = 128 for T = 1, 4, 16, 64,
                                plus the mean total latency (s)
out/data/restoration-pbs.csv:   restoration bootstrappings on one thread (W = 16..256)

Phases: sel = circuit-bootstrapping selector lifts (cbs_ms), look = lookup /
external products (ext_ms), rest = restoration (normalization_ms). Warm-up rows
are excluded. Source: results/stats-20261004/stats-main-w*-20261004 (5 runs, 3 at W=256)
and stats-controls-20261004 for the 4x4 local-product control.
"""
import csv, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATS = HERE.parent / "results" / "stats-20261004"
OUT = HERE / "out" / "data"
PS, CP, PP4 = "hybrid-grouped-rev-ld", "hybrid-cached-rev-ld", "hybrid-4x4-rev-ld"


def rows(run, method, w, t):
    p = STATS / run / f"{method}-w{w}-t{t}" / "raw" / "timings.csv"
    if not p.exists():
        return None
    r = [x for x in csv.DictReader(open(p)) if x.get("warmup", "0") not in ("1", "true", "True")]
    return r or None


def mean(r, col):
    return sum(float(x[col]) for x in r) / len(r)


def phases(method, w, t):
    r = rows(f"stats-main-w{w}-20261004", method, w, t)
    return [mean(r, c) / 1000 for c in ("cbs_ms", "ext_ms", "normalization_ms")], mean(r, "total_ms") / 1000


OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "breakdown-wide.csv", "w") as f:
    f.write("i W ps_sel ps_look ps_rest cp_sel cp_look cp_rest\n")
    for i, w in enumerate((64, 128, 256), 1):
        vals = phases(PS, w, 1)[0] + phases(CP, w, 1)[0]
        f.write(f"{i} {w} " + " ".join(f"{v:.4f}" for v in vals) + "\n")
with open(OUT / "breakdown-threads.csv", "w") as f:
    f.write("i T ps_sel ps_look ps_rest cp_sel cp_look cp_rest ps_total cp_total\n")
    for i, t in enumerate((1, 4, 16, 64), 1):
        (a, ta), (b, tb) = phases(PS, 128, t), phases(CP, 128, t)
        pct = [100 * x / ta for x in a] + [100 * x / tb for x in b]
        f.write(f"{i} {t} " + " ".join(f"{v:.2f}" for v in pct) + f" {ta:.3f} {tb:.3f}\n")
# ps/cp: measured normalization_pbs (constant over trials). pp4 (4x4 local-product
# control): planner count reduction_pbs + sequential final additions, recorded in
# results/planner-20261006 (equals the measured 1431 at W=128).
PLAN = json.load(open(HERE.parent / "results" / "planner-20261006" / "hybrid-4x4-rev-ld-restoration-pbs.json"))["widths"]
with open(OUT / "restoration-pbs.csv", "w") as f:
    f.write("W ps cp pp4\n")
    for w in (16, 32, 64, 128, 256):
        cells = [str(int(mean(rows(f"stats-main-w{w}-20261004", m, w, 1), "normalization_pbs"))) for m in (PS, CP)]
        pl = PLAN[str(w)]
        cells.append(str(pl["reduction_pbs"] + pl["final_addition_pbs"]["sequential"]))
        f.write(f"{w} " + " ".join(cells) + "\n")
print("exported breakdown-wide.csv breakdown-threads.csv restoration-pbs.csv")
