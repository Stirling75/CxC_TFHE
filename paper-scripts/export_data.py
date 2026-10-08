"""Export per-method latency tables for the pgfplots figures.

out/data/<method>.csv: columns W, t1, t2, t4, t8, t16, t32, t64, best (seconds).
Later summaries override earlier ones for the same (method, W, threads)."""
import csv, sys
from pathlib import Path
OUT = Path(__file__).resolve().parent / "out" / "data"
METHODS = {"hybrid-grouped-rev-ld": "ours-ps", "hybrid-grouped-rev-mvb": "ours-ps-mvb", "hybrid-cached-rev-ld": "ours-cp", "st-r2-n688": "st",
           "st-reported-r2": "st-paper", "tfhe-rs-ks28": "tfhe-rs", "trifan": "trifan",
           "clot-bfv": "bfv", "parmesan": "parmesan",
           "bernard-mvb-ks36": "bernard", "tfhe-rs": "tfhe-rs-default", "bernard-mvb": "bernard-paper"}
T = (1, 2, 4, 8, 16, 32, 64)
data = {}
for path in sys.argv[1:]:
    for r in csv.DictReader(open(path)):
        if r["method"] in METHODS:
            data.setdefault((r["method"], int(r["width"])), {})[int(r["threads"])] = float(r["mean_seconds"])
OUT.mkdir(parents=True, exist_ok=True)
# BFV adaptation misses the target at W=256: solid up to 128, dashed segment 128-256.
FAILS_FROM = {"clot-bfv": 256}
for method, name in METHODS.items():
    widths = sorted(w for (m, w) in data if m == method)
    if method in FAILS_FROM:
        cut = FAILS_FROM[method]
        tail = [w for w in widths if w >= cut]
        prev = [w for w in widths if w < cut][-1:]
        with open(OUT / f"{name}-miss.csv", "w") as f:
            f.write("W " + " ".join(f"t{t}" for t in T) + " best\n")
            for w in prev + tail:
                d = data[(method, w)]
                f.write(f"{w} " + " ".join(f"{d[t]:.6g}" if t in d else "nan" for t in T)
                        + f" {min(d.values()):.6g}\n")
        widths = [w for w in widths if w < cut]
    with open(OUT / f"{name}.csv", "w") as f:
        f.write("W " + " ".join(f"t{t}" for t in T) + " best\n")
        for w in widths:
            d = data[(method, w)]
            if 1 not in d: continue
            f.write(f"{w} " + " ".join(f"{d[t]:.6g}" if t in d else "nan" for t in T)
                    + f" {min(d.values()):.6g}\n")
for w in sorted({w for (_, w) in data}):
    with open(OUT / f"threads-w{w}.csv", "w") as f:
        f.write("T " + " ".join(METHODS[m] for m in METHODS) + "\n")
        for t in T:
            f.write(f"{t} " + " ".join(f"{data[(m, w)][t]:.6g}" if t in data.get((m, w), {}) else "nan"
                                       for m in METHODS) + "\n")
print("exported", sorted(p.name for p in OUT.iterdir()))
