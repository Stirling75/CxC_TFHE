"""Write <data dir>/rel-*.csv (default out/data): latency of each method divided by that of
Ours (product-sum) at the same width and thread budget (Figure 5)."""
import csv, pathlib
import sys
D = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "out/data")
def load(name):
    with open(D / f"{name}.csv") as f:
        rows = list(csv.reader(f, delimiter=" "))
    head = rows[0]
    return {int(r[0]): dict(zip(head[1:], map(float, r[1:]))) for r in rows[1:]}
base = load("ours-ps")
for name in ["ours-ps", "ours-ps-mvb", "ours-cp", "st", "bfv", "bfv-miss", "bernard",
             "tfhe-rs", "trifan", "st-paper", "parmesan", "tfhe-rs-default", "bernard-paper"]:
    rows = load(name)
    with open(D / f"rel-{name}.csv", "w") as f:
        f.write("W t1 t64\n")
        for w in sorted(rows):
            f.write(f"{w} {rows[w]['t1']/base[w]['t1']:.4f} {rows[w]['t64']/base[w]['t64']:.4f}\n")
# Latency relative to Ours (product-sum) by thread budget at every width (Figure 5).
for w in (16, 32, 64, 128, 256):
    with open(D / f"threads-w{w}.csv") as f:
        rows = list(csv.reader(f, delimiter=" "))
    head = rows[0]
    with open(D / f"rel-threads-w{w}.csv", "w") as f:
        f.write(" ".join(head) + "\n")
        for r in rows[1:]:
            d = dict(zip(head, r)); b = float(d["ours-ps"])
            f.write(d["T"] + " " + " ".join(f"{float(d[h]) / b:.4f}" if d[h] != "nan" else "nan"
                                            for h in head[1:]) + "\n")
