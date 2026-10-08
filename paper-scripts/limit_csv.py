"""Write out/data/limit-{ps,cp}{128,256}.csv (Figure 4) from
../results/limit-sweep/sweep.txt: for each limit L, the reduction
bootstrappings saved relative to L=0 (%), the log2 failure estimate (worse of
separate and identical operands), and the plotted value clipped at -124.4."""
import re, collections
from pathlib import Path
HERE = Path(__file__).resolve().parent
rows = collections.defaultdict(dict)
for line in (HERE.parent / "results" / "limit-sweep" / "sweep.txt").read_text().splitlines():
    m = re.match(r'(\S+) W(\d+) cap=\s*(\d+) red_pbs=\s*(\d+) AB=(\S+) AA=(\S+)', line)
    if m:
        mode = "ps" if m.group(1).startswith("grouped") else "cp"
        rows[(mode, int(m.group(2)))][int(m.group(3))] = (int(m.group(4)), float(m.group(5)), float(m.group(6)))
out = HERE / "out" / "data"
out.mkdir(parents=True, exist_ok=True)
for (mode, W), d in sorted(rows.items()):
    base = d[0][0]
    with open(out / f"limit-{mode}{W}.csv", "w") as f:
        f.write("L saved logp y cut\n")
        for L in sorted(d):
            p, ab, aa = d[L]
            lp = float(f"{max(ab, aa):.2f}")
            f.write(f"{L} {100 * (base - p) / base:.2f} {lp:.2f} {min(lp, -124.4):.2f} {1 if lp > -124.4 else 0}\n")
print("wrote", sorted(p.name for p in out.glob("limit-*.csv")))
