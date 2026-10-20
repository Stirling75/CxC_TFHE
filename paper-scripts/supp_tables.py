"""Tables S3 and S4 of the supplement from results/estimates/estimates.json and
results/limit-sweep/sweep.txt. Writes out/failure-parts.tex and out/limit-table.tex."""
import json, re
from pathlib import Path
HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
est = json.loads((RES / "estimates" / "estimates.json").read_text())
f = lambda x: f"${x:.1f}$"
pair = lambda a, b: f"{f(a)}\\,/\\,{f(b)}"
rows = []
for name, m in (("Product-sum", "hybrid-grouped-rev-ld"), ("Chunk-product", "hybrid-cached-rev-ld")):
    for i, w in enumerate((16, 32, 64, 128, 256)):
        e = est[f"{m}|{w}"]; ab, aa = e["AB"], e["AA"]
        cells = [pair(ab[k], aa[k]) for k in ("lift_log2", "reduction_log2", "addition_log2", "conditional_union_log2")]
        rows.append(f"{name if i == 0 else ' '} & ${w}$ & " + " & ".join(cells) + " \\\\")
    rows.append("\\midrule")
(HERE / "out").mkdir(exist_ok=True)
(HERE / "out" / "failure-parts.tex").write_text("\n".join(rows[:-1]) + "\n")

sweep = {}
for line in (RES / "limit-sweep" / "sweep.txt").read_text().splitlines():
    m = re.match(r"(\w+)\S* W(\d+) cap=\s*(\d+) red_pbs=\s*(\d+) AB=(\S+) AA=(\S+)", line)
    if m:
        mode, w, cap, pbs, ab, aa = m.groups()
        sweep[(mode[:6], int(w), int(cap))] = (int(pbs), float(ab), float(aa))
USED = {("groupe", 128): 40, ("groupe", 256): 40, ("cached", 128): 30, ("cached", 256): 20}
def cell(mode, w, cap):
    key = (mode, w, cap if cap is not None else 10**9)
    pbs, ab, aa = sweep[key]
    bold = cap is not None and USED[(mode, w)] == cap
    b = (lambda s: f"\\mathbf{{{s}}}") if bold else (lambda s: s)
    return f"${b(f'{pbs:,}'.replace(',', '{,}'))}$ & ${b(f'{ab:.1f}')}$ / ${b(f'{aa:.1f}')}$"
out = []
for w in (128, 256):
    for cap in (0, 20, 30, 40, 50, 60, 80, None):
        lab = "\\infty" if cap is None else str(cap)
        out.append(f"${lab}$ & ${w}$ & {cell('groupe', w, cap)} & ${w}$ & {cell('cached', w, cap)} \\\\")
    out.append("\\midrule")
(HERE / "out" / "limit-table.tex").write_text("\n".join(out[:-1]) + "\n")
print((HERE / "out" / "failure-parts.tex").read_text()); print((HERE / "out" / "limit-table.tex").read_text())
