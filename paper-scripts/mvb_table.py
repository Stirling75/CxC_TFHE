"""Write out/mvb-table.tex: the body of the table comparing the restoration
without and with multi-value folds (MVB), from ../results.

Latency without MVB is the merged main comparison (summaries.txt, later files
override earlier ones, as for the figures); latency with MVB, restoration PBS
counts come from the run records of each campaign, and failure estimates from
../results/estimates-20261007/estimates.json (estimates.py)."""
import csv, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
OUT = HERE / "out" / "mvb-table.tex"
MODES = (("hybrid-grouped-rev", "Product-sum"), ("hybrid-cached-rev", "Chunk-product"))
WIDTHS = (64, 128, 256)

lat = {}
for line in (HERE / "summaries.txt").read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    for r in csv.DictReader(open(HERE / line)):
        lat[(r["method"], int(r["width"]), int(r["threads"]))] = float(r["mean_seconds"])

EST = json.loads((RES / "estimates-20261007" / "estimates.json").read_text())

def run(method, w):
    camp = (f"mvb-20261005/mvb-w{w}-20261005" if method.endswith("mvb")
            else f"stats-20261004/stats-main-w{w}-20261004")
    return json.loads((RES / camp / f"{method}-w{w}-t1" / "run.json").read_text())["resolved"]

def pbs(res, w):
    # restoration bootstrappings: reduction groups plus the sequential final addition
    return res["analysis"]["reduction_pbs"] + res["analysis"]["final_addition_pbs"]["sequential"]

def f2(x): return f"{x:.2f}"
def f3(x): return f"{x:.3f}"
def num(n): return f"{n:,}".replace(",", "{,}")

rows = []
for prefix, name in MODES:
    if rows:
        rows.append(r"\midrule")
    for k, w in enumerate(WIDTHS):
        ld, mv = f"{prefix}-ld", f"{prefix}-mvb"
        r_ld, r_mv = run(ld, w), run(mv, w)
        ab = EST[f"{mv}|{w}"]["AB"]["conditional_union_log2"]
        aa = EST[f"{mv}|{w}"]["AA"]["conditional_union_log2"]
        rows.append((name if k == 0 else "") + f" & ${w}$ & ${num(pbs(r_ld, w))}$ & ${num(pbs(r_mv, w))}$ & "
                    f"${ab:.1f}$ (${aa:.1f}$) & ${f2(lat[(ld, w, 1)])}$ & ${f2(lat[(mv, w, 1)])}$ & "
                    f"${f3(lat[(ld, w, 64)])}$ & ${f3(lat[(mv, w, 64)])}$ \\\\")
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(rows) + "\n")
print("wrote", OUT)
