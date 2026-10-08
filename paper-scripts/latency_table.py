"""Write out/appendix-latency.tex: mean latency of every method, width and
thread budget, from campaign summary CSVs (later files override earlier ones).
In each width block, the fastest method meeting the failure target is set in bold
for every thread budget."""
import csv, sys
from pathlib import Path
OUT = Path(__file__).resolve().parent / "out" / "appendix-latency.tex"
OUT.parent.mkdir(parents=True, exist_ok=True)
MEETS = [("hybrid-grouped-rev-ld", "Ours, product-sum"), ("hybrid-cached-rev-ld", "Ours, chunk-product"),
         ("hybrid-grouped-rev-mvb", "Ours+MVB, product-sum$^\\dagger$"), ("hybrid-cached-rev-mvb", "Ours+MVB, chunk-product$^\\dagger$"),
         ("st-r2-n688", "Shokri--Tsoutsos, $n=688$"), ("bernard-mvb-ks36", "Bernard et al., KS $(3,6)^\\dagger$"),
         ("tfhe-rs-ks28", "TFHE-rs, KS $(2,8)$"), ("clot-bfv", "BFV adaptation")]
MISSES = [("st-reported-r2", "Shokri--Tsoutsos, paper"), ("bernard-mvb", "Bernard et al., paper$^\\dagger$"), ("tfhe-rs", "TFHE-rs, default"),
          ("trifan", "Trifan et al."), ("parmesan", "PARMESAN")]
T = (1, 2, 4, 8, 16, 32, 64)
data = {}
for path in sys.argv[1:]:
    for r in csv.DictReader(open(path)):
        data[(r["method"], int(r["width"]), int(r["threads"]))] = float(r["mean_seconds"])
def fmt(x):
    return f"{x:.3f}" if x < 1 else f"{x:.2f}" if x < 10 else f"{x:.1f}" if x < 100 else f"{x:.0f}"
def meets(m, w):
    return m != "clot-bfv" or w < 256
lines = [r"\section{Latency Table}\label{app:latency}",
         r"Tables~\ref{tab:latency} and~\ref{tab:latency-wide} list the mean latency of every method of the main comparison, for all widths and thread budgets of Section~\ref{subsec:setup}; Figure~\ref{fig:overview} plots parts of them. The controls of Table~\ref{tab:breakdown} were measured separately."]
CAPTION = (r"Mean latency in seconds of one multiplication by operand width and number of threads, for {part}. "
           r"In each block, the methods above the inner rule meet the $2^{{-128}}$ target of Table~\ref{{tab:failure}} "
           r"(the BFV adaptation only up to $W=128$; the paper set of Bernard et al.\ also meets it for $W\le32$), "
           r"and the fastest of them for each number of threads is set in bold. $^\dagger$Under the model of Bernard et al.~\cite{{bernard2026lownoise}} for multi-value outputs (Section~\ref{{subsec:error}}).")
for widths, part, label in (((16, 32, 64), r"$W\le64$", "tab:latency"), ((128, 256), r"$W\ge128$", "tab:latency-wide")):
    lines += [r"\begin{table}[p]", r"\centering", r"\small", r"\renewcommand{\arraystretch}{1.0}",
              r"\caption{" + CAPTION.format(part=part) + "}", rf"\label{{{label}}}", r"\setlength{\tabcolsep}{7pt}",
              r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
              r"& \multicolumn{7}{c}{Number of threads} \\", r"\cmidrule(l){2-8}",
              "Method & " + " & ".join(map(str, T)) + r" \\"]
    for w in widths:
        lines += [r"\midrule[0.8pt]", r"\rowcolor{black!8}" + rf"\multicolumn{{8}}{{@{{}}l}}{{\textbf{{$W={w}$}}}} \\", r"\midrule"]
        best = {t: min((data[(m, w, t)] for m, _ in MEETS if (m, w, t) in data and meets(m, w)), default=None) for t in T}
        for group in (MEETS, MISSES):
            rows = [(m, n) for m, n in group if any((m, w, t) in data for t in T)]
            if group is MISSES and rows:
                lines.append(r"\cmidrule(l){1-8}")
            for m, name in rows:
                cells = []
                for t in T:
                    if (m, w, t) not in data:
                        cells.append("--"); continue
                    v = fmt(data[(m, w, t)])
                    bold = group is MEETS and meets(m, w) and data[(m, w, t)] == best[t]
                    cells.append(rf"$\mathbf{{{v}}}$" if bold else f"${v}$")
                lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
OUT.write_text("\n".join(lines) + "\n")
print("wrote", OUT)
