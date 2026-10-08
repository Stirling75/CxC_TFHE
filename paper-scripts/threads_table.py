"""Write the table of latencies at W=128 and 256 by thread budget (Table
tab:threads) from <data dir>/threads-w{128,256}.csv.
Usage: threads_table.py DATA_DIR OUT_TEX"""
import csv, sys
from pathlib import Path
D = Path(sys.argv[1]); OUT = Path(sys.argv[2])
T = ["1", "2", "4", "8", "16", "32", "64"]
rows = [("Ours, product-sum", "ours-ps"), ("Ours, chunk-product", "ours-cp"),
        ("Ours+MVB, product-sum$^\\dagger$", "ours-ps-mvb"),
        ("Shokri--Tsoutsos~\\cite{shokri2026accelerating}, retuned", "st"),
        ("Bernard et al.~\\cite{bernard2026lownoise}, KS $(3,6)^\\dagger$", "bernard"),
        ("TFHE-rs~\\cite{tfhe_rs}, KS $(2,8)$", "tfhe-rs"),
        ("BFV adaptation~\\cite{chillotti2021improved}", "bfv")]
MISS = {(256, "bfv")}  # misses the target at this width (Table tab:failure)
def fmt(x): return f"{x:.3f}" if x < 1 else f"{x:.2f}" if x < 10 else f"{x:.1f}" if x < 100 else f"{x:.0f}"
lines = []
for w in (128, 256):
    r = list(csv.reader(open(D / f"threads-w{w}.csv"), delimiter=" ")); h = r[0]
    d = {row[0]: dict(zip(h[1:], row[1:])) for row in r[1:]}
    best = {t: min(float(d[t][k]) for _, k in rows if (w, k) not in MISS) for t in T}
    if w != 128: lines.append("\\midrule")
    lines.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\emph{{$W={w}$}}}} \\\\")
    for name, k in rows:
        cells = []
        for t in T:
            v = float(d[t][k]); c = fmt(v)
            if (w, k) in MISS: cells.append(f"\\textcolor{{black!45}}{{${c}$}}")
            elif v == best[t]: cells.append(f"$\\mathbf{{{c}}}$")
            else: cells.append(f"${c}$")
        lines.append(name + " & " + " & ".join(cells) + " \\\\")
OUT.write_text(r"""\begin{table}[!htb]
\centering
\small
\caption{Latency in seconds at $W=128$ and $256$ by the number of threads $T$
for the methods that meet the target, with the fastest of each column in
bold. The gray row misses the target at that width.
$^\dagger$Under the model of Bernard et al.~\cite{bernard2026lownoise} for
multi-value outputs (Section~\ref{subsec:error}).}
\label{tab:threads}
\setlength{\tabcolsep}{7pt}
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
Method & $T=1$ & $2$ & $4$ & $8$ & $16$ & $32$ & $64$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
print("wrote", OUT)
