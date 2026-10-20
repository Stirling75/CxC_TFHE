"""Table 5: estimated log2 failure probability of one multiplication for every method,
from code/scripts/cases.py (the model of Section 5, or for Trifan et al. and PARMESAN
per-bootstrapping estimates combined over one multiplication). Writes out/failure-table.tex."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code" / "scripts"))
import cases

WIDTHS = (16, 32, 64, 128, 256)
OURS = [("Ours, product-sum", "hybrid-grouped-rev-ld"), ("Ours, chunk-product", "hybrid-cached-rev-ld"),
        ("Ours+MVB, product-sum$^\\dagger$", "hybrid-grouped-rev-mvb"),
        ("Ours+MVB, chunk-product$^\\dagger$", "hybrid-cached-rev-mvb"),
        ("Per product, 4-bit chunks", "hybrid-4x4-rev-ld")]
OTHERS = [("TFHE-rs~\\cite{tfhe_rs}, KS $(2,8)$", "tfhe-rs-ks28"),
          ("TFHE-rs~\\cite{tfhe_rs}, KS $(3,5)$", "tfhe-rs"),
          ("Shokri--Tsoutsos~\\cite{shokri2026accelerating}, retuned", "st-sign-retuned"),
          ("Shokri--Tsoutsos~\\cite{shokri2026accelerating}, paper", "st-reported-r2"),
          ("Bernard et al.~\\cite{bernard2026lownoise}, KS $(3,6)^\\dagger$", "bernard-mvb-ks36"),
          ("Bernard et al.~\\cite{bernard2026lownoise}, paper$^\\dagger$", "bernard-mvb"),
          ("BFV adaptation~\\cite{chillotti2021improved}", "clot-bfv"),
          ("Trifan et al.~\\cite{trifan2023arithmetic}", "trifan"),
          ("PARMESAN~\\cite{klemsa2023parmesan}", "parmesan")]

def cell(x):
    if x is None:
        return "--"
    s = f"{x:.1f}" if f"{x:.1f}" != "-128.0" else f"{x:.2f}"
    s = f"${s}$".replace("$-", "$-")
    return s if x < -128 else f"\\textcolor{{black!45}}{{{s}}}"

def other(method, width):
    if method == "parmesan" and width > 32:
        return None
    r = cases.resolve(method, width)
    if method.startswith("bernard"):
        return max(row["ab"] for row in r["analysis"].values())
    if method.startswith("st-"):
        return r["analysis"]["conditional_log2_union"]
    return r["whole_product_log2_estimate"]

import json
EST = json.loads((HERE.parent / "results" / "estimates" / "estimates.json").read_text())  # measured plans
lines = []
for name, method in OURS:
    rows = [EST[f"{method}|{w}"] for w in WIDTHS]
    lines.append(f"{name} & separate & " + " & ".join(cell(r["AB"]["conditional_union_log2"]) for r in rows) + " \\\\")
    lines.append(f" & identical & " + " & ".join(cell(r["AA"]["conditional_union_log2"]) for r in rows) + " \\\\")
lines.append("\\midrule")
lines.append("\\multicolumn{7}{@{}l}{\\emph{Other methods}} \\\\")
for name, method in OTHERS:
    lines.append(f"{name} & separate & " + " & ".join(cell(other(method, w)) for w in WIDTHS) + " \\\\")
out = HERE / "out" / "failure-table.tex"
out.parent.mkdir(exist_ok=True)
out.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
