"""Compare noise-probe residuals with the model variances of the plan (pre-key-switch).
Categories: reduction inputs of wave 0 (product digits), later waves with / without linear
digits, and the block sums of the final addition with / without unrefreshed digits.
Per category: mean(res^2/model) (model conservative if <= 1), excess kurtosis of z,
max |z|, slot mismatches, and the largest per-position measured/model variance ratio.
Usage: probe_analyze.py DIR..."""
import csv, json, math, sys, collections
import numpy as np

def kurt(z):
    z = np.asarray(z, float); return float(np.mean(z**4) / np.mean(z**2)**2 - 3)

for d in sys.argv[1:]:
    model = json.load(open(f"{d}/model.json"))["variances"]
    plan = json.load(open(f"{d}/plan.json"))
    linear_jobs = []
    for wave in plan["waves"]:
        flags = []
        for rec in wave:
            lin = rec.get("linear") or [False] * len(rec["groups"])
            flags += [lin[i] for i, g in enumerate(rec["groups"]) if len(g) >= 3]
        linear_jobs.append(flags)
    cats = collections.defaultdict(list); pos = collections.defaultdict(list); mism = collections.Counter()
    for r in csv.DictReader(open(f"{d}/probe.csv")):
        st, w, q, ev = r["stage"], int(r["wave"]), int(r["column"]), int(r["event"])
        if st == "chunk":
            mv = model["waves"][w][ev]
            if w == 0: c = "wave0 (product digits)"
            elif int(r["unrefreshed"]) > 0: c = "wave>=1 with unrefreshed terms"
            else: c = "wave>=1 refreshed only"
            key = (st, w, ev)
        elif st == "final-input":
            mv = model["final_blocks"][q]
            c = "final input, unrefreshed" if int(r["unrefreshed"]) > 0 else "final input, refreshed"
            key = (st, q)
        else:
            continue
        res = float(r["residual"])
        cats[c].append(res / math.sqrt(mv)); pos[(c, key)].append((res, mv))
        mism[c] += int(r["slot_mismatch"])
    print(f"== {d}  (linear jobs per wave: {[sum(f) for f in linear_jobs]} of {[len(f) for f in linear_jobs]})")
    for c, z in sorted(cats.items()):
        z = np.array(z)
        ratios = [np.mean([x*x for x, _ in v]) / v[0][1] for (cc, _), v in pos.items() if cc == c and len(v) >= 10]
        print(f"  {c:34s} n={len(z):6d} E[z^2]={np.mean(z**2):.3f} exkurt={kurt(z):+.3f} max|z|={np.max(abs(z)):.2f} "
              f"mismatch={mism[c]} pos-ratio max={max(ratios) if ratios else float('nan'):.3f} (positions {len(ratios)})")
