"""Table 6: compare the errors of decrypted intermediate ciphertexts in
results/noise with the modeled variances, and write out/data/noise-ratios.csv
and out/data/noise-kurtosis.csv."""
import collections, csv, gzip, json, math, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
N = HERE.parent / "results" / "noise"
sys.path.insert(0, str(HERE.parent / "code" / "scripts"))
sys.path.insert(0, str(HERE.parent / "code" / "code" / "ring-variant-multiplier" / "model" / "bernard"))
import cases

def rows(path):
    with gzip.open(path, "rt") as f:
        yield from csv.DictReader(f)

def var(xs):
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)

def exkurt(z):
    m = sum(z) / len(z)
    v = sum((x - m) ** 2 for x in z) / len(z)
    return sum((x - m) ** 4 for x in z) / len(z) / v / v - 3

out, kurt = [], []
# selector rows: largest measured row variance against the model
for name, method in (("product-sum", "hybrid-grouped-rev-ld"), ("chunk-product", "hybrid-cached-rev-ld"),
                     ("4-bit", "hybrid-4x4-rev-ld")):
    g = collections.defaultdict(list)
    for r in rows(N / "selectors" / name / "selectors.csv.gz"):
        key = "row0" if r["row"] == "0" else ("c0" if r["coefficient"] == "0" else "rest")
        g[key].append(int(r["error"]) ** 2)
    meas = max(sum(v) / len(v) for v in g.values())
    model = 2.0 ** cases.resolve(method, 16)["analysis"]["selector_log2_variance"]
    out.append(("selector", name, 16, meas / model))
# reduction and final-addition inputs: mean of residual^2 / V_T per category
for w in (64, 128, 256):
    d = N / "restoration" / f"w{w}"
    model = json.load(open(d / "model.json"))["variances"]
    cats = collections.defaultdict(list)
    for r in rows(d / "probe.csv.gz"):
        wave, col, ev, unref = int(r["wave"]), int(r["column"]), int(r["event"]), int(r["unrefreshed"])
        if r["stage"] == "chunk":
            v = model["waves"][wave][ev]
            c = "wave 0" if wave == 0 else ("later wave, unrefreshed terms" if unref else "later wave, refreshed")
        else:
            v = model["final_blocks"][col]
            c = "final addition, unrefreshed terms" if unref else "final addition, refreshed"
        cats[c].append(float(r["residual"]) / math.sqrt(v))
    for c, z in sorted(cats.items()):
        out.append(("reduction", c, w, sum(x * x for x in z) / len(z)))
        kurt.append((c, w, len(z), exkurt(z)))
# Ours+MVB digits against 8 V_PBS
for w in (64, 128, 256):
    import heterogeneous_screen as screen
    cap = {}
    def spy(plan, p, v, _orig=screen.analyze, **k):
        cap["v"] = v
        return _orig(plan, p, v, **k)
    screen.analyze, saved = spy, screen.analyze
    cases.resolve.cache_clear(); cases.resolve("hybrid-grouped-rev-mvb", w)
    screen.analyze = saved
    vpbs = cap["v"].normalizer_pbs_var
    dig = [float(r["residual"]) for r in rows(N / "mvb-digit" / f"w{w}" / "probe.csv.gz") if r["stage"] == "mvb-digit"]
    out.append(("mvb-digit", "digit", w, var(dig) / (8 * vpbs)))
# Bernard et al.: blind-rotation inputs per class against ||w||_2^2 V_PBS
import failure as F
vpbs, _ = F.primitive((4, 4))
for w in (64, 128):
    g = collections.defaultdict(list)
    for r in rows(N / "bernard" / f"w{w}" / "probe.csv.gz"):
        nu2 = 17.0 if r["kind"] in ("pmvb", "plsb") else round(sum(F.V2[t] for t in r["in_types"].split("+")), 3)
        g[nu2].append(int(r["pre_ks_error"]))
    for nu2, xs in sorted(g.items()):
        if len(xs) >= 200:
            out.append(("bernard", f"||w||^2={nu2:g}", w, var(xs) / (nu2 * vpbs)))

(HERE / "out" / "data").mkdir(parents=True, exist_ok=True)
with open(HERE / "out" / "data" / "noise-ratios.csv", "w") as f:
    f.write("row,category,width,ratio\n")
    for r in out:
        f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.3f}\n")
with open(HERE / "out" / "data" / "noise-kurtosis.csv", "w") as f:
    f.write("category,width,n,excess_kurtosis,standard_error\n")
    for c, w, n, k in kurt:
        f.write(f"{c},{w},{n},{k:.3f},{math.sqrt(24 / n):.3f}\n")
print(f"kurtosis   {min(k for *_, k in kurt):.2f} to {max(k for *_, k in kurt):.2f}")
for key in ("selector", "reduction", "mvb-digit", "bernard"):
    v = [r[3] for r in out if r[0] == key]
    print(f"{key:10s} {min(v):.2f} to {max(v):.2f}")
