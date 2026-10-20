#!/usr/bin/env python3
"""ST R2 noise probe: measured per-stage noise vs the st_retune screen, and a calibrated estimate.

probe.csv rows: stage, theta, trial, index, pre, post (signed offsets in units of 2^-64 from the
nearest multiple of the stage spacing). Blind-rotation stages (pre = small-key error before MS,
post = decision error after the exact centered binary MS):
  9 input lift (theta 2, spacing 2^62)  1 compressor (theta 1)  2 terminal read (theta 2)
  0 emission (theta 0; index = block; blocks < 4 are direct column sums, >= 4 the CMux chain)
Pre-key-switch big-key errors (post = 0): 19 lift (scaled by 2^10), 11 compressor input,
  12 terminal-read input (index = column), 10 emission input (index = block).
Decision rule: lane j of a 2^theta-lane accumulator reads
rot + j, so a read is correct iff post in [-s/2, s/2 - (2^theta - 1) q/(2N)), s = 2^61 (2^62 for 9).
"""
import csv, collections, gzip, json, math, sys
from pathlib import Path
ART = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ART / "code" / "scripts"), str(ART / "code" / "code" / "server-campaign")]
import cases
from st_gaussian import log2_gaussian_outside

Q, N = 2.0**64, 1024
L2 = lambda x: math.log2(x) if x > 0 else float("-inf")
SCREEN = {"9": "input_grouped_lift", "1": "compressor", "2": "terminal_binary_lift",
          "0d": "emission_pbs", "0t": "emission_pbs"}


def tail(z):
    return math.log2(0.5 * math.erfc(z / math.sqrt(2))) if z < 30 else \
        (-z * z / 2 - math.log(z * math.sqrt(2 * math.pi))) / math.log(2)


def two_sided(lo, hi, mean, sd):
    a, b = tail((lo + mean) / sd), tail((hi - mean) / sd)
    m = max(a, b); return m + math.log2(2**(a - m) + 2**(b - m))


def moments(xs):
    n = len(xs); m = math.fsum(xs) / n
    c2 = math.fsum((x - m)**2 for x in xs) / n; c4 = math.fsum((x - m)**4 for x in xs) / n
    return m, c2, c4 / c2**2 - 3 if c2 else float("nan")


def load(path):
    g = collections.defaultdict(list); trials = set()
    for r in csv.DictReader(gzip.open(path, "rt")):
        s = r["stage"]; trials.add(r["trial"])
        if s in ("0", "10"):
            s += "d" if int(r["index"]) < 4 else "t"
        g[s].append((float(r["pre"]), float(r["post"]), int(r["theta"])))
        if s == "12":
            g["12c" + r["index"]].append((float(r["pre"]), 0.0, 0))
    g["_trials"] = len(trials)
    return g


def screen_with(width, read):
    """Model events of the paper set with the given terminal read. The samples were
    recorded with the padded read, so the ratios are fitted against that model."""
    import st_retune
    params = cases.resolve("st-reported-r2", width)["parameters"]
    return st_retune.screen(dict(params, terminal_read=read), width)


def model_groups(width):
    a = screen_with(width, "padded")
    g = collections.defaultdict(list)
    for e in a["events"]:
        key = {v: k for k, v in SCREEN.items() if k[0] != "0"}.get(e["stage"])
        if e["stage"] == "emission_pbs":
            key = "0d" if e["block"] < 4 else "0t"
        if key: g[key].append(e)
    return a, g


def analyze(width, path):
    g = load(path); trials = g.pop("_trials"); a, mg = model_groups(width)
    ks = a["variances"]["ks"]
    out = {"width": width, "trials": trials, "stages": {}}
    print(f"\n=== W{width}: w{width}/{path.name}  trials={trials}")
    print(f"{'stage':6s} {'n':>6s} {'sd pre':>7s} {'model':>7s} {'ratio':>7s} {'kurt':>6s} "
          f"{'sd ms':>6s} {'model':>6s} {'sd post':>7s} {'mean':>7s} {'max|post|':>9s} {'z':>6s} {'log2p/read':>10s}")
    out["columns"] = {int(k[3:]): math.fsum(x[0]**2 for x in g[k]) / len(g[k]) for k in g if k.startswith("12c")}
    for k in sorted((k for k in g if not k.startswith("12c")), key=lambda s: (len(s) > 2, s)):
        v = g[k]; n = len(v); theta = v[0][2]
        pre = [x[0] for x in v]; post = [x[1] for x in v]
        mpre, vpre, kpre = moments(pre)
        rms_pre = math.fsum(x * x for x in pre) / n
        row = {"n": n, "sd_pre_log2": L2(rms_pre) / 2, "mean_pre": mpre, "excess_kurtosis_pre": kpre}
        model = None
        if k in mg:
            model = math.fsum(e["variance_before_ms"] for e in mg[k]) / len(mg[k])
            row.update(model_var_pre_log2=L2(model), ratio_var=rms_pre / model,
                       model_ms_var_log2=L2(mg[k][0]["ms_variance"]))
        line = f"{k:6s} {n:6d} {L2(rms_pre)/2:7.2f} " + (f"{L2(model)/2:7.2f} {rms_pre/model:7.4f} " if model else " " * 16) + f"{kpre:6.2f} "
        if k in ("0d", "0t", "1", "2", "9"):
            spacing = 2.0**(62 if k == "9" else 61)
            mpost, vpost, kpost = moments(post)
            ms = [b - a_ for a_, b in zip(pre, post)]
            _, vms, _ = moments(ms)
            sd = math.sqrt(vpost)
            lo = spacing / 2; hi = spacing / 2 - (2**theta - 1) * Q / (2 * N)
            z = min(lo + mpost, hi - mpost) / sd
            lp = two_sided(lo, hi, mpost, sd)
            row.update(sd_ms_log2=L2(vms) / 2, sd_post_log2=L2(vpost) / 2, mean_post=mpost,
                       max_abs_post_log2=L2(max(abs(x) for x in post)), excess_kurtosis_post=kpost,
                       z=z, log2_p_read=lp, margin_lo=lo, margin_hi=hi)
            line += (f"{L2(vms)/2:6.2f} " + (f"{L2(mg[k][0]['ms_variance'])/2:6.2f} " if model else " " * 7)
                     + f"{L2(vpost)/2:7.2f} {mpost/2**55:+6.2f}u {L2(max(abs(x) for x in post)):9.2f} {z:6.2f} {lp:10.1f}")
        print(line)
        out["stages"][k] = row
    return out


def implied_cmux(width, res):
    """Chain variance (stage 10t, pre-KS) per terminal selector use."""
    if "10t" not in res["stages"]:
        return None
    from reference import make_plan
    plan = make_plan(width)
    sel = sum(1 + int(c > 8 and len(plan.final[c - 1]) > 1) for c in range(8, width))
    v = 2 ** (2 * res["stages"]["10t"]["sd_pre_log2"])
    return sel, v / sel


PREKS = {"9": "19", "1": "11", "2": "12", "0d": "10d", "0t": "10t"}


def calibrate(width, res, read="padded"):
    """Per event: variance_before_ms = r_ks * ks + r_src * (model - ks); MS model kept.
    r_src = measured pre-KS variance / mean modelled source variance of that stage (terminal reads:
    per column); r_ks = pooled measured KS variance (pre-MS minus pre-KS) / model KS variance."""
    a, mg = model_groups(width)
    ks = a["variances"]["ks"]
    st = res["stages"]
    var = lambda k: 2 ** (2 * st[k]["sd_pre_log2"])
    ks_meas = [var(k) - var(PREKS[k]) for k in PREKS if k in st and PREKS[k] in st]
    r_ks = (math.fsum(ks_meas) / len(ks_meas)) / ks
    r_src = {k: var(PREKS[k]) / (math.fsum(e["variance_before_ms"] - ks for e in mg[k]) / len(mg[k]))
             for k in PREKS if k in st and PREKS[k] in st}
    col_model = collections.defaultdict(list)
    for e in mg.get("2", []):
        col_model[e["column"]].append(e["variance_before_ms"] - ks)
    r_col = {c: res["columns"][c] / (sum(v) / len(v)) for c, v in col_model.items() if c in res["columns"]}
    logs, per = [], collections.defaultdict(list)
    rev = {v: k for k, v in SCREEN.items()}
    target = a if read == "padded" else screen_with(width, read)
    assert len(target["events"]) == len(a["events"])
    for e, t in zip(a["events"], target["events"]):
        assert e["stage"] == t["stage"]
        if e["stage"] not in rev:
            lp = t["log2_p_fail"]
        else:
            key = rev[e["stage"]] if e["stage"] != "emission_pbs" else ("0d" if e["block"] < 4 else "0t")
            r = r_col.get(e["column"], r_src[key]) if key == "2" else r_src[key]
            v = r_ks * ks + r * (e["variance_before_ms"] - ks)
            # A sign read scales the measured pre-MS noise by the factor of its read.
            v *= t["variance_before_ms"] / e["variance_before_ms"]
            lp = log2_gaussian_outside(t["margin"], t["mean_bound"], v + t["ms_variance"])
        lp += math.log2(e.get("multiplicity", 1))
        logs.append(lp); per[e["stage"]].append(lp)
    union = lambda xs: max(xs) + math.log2(math.fsum(2**(x - max(xs)) for x in xs))
    return (target["conditional_log2_union"], union(logs), {k: (max(v), union(v)) for k, v in per.items()},
            {"r_ks": r_ks, "r_src": r_src, "r_col_max": max(r_col.values()) if r_col else None})


if __name__ == "__main__":
    D = Path(__file__).resolve().parent
    results = {}
    for w in (16, 32, 64, 128, 256):
        p = D / f"w{w}" / "probe.csv.gz"
        if p.exists():
            results[w] = analyze(w, p)
            ic = implied_cmux(w, results[w])
            if ic:
                sel, c = ic
                results[w]["implied_cmux_var_log2"] = L2(c)
                print(f"chain: {sel} selector uses; stage-10t var / uses = 2^{L2(c):.2f} "
                      f"(screen cmux 2^{L2(model_groups(w)[0]['variances']['cmux']):.2f}, per use 4*cmux)")
    print("\n=== Calibrated whole-multiplication log2 failure (per-event pre-MS variance rescaled by measured KS and source factors; MS model kept)")
    for w in (16, 32, 64, 128, 256):
        if w not in results:
            continue
        screen, cal, per, factors = calibrate(w, results[w])
        results[w]["calibrated"] = {"screen": screen, "calibrated": cal, "per_stage": per, **factors}
        print(f"W{w:3d}: screen {screen:7.1f} -> calibrated {cal:7.1f}  r_ks={factors['r_ks']:.3f} "
              + " ".join(f"r_src[{k}]=2^{L2(v):.2f}" for k, v in factors["r_src"].items())
              + (f" max r_col=2^{L2(factors['r_col_max']):.2f}" if factors["r_col_max"] else ""))
        print("      " + "  ".join(f"{k}: max {m:.1f} union {u:.1f}" for k, (m, u) in per.items() if k != "output_decode"))
    print("\n=== Same calibration with the sign terminal read (the read of the reconstruction in the paper)")
    for w in (16, 32, 64, 128, 256):
        if w not in results:
            continue
        screen, cal, per, factors = calibrate(w, results[w], "sign")
        results[w]["calibrated_sign"] = {"screen": screen, "calibrated": cal, "per_stage": per}
        print(f"W{w:3d}: screen {screen:7.1f} -> calibrated {cal:7.1f}")
        print("      " + "  ".join(f"{k}: max {m:.1f} union {u:.1f}" for k, (m, u) in per.items() if k != "output_decode"))
    (D / "analysis.json").write_text(json.dumps(results, indent=1, default=float) + "\n")
