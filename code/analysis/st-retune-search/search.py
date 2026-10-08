#!/usr/bin/env python3
"""Decomposition-only retune search for the revised-ST candidates.

n, N, k and both noise sigmas (hence lattice security) are fixed; only the
PBS/KS/auto/SS/CBS-base decompositions vary (CBS levels are fixed at 4 by the
Rust implementation). Feasibility = packaged Gaussian screen (st_retune.screen)
below TARGET at every width. Cost proxy (relative to the baseline) is dominated
by blind rotations x PBS levels, then key switches x KS levels.

Usage: python3 -B analysis/st-retune-search/search.py BASE.json [fft_scale]
"""
import copy, itertools, json, math, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/server-campaign"))
import st_retune
from reference import make_plan

TARGET = -128.0
base = json.loads(Path(sys.argv[1]).read_text())
fft_scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
if fft_scale != 1.0:
    P = st_retune.primitive
    for name in ("get_var_fft_pbs", "get_var_fft_ext_prod", "get_var_fft_glwe_ks"):
        f = getattr(P, name)
        setattr(P, name, (lambda f: lambda *a, **k: fft_scale * f(*a, **k))(f))

N, K, n = base["polynomial_size"], base["glwe_dimension"], base["lwe_dimension"]
U_BR = n * (K + 1)           # FFT-sized polynomial ops per blind-rotation level
U_KS = K * N * n / (N * 11)  # KS level in the same units (integer MACs / N log N)
U_CMUX, U_AUTO, U_SS = 2 * (K + 1), K + 2, 2 * (K + 1)

def cost(p, width=256):
    c = make_plan(width).counts()
    br = c["total_blind_rotations"]
    ks = c["dadda_pbs_manylut"] + c["terminal_binary_cbs"] + c["emission_pbs"] + width
    conv = c["input_grouped_cbs"] + c["terminal_binary_cbs"]
    cmux = c["product_cmux_standard_scalar_vp"]
    return (br * p["pbs"][1] * U_BR + ks * p["ks"][1] * U_KS
            + cmux * p["cbs"][1] * U_CMUX
            + conv * p["cbs"][1] * (int(math.log2(N)) * p["auto"][1] * U_AUTO + p["ss"][1] * U_SS))

def union(p, widths=(16, 32, 64, 128, 256)):
    try:
        return max(st_retune.screen(p, w)["conditional_log2_union"] for w in widths)
    except (ValueError, AssertionError, ZeroDivisionError, OverflowError):
        return float("inf")

def decomps(bases, levels, lo=1, hi=63):
    return [(b, l) for b in bases for l in levels if lo <= b * l <= hi]

base_cost = cost(base)
# Best-noise settings for the cheap-to-vary parts, chosen per (pbs, ks) below.
AUTO = decomps(range(4, 17), range(2, 11), 24, 62)
SS = decomps(range(8, 21), range(2, 6), 24, 62)
CBS = [(b, 4) for b in range(3, 9)]

def best_rest(p):
    best = None
    for auto in AUTO:
        q = dict(p, auto=list(auto))
        v = st_retune.variances(q)
        split = st_retune.split_high_rounding(q, 2 * 256 + 496)
        if split and split["log2_p_fail"] + math.log2(split["multiplicity"]) > TARGET - 20:
            continue
        for ss in SS:
            for cbs in CBS:
                r = dict(q, ss=list(ss), cbs=list(cbs))
                cm = st_retune.variances(r)["cmux"]
                if best is None or cm < best[0]:
                    best = (cm, r)
    return best[1]

rows = []
for pbs in decomps(range(6, 25), (1, 2, 3), 12, 62):
    for ks in decomps(range(1, 9), range(2, 31), 14, 40):
        p = dict(copy.deepcopy(base), pbs=list(pbs), ks=list(ks))
        if cost(p) > base_cost * 1.001:
            continue
        # Cheap necessary condition: the W=256 input-lift family alone.
        v = st_retune.variances(p)
        if p.get("cc2_big_key", False):
            lift = st_retune.event(p, "input_grouped_lift", 2**20 * 2 * v["pbs"] + v["ks"], theta=2,
                                   spacing=2**64/4, mean_bound=abs(v["ks_mean"]))
        else:
            lift = st_retune.event(p, "input_grouped_lift", 2**20 * 2 * v["closed"], theta=2,
                                   spacing=2**64/4, mean_bound=2**10 * 2 * abs(v["ks_mean"]))
        if lift["log2_p_fail"] + 8 > TARGET:
            continue
        rows.append((cost(p), p))
rows.sort(key=lambda x: x[0])
print(f"base cost 1.000, union {union(base):.2f}; {len(rows)} (pbs,ks) pairs pass the lift pre-filter",
      flush=True)
found, seen = [], set()
for c, p in rows:
    key = (tuple(p["pbs"]), tuple(p["ks"]))
    if key in seen:
        continue
    seen.add(key)
    q = best_rest(p)
    u = union(q)
    if u <= TARGET:
        found.append({"relative_cost": round(cost(q) / base_cost, 3), "max_union_log2": round(u, 2),
                      **{k: q[k] for k in ("pbs", "ks", "auto", "ss", "cbs")}})
        print(json.dumps(found[-1]), flush=True)
        if len(found) >= 8:
            break
print(json.dumps({"fft_scale": fft_scale, "target": TARGET, "found": len(found)}))
