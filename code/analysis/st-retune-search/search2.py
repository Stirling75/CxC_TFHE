#!/usr/bin/env python3
"""Retune search for the Shokri-Tsoutsos reconstruction under the common model.

For every (n, sigma) pair with a lattice-estimator result of at least 128 bits,
both ring shapes (N, k) = (2048, 1) and (1024, 2) (GLWE noise 2^-49.9, i.e. 129.2
bits for kN = 2048), terminal packing theta in {1, 2}, and the sign terminal
read, it finds the cheapest decompositions whose estimate meets 2^-128 at every
width. Unlike search.py, the remaining decompositions are chosen by cost, not by
CMux variance, and the search does not stop early. The cost is a proxy in
FFT-sized polynomial operations; the final choice is made by measurement.
Usage: python3 -B search2.py [shape] [theta]
"""
import itertools, json, math, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/server-campaign"))
import st_retune
from reference import make_plan

TARGET = -128.0
GLWE = 2.0**-49.9
LWE = [(656, 2.0**-13.925), (672, 2.0**-14.35), (688, 2.0**-14.775), (688, 2.0**-15.05),
       (696, 2.0**-15.0), (696, 2.0**-15.25), (704, 2.0**-15.2), (768, 2.0**-16.8)]
SHAPES = {"2048x1": (2048, 1), "1024x2": (1024, 2)}

def decomps(bases, levels, lo, hi):
    return [(b, l) for b in bases for l in levels if lo <= b * l <= hi]

def cost(p, width=128):
    N, k, n = p["polynomial_size"], p["glwe_dimension"], p["lwe_dimension"]
    c = st_retune.expected_runtime_counts(p, make_plan(width))
    fft = N * math.log2(N)
    br = c["blind_rotations"] * p["pbs"][1] * n * (k + 1) ** 2 * fft
    ks = c["key_switches"] * p["ks"][1] * k * N * n / 8
    cmux = c["cmux"] * p["cbs"][1] * (k + 1) ** 2 * fft
    conv = c["ggsw_conversions"] * p["cbs"][1] * (math.log2(N) * p["auto"][1] * k * (k + 1)
                                                  + p["ss"][1] * (k + 1) ** 2) * fft
    return br + ks + cmux + conv

def ok(p, widths=(256, 128, 64, 32, 16)):
    try:
        return all(st_retune.screen(p, w)["conditional_log2_union"] <= TARGET for w in widths)
    except (ValueError, AssertionError, ZeroDivisionError, OverflowError):
        return False

PBS = decomps(range(6, 25), (1, 2, 3), 12, 62)
KS = decomps(range(1, 9), range(2, 31), 14, 40)
AUTO = decomps(range(4, 17), range(2, 11), 24, 62)
SS = decomps(range(8, 21), range(2, 6), 24, 62)
CBS = [(b, 4) for b in range(3, 9)]
shapes = [sys.argv[1]] if len(sys.argv) > 1 else list(SHAPES)
thetas = [int(sys.argv[2])] if len(sys.argv) > 2 else [1, 2]
if len(sys.argv) > 3:
    LWE = [LWE[int(sys.argv[3])]]
results = []
for shape in shapes:
    N, k = SHAPES[shape]
    for theta in thetas:
        for n, sigma in LWE:
            base = dict(name=f"st-sign-n{n}-{shape}-t{theta}", lwe_dimension=n, polynomial_size=N,
                        glwe_dimension=k, lwe_sigma=sigma, glwe_sigma=GLWE, terminal_lut_count_log=theta,
                        split_trace_fft=True, cc2_big_key=True, terminal_read="sign")
            rest_all = [(a, s_, cb) for a in AUTO for s_ in SS for cb in CBS]
            quiet = {}
            for pb in PBS:
                # The decompositions with the least CMux variance give the best chance
                # to meet the target; they decide whether this PBS can work at all.
                q = dict(base, pbs=list(pb), ks=[2, 8])
                quiet[pb] = min(rest_all, key=lambda r: st_retune.variances(
                    dict(q, auto=list(r[0]), ss=list(r[1]), cbs=list(r[2])))["cmux"])
            pairs = sorted(((cost(dict(base, pbs=list(pb), ks=list(ks), auto=list(quiet[pb][0]),
                                       ss=list(quiet[pb][1]), cbs=list(quiet[pb][2]))), pb, ks)
                            for pb in PBS for ks in KS))
            best = None
            for c0, pb, ks in pairs:
                if best and c0 > best[0] * 1.3:
                    break
                a0, s0, cb0 = quiet[pb]
                q0 = dict(base, pbs=list(pb), ks=list(ks), auto=list(a0), ss=list(s0), cbs=list(cb0))
                if not ok(q0, (256,)) or not ok(q0):
                    continue
                rest = sorted((cost(dict(q0, auto=list(a), ss=list(s_), cbs=list(cb))), a, s_, cb)
                              for a, s_, cb in rest_all)
                for c1, a, s_, cb in rest:
                    if best and c1 >= best[0]:
                        break
                    q = dict(q0, auto=list(a), ss=list(s_), cbs=list(cb))
                    if ok(q, (256,)) and ok(q):
                        best = (c1, q)
                        break
            if best:
                c1, q = best
                u = max(st_retune.screen(q, w)["conditional_log2_union"] for w in (16, 32, 64, 128, 256))
                row = {"cost": c1, "max_union": round(u, 2), **{x: q[x] for x in
                       ("lwe_dimension", "lwe_sigma", "polynomial_size", "glwe_dimension",
                        "terminal_lut_count_log", "pbs", "ks", "auto", "ss", "cbs")}}
                results.append((c1, q, row))
                print(json.dumps(row), flush=True)
            else:
                print(json.dumps({"none": base["name"]}), flush=True)
results.sort(key=lambda x: x[0])
tag = f"-i{sys.argv[3]}" if len(sys.argv) > 3 else ""
out = Path(__file__).with_name(f"search2-{'-'.join(shapes)}-t{'-'.join(map(str, thetas))}{tag}.json")
out.write_text(json.dumps([dict(q, name=f"st-sign-n{q['lwe_dimension']}-{q['polynomial_size']}x{q['glwe_dimension']}-t{q['terminal_lut_count_log']}") for _, q, _ in results], indent=1))
print("wrote", out)
