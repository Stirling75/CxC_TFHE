"""Offline schedule of the Bernard et al. (ePrint 2026/2310) radix multiplier:
multi-value partial products, noise-aware folding (heaviest rule, balanced
column order) and TFHE-rs carry propagation.  Our reimplementation."""
import itertools, json, math, sys

# Normalized output standard deviation (coefficient norm) per output type.
SIGMA = {"Pls": math.sqrt(20), "Pms": math.sqrt(5), "Fls": math.sqrt(8), "Fms": 1.0,
         "P1": 1.0, "F1": 1.0}       # P1/F1: single-output PBS (top column)
CLASSIC = {k: 1.0 for k in SIGMA}    # TFHE-rs: every output is a fresh PBS
ORDER = ["Pls", "Fls", "Pms", "P1", "F1", "Fms"]  # noisiest first


def products(nb, mvb=True):
    """Partial-product BRs. Each item: (i, j, outputs[(column, type)])."""
    ops = []
    for i in range(nb):
        for j in range(nb - i):
            c = i + j
            if c + 1 < nb:
                outs = [(c, "Pls"), (c + 1, "Pms")] if mvb else None
                if mvb:
                    ops.append(("pmvb", i, j, outs))
                else:
                    ops.append(("plsb", i, j, [(c, "P1")]))
                    ops.append(("pmsb", i, j, [(c + 1, "P1")]))
            else:
                ops.append(("plsb", i, j, [(c, "P1")]))
    return ops


def best_fold(terms, phi, sigma, rule):
    """terms: list of term ids with types; returns a list of ids to fold."""
    by_type = {}
    for tid, ty in terms:
        by_type.setdefault(ty, []).append(tid)
    types = [t for t in ORDER if t in by_type]
    best, best_key = None, None
    def rec(k, chosen, count, noise):
        nonlocal best, best_key
        if k == len(types):
            if count >= 2:
                distinct = sum(1 for c in chosen if c)
                if rule == "heaviest":
                    key = (noise, distinct, count)
                elif rule == "cheapest":
                    key = (count, -noise)
                else:  # expensive: lexicographic by noisiest types
                    key = tuple(chosen)
                if best_key is None or key > best_key:
                    best, best_key = list(chosen), key
            return
        t = types[k]
        for m in range(0, min(len(by_type[t]), 5 - count) + 1):
            nn = noise + m * sigma[t]
            if nn > phi + 1e-9:
                break
            rec(k + 1, chosen + [m], count + m, nn)
    rec(0, [], 0, 0.0)
    if best is None:
        return None
    picked = []
    for t, m in zip(types, best):
        picked += [(tid, t) for tid in by_type[t][:m]]
    return picked


def column_ok(terms, phi, sigma):
    return len(terms) <= 5 and sum(sigma[t] for _, t in terms) <= phi + 1e-9


def schedule(width, threads, phi=12.8, mvb=True, rule="heaviest", order="balanced", classic=False):
    nb = width // 2
    sigma = CLASSIC if classic else SIGMA
    if classic:
        phi = 5.0  # TFHE-rs: arithmetic condition only (five terms)
    ids = itertools.count()
    cols = [[] for _ in range(nb)]
    ticks = []
    # Partial products: data independent, issued first in ticks of T.
    pops = products(nb, mvb)
    out_ops = []
    for op in pops:
        kind, i, j, outs = op
        o = [(c, t, next(ids)) for c, t in outs]
        out_ops.append({"kind": kind, "i": i, "j": j, "out": [[c, t, tid] for c, t, tid in o]})
    for k in range(0, len(out_ops), threads):
        ticks.append(out_ops[k:k + threads])
    for op in out_ops:
        for c, t, tid in op["out"]:
            cols[c].append((tid, t))
    # Column folding.
    while not all(column_ok(c, phi, sigma) for c in cols):
        tick, landing = [], [[] for _ in range(nb)]
        avail = [list(c) for c in cols]
        progress = True
        while progress and len(tick) < threads:
            progress = False
            rng = range(nb) if order in ("balanced", "forward") else range(nb - 1, -1, -1)
            for c in rng:
                if len(tick) >= threads:
                    break
                if column_ok(avail[c] + landing[c], phi, sigma):
                    continue
                if classic:
                    if len(avail[c]) < 5:
                        continue
                    pick = avail[c][:5]
                else:
                    pick = best_fold(avail[c], phi, sigma, rule)
                    if pick is None:
                        continue
                ids_in = {tid for tid, _ in pick}
                avail[c] = [x for x in avail[c] if x[0] not in ids_in]
                if c + 1 < nb:
                    lo, hi = (c, "Fls" if not classic else "F1", next(ids)), (c + 1, "Fms" if not classic else "F1", next(ids))
                    outs, kind = [lo, hi], ("fmvb" if not classic else "fpair")
                    landing[c].append((lo[2], lo[1])); landing[c + 1].append((hi[2], hi[1]))
                else:
                    lo = (c, "F1", next(ids)); outs, kind = [lo], "fsingle"
                    landing[c].append((lo[2], lo[1]))
                tick.append({"kind": kind, "col": c, "in": [tid for tid, _ in pick],
                             "in_types": [t for _, t in pick],
                             "out": [[a, b, d] for a, b, d in outs]})
                progress = True
                if order != "balanced":
                    break
        if not tick:
            raise RuntimeError("stuck")
        ticks.append(tick)
        cols = [avail[c] + landing[c] for c in range(nb)]
    return {"width": width, "threads": threads, "phi": phi, "ticks": ticks,
            "final": [[[tid, t] for tid, t in c] for c in cols]}


if __name__ == "__main__":
    for T in (1, 8, 12, 64, 96, 192):
        s = schedule(64, T)
        print(T, len(s["ticks"]))


def export(directory, widths=(16, 32, 64, 128, 256), threads=(1, 4, 16, 64)):
    import os
    os.makedirs(directory, exist_ok=True)
    for W in widths:
        for T in threads:
            s = schedule(W, T)
            with open(f"{directory}/w{W}-t{T}.json", "w") as f:
                json.dump(s, f)
