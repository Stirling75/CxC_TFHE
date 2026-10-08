"""Whole-multiplication failure estimate of our reimplementation of the
Bernard et al. multiplier, under the model used for every other method:
Gaussian decision noise at each blind-rotation input (variance-additive
inputs; KS, centered binary MS and the asymmetric LUT cell of TFHE-rs) and a
union bound over all blind rotations of the operation."""
import collections, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
RING = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, RING, os.path.join(RING, "model"), os.path.join(RING, "model", "source_snapshot")]
import schedule as S
import heterogeneous_screen as hs
e = hs.e
Q = 2.0**64
V2 = {t: s * s for t, s in S.SIGMA.items()}
# PBS of TFHE-rs 1.6.1 full_propagate_parallelized on the folded columns
# (pbs-stats, all widths, identical across thread counts and trials).
PROPAGATION_PBS = {16: 25, 32: 72, 64: 156, 128: 324, 256: 754}
PG128 = dict(n=930, lwe_std=6.782362904013915e-07, glwe_std=2.845267479601915e-15, pbs=(23, 1), N=2048, k=1)


def primitive(ks, p=PG128):
    vpbs = (e.get_var_pbs(p["N"], p["k"], p["n"], Q, p["glwe_std"]**2, *p["pbs"])
            + e.get_var_fft_pbs(p["N"], p["k"], p["n"], *p["pbs"]))
    vks = e.get_var_lwe_ks(p["k"] * p["N"], Q, p["lwe_std"]**2, *ks)
    return vpbs, vks


def log2_pfail(nu2, ks, p=PG128):
    vpbs, vks = primitive(ks, p)
    return hs.normalizer_input_log2_pfail(p["n"], nu2 * vpbs + vks)


def events(s, identical=False):
    """Blind-rotation inputs by variance multiplier (units of the PBS output variance)."""
    ev = collections.Counter()
    for tick in s["ticks"]:
        for op in tick:
            if op["kind"] in ("pmvb", "plsb"):
                # TFHE-rs bivariate input 4x+y of two nominal blocks (25 when x is y).
                ev[25 if identical and op["i"] == op["j"] else 17] += 1
            else:
                ev[round(sum(V2[t] for t in op["in_types"]), 6)] += 1
    # Propagation: message and carry extraction of every folded column sum,
    # the remaining PBS at the library's maximum noise level 5.
    extraction = 0
    for col in s["final"]:
        if col:
            ev[round(sum(V2[t] for _, t in col), 6)] += 2
            extraction += 2
    ev[5] += max(0, PROPAGATION_PBS[s["width"]] - extraction)
    return ev


def union(ev, ks, p=PG128):
    return e.log2_sum_exp(math.log2(c) + log2_pfail(nu2, ks, p) for nu2, c in ev.items())


def estimate(width, threads, ks, phi=12.8):
    s = S.schedule(width, threads, phi)
    return {"ab": union(events(s), ks), "identical": union(events(s, True), ks),
            "blind_rotations": sum(map(len, s["ticks"])), "ticks": len(s["ticks"])}


if __name__ == "__main__":
    for ks in ((4, 4), (3, 6)):
        for W in (16, 32, 64, 128, 256):
            r = [estimate(W, T, ks) for T in (1, 4, 16, 64)]
            print(ks, W, " ".join(f"T{T}:{x['ab']:.1f}/{x['identical']:.1f}" for T, x in zip((1, 4, 16, 64), r)))
