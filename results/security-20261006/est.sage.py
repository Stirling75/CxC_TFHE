import sys, math
sys.path.insert(0, 'LATTICE_ESTIMATOR_PATH')
from sage.all import oo
from estimator import LWE, ND
q = 2**64
n, ls, tag = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
params = LWE.Parameters(n=n, q=q, Xs=ND.Uniform(0, 1), Xe=ND.DiscreteGaussian(2.0**ls * q))
r = LWE.estimate(params)
for k, v in r.items(): print(tag, k, float(math.log2(v["rop"])) if v["rop"] != oo else "inf", flush=True)
best = min((float(math.log2(v["rop"])), k) for k, v in r.items() if v["rop"] != oo)
print(f"RESULT {tag} n={n} log2sigma={ls}: {best[0]:.2f} ({best[1]})", flush=True)
