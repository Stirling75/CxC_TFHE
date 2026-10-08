"""Section 6.2: the worst estimate of product-sum lookup (W=256, identical
operands) with every modeled variance scaled by the largest measured ratio of
Table 6, rounded to 0.61, on the same schedule."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code" / "scripts"))
import cases
cases.resolve("hybrid-grouped-rev-ld", 16)
est = sys.modules["cbs_variance_estimator"]
orig = est.log2_erfc
for scale in (1.0, 0.61):
    est.log2_erfc = lambda x, s=scale: orig(x / math.sqrt(s))
    cases.resolve.cache_clear()
    r = cases.resolve("hybrid-grouped-rev-ld", 256)
    print(f"variance x{scale}: AB {r['analysis']['conditional_union_log2']:.1f}, "
          f"A=B {r['analysis_identical']['conditional_union_log2']:.1f}")
