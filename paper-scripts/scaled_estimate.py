"""Section 6.2: the worst estimate of product-sum lookup with the modeled
variances before the key switch scaled by the largest measured ratio of
Table 6, rounded to 0.61, on the same schedule. The key-switching and
modulus-switching variances are kept, since Table 6 does not measure them."""
import copy, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code" / "scripts"))
import cases
cases.resolve("hybrid-grouped-rev-ld", 16)
est = sys.modules["cbs_variance_estimator"]
screen = sys.modules["heterogeneous_screen"]
chunk, term, assign = est.chunk_input_var_source_aware, est.term_var_source_aware, screen.assign_linear
for width in (128, 256):
    cases.resolve.cache_clear()
    base = cases.resolve("hybrid-grouped-rev-ld", width)
    plan = copy.deepcopy(base["plan"])
    screen.assign_linear = lambda *a, **k: copy.deepcopy(plan)
    est.chunk_input_var_source_aware = lambda *a, **k: 0.61 * chunk(*a, **k)
    est.term_var_source_aware = lambda *a, **k: 0.61 * term(*a, **k)
    cases.resolve.cache_clear()
    r = cases.resolve("hybrid-grouped-rev-ld", width)
    assert r["plan"] == plan
    print(f"W={width}: AB {base['analysis']['conditional_union_log2']:.1f} -> {r['analysis']['conditional_union_log2']:.1f}, "
          f"A=B {base['analysis_identical']['conditional_union_log2']:.1f} -> {r['analysis_identical']['conditional_union_log2']:.1f}")
    screen.assign_linear, est.chunk_input_var_source_aware, est.term_var_source_aware = assign, chunk, term
