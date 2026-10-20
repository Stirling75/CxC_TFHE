"""Model values quoted in the text of the paper (make extra, several minutes):
the selector-error share of the largest reduction input (Section 5.4), the
estimate of product-sum lookup with groups of 36 (Section 7), and Trifan et al.
with the one-bit TUniform set of TFHE-rs 1.7.0 (Section 6.1). Writes
out/model-checks.txt (compare with expected-extra/model-checks.txt)."""
import copy, sys
from dataclasses import replace
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code" / "scripts"))
import cases, heterogeneous_screen as hs
import parameter_checks

e, out, seen = hs.e, [], []
analyze = hs.analyze
def spy(plan, profile, prim, identical=False, **k):
    seen.append((plan, prim, identical))
    return analyze(plan, profile, prim, identical=identical, **k)
hs.analyze = spy

def inputs(plan, prim, identical):
    columns = hs.product_columns(plan, prim)
    if identical:
        columns = hs.identical_operands(columns)
    _, jobs = hs.replay(plan, prim, columns)
    return [e.chunk_input_var_source_aware(g) for wave in jobs for _, g, _, _, _, _ in wave]

# Selector error times decomposition digits: the cmux_key_var part of every product term.
shares = []
for m in ("hybrid-grouped-rev-ld", "hybrid-cached-rev-ld"):
    for w in (16, 32, 64, 128, 256):
        seen.clear(); cases.resolve.cache_clear(); cases.resolve(m, w)
        for plan, prim, identical in seen[-2:]:
            ms, _ = e.centered_binary_ms_decision_noise(
                hs.base.profile_for(plan["chunk_bits"]).lwe_n, e.Q, hs.NORMALIZER_MS_MODULUS)
            full = inputs(plan, prim, identical)
            nokey = inputs(plan, replace(prim, cmux_key_var=0.0), identical)
            i = max(range(len(full)), key=full.__getitem__)
            share = (full[i] - nokey[i]) / (full[i] + prim.normalizer_ks_var + ms)
            shares.append(share)
            out.append(f"selector share {m} W={w} {'AA' if identical else 'AB'}: {100 * share:.1f}%")
out.append(f"selector share, largest: {100 * max(shares):.1f}%")

catalog = copy.deepcopy(cases.CATALOG)
for size in (32, 36):
    cases.CATALOG["hybrid-grouped-rev-ld"]["group_size"] = size
    cases.resolve.cache_clear()
    r = cases.resolve("hybrid-grouped-rev-ld", 256)
    out.append(f"product-sum W=256 M={size}: separate {r['analysis']['conditional_union_log2']:.1f}, "
               f"identical {r['analysis_identical']['conditional_union_log2']:.1f}")
cases.CATALOG.clear(); cases.CATALOG.update(catalog)

tuniform = parameter_checks.library_parameters("m1c1-tuniform")["log2_p_fail"]
cases.CATALOG["trifan"]["primitive_log2_p_fail"] = tuniform
for w in (128, 256):
    cases.resolve.cache_clear()
    r = cases.resolve("trifan", w)
    out.append(f"Trifan et al., TUniform ({tuniform} per PBS), W={w}: {r['whole_product_log2_estimate']:.1f}")

(HERE / "out").mkdir(exist_ok=True)
(HERE / "out" / "model-checks.txt").write_text("\n".join(out) + "\n")
print("\n".join(out))
