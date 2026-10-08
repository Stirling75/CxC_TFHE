"""Conditional Gaussian sensitivity review, NOT an approved parameter selector.

Retains the primitive marginal Gaussian estimates in the frozen estimator, but
does not assume independent trace coefficients or independent trace stages.
FFT marginal estimates remain implementation-dependent model assumptions.
"""
import argparse
import csv
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import baseline_study as base

e = base.est


def split_high_rounding(profile, width, split_bits):
    """Conditional Gaussian screen for rounding the high FFT part to integers.

    Uses the inherited single-convolution FFT model, amplitude-added over the
    k*ell products in one KS output coefficient. Neither these products nor
    different rounding events need be independent for this screen. The FFT
    marginal model itself is still an assumption, not a proved error bound.
    """
    if split_bits is None:
        return {"events": 0, "variance": 0.0, "union_log2": float("-inf")}
    if not isinstance(split_bits, int) or not 0 < split_bits < 64:
        raise ValueError("split_bits must be an integer strictly between 0 and 64")
    if not isinstance(width, int) or width <= 0:
        raise ValueError("positive integer width required")
    n, k, ell = profile.poly_n, profile.glwe_k, profile.auto_level
    count = k * ell
    single = e.get_var_fft_glwe_ks(n, 1, profile.auto_base_log, 1, 2.0**(64-split_bits))
    variance = count**2 * single
    # Upper-count both selectors of every input block and all their GLev levels.
    events = 2 * width * profile.cbs_level * (n.bit_length()-1) * (k+1) * n
    tail = e.two_sided_log2_pfail(0.5, 0.0, variance)
    return {"events": events, "variance": variance,
            "union_log2": math.log2(events)+tail}


def reassess(profile, *, split_bits=None):
    split_high_rounding(profile, 1, split_bits)
    n, k, q = profile.poly_n, profile.glwe_k, e.Q
    if k != 1:
        raise ValueError("this review is specialized to the artifact's GLWE dimension one")
    legacy = e.primitive_vars(profile, 4, 2, "linear", "sage", "revhomtrace", "refined-cbs")
    auto_crypto = e.get_var_glwe_ks(n,k,q,profile.glwe_var,profile.auto_base_log,profile.auto_level)
    fft_bound = q if split_bits is None else 2.0**split_bits
    auto_fft = e.get_var_fft_glwe_ks(n,k,profile.auto_base_log,profile.auto_level,fft_bound)
    # Independent addition is not needed between these two component errors.
    auto_sigma = math.sqrt(auto_crypto)+math.sqrt(auto_fft)
    ms_sigma = 2*math.sqrt(e.get_var_modswitch_1bit(n,k))
    # Remaining trace is a stride projection. Each output of S*P_r e_r
    # contains at most 2^r coefficients; sum_{r=1}^logN 2^r = 2(N-1).
    trace_sigma = 2*(n-1)*(auto_sigma+ms_sigma)
    pbs = e.get_var_pbs(n,k,profile.lwe_n,q,profile.glwe_var,
        profile.cbs_lift_pbs_base_log,profile.cbs_lift_pbs_level)
    pbs += e.get_var_fft_pbs(n,k,profile.lwe_n,profile.cbs_lift_pbs_base_log,profile.cbs_lift_pbs_level)
    b = 2.0**profile.ss_base_log
    ell = profile.ss_level
    rounding = (q*q/b**(2*ell)-1)/12
    # Conditional iid rounding-coordinate model: ||S||_2^2 <= N and
    # ||S*S||_2^2 <= N^3. This is not (N/2) times a scalar phase variance.
    ss_rounding = rounding*(n+n**3)
    ss_key = (k+1)*ell*n*((b*b+2)/12)*legacy.input_glwe_var
    ss_fft = e.get_var_fft_ext_prod(n,k,q,profile.ss_base_log,ell)
    ss_sigma = math.sqrt(ss_rounding)+math.sqrt(ss_key)+math.sqrt(ss_fft)
    selector_var = (math.sqrt(pbs)+trace_sigma+ss_sigma)**2
    ext = e.get_var_ext_prod(n,k,q,selector_var,profile.cbs_base_log,profile.cbs_level)
    ext += e.get_var_fft_ext_prod(n,k,q,profile.cbs_base_log,profile.cbs_level)
    updated = replace(legacy,cbs_lift_var=selector_var,cmux_ext_var=ext,
        cmux_key_var=ext-legacy.cmux_gadget_var,
        product_diag_var=4*ext,product_fold_var=8*ext,
        product_chunk2x2_var=8*ext,product_chunk4x4_split_var=16*ext)
    return updated, {"legacy_selector_var":legacy.cbs_lift_var,"selector_var":selector_var,
        "pbs":pbs,"trace_sigma":trace_sigma,"ss_sigma":ss_sigma,
        "auto_crypto":auto_crypto,"auto_fft":auto_fft,
        "scope":"Conditional trace covariance envelope with inherited primitive Gaussian marginal models. High-part rounding is added per width; no approval is issued."}


def multiplier(bits, width, profile, primitive, split_bits=None):
    columns=e.product_terms(width//2,primitive,contract=base.CONTRACTS[bits],identify_operands=False)
    result=base.simulate(columns,width//2,profile,primitive,policy="baseline")
    rounding = split_high_rounding(profile, width, split_bits)
    union = e.log2_sum_exp([result.lift_union_log2_pfail,
        result.product_union_log2_pfail, result.normalizer_union_log2_pfail,
        result.final_union_log2_pfail, rounding["union_log2"]])
    return {"bits":bits,"width":width,"raw_log2_union":union,
        "log2_probability_bound":min(0,union),"pbs":result.pbs,
        "lift_log2":result.lift_union_log2_pfail,
        "normalizer_log2":result.normalizer_union_log2_pfail,
        "final_log2":result.final_union_log2_pfail,
        "split_high_rounding_events":rounding["events"],
        "split_high_rounding_log2":rounding["union_log2"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    paths = [Path(__file__), Path(base.__file__),
             base.ROOT / "source_snapshot/cbs_variance_estimator.py",
             base.ROOT / "source_snapshot/parameters.json"]
    metadata = {
        "approved": False,
        "scope": "General XY; conditional Gaussian sensitivity only, not a complete new failure estimator or parameter approval.",
        "assumptions": [
            "Inherited primitive marginal Gaussian models, including the published FFT heuristic",
            "Amplitude addition across trace coefficients and stages",
            "Independent scheme-switch rounding-coordinate model conditional on the secret",
            "Inherited external-product and normalizer source model; no new proof of coefficient independence in these stages",
            "No empirical floor, fitted factor, or Chernoff tail",
        ],
        "source_sha256": {str(p.relative_to(base.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2)+"\n")
    rows=[]
    for bits in (4,8):
        original=base.profile_for(bits)
        candidates=[("legacy",original,None), ("trace-envelope",original,None),
            ("split40-only",original,40)]
        for a,l in ((4,12),(3,16),(2,24)):
            candidates.append((f"split40-auto{a}x{l}-ss13x4",
                               replace(original,auto_base_log=a,auto_level=l,ss_level=4),40))
        for name, profile, split in candidates:
            if name=="legacy":
                _, primitive=base.primitive_for(bits)
                detail={"scope":"Prior model, unapproved"}
            else:
                primitive,detail=reassess(profile,split_bits=split)
            for width in base.WIDTHS:
                row=multiplier(bits,width,profile,primitive,split)
                row.update(model=name,approved=False,auto=f"{profile.auto_base_log}x{profile.auto_level}",
                    ss=f"{profile.ss_base_log}x{profile.ss_level}",
                    selector_log2_variance=math.log2(primitive.cbs_lift_var))
                rows.append(row)
            (args.output/f"{bits}-{name}.json").write_text(json.dumps({"profile":vars(profile),"detail":detail},indent=2)+"\n")
    with (args.output/"comparison.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    for row in rows:
        if row["width"]==256: print(json.dumps(row))


if __name__=="__main__": main()
