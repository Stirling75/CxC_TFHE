#!/usr/bin/env python3
"""Conditional CLOT-style variance screen of the encrypted BFV probe.

No fitted probe floor, parameter approval, or security certification. The
layout-aware extension assumes variance-additive primitive errors, as do the
underlying literature estimates. See NOISE_ANALYSIS.md for its scope.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SNAPSHOT = HERE.parents[2] / "ring-variant-multiplier/model/source_snapshot"
sys.path.insert(0, str(SNAPSHOT))
import cbs_variance_estimator as primitive

Q = 2**64
NATIVE_DELTA = Q // 32


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def packing_variances(n, digits, incoming, sigma, base_log, levels):
    """CLOT Appendix D: the input/rounding term is present only at filled slots."""
    base = 2**base_log
    rounding = n * ((Q**2 / base**(2*levels) - 1) / 24 + 1/16)
    key = digits * n * levels * (base*base + 2)/12 * (sigma*Q)**2
    return {"filled": incoming + rounding + key, "empty": key,
            "rounding": rounding, "key": key, "incoming": incoming}


def tensor_terms(n, digits, column, delta, fill, empty, sigma, relin, dense=False):
    """k=1, binary S: CLOT Eq. (1), or its C.1 coefficient-sum adaptation.

    Message coefficients are bounded by 3, including carry-producing inputs.
    C.2's single-product specialization is NOT used for our convolution.
    """
    if dense:
        linear = 18*n*fill
        quadratic = n*fill*fill/delta**2
        noise_sum = 2*n*fill
    else:
        overlap = sum((column-i) % n < digits for i in range(digits))
        linear = 18*(overlap*fill + (digits-overlap)*empty)
        quadratic = (overlap*fill*fill + 2*(digits-overlap)*fill*empty
                     + (n-2*digits+overlap)*empty*empty)/delta**2
        noise_sum = 2*(digits*fill + (n-digits)*empty)
    # q^2 E[U^2], with the quotient approximation in CLOT C.1, pp. 55-56.
    quotient_moment = ((Q*Q-1)/12 * (1+n/2) + n/16 + (1+n/2)**2/4)
    quotient = quotient_moment * noise_sum / delta**2
    odd = 3*n/8
    even = odd - 1/4
    mean_square = (n*n+2)/48
    second_moment_sum = odd + even + 2*mean_square
    rounding = (1/12 + n/(12*delta**2)*((delta**2-1)/2+3/4)
                + n/(24*delta**2)*((delta**2-1)*second_moment_sum+3*(odd+even)))
    base_log, levels = relin
    base = 2**base_log
    relin_key = levels*n*(sigma*Q)**2*(base*base+2)/12
    relin_rounding = (n/2*(Q*Q/(12*base**(2*levels))-1/12)*second_moment_sum
                     + n/8*(odd+even))
    return {"message_times_error": linear, "error_times_error": quadratic,
            "quotient_times_error": quotient, "tensor_rounding": rounding,
            "relin_key": relin_key, "relin_rounding": relin_rounding}


def logsum(values):
    top = max(values)
    return top + math.log2(sum(2**(x-top) for x in values))


def make_plan(width, degree_aware=False, normalizer_domain=16):
    d = width//2
    columns = [[] for _ in range(d)]
    bits_total = 0
    for col in range(d):
        bound = 9*(col+1)
        bits = min(bound.bit_length(), 2*(d-col))
        bits_total += bits
        for j in range((bits+1)//2):
            columns[col+j].append(min(3, bound//4**j))
    norm = 0
    for col in range(d):
        while (sum(columns[col]) >= normalizer_domain if degree_aware
               else len(columns[col]) > 4):
            take = 0
            total = 0
            for degree in columns[col]:
                if degree_aware and total+degree >= normalizer_domain:
                    break
                if not degree_aware and take == 4:
                    break
                total += degree
                take += 1
            assert take >= 2
            columns[col] = columns[col][take:] + [min(3, total)]
            norm += 1
            if col+1 < d:
                columns[col+1].append(total//4)
                norm += 1
        total = sum(columns[col])
        norm += 1
        if col+1 < d and (total >= 4 if degree_aware else len(columns[col]) > 1):
            columns[col+1].append(total//4)
            norm += 1
    return {"width": width, "digits": d, "encoding_precision": (9*d).bit_length(),
            "pbs_input_rescale": 2*d, "pbs_bit_extraction": bits_total,
            "pbs_normalization": norm, "pbs_total": 2*d+bits_total+norm}


def screen(parameters, plan, fft, dense):
    n, small = parameters["polynomial_size"], parameters["n"]
    d, precision = plan["digits"], plan["encoding_precision"]
    delta = Q // 2**precision
    sigma = parameters["glwe_sigma"]
    input_factor = 16 // parameters.get("input_lut_domain", 16)
    norm_factor = 16 // parameters.get("normalizer_lut_domain", 16)
    degree_aware = parameters.get("degree_aware", False)
    capacity = parameters.get("normalizer_lut_domain", 16)-1

    def pbs_variance(decomposition):
        algebraic = primitive.get_var_pbs(n, 1, small, Q, sigma*sigma, *decomposition)
        numerical = primitive.get_var_fft_pbs(n, 1, small, *decomposition) if fft else 0.0
        return {"algebraic": algebraic, "fft_model": numerical,
                "total": algebraic+numerical}

    ordinary = pbs_variance(parameters["pbs"])
    encoding = pbs_variance(parameters["encoding_pbs"])
    ks = primitive.get_var_lwe_ks(n, Q, parameters["lwe_sigma"]**2, *parameters["ks"])
    packing = packing_variances(n, d, encoding["total"], sigma, *parameters["packing"])
    terms = [tensor_terms(n, d, col, delta, packing["filled"], packing["empty"],
                          sigma, parameters["relin"], dense) for col in range(d)]
    product = [sum(parts.values()) for parts in terms]
    event_rows = []
    source_variances = {}

    def fresh(variance):
        index = len(source_variances)
        source_variances[index] = variance
        return {index: 1.0}

    def variance(source):
        return sum(weight*weight*source_variances[i] for i, weight in source.items())

    def combine(*sources):
        result = {}
        for source in sources:
            for i, weight in source.items():
                result[i] = result.get(i, 0.0) + weight
        return result

    def scaled(source, scale):
        return {i: weight*scale for i, weight in source.items()}

    def decision(kind, label, incoming, spacing):
        tail = primitive.pbs_input_log2_pfail(small, Q, n, 0, spacing, incoming+ks)
        event_rows.append({"kind": kind, "label": label,
                           "variance_before_ks": incoming, "spacing": spacing,
                           "log2_gaussian": tail})

    for i in range(2*d):
        decision("input_reencoding", i, input_factor**2*ordinary["total"],
                 input_factor*NATIVE_DELTA)
    columns = [[] for _ in range(d)]
    degrees = [[] for _ in range(d)]
    for col in range(d):
        residual = fresh(product[col])
        bits = min((9*(col+1)).bit_length(), 2*(d-col))
        digit = None
        for j in range(bits):
            weight = delta * 2**j
            base = min(weight, NATIVE_DELTA)
            factor = 2**(precision-j-1)
            decision("bit_extraction", [col, j], factor**2*variance(residual), Q/2)
            bit = fresh(ordinary["total"])
            residual = combine(residual, scaled(bit, -weight/base))
            native_bit = scaled(bit, NATIVE_DELTA/base)
            if j % 2 == 0:
                digit = native_bit
                columns[col+j//2].append(digit)
                degrees[col+j//2].append(min(3, 9*(col+1)//4**(j//2)))
            else:
                columns[col+j//2][-1] = combine(digit, scaled(native_bit, 2))

    for col in range(d):
        while (sum(degrees[col]) > capacity if degree_aware else len(columns[col]) > 4):
            take, total = 0, 0
            for degree in degrees[col]:
                if (degree_aware and total+degree > capacity) or (not degree_aware and take == 4):
                    break
                take, total = take+1, total+degree
            assert take >= 2
            batch, columns[col] = columns[col][:take], columns[col][take:]
            degrees[col] = degrees[col][take:] + [min(3, total)]
            incoming = norm_factor**2*variance(combine(*batch))
            decision("normalization", [col, "low"], incoming, norm_factor*NATIVE_DELTA)
            columns[col].append(fresh(ordinary["total"]))
            if col+1 < d:
                decision("normalization", [col, "carry"], incoming, norm_factor*NATIVE_DELTA)
                columns[col+1].append(fresh(ordinary["total"]))
                degrees[col+1].append(total//4)
        incoming = norm_factor**2*variance(combine(*columns[col]))
        decision("normalization", [col, "final"], incoming, norm_factor*NATIVE_DELTA)
        total = sum(degrees[col])
        if col+1 < d and (total >= 4 if degree_aware else len(columns[col]) > 1):
            decision("normalization", [col, "final-carry"], incoming, norm_factor*NATIVE_DELTA)
            columns[col+1].append(fresh(ordinary["total"]))
            degrees[col+1].append(total//4)
        event_rows.append({"kind": "output_decoding", "label": col,
                           "variance": ordinary["total"],
                           "log2_gaussian": primitive.tail_log2_pfail(
                               NATIVE_DELTA/2, ordinary["total"], "gaussian", 1)})

    families = {}
    for kind in ("input_reencoding", "bit_extraction", "normalization", "output_decoding"):
        rows = [row for row in event_rows if row["kind"] == kind]
        families[kind] = {"count": len(rows),
                          "raw_log2_union": logsum([row["log2_gaussian"] for row in rows])}
    for kind, key in (("input_reencoding", "pbs_input_rescale"),
                      ("bit_extraction", "pbs_bit_extraction"),
                      ("normalization", "pbs_normalization")):
        assert families[kind]["count"] == plan[key], (kind, plan)
    assert len(event_rows) == plan["pbs_total"] + d
    raw = logsum([row["log2_gaussian"] for row in event_rows])
    worst = max(event_rows, key=lambda row: row["log2_gaussian"])
    return {"width": plan["width"], "precision": precision,
            "layout_model": "dense-envelope" if dense else "layout-aware",
            "pbs_fft_model_included": fft, "ordinary_pbs": ordinary,
            "encoding_pbs": encoding, "ks_increment": ks, "packing": packing,
            "max_product_variance": max(product),
            "max_product_sigma_over_margin": math.sqrt(max(product))/(delta/2),
            "last_product_variance_terms": terms[-1], "families": families,
            "events": len(event_rows), "worst_event": worst,
            "raw_log2_union": raw, "log2_union_capped": min(0, raw),
            "conditional_model_target_pass": raw < -128,
            "whole_multiplier_approved": False, "event_trace": event_rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["method"] != "independent-clot-style-convolution-probe":
        parser.error("not a BFV probe manifest")
    rows = [screen(manifest["parameters"], plan, fft, dense)
            for plan in manifest["plans"] for fft in (False, True) for dense in (False, True)]
    report = {"model": "CLOT-k1-binary-conditional-Gaussian-screen-v2-range-aware",
              "input_manifest_sha256": digest(args.manifest),
              "input_manifest": str(args.manifest), "parameters": manifest["parameters"],
              "source_sha256": {str(path): digest(path) for path in
                  (Path(__file__), SNAPSHOT / "cbs_variance_estimator.py")},
              "references": ["CLOT 2021, Eq. (1), Table 5, Appendices B/C/D",
                             "TFHE-rs Handbook v1.6, Section 2.5.3, Theorem 11",
                             "Existing primitive estimator for PBS/KS/mod-switch/FFT"],
              "assumptions": ["General XY, not identified/squaring operands",
                  "Centered Gaussian approximation and literature variance-additive heuristic",
                  "Shared evaluation-key and cross-primitive correlations not certified",
                  "PBS numerical term is the existing Refined-style FFT model, not a measured floor",
                  "Dense-envelope is within the same stochastic model, not an assumption-free bound",
                  "Successful PBS output errors are modeled under the prior-correctness approximation",
                  "Exact integer tensor/packing/relinearization introduce no floating-point error"],
              "whole_multiplier_approved": False, "security_estimate": None, "rows": rows}
    with args.output.open("x") as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write("\n")
    for row in rows:
        print(f"W={row['width']:3} {row['layout_model']:14} FFT={row['pbs_fft_model_included']} "
              f"log2_union={row['raw_log2_union']:.6f} "
              f"sigma/margin={row['max_product_sigma_over_margin']:.6g}")


if __name__ == "__main__":
    main()
