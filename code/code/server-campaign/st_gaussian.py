"""Gaussian screen of the hypothesized revised-ST closing LWE key switch.

The paper's Table 2 KS decomposition is tested at the Section 3.1 CC2 scale.
The placement of this KS after emission is a reconstruction assumption, not
confirmed author code. The screen is not a whole-ST failure certificate.
"""
import math


def _log2_erfc(x):
    if x < 0:
        return math.log2(math.erfc(x))
    direct = math.erfc(x)
    if direct:
        return math.log2(direct)
    inverse_square = 1 / (x * x)
    correction = 1 - inverse_square / 2 + 3 * inverse_square**2 / 4
    return (-x*x - math.log(x) - math.log(math.pi)/2 + math.log(correction)) / math.log(2)


def log2_gaussian_outside(margin, mean, variance):
    if not all(math.isfinite(v) for v in (margin, mean, variance)):
        raise ValueError("finite margin, mean, and variance required")
    if margin <= 0 or variance < 0:
        raise ValueError("positive margin and nonnegative variance required")
    if variance == 0:
        return -math.inf if abs(mean) < margin else 0.0
    scale = math.sqrt(2 * variance)
    halves = [_log2_erfc((margin - mean) / scale) - 1,
              _log2_erfc((margin + mean) / scale) - 1]
    pivot = max(halves)
    return min(0.0, pivot + math.log2(sum(2**(v-pivot) for v in halves)))


def rounding_moments(dimension, precision, q_bits=64):
    """Uniform mask residues and iid Bernoulli(1/2) source secret.

    Nearest rounding with upward ties has r in [-step/2, step/2-1].
    E[s*r]=-1/4 and Var(s*r)=(step^2-1)/24+1/16 per coordinate.
    These are marginal moments, not a proof about PBS-produced masks.
    """
    if not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("positive integer dimension required")
    if not isinstance(precision, int) or not 0 < precision < q_bits:
        raise ValueError("precision must lie strictly between zero and q_bits")
    step = 2 ** (q_bits - precision)
    return -dimension / 4, dimension * ((step*step - 1) / 24 + 1 / 16)


def signed_digit_second_moments(base_log, levels, q_bits=64):
    """Exact per-level E[d^2] of the TFHE-rs native signed decomposition.

    Mirrors tfhe 1.6.1 SignedDecomposer::init_decomposer_state (balanced
    rounding of the top base_log*levels bits) and decompose_one_level (digit
    res-B*carry, carry iff res>B/2 or res==B/2 and the next state's bit
    base_log-1 is set). For a uniform input the rounded value is uniform, so
    the raw base-B chunks are iid uniform. A finite Markov chain over
    (incoming carry, raw chunk, all-lower-chunks-zero) is exact; the last
    flag resolves the balancing tie. Returned in decomposition order, least
    significant level first. Verified exhaustively against a bit-exact u64
    simulation in test_st_retune.py.

    The digits are correlated through the carry, so the result differs from
    the independent-uniform value (B^2+2)/12: about 1/3 instead of 1/2 for
    B=2, and 5.278 instead of 5.5 per inner level for B=8.
    """
    if base_log <= 0 or levels <= 0 or base_log * levels >= q_bits:
        raise ValueError("invalid signed decomposition")
    base = 1 << base_log
    half = base // 2
    weight = 1.0 / base
    state = {(0, x, True): weight for x in range(base)}
    moments = []
    for level in range(levels):
        second = 0.0
        following = {}
        for (carry_in, chunk, lower_zero), probability in state.items():
            total = chunk + carry_in
            overflow = int(total == base)
            residue = total - base * overflow
            if level + 1 < levels:
                successors = [(nxt, (nxt + overflow) % base, weight) for nxt in range(base)]
            else:
                # Bits above the decomposition are the balanced sign extension.
                if chunk > half or (chunk == half and not lower_zero):
                    signs = [(1, 1.0)]
                elif chunk == half:
                    signs = [(1, 0.5), (0, 0.5)]
                else:
                    signs = [(0, 1.0)]
                successors = [(None, ((base - 1) * sign + overflow) % base, p) for sign, p in signs]
            for nxt, visible, p in successors:
                tie = (visible >> (base_log - 1)) & 1
                carry = int(residue > half or (residue == half and tie))
                digit = residue - base * carry
                second += probability * p * digit * digit
                if nxt is not None:
                    key = (overflow + carry, nxt, lower_zero and chunk == 0)
                    following[key] = following.get(key, 0.0) + probability * p
        moments.append(second)
        state = following
    return moments


def key_error_variance(dimension, base_log, levels, normalized_sigma, q_bits=64):
    """Key-switching-key noise variance for uniform input mask coefficients.

    Uses the exact digit second moments of the TFHE-rs signed decomposer
    (see signed_digit_second_moments), not the independent-digit value
    (B^2+2)/12. The Gaussian reporting basis is unchanged.
    """
    if base_log <= 0 or levels <= 0 or base_log * levels >= q_bits:
        raise ValueError("invalid signed decomposition")
    if not math.isfinite(normalized_sigma) or normalized_sigma < 0:
        raise ValueError("nonnegative finite Gaussian sigma required")
    second = math.fsum(signed_digit_second_moments(base_log, levels, q_bits))
    return dimension * second * (normalized_sigma * 2**q_bits)**2


def screen(width, base_log=2, levels=7, delta_log2=52, key_sigma=0.0):
    if not isinstance(width, int) or width < 2 or width % 2:
        raise ValueError("positive even integer width required")
    mean, rounding = rounding_moments(2048, base_log * levels)
    key = key_error_variance(2048, base_log, levels, key_sigma)
    variance = rounding + key
    delta = 2.0**delta_log2
    per_output = log2_gaussian_outside(delta / 2, mean, variance)
    union = math.log2(width // 2) + per_output
    return {"width": width, "ks_base_log": base_log, "ks_levels": levels,
            "delta_log2": delta_log2, "key_sigma": key_sigma,
            "rounding_variance": rounding, "key_variance": key,
            "sigma_in_message_steps": math.sqrt(variance) / delta,
            "event_log2_gaussian": per_output,
            "output_union_raw_log2": union,
            "output_union_log2_capped": min(0, union),
            "rounding_only_target_pass": key_sigma == 0 and union < -128,
            "whole_multiplier_approved": False}
