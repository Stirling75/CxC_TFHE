#!/usr/bin/env python3
"""Failure estimator for the CBS-CMUX single-multiplication circuit.

The estimator follows the actual decision boundaries of the evaluator.  A
selector-lift blind rotation and a normalizer blind rotation must select the
correct accumulator cell; CBS conversion and CMUX evaluation do not decode an
intermediate plaintext and their noise is propagated to the next decision
boundary instead.  The variance recurrences come from Refined TFHE Analysis,
RevHomTrace, and TFHE-rs 1.6.1, while the term schedule mirrors the released
height-two normalizer.

  * refined-bit CBS lift for two selector bits per radix digit,
    with RevHomTrace trace variance by default,
  * 4-CMUX diagonal product cells,
  * 8-CMUX folded off-diagonal product cells,
  * 8-CMUX 4-bit-by-4-bit chunk product cells,
  * bounded-term chunk normalization with an interleaved digit/carry ManyLUT.

All variances are represented in the integer torus domain q = 2^64.  The
reported Gaussian tail probabilities remain model-based estimates; finite
noise measurements are validation data and are not used as a proof of a
2^-128 tail.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Iterable

Q = float(2**64)
AUTO_FFT_TYPE = "vanilla"
AUTO_FFT_COEFFICIENT_BOUND = Q
MIN_VALIDATED_AUTO_LEVEL = 3


@dataclass(frozen=True)
class Profile:
    name: str
    lwe_n: int
    glwe_k: int
    poly_n: int
    lwe_var: float  # normalized torus variance
    glwe_var: float  # normalized torus variance
    pbs_base_log: int
    pbs_level: int
    cbs_lift_pbs_base_log: int
    cbs_lift_pbs_level: int
    ks_base_log: int
    ks_level: int
    cbs_lift_ks_base_log: int
    cbs_lift_ks_level: int
    auto_base_log: int
    auto_level: int
    ss_base_log: int
    ss_level: int
    cbs_base_log: int
    cbs_level: int


def validate_auto_backend(profile: Profile) -> None:
    """Fail closed outside the validated RevHomTrace AUTO backend envelope.

    The Rust artifact instantiates ``FftType::Vanilla`` for every automorphism
    key.  Level-two AUTO decompositions produced immediate encrypted selector
    mismatches for both 16x2 and 17x2 even though the former remains barely
    below the target after correcting the Fourier variance term.  Until that
    residual model gap is closed, the estimator must not approve level < 3.
    """
    if profile.auto_level < MIN_VALIDATED_AUTO_LEVEL:
        raise ValueError(
            "RevHomTrace AUTO level_count < "
            f"{MIN_VALIDATED_AUTO_LEVEL} is outside the validated "
            f"{AUTO_FFT_TYPE} FFT envelope; got "
            f"{profile.auto_base_log}x{profile.auto_level}"
        )


@dataclass
class Term:
    bound: int
    var: float
    kind: str
    refreshed: bool = False
    # Source-aware decomposition (E_tau = sum_s a_{tau,s} xi_s + eta_tau):
    # `private_var` is Var(eta_tau); `shared` maps a shared-source key to the
    # term's amplitude bound a_{tau,s} * sqrt(Var(xi_s)).  Populated for the
    # local-product contracts of the paper; None elsewhere.
    private_var: float = 0.0
    shared: dict[tuple, float] | None = None
    # Standard-deviation amplitude used only by the correlation stress model.
    # Unlike ``var``, this field may add the primitive noise increments along
    # one CMux path before squaring.  Keeping it explicit prevents the stress
    # calculation from silently reverting to the independent chain variance.
    worst_amp: float | None = None


def term_corr_amp(term: Term) -> float:
    """Correlation-stress amplitude of one encrypted term.

    Product terms carry an explicit path-level amplitude.  Refreshed terms
    are treated as atomic PBS outputs, so their fallback amplitude is the
    square root of the ordinary output variance.  This diagnostic therefore
    stresses correlation between CMux increments and between terms entering
    one decision event; it does not claim a distribution-free expansion of
    every internal noise source of PBS/CBS.
    """
    if term.worst_amp is not None:
        return term.worst_amp
    return math.sqrt(term.var)


def term_var_source_aware(term: Term) -> float:
    """Variance of one unrefreshed term under its source decomposition."""
    if term.refreshed:
        return term.var
    if term.shared is None:
        raise ValueError(f"term {term.kind} lacks a source decomposition")
    return term.private_var + sum(amp * amp for amp in term.shared.values())


def chunk_input_var_source_aware(chunk: list[Term]) -> float:
    """Source-aware chunk variance
        V_T = sum_s (sum_tau a_{tau,s})^2 V(xi_s) + sum_tau V(eta_tau).

    Amplitudes of the same shared source add across the members of one PBS
    input (upper-bounding the covariance by Cauchy-Schwarz for every pair
    that reuses that source), while private components and fresh PBS outputs
    add independently.  Each unrefreshed term's shared amplitudes already
    carry the grouped-refresh worst case (the two bits of one radix digit are
    treated as fully correlated).
    """
    private = 0.0
    shared_amplitudes: dict[tuple, float] = {}
    for term in chunk:
        if term.refreshed:
            private += term.var
            continue
        if term.shared is None:
            raise ValueError(f"term {term.kind} lacks a source decomposition")
        private += term.private_var
        for key, amplitude in term.shared.items():
            shared_amplitudes[key] = shared_amplitudes.get(key, 0.0) + amplitude
    return private + sum(amp * amp for amp in shared_amplitudes.values())


def tfhe_final_add_event_multipliers(
    num_blocks: int, grouping_size: int = 4
) -> tuple[list[float], list[float], int, int]:
    """Model the PBS-input variances of TFHE-rs' clean radix addition.

    Every returned multiplier ``a`` denotes a PBS input whose pre-key-switch
    variance is upper-bounded by ``a * V_PBS``, where ``V_PBS`` is the noise
    variance of one clean input block.  The two lists correspond to the
    sequential and parallel carry-propagation paths selected by TFHE-rs 1.6.1.

    The sequential path evaluates separate message and carry LUTs on the same
    block sum.  Their input-cell selection event is shared, so it contributes
    one event per block, although it still performs two PBS calls.  The
    parallel path consists of a two-output PBSManyLUT layer, the intra-group
    carry-state LUTs, an optional inter-group resolution, and the final cleanup
    LUTs.  A Hillis--Steele resolution packs two radix-4 states as ``4*x+y``;
    its input variance is therefore at most ``(4^2+1) V_PBS``.
    """
    if num_blocks < 1:
        return [], [], 0, 0
    if grouping_size < 2:
        raise ValueError("grouping_size must be at least two")

    sequential = [2.0] + [3.0] * (num_blocks - 1)
    sequential_pbs_calls = 2 * num_blocks

    # First layer: lhs_i + rhs_i enters one two-output PBSManyLUT.
    parallel = [2.0] * num_blocks

    # Second layer: carry states are summed within groups before a PBS.  The
    # cumulative sums contain one through `grouping_size` refreshed blocks.
    parallel.extend(
        float((index % grouping_size) + 1) for index in range(num_blocks - 1)
    )

    # One carry-state block is retained at the end of every completed group
    # after the first.  TFHE-rs resolves these blocks either sequentially or by
    # a Hillis--Steele scan, using the same criterion as add.rs.
    carries_to_resolve = (num_blocks - 1) // grouping_size
    sequential_depth = max(0, carries_to_resolve - 1) // (grouping_size - 1)
    hillis_steele_depth = (
        0 if carries_to_resolve == 0 else (carries_to_resolve - 1).bit_length()
    )
    if sequential_depth <= hillis_steele_depth:
        remaining = max(0, carries_to_resolve - 1)
        while remaining:
            chunk_len = min(grouping_size - 1, remaining)
            # The first state is added to the previously resolved carry; later
            # states extend that cumulative sum.
            parallel.extend(float(2 + index) for index in range(chunk_len))
            remaining -= chunk_len
    else:
        space = 1
        while space < carries_to_resolve:
            parallel.extend([17.0] * (carries_to_resolve - space))
            space *= 2

    # Final cleanup adds a shifted block, a propagation simulator, and a
    # resolved carry.  Three refreshed-output variances upper-bound every block.
    parallel.extend([3.0] * num_blocks)
    parallel_pbs_calls = len(parallel)
    return sequential, parallel, sequential_pbs_calls, parallel_pbs_calls


def align_source_split(prim: PrimitiveVars, contract: str) -> PrimitiveVars:
    """Fold any empirical raise of the per-term product variance into the
    shared source bucket.

    When calibration lifts the per-term variance above the independent formula
    value m*(gadget+key), the excess is attributed entirely to the shared
    key-dependent component, which is the conservative choice for the
    source-aware chunk bound (shared amplitudes add, private variances do
    not).  Without calibration this is a no-op because the chain model gives
    product_var = m*(gadget+key) exactly.
    """
    chains = {
        "chunk4x4-tree-digits": (8, "product_chunk2x2_var"),
        "chunk8x8-direct-digits": (16, "product_chunk4x4_split_var"),
    }
    if contract not in chains:
        return prim
    slots, field = chains[contract]
    product_var = getattr(prim, field)
    independent_model = slots * (prim.cmux_gadget_var + prim.cmux_key_var)
    if product_var <= independent_model:
        return prim
    raised_key = (product_var - slots * prim.cmux_gadget_var) / slots
    return replace(prim, cmux_key_var=max(prim.cmux_key_var, raised_key))


def chunk_input_var_worst_case(chunk: list[Term]) -> float:
    """Cauchy-Schwarz worst-case variance of a chunk sum.

    Var(sum X_i) <= (sum sigma_i)^2 holds under arbitrary correlation, so
    summing every member's amplitude and squaring never underestimates,
    whatever the true joint distribution of the shared selector-lift,
    grouped-refresh, CMux, and refreshed-output noise components is.  This is
    a sensitivity analysis rather than the analytic gating model.
    """
    amp = sum(term_corr_amp(t) for t in chunk)
    return amp * amp


@dataclass
class PrimitiveVars:
    input_lwe_var: float
    input_glwe_var: float
    lift_input_var: float
    normalizer_ks_var: float
    normalizer_pbs_var: float
    cbs_lift_var: float
    cmux_ext_var: float
    product_diag_var: float
    product_fold_var: float
    product_chunk2x2_var: float
    product_chunk4x4_split_var: float
    normalizer_input_var_floor: float = 0.0
    source: str = "formula"
    # Source-aware split of one CMux external product: the accumulator
    # gadget-decomposition rounding is private to each lookup path, while the
    # key-dependent part (selector-error and FFT-representation terms) is
    # shared between every lookup that reuses the same GGSW selector.
    cmux_gadget_var: float = 0.0
    cmux_key_var: float = 0.0


@dataclass
class ScheduleStats:
    width_bits: int
    digits: int
    mode: str
    row_bits: int
    pbs: int
    rounds: int
    max_chunks: int
    max_column_height: int
    max_column_bound: int
    lift_event_count: int
    lift_threshold_log2: float
    max_lift_log2_pfail: float
    lift_union_log2_pfail: float
    product_threshold_bits: int
    product_padding_bits: int
    product_threshold_log2: float
    max_product_log2_var: float
    max_product_log2_pfail: float
    product_event_count: int
    product_union_log2_pfail: float
    normalizer_cap_bits: int
    normalizer_threshold_bits: int
    normalizer_padding_bits: int
    normalizer_threshold_log2: float
    max_chunk_input_log2_var: float
    max_chunk_input_log2_pfail: float
    final_threshold_bits: int
    final_padding_bits: int
    final_threshold_log2: float
    max_final_log2_var: float
    max_final_log2_pfail: float
    normalizer_event_count: int
    normalizer_union_log2_pfail: float
    final_event_count: int
    final_union_log2_pfail: float
    post_product_union_log2_pfail: float
    union_log2_pfail: float


def log2_or_neginf(x: float) -> float:
    if x <= 0.0:
        return float("-inf")
    return math.log2(x)


def log2_sum_exp(log2_values: Iterable[float]) -> float:
    values = [x for x in log2_values if x != float("-inf")]
    if not values:
        return float("-inf")
    pivot = max(values)
    if pivot == float("inf"):
        return float("inf")
    return pivot + math.log2(sum(2.0 ** (x - pivot) for x in values))


def log2_erfc(x: float) -> float:
    if x <= 0.0:
        return 0.0
    direct = math.erfc(x)
    if direct > 0.0:
        return math.log2(direct)

    # erfc(x) underflows in double precision for large x.  Use the first terms
    # of the asymptotic expansion
    #   erfc(x) = exp(-x^2)/(x sqrt(pi)) * (1 - 1/(2x^2) + 3/(4x^4) - ...)
    # for log-domain screening.  This is only used once the direct value is
    # already far below representable probability.
    inv_x2 = 1.0 / (x * x)
    correction = 1.0 - 0.5 * inv_x2 + 0.75 * inv_x2 * inv_x2
    if correction <= 0.0:
        correction = 1.0
    return (
        -x * x
        - math.log(x)
        - 0.5 * math.log(math.pi)
        + math.log(correction)
    ) / math.log(2.0)


def two_sided_log2_pfail(theta: float, mu: float, var: float) -> float:
    """log2 Pr[|X + mu| >= theta] for X ~ N(0, var).

    Equals log2 erfc(theta / sqrt(2 var)) when mu = 0; with a nonzero known
    mean the two one-sided halves are combined exactly:
        1/2 erfc((theta - mu)/sqrt(2V)) + 1/2 erfc((theta + mu)/sqrt(2V)).
    """
    if var <= 0.0:
        return float("-inf") if abs(mu) < theta else 0.0
    scale = math.sqrt(2.0 * var)
    lower = log2_erfc((theta - mu) / scale) - 1.0
    upper = log2_erfc((theta + mu) / scale) - 1.0
    return log2_sum_exp([lower, upper])


def tail_log2_pfail(
    threshold: float,
    var_int: float,
    tail_model: str,
    subgaussian_proxy_scale: float,
) -> float:
    if var_int <= 0.0:
        return float("-inf")
    if threshold <= 0.0:
        return 0.0

    if tail_model == "gaussian":
        gamma = threshold / math.sqrt(var_int)
        return log2_erfc(gamma / math.sqrt(2.0))

    if tail_model in ("tuniform-hoeffding", "tuniform-proxy", "subgaussian"):
        # Conservative two-sided bounded-tail proxy:
        #   Pr[|X| >= t] <= 2 exp(-t^2 / (2 kappa V)).
        # For a centered uniform-like bounded term, kappa = 3 corresponds to
        # replacing support radius squared by about 3 times its variance.
        # This is an engineering screen, not a complete Hoeffding theorem for
        # the full composed circuit, because the script does not track the
        # support of every independent summand.
        denom = 2.0 * subgaussian_proxy_scale * var_int
        log2_bound = 1.0 - (threshold * threshold) / (denom * math.log(2.0))
        return min(0.0, log2_bound)

    raise ValueError(f"unknown tail_model={tail_model}")


def t_uniform_normalized_var(bound_log2: int, modulus: float = Q) -> float:
    # TFHE-rs TUniform variance:
    # ((2^{2b+1}+1)/6) * modulus^{-2}.
    return ((2.0 ** (2 * bound_log2 + 1) + 1.0) / 6.0) / (modulus * modulus)


def get_var_pbs(N: int, k: int, n: int, q: float, var_glwe: float, base_log: int, level: int) -> float:
    B = 2.0**base_log
    B_2l = B ** (2 * level)
    var_gadget = 0.0
    var_gadget += n * (q * q - B_2l) / (24.0 * B_2l) * (1.0 + k * N / 2.0)
    var_gadget += (n * k * N) / 32.0
    var_gadget += (n / 16.0) * (1.0 - k * N / 2.0) ** 2
    var_key = n * level * (k + 1) * N * ((B * B + 2.0) / 12.0) * (var_glwe * q * q)
    return var_gadget + var_key


def get_var_fft_pbs(N: int, k: int, n: int, base_log: int, level: int) -> float:
    B = 2.0**base_log
    return n * (2.0 ** (22 - 2.6)) * level * B * B * N * N * (k + 1)


def get_var_ext_prod(N: int, k: int, q: float, var_in_int: float, base_log: int, level: int) -> float:
    B = 2.0**base_log
    B_2l = B ** (2 * level)
    gadget = (1.0 + k * N) * ((q * q - B_2l) / (24.0 * B_2l) + 1.0 / 16.0)
    inc = (k + 1) * level * N * ((B * B + 2.0) / 12.0) * var_in_int
    return gadget + inc


def get_var_fft_ext_prod(N: int, k: int, q: float, base_log: int, level: int) -> float:
    B = 2.0**base_log
    return (2.0 ** (-2 * 53 - 2.6)) * (k + 1) * level * B * B * q * q * N * N


def get_var_lwe_ks(source_lwe_dimension: int, q: float, var_lwe: float, base_log: int, level: int) -> float:
    B = 2.0**base_log
    B_2l = B ** (2 * level)
    gadget = source_lwe_dimension * ((q * q - B_2l) / (24.0 * B_2l) + 1.0 / 16.0)
    key = source_lwe_dimension * level * (var_lwe * q * q) * (B * B / 12.0 + 1.0 / 6.0)
    return gadget + key


def get_var_glwe_ks(N: int, k_src: int, q: float, var_dst: float, base_log: int, level: int) -> float:
    B = 2.0**base_log
    B_2l = B ** (2 * level)
    gadget = k_src * N * ((q * q - B_2l) / (24.0 * B_2l) + 1.0 / 16.0)
    key = k_src * N * level * (var_dst * q * q) * ((B * B + 2.0) / 12.0)
    return gadget + key


def get_var_fft_glwe_ks(N: int, k: int, base_log: int, level: int, split_base: float) -> float:
    B = 2.0**base_log
    return (2.0 ** (-2 * 53 - 2.6)) * k * level * B * B * split_base * split_base * N * N


def get_var_homtrace(N: int, k: int, q: float, var_glwe: float, base_log: int, level: int, split_base: float) -> float:
    # Original HomTrace / PreHomTrace bound used by Refined TFHE Analysis:
    #   Var_trace <= ((N^2 - 1) / 3) * Var_auto.
    # Since Var_auto itself is O(kN), this is the old cubic-noise shape.
    base = get_var_glwe_ks(N, k, q, var_glwe, base_log, level)
    fft = get_var_fft_glwe_ks(N, k, base_log, level, split_base)
    return ((N * N - 1.0) / 3.0) * (base + fft)


def get_var_modswitch_1bit(N: int, k: int) -> float:
    # TCHES 2026/1/05, Lemma 1, specialized to q -> q/2 -> q modulus
    # switching with a binary GLWE key.  The bound is in integer torus units.
    return (1.0 + k * N) / 12.0


def get_var_revhomtrace(
    N: int,
    k: int,
    q: float,
    var_glwe: float,
    base_log: int,
    level: int,
    split_base: float,
) -> float:
    # RevHomTrace bound from TCHES 2026/1/05, Theorem 4:
    #   Var(C') <= Var(C) + 4 log N * V_MS + log N * V_Auto.
    # Here we return only the additional trace variance term.
    log_n = math.log2(N)
    v_auto = get_var_glwe_ks(N, k, q, var_glwe, base_log, level)
    v_fft_auto = get_var_fft_glwe_ks(N, k, base_log, level, split_base)
    v_ms = get_var_modswitch_1bit(N, k)
    return 4.0 * log_n * v_ms + log_n * (v_auto + v_fft_auto)


def get_var_trace(
    N: int,
    k: int,
    q: float,
    var_glwe: float,
    base_log: int,
    level: int,
    split_base: float,
    trace_model: str,
) -> float:
    if trace_model == "revhomtrace":
        return get_var_revhomtrace(N, k, q, var_glwe, base_log, level, split_base)
    if trace_model == "homtrace":
        return get_var_homtrace(N, k, q, var_glwe, base_log, level, split_base)
    raise ValueError(f"unknown trace_model={trace_model}")


def get_var_scheme_switch(N: int, k: int, q: float, var_in_int: float, base_log: int, level: int) -> float:
    B = 2.0**base_log
    B_2l = B ** (2 * level)
    gadget = (1.0 + k * N) * ((q * q - B_2l) / (24.0 * B_2l) + 1.0 / 16.0) * N / 2.0
    inc = (k + 1) * level * N * ((B * B + 2.0) / 12.0) * var_in_int
    fft = get_var_fft_ext_prod(N, k, q, base_log, level)
    return gadget + inc + fft


def centered_binary_ms_decision_noise(
    lwe_n: int, q: float, target_modulus: float
) -> tuple[float, float]:
    """Variance proxy and worst-case bias for centered binary MS selection.

    Write ``D = q / target_modulus`` and let

        r_i = D * round(a_i / D) - a_i

    be the rounding residual of mask coefficient ``a_i``.  The body
    correction used by TFHE-rs changes the perturbation presented to the LUT
    boundary from ``-sum_i s_i r_i`` to

        A = sum_i (1/2 - s_i) r_i + zeta,

    where ``s_i`` is binary and the integer-halving correction satisfies
    ``|zeta| <= 1/2`` in the native integer torus.  For a uniform native mask,
    ``Var(r_i) = (D^2 - 1) / 12``.  Thus the centered random mask component has
    variance ``n(D^2 - 1)/48``, independently of the secret-key Hamming
    weight.  The remaining mean is at most ``n/4`` in absolute value because
    the discrete round-to-nearest residual has mean 1/2.

    If the binary secret is also averaged as independent Bernoulli(1/2), then
    ``E[(1/2-s_i)^2 r_i^2] = (D^2+2)/48`` and TFHE-rs 1.6.1 obtains

        n/24 + n D^2/48.

    We retain that random-key/uniform-mask second moment as the variance proxy
    and account separately for the worst fixed-key mean and the halving
    remainder through ``bias_bound``.  TFHE-rs labels the formula heuristic
    because an evaluated or key-switched ciphertext is not proved here to have
    iid uniform mask residues.
    This function models the perturbation that can cross a repeated-LUT cell
    boundary; the final body quantization is not added again as an independent
    Gaussian term.
    """
    if lwe_n < 0:
        raise ValueError("lwe_n must be non-negative")
    if q <= 0.0 or target_modulus <= 0.0 or target_modulus > q:
        raise ValueError("expected 0 < target_modulus <= q")

    step = q / target_modulus
    exact_mask_variance = lwe_n * (step * step - 1.0) / 48.0
    tfhe_rs_variance_proxy = lwe_n / 24.0 + lwe_n * step * step / 48.0
    variance_proxy = max(0.0, exact_mask_variance, tfhe_rs_variance_proxy)
    bias_bound = lwe_n / 4.0 + 0.5
    return variance_proxy, bias_bound


def pbs_modulus_switch_target(N: int, theta: int) -> float:
    """Return the index modulus used by the corresponding blind rotation.

    Ordinary TFHE-rs PBS uses ``theta = 0`` and switches to ``2N``.  The
    grouped selector-lift backend drops ``theta = log_lut_count`` low index
    bits inside ``fast_pbs_modulus_switch_u64`` and therefore rounds directly
    to ``2N / 2^theta``.  This target controls the rounding residual; the
    independently supplied ``delta_in`` controls the admissible LUT-cell
    margin.
    """
    if N <= 0:
        raise ValueError("N must be positive")
    if theta < 0 or (1 << theta) > 2 * N:
        raise ValueError("theta must satisfy 0 <= 2^theta <= 2N")
    return (2.0 * N) / (2.0**theta)


def pbs_input_log2_pfail(
    lwe_n: int,
    q: float,
    N: int,
    theta: int,
    delta_in: float,
    var_in_int: float,
    tail_model: str = "gaussian",
    subgaussian_proxy_scale: float = 3.0,
    centered_binary_ms: bool = False,
) -> float:
    # Refined var.sage get_fp_pbs for ordinary modulus switching.  For the
    # centered binary path, model the centered mask-rounding perturbation that
    # is actually compared with the repeated-LUT cell boundary.  The body
    # quantization maps this perturbation to an integer blind-rotation index;
    # it is not a second independent error source at that same boundary.
    ms_target_modulus = pbs_modulus_switch_target(N, theta)
    if centered_binary_ms:
        additive, bias_bound = centered_binary_ms_decision_noise(
            lwe_n, q, ms_target_modulus
        )
    else:
        additive = (
            q * q / (12.0 * ms_target_modulus * ms_target_modulus)
            - 1.0 / 12.0
            + lwe_n
            * q
            * q
            / (24.0 * ms_target_modulus * ms_target_modulus)
            + lwe_n / 48.0
        )
        bias_bound = 0.0
    denom = var_in_int + additive
    if denom <= 0.0:
        return float("-inf")
    threshold = delta_in / 2.0 - bias_bound
    if threshold <= 0.0:
        return 0.0
    return tail_log2_pfail(
        threshold,
        denom,
        tail_model,
        subgaussian_proxy_scale,
    )


def centered_decode_log2_pfail(
    bits: int,
    var_int: float,
    tail_model: str = "gaussian",
    subgaussian_proxy_scale: float = 3.0,
    padding_bits: int = 0,
) -> float:
    half_delta = 2.0 ** decision_threshold_log2(bits, padding_bits)
    return tail_log2_pfail(
        half_delta,
        var_int,
        tail_model,
        subgaussian_proxy_scale,
    )


def ceil_log2_domain_size(size: int) -> int:
    if size <= 1:
        return 0
    return (size - 1).bit_length()


def decision_threshold_log2(effective_bits: int, padding_bits: int = 0) -> int:
    if effective_bits < 0 or padding_bits < 0:
        raise ValueError("effective_bits and padding_bits must be non-negative")
    # Rust encodes a b-bit row value with encoding_delta(b) = 2^(63-b).
    # Centered decoding is correct while |phase error| < encoding_delta/2.
    return 64 - effective_bits - padding_bits - 2


def delta_from_bits(effective_bits: int, padding_bits: int = 0) -> float:
    if effective_bits < 0 or padding_bits < 0:
        raise ValueError("effective_bits and padding_bits must be non-negative")
    return 2.0 ** (63 - effective_bits - padding_bits)


def primitive_vars(
    profile: Profile,
    row_bits: int,
    cbs_extract_bits: int,
    cmux_noise_model: str,
    cbs_lift_model: str,
    trace_model: str,
    trace_scale_model: str,
    cbs_lift_log2_var_floor: float | None = None,
) -> PrimitiveVars:
    validate_auto_backend(profile)
    N = profile.poly_n
    k = profile.glwe_k
    n = profile.lwe_n
    big_lwe_dimension = k * N
    input_lwe_var = profile.lwe_var * Q * Q
    input_glwe_var = profile.glwe_var * Q * Q

    normalizer_ks = get_var_lwe_ks(
        big_lwe_dimension,
        Q,
        profile.lwe_var,
        profile.ks_base_log,
        profile.ks_level,
    )
    normalizer_pbs = get_var_pbs(
        N, k, n, Q, profile.glwe_var, profile.pbs_base_log, profile.pbs_level
    ) + get_var_fft_pbs(N, k, n, profile.pbs_base_log, profile.pbs_level)
    cbs_lift_ks = get_var_lwe_ks(
        big_lwe_dimension,
        Q,
        profile.lwe_var,
        profile.cbs_lift_ks_base_log,
        profile.cbs_lift_ks_level,
    )
    # The multiplier accepts a clean m2c2 radix block produced by PBS.  Before
    # selector-lift key switching, its public encoding is rescaled from q/32 to
    # q/4, multiplying the input error variance by 8^2.
    lift_input_var = 64.0 * normalizer_pbs + cbs_lift_ks
    cbs_lift_pbs = get_var_pbs(
        N,
        k,
        n,
        Q,
        profile.glwe_var,
        profile.cbs_lift_pbs_base_log,
        profile.cbs_lift_pbs_level,
    ) + get_var_fft_pbs(
        N,
        k,
        n,
        profile.cbs_lift_pbs_base_log,
        profile.cbs_lift_pbs_level,
    )

    # CBS-lift composition.  RevHomTrace changes the trace recurrence, but not
    # the scheme-switching geometry around it.  The trace-added polynomial
    # error is multiplied by a binary GLWE secret while forming a GGSW mask
    # row, contributing (N/2) * V_trace per coefficient as in the Refined CBS
    # composition.  The PBS error was first sample-extracted and embedded in
    # the constant coefficient, so it is not multiplied by N/2.
    # The artifact generates automorphism keys with FftType::Vanilla.  Its
    # Fourier conversion keeps the full 64-bit torus coefficient (num_split=1,
    # split_base_log=64), so the FFT error recurrence must use b_fft=q.  The
    # former 2^35 value modeled a split representation that the Rust path does
    # not instantiate and understated this term by (2^64/2^35)^2 = 2^58.
    auto_fft_coefficient_bound = AUTO_FFT_COEFFICIENT_BOUND
    v_pbs_lift = cbs_lift_pbs
    v_trace = get_var_trace(
        N,
        k,
        Q,
        profile.glwe_var,
        profile.auto_base_log,
        profile.auto_level,
        auto_fft_coefficient_bound,
        trace_model,
    )
    if trace_scale_model == "refined-cbs":
        trace_factor = N / 2.0
    elif trace_scale_model == "direct":
        raise ValueError(
            "trace_scale_model=direct omits the N/2 scheme-switch amplification "
            "and is retained only in historical result files"
        )
    else:
        raise ValueError(f"unknown trace_scale_model={trace_scale_model}")

    if cbs_lift_model == "sage":
        v_ss = get_var_scheme_switch(
            N,
            k,
            Q,
            input_glwe_var,
            profile.ss_base_log,
            profile.ss_level,
        )
        v_cbs = v_pbs_lift + trace_factor * v_trace + v_ss
    elif cbs_lift_model == "engineering":
        v_ss = get_var_scheme_switch(
            N,
            k,
            Q,
            v_pbs_lift + trace_factor * v_trace,
            profile.ss_base_log,
            profile.ss_level,
        )
        # One modulus-switching rounding unit per extracted bit is a rough guard.
        v_ms_guard = (Q * Q) / (12.0 * (2.0 * N) ** 2)
        v_cbs = (
            v_pbs_lift
            + (cbs_extract_bits**2) * v_ms_guard
            + trace_factor * v_trace
            + v_ss
        )
    else:
        raise ValueError(f"unknown cbs_lift_model={cbs_lift_model}")

    if cbs_lift_log2_var_floor is not None:
        # Conservative floor on the effective GGSW selector-error variance,
        # fitted from encrypted noise measurements.  The measured per-term
        # product noise exceeds this formula's v_cbs-driven prediction by a
        # component that scales with level*base^2 of the CBS gadget; raising
        # v_cbs to the fitted floor makes the formula upper-bound every
        # measured configuration (see analyze_noise_probe.py).
        v_cbs = max(v_cbs, 2.0**cbs_lift_log2_var_floor)

    v_ext_gadget = get_var_ext_prod(N, k, Q, 0.0, profile.cbs_base_log, profile.cbs_level)
    v_ext = get_var_ext_prod(
        N, k, Q, v_cbs, profile.cbs_base_log, profile.cbs_level
    ) + get_var_fft_ext_prod(N, k, Q, profile.cbs_base_log, profile.cbs_level)
    v_ext_key = v_ext - v_ext_gadget

    def chain(rounds: int) -> float:
        v = 0.0
        for _ in range(rounds):
            if cmux_noise_model == "guarded":
                v = 2.0 * v + v_ext
            elif cmux_noise_model == "linear":
                v = v + v_ext
            else:
                raise ValueError(f"unknown cmux_noise_model={cmux_noise_model}")
        return v

    return PrimitiveVars(
        input_lwe_var=input_lwe_var,
        input_glwe_var=input_glwe_var,
        lift_input_var=lift_input_var,
        normalizer_ks_var=normalizer_ks,
        normalizer_pbs_var=normalizer_pbs,
        cbs_lift_var=v_cbs,
        cmux_ext_var=v_ext,
        product_diag_var=chain(4),
        product_fold_var=chain(8),
        product_chunk2x2_var=chain(8),
        product_chunk4x4_split_var=chain(16),
        cmux_gadget_var=v_ext_gadget,
        cmux_key_var=v_ext_key,
    )


def load_noise_probe_variances(path: Path) -> dict[str, dict[str, float]]:
    by_stage: dict[str, list[float]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            stage = row.get("stage", "")
            if not stage:
                continue
            try:
                residual = float(row["residual_torus_signed"])
            except (KeyError, ValueError):
                continue
            by_stage.setdefault(stage, []).append(residual)

    out: dict[str, dict[str, float]] = {}
    for stage, values in by_stage.items():
        if len(values) == 1:
            var = 0.0
        else:
            var = statistics.pvariance(values)
        out[stage] = {
            "count": float(len(values)),
            "var": var,
            "max_abs": max(abs(v) for v in values) if values else 0.0,
        }
    return out


def apply_calibration(
    prim: PrimitiveVars,
    noise_csv: Path | None,
    product_log2_var: float | None,
    normalizer_log2_var: float | None,
    std_factor: float,
) -> PrimitiveVars:
    if noise_csv is None and product_log2_var is None and normalizer_log2_var is None:
        return prim

    calibrated = replace(prim)
    source_parts = []
    if product_log2_var is not None:
        product_var = 2.0**product_log2_var
        calibrated.product_diag_var = product_var
        calibrated.product_fold_var = product_var
        calibrated.product_chunk2x2_var = product_var
        calibrated.product_chunk4x4_split_var = product_var
        source_parts.append(f"product-log2={product_log2_var:.2f}")

    if normalizer_log2_var is not None:
        norm_var = 2.0**normalizer_log2_var
        calibrated.normalizer_pbs_var = norm_var
        source_parts.append(f"norm-log2={normalizer_log2_var:.2f}")

    if noise_csv is not None:
        stages = load_noise_probe_variances(noise_csv)
        scale = std_factor * std_factor
        if "term" in stages:
            # Empirical values may only RAISE the analytic model, never lower
            # it: a finite-sample point estimate below the formula must not be
            # used to claim a smaller failure probability.  The measurement
            # acts as a floor that exposes under-modeled noise; the formula
            # remains the baseline.
            measured = (
                max(stages["term"]["var"], stages["term"]["max_abs"] ** 2 / 36.0) * scale
            )
            calibrated.product_diag_var = max(prim.product_diag_var, measured)
            calibrated.product_fold_var = max(prim.product_fold_var, measured)
            calibrated.product_chunk2x2_var = max(prim.product_chunk2x2_var, measured)
            calibrated.product_chunk4x4_split_var = max(
                prim.product_chunk4x4_split_var, measured
            )
            source_parts.append(
                f"term:{noise_csv.name}:n={int(stages['term']['count'])}:stdx{std_factor:g}:max-with-formula"
            )
        # `digit` and `carry` are fresh outputs of the normalizer PBS/ManyLUT.
        # `final_sum` is a later LWE sum of already-normalized terms, so using
        # it as a PBS-output calibration floor would conflate two stages.
        norm_candidates = [
            stages[name]["var"]
            for name in ("digit", "carry")
            if name in stages and stages[name]["var"] > 0.0
        ]
        if norm_candidates:
            calibrated.normalizer_pbs_var = max(
                prim.normalizer_pbs_var, max(norm_candidates) * scale
            )
            source_parts.append(f"norm:{noise_csv.name}:stdx{std_factor:g}")
        if "chunk_sum" in stages:
            v = max(
                stages["chunk_sum"]["var"],
                stages["chunk_sum"]["max_abs"] ** 2 / 36.0,
            ) * scale
            calibrated.normalizer_input_var_floor = v
            source_parts.append(
                f"chunk-sum:{noise_csv.name}:n={int(stages['chunk_sum']['count'])}:stdx{std_factor:g}"
            )
    calibrated.source = "+".join(source_parts) if source_parts else "formula"
    return calibrated


def normalize_contract(contract: str | None, folded: bool | None = None) -> str:
    if contract:
        aliases = {
            "folded": "folded-lh",
            "folded8": "folded-lh",
            "folded-lh": "folded-lh",
            "raw": "raw-lh",
            "raw-lh": "raw-lh",
            "chunk2x2": "chunk2x2-digits",
            "chunk2x2-digits": "chunk2x2-digits",
            "chunk2x2-base16": "chunk2x2-base16-first",
            "base16-first": "chunk2x2-base16-first",
            "chunk2x2-base16-first": "chunk2x2-base16-first",
            "chunk4x4": "chunk4x4-tree-digits",
            "chunk4x4-tree": "chunk4x4-tree-digits",
            "chunk4x4-tree-digits": "chunk4x4-tree-digits",
            "chunk4x4-split": "chunk8x8-direct-digits",
            "chunk4x4-split-digits": "chunk8x8-direct-digits",
            "chunk8x8": "chunk8x8-direct-digits",
            "chunk8x8-direct": "chunk8x8-direct-digits",
            "chunk8x8-direct-digits": "chunk8x8-direct-digits",
        }
        try:
            return aliases[contract]
        except KeyError as exc:
            raise ValueError(f"unknown product contract={contract}") from exc
    return "folded-lh" if folded is not False else "raw-lh"


def product_contract_stats(digits: int, contract: str, chunk4x4_split_prefix_bits: int = 8) -> dict[str, int]:
    contract = normalize_contract(contract)
    stats = {
        "selector_bits_per_cell": 0,
        "outputs_per_cell": 0,
        "product_cells": 0,
        "product_terms": 0,
        "cmux_count": 0,
    }
    if contract == "folded-lh":
        stats["selector_bits_per_cell"] = 8
        stats["outputs_per_cell"] = 2
        for i in range(digits):
            for j in range(i, digits):
                if i + j >= digits:
                    continue
                stats["product_cells"] += 1
                stats["product_terms"] += 2
                stats["cmux_count"] += 4 if i == j else 8
        return stats
    if contract == "raw-lh":
        stats["selector_bits_per_cell"] = 4
        stats["outputs_per_cell"] = 2
        for i in range(digits):
            for j in range(digits):
                if i + j >= digits:
                    continue
                stats["product_cells"] += 1
                stats["product_terms"] += 2
                stats["cmux_count"] += 4
        return stats
    if contract == "chunk2x2-digits":
        if digits % 2 != 0:
            raise ValueError("chunk2x2-digits requires an even number of radix-4 digits")
        stats["selector_bits_per_cell"] = 8
        stats["outputs_per_cell"] = 4
        chunk_count = digits // 2
        for a_chunk in range(chunk_count):
            for b_chunk in range(chunk_count):
                if 2 * (a_chunk + b_chunk) >= digits:
                    continue
                stats["product_cells"] += 1
                stats["product_terms"] += 4
                stats["cmux_count"] += 8
        return stats
    if contract == "chunk2x2-base16-first":
        if digits % 2 != 0:
            raise ValueError("chunk2x2-base16-first requires an even number of radix-4 digits")
        stats["selector_bits_per_cell"] = 8
        stats["outputs_per_cell"] = 2
        chunk_count = digits // 2
        for a_chunk in range(chunk_count):
            for b_chunk in range(chunk_count):
                if a_chunk + b_chunk >= chunk_count:
                    continue
                stats["product_cells"] += 1
                stats["product_terms"] += 2
                stats["cmux_count"] += 8
        return stats
    if contract == "chunk4x4-tree-digits":
        if digits % 4 != 0:
            raise ValueError("chunk4x4-tree-digits requires a multiple of four radix-4 digits")
        stats["selector_bits_per_cell"] = 8
        stats["outputs_per_cell"] = 4
        chunk_count = digits // 4
        for a_chunk in range(chunk_count):
            for b_chunk in range(chunk_count):
                base_diagonal = 4 * (a_chunk + b_chunk)
                if base_diagonal >= digits:
                    continue
                for sub_offset in (0, 2, 2, 4):
                    sub_diagonal = base_diagonal + sub_offset
                    if sub_diagonal < digits:
                        stats["product_cells"] += 1
                        # A final lookup may straddle the W-bit truncation
                        # boundary.  It is evaluated, but only the retained
                        # coefficients are routed to the normalizer.
                        stats["product_terms"] += min(4, digits - sub_diagonal)
                        stats["cmux_count"] += 8
        return stats
    if contract == "chunk8x8-direct-digits":
        if digits % 4 != 0:
            raise ValueError("chunk8x8-direct-digits requires a multiple of four radix-4 digits")
        stats["selector_bits_per_cell"] = 16
        stats["outputs_per_cell"] = 8
        prefix_bits = chunk4x4_split_prefix_bits
        if prefix_bits < 0 or prefix_bits >= 16:
            raise ValueError("chunk8x8-direct-digits requires 0 <= prefix_bits < 16")
        cmux_per_cell = ((1 << prefix_bits) - 1) + (16 - prefix_bits)
        chunk_count = digits // 4
        for a_chunk in range(chunk_count):
            for b_chunk in range(chunk_count):
                base_diagonal = 4 * (a_chunk + b_chunk)
                if base_diagonal >= digits:
                    continue
                stats["product_cells"] += 1
                stats["product_terms"] += min(8, digits - base_diagonal)
                stats["cmux_count"] += cmux_per_cell
        return stats
    raise ValueError(f"unknown product contract={contract}")


def product_terms(
    digits: int,
    prim: PrimitiveVars,
    folded: bool | None = None,
    contract: str | None = None,
    identify_operands: bool = False,
) -> list[list[Term]]:
    contract = normalize_contract(contract, folded)
    columns: list[list[Term]] = [[] for _ in range(2 * digits)]

    def selector_sources(lhs_digits, rhs_digits, amplitude):
        """Shared-source amplitudes of one lookup path.

        With ``identify_operands`` the second operand reuses the first
        operand's selector sources, modelling the squaring case Y = X: the
        same GGSW selectors then appear on both sides of every cell, and on
        diagonal cells the same digit source is traversed twice, so its
        amplitude accumulates instead of being counted once per side.
        """
        shared: dict[tuple, float] = {}
        for digit in lhs_digits:
            key = ("X", digit)
            shared[key] = shared.get(key, 0.0) + amplitude
        rhs_side = "X" if identify_operands else "Y"
        for digit in rhs_digits:
            key = (rhs_side, digit)
            shared[key] = shared.get(key, 0.0) + amplitude
        return shared

    def path_worst_amp(rounds: int, ordinary_var: float) -> float:
        """Amplitude-additive stress bound for one selected CMux path.

        The ordinary model adds the gadget and key-dependent variance of each
        external product.  The stress model instead permits both components,
        and all selected CMux levels, to be perfectly correlated.  An
        empirical/calibrated variance floor can only increase this bound.
        """
        primitive_amp = rounds * (
            math.sqrt(prim.cmux_gadget_var) + math.sqrt(prim.cmux_key_var)
        )
        return max(math.sqrt(ordinary_var), primitive_amp)

    if contract == "folded-lh":
        for i in range(digits):
            for j in range(i, digits):
                q = i + j
                if q >= digits:
                    continue
                if i == j:
                    amp = path_worst_amp(4, prim.product_diag_var)
                    columns[q].append(
                        Term(3, prim.product_diag_var, "diag-low", worst_amp=amp)
                    )
                    columns[q + 1].append(
                        Term(2, prim.product_diag_var, "diag-high", worst_amp=amp)
                    )
                else:
                    amp = path_worst_amp(8, prim.product_fold_var)
                    columns[q].append(
                        Term(3, prim.product_fold_var, "fold-low", worst_amp=amp)
                    )
                    columns[q + 1].append(
                        Term(4, prim.product_fold_var, "fold-high", worst_amp=amp)
                    )
    elif contract == "raw-lh":
        for i in range(digits):
            for j in range(digits):
                q = i + j
                if q >= digits:
                    continue
                amp = path_worst_amp(4, prim.product_diag_var)
                columns[q].append(
                    Term(3, prim.product_diag_var, "cell-low", worst_amp=amp)
                )
                columns[q + 1].append(
                    Term(2, prim.product_diag_var, "cell-high", worst_amp=amp)
                )
    elif contract == "chunk2x2-digits":
        if digits % 2 != 0:
            raise ValueError("chunk2x2-digits requires an even number of radix-4 digits")
        chunk_count = digits // 2
        for a_chunk in range(chunk_count):
            for b_chunk in range(chunk_count):
                base_diagonal = 2 * (a_chunk + b_chunk)
                if base_diagonal >= digits:
                    continue
                for local_digit in range(4):
                    q = base_diagonal + local_digit
                    if q < len(columns):
                        columns[q].append(
                            Term(
                                3,
                                prim.product_chunk2x2_var,
                                f"chunk2x2-d{local_digit}",
                                worst_amp=path_worst_amp(
                                    8, prim.product_chunk2x2_var
                                ),
                            )
                        )
    elif contract == "chunk2x2-base16-first":
        if digits % 2 != 0:
            raise ValueError("chunk2x2-base16-first requires an even number of radix-4 digits")
        chunk_count = digits // 2
        columns = [[] for _ in range(chunk_count)]
        for a_chunk in range(chunk_count):
            for b_chunk in range(chunk_count):
                q = a_chunk + b_chunk
                if q >= chunk_count:
                    continue
                amp = path_worst_amp(8, prim.product_chunk2x2_var)
                columns[q].append(
                    Term(15, prim.product_chunk2x2_var, "base16-lo", worst_amp=amp)
                )
                if q + 1 < chunk_count:
                    columns[q + 1].append(
                        Term(
                            14,
                            prim.product_chunk2x2_var,
                            "base16-hi",
                            worst_amp=amp,
                        )
                    )
    elif contract == "chunk4x4-tree-digits":
        if digits % 4 != 0:
            raise ValueError("chunk4x4-tree-digits requires a multiple of four radix-4 digits")
        chunk_count = digits // 4
        for a_chunk in range(chunk_count):
            for b_chunk in range(chunk_count):
                base_diagonal = 4 * (a_chunk + b_chunk)
                if base_diagonal >= digits:
                    continue
                for sub_idx, sub_offset in enumerate((0, 2, 2, 4)):
                    sub_diagonal = base_diagonal + sub_offset
                    if sub_diagonal >= digits:
                        continue
                    lhs_start = 4 * a_chunk + 2 * (sub_idx // 2)
                    rhs_start = 4 * b_chunk + 2 * (sub_idx % 2)
                    shared = selector_sources(
                        (lhs_start, lhs_start + 1),
                        (rhs_start, rhs_start + 1),
                        2.0 * math.sqrt(prim.cmux_key_var),
                    )
                    for local_digit in range(4):
                        q = sub_diagonal + local_digit
                        if q < len(columns):
                            columns[q].append(
                                Term(
                                    3,
                                    prim.product_chunk2x2_var,
                                    f"chunk4x4-sub{sub_idx}-d{local_digit}",
                                    private_var=8.0 * prim.cmux_gadget_var,
                                    shared=dict(shared),
                                    worst_amp=path_worst_amp(
                                        8, prim.product_chunk2x2_var
                                    ),
                                )
                            )
    elif contract == "chunk8x8-direct-digits":
        if digits % 4 != 0:
            raise ValueError("chunk8x8-direct-digits requires a multiple of four radix-4 digits")
        chunk_count = digits // 4
        for a_chunk in range(chunk_count):
            for b_chunk in range(chunk_count):
                base_diagonal = 4 * (a_chunk + b_chunk)
                if base_diagonal >= digits:
                    continue
                shared = selector_sources(
                    tuple(4 * a_chunk + offset for offset in range(4)),
                    tuple(4 * b_chunk + offset for offset in range(4)),
                    2.0 * math.sqrt(prim.cmux_key_var),
                )
                for local_digit in range(8):
                    q = base_diagonal + local_digit
                    if q < len(columns):
                        columns[q].append(
                            Term(
                                3,
                                prim.product_chunk4x4_split_var,
                                f"chunk8x8-direct-d{local_digit}",
                                private_var=16.0 * prim.cmux_gadget_var,
                                shared=dict(shared),
                                worst_amp=path_worst_amp(
                                    16, prim.product_chunk4x4_split_var
                                ),
                            )
                        )
    else:
        raise ValueError(f"unknown product contract={contract}")
    return columns


def exact_small_bound_groups(bounds: list[int], cap: int, state_limit: int = 200_000) -> list[list[int]] | None:
    if not bounds:
        return []
    if cap <= 0 or any(bound <= 0 or bound > 3 or bound > cap for bound in bounds):
        return None

    counts = [0, 0, 0]
    by_bound: list[list[int]] = [[], [], []]
    for idx, bound in enumerate(bounds):
        bucket = bound - 1
        counts[bucket] += 1
        by_bound[bucket].append(idx)

    states = (counts[0] + 1) * (counts[1] + 1) * (counts[2] + 1)
    if states > state_limit:
        return None

    patterns: list[tuple[int, int, int]] = []
    for c1 in range(cap + 1):
        for c2 in range(cap // 2 + 1):
            for c3 in range(cap // 3 + 1):
                fill = c1 + 2 * c2 + 3 * c3
                if 0 < fill <= cap:
                    patterns.append((c1, c2, c3))
    patterns.sort(key=lambda p: (p[0] + 2 * p[1] + 3 * p[2], p[0] + p[1] + p[2]), reverse=True)

    @lru_cache(maxsize=None)
    def solve(c1: int, c2: int, c3: int) -> tuple[int, tuple[int, int, int] | None]:
        if (c1, c2, c3) == (0, 0, 0):
            return 0, None
        best = 10**9
        best_pattern: tuple[int, int, int] | None = None
        best_fill = -1
        for p1, p2, p3 in patterns:
            if p1 > c1 or p2 > c2 or p3 > c3:
                continue
            tail, _ = solve(c1 - p1, c2 - p2, c3 - p3)
            candidate = 1 + tail
            fill = p1 + 2 * p2 + 3 * p3
            if candidate < best or (candidate == best and fill > best_fill):
                best = candidate
                best_pattern = (p1, p2, p3)
                best_fill = fill
        return best, best_pattern

    groups: list[list[int]] = []
    state = tuple(counts)
    while state != (0, 0, 0):
        _, pattern = solve(*state)
        if pattern is None:
            return None
        group: list[int] = []
        for bucket, take in enumerate(pattern):
            for _ in range(take):
                group.append(by_bound[bucket].pop())
        groups.append(group)
        state = (state[0] - pattern[0], state[1] - pattern[1], state[2] - pattern[2])
    return groups


def chunk_terms_by_bound(terms: list[Term], cap: int, token_aware: bool = False) -> list[list[Term]]:
    if token_aware:
        groups = exact_small_bound_groups([term.bound for term in terms], cap)
        if groups is not None:
            return [[terms[idx] for idx in group] for group in groups]

    chunks: list[list[Term]] = []
    chunk_bounds: list[int] = []
    for term in sorted(terms, key=lambda t: t.bound, reverse=True):
        if term.bound > cap:
            raise ValueError(f"term bound {term.bound} exceeds cap {cap}")
        eligible = [
            (bound, idx)
            for idx, bound in enumerate(chunk_bounds)
            if bound + term.bound <= cap
        ]
        if eligible:
            _, idx = max(eligible)
            chunk_bounds[idx] += term.bound
            chunks[idx].append(term)
        else:
            chunk_bounds.append(term.bound)
            chunks.append([term])
    return chunks


def simulate_normalization(
    columns_in: list[list[Term]],
    digits: int,
    prim: PrimitiveVars,
    profile: Profile,
    mode: str,
    normalizer_kernel: str,
    row_bits: int,
    tile_size: int,
    pbs_theta: int,
    lift_pbs_theta: int,
    cbs_extract_bits: int,
    chunk_cap: int | None,
    tail_model: str = "gaussian",
    subgaussian_proxy_scale: float = 3.0,
    product_padding_bits: int = 0,
    normalizer_padding_bits: int = 0,
    final_padding_bits: int = 0,
    normalizer_centered_ms: bool = False,
    token_aware_normalizer: bool = False,
    base: int = 4,
    width_bits_override: int | None = None,
    final_split_radix4: bool = False,
    output_row_bits: int | None = None,
    base16_split_kernel: str = "manylut",
    lift_centered_ms: bool = False,
    lift_box_centered: bool = False,
    correlation_model: str = "independent",
) -> ScheduleStats:
    if correlation_model not in ("independent", "worst-case", "source-aware"):
        raise ValueError(f"unknown correlation_model={correlation_model}")
    if correlation_model != "independent" and mode not in (
        "height2",
        "height2-add",
        "height2-parallel-add",
    ):
        raise ValueError(
            f"correlation_model={correlation_model} is implemented for the height2 schedule"
        )
    worst_case_correlation = correlation_model == "worst-case"
    source_aware_correlation = correlation_model == "source-aware"
    if source_aware_correlation and normalizer_kernel == "carry-only":
        raise ValueError(
            "source-aware correlation is not implemented for carry-only dirty digits"
        )
    output_bits = row_bits if output_row_bits is None else output_row_bits
    row_cap = chunk_cap if chunk_cap is not None else (1 << row_bits) - 1
    min_cap = 2 * (base - 1)
    if row_cap < min_cap or row_cap > (1 << row_bits) - 1:
        raise ValueError(f"chunk cap must be in {min_cap}..{(1 << row_bits) - 1}")
    normalizer_cap_bits = ceil_log2_domain_size(row_cap + 1)
    # The implementation builds the normalizer accumulator over
    # input_modulus = 2^row_bits, so the decision spacing is the row spacing
    # even when the public cap is smaller than 2^row_bits - 1.  Keep the cap
    # bits visible in the CSV to prevent confusing the logical bound with the
    # actual phase spacing.
    normalizer_threshold_bits = max(row_bits, normalizer_cap_bits)
    product_threshold_bits = row_bits
    final_threshold_bits = output_bits
    product_threshold_log2 = float(
        decision_threshold_log2(product_threshold_bits, product_padding_bits)
    )
    normalizer_threshold_log2 = float(
        decision_threshold_log2(normalizer_threshold_bits, normalizer_padding_bits)
    )
    final_threshold_log2 = float(
        decision_threshold_log2(final_threshold_bits, final_padding_bits)
    )
    columns = [list(col) for col in columns_in]
    if len(columns) < digits + 1:
        columns.extend([] for _ in range(digits + 1 - len(columns)))

    lift_event_count = 2 * digits
    # pbs_input_log2_pfail screens at delta_in / 2.  The current selector
    # accumulator has only q/16 geometric margin; centering its boxes on the
    # radix-4 inputs would restore q/8.  This geometry choice is independent
    # of whether ordinary or centered modulus switching supplies the variance.
    lift_threshold_log2 = float(
        64 - cbs_extract_bits - (1 if lift_box_centered else 2)
    )
    lift_delta_in = 2.0 ** (
        64 - cbs_extract_bits - (0 if lift_box_centered else 1)
    )
    lift_event_log2 = pbs_input_log2_pfail(
        profile.lwe_n,
        Q,
        profile.poly_n,
        lift_pbs_theta,
        lift_delta_in,
        prim.lift_input_var,
        tail_model,
        subgaussian_proxy_scale,
        centered_binary_ms=lift_centered_ms,
    )
    lift_union_log2 = (
        log2_sum_exp([lift_event_log2] * lift_event_count)
        if lift_event_count
        else float("-inf")
    )

    # Product digits are not decoded between CMUX extraction and normalization.
    # Their noise is carried by Term metadata into the first normalizer PBS (or
    # into the final output decision if a term survives without refresh).  A
    # separate per-product decode event would count a decision that the circuit
    # never makes and would double-count the same noise.
    product_vars = [
        term_var_source_aware(term) if source_aware_correlation else term.var
        for column in columns_in[:digits]
        for term in column
    ]
    max_product_var = max(product_vars) if product_vars else 0.0
    product_log2_events: list[float] = []
    max_product_log2_pfail = float("-inf")
    product_union_log2 = float("-inf")

    pbs = 0
    rounds = 0
    max_chunks = 0
    max_column_height = 0
    max_column_bound = 0
    max_chunk_input_var = 0.0
    max_chunk_input_log2_pfail = float("-inf")
    normalizer_log2_events: list[float] = []

    def normalized_terms(chunk_bound: int, chunk_var: float) -> tuple[int, Term, Term]:
        carry = Term(
            chunk_bound // base,
            prim.normalizer_pbs_var,
            "norm-carry",
            refreshed=True,
        )
        if normalizer_kernel == "manylut":
            # One interleaved blind rotation extracts both outputs.  The two
            # samples are correlated, but each output has the post-PBS variance
            # scale; the scheduler only needs per-term variance magnitudes.
            digit = Term(
                base - 1,
                prim.normalizer_pbs_var,
                "norm-digit",
                refreshed=True,
            )
            return 1, digit, carry
        if normalizer_kernel == "refresh-digit":
            digit = Term(
                base - 1,
                prim.normalizer_pbs_var,
                "norm-digit",
                refreshed=True,
            )
            return 2, digit, carry
        if normalizer_kernel == "carry-only":
            # Rust's non-refresh path computes digit = chunk_sum - B*carry.
            # The digit is not PBS-refreshed, so it keeps chunk-sum noise plus
            # the scaled carry output noise.
            digit = Term(
                base - 1,
                chunk_var + (base * base) * prim.normalizer_pbs_var,
                "dirty-digit",
                refreshed=False,
                worst_amp=(
                    math.sqrt(chunk_var)
                    + base * math.sqrt(prim.normalizer_pbs_var)
                    if worst_case_correlation
                    else None
                ),
            )
            return 1, digit, carry
        raise ValueError(f"unknown normalizer_kernel={normalizer_kernel}")

    def normalize_jobs(jobs: list[tuple[int, list[Term]]]) -> None:
        nonlocal pbs, max_chunk_input_var, max_chunk_input_log2_pfail
        normalized: list[tuple[int, int, Term, Term]] = []
        for q, chunk in jobs:
            chunk_bound = sum(t.bound for t in chunk)
            if worst_case_correlation:
                chunk_var = chunk_input_var_worst_case(chunk)
            elif source_aware_correlation:
                chunk_var = chunk_input_var_source_aware(chunk)
            else:
                chunk_var = sum(t.var for t in chunk)
            input_var = max(chunk_var, prim.normalizer_input_var_floor) + prim.normalizer_ks_var
            max_chunk_input_var = max(max_chunk_input_var, input_var)
            log2p = pbs_input_log2_pfail(
                profile.lwe_n,
                Q,
                profile.poly_n,
                pbs_theta,
                delta_from_bits(normalizer_threshold_bits, normalizer_padding_bits),
                input_var,
                tail_model,
                subgaussian_proxy_scale,
                normalizer_centered_ms,
            )
            max_chunk_input_log2_pfail = max(max_chunk_input_log2_pfail, log2p)
            pbs_count, digit, carry = normalized_terms(chunk_bound, chunk_var)
            # Separate digit and carry PBS calls repeat the same deterministic
            # big-to-small key switch and modulus switch on the same input.
            # Hence they share one input-cell selection event.  Their two output
            # noises are propagated separately by `digit` and `carry`.
            normalizer_log2_events.append(log2p)
            normalized.append((q, chunk_bound, digit, carry))
            pbs += pbs_count
        for q, _chunk_bound, digit, carry in normalized:
            columns[q].append(digit)
            if carry.bound > 0 and q + 1 < digits:
                columns[q + 1].append(carry)

    def record_normalizer_events(input_vars: Iterable[float]) -> None:
        """Record PBS input-screen events using the normalizer parameters."""
        nonlocal max_chunk_input_var, max_chunk_input_log2_pfail
        for raw_var in input_vars:
            input_var = max(raw_var, prim.normalizer_input_var_floor) + prim.normalizer_ks_var
            max_chunk_input_var = max(max_chunk_input_var, input_var)
            log2p = pbs_input_log2_pfail(
                profile.lwe_n,
                Q,
                profile.poly_n,
                pbs_theta,
                delta_from_bits(normalizer_threshold_bits, normalizer_padding_bits),
                input_var,
                tail_model,
                subgaussian_proxy_scale,
                normalizer_centered_ms,
            )
            max_chunk_input_log2_pfail = max(max_chunk_input_log2_pfail, log2p)
            normalizer_log2_events.append(log2p)

    if mode in ("parallel", "serial"):
        q = 0
        while q < digits:
            if q == len(columns):
                columns.append([])
            while True:
                bound_sum = sum(t.bound for t in columns[q])
                if not columns[q] or bound_sum < base:
                    break
                rounds += 1
                max_column_height = max(max_column_height, len(columns[q]))
                max_column_bound = max(max_column_bound, bound_sum)
                current = columns[q]
                columns[q] = []
                chunks = chunk_terms_by_bound(current, row_cap, token_aware_normalizer)
                max_chunks = max(max_chunks, len(chunks))
                # Digit terms remain in the current column; carries move right.
                next_current: list[Term] = []
                next_carry: list[Term] = []
                for chunk in chunks:
                    chunk_bound = sum(t.bound for t in chunk)
                    chunk_var = sum(t.var for t in chunk)
                    input_var = max(chunk_var, prim.normalizer_input_var_floor) + prim.normalizer_ks_var
                    max_chunk_input_var = max(max_chunk_input_var, input_var)
                    log2p = pbs_input_log2_pfail(
                        profile.lwe_n,
                        Q,
                        profile.poly_n,
                        pbs_theta,
                        delta_from_bits(normalizer_threshold_bits, normalizer_padding_bits),
                        input_var,
                        tail_model,
                        subgaussian_proxy_scale,
                        normalizer_centered_ms,
                    )
                    max_chunk_input_log2_pfail = max(max_chunk_input_log2_pfail, log2p)
                    pbs_count, digit, carry = normalized_terms(chunk_bound, chunk_var)
                    normalizer_log2_events.append(log2p)
                    pbs += pbs_count
                    next_current.append(digit)
                    if carry.bound > 0:
                        next_carry.append(carry)
                columns[q] = next_current
                if q + 1 < digits:
                    columns[q + 1].extend(next_carry)
            q += 1
    elif mode.startswith("tile"):
        if tile_size <= 0:
            if mode == "tile4":
                tile_size = 4
            else:
                tile_size = int(mode.removeprefix("tile"))
        for tile_start in range(0, digits, tile_size):
            tile_end = min(tile_start + tile_size, digits)
            while True:
                jobs: list[tuple[int, list[Term]]] = []
                for q in range(tile_start, tile_end):
                    current = columns[q]
                    columns[q] = []
                    if not current:
                        continue
                    bound_sum = sum(t.bound for t in current)
                    if bound_sum < base:
                        columns[q] = current
                        continue
                    max_column_height = max(max_column_height, len(current))
                    max_column_bound = max(max_column_bound, bound_sum)
                    chunks = chunk_terms_by_bound(current, row_cap, token_aware_normalizer)
                    max_chunks = max(max_chunks, len(chunks))
                    for chunk in chunks:
                        jobs.append((q, chunk))
                if not jobs:
                    break
                rounds += 1
                normalize_jobs(jobs)
    elif mode == "wavefront":
        while True:
            next_columns: list[list[Term]] = [[] for _ in range(digits)]
            jobs = []
            for q in range(digits):
                current = columns[q]
                columns[q] = []
                if not current:
                    continue
                bound_sum = sum(t.bound for t in current)
                if bound_sum < base:
                    next_columns[q] = current
                    continue
                max_column_height = max(max_column_height, len(current))
                max_column_bound = max(max_column_bound, bound_sum)
                chunks = chunk_terms_by_bound(current, row_cap, token_aware_normalizer)
                max_chunks = max(max_chunks, len(chunks))
                for chunk in chunks:
                    jobs.append((q, chunk))
            columns = next_columns + [[] for _ in range(max(0, len(columns) - digits))]
            if not jobs:
                break
            rounds += 1
            normalize_jobs(jobs)
    elif mode in ("height2", "height2-add", "height2-parallel-add"):
        if base != 4 or row_bits != 4:
            raise ValueError("height2 normalization requires radix 4 with row_bits=4")

        # Match normalize_bounded_terms_height2 in the Rust evaluator.  Every
        # level reduces only columns above height two.  The remaining two rows
        # are handed to TFHE-rs for one ordinary radix addition.
        columns = columns[:digits]
        while any(len(column) > 2 for column in columns):
            before_terms = sum(len(column) for column in columns)
            next_columns: list[list[Term]] = [[] for _ in range(digits)]
            jobs: list[tuple[int, list[Term]]] = []
            for q, current in enumerate(columns):
                bound_sum = sum(term.bound for term in current)
                max_column_height = max(max_column_height, len(current))
                max_column_bound = max(max_column_bound, bound_sum)
                if len(current) <= 2:
                    next_columns[q].extend(current)
                    continue

                chunks = chunk_terms_by_bound(current, row_cap, token_aware_normalizer)
                max_chunks = max(max_chunks, len(chunks))
                for chunk in chunks:
                    chunk_bound = sum(term.bound for term in chunk)
                    if len(chunk) < 3:
                        next_columns[q].extend(chunk)
                    else:
                        jobs.append((q, chunk))

            columns = next_columns
            if jobs:
                rounds += 1
                normalize_jobs(jobs)
            after_terms = sum(len(column) for column in columns)
            if after_terms >= before_terms:
                raise ValueError("height2 compression made no progress")

        if any(len(column) > 1 for column in columns):
            # Rust full-propagates the two rows only when a surviving term did
            # not come from a normalizer PBS.  Its accounting charges one PBS
            # per such term; record each term's actual input variance here.
            unrefreshed = [
                term for column in columns for term in column if not term.refreshed
            ]
            if unrefreshed:
                if source_aware_correlation:
                    refresh_inputs = (term_var_source_aware(term) for term in unrefreshed)
                elif worst_case_correlation:
                    refresh_inputs = (term_corr_amp(term) ** 2 for term in unrefreshed)
                else:
                    refresh_inputs = (term.var for term in unrefreshed)
                record_normalizer_events(refresh_inputs)
                pbs += len(unrefreshed)
                rounds += 1
                columns = [
                    [
                        Term(term.bound, prim.normalizer_pbs_var, "height2-row-refresh", True)
                        for term in column
                    ]
                    for column in columns
                ]

            # TFHE-rs selects its sequential or parallel carry propagation from
            # the active Rayon thread count.  Screen both paths and retain the
            # larger failure union, so one parameter estimate covers every
            # thread count used by the artifact.  All row blocks are clean at
            # this point; the event multipliers below model the internal linear
            # combinations and the radix-4 bivariate packing of add.rs.
            (
                sequential_multipliers,
                parallel_multipliers,
                sequential_pbs_calls,
                parallel_pbs_calls,
            ) = tfhe_final_add_event_multipliers(digits)

            def add_path_log2_union(multipliers: list[float]) -> float:
                return log2_sum_exp(
                    pbs_input_log2_pfail(
                        profile.lwe_n,
                        Q,
                        profile.poly_n,
                        pbs_theta,
                        delta_from_bits(
                            normalizer_threshold_bits, normalizer_padding_bits
                        ),
                        multiplier * prim.normalizer_pbs_var
                        + prim.normalizer_ks_var,
                        tail_model,
                        subgaussian_proxy_scale,
                        normalizer_centered_ms,
                    )
                    for multiplier in multipliers
                )

            sequential_union = add_path_log2_union(sequential_multipliers)
            parallel_union = add_path_log2_union(parallel_multipliers)
            add_multipliers = (
                parallel_multipliers
                if parallel_union >= sequential_union
                else sequential_multipliers
            )
            record_normalizer_events(
                multiplier * prim.normalizer_pbs_var
                for multiplier in add_multipliers
            )
            # `pbs` is a work counter rather than a failure-event counter.  The
            # sequential path performs two PBS calls on each shared input, while
            # the parallel path performs one call per modeled event.
            pbs += max(sequential_pbs_calls, parallel_pbs_calls)
            rounds += 1
            columns = [
                [Term(base - 1, prim.normalizer_pbs_var, "height2-final-add", True)]
                for _ in range(digits)
            ]
    else:
        raise ValueError(f"unknown mode={mode}")

    def terminal_column_var(column: list[Term]) -> float:
        """Variance presented to the final radix decoder.

        The height-two path normally refreshes unrefreshed terms before its
        final addition.  A one-row state, however, is returned directly by the
        Rust evaluator.  Preserve the selected covariance model on that path
        instead of silently falling back to an independent variance sum.
        """
        if worst_case_correlation:
            return chunk_input_var_worst_case(column)
        if source_aware_correlation:
            return chunk_input_var_source_aware(column)
        return sum(term.var for term in column)

    final_vars = [terminal_column_var(columns[q]) for q in range(digits)]
    if final_split_radix4:
        split_count = digits
        if base16_split_kernel not in ("manylut", "separate-pbs"):
            raise ValueError(f"unknown base16_split_kernel={base16_split_kernel}")
        rounds += 1
        max_chunks = max(max_chunks, 1)
        max_column_height = max(max_column_height, 1)
        max_column_bound = max(max_column_bound, base - 1)
        split_padding_bits = (
            normalizer_padding_bits if base16_split_kernel == "manylut" else final_padding_bits
        )
        split_pbs_per_input = 1 if base16_split_kernel == "manylut" else 2
        split_log2_events = []
        for var in final_vars:
            log2p = pbs_input_log2_pfail(
                profile.lwe_n,
                Q,
                profile.poly_n,
                pbs_theta,
                delta_from_bits(normalizer_threshold_bits, split_padding_bits),
                max(var, prim.normalizer_input_var_floor) + prim.normalizer_ks_var,
                tail_model,
                subgaussian_proxy_scale,
                normalizer_centered_ms,
            )
            split_log2_events.extend([log2p] * split_pbs_per_input)
        normalizer_log2_events.extend(split_log2_events)
        pbs += split_count * split_pbs_per_input
        final_vars = [prim.normalizer_pbs_var for _ in range(split_count * 2)]

    # Rust restores final digits to CBS_OUTPUT_ROW_BITS when the internal row
    # spacing differs.  This is one ordinary PBS per output digit after the
    # base16 split (if any).
    if output_bits != row_bits:
        restore_log2_events = [
            pbs_input_log2_pfail(
                profile.lwe_n,
                Q,
                profile.poly_n,
                pbs_theta,
                delta_from_bits(row_bits, final_padding_bits),
                max(var, prim.normalizer_input_var_floor) + prim.normalizer_ks_var,
                tail_model,
                subgaussian_proxy_scale,
                normalizer_centered_ms,
            )
            for var in final_vars
        ]
        normalizer_log2_events.extend(restore_log2_events)
        pbs += len(final_vars)
        final_vars = [prim.normalizer_pbs_var for _ in final_vars]

    final_log2_events = [
        centered_decode_log2_pfail(
            final_threshold_bits,
            var,
            tail_model,
            subgaussian_proxy_scale,
            final_padding_bits,
        )
        for var in final_vars
    ]
    max_final_var = max(final_vars) if final_vars else 0.0
    max_final_log2_pfail = max(final_log2_events) if final_log2_events else float("-inf")
    normalizer_union_log2 = log2_sum_exp(normalizer_log2_events)
    final_union_log2 = log2_sum_exp(final_log2_events)
    post_product_union_log2 = min(
        0.0,
        log2_sum_exp([normalizer_union_log2, final_union_log2]),
    )
    union_log2 = min(
        0.0,
        log2_sum_exp([lift_union_log2, product_union_log2, post_product_union_log2]),
    )

    return ScheduleStats(
        width_bits=width_bits_override if width_bits_override is not None else digits * 2,
        digits=digits,
        mode=mode,
        row_bits=row_bits,
        pbs=pbs,
        rounds=rounds,
        max_chunks=max_chunks,
        max_column_height=max_column_height,
        max_column_bound=max_column_bound,
        lift_event_count=lift_event_count,
        lift_threshold_log2=lift_threshold_log2,
        max_lift_log2_pfail=lift_event_log2,
        lift_union_log2_pfail=lift_union_log2,
        product_threshold_bits=product_threshold_bits,
        product_padding_bits=product_padding_bits,
        product_threshold_log2=product_threshold_log2,
        max_product_log2_var=log2_or_neginf(max_product_var),
        max_product_log2_pfail=max_product_log2_pfail,
        product_event_count=len(product_log2_events),
        product_union_log2_pfail=product_union_log2,
        normalizer_cap_bits=normalizer_cap_bits,
        normalizer_threshold_bits=normalizer_threshold_bits,
        normalizer_padding_bits=normalizer_padding_bits,
        normalizer_threshold_log2=normalizer_threshold_log2,
        max_chunk_input_log2_var=log2_or_neginf(max_chunk_input_var),
        max_chunk_input_log2_pfail=max_chunk_input_log2_pfail,
        final_threshold_bits=final_threshold_bits,
        final_padding_bits=final_padding_bits,
        final_threshold_log2=final_threshold_log2,
        max_final_log2_var=log2_or_neginf(max_final_var),
        max_final_log2_pfail=max_final_log2_pfail,
        normalizer_event_count=len(normalizer_log2_events),
        normalizer_union_log2_pfail=normalizer_union_log2,
        final_event_count=len(final_log2_events),
        final_union_log2_pfail=final_union_log2,
        post_product_union_log2_pfail=post_product_union_log2,
        union_log2_pfail=union_log2,
    )


def profiles() -> dict[str, Profile]:
    return {
        "m2c2-tuniform": Profile(
            name="m2c2-tuniform",
            lwe_n=918,
            glwe_k=1,
            poly_n=2048,
            lwe_var=t_uniform_normalized_var(45),
            glwe_var=t_uniform_normalized_var(17),
            pbs_base_log=23,
            pbs_level=1,
            cbs_lift_pbs_base_log=23,
            cbs_lift_pbs_level=1,
            ks_base_log=4,
            ks_level=4,
            cbs_lift_ks_base_log=4,
            cbs_lift_ks_level=4,
            auto_base_log=7,
            auto_level=7,
            ss_base_log=17,
            ss_level=2,
            cbs_base_log=4,
            cbs_level=4,
        ),
        "m2c2-gaussian": Profile(
            name="m2c2-gaussian",
            lwe_n=866,
            glwe_k=1,
            poly_n=2048,
            lwe_var=2.046151696979124e-6**2,
            glwe_var=2.845267479601915e-15**2,
            pbs_base_log=23,
            pbs_level=1,
            cbs_lift_pbs_base_log=23,
            cbs_lift_pbs_level=1,
            ks_base_log=3,
            ks_level=5,
            cbs_lift_ks_base_log=3,
            cbs_lift_ks_level=5,
            auto_base_log=7,
            auto_level=7,
            ss_base_log=17,
            ss_level=2,
            cbs_base_log=4,
            cbs_level=4,
        ),
        "m2c1-gaussian": Profile(
            name="m2c1-gaussian",
            lwe_n=884,
            glwe_k=2,
            poly_n=1024,
            lwe_var=1.4999005934396873e-6**2,
            glwe_var=2.845267479601915e-15**2,
            pbs_base_log=23,
            pbs_level=1,
            cbs_lift_pbs_base_log=23,
            cbs_lift_pbs_level=1,
            ks_base_log=5,
            ks_level=3,
            cbs_lift_ks_base_log=5,
            cbs_lift_ks_level=3,
            auto_base_log=7,
            auto_level=7,
            ss_base_log=17,
            ss_level=2,
            cbs_base_log=4,
            cbs_level=4,
        ),
        "m2c3-gaussian": Profile(
            name="m2c3-gaussian",
            lwe_n=930,
            glwe_k=1,
            poly_n=4096,
            lwe_var=6.782362904013915e-7**2,
            glwe_var=2.168404344971009e-19**2,
            pbs_base_log=15,
            pbs_level=2,
            cbs_lift_pbs_base_log=15,
            cbs_lift_pbs_level=2,
            ks_base_log=3,
            ks_level=6,
            cbs_lift_ks_base_log=3,
            cbs_lift_ks_level=6,
            auto_base_log=7,
            auto_level=7,
            ss_base_log=17,
            ss_level=2,
            cbs_base_log=4,
            cbs_level=4,
        ),
        "m2c4-gaussian": Profile(
            name="m2c4-gaussian",
            lwe_n=1007,
            glwe_k=1,
            poly_n=8192,
            lwe_var=1.796446316728823e-7**2,
            glwe_var=2.168404344971009e-19**2,
            pbs_base_log=15,
            pbs_level=2,
            cbs_lift_pbs_base_log=15,
            cbs_lift_pbs_level=2,
            ks_base_log=3,
            ks_level=7,
            cbs_lift_ks_base_log=3,
            cbs_lift_ks_level=7,
            auto_base_log=7,
            auto_level=7,
            ss_base_log=17,
            ss_level=2,
            cbs_base_log=4,
            cbs_level=4,
        ),
        "revhomtrace-base16": Profile(
            name="revhomtrace-base16",
            lwe_n=769,
            glwe_k=1,
            poly_n=2048,
            lwe_var=8.763872947670246e-06**2,
            glwe_var=9.25119974676756e-16**2,
            pbs_base_log=15,
            pbs_level=2,
            cbs_lift_pbs_base_log=15,
            cbs_lift_pbs_level=2,
            ks_base_log=4,
            ks_level=3,
            cbs_lift_ks_base_log=4,
            cbs_lift_ks_level=3,
            auto_base_log=7,
            auto_level=7,
            ss_base_log=17,
            ss_level=2,
            cbs_base_log=4,
            cbs_level=4,
        ),
        "tetris-set-i-like": Profile(
            name="tetris-set-i-like",
            lwe_n=710,
            glwe_k=1,
            poly_n=1024,
            lwe_var=(2.8147e14 / Q) ** 2,
            glwe_var=(8.192e3 / Q) ** 2,
            pbs_base_log=12,
            pbs_level=3,
            cbs_lift_pbs_base_log=12,
            cbs_lift_pbs_level=3,
            ks_base_log=4,
            ks_level=4,
            cbs_lift_ks_base_log=4,
            cbs_lift_ks_level=4,
            auto_base_log=10,
            auto_level=4,
            ss_base_log=13,
            ss_level=3,
            cbs_base_log=8,
            cbs_level=2,
        ),
        "tetris-set-ii-like": Profile(
            name="tetris-set-ii-like",
            lwe_n=710,
            glwe_k=1,
            poly_n=1024,
            lwe_var=(2.8147e14 / Q) ** 2,
            glwe_var=(8.192e3 / Q) ** 2,
            pbs_base_log=5,
            pbs_level=9,
            cbs_lift_pbs_base_log=5,
            cbs_lift_pbs_level=9,
            ks_base_log=4,
            ks_level=4,
            cbs_lift_ks_base_log=4,
            cbs_lift_ks_level=4,
            auto_base_log=4,
            auto_level=12,
            ss_base_log=8,
            ss_level=6,
            cbs_base_log=16,
            cbs_level=1,
        ),
    }


def parse_widths(text: str) -> list[int]:
    widths: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            parts = [int(x) for x in item.split(":")]
            if len(parts) == 2:
                start, stop = parts
                step = 1
            elif len(parts) == 3:
                start, stop, step = parts
            else:
                raise ValueError(f"bad width range: {item}")
            widths.extend(range(start, stop + 1, step))
        else:
            widths.append(int(item))
    return widths


def apply_overrides(profile: Profile, args: argparse.Namespace) -> Profile:
    mapping = {
        "pbs_base_log": args.pbs_base_log,
        "pbs_level": args.pbs_level,
        "cbs_lift_pbs_base_log": args.cbs_lift_pbs_base_log,
        "cbs_lift_pbs_level": args.cbs_lift_pbs_level,
        "ks_base_log": args.ks_base_log,
        "ks_level": args.ks_level,
        "cbs_lift_ks_base_log": args.cbs_lift_ks_base_log,
        "cbs_lift_ks_level": args.cbs_lift_ks_level,
        "auto_base_log": args.auto_base_log,
        "auto_level": args.auto_level,
        "ss_base_log": args.ss_base_log,
        "ss_level": args.ss_level,
        "cbs_base_log": args.cbs_base_log,
        "cbs_level": args.cbs_level,
    }
    updates = {key: value for key, value in mapping.items() if value is not None}
    return replace(profile, **updates)


def fmt_log2(x: float) -> str:
    if x == float("-inf"):
        return "-inf"
    if x == float("inf"):
        return "inf"
    return f"{x:.2f}"


def iter_rows(args: argparse.Namespace) -> Iterable[tuple[Profile, PrimitiveVars, ScheduleStats, str]]:
    profile_map = profiles()
    selected = args.profile
    if selected == "all":
        names = list(profile_map)
    else:
        names = [selected]
    widths = parse_widths(args.widths)
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    contract = normalize_contract(args.product_contract, folded=not args.no_fold)
    if args.correlation_model != "independent" and args.cmux_noise_model != "linear":
        raise SystemExit(
            "source-aware and path-correlation stress models are validated only "
            "with --cmux-noise-model linear"
        )
    for name in names:
        if name not in profile_map:
            raise SystemExit(f"unknown profile {name}; choices: {', '.join(profile_map)}")
        profile = apply_overrides(profile_map[name], args)
        prim = primitive_vars(
            profile,
            args.row_bits,
            args.cbs_extract_bits,
            args.cmux_noise_model,
            args.cbs_lift_model,
            args.trace_model,
            args.trace_scale_model,
            cbs_lift_log2_var_floor=args.cbs_lift_log2_var_floor,
        )
        prim = apply_calibration(
            prim,
            args.calibrate_noise_csv,
            args.product_log2_var,
            args.normalizer_log2_var,
            args.calibration_std_factor,
        )
        prim = align_source_split(prim, contract)
        for width in widths:
            if width % 2 != 0:
                raise SystemExit(f"width must be a multiple of 2 for base-4 digits: {width}")
            digits = width // 2
            columns = product_terms(
                digits, prim, contract=contract, identify_operands=args.identify_operands
            )
            normalizer_digits = digits // 2 if contract == "chunk2x2-base16-first" else digits
            normalizer_base = 16 if contract == "chunk2x2-base16-first" else 4
            final_split_radix4 = contract == "chunk2x2-base16-first"
            for mode in modes:
                stats = simulate_normalization(
                    columns,
                    normalizer_digits,
                    prim,
                    profile,
                    mode,
                    args.normalizer_kernel,
                    args.row_bits,
                    args.tile_size,
                    args.pbs_theta,
                    args.lift_pbs_theta
                    if args.lift_pbs_theta is not None
                    else args.cbs_extract_bits,
                    args.cbs_extract_bits,
                    args.chunk_cap,
                    args.tail_model,
                    args.subgaussian_proxy_scale,
                    args.product_padding_bits,
                    args.normalizer_padding_bits,
                    args.final_padding_bits,
                    args.normalizer_centered_ms,
                    args.token_aware_normalizer,
                    normalizer_base,
                    width,
                    final_split_radix4,
                    args.output_row_bits,
                    args.base16_split_kernel,
                    lift_centered_ms=args.lift_centered_ms,
                    lift_box_centered=args.lift_box_centered,
                    correlation_model=args.correlation_model,
                )
                yield profile, prim, stats, contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="m2c2-gaussian")
    parser.add_argument("--widths", default="16,32,64,128,256")
    parser.add_argument("--modes", default="parallel,tile4")
    parser.add_argument("--row-bits", type=int, default=4)
    parser.add_argument("--tile-size", type=int, default=4)
    parser.add_argument("--pbs-theta", type=int, default=0)
    parser.add_argument(
        "--lift-pbs-theta",
        type=int,
        help="log LUT count / theta used for refined selector-lift cell selection; default is --cbs-extract-bits",
    )
    parser.add_argument(
        "--lift-centered-ms",
        action="store_true",
        help="use TFHE-rs centered binary modulus-switch variance for selector-lift PBS input events; independent of --lift-pbs-theta",
    )
    parser.add_argument(
        "--lift-box-centered",
        action="store_true",
        help="model selector boxes centered on radix-4 inputs (q/8 threshold); default models the current uncentered boxes (q/16 threshold), independently of --lift-centered-ms",
    )
    parser.add_argument(
        "--cbs-lift-log2-var-floor",
        type=float,
        help=(
            "conservative floor (log2, absolute torus variance) on the effective "
            "GGSW selector-error variance, fitted from encrypted measurements; "
            "raises v_cbs before the CMux external-product model"
        ),
    )
    parser.add_argument(
        "--identify-operands",
        action="store_true",
        help=(
            "squaring sensitivity: treat the second operand as the first (Y = X), so "
            "every lookup reuses one operand's selector sources and diagonal cells "
            "traverse each digit source twice"
        ),
    )
    parser.add_argument(
        "--correlation-model",
        choices=["source-aware"],
        default="source-aware",
        help=(
            "source-aware dependency model for normalizer PBS inputs: decompose every unrefreshed term "
            "into shared selector sources and a private part, adds amplitudes of reused "
            "sources within one PBS input (Cauchy-Schwarz per shared source, grouped-refresh "
            "bits fully correlated), and adds private components and refreshed PBS outputs "
            "in variance"
        ),
    )
    parser.add_argument("--chunk-cap", type=int, help="override normalizer chunk cap; default is 2^row_bits-1")
    parser.add_argument(
        "--normalizer-kernel",
        choices=["manylut", "refresh-digit", "carry-only"],
        default="manylut",
        help="match the Rust normalization kernel: ManyLUT, two-PBS digit refresh, or carry-only dirty digit",
    )
    parser.add_argument("--cbs-extract-bits", type=int, default=2)
    parser.add_argument(
        "--cbs-lift-model",
        choices=["sage", "engineering"],
        default="sage",
        help="CBS lift composition model; `sage` keeps the Refined/Tetris surrounding CBS composition, `engineering` keeps the older pessimistic composition",
    )
    parser.add_argument(
        "--trace-model",
        choices=["revhomtrace", "homtrace"],
        default="revhomtrace",
        help="trace primitive variance model; `revhomtrace` uses the O(N log N) TCHES 2026/1/05 bound, `homtrace` keeps the old O(N^3) bound",
    )
    parser.add_argument(
        "--trace-scale-model",
        choices=["refined-cbs"],
        default="refined-cbs",
        help=(
            "conversion-layer scale applied to the trace variance; the actual "
            "RevHomTrace-to-scheme-switch composition multiplies the trace-added "
            "polynomial error by N/2"
        ),
    )
    parser.add_argument("--cmux-noise-model", choices=["linear", "guarded"], default="linear")
    parser.add_argument(
        "--tail-model",
        choices=["gaussian"],
        default="gaussian",
        help="failure-probability tail model; Gaussian uses a log-domain erfc estimate",
    )
    parser.add_argument(
        "--subgaussian-proxy-scale",
        type=float,
        default=3.0,
        help="kappa in Pr[|X|>=t] <= 2 exp(-t^2/(2 kappa V)) for tuniform-proxy/subgaussian tails",
    )
    parser.add_argument("--no-fold", action="store_true")
    parser.add_argument(
        "--product-contract",
        choices=[
            "folded-lh",
            "raw-lh",
            "chunk2x2-digits",
            "chunk2x2-base16-first",
            "chunk2x2-base16",
            "base16-first",
            "folded8",
            "chunk2x2",
            "chunk4x4",
            "chunk4x4-tree",
            "chunk4x4-tree-digits",
            "chunk4x4-split",
            "chunk4x4-split-digits",
            "chunk8x8",
            "chunk8x8-direct",
            "chunk8x8-direct-digits",
        ],
        help="product-generator output contract; default keeps legacy --no-fold behavior",
    )
    parser.add_argument(
        "--chunk4x4-split-prefix-bits",
        "--chunk8x8-prefix-bits",
        dest="chunk4x4_split_prefix_bits",
        type=int,
        default=8,
        help="prefix-bank selector bits for chunk8x8-direct-digits CMUX count; Rust default is CBS_CHUNK8X8_PREFIX_BITS=8",
    )
    parser.add_argument(
        "--calibrate-noise-csv",
        type=Path,
        help="noise-probe CSV; stage=term calibrates product terms, stage=digit/carry/final_sum calibrates normalizer outputs",
    )
    parser.add_argument(
        "--calibration-std-factor",
        type=float,
        default=1.0,
        help="optional multiplier for calibrated empirical standard deviations (default: no inflation)",
    )
    parser.add_argument("--product-log2-var", type=float, help="override product term variance in log2 integer-torus units")
    parser.add_argument("--normalizer-log2-var", type=float, help="override normalizer output variance in log2 integer-torus units")
    parser.add_argument(
        "--product-padding-bits",
        type=int,
        default=0,
        help="extra effective bits consumed by the product-output decision screen",
    )
    parser.add_argument(
        "--normalizer-padding-bits",
        type=int,
        default=0,
        help="extra effective bits consumed by the normalizer PBS/ManyLUT input screen",
    )
    parser.add_argument(
        "--final-padding-bits",
        type=int,
        default=0,
        help="extra effective bits consumed by the final raw-decode screen",
    )
    parser.add_argument(
        "--output-row-bits",
        type=int,
        help="ordinary output row spacing; when different from --row-bits, model one restore PBS per output digit",
    )
    parser.add_argument(
        "--base16-split-kernel",
        choices=["manylut", "separate-pbs"],
        default="manylut",
        help="how the optional base16-first final split returns radix-4 digits",
    )
    parser.add_argument(
        "--normalizer-centered-ms",
        action="store_true",
        help="use TFHE-rs centered binary modulus-switch variance for normalizer PBS/ManyLUT input events",
    )
    parser.add_argument(
        "--token-aware-normalizer",
        action="store_true",
        help="match CBS_TOKEN_AWARE_NORMALIZER=1: exact public-bound packing for clean digit/carry tokens when feasible",
    )
    parser.add_argument("--pbs-base-log", type=int)
    parser.add_argument("--pbs-level", type=int)
    parser.add_argument("--cbs-lift-pbs-base-log", type=int)
    parser.add_argument("--cbs-lift-pbs-level", type=int)
    parser.add_argument("--ks-base-log", type=int)
    parser.add_argument("--ks-level", type=int)
    parser.add_argument("--cbs-lift-ks-base-log", type=int)
    parser.add_argument("--cbs-lift-ks-level", type=int)
    parser.add_argument("--auto-base-log", type=int)
    parser.add_argument("--auto-level", type=int)
    parser.add_argument("--ss-base-log", type=int)
    parser.add_argument("--ss-level", type=int)
    parser.add_argument("--cbs-base-log", type=int)
    parser.add_argument("--cbs-level", type=int)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    rows = list(iter_rows(args))
    fieldnames = [
        "profile",
        "width_bits",
        "digits",
        "mode",
        "row_bits",
        "chunk_cap",
        "tile_size",
        "normalizer_kernel",
        "normalizer_centered_ms",
        "lift_centered_ms",
        "lift_box_centered",
        "correlation_model",
        "identify_operands",
        "cbs_lift_log2_var_floor",
        "token_aware_normalizer",
        "product_contract",
        "chunk4x4_split_prefix_bits",
        "base16_split_kernel",
        "selector_bits_per_cell",
        "outputs_per_cell",
        "product_cells",
        "product_terms",
        "cmux_count",
        "pbs",
        "rounds",
        "max_chunks",
        "max_column_height",
        "max_column_bound",
        "log2_var_cbs_lift",
        "log2_var_cmux_ext",
        "log2_var_lift_input",
        "log2_var_prod_diag",
        "log2_var_prod_fold",
        "log2_var_prod_chunk2x2",
        "log2_var_prod_chunk4x4_split",
        "log2_var_norm_ks",
        "log2_var_norm_pbs",
        "log2_var_norm_input_floor",
        "lift_event_count",
        "lift_threshold_log2",
        "max_lift_log2_pfail",
        "lift_union_log2_pfail",
        "product_threshold_bits",
        "product_padding_bits",
        "product_threshold_log2",
        "max_product_log2_var",
        "max_product_log2_pfail",
        "product_event_count",
        "product_union_log2_pfail",
        "normalizer_cap_bits",
        "normalizer_threshold_bits",
        "normalizer_padding_bits",
        "normalizer_threshold_log2",
        "max_chunk_input_log2_var",
        "max_chunk_input_log2_pfail",
        "final_threshold_bits",
        "final_padding_bits",
        "final_threshold_log2",
        "max_final_log2_var",
        "max_final_log2_pfail",
        "normalizer_event_count",
        "normalizer_union_log2_pfail",
        "final_event_count",
        "final_union_log2_pfail",
        "post_product_union_log2_pfail",
        "union_log2_pfail",
        "union_log2_pfail_raw",
        "pbs_base_log",
        "pbs_level",
        "cbs_lift_pbs_base_log",
        "cbs_lift_pbs_level",
        "ks_base_log",
        "ks_level",
        "cbs_lift_ks_base_log",
        "cbs_lift_ks_level",
        "auto_base_log",
        "auto_level",
        "ss_base_log",
        "ss_level",
        "cbs_base_log",
        "cbs_level",
        "pbs_theta",
        "lift_pbs_theta",
        "cbs_extract_bits",
        "cbs_lift_model",
        "trace_model",
        "trace_scale_model",
        "cmux_noise_model",
        "tail_model",
        "subgaussian_proxy_scale",
        "calibration_csv",
        "calibration_std_factor",
        "variance_source",
    ]
    out_rows = []
    for profile, prim, stats, contract in rows:
        contract_stats = product_contract_stats(
            stats.width_bits // 2,
            contract,
            args.chunk4x4_split_prefix_bits,
        )
        out_rows.append(
            {
                "profile": profile.name,
                "width_bits": stats.width_bits,
                "digits": stats.digits,
                "mode": stats.mode,
                "row_bits": args.row_bits,
                "chunk_cap": args.chunk_cap if args.chunk_cap is not None else (1 << args.row_bits) - 1,
                "tile_size": args.tile_size,
                "normalizer_kernel": args.normalizer_kernel,
                "normalizer_centered_ms": int(args.normalizer_centered_ms),
                "lift_centered_ms": int(args.lift_centered_ms),
                "lift_box_centered": int(args.lift_box_centered),
                "correlation_model": args.correlation_model,
                "identify_operands": int(args.identify_operands),
                "cbs_lift_log2_var_floor": (
                    "" if args.cbs_lift_log2_var_floor is None
                    else f"{args.cbs_lift_log2_var_floor:.2f}"
                ),
                "token_aware_normalizer": int(args.token_aware_normalizer),
                "product_contract": contract,
                "chunk4x4_split_prefix_bits": args.chunk4x4_split_prefix_bits,
                "base16_split_kernel": args.base16_split_kernel,
                "selector_bits_per_cell": contract_stats["selector_bits_per_cell"],
                "outputs_per_cell": contract_stats["outputs_per_cell"],
                "product_cells": contract_stats["product_cells"],
                "product_terms": contract_stats["product_terms"],
                "cmux_count": contract_stats["cmux_count"],
                "pbs": stats.pbs,
                "rounds": stats.rounds,
                "max_chunks": stats.max_chunks,
                "max_column_height": stats.max_column_height,
                "max_column_bound": stats.max_column_bound,
                "log2_var_cbs_lift": fmt_log2(log2_or_neginf(prim.cbs_lift_var)),
                "log2_var_cmux_ext": fmt_log2(log2_or_neginf(prim.cmux_ext_var)),
                "log2_var_lift_input": fmt_log2(log2_or_neginf(prim.lift_input_var)),
                "log2_var_prod_diag": fmt_log2(log2_or_neginf(prim.product_diag_var)),
                "log2_var_prod_fold": fmt_log2(log2_or_neginf(prim.product_fold_var)),
                "log2_var_prod_chunk2x2": fmt_log2(log2_or_neginf(prim.product_chunk2x2_var)),
                "log2_var_prod_chunk4x4_split": fmt_log2(
                    log2_or_neginf(prim.product_chunk4x4_split_var)
                ),
                "log2_var_norm_ks": fmt_log2(log2_or_neginf(prim.normalizer_ks_var)),
                "log2_var_norm_pbs": fmt_log2(log2_or_neginf(prim.normalizer_pbs_var)),
                "log2_var_norm_input_floor": fmt_log2(
                    log2_or_neginf(prim.normalizer_input_var_floor)
                ),
                "lift_event_count": stats.lift_event_count,
                "lift_threshold_log2": fmt_log2(stats.lift_threshold_log2),
                "max_lift_log2_pfail": fmt_log2(stats.max_lift_log2_pfail),
                "lift_union_log2_pfail": fmt_log2(stats.lift_union_log2_pfail),
                "product_threshold_bits": stats.product_threshold_bits,
                "product_padding_bits": stats.product_padding_bits,
                "product_threshold_log2": fmt_log2(stats.product_threshold_log2),
                "max_product_log2_var": fmt_log2(stats.max_product_log2_var),
                "max_product_log2_pfail": fmt_log2(stats.max_product_log2_pfail),
                "product_event_count": stats.product_event_count,
                "product_union_log2_pfail": fmt_log2(stats.product_union_log2_pfail),
                "normalizer_cap_bits": stats.normalizer_cap_bits,
                "normalizer_threshold_bits": stats.normalizer_threshold_bits,
                "normalizer_padding_bits": stats.normalizer_padding_bits,
                "normalizer_threshold_log2": fmt_log2(stats.normalizer_threshold_log2),
                "max_chunk_input_log2_var": fmt_log2(stats.max_chunk_input_log2_var),
                "max_chunk_input_log2_pfail": fmt_log2(stats.max_chunk_input_log2_pfail),
                "final_threshold_bits": stats.final_threshold_bits,
                "final_padding_bits": stats.final_padding_bits,
                "final_threshold_log2": fmt_log2(stats.final_threshold_log2),
                "max_final_log2_var": fmt_log2(stats.max_final_log2_var),
                "max_final_log2_pfail": fmt_log2(stats.max_final_log2_pfail),
                "normalizer_event_count": stats.normalizer_event_count,
                "normalizer_union_log2_pfail": fmt_log2(stats.normalizer_union_log2_pfail),
                "final_event_count": stats.final_event_count,
                "final_union_log2_pfail": fmt_log2(stats.final_union_log2_pfail),
                "post_product_union_log2_pfail": fmt_log2(
                    stats.post_product_union_log2_pfail
                ),
                "union_log2_pfail": fmt_log2(stats.union_log2_pfail),
                "union_log2_pfail_raw": f"{stats.union_log2_pfail:.17g}",
                "pbs_base_log": profile.pbs_base_log,
                "pbs_level": profile.pbs_level,
                "cbs_lift_pbs_base_log": profile.cbs_lift_pbs_base_log,
                "cbs_lift_pbs_level": profile.cbs_lift_pbs_level,
                "ks_base_log": profile.ks_base_log,
                "ks_level": profile.ks_level,
                "cbs_lift_ks_base_log": profile.cbs_lift_ks_base_log,
                "cbs_lift_ks_level": profile.cbs_lift_ks_level,
                "auto_base_log": profile.auto_base_log,
                "auto_level": profile.auto_level,
                "ss_base_log": profile.ss_base_log,
                "ss_level": profile.ss_level,
                "cbs_base_log": profile.cbs_base_log,
                "cbs_level": profile.cbs_level,
                "pbs_theta": args.pbs_theta,
                "lift_pbs_theta": args.lift_pbs_theta
                if args.lift_pbs_theta is not None
                else args.cbs_extract_bits,
                "cbs_extract_bits": args.cbs_extract_bits,
                "cbs_lift_model": args.cbs_lift_model,
                "trace_model": args.trace_model,
                "trace_scale_model": args.trace_scale_model,
                "cmux_noise_model": args.cmux_noise_model,
                "tail_model": args.tail_model,
                "subgaussian_proxy_scale": args.subgaussian_proxy_scale,
                "calibration_csv": str(args.calibrate_noise_csv or ""),
                "calibration_std_factor": args.calibration_std_factor,
                "variance_source": prim.source,
            }
        )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_rows)

    writer = csv.DictWriter(
        # Small stdout adapter.
        type("Stdout", (), {"write": staticmethod(lambda s: print(s, end=""))})(),
        fieldnames=fieldnames,
        delimiter="\t",
    )
    writer.writeheader()
    writer.writerows(out_rows)


if __name__ == "__main__":
    main()
