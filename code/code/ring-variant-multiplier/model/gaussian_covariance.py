"""Gaussian linear-error propagation without a diagonal-covariance shortcut.

These identities do not infer a covariance matrix from samples. In particular,
they do not yet supply the trace/FFT source matrix needed by the fused kernel.
The envelope is conditional on valid coefficient-wise input variance bounds.
"""

import math


def negacyclic_weights(polynomial, index):
    """Weights of input coefficients in one coefficient of polynomial * input."""
    n = len(polynomial)
    if n == 0 or not 0 <= index < n:
        raise ValueError("coefficient outside the ring")
    return [polynomial[index-j] if j <= index else -polynomial[n+index-j]
            for j in range(n)]


def variance_from_sources(weights, coefficient_sources):
    """Each dict holds signed amplitudes of independent unit-variance sources.

    A repeated source name denotes the SAME random variable, including uses in
    different coefficients. Sources must come from a derivation, not a fitted
    collection of independent per-coefficient placeholders.
    """
    if len(weights) != len(coefficient_sources):
        raise ValueError("one source dictionary is required per coefficient")
    combined = {}
    for weight, sources in zip(weights, coefficient_sources):
        for source, amplitude in sources.items():
            if not math.isfinite(amplitude) or not math.isfinite(weight):
                raise ValueError("non-finite error amplitude")
            combined[source] = combined.get(source, 0.0) + weight * amplitude
    return math.fsum(amplitude * amplitude for amplitude in combined.values())


def covariance_envelope(weights, marginal_variances):
    """Sharp covariance-free variance bound (sum |w_i| sigma_i)^2.

    This is Cauchy--Schwarz applied to covariance entries, not a tail bound,
    empirical multiplier, or assertion that all coefficients are correlated.
    """
    if len(weights) != len(marginal_variances):
        raise ValueError("one marginal variance is required per coefficient")
    if any(not math.isfinite(v) or v < 0 for v in marginal_variances):
        raise ValueError("variances must be finite and nonnegative")
    if any(not math.isfinite(w) for w in weights):
        raise ValueError("weights must be finite")
    return math.fsum(abs(w) * math.sqrt(v)
                     for w, v in zip(weights, marginal_variances)) ** 2


def binary_trace_envelope(ring_degree, trace_coefficient_var):
    """Public-parameter envelope for -S * trace_error, binary S of degree < N."""
    if not isinstance(ring_degree, int) or ring_degree < 1:
        raise ValueError("ring degree must be a positive integer")
    return covariance_envelope([1] * ring_degree,
                               [trace_coefficient_var] * ring_degree)


def sparse_pbs_plus_trace_envelope(ring_degree, pbs_var, trace_coefficient_var):
    """PBS error occurs at coefficient zero; trace/PBS covariance is unrestricted.

    Scheme-switch additive error is NOT included. A full selector-row bound
    still needs that term and valid marginal models for trace and FFT errors.
    """
    return covariance_envelope([1, 1],
        [pbs_var, binary_trace_envelope(ring_degree, trace_coefficient_var)])
