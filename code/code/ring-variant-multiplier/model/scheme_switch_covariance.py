"""Exact shared-GLev external-product algebra and bounded joint moments.

Integer negacyclic identities are independent of a distribution. Applying a
Gaussian tail still requires a justified centered event-error model. None of
these functions approve parameters or infer independence from an experiment.
"""
import math


def negacyclic_product(left, right):
    if not left or len(left) != len(right):
        raise ValueError("equal nonempty polynomial lengths required")
    n = len(left)
    out = [0] * n
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            out[(i + j) % n] += a * b * (1 if i + j < n else -1)
    return out


def polynomial_sum(polynomials, n):
    out = [0] * n
    for polynomial in polynomials:
        if len(polynomial) != n:
            raise ValueError("polynomial size mismatch")
        out = [a + b for a, b in zip(out, polynomial)]
    return out


def decomposition_phase(rows, secret):
    """D_body - sum_j D_j*S_j for one gadget level."""
    if not rows or len(rows) != len(secret) + 1:
        raise ValueError("one mask row per secret component and one body row required")
    n = len(rows[-1])
    mask = polynomial_sum((negacyclic_product(d, s) for d, s in zip(rows, secret)), n)
    return [body - a for body, a in zip(rows[-1], mask)]


def shared_row_errors(secret, glev_error, switch_errors):
    """Mask row: -S_j*e + zeta_j. Body row: e, copied without new SS error."""
    if len(secret) != len(switch_errors):
        raise ValueError("one new scheme-switch error per mask row required")
    n = len(glev_error)
    rows = []
    for s, zeta in zip(secret, switch_errors):
        if len(zeta) != n:
            raise ValueError("scheme-switch error size mismatch")
        transformed = negacyclic_product(s, glev_error)
        rows.append([z - x for z, x in zip(zeta, transformed)])
    return rows + [list(glev_error)]


def direct_error(rows, errors):
    if not rows or len(rows) != len(errors):
        raise ValueError("one error polynomial per decomposition row required")
    return polynomial_sum((negacyclic_product(d, u) for d, u in zip(rows, errors)), len(rows[0]))


def factored_error(rows, secret, glev_error, switch_errors):
    """sum D_j*U_j = Phi(D)*e + sum_mask D_j*zeta_j, exactly."""
    shared = negacyclic_product(decomposition_phase(rows, secret), glev_error)
    fresh = polynomial_sum((negacyclic_product(d, z) for d, z in zip(rows, switch_errors)), len(shared))
    return [a + b for a, b in zip(shared, fresh)]


def bounded_joint_second_moment(coefficient_bounds, error_second_moments):
    """E[(sum D_i U_i)^2] <= (sum M_i sqrt(E[U_i^2]))^2, |D_i|<=M_i.

    Allows arbitrary dependence between D and U and among U coordinates.
    Raw second moments retain biases; a centered Gaussian tail cannot be
    inferred from this inequality alone.
    """
    if not coefficient_bounds or len(coefficient_bounds) != len(error_second_moments):
        raise ValueError("nonempty matching bounds and moments required")
    if any(not math.isfinite(x) or x < 0 for x in [*coefficient_bounds, *error_second_moments]):
        raise ValueError("finite nonnegative bounds and moments required")
    return math.fsum(m * math.sqrt(v) for m, v in zip(coefficient_bounds, error_second_moments)) ** 2


def uniform_external_product_moment(n, k, base_log, levels, row_second_moment):
    """Compact bound for (k+1)*ell*N signed digits bounded by B/2."""
    if any(not isinstance(x, int) or x < 1 for x in (n, k, base_log, levels)):
        raise ValueError("positive dimensions and decomposition required")
    if base_log * levels >= 64:
        raise ValueError("native signed decomposition requires fewer than 64 bits")
    if not math.isfinite(row_second_moment) or row_second_moment < 0:
        raise ValueError("invalid row second moment")
    return ((k + 1) * levels * n * 2 ** (base_log - 1)) ** 2 * row_second_moment
