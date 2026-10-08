"""Second moments of a decomposition inner product, not a parameter approval.

The product identity requires the decomposition vector and row-error vector to
be independent. Reusing selectors or choosing state-dependent inputs does not
establish that hypothesis. No covariance matrix is inferred from measurements.
"""

import math


def inner_product_second_moment(decomposition_second_moment, error_second_moment):
    """E[(D^T E)^2] = sum_ij E[D_i D_j] E[E_i E_j], for independent D,E.

    Matrices must be valid second-moment matrices supplied by the caller.
    Nonzero means are retained. Shape and symmetry are checked here; this
    function is not a general positive-semidefiniteness validator.
    """
    n = len(decomposition_second_moment)
    matrices = (decomposition_second_moment, error_second_moment)
    if not n or any(len(a) != n or any(len(row) != n for row in a) for a in matrices):
        raise ValueError("two nonempty square matrices of equal size required")
    for a in matrices:
        for i in range(n):
            for j in range(n):
                if not math.isfinite(a[i][j]) or not math.isclose(a[i][j], a[j][i]):
                    raise ValueError("finite symmetric second-moment matrices required")
    result = math.fsum(decomposition_second_moment[i][j] * error_second_moment[i][j]
                       for i in range(n) for j in range(n))
    if result < 0:
        raise ValueError("negative second moment: input matrices are not valid")
    return result


def signed_digit_second_moment(base, signs):
    """Hypothetical iid digits uniform on {-B/2,...,B/2-1}, with ring signs.

    Their mean is -1/2, not zero. This supplies a counterexample to silently
    imposing a diagonal second-moment matrix, not a model of an actual CMux.
    """
    if not isinstance(base, int) or base < 2 or base & (base - 1):
        raise ValueError("base must be a power of two")
    if not signs or any(s not in (-1, 1) for s in signs):
        raise ValueError("one negacyclic sign per coordinate required")
    variance = (base * base - 1) / 12
    return [[(variance if i == j else 0) + signs[i] * signs[j] / 4
             for j in range(len(signs))] for i in range(len(signs))]


def fixed_weight_second_moment(weights, error_second_moment):
    """Exact quadratic form for fixed decomposition weights."""
    matrix = [[a * b for b in weights] for a in weights]
    return inner_product_second_moment(matrix, error_second_moment)
