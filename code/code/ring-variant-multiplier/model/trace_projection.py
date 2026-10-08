"""Exact linear trace projections and binary-secret second moments (k=1).

The algebraic identities are independent of any Gaussian or FFT noise model.
Applying them to FFT noise requires a separately stated component-noise model.
"""
from fractions import Fraction


def require_degree(n):
    if not isinstance(n, int) or n < 2 or n & (n-1):
        raise ValueError("power-of-two degree >= 2 required")


def automorphism(vector, exponent):
    n = len(vector)
    require_degree(n)
    if exponent % 2 == 0:
        raise ValueError("automorphism exponent must be odd")
    out = [0] * n
    for j, value in enumerate(vector):
        k = (j * exponent) % (2*n)
        out[k % n] = value if k < n else -value
    return out


def remaining_trace(vector, completed_stages):
    n = len(vector)
    require_degree(n)
    log_n = n.bit_length()-1
    if not 1 <= completed_stages <= log_n:
        raise ValueError("completed stages outside trace")
    out = list(map(Fraction, vector))
    for r in range(completed_stages+1, log_n+1):
        other = automorphism(out, 2**r+1)
        out = [(x+y)/2 for x, y in zip(out, other)]
    return out


def binary_product_energy(m, same):
    """E ||A B||_2^2 in R[X]/(X^m+1); A,B iid binary, or B=A."""
    if m == 1:
        return Fraction(1, 2 if same else 4)
    require_degree(m)
    return Fraction(m**3+18*m*m-4*m if same else m**3+9*m*m+2*m, 48)


def projected_row_energy(n, stride):
    """E ||row_t(H_S P_stride H_S)||_2^2, independent of output t.

    P_stride retains coefficients divisible by stride. Splitting S into residue
    classes modulo stride reduces the row energy to one self-product and
    stride-1 products of independent binary polynomials of length n/stride.
    """
    require_degree(n)
    if stride < 1 or n % stride or stride & (stride-1):
        raise ValueError("stride must divide the ring degree and be a power of two")
    m = n // stride
    return binary_product_energy(m, True)+(stride-1)*binary_product_energy(m, False)


def trace_component_fft_energy(n):
    """Postmultiplication multiplier for iid unit-variance mask/body FFT errors.

    At stage r, phase noise is delta_b-H_S delta_a. Its contribution after the
    remaining trace and multiplication by S has energy M/2 + row_energy,
    M=2^r. Different stages and component coordinates are assumed independent
    ONLY when this multiplier is used as a variance.
    """
    require_degree(n)
    return sum(Fraction(2**r, 2)+projected_row_energy(n, n//2**r)
               for r in range(1, n.bit_length()))
