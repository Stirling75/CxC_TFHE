"""Column model of product-sum lookup used by the schedule and the failure
estimate (heterogeneous_screen.py)."""
import math
from dataclasses import replace

import baseline_study as base

ANALYSIS_STATUS = "unresolved-trace-to-ggsw-covariance"


def primitive(level=7):
    profile = replace(base.profile_for(8), cbs_level=level)
    return profile, base.est.primitive_vars(
        profile, 4, 2, "linear", "sage", "revhomtrace", "refined-cbs")


def columns(width, capacity, prim, shared_prefix=False):
    """`shared_prefix` models the first candidates shared by full groups that
    start at the same index (used by the failure estimate)."""
    assert 1 <= capacity <= 45
    result = [[] for _ in range(width // 2)]
    for q in range(len(result)):
        for offset in range(0, q + 1, capacity):
            length = min(capacity, q + 1 - offset)
            maximum = 9 * length
            count = (maximum.bit_length() + 1) // 2
            assert count * (maximum + 1) <= 2048
            if shared_prefix and length == capacity and offset + capacity <= q:
                # Full groups that start at the same index share their first
                # candidates, whose two external products on the selected path
                # carry the same gadget error in every such group.
                shared = {("sum-gadget", q, offset): math.sqrt((4 * length - 2) * prim.cmux_gadget_var),
                          ("prefix-gadget", offset): math.sqrt(2 * prim.cmux_gadget_var)}
            else:
                shared = {("sum-gadget", q, offset): math.sqrt(4 * length * prim.cmux_gadget_var)}
            for i in range(offset, offset + length):
                for key in (("X", i), ("Y", q - i)):
                    shared[key] = shared.get(key, 0) + 2 * math.sqrt(prim.cmux_key_var)
            variance = sum(a * a for a in shared.values())
            for t in range(min(count, len(result) - q)):
                term = base.est.Term(min(3, maximum // 4**t), variance,
                    "fused-product-sum", False, private_var=0, shared=dict(shared))
                term.routing = [q, offset, t]
                result[q + t].append(term)
    return result


def public_digits(maximum):
    """Radix-4 digits needed for group sums in [0, maximum]."""
    return max(1, (maximum.bit_length() + 1) // 2)


def public_groups(scalar, nonzero_cap, ring=2048):
    """Scalar-aware groups for a public second operand (radix-4 digits `scalar`).

    Column q collects the pairs (i, q-i) with scalar[q-i] != 0 in order of i and
    closes a group when another product would exceed `nonzero_cap` rotations or
    the table of sums up to 3*sum(b) would not fit in the ring.
    """
    d = len(scalar)
    groups = []
    for q in range(d):
        column, current, maximum = [], [], 0
        for i in range(q + 1):
            b = scalar[q - i]
            if b == 0:
                continue
            nxt = maximum + 3 * b
            if current and (len(current) == nonzero_cap
                            or public_digits(nxt) * (nxt + 1) > ring):
                column.append(current)
                current, maximum, nxt = [], 0, 3 * b
            current.append(i)
            maximum = nxt
        if current:
            column.append(current)
        groups.append(column)
    return groups


def public_columns(scalar, groups, prim):
    """Product terms of the ciphertext-plaintext product-sum lookup.

    Each product is one external product with the low selector and one CMux
    with the high selector of the encrypted digit, so a group of m products
    carries 2m gadget errors and two selector-key errors per encrypted digit.
    """
    d = len(scalar)
    result = [[] for _ in range(d)]
    for q, column in enumerate(groups):
        for group in column:
            maximum = 3 * sum(scalar[q - i] for i in group)
            count = public_digits(maximum)
            assert count * (maximum + 1) <= 2048
            shared = {("sum-gadget", q, group[0]): math.sqrt(2 * len(group) * prim.cmux_gadget_var)}
            for i in group:
                shared[("X", i)] = 2 * math.sqrt(prim.cmux_key_var)
            variance = sum(a * a for a in shared.values())
            for t in range(min(count, d - q)):
                term = base.est.Term(min(3, maximum // 4**t), variance,
                    "public-product-sum", False, private_var=0, shared=dict(shared))
                term.routing = [q, group[0], t]
                result[q + t].append(term)
    return result
