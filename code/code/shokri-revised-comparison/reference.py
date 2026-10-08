"""Plaintext execution of 2026/810 (July 30), Algorithm 1 and Eq. (12).

This is an independent semantic reference, not the authors' implementation.
It does not perform cryptography, estimate failure, or measure FHE latency.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProductBit:
    left: int
    right: int
    bit: int
    column: int


@dataclass(frozen=True)
class Compressor:
    column: int
    inputs: tuple[int, ...]
    parity: int
    carry: int | None


@dataclass(frozen=True)
class Layer:
    target: int
    before: tuple[tuple[int, ...], ...]
    jobs: tuple[Compressor, ...]
    after: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class Plan:
    width: int
    products: tuple[ProductBit, ...]
    initial: tuple[tuple[int, ...], ...]
    layers: tuple[Layer, ...]
    final: tuple[tuple[int, ...], ...]

    def counts(self):
        active = [len(col) for col in self.final[8:]]
        needs_terminal = any(h > 1 for h in active)
        terminal_reads = sum(min(h, 2) for h in active) if needs_terminal else 0
        # The paper reports three CxC segments at W=512, one at smaller widths.
        # This is a published plan, not a fresh derivation of the noise budget.
        segments = 3 if self.width == 512 else 1
        compressors = sum(len(layer.jobs) for layer in self.layers)
        full = sum(len(job.inputs) == 3 for layer in self.layers for job in layer.jobs)
        cbs = self.width + terminal_reads + segments - 1
        emission = self.width // 2
        return {
            "width": self.width,
            "product_families": (self.width // 8) * (self.width // 8 + 1) // 2,
            "product_bits": len(self.products),
            "input_grouped_cbs": self.width,
            "product_cmux_standard_scalar_vp": len(self.products) * (63 + 10),
            "dadda_layers": len(self.layers),
            "dadda_full": full,
            "dadda_half": compressors - full,
            "dadda_pbs_manylut": compressors,
            "terminal_binary_cbs": terminal_reads + segments - 1,
            "published_segments": segments,
            "emission_pbs": emission,
            "standalone_pbs": compressors + emission,
            "cbs_total": cbs,
            "total_blind_rotations": compressors + emission + cbs,
            "homtrace": 4 * (2 * self.width + terminal_reads + segments - 1),
            "terminal_cmux_gate_count": None,
        }


def frozen(columns):
    return tuple(tuple(column) for column in columns)


def make_plan(width):
    if width < 8 or width > 512 or width % 8:
        raise ValueError("width must be a multiple of 8 in [8, 512]")
    products = []
    columns = [[] for _ in range(width)]
    for left in range(width // 8):
        for right in range(width // 8 - left):
            offset = 8 * (left + right)
            for bit in range(min(16, width - offset)):
                columns[offset + bit].append(len(products))
                products.append(ProductBit(left, right, bit, offset + bit))
    initial = frozen(columns)
    next_id = len(products)
    targets = []
    target = 3
    while target < max(map(len, columns)):
        targets.append(target)
        target = 3 * target // 2

    layers = []
    for target in reversed(targets):
        before = frozen(columns)
        after = [[] for _ in range(width)]
        jobs = []
        for column, old in enumerate(before):
            # Incoming carries count toward the next height but cannot be read
            # by another job in this layer (paper Eq. 12).
            incoming = len(after[column])
            excess = max(len(old) + incoming - target, 0)
            full, half = divmod(excess, 2)
            cursor = 0
            for arity in [3] * full + [2] * half:
                inputs = old[cursor:cursor + arity]
                if len(inputs) != arity:
                    raise AssertionError("Dadda layer would consume its own carries")
                cursor += arity
                parity = next_id
                next_id += 1
                carry = next_id if column + 1 < width else None
                next_id += int(carry is not None)
                after[column].append(parity)
                if carry is not None:
                    after[column + 1].append(carry)
                jobs.append(Compressor(column, inputs, parity, carry))
            after[column].extend(old[cursor:])
            if len(after[column]) > target:
                raise AssertionError("Dadda target not reached")
        columns = after
        layers.append(Layer(target, before, tuple(jobs), frozen(after)))
    return Plan(width, tuple(products), initial, tuple(layers), frozen(columns))


def weighted_value(columns, values, width):
    return sum(sum(values[i] for i in col) << c for c, col in enumerate(columns)) % (1 << width)


def terminal_reference(sums):
    """Two-state suffix evaluation of Eq. (6), not a CMux gate realization.

    Direct four-entry selection evaluates the plaintext function of each state.
    No cryptographic gate count or terminal noise is inferred from this function.
    """
    lo = [s & 1 for s in sums]
    hi = [s >> 1 for s in sums]
    states = (0, 0)
    for c in range(len(sums) - 1, -1, -1):
        previous_high = hi[c - 1] if c else 0
        candidates = []
        for incoming in (0, 1):
            total = lo[c] + previous_high + incoming
            candidates.append(states[total >> 1] | ((total & 1) << c))
        states = tuple(candidates)
    return states[0]


def evaluate(plan, x, y):
    if not (0 <= x < (1 << plan.width) and 0 <= y < (1 << plan.width)):
        raise ValueError("operands must be unsigned width-bit integers")
    values = {}
    for identifier, term in enumerate(plan.products):
        left = (x >> (8 * term.left)) & 255
        right = (y >> (8 * term.right)) & 255
        values[identifier] = (left * right >> term.bit) & 1
    expected = (x * y) % (1 << plan.width)
    if weighted_value(plan.initial, values, plan.width) != expected:
        raise AssertionError("product routing changes the weighted value")
    for layer in plan.layers:
        outputs = {}
        for job in layer.jobs:
            total = sum(values[i] for i in job.inputs)
            outputs[job.parity] = total & 1
            if job.carry is not None:
                outputs[job.carry] = total >> 1
        values.update(outputs)
        if weighted_value(layer.after, values, plan.width) != expected:
            raise AssertionError("compression changes the weighted value")
    sums = [sum(values[i] for i in column) for column in plan.final]
    if any(s < 0 or s > 3 for s in sums):
        raise AssertionError("terminal plaintext outside [0, 3]")
    result = terminal_reference(sums)
    chunks = [(result >> c) & 3 for c in range(0, plan.width, 2)]
    if result != expected:
        raise AssertionError("terminal result does not equal XY modulo 2^W")
    return chunks
