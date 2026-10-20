use tfhe::core_crypto::algorithms::polynomial_algorithms::polynomial_wrapping_mul;
use tfhe::core_crypto::prelude::*;

pub const NATIVE_DELTA: u64 = 1 << 59;

pub fn bits_needed(bound: u64) -> usize {
    (64 - bound.leading_zeros()) as usize
}

pub fn precision(digits: usize) -> usize {
    // Sign-based bit extraction accepts the full plaintext interval.
    bits_needed(9 * digits as u64)
}

pub fn public_pbs_counts(d: usize) -> (usize, usize, usize) {
    let mut columns = vec![0usize; d];
    let mut bit_extraction = 0;
    for q in 0..d {
        let bits = bits_needed(9 * (q + 1) as u64).min(2 * (d - q));
        bit_extraction += bits;
        for j in 0..bits.div_ceil(2) {
            columns[q + j] += 1;
        }
    }
    let mut normalization = 0;
    for q in 0..d {
        while columns[q] > 4 {
            columns[q] -= 3;
            normalization += 1;
            if q + 1 < d {
                columns[q + 1] += 1;
                normalization += 1;
            }
        }
        normalization += 1;
        if q + 1 < d && columns[q] > 1 {
            columns[q + 1] += 1;
            normalization += 1;
        }
    }
    (2 * d, bit_extraction, normalization)
}

pub fn bounded_group_length(degrees: &[usize], capacity: usize) -> usize {
    let mut sum = 0;
    degrees
        .iter()
        .take_while(|&&degree| {
            if sum + degree > capacity {
                false
            } else {
                sum += degree;
                true
            }
        })
        .count()
}

pub fn range_aware_pbs_counts(d: usize, domain: usize) -> (usize, usize, usize) {
    let mut columns = vec![Vec::new(); d];
    let mut bit_extraction = 0;
    for q in 0..d {
        let bound = 9 * (q + 1);
        let bits = bits_needed(bound as u64).min(2 * (d - q));
        bit_extraction += bits;
        for j in 0..bits.div_ceil(2) {
            columns[q + j].push((bound >> (2 * j)).min(3));
        }
    }
    let mut normalization = 0;
    for q in 0..d {
        while columns[q].iter().sum::<usize>() >= domain {
            let take = bounded_group_length(&columns[q], domain - 1);
            assert!(take >= 2);
            let total = columns[q].drain(..take).sum::<usize>();
            columns[q].push(total.min(3));
            normalization += 1;
            if q + 1 < d {
                columns[q + 1].push(total / 4);
                normalization += 1;
            }
        }
        let total = columns[q].iter().sum::<usize>();
        normalization += 1;
        if q + 1 < d && total >= 4 {
            columns[q + 1].push(total / 4);
            normalization += 1;
        }
    }
    (2 * d, bit_extraction, normalization)
}

#[derive(Debug)]
pub struct NormalizerTask {
    pub inputs: Vec<usize>,
    pub low: usize,
    pub carry: Option<usize>,
}

pub struct NormalizerPlan {
    pub stages: Vec<Vec<NormalizerTask>>,
    pub outputs: Vec<usize>,
    pub value_count: usize,
    pub initial_count: usize,
}

impl NormalizerPlan {
    fn add(
        &mut self,
        inputs: Vec<usize>,
        carry: bool,
        levels: &mut Vec<usize>,
    ) -> (usize, Option<usize>) {
        let level = inputs.iter().map(|&i| levels[i]).max().unwrap() + 1;
        while self.stages.len() < level {
            self.stages.push(Vec::new());
        }
        let low = levels.len();
        levels.push(level);
        let carry = carry.then(|| {
            let index = levels.len();
            levels.push(level);
            index
        });
        self.stages[level - 1].push(NormalizerTask { inputs, low, carry });
        self.value_count = levels.len();
        (low, carry)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn pbs_count(&self) -> usize {
        self.stages
            .iter()
            .flatten()
            .map(|task| 1 + usize::from(task.carry.is_some()))
            .sum()
    }

    pub fn ks_count(&self) -> usize {
        self.stages.iter().map(Vec::len).sum()
    }
}

// Preserve every group and within-group input order of the serial normalizer.
// Only independent nodes are moved into the same execution stage.
pub fn normalizer_plan(d: usize, domain: usize) -> NormalizerPlan {
    assert!([8, 16].contains(&domain));
    let mut degrees = vec![Vec::new(); d];
    for q in 0..d {
        let bound = 9 * (q + 1);
        let bits = bits_needed(bound as u64).min(2 * (d - q));
        for j in 0..bits.div_ceil(2) {
            degrees[q + j].push((bound >> (2 * j)).min(3));
        }
    }
    let mut columns = vec![Vec::new(); d];
    let mut levels = Vec::new();
    for (column, bounds) in columns.iter_mut().zip(&degrees) {
        for _ in bounds {
            column.push(levels.len());
            levels.push(0);
        }
    }
    let mut plan = NormalizerPlan {
        stages: Vec::new(),
        outputs: Vec::new(),
        value_count: levels.len(),
        initial_count: levels.len(),
    };
    for q in 0..d {
        while degrees[q].iter().sum::<usize>() >= domain {
            let take = bounded_group_length(&degrees[q], domain - 1);
            assert!(take >= 2);
            let total: usize = degrees[q].drain(..take).sum();
            let inputs = columns[q].drain(..take).collect();
            let (low, carry) = plan.add(inputs, q + 1 < d, &mut levels);
            columns[q].push(low);
            degrees[q].push(total.min(3));
            if let Some(carry) = carry {
                columns[q + 1].push(carry);
                degrees[q + 1].push(total / 4);
            }
        }
        let total: usize = degrees[q].iter().sum();
        let (low, carry) = plan.add(columns[q].clone(), q + 1 < d && total >= 4, &mut levels);
        plan.outputs.push(low);
        if let Some(carry) = carry {
            columns[q + 1].push(carry);
            degrees[q + 1].push(total / 4);
        }
    }
    plan
}

pub fn convolution(a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut c = vec![0; a.len() + b.len() - 1];
    for (i, &x) in a.iter().enumerate() {
        for (j, &y) in b.iter().enumerate() {
            c[i + j] += x * y;
        }
    }
    c
}

pub fn mul64(a: &[u64], b: &[u64]) -> Vec<u64> {
    assert_eq!(a.len(), b.len());
    let mut out = Polynomial::new(0, PolynomialSize(a.len()));
    polynomial_wrapping_mul(
        &mut out,
        &Polynomial::from_container(a),
        &Polynomial::from_container(b),
    );
    out.into_container()
}

fn mul128(a: &[u64], b: &[u64]) -> Vec<u128> {
    assert_eq!(a.len(), b.len());
    // Centered lifts match CLOT's noise analysis; do not zero-extend u64 masks.
    let a: Vec<u128> = a.iter().map(|&x| (x as i64 as i128) as u128).collect();
    let b: Vec<u128> = b.iter().map(|&x| (x as i64 as i128) as u128).collect();
    let mut out = Polynomial::new(0, PolynomialSize(a.len()));
    polynomial_wrapping_mul(
        &mut out,
        &Polynomial::from_container(a),
        &Polynomial::from_container(b),
    );
    out.into_container()
}

pub fn round_scale(x: u128, shift: usize) -> u64 {
    assert!((1..64).contains(&shift));
    ((x >> shift) as u64).wrapping_add(((x >> (shift - 1)) & 1) as u64)
}

// k=1 specialization of CLOT Algorithm 1. Products live modulo q^2, not q.
pub fn tensor(a: &[u64], b: &[u64], shift: usize) -> [Vec<u64>; 3] {
    assert_eq!(a.len(), b.len());
    let n = a.len() / 2;
    assert_eq!(a.len(), 2 * n);
    let t = mul128(&a[..n], &b[..n]);
    let ab = mul128(&a[..n], &b[n..]);
    let ba = mul128(&a[n..], &b[..n]);
    let bb = mul128(&a[n..], &b[n..]);
    [
        t.into_iter().map(|x| round_scale(x, shift)).collect(),
        ab.into_iter()
            .zip(ba)
            .map(|(x, y)| round_scale(x.wrapping_add(y), shift))
            .collect(),
        bb.into_iter().map(|x| round_scale(x, shift)).collect(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::{BigInt, BigUint, Sign};
    use rand::{Rng, SeedableRng};

    #[test]
    fn range_aware_normalizer_preserves_products_and_public_bounds() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(20260919);
        for w in [8, 16, 32, 64, 128, 256] {
            let d = w / 2;
            for domain in [8, 16] {
                let plan = normalizer_plan(d, domain);
                for case in 0..100 {
                    let a: Vec<u64> = (0..d)
                        .map(|_| if case == 0 { 3 } else { rng.gen_range(0..4) })
                        .collect();
                    let b: Vec<u64> = (0..d)
                        .map(|_| if case == 0 { 3 } else { rng.gen_range(0..4) })
                        .collect();
                    let c = convolution(&a, &b);
                    let mut values = vec![Vec::new(); d];
                    let mut degrees = vec![Vec::new(); d];
                    for q in 0..d {
                        let bound = 9 * (q + 1);
                        let bits = bits_needed(bound as u64).min(2 * (d - q));
                        for j in 0..bits.div_ceil(2) {
                            values[q + j].push((c[q] >> (2 * j)) & 3);
                            degrees[q + j].push((bound >> (2 * j)).min(3));
                        }
                    }
                    let mut graph: Vec<Option<u64>> =
                        values.iter().flatten().copied().map(Some).collect();
                    assert_eq!(graph.len(), plan.initial_count);
                    graph.resize(plan.value_count, None);
                    for stage in &plan.stages {
                        let results: Vec<_> = stage
                            .iter()
                            .map(|task| {
                                let sum: u64 = task.inputs.iter().map(|&i| graph[i].unwrap()).sum();
                                assert!(sum < domain as u64);
                                (sum % 4, sum / 4)
                            })
                            .collect();
                        for (task, (low, carry)) in stage.iter().zip(results) {
                            assert!(graph[task.low].is_none());
                            graph[task.low] = Some(low);
                            if let Some(i) = task.carry {
                                assert!(graph[i].is_none());
                                graph[i] = Some(carry);
                            }
                        }
                    }
                    let mut output = Vec::new();
                    let mut calls = 0;
                    for q in 0..d {
                        while degrees[q].iter().sum::<usize>() >= domain {
                            let take = bounded_group_length(&degrees[q], domain - 1);
                            assert!(take >= 2);
                            let value: u64 = values[q].drain(..take).sum();
                            let bound: usize = degrees[q].drain(..take).sum();
                            assert!(value <= bound as u64 && bound < domain);
                            values[q].push(value % 4);
                            degrees[q].push(bound.min(3));
                            calls += 1;
                            if q + 1 < d {
                                values[q + 1].push(value / 4);
                                degrees[q + 1].push(bound / 4);
                                calls += 1;
                            }
                        }
                        let value: u64 = values[q].iter().sum();
                        let bound: usize = degrees[q].iter().sum();
                        assert!(value <= bound as u64 && bound < domain);
                        output.push(value % 4);
                        calls += 1;
                        if q + 1 < d && bound >= 4 {
                            values[q + 1].push(value / 4);
                            degrees[q + 1].push(bound / 4);
                            calls += 1;
                        }
                    }
                    let integer = |v: &[u64]| {
                        v.iter()
                            .rev()
                            .fold(BigUint::from(0u8), |a, x| (a << 2usize) + x)
                    };
                    assert_eq!(
                        integer(&output),
                        integer(&a) * integer(&b) % (BigUint::from(1u8) << w)
                    );
                    assert_eq!(calls, range_aware_pbs_counts(d, domain).2);
                    assert_eq!(calls, plan.pbs_count());
                    assert_eq!(
                        output,
                        plan.outputs
                            .iter()
                            .map(|&i| graph[i].unwrap())
                            .collect::<Vec<_>>()
                    );
                }
            }
        }
    }

    #[test]
    fn dependency_graph_retains_serial_groups_and_input_order() {
        let plan = normalizer_plan(4, 8);
        assert_eq!(plan.initial_count, 8);
        assert_eq!(plan.stages.len(), 3);
        assert_eq!(plan.pbs_count(), 7);
        assert_eq!(plan.ks_count(), 5);
        assert_eq!(plan.outputs, vec![8, 9, 11, 14]);
        let mut tasks: Vec<_> = plan.stages.iter().flatten().collect();
        tasks.sort_by_key(|task| task.low);
        let expected = [
            (vec![0], 8, None),
            (vec![1, 2], 9, Some(10)),
            (vec![3, 4, 10], 11, Some(12)),
            (vec![5, 6, 7], 13, None),
            (vec![12, 13], 14, None),
        ];
        for (task, (inputs, low, carry)) in tasks.iter().zip(expected) {
            assert_eq!(task.inputs, inputs);
            assert_eq!(task.low, low);
            assert_eq!(task.carry, carry);
        }
    }

    #[test]
    fn widened_tensor_matches_signed_big_integer_oracle() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(20260918);
        for n in [2, 8, 64] {
            for shift in [52, 59, 60] {
                let a: Vec<u64> = (0..2 * n).map(|_| rng.gen()).collect();
                let b: Vec<u64> = (0..2 * n).map(|_| rng.gen()).collect();
                let actual = tensor(&a, &b, shift);
                for part in 0..3 {
                    let pairs = match part {
                        0 => vec![(0, 0)],
                        1 => vec![(0, n), (n, 0)],
                        _ => vec![(n, n)],
                    };
                    let mut exact = vec![BigInt::from(0); n];
                    for (oa, ob) in pairs {
                        for i in 0..n {
                            for j in 0..n {
                                let p =
                                    BigInt::from(a[oa + i] as i64) * BigInt::from(b[ob + j] as i64);
                                if i + j < n {
                                    exact[i + j] += p;
                                } else {
                                    exact[i + j - n] -= p;
                                }
                            }
                        }
                    }
                    let q = BigInt::from(1u128 << 64);
                    for (x, got) in exact.into_iter().zip(&actual[part]) {
                        let rounded: BigInt = (x + (BigInt::from(1) << (shift - 1))) >> shift;
                        let reduced: BigInt = ((rounded % &q) + &q) % &q;
                        let (sign, digits) = reduced.to_u64_digits();
                        assert_ne!(sign, Sign::Minus);
                        assert_eq!(*got, digits.first().copied().unwrap_or(0));
                    }
                }
            }
        }
    }

    #[test]
    fn radix_convolution_matches_big_integer_product_through_256_bits() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        for w in [8, 16, 32, 64, 128, 256] {
            let d = w / 2;
            for case in 0..100 {
                let a: Vec<u64> = (0..d)
                    .map(|_| if case == 0 { 3 } else { rng.gen_range(0..4) })
                    .collect();
                let b: Vec<u64> = (0..d)
                    .map(|_| if case == 0 { 3 } else { rng.gen_range(0..4) })
                    .collect();
                let integer = |x: &[u64]| {
                    x.iter()
                        .rev()
                        .fold(BigUint::from(0u8), |v, x| (v << 2usize) + x)
                };
                let c = convolution(&a, &b);
                assert!(c.iter().all(|&x| x < 1 << precision(d)));
                let mut carry = 0;
                let out: Vec<u64> = c[..d]
                    .iter()
                    .map(|&x| {
                        let s = x + carry;
                        carry = s / 4;
                        s % 4
                    })
                    .collect();
                assert_eq!(
                    integer(&out),
                    (integer(&a) * integer(&b)) % (BigUint::from(1u8) << w)
                );
                let mut columns = vec![Vec::<u64>::new(); d];
                for q in 0..d {
                    let bits = bits_needed(9 * (q + 1) as u64).min(2 * (d - q));
                    let mut residual = c[q];
                    for j in (0..bits).step_by(2) {
                        let low = (residual >> j) & 1;
                        residual -= low << j;
                        let high = if j + 1 < bits {
                            (residual >> (j + 1)) & 1
                        } else {
                            0
                        };
                        residual -= high << (j + 1);
                        columns[q + j / 2].push(low + 2 * high);
                    }
                }
                let mut restored = Vec::new();
                let mut normalization_pbs = 0;
                for q in 0..d {
                    while columns[q].len() > 4 {
                        let sum: u64 = columns[q].drain(..4).sum();
                        assert!(sum <= 12);
                        columns[q].push(sum % 4);
                        normalization_pbs += 1;
                        if q + 1 < d {
                            columns[q + 1].push(sum / 4);
                            normalization_pbs += 1;
                        }
                    }
                    let sum: u64 = columns[q].iter().sum();
                    assert!(sum <= 12);
                    restored.push(sum % 4);
                    normalization_pbs += 1;
                    if q + 1 < d && columns[q].len() > 1 {
                        columns[q + 1].push(sum / 4);
                        normalization_pbs += 1;
                    }
                }
                assert_eq!(restored, out);
                assert_eq!(normalization_pbs, public_pbs_counts(d).2);
            }
        }
    }

    #[test]
    fn rounding_handles_negative_values_and_wraparound() {
        assert_eq!(round_scale((-33i128) as u128, 4), u64::MAX - 1);
        assert_eq!(round_scale((-8i128) as u128, 4), 0);
        assert_eq!(round_scale(8, 4), 1);
        assert_eq!(precision(128), 11);
    }

    #[test]
    fn public_schedule_has_expected_small_case() {
        assert_eq!(public_pbs_counts(4), (8, 15, 6));
    }

    #[test]
    fn reducing_tensor_mod_q_before_rescaling_is_not_valid() {
        let delta = 1u64 << 59;
        let a = [0, 0, 3 * delta, 0];
        let b = [0, 0, 2 * delta, 0];
        assert_eq!(tensor(&a, &b, 59)[2][0], 6 * delta);
        assert_ne!(a[2].wrapping_mul(b[2]) >> 59, 6 * delta);
    }

    #[test]
    fn sign_extraction_supports_the_full_unpadded_domain() {
        let p = 11;
        let delta = 1u64 << (64 - p);
        for c in 0u64..1 << p {
            let mut residual = c * delta;
            let mut reconstructed = 0;
            for j in 0..p {
                let parity_phase = residual
                    .wrapping_mul(1 << (p - j - 1))
                    .wrapping_add(1 << 62);
                // A constant negative negacyclic LUT changes sign at q/2.
                let bit = u64::from(parity_phase >= 1 << 63);
                reconstructed += bit << j;
                residual = residual.wrapping_sub(bit * (delta << j));
            }
            assert_eq!(reconstructed, c);
            assert_eq!(residual, 0);
        }
    }
}
