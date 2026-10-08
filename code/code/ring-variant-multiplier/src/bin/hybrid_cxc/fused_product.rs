use super::*;
use dyn_stack::{PodBuffer, PodStack};

type Selectors = FourierGgswCiphertextList<Vec<tfhe_fft::c64>>;

#[path = "fused_reuse.rs"]
mod reuse;

#[cfg(feature = "noise-audit")]
#[path = "fused_trace.rs"]
pub(super) mod trace;

#[derive(Default, Clone, Copy)]
pub(super) struct Work {
    pub(super) cmux: usize,
    pub(super) direct: usize,
    pub(super) cache_entries: usize,
    pub(super) cache_hits: usize,
}

impl Work {
    fn add(&mut self, other: Self) {
        self.cmux += other.cmux;
        self.direct += other.direct;
        self.cache_entries += other.cache_entries;
        self.cache_hits += other.cache_hits;
    }
}

pub(super) struct Evaluation {
    pub(super) columns: Vec<Vec<BoundedLweTerm>>,
    pub(super) elapsed_ms: f64,
    pub(super) work: Work,
}

pub(super) fn evaluate(lhs: &[Selectors], rhs: &[Selectors], params: BenchParams) -> Result<Evaluation> {
    match std::env::var("FUSED_KERNEL").as_deref() {
        Ok("reuse") => reuse::evaluate(lhs, rhs, params, false),
        Ok("reuse-cache") => reuse::evaluate(lhs, rhs, params, true),
        Ok("tree") | Err(_) => {
            // CMuxes are counted at execution time by `record_cmux`.
            let (columns, elapsed_ms) = evaluate_tree(lhs, rhs, params)?;
            Ok(Evaluation { columns, elapsed_ms, work: Work::default() })
        }
        Ok(value) => bail!("unsupported FUSED_KERNEL={value}"),
    }
}

pub(super) fn evaluate_public(lhs: &[Selectors], scalar: &[u64], groups: Option<&[Vec<Vec<usize>>]>,
                              params: BenchParams) -> Result<Evaluation> {
    match groups {
        Some(groups) => reuse::evaluate_scalar_aware(lhs, scalar, groups, params),
        None => reuse::evaluate_public(lhs, scalar, params),
    }
}

/// Radix-4 digits needed for sums in [0, maximum].
fn public_digits(maximum: usize) -> usize {
    ((usize::BITS - maximum.leading_zeros()) as usize + 1).max(2) / 2
}

/// Table of the radix-4 digits of every sum in [0, maximum], `public_digits` apart.
fn sum_table(maximum: usize, params: BenchParams) -> GlweCiphertextOwned<u64> {
    let stride = public_digits(maximum);
    assert!(stride * (maximum + 1) <= params.polynomial_size.0);
    let mut coefficients = vec![0; params.polynomial_size.0];
    for s in 0..=maximum {
        for t in 0..stride {
            coefficients[stride * s + t] = ((s >> (2 * t)) & 3) as u64 * encoding_delta(4);
        }
    }
    allocate_and_trivially_encrypt_new_glwe_ciphertext(
        params.glwe_dimension.to_glwe_size(),
        &PlaintextList::from_container(coefficients),
        params.ciphertext_modulus,
    )
}

struct Scratch {
    fft: Fft,
    memory: PodBuffer,
    nodes: Vec<GlweCiphertextOwned<u64>>,
}

fn output_digits(length: usize) -> usize {
    ((9 * length).ilog2() as usize + 2) / 2
}

fn rotation(index: usize, stride: usize) -> usize {
    let a = decode_refined_digit((index >> 3) & 1, (index >> 2) & 1);
    let b = decode_refined_digit((index >> 1) & 1, index & 1);
    stride * a * b
}

fn public_table(length: usize, params: BenchParams) -> GlweCiphertextOwned<u64> {
    let maximum = 9 * length;
    let stride = output_digits(length);
    assert!(stride * (maximum + 1) <= params.polynomial_size.0);
    let mut coefficients = vec![0; params.polynomial_size.0];
    for s in 0..=maximum {
        for t in 0..stride {
            coefficients[stride * s + t] = ((s >> (2 * t)) & 3) as u64 * encoding_delta(4);
        }
    }
    allocate_and_trivially_encrypt_new_glwe_ciphertext(
        params.glwe_dimension.to_glwe_size(),
        &PlaintextList::from_container(coefficients),
        params.ciphertext_modulus,
    )
}

impl Scratch {
    fn new(params: BenchParams) -> Self {
        let fft = Fft::new(params.polynomial_size);
        let memory = PodBuffer::try_new(cmux_assign_mem_optimized_requirement::<u64>(
            params.glwe_dimension.to_glwe_size(), params.polynomial_size, fft.as_view(),
        )).expect("CMux scratch allocation");
        Self { fft, memory, nodes: (0..16).map(|_| zero_glwe(params)).collect() }
    }

    fn accumulate(&mut self, state: &mut GlweCiphertextOwned<u64>,
                  a: &Selectors, b: &Selectors, stride: usize) {
        for (index, node) in self.nodes.iter_mut().enumerate() {
            monomial_div_into(node, state, MonomialDegree(rotation(index, stride)));
        }
        let mut count = 16;
        // Eliminate index bits from most to least significant, preserving
        // the refined two-selector recoding of each native radix block.
        for (bits, bit) in [(a, 0), (a, 1), (b, 0), (b, 1)] {
            let half = count / 2;
            let (zero, one) = self.nodes[..count].split_at_mut(half);
            let selector = bits.as_view().into_ggsw_iter().nth(bit).expect("radix selector");
            for (zero, one) in zero.iter_mut().zip(one) {
                cmux_assign_mem_optimized::<u64, _, _, _>(zero, one, &selector,
                    self.fft.as_view(), &mut PodStack::new(&mut self.memory));
                record_cmux();
            }
            count = half;
        }
        std::mem::swap(state, &mut self.nodes[0]);
    }
}

pub(super) fn evaluate_tree(lhs: &[Selectors], rhs: &[Selectors], params: BenchParams)
    -> Result<(Vec<Vec<BoundedLweTerm>>, f64)> {
    let started = Instant::now();
    let d = lhs.len();
    let capacity = crate::config::fused_products_per_group();
    assert_eq!(d, rhs.len());
    assert!((1..=45).contains(&capacity));
    let tables: Vec<_> = (1..=capacity.min(d)).map(|r| public_table(r, params)).collect();
    let jobs: Vec<_> = (0..d).flat_map(|q| (0..=q).step_by(capacity)
        .map(move |offset| (q, offset, capacity.min(q + 1 - offset)))).collect();
    let output_size = params.glwe_dimension.to_equivalent_lwe_dimension(params.polynomial_size).to_lwe_size();
    let query = |scratch: &mut Scratch, (q, offset, length)| {
        let stride = output_digits(length);
        let mut state = tables[length - 1].clone();
        for i in offset..offset + length {
            scratch.accumulate(&mut state, &lhs[i], &rhs[q - i], stride);
        }
        let mut digits = extract_clean_digits(&state, q, (q, offset), stride, d, output_size, params);
        for (_, term) in &mut digits {
            let t = term.source.unwrap().2;
            term.bound = ((9 * length) >> (2 * t)).min(3) as u64;
        }
        digits
    };
    let cells: Vec<_> = if parallel_cmux_cells() {
        jobs.into_par_iter().map_init(|| Scratch::new(params), query).collect()
    } else {
        let mut scratch = Scratch::new(params);
        jobs.into_iter().map(|job| query(&mut scratch, job)).collect()
    };
    let mut columns: Vec<Vec<_>> = (0..d).map(|_| Vec::new()).collect();
    for cell in cells {
        for (q, term) in cell { columns[q].push(term); }
    }
    Ok((columns, duration_ms(started.elapsed())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_sum_and_selector_rotation_matches_the_public_table() {
        let params = BenchParams::from_env(ProductVariant::Hybrid8x8).unwrap();
        for length in 1..=45 {
            let table = public_table(length, params);
            let stride = output_digits(length);
            for sum in 0..=9 * length {
                let mut moved = zero_glwe(params);
                monomial_div_into(&mut moved, &table, MonomialDegree(stride * sum));
                let result: u64 = (0..stride).map(|t|
                    (moved.get_body().as_ref()[t] / encoding_delta(4)) << (2 * t)).sum();
                assert_eq!(result as usize, sum);
            }
        }
        for index in 0..16 {
            let bits: Vec<_> = (0..4).map(|i| (index >> (3-i)) & 1).collect();
            let a = (bits[0] ^ bits[1]) + 2 * bits[1];
            let b = (bits[2] ^ bits[3]) + 2 * bits[3];
            assert_eq!(rotation(index, 4), 4*a*b);
        }
    }
}
