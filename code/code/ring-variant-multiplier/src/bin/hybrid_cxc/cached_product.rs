use super::*;
use dyn_stack::{PodBuffer, PodStack};

type Selectors = FourierGgswCiphertextList<Vec<tfhe_fft::c64>>;

struct Scratch {
    fft: Fft,
    memory: PodBuffer,
    rotated: GlweCiphertextOwned<u64>,
}

impl Scratch {
    fn new(params: BenchParams) -> Self {
        let fft = Fft::new(params.polynomial_size);
        let memory = PodBuffer::try_new(cmux_assign_mem_optimized_requirement::<u64>(
            params.glwe_dimension.to_glwe_size(),
            params.polynomial_size,
            fft.as_view(),
        ))
        .expect("CMux scratch allocation");
        Self {
            fft,
            memory,
            rotated: zero_glwe(params),
        }
    }

    fn select(
        &mut self,
        zero: &mut GlweCiphertextOwned<u64>,
        one: &mut GlweCiphertextOwned<u64>,
        bits: &Selectors,
        bit: usize,
    ) {
        let selector = bits
            .as_view()
            .into_ggsw_iter()
            .nth(bit)
            .expect("radix selector");
        cmux_assign_mem_optimized::<u64, _, _, _>(
            zero,
            one,
            &selector,
            self.fft.as_view(),
            &mut PodStack::new(&mut self.memory),
        );
        record_cmux();
    }

    fn query(
        &mut self,
        seed: &GlweCiphertextOwned<u64>,
        rhs: &[Selectors],
        params: BenchParams,
    ) -> GlweCiphertextOwned<u64> {
        let mut result = seed.clone();
        for round in 0..8 {
            monomial_div_into(
                &mut self.rotated,
                &result,
                MonomialDegree(params.polynomial_size.0 >> (round + 1)),
            );
            let selector = rhs[round % 4]
                .as_view()
                .into_ggsw_iter()
                .nth(round / 4)
                .expect("radix selector");
            cmux_assign_mem_optimized::<u64, _, _, _>(
                &mut result,
                &mut self.rotated,
                &selector,
                self.fft.as_view(),
                &mut PodStack::new(&mut self.memory),
            );
            record_cmux();
        }
        result
    }
}

fn decoded_byte(index: usize) -> usize {
    (0..4)
        .map(|i| {
            let y0 = (index >> (7 - i)) & 1;
            let y1 = (index >> (3 - i)) & 1;
            decode_refined_digit(y0, y1) << (2 * i)
        })
        .sum()
}

fn public_bank(params: BenchParams) -> Vec<GlweCiphertextOwned<u64>> {
    assert_eq!(params.polynomial_size.0, 2048);
    (0..256)
        .map(|left| {
            let mut coefficients = vec![0; 2048];
            for right in 0..256 {
                let product = decoded_byte(left) * decoded_byte(right);
                for t in 0..8 {
                    coefficients[8 * right + t] =
                        ((product >> (2 * t)) & 3) as u64 * encoding_delta(4);
                }
            }
            allocate_and_trivially_encrypt_new_glwe_ciphertext(
                params.glwe_dimension.to_glwe_size(),
                &PlaintextList::from_container(coefficients),
                params.ciphertext_modulus,
            )
        })
        .collect()
}

fn select_left(
    bank: &[GlweCiphertextOwned<u64>],
    lhs: &[Selectors],
    scratch: &mut Scratch,
) -> GlweCiphertextOwned<u64> {
    let mut nodes = bank.to_vec();
    for round in 0..8 {
        let half = nodes.len() / 2;
        let (zero, one) = nodes.split_at_mut(half);
        for (zero, one) in zero.iter_mut().zip(one) {
            scratch.select(zero, one, &lhs[round % 4], round / 4);
        }
        nodes.truncate(half);
    }
    nodes.pop().expect("selected encrypted table")
}

pub(super) fn evaluate(
    lhs: &[Selectors],
    rhs: &[Selectors],
    params: BenchParams,
) -> Result<(Vec<Vec<BoundedLweTerm>>, f64)> {
    assert_eq!(lhs.len(), rhs.len());
    assert_eq!(lhs.len() % 4, 0);
    let started = Instant::now();
    let digits = lhs.len();
    let h = digits / 4;
    let bank = public_bank(params);
    let seeds: Vec<_> = if parallel_cmux_cells() {
        (0..h)
            .into_par_iter()
            .map_init(
                || Scratch::new(params),
                |scratch, u| select_left(&bank, &lhs[4 * u..4 * u + 4], scratch),
            )
            .collect()
    } else {
        let mut scratch = Scratch::new(params);
        (0..h)
            .map(|u| select_left(&bank, &lhs[4 * u..4 * u + 4], &mut scratch))
            .collect()
    };
    let jobs: Vec<_> = (0..h)
        .flat_map(|u| (0..h - u).map(move |v| (u, v)))
        .collect();
    let output_size = params
        .glwe_dimension
        .to_equivalent_lwe_dimension(params.polynomial_size)
        .to_lwe_size();
    let query = |scratch: &mut Scratch, (u, v)| {
        let selected = scratch.query(&seeds[u], &rhs[4 * v..4 * v + 4], params);
        extract_clean_digits(
            &selected,
            4 * (u + v),
            (4 * u, 4 * v),
            8,
            digits,
            output_size,
            params,
        )
    };
    let cells: Vec<_> = if parallel_cmux_cells() {
        jobs.into_par_iter()
            .map_init(|| Scratch::new(params), query)
            .collect()
    } else {
        let mut scratch = Scratch::new(params);
        jobs.into_iter()
            .map(|job| query(&mut scratch, job))
            .collect()
    };
    let mut columns = (0..2 * digits).map(|_| Vec::new()).collect::<Vec<_>>();
    for cell in cells {
        for (q, term) in cell {
            columns[q].push(term);
        }
    }
    Ok((columns, duration_ms(started.elapsed())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_byte_product_has_the_correct_bank_and_offset() {
        let params = BenchParams::from_env(ProductVariant::Hybrid8x8).unwrap();
        let bank = public_bank(params);
        for (a, polynomial) in bank.iter().enumerate() {
            for b in 0..256 {
                let coefficients = polynomial.get_body();
                let product: u64 = (0..8)
                    .map(|t| (coefficients.as_ref()[8 * b + t] / encoding_delta(4)) << (2 * t))
                    .sum();
                assert_eq!(product as usize, decoded_byte(a) * decoded_byte(b));
            }
        }
    }
}
