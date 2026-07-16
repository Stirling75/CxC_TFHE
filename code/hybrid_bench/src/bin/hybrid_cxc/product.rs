use crate::config::{
    encoding_delta, parallel_cmux_cells, parallel_digit_lift, BenchParams, ProductVariant,
};
use crate::keys::EvaluationKeys;
use crate::revhomtrace_lift_tfhe16;
use crate::types::{BoundedLweTerm, ProductDigitTerm, Timings};
use anyhow::{bail, Result};
use rayon::prelude::*;
use std::time::{Duration, Instant};
use tfhe::core_crypto::algorithms::polynomial_algorithms::polynomial_wrapping_monic_monomial_div;
use tfhe::core_crypto::fft_impl::fft64::crypto::ggsw::FourierGgswCiphertextList;
use tfhe::core_crypto::prelude::*;

pub(crate) struct EncryptedOperands {
    pub(crate) lhs: Vec<LweCiphertextOwned<u64>>,
    pub(crate) rhs: Vec<LweCiphertextOwned<u64>>,
}

pub(crate) struct ProductEvaluation {
    pub(crate) columns: Vec<Vec<ProductDigitTerm>>,
    pub(crate) timings: Timings,
    pub(crate) elapsed: Duration,
}

/// Evaluator-only entry point: no secret key or cleartext operand crosses this boundary.
pub(crate) fn evaluate_product(
    inputs: EncryptedOperands,
    eval_keys: &EvaluationKeys,
    params: BenchParams,
    variant: ProductVariant,
) -> Result<ProductEvaluation> {
    if inputs.lhs.len() != inputs.rhs.len() {
        bail!("hybrid product requires equal public operand widths");
    }
    if inputs.lhs.len() % 4 != 0 {
        bail!("hybrid product requires a multiple of four radix-4 digits");
    }

    // Keep this boundary aligned with the released measurements: it starts
    // after input encryption and includes lifting, LUT construction, CMUX, and extraction.
    let elapsed = Instant::now();
    let mut timings = Timings::default();
    let (lhs_lifted, rhs_lifted) = lift_operands(inputs, eval_keys, params, &mut timings)?;
    let (columns, cmux_ms) = match variant {
        ProductVariant::Hybrid4x4 => evaluate_hybrid_4x4(&lhs_lifted, &rhs_lifted, params)?,
        ProductVariant::Hybrid8x8 => evaluate_hybrid_8x8(&lhs_lifted, &rhs_lifted, params)?,
    };
    timings.ext_ms = cmux_ms;
    timings.cmux_count = cmux_count(variant, lhs_lifted.len());

    Ok(ProductEvaluation {
        columns,
        timings,
        elapsed: elapsed.elapsed(),
    })
}

fn lift_operands(
    inputs: EncryptedOperands,
    eval_keys: &EvaluationKeys,
    params: BenchParams,
    timings: &mut Timings,
) -> Result<(
    Vec<FourierGgswCiphertextList<Vec<tfhe_fft::c64>>>,
    Vec<FourierGgswCiphertextList<Vec<tfhe_fft::c64>>>,
)> {
    let lhs_len = inputs.lhs.len();
    let rhs_len = inputs.rhs.len();
    let all_lwes = inputs.lhs.into_iter().chain(inputs.rhs).collect::<Vec<_>>();
    timings.cbs_lifts = all_lwes.len();

    let lifted = if parallel_digit_lift() && all_lwes.len() > 1 {
        let started = Instant::now();
        let output = all_lwes
            .par_iter()
            .map(|lwe| lift_radix_digit(lwe, eval_keys, params))
            .collect::<Result<Vec<_>>>()?;
        timings.cbs_ms += started.elapsed().as_millis();
        output
    } else {
        let mut output = Vec::with_capacity(all_lwes.len());
        for lwe in &all_lwes {
            let started = Instant::now();
            output.push(lift_radix_digit(lwe, eval_keys, params)?);
            timings.cbs_ms += started.elapsed().as_millis();
        }
        output
    };

    let mut lifted = lifted.into_iter();
    let lhs = lifted.by_ref().take(lhs_len).collect::<Vec<_>>();
    let rhs = lifted.by_ref().take(rhs_len).collect::<Vec<_>>();
    Ok((lhs, rhs))
}

fn lift_radix_digit(
    digit: &LweCiphertextOwned<u64>,
    eval_keys: &EvaluationKeys,
    params: BenchParams,
) -> Result<FourierGgswCiphertextList<Vec<tfhe_fft::c64>>> {
    let direct = &eval_keys.direct_lift;
    let count = 2usize;
    let mut standard = GgswCiphertextList::new(
        0u64,
        params.glwe_dimension.to_glwe_size(),
        params.polynomial_size,
        direct.cbs_base_log,
        direct.cbs_level,
        GgswCiphertextCount(count),
        params.ciphertext_modulus,
    );
    let fourier_len = count
        * params.polynomial_size.to_fourier_polynomial_size().0
        * params.glwe_dimension.to_glwe_size().0
        * params.glwe_dimension.to_glwe_size().0
        * direct.cbs_level.0;
    let mut fourier = FourierGgswCiphertextList::new(
        vec![tfhe_fft::c64::default(); fourier_len],
        count,
        params.glwe_dimension.to_glwe_size(),
        params.polynomial_size,
        direct.cbs_base_log,
        direct.cbs_level,
    );

    revhomtrace_lift_tfhe16::improved_wopbs_multi_bits_revtrace_tfhe16(
        digit,
        &mut standard,
        &mut fourier,
        2,
        &eval_keys.cbs_big_to_small_ksk,
        eval_keys.cbs_fourier_bsk.as_view(),
        &direct.auto_keys,
        direct.ss_key.as_view(),
        direct.log_lut_count,
    );
    Ok(fourier)
}

fn zero_glwe(params: BenchParams) -> GlweCiphertextOwned<u64> {
    GlweCiphertext::new(
        0u64,
        params.glwe_dimension.to_glwe_size(),
        params.polynomial_size,
        params.ciphertext_modulus,
    )
}

fn monomial_div_into<OutputCont, InputCont>(
    output: &mut GlweCiphertext<OutputCont>,
    input: &GlweCiphertext<InputCont>,
    degree: MonomialDegree,
) where
    OutputCont: ContainerMut<Element = u64>,
    InputCont: Container<Element = u64>,
{
    assert_eq!(output.glwe_size(), input.glwe_size());
    assert_eq!(output.polynomial_size(), input.polynomial_size());
    assert_eq!(output.ciphertext_modulus(), input.ciphertext_modulus());
    for (mut output_poly, input_poly) in output
        .as_mut_polynomial_list()
        .iter_mut()
        .zip(input.as_polynomial_list().iter())
    {
        polynomial_wrapping_monic_monomial_div(&mut output_poly, &input_poly, degree);
    }
}

fn hybrid_4x4_accumulator(params: BenchParams, output_bits: usize) -> GlweCiphertextOwned<u64> {
    assert!(
        params.polynomial_size.0 >= 256 && params.polynomial_size.0 % 256 == 0,
        "4x4 CMUX product LUT expects the polynomial to split into 256 equal slots"
    );
    let segment_len = params.polynomial_size.0 / 256;
    assert!(
        segment_len >= 4,
        "4x4 CMUX product LUT needs at least four coefficients per selected slot"
    );
    let mut coefficients = vec![0u64; params.polynomial_size.0];
    let delta = encoding_delta(output_bits);

    for a0_y0 in 0..=1usize {
        for b0_y0 in 0..=1usize {
            for a1_y0 in 0..=1usize {
                for b1_y0 in 0..=1usize {
                    for a0_y1 in 0..=1usize {
                        for b0_y1 in 0..=1usize {
                            for a1_y1 in 0..=1usize {
                                for b1_y1 in 0..=1usize {
                                    let a0 = (a0_y0 ^ a0_y1) + 2 * a0_y1;
                                    let b0 = (b0_y0 ^ b0_y1) + 2 * b0_y1;
                                    let a1 = (a1_y0 ^ a1_y1) + 2 * a1_y1;
                                    let b1 = (b1_y0 ^ b1_y1) + 2 * b1_y1;
                                    let product = (a0 + 4 * a1) * (b0 + 4 * b1);
                                    let digits = [
                                        product % 4,
                                        (product / 4) % 4,
                                        (product / 16) % 4,
                                        (product / 64) % 4,
                                    ];
                                    let offset = a0_y0 * (params.polynomial_size.0 >> 1)
                                        + b0_y0 * (params.polynomial_size.0 >> 2)
                                        + a1_y0 * (params.polynomial_size.0 >> 3)
                                        + b1_y0 * (params.polynomial_size.0 >> 4)
                                        + a0_y1 * (params.polynomial_size.0 >> 5)
                                        + b0_y1 * (params.polynomial_size.0 >> 6)
                                        + a1_y1 * (params.polynomial_size.0 >> 7)
                                        + b1_y1 * (params.polynomial_size.0 >> 8);
                                    for local_idx in 0..segment_len {
                                        coefficients[offset + local_idx] =
                                            digits[local_idx % 4] as u64 * delta;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    allocate_and_trivially_encrypt_new_glwe_ciphertext(
        params.glwe_dimension.to_glwe_size(),
        &PlaintextList::from_container(coefficients),
        params.ciphertext_modulus,
    )
}

fn refined_selector_cmux_assign(
    selected: &mut GlweCiphertextOwned<u64>,
    selector_bits: &FourierGgswCiphertextList<Vec<tfhe_fft::c64>>,
    refined_bit_idx: usize,
    round_idx: usize,
    params: BenchParams,
) {
    let mut rotated = zero_glwe(params);
    let shift = params.polynomial_size.0 >> (round_idx + 1);
    monomial_div_into(&mut rotated, selected, MonomialDegree(shift));
    let mut iter = selector_bits.as_view().into_ggsw_iter();
    let selector = match refined_bit_idx {
        0 => iter
            .next()
            .expect("refined digit extraction must output y0"),
        1 => {
            let _ = iter.next();
            iter.next()
                .expect("refined digit extraction must output y1")
        }
        _ => unreachable!("2-bit refined selector index"),
    };
    cmux_assign::<u64, _, _, _>(selected, &mut rotated, &selector);
}

fn evaluate_hybrid_4x4(
    lhs: &[FourierGgswCiphertextList<Vec<tfhe_fft::c64>>],
    rhs: &[FourierGgswCiphertextList<Vec<tfhe_fft::c64>>],
    params: BenchParams,
) -> Result<(Vec<Vec<BoundedLweTerm>>, u128)> {
    if lhs.is_empty() || rhs.is_empty() {
        return Ok((Vec::new(), 0));
    }
    if lhs.len() != rhs.len() || lhs.len() % 4 != 0 {
        bail!("hybrid-4x4 expects equal lengths divisible by four");
    }
    let digit_count = lhs.len();
    let chunk_count = digit_count / 4;
    let output_lwe_size = params
        .glwe_dimension
        .to_equivalent_lwe_dimension(params.polynomial_size)
        .to_lwe_size();
    let mut columns = (0..2 * digit_count).map(|_| Vec::new()).collect::<Vec<_>>();
    let accumulator = hybrid_4x4_accumulator(params, 4);
    let started = Instant::now();
    let jobs = hybrid_4x4_jobs(chunk_count, digit_count);
    let run_cell = |job| {
        evaluate_hybrid_4x4_cell(
            job,
            &accumulator,
            lhs,
            rhs,
            digit_count,
            output_lwe_size,
            params,
        )
    };

    let results = if parallel_cmux_cells() && jobs.len() > 1 {
        jobs.into_par_iter().map(run_cell).collect::<Vec<_>>()
    } else {
        jobs.into_iter().map(run_cell).collect::<Vec<_>>()
    };
    for cell in results {
        for (diagonal, term) in cell {
            columns[diagonal].push(term);
        }
    }
    Ok((columns, started.elapsed().as_millis()))
}

type Hybrid4x4Job = (usize, usize, usize, usize, usize);

fn hybrid_4x4_jobs(chunk_count: usize, digit_count: usize) -> Vec<Hybrid4x4Job> {
    let mut jobs = Vec::new();
    for lhs_chunk in 0..chunk_count {
        for rhs_chunk in 0..chunk_count {
            let base_diagonal = 4 * (lhs_chunk + rhs_chunk);
            if base_diagonal >= digit_count {
                continue;
            }
            let lhs_base = 4 * lhs_chunk;
            let rhs_base = 4 * rhs_chunk;
            for job in [
                (
                    lhs_base,
                    lhs_base + 1,
                    rhs_base,
                    rhs_base + 1,
                    base_diagonal,
                ),
                (
                    lhs_base,
                    lhs_base + 1,
                    rhs_base + 2,
                    rhs_base + 3,
                    base_diagonal + 2,
                ),
                (
                    lhs_base + 2,
                    lhs_base + 3,
                    rhs_base,
                    rhs_base + 1,
                    base_diagonal + 2,
                ),
                (
                    lhs_base + 2,
                    lhs_base + 3,
                    rhs_base + 2,
                    rhs_base + 3,
                    base_diagonal + 4,
                ),
            ] {
                if job.4 < digit_count {
                    jobs.push(job);
                }
            }
        }
    }
    jobs
}

fn evaluate_hybrid_4x4_cell(
    (a0_idx, a1_idx, b0_idx, b1_idx, base_diagonal): Hybrid4x4Job,
    accumulator: &GlweCiphertextOwned<u64>,
    lhs: &[FourierGgswCiphertextList<Vec<tfhe_fft::c64>>],
    rhs: &[FourierGgswCiphertextList<Vec<tfhe_fft::c64>>],
    digit_count: usize,
    output_lwe_size: LweSize,
    params: BenchParams,
) -> Vec<(usize, BoundedLweTerm)> {
    let mut selected = accumulator.clone();
    refined_selector_cmux_assign(&mut selected, &lhs[a0_idx], 0, 0, params);
    refined_selector_cmux_assign(&mut selected, &rhs[b0_idx], 0, 1, params);
    refined_selector_cmux_assign(&mut selected, &lhs[a1_idx], 0, 2, params);
    refined_selector_cmux_assign(&mut selected, &rhs[b1_idx], 0, 3, params);
    refined_selector_cmux_assign(&mut selected, &lhs[a0_idx], 1, 4, params);
    refined_selector_cmux_assign(&mut selected, &rhs[b0_idx], 1, 5, params);
    refined_selector_cmux_assign(&mut selected, &lhs[a1_idx], 1, 6, params);
    refined_selector_cmux_assign(&mut selected, &rhs[b1_idx], 1, 7, params);
    extract_clean_digits(
        &selected,
        base_diagonal,
        4,
        digit_count,
        output_lwe_size,
        params,
    )
}

#[derive(Clone, Copy)]
struct SelectorEntry {
    lhs: bool,
    digit_idx: usize,
    refined_bit_idx: usize,
}

fn selector_order(lhs_base: usize, rhs_base: usize) -> [SelectorEntry; 16] {
    [
        selector(true, lhs_base, 0),
        selector(false, rhs_base, 0),
        selector(true, lhs_base + 1, 0),
        selector(false, rhs_base + 1, 0),
        selector(true, lhs_base + 2, 0),
        selector(false, rhs_base + 2, 0),
        selector(true, lhs_base + 3, 0),
        selector(false, rhs_base + 3, 0),
        selector(true, lhs_base, 1),
        selector(false, rhs_base, 1),
        selector(true, lhs_base + 1, 1),
        selector(false, rhs_base + 1, 1),
        selector(true, lhs_base + 2, 1),
        selector(false, rhs_base + 2, 1),
        selector(true, lhs_base + 3, 1),
        selector(false, rhs_base + 3, 1),
    ]
}

fn selector(lhs: bool, digit_idx: usize, refined_bit_idx: usize) -> SelectorEntry {
    SelectorEntry {
        lhs,
        digit_idx,
        refined_bit_idx,
    }
}

fn refined_selector_cmux_assign_entry(
    selected: &mut GlweCiphertextOwned<u64>,
    choice_if_one: &mut GlweCiphertextOwned<u64>,
    entry: SelectorEntry,
    lhs: &[FourierGgswCiphertextList<Vec<tfhe_fft::c64>>],
    rhs: &[FourierGgswCiphertextList<Vec<tfhe_fft::c64>>],
) {
    let selector_bits = if entry.lhs {
        &lhs[entry.digit_idx]
    } else {
        &rhs[entry.digit_idx]
    };
    let mut iter = selector_bits.as_view().into_ggsw_iter();
    let selector = match entry.refined_bit_idx {
        0 => iter
            .next()
            .expect("refined digit extraction must output y0"),
        1 => {
            let _ = iter.next();
            iter.next()
                .expect("refined digit extraction must output y1")
        }
        _ => unreachable!("2-bit refined selector index"),
    };
    cmux_assign::<u64, _, _, _>(selected, choice_if_one, &selector);
}

fn offset_from_msb_bits(bits: &[usize], polynomial_size: usize) -> usize {
    bits.iter()
        .enumerate()
        .map(|(round_idx, &bit)| bit * (polynomial_size >> (round_idx + 1)))
        .sum()
}

fn decode_refined_digit(y0: usize, y1: usize) -> usize {
    (y0 ^ y1) + 2 * y1
}

fn hybrid_8x8_accumulator_bank(params: BenchParams) -> Vec<GlweCiphertextOwned<u64>> {
    const PREFIX_BITS: usize = 8;
    const TOTAL_SELECTOR_BITS: usize = 16;
    let suffix_bits = TOTAL_SELECTOR_BITS - PREFIX_BITS;
    assert!(
        params.polynomial_size.0 >= (1usize << suffix_bits)
            && params.polynomial_size.0 % (1usize << suffix_bits) == 0,
        "8x8 direct LUT expects each suffix table to divide the polynomial"
    );
    let segment_len = params.polynomial_size.0 >> suffix_bits;
    assert!(
        segment_len >= 8,
        "8x8 direct LUT needs at least eight coefficients per selected slot"
    );
    let delta = encoding_delta(4);

    (0..(1usize << PREFIX_BITS))
        .map(|prefix| {
            let mut coefficients = vec![0u64; params.polynomial_size.0];
            for suffix in 0..(1usize << suffix_bits) {
                let mut refined = [0usize; TOTAL_SELECTOR_BITS];
                for (bit_idx, bit) in refined.iter_mut().take(PREFIX_BITS).enumerate() {
                    *bit = (prefix >> (PREFIX_BITS - 1 - bit_idx)) & 1;
                }
                for (bit_idx, bit) in refined.iter_mut().skip(PREFIX_BITS).enumerate() {
                    *bit = (suffix >> (suffix_bits - 1 - bit_idx)) & 1;
                }

                let a0 = decode_refined_digit(refined[0], refined[8]);
                let b0 = decode_refined_digit(refined[1], refined[9]);
                let a1 = decode_refined_digit(refined[2], refined[10]);
                let b1 = decode_refined_digit(refined[3], refined[11]);
                let a2 = decode_refined_digit(refined[4], refined[12]);
                let b2 = decode_refined_digit(refined[5], refined[13]);
                let a3 = decode_refined_digit(refined[6], refined[14]);
                let b3 = decode_refined_digit(refined[7], refined[15]);
                let lhs = a0 + 4 * a1 + 16 * a2 + 64 * a3;
                let rhs = b0 + 4 * b1 + 16 * b2 + 64 * b3;
                let product = lhs * rhs;
                let offset =
                    offset_from_msb_bits(&refined[PREFIX_BITS..], params.polynomial_size.0);
                for local_idx in 0..segment_len {
                    let digit = (product >> (2 * (local_idx % 8))) & 3;
                    coefficients[offset + local_idx] = digit as u64 * delta;
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

fn evaluate_hybrid_8x8(
    lhs: &[FourierGgswCiphertextList<Vec<tfhe_fft::c64>>],
    rhs: &[FourierGgswCiphertextList<Vec<tfhe_fft::c64>>],
    params: BenchParams,
) -> Result<(Vec<Vec<BoundedLweTerm>>, u128)> {
    if lhs.is_empty() || rhs.is_empty() {
        return Ok((Vec::new(), 0));
    }
    if lhs.len() != rhs.len() || lhs.len() % 4 != 0 {
        bail!("hybrid-8x8 expects equal lengths divisible by four");
    }
    let digit_count = lhs.len();
    let chunk_count = digit_count / 4;
    let output_lwe_size = params
        .glwe_dimension
        .to_equivalent_lwe_dimension(params.polynomial_size)
        .to_lwe_size();
    let mut columns = (0..2 * digit_count).map(|_| Vec::new()).collect::<Vec<_>>();
    let accumulator_bank = hybrid_8x8_accumulator_bank(params);
    let started = Instant::now();
    let jobs = (0..chunk_count)
        .flat_map(|lhs_chunk| {
            (0..chunk_count).filter_map(move |rhs_chunk| {
                let base_diagonal = 4 * (lhs_chunk + rhs_chunk);
                (base_diagonal < digit_count).then_some((lhs_chunk, rhs_chunk, base_diagonal))
            })
        })
        .collect::<Vec<_>>();
    let run_cell = |job| {
        evaluate_hybrid_8x8_cell(
            job,
            &accumulator_bank,
            lhs,
            rhs,
            digit_count,
            output_lwe_size,
            params,
        )
    };

    let results = if parallel_cmux_cells() && jobs.len() > 1 {
        jobs.into_par_iter().map(run_cell).collect::<Vec<_>>()
    } else {
        jobs.into_iter().map(run_cell).collect::<Vec<_>>()
    };
    for cell in results {
        for (diagonal, term) in cell {
            columns[diagonal].push(term);
        }
    }
    Ok((columns, started.elapsed().as_millis()))
}

fn evaluate_hybrid_8x8_cell(
    (lhs_chunk, rhs_chunk, base_diagonal): (usize, usize, usize),
    accumulator_bank: &[GlweCiphertextOwned<u64>],
    lhs: &[FourierGgswCiphertextList<Vec<tfhe_fft::c64>>],
    rhs: &[FourierGgswCiphertextList<Vec<tfhe_fft::c64>>],
    digit_count: usize,
    output_lwe_size: LweSize,
    params: BenchParams,
) -> Vec<(usize, BoundedLweTerm)> {
    const PREFIX_BITS: usize = 8;
    let order = selector_order(4 * lhs_chunk, 4 * rhs_chunk);
    let mut bank = accumulator_bank.to_vec();
    for &entry in order.iter().take(PREFIX_BITS) {
        let half = bank.len() / 2;
        let mut next_bank = Vec::with_capacity(half);
        for idx in 0..half {
            let mut selected = bank[idx].clone();
            let mut choice_if_one = bank[idx + half].clone();
            refined_selector_cmux_assign_entry(&mut selected, &mut choice_if_one, entry, lhs, rhs);
            next_bank.push(selected);
        }
        bank = next_bank;
    }
    debug_assert_eq!(bank.len(), 1);
    let mut selected = bank
        .pop()
        .expect("prefix selection must leave one accumulator");
    for (round, &entry) in order.iter().skip(PREFIX_BITS).enumerate() {
        let mut rotated = zero_glwe(params);
        let shift = params.polynomial_size.0 >> (round + 1);
        monomial_div_into(&mut rotated, &selected, MonomialDegree(shift));
        refined_selector_cmux_assign_entry(&mut selected, &mut rotated, entry, lhs, rhs);
    }
    extract_clean_digits(
        &selected,
        base_diagonal,
        8,
        digit_count,
        output_lwe_size,
        params,
    )
}

fn extract_clean_digits(
    selected: &GlweCiphertextOwned<u64>,
    base_diagonal: usize,
    count: usize,
    digit_count: usize,
    output_lwe_size: LweSize,
    params: BenchParams,
) -> Vec<(usize, BoundedLweTerm)> {
    (0..count)
        .filter_map(|local_digit| {
            let diagonal = base_diagonal + local_digit;
            if diagonal >= digit_count {
                return None;
            }
            let mut digit = LweCiphertext::new(0u64, output_lwe_size, params.ciphertext_modulus);
            extract_lwe_sample_from_glwe_ciphertext(
                selected,
                &mut digit,
                MonomialDegree(local_digit),
            );
            Some((
                diagonal,
                BoundedLweTerm {
                    ciphertext: digit,
                    bound: 3,
                    refreshed: false,
                },
            ))
        })
        .collect()
}

fn cmux_count(variant: ProductVariant, digit_count: usize) -> usize {
    let chunk_count = digit_count / 4;
    let active_chunk_pairs = (0..chunk_count)
        .flat_map(|lhs| (0..chunk_count).map(move |rhs| (lhs, rhs)))
        .filter(|(lhs, rhs)| 4 * (lhs + rhs) < digit_count)
        .count();
    match variant {
        ProductVariant::Hybrid4x4 => {
            let mut subcells = 0usize;
            for lhs in 0..chunk_count {
                for rhs in 0..chunk_count {
                    let diagonal = 4 * (lhs + rhs);
                    if diagonal >= digit_count {
                        continue;
                    }
                    subcells += [0usize, 2, 2, 4]
                        .into_iter()
                        .filter(|offset| diagonal + offset < digit_count)
                        .count();
                }
            }
            8 * subcells
        }
        ProductVariant::Hybrid8x8 => {
            let per_cell = ((1usize << 8) - 1) + (16 - 8);
            active_chunk_pairs * per_cell
        }
    }
}
