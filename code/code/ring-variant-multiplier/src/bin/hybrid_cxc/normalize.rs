use crate::config::{centered_normalizer_modulus_switch, encoding_delta, env_usize, BenchParams};
use crate::keys::EvaluationKeys;
use crate::plan::{PlannedColumn, ReductionPlan};
use crate::probe::NoiseProbeFn;
use crate::types::{BoundedLweTerm, NormalizationStats};
use dyn_stack::{PodBuffer, PodStack};
use rayon::prelude::*;
use std::collections::HashMap;
use tfhe::core_crypto::algorithms::{
    blind_rotate_assign_mem_optimized, blind_rotate_assign_mem_optimized_requirement,
    generate_programmable_bootstrap_glwe_lut, lwe_ciphertext_centered_binary_modulus_switch,
    lwe_ciphertext_modulus_switch,
};
use tfhe::core_crypto::prelude::*;
use tfhe::integer::ciphertext::IntegerRadixCiphertext;
use tfhe::integer::RadixCiphertext;
use tfhe::shortint::atomic_pattern::AtomicPatternKind;
use tfhe::shortint::ciphertext::{Degree, NoiseLevel};
use tfhe::shortint::parameters::{CarryModulus, MessageModulus};
use tfhe::shortint::{Ciphertext as ShortintCiphertext, PBSOrder};

/// Level-synchronous PBS radix normalization (Algorithm 2 of the paper).
///
/// Relation to the paper's pseudocode: Algorithm 2 leaves the group-forming
/// rule abstract behind `Partition`; any public rule that forms admissible
/// groups (`bd(T) <= cap`) from public bounds preserves Lemma 2 and
/// Theorem 1, because the digit/carry replacement argument is per group.
/// This implementation instantiates `Partition` as an exact small-bound
/// packing with a best-fit-decreasing fallback (`token_aware_chunks`). Both
/// choices depend only on public metadata, never on ciphertext contents.
pub(crate) fn normalize_height2(
    mut columns: Vec<Vec<BoundedLweTerm>>,
    output_digits: usize,
    eval_keys: &EvaluationKeys,
    params: BenchParams,
    noise_probe: Option<NoiseProbeFn>,
    plan: &ReductionPlan,
) -> (Vec<LweCiphertextOwned<u64>>, NormalizationStats) {
    let params = params.normalizer_params();
    const BASE: u64 = 4;

    if columns.len() < output_digits {
        columns.resize_with(output_digits, Vec::new);
    } else {
        columns.truncate(output_digits);
    }
    assert!(
        columns
            .iter()
            .flatten()
            .all(|term| 0 < term.bound && term.bound < BASE),
        "height-2 normalization accepts only clean radix terms"
    );

    let row_cap = normalizer_chunk_cap();
    assert!(
        row_cap >= 3 * (BASE - 1),
        "height-2 normalization needs cap >= {}",
        3 * (BASE - 1)
    );
    let big_lwe_size = params
        .glwe_dimension
        .to_equivalent_lwe_dimension(params.polynomial_size)
        .to_lwe_size();
    let carry_accumulator = generate_carry_accumulator(params);
    let digit_accumulator = generate_digit_accumulator(params);
    let mvb_accumulator = generate_mvb_accumulator(params);
    let mut stats = NormalizationStats::default();

    let mut wave = 0usize;
    while columns.iter().any(|column| column.len() > 2) {
        if let Some(probe) = noise_probe {
            for (q, column) in columns.iter().enumerate() {
                for (idx, term) in column.iter().enumerate() {
                    probe("wave-term", wave, q, idx, term.bound, 1,
                        usize::from(!term.refreshed), term.audit_expected(), &term.ciphertext);
                }
            }
        }
        let before_terms = columns.iter().map(Vec::len).sum::<usize>();
        let (mut next_columns, jobs) = plan_compression_round(
            columns,
            output_digits,
            row_cap,
            &mut stats,
            plan.waves.get(wave).expect("unexpected reduction wave"),
        );
        let job_count = jobs.len();
        let normalized = crate::config::with_phase_threads(job_count, || execute_compression_jobs(
            jobs,
            wave,
            noise_probe,
            big_lwe_size,
            &carry_accumulator,
            &digit_accumulator,
            &mvb_accumulator,
            eval_keys,
            params,
            output_digits,
        ));
        record_compression_round(&mut next_columns, normalized, &mut stats);
        wave += 1;

        let after_terms = next_columns.iter().map(Vec::len).sum::<usize>();
        assert!(
            after_terms < before_terms,
            "height-2 compression made no progress"
        );
        columns = next_columns;
    }

    plan.finish(&columns, wave);
    let output = add_height2_rows(columns, output_digits, eval_keys, &mut stats, noise_probe,
                                  plan.final_direct);
    (output, stats)
}

type CompressionJob = (usize, u64, Vec<BoundedLweTerm>, bool, bool);
/// (column, bound, expected sum, digit, carry, linear digit, multi-value).
type NormalizedJob = (
    usize,
    u64,
    Option<u64>,
    LweCiphertextOwned<u64>,
    Option<LweCiphertextOwned<u64>>,
    bool,
    bool,
);

fn plan_compression_round(
    columns: Vec<Vec<BoundedLweTerm>>,
    output_digits: usize,
    row_cap: u64,
    stats: &mut NormalizationStats,
    planned: &[PlannedColumn],
) -> (Vec<Vec<BoundedLweTerm>>, Vec<CompressionJob>) {
    let mut next_columns = (0..output_digits).map(|_| Vec::new()).collect::<Vec<_>>();
    let mut pbs_jobs = Vec::new();
    assert_eq!(planned.len(), output_digits);
    for (column_idx, current) in columns.into_iter().enumerate() {
        let bound_sum = current.iter().map(|term| term.bound).sum::<u64>();
        stats.max_column_height = stats.max_column_height.max(current.len());
        stats.max_column_bound = stats.max_column_bound.max(bound_sum);
        let chunks = planned[column_idx].partition(current, row_cap);
        stats.max_chunks = stats.max_chunks.max(chunks.len());
        for (group_idx, chunk) in chunks.into_iter().enumerate() {
            let chunk_bound = chunk.iter().map(|term| term.bound).sum::<u64>();
            if chunk.len() < 3 {
                next_columns[column_idx].extend(chunk);
            } else {
                let linear = planned[column_idx].is_linear(group_idx);
                let mvb = planned[column_idx].is_mvb(group_idx);
                pbs_jobs.push((column_idx, chunk_bound, chunk, linear, mvb));
            }
        }
    }
    (next_columns, pbs_jobs)
}

fn execute_compression_jobs(
    jobs: Vec<CompressionJob>,
    wave: usize,
    noise_probe: Option<NoiseProbeFn>,
    big_lwe_size: LweSize,
    carry_accumulator: &GlweCiphertextOwned<u64>,
    digit_accumulator: &GlweCiphertextOwned<u64>,
    mvb_accumulator: &GlweCiphertextOwned<u64>,
    eval_keys: &EvaluationKeys,
    params: BenchParams,
    output_digits: usize,
) -> Vec<NormalizedJob> {
    jobs.into_par_iter().enumerate()
        .map_init(
            || crate::config::shared_preprocessing().then(|| NormalizerScratch::new(params)),
            |scratch, (job_idx, (column_idx, chunk_bound, chunk, linear, mvb))| {
                let chunk_sum = sum_terms(&chunk, big_lwe_size, params);
                let chunk_expected = chunk
                    .iter()
                    .map(BoundedLweTerm::audit_expected)
                    .sum::<Option<u64>>();
                if let Some(probe) = noise_probe {
                    let unrefreshed = chunk.iter().filter(|term| !term.refreshed).count();
                    probe(
                        "chunk",
                        wave,
                        column_idx,
                        job_idx,
                        chunk_bound,
                        chunk.len(),
                        unrefreshed,
                        chunk_expected,
                        &chunk_sum,
                    );
                }
                let retain_carry = column_idx + 1 < output_digits && chunk_bound >= 4;
                let (digit, carry) = if mvb {
                    assert!(retain_carry, "a multi-value group needs its carry");
                    let (digit, carry, coefficients) = normalize_mvb(
                        &chunk_sum, mvb_accumulator, eval_keys, params, scratch);
                    if let Some(probe) = noise_probe {
                        // Five extracted coefficients of one blind rotation, then the outputs.
                        for (j, coefficient) in coefficients.iter().enumerate() {
                            probe("mvb-coef", wave, column_idx, job_idx * 8 + j, 0, 1, 0,
                                  None, coefficient);
                        }
                        probe("mvb-digit", wave, column_idx, job_idx, 3, 1, 0,
                              chunk_expected.map(|v| v % 4), &digit);
                        probe("mvb-carry", wave, column_idx, job_idx, chunk_bound / 4, 1, 0,
                              chunk_expected.map(|v| v / 4), &carry);
                    }
                    (digit, Some(carry))
                } else if linear {
                    // Carry-only reduction: digit = sum - 4*carry, not refreshed.
                    assert!(retain_carry || chunk_bound < 4, "a linear digit needs its carry");
                    let carry = retain_carry.then(|| normalize_carry(
                        &chunk_sum, carry_accumulator, eval_keys, params, scratch));
                    let mut digit = chunk_sum;
                    if let Some(carry) = &carry {
                        for (dst, &src) in digit.as_mut().iter_mut().zip(carry.as_ref()) {
                            *dst = dst.wrapping_sub(src.wrapping_mul(4));
                        }
                    }
                    (digit, carry)
                } else {
                    normalize_chunk_sum(
                        &chunk_sum,
                        chunk_bound,
                        carry_accumulator,
                        digit_accumulator,
                        eval_keys,
                        params,
                        retain_carry,
                        scratch,
                    )
                };
                (column_idx, chunk_bound, chunk_expected, digit, carry, linear, mvb)
            },
        )
        .collect()
}

fn record_compression_round(
    next_columns: &mut [Vec<BoundedLweTerm>],
    normalized: Vec<NormalizedJob>,
    stats: &mut NormalizationStats,
) {
    if !normalized.is_empty() {
        // Count only executed blind rotations: one digit PBS per job plus one
        // carry PBS when the carry is retained (it is skipped otherwise).
        let executed = normalized
            .iter()
            .map(|job| if job.6 { 1 } else { usize::from(!job.5) + usize::from(job.4.is_some()) })
            .sum::<usize>();
        stats.reduction_pbs += executed;
        // The shared path key-switches once per job with a PBS; the baseline
        // path key-switches once per executed PBS.
        stats.key_switches += if crate::config::shared_preprocessing() {
            normalized.iter().filter(|job| job.6 || !job.5 || job.4.is_some()).count()
        } else {
            executed
        };
        stats.rounds += 2 * normalized
            .len()
            .div_ceil(rayon::current_num_threads().max(1));
    }
    for (column_idx, chunk_bound, chunk_expected, digit, carry, linear, _) in normalized {
        #[cfg(not(feature = "noise-audit"))]
        let _ = chunk_expected;
        next_columns[column_idx].push(BoundedLweTerm {
            ciphertext: digit,
            bound: 3,
            refreshed: !linear,
            source: None,
            #[cfg(feature = "noise-audit")]
            expected: chunk_expected.map(|value| value % 4),
        });
        let carry_bound = chunk_bound / 4;
        if carry_bound > 0 && column_idx + 1 < next_columns.len() {
            assert!(carry_bound < 4, "compressor emitted a dirty carry");
            next_columns[column_idx + 1].push(BoundedLweTerm {
                ciphertext: carry.expect("retained carry was not computed"),
                bound: carry_bound,
                refreshed: true,
                source: None,
                #[cfg(feature = "noise-audit")]
                expected: chunk_expected.map(|value| value / 4),
            });
        }
    }
}

fn normalizer_chunk_cap() -> u64 {
    let cap = env_usize("CBS_NORMALIZER_CHUNK_CAP", 15) as u64;
    assert!(
        (9..=15).contains(&cap),
        "height-2 normalizer chunk cap must be in 9..=15"
    );
    cap
}

/// Public partition rule instantiating Algorithm 2's `Partition`: an exact
/// dynamic-programming packing over the public bound multiset when every
/// bound is at most three (minimizing the number of groups), with a
/// best-fit-decreasing fallback otherwise.  Only public bounds are read.
#[allow(dead_code)]
fn token_aware_chunks(mut terms: Vec<BoundedLweTerm>, cap: u64) -> Vec<Vec<BoundedLweTerm>> {
    let bounds = terms.iter().map(|term| term.bound).collect::<Vec<_>>();
    if let Some(groups) = exact_small_bound_groups(&bounds, cap) {
        let mut slots = terms.into_iter().map(Some).collect::<Vec<_>>();
        return groups
            .into_iter()
            .map(|group| {
                group
                    .into_iter()
                    .map(|idx| slots[idx].take().expect("packing index used twice"))
                    .collect()
            })
            .collect();
    }

    let mut chunks = Vec::<Vec<BoundedLweTerm>>::new();
    let mut chunk_bounds = Vec::<u64>::new();
    terms.sort_by(|lhs, rhs| rhs.bound.cmp(&lhs.bound));
    for term in terms {
        assert!(term.bound <= cap, "single term does not fit chunk cap");
        let best_fit = chunk_bounds
            .iter()
            .enumerate()
            .filter(|(_, bound)| **bound + term.bound <= cap)
            .max_by_key(|(_, bound)| *bound)
            .map(|(idx, _)| idx);
        if let Some(chunk_idx) = best_fit {
            chunk_bounds[chunk_idx] += term.bound;
            chunks[chunk_idx].push(term);
        } else {
            chunk_bounds.push(term.bound);
            chunks.push(vec![term]);
        }
    }
    chunks
}

fn exact_small_bound_groups(bounds: &[u64], cap: u64) -> Option<Vec<Vec<usize>>> {
    if bounds.is_empty() {
        return Some(Vec::new());
    }
    if cap == 0
        || bounds
            .iter()
            .any(|bound| *bound == 0 || *bound > 3 || *bound > cap)
    {
        return None;
    }

    let mut counts = [0usize; 3];
    let mut by_bound = vec![Vec::<usize>::new(), Vec::new(), Vec::new()];
    for (idx, bound) in bounds.iter().enumerate() {
        let bucket = (*bound as usize) - 1;
        counts[bucket] += 1;
        by_bound[bucket].push(idx);
    }
    let state_count = (counts[0] + 1)
        .saturating_mul(counts[1] + 1)
        .saturating_mul(counts[2] + 1);
    if state_count > 200_000 {
        return None;
    }

    let mut patterns = Vec::<[usize; 3]>::new();
    for ones in 0..=cap as usize {
        for twos in 0..=(cap as usize / 2) {
            for threes in 0..=(cap as usize / 3) {
                let fill = ones + 2 * twos + 3 * threes;
                if fill > 0 && fill <= cap as usize {
                    patterns.push([ones, twos, threes]);
                }
            }
        }
    }
    patterns.sort_by(|lhs, rhs| {
        let lhs_fill = lhs[0] + 2 * lhs[1] + 3 * lhs[2];
        let rhs_fill = rhs[0] + 2 * rhs[1] + 3 * rhs[2];
        rhs_fill
            .cmp(&lhs_fill)
            .then_with(|| (rhs[0] + rhs[1] + rhs[2]).cmp(&(lhs[0] + lhs[1] + lhs[2])))
    });

    fn solve(
        state: (usize, usize, usize),
        patterns: &[[usize; 3]],
        memo: &mut HashMap<(usize, usize, usize), (usize, Option<[usize; 3]>)>,
    ) -> usize {
        if state == (0, 0, 0) {
            return 0;
        }
        if let Some((best, _)) = memo.get(&state) {
            return *best;
        }
        let mut best = usize::MAX / 4;
        let mut best_pattern = None;
        let mut best_fill = 0usize;
        for pattern in patterns {
            if pattern[0] > state.0 || pattern[1] > state.1 || pattern[2] > state.2 {
                continue;
            }
            let next = (
                state.0 - pattern[0],
                state.1 - pattern[1],
                state.2 - pattern[2],
            );
            let candidate = 1 + solve(next, patterns, memo);
            let fill = pattern[0] + 2 * pattern[1] + 3 * pattern[2];
            if candidate < best || (candidate == best && fill > best_fill) {
                best = candidate;
                best_pattern = Some(*pattern);
                best_fill = fill;
            }
        }
        memo.insert(state, (best, best_pattern));
        best
    }

    let initial = (counts[0], counts[1], counts[2]);
    let mut memo = HashMap::new();
    solve(initial, &patterns, &mut memo);
    let mut state = initial;
    let mut groups = Vec::new();
    while state != (0, 0, 0) {
        let (_, pattern) = memo.get(&state)?;
        let pattern = (*pattern)?;
        let mut group = Vec::with_capacity(pattern[0] + pattern[1] + pattern[2]);
        for bucket in 0..3 {
            for _ in 0..pattern[bucket] {
                group.push(by_bound[bucket].pop()?);
            }
        }
        state = (
            state.0 - pattern[0],
            state.1 - pattern[1],
            state.2 - pattern[2],
        );
        groups.push(group);
    }
    Some(groups)
}

fn sum_terms(
    terms: &[BoundedLweTerm],
    lwe_size: LweSize,
    params: BenchParams,
) -> LweCiphertextOwned<u64> {
    let mut sum = LweCiphertext::new(0u64, lwe_size, params.ciphertext_modulus);
    for term in terms {
        lwe_ciphertext_add_assign(&mut sum, &term.ciphertext);
    }
    sum
}

fn generate_carry_accumulator(params: BenchParams) -> GlweCiphertextOwned<u64> {
    generate_programmable_bootstrap_glwe_lut(
        params.polynomial_size,
        params.glwe_dimension.to_glwe_size(),
        16,
        params.ciphertext_modulus,
        encoding_delta(4),
        |sum: u64| sum / 4,
    )
}

fn generate_digit_accumulator(params: BenchParams) -> GlweCiphertextOwned<u64> {
    generate_programmable_bootstrap_glwe_lut(
        params.polynomial_size,
        params.glwe_dimension.to_glwe_size(),
        16,
        params.ciphertext_modulus,
        encoding_delta(4),
        |sum: u64| sum % 4,
    )
}

fn normalize_chunk_sum(
    chunk_sum: &LweCiphertextOwned<u64>,
    chunk_bound: u64,
    carry_accumulator: &GlweCiphertextOwned<u64>,
    digit_accumulator: &GlweCiphertextOwned<u64>,
    eval_keys: &EvaluationKeys,
    params: BenchParams,
    retain_carry: bool,
    scratch: &mut Option<NormalizerScratch>,
) -> (LweCiphertextOwned<u64>, Option<LweCiphertextOwned<u64>>) {
    assert!(chunk_bound < 16);
    if crate::config::shared_preprocessing() {
        let scratch = scratch.as_mut().expect("shared normalizer scratch");
        let mut input_ks = LweCiphertext::new(
            0u64,
            params.lwe_dimension.to_lwe_size(),
            params.ciphertext_modulus,
        );
        keyswitch_lwe_ciphertext(
            eval_keys.normalizer_big_to_small_ksk(),
            chunk_sum,
            &mut input_ks,
        );
        let log_modulus = params.polynomial_size.to_blind_rotation_input_modulus_log();
        let switched = if centered_normalizer_modulus_switch() {
            lwe_ciphertext_centered_binary_modulus_switch(input_ks.as_view(), log_modulus)
        } else {
            lwe_ciphertext_modulus_switch(input_ks.as_view(), log_modulus)
        };
        let mut apply = |acc: &GlweCiphertextOwned<u64>| {
            let mut rotated = acc.clone();
            blind_rotate_assign_mem_optimized(
                &switched,
                &mut rotated,
                eval_keys.normalizer_fourier_bsk(),
                scratch.fft.as_view(),
                &mut PodStack::new(&mut scratch.memory),
            );
            let mut out = LweCiphertext::new(
                0u64,
                eval_keys
                    .normalizer_fourier_bsk()
                    .output_lwe_dimension()
                    .to_lwe_size(),
                params.ciphertext_modulus,
            );
            extract_lwe_sample_from_glwe_ciphertext(&rotated, &mut out, MonomialDegree(0));
            out
        };
        let carry = retain_carry.then(|| apply(carry_accumulator));
        let digit = apply(digit_accumulator);
        if crate::config::env_flag("CACHED_MULT_VERIFY_PBS") {
            let reference = pbs_with_accumulator(chunk_sum, digit_accumulator, eval_keys, params);
            assert_eq!(
                digit.as_ref(),
                reference.as_ref(),
                "shared digit PBS differs"
            );
            if let Some(carry) = &carry {
                let reference =
                    pbs_with_accumulator(chunk_sum, carry_accumulator, eval_keys, params);
                assert_eq!(
                    carry.as_ref(),
                    reference.as_ref(),
                    "shared carry PBS differs"
                );
            }
        }
        return (digit, carry);
    }
    // A discarded carry (last output column or chunk bound below the base) is
    // not evaluated, so its key switch and blind rotation are not timed.
    let carry = retain_carry
        .then(|| pbs_with_accumulator(chunk_sum, carry_accumulator, eval_keys, params));
    let digit = pbs_with_accumulator(chunk_sum, digit_accumulator, eval_keys, params);
    (digit, carry)
}

/// The carry PBS of a linear-digit job (same key switch, modulus switch, and
/// blind rotation as `normalize_chunk_sum`).
fn normalize_carry(
    chunk_sum: &LweCiphertextOwned<u64>,
    carry_accumulator: &GlweCiphertextOwned<u64>,
    eval_keys: &EvaluationKeys,
    params: BenchParams,
    scratch: &mut Option<NormalizerScratch>,
) -> LweCiphertextOwned<u64> {
    if !crate::config::shared_preprocessing() {
        return pbs_with_accumulator(chunk_sum, carry_accumulator, eval_keys, params);
    }
    let scratch = scratch.as_mut().expect("shared normalizer scratch");
    let mut input_ks = LweCiphertext::new(0u64, params.lwe_dimension.to_lwe_size(), params.ciphertext_modulus);
    keyswitch_lwe_ciphertext(eval_keys.normalizer_big_to_small_ksk(), chunk_sum, &mut input_ks);
    let log_modulus = params.polynomial_size.to_blind_rotation_input_modulus_log();
    let switched = if centered_normalizer_modulus_switch() {
        lwe_ciphertext_centered_binary_modulus_switch(input_ks.as_view(), log_modulus)
    } else {
        lwe_ciphertext_modulus_switch(input_ks.as_view(), log_modulus)
    };
    let mut rotated = carry_accumulator.clone();
    blind_rotate_assign_mem_optimized(&switched, &mut rotated, eval_keys.normalizer_fourier_bsk(),
        scratch.fft.as_view(), &mut PodStack::new(&mut scratch.memory));
    let mut out = LweCiphertext::new(0u64,
        eval_keys.normalizer_fourier_bsk().output_lwe_dimension().to_lwe_size(), params.ciphertext_modulus);
    extract_lwe_sample_from_glwe_ciphertext(&rotated, &mut out, MonomialDegree(0));
    out
}

/// Common-factor accumulator for multi-value folding (Bernard et al., ePrint
/// 2026/2310, Sect. 4.2.1): body coefficients -1, 0, 1, 2 times Delta in four
/// blocks of N/4, used as is (not centered again).
fn generate_mvb_accumulator(params: BenchParams) -> GlweCiphertextOwned<u64> {
    let n = params.polynomial_size.0;
    let delta = encoding_delta(4);
    let coefficients: Vec<u64> = (0..n)
        .map(|i| ((i / (n / 4)) as i64 - 1).wrapping_mul(delta as i64) as u64)
        .collect();
    allocate_and_trivially_encrypt_new_glwe_ciphertext(
        params.glwe_dimension.to_glwe_size(),
        &PlaintextList::from_container(coefficients),
        params.ciphertext_modulus,
    )
}

/// Extracted coefficients of the multi-value fold, for N >= 32.
fn mvb_extraction_indices(n: usize) -> [usize; 5] {
    let w = n / 16;
    [n - w / 2, n - 3 * w / 2, n - 5 * w / 2, n - 7 * w / 2, w / 2]
}

/// Digit and carry of a group sum from one blind rotation: the digit is
/// -E0-E1-E2+2E3-E4+Delta and the carry E4+Delta over the extracted samples.
/// Also returns the five samples for the noise probe.
fn normalize_mvb(
    chunk_sum: &LweCiphertextOwned<u64>,
    accumulator: &GlweCiphertextOwned<u64>,
    eval_keys: &EvaluationKeys,
    params: BenchParams,
    scratch: &mut Option<NormalizerScratch>,
) -> (LweCiphertextOwned<u64>, LweCiphertextOwned<u64>, Vec<LweCiphertextOwned<u64>>) {
    let mut input_ks = LweCiphertext::new(0u64, params.lwe_dimension.to_lwe_size(), params.ciphertext_modulus);
    keyswitch_lwe_ciphertext(eval_keys.normalizer_big_to_small_ksk(), chunk_sum, &mut input_ks);
    let log_modulus = params.polynomial_size.to_blind_rotation_input_modulus_log();
    let switched = if centered_normalizer_modulus_switch() {
        lwe_ciphertext_centered_binary_modulus_switch(input_ks.as_view(), log_modulus)
    } else {
        lwe_ciphertext_modulus_switch(input_ks.as_view(), log_modulus)
    };
    let mut rotated = accumulator.clone();
    let mut local = None;
    let scratch = match scratch.as_mut() {
        Some(scratch) => scratch,
        None => local.insert(NormalizerScratch::new(params)),
    };
    blind_rotate_assign_mem_optimized(&switched, &mut rotated, eval_keys.normalizer_fourier_bsk(),
        scratch.fft.as_view(), &mut PodStack::new(&mut scratch.memory));
    let size = eval_keys.normalizer_fourier_bsk().output_lwe_dimension().to_lwe_size();
    let samples: Vec<_> = mvb_extraction_indices(params.polynomial_size.0).iter().map(|&j| {
        let mut out = LweCiphertext::new(0u64, size, params.ciphertext_modulus);
        extract_lwe_sample_from_glwe_ciphertext(&rotated, &mut out, MonomialDegree(j));
        out
    }).collect();
    let delta = encoding_delta(4);
    let mut digit = LweCiphertext::new(0u64, size, params.ciphertext_modulus);
    for (sample, weight) in samples.iter().zip([-1i64, -1, -1, 2, -1]) {
        for (dst, &src) in digit.as_mut().iter_mut().zip(sample.as_ref()) {
            *dst = dst.wrapping_add(src.wrapping_mul(weight as u64));
        }
    }
    *digit.get_mut_body().data = digit.get_body().data.wrapping_add(delta);
    let mut carry = samples[4].clone();
    *carry.get_mut_body().data = carry.get_body().data.wrapping_add(delta);
    (digit, carry, samples)
}

struct NormalizerScratch {
    fft: Fft,
    memory: PodBuffer,
}

impl NormalizerScratch {
    fn new(params: BenchParams) -> Self {
        let fft = Fft::new(params.polynomial_size);
        let memory = PodBuffer::try_new(blind_rotate_assign_mem_optimized_requirement::<u64>(
            params.glwe_dimension.to_glwe_size(),
            params.polynomial_size,
            fft.as_view(),
        ))
        .expect("normalizer scratch allocation");
        Self { fft, memory }
    }
}

pub(crate) fn refresh_inputs(
    inputs: &[LweCiphertextOwned<u64>],
    keys: &EvaluationKeys,
    params: BenchParams,
) -> Vec<LweCiphertextOwned<u64>> {
    let params = params.normalizer_params();
    let accumulator = generate_digit_accumulator(params);
    inputs
        .par_iter()
        .map(|input| pbs_with_accumulator(input, &accumulator, keys, params))
        .collect()
}

fn pbs_with_accumulator(
    input: &LweCiphertextOwned<u64>,
    accumulator: &GlweCiphertextOwned<u64>,
    eval_keys: &EvaluationKeys,
    params: BenchParams,
) -> LweCiphertextOwned<u64> {
    let mut input_ks = LweCiphertext::new(
        0u64,
        params.lwe_dimension.to_lwe_size(),
        params.ciphertext_modulus,
    );
    keyswitch_lwe_ciphertext(
        eval_keys.normalizer_big_to_small_ksk(),
        input,
        &mut input_ks,
    );

    let mut output = LweCiphertext::new(
        0u64,
        eval_keys
            .normalizer_fourier_bsk()
            .output_lwe_dimension()
            .to_lwe_size(),
        params.ciphertext_modulus,
    );
    let log_modulus = params.polynomial_size.to_blind_rotation_input_modulus_log();
    let switched = if centered_normalizer_modulus_switch() {
        lwe_ciphertext_centered_binary_modulus_switch(input_ks.as_view(), log_modulus)
    } else {
        lwe_ciphertext_modulus_switch(input_ks.as_view(), log_modulus)
    };
    let mut rotated = accumulator.clone();
    let fft = Fft::new(params.polynomial_size);
    let fft = fft.as_view();
    let mut memory = PodBuffer::try_new(blind_rotate_assign_mem_optimized_requirement::<u64>(
        params.glwe_dimension.to_glwe_size(),
        params.polynomial_size,
        fft,
    ))
    .expect("allocate normalizer blind-rotate scratch");
    blind_rotate_assign_mem_optimized(
        &switched,
        &mut rotated,
        eval_keys.normalizer_fourier_bsk(),
        fft,
        &mut PodStack::new(&mut memory),
    );
    extract_lwe_sample_from_glwe_ciphertext(&rotated, &mut output, MonomialDegree(0));
    output
}

fn shortint_block(term: BoundedLweTerm, direct: bool) -> ShortintCiphertext {
    assert!(term.bound < 4, "height-2 output must be a clean radix term");
    ShortintCiphertext::new(
        term.ciphertext,
        Degree::new(term.bound),
        // `direct`: the public plan screened these inputs of the final
        // addition with their own variances, so they need no refresh.
        if term.refreshed || direct {
            NoiseLevel::NOMINAL
        } else {
            NoiseLevel::UNKNOWN
        },
        MessageModulus(4),
        CarryModulus(4),
        AtomicPatternKind::Standard(PBSOrder::KeyswitchBootstrap),
    )
}

fn add_height2_rows(
    columns: Vec<Vec<BoundedLweTerm>>,
    output_digits: usize,
    eval_keys: &EvaluationKeys,
    stats: &mut NormalizationStats,
    noise_probe: Option<NoiseProbeFn>,
    direct: bool,
) -> Vec<LweCiphertextOwned<u64>> {
    let shortint_sks: &tfhe::shortint::ServerKey = eval_keys.normalizer_radix_sks.as_ref();
    if let Some(probe) = noise_probe {
        for (column_idx, column) in columns.iter().take(output_digits).enumerate() {
            for (term_idx, term) in column.iter().enumerate().filter(|(_, term)| !term.refreshed) {
                probe(
                    "surviving-row",
                    0,
                    column_idx,
                    term_idx,
                    term.bound,
                    1,
                    1,
                    term.audit_expected(),
                    &term.ciphertext,
                );
            }
        }
    }

    if columns
        .iter()
        .take(output_digits)
        .all(|column| column.len() <= 1)
    {
        let mut output = Vec::with_capacity(output_digits);
        for mut column in columns.into_iter().take(output_digits) {
            output.push(
                column
                    .pop()
                    .map_or_else(|| shortint_sks.create_trivial(0).ct, |term| term.ciphertext),
            );
        }
        while output.len() < output_digits {
            output.push(shortint_sks.create_trivial(0).ct);
        }
        return output;
    }

    let mut lhs_blocks = Vec::with_capacity(output_digits);
    let mut rhs_blocks = Vec::with_capacity(output_digits);
    let mut unrefreshed = 0usize;
    for (column_idx, mut column) in columns.into_iter().take(output_digits).enumerate() {
        assert!(column.len() <= 2, "compressor left an overfull column");
        if let Some(probe) = noise_probe.filter(|_| !column.is_empty()) {
            // Block sum read by the first layer of the final addition.
            let mut sum = column[0].ciphertext.clone();
            for term in &column[1..] {
                lwe_ciphertext_add_assign(&mut sum, &term.ciphertext);
            }
            let expected = column.iter().map(BoundedLweTerm::audit_expected).sum::<Option<u64>>();
            let unrefreshed = column.iter().filter(|term| !term.refreshed).count();
            let bound = column.iter().map(|term| term.bound).sum::<u64>();
            probe("final-input", 0, column_idx, 0, bound, column.len(), unrefreshed, expected, &sum);
        }
        column.sort_by_key(|term| std::cmp::Reverse(term.bound));
        let lhs = column.pop();
        let rhs = column.pop();
        unrefreshed += lhs.as_ref().is_some_and(|term| !term.refreshed) as usize;
        unrefreshed += rhs.as_ref().is_some_and(|term| !term.refreshed) as usize;
        lhs_blocks.push(lhs.map_or_else(|| shortint_sks.create_trivial(0), |t| shortint_block(t, direct)));
        rhs_blocks.push(rhs.map_or_else(|| shortint_sks.create_trivial(0), |t| shortint_block(t, direct)));
    }
    while lhs_blocks.len() < output_digits {
        lhs_blocks.push(shortint_sks.create_trivial(0));
        rhs_blocks.push(shortint_sks.create_trivial(0));
    }

    let mut lhs = RadixCiphertext::from(lhs_blocks);
    let mut rhs = RadixCiphertext::from(rhs_blocks);
    let library_pbs_before = tfhe::shortint::server_key::get_pbs_count();
    if unrefreshed > 0 && !direct {
        rayon::join(
            || {
                eval_keys
                    .normalizer_radix_sks
                    .full_propagate_parallelized(&mut lhs)
            },
            || {
                eval_keys
                    .normalizer_radix_sks
                    .full_propagate_parallelized(&mut rhs)
            },
        );
        stats.rounds += unrefreshed.div_ceil(rayon::current_num_threads().max(1));
    }

    if let Err(error) = eval_keys.normalizer_radix_sks.is_add_possible(&lhs, &rhs) {
        panic!("height-2 final radix addition is not admissible: {error}");
    }
    // The addition has at most one independent job per block.
    let result = crate::config::with_phase_threads(output_digits, ||
        eval_keys.normalizer_radix_sks.add_parallelized(&lhs, &rhs));
    let (_, add_depth) = tfhe_parallel_add_model(output_digits);
    stats.final_pbs += (tfhe::shortint::server_key::get_pbs_count() - library_pbs_before) as usize;
    stats.rounds += add_depth;
    result
        .into_blocks()
        .into_iter()
        .map(|block| block.ct)
        .collect()
}

fn tfhe_parallel_add_model(num_blocks: usize) -> (usize, usize) {
    if num_blocks == 0 {
        return (0, 0);
    }
    let num_threads = rayon::current_num_threads().max(1);
    let layer_latency = |jobs: usize| jobs.div_ceil(num_threads);
    let grouping_size = 4usize;
    let num_groups = num_blocks.div_ceil(grouping_size);
    let carries_to_resolve = num_groups.saturating_sub(1);
    let sequential_depth = carries_to_resolve.saturating_sub(1) / (grouping_size - 1);
    let hillis_steele_depth = ceil_log2(carries_to_resolve);
    let use_sequential_groups = sequential_depth <= hillis_steele_depth;

    let mut parallel_depth = 3 * layer_latency(num_blocks);
    if use_sequential_groups {
        parallel_depth += sequential_depth * layer_latency(grouping_size);
    } else {
        let mut space = 1usize;
        while space < num_blocks {
            parallel_depth += layer_latency(num_blocks - space);
            space *= 2;
        }
    }
    if parallel_depth >= num_blocks {
        return (2 * num_blocks, num_blocks);
    }

    let group_resolution_pbs = if carries_to_resolve == 0 {
        0
    } else if use_sequential_groups {
        carries_to_resolve - 1
    } else {
        let mut count = 0usize;
        let mut space = 1usize;
        while space < carries_to_resolve {
            count += carries_to_resolve - space;
            space *= 2;
        }
        count
    };
    (3 * num_blocks - 1 + group_resolution_pbs, parallel_depth)
}

fn ceil_log2(value: usize) -> usize {
    if value <= 1 {
        0
    } else {
        (usize::BITS - (value - 1).leading_zeros()) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::exact_small_bound_groups;

    #[test]
    fn exact_packing_respects_cap_and_uses_minimum_groups() {
        let bounds = [3, 3, 3, 3, 3, 2, 2, 1];
        let groups = exact_small_bound_groups(&bounds, 15).expect("small-bound packing");
        assert_eq!(groups.len(), 2);
        assert!(groups
            .iter()
            .all(|group| group.iter().map(|&idx| bounds[idx]).sum::<u64>() <= 15));
    }
}
