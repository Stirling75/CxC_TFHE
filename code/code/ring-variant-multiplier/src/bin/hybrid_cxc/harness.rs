use crate::config::{env_flag, validate_runtime_contract, BenchParams, ProductVariant};
use crate::keys::{
    decrypt_radix_digits, encrypt_radix_digits, generate_keys, KeySet, RngResources,
};
use crate::normalize::normalize_height2;
use crate::plan::ReductionPlan;
use crate::probe::{NoiseProbe, NoiseProbeFn};
use crate::product::{evaluate_product, EncryptedOperands};
use crate::report::append_timing_csv;
use crate::types::duration_ms;
use anyhow::{bail, Context, Result};
use std::env;
use std::time::Instant;
use tfhe::core_crypto::prelude::LweCiphertextOwned;

pub(crate) fn run() -> Result<()> {
    let args = env::args().collect::<Vec<_>>();
    let width_bits = parse_arg(&args, 1, "width_bits")?;
    let product_count = parse_optional_arg(&args, 2, 1usize, "product_count")?;
    let trial_count = parse_optional_arg(&args, 3, 1usize, "trial_count")?;
    validate_run_shape(width_bits, product_count, trial_count)?;
    validate_runtime_contract()?;
    let centered_ms = explicit_lift_centered_ms(env::var("CBS_CENTERED_MS").ok().as_deref())?;
    validate_selector_box_centering(env_flag("CBS_CENTER_SELECTOR_BOX"), centered_ms)?;
    let operands = OperandSource::parse(
        env::var("CACHED_MULT_PATTERN").ok().as_deref(),
        env::var("CBS_RANDOM_OPERANDS_SEED").ok().as_deref(),
    )?;

    let variant = ProductVariant::from_env()?;
    let params = BenchParams::from_env(variant)?;
    let plan = ReductionPlan::load(width_bits, params, variant)?;
    print_configuration(params, variant);
    println!("direct_auto_fft={}", params.auto_fft.label());

    let mut rng = RngResources::new();
    let keygen = Instant::now();
    let keys = generate_keys(params, variant, &mut rng);
    println!("keygen_ms={}", keygen.elapsed().as_millis());
    print_direct_lift_parameters(&keys);
    crate::report::write_parameters(params, &keys.evaluator, operands)?;
    rng.keep_seeder_alive();

    for trial_idx in 0..trial_count {
        run_trial(
            width_bits, trial_idx, variant, &keys, params, &mut rng, &plan, operands,
        )?;
    }
    println!("ok=true");
    Ok(())
}

/// The selector lift's modulus-switch mode must be explicit for this binary:
/// `CBS_CENTERED_MS` is read by the shared lift with a default of off, so a
/// missing value would silently change the analysed lift.  Only `0` or `1` is
/// accepted; the campaign plan sets `1` and `ReductionPlan::load` requires it.
fn explicit_lift_centered_ms(value: Option<&str>) -> Result<bool> {
    match value {
        Some("1") => Ok(true),
        Some("0") => Ok(false),
        Some(other) => bail!("CBS_CENTERED_MS must be 0 or 1; got {other:?}"),
        None => bail!("CBS_CENTERED_MS must be set explicitly (0 or 1) for the hybrid selector lift"),
    }
}

/// Plaintext operand generator.  `random` requires an explicit u64 seed so the
/// operands of every recorded trial are reproducible from `parameters.json`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OperandSource {
    /// Legacy deterministic sequence (no pattern and no seed given).
    Deterministic,
    /// Seeded pseudorandom digits; `explicit` records whether `CACHED_MULT_PATTERN=random` was set.
    Random { seed: u64, explicit: bool },
    Zero,
    Max,
    Alternating,
}

impl OperandSource {
    pub(crate) fn parse(pattern: Option<&str>, seed: Option<&str>) -> Result<Self> {
        let parsed_seed = match seed {
            None => None,
            Some(text) => Some(text.parse::<u64>().map_err(|_| {
                anyhow::anyhow!("CBS_RANDOM_OPERANDS_SEED must be an unsigned 64-bit integer; got {text:?}")
            })?),
        };
        Ok(match pattern {
            Some("random") => Self::Random {
                seed: parsed_seed.context("CACHED_MULT_PATTERN=random requires CBS_RANDOM_OPERANDS_SEED")?,
                explicit: true,
            },
            Some("zero") => Self::Zero,
            Some("max") => Self::Max,
            Some("alternating") => Self::Alternating,
            Some(other) => bail!("CACHED_MULT_PATTERN must be random, zero, max or alternating; got {other:?}"),
            None => match parsed_seed {
                Some(seed) => Self::Random { seed, explicit: false },
                None => Self::Deterministic,
            },
        })
    }

    pub(crate) fn pattern_label(self) -> Option<&'static str> {
        match self {
            Self::Deterministic | Self::Random { explicit: false, .. } => None,
            Self::Random { explicit: true, .. } => Some("random"),
            Self::Zero => Some("zero"),
            Self::Max => Some("max"),
            Self::Alternating => Some("alternating"),
        }
    }

    pub(crate) fn seed(self) -> Option<u64> {
        match self {
            Self::Random { seed, .. } => Some(seed),
            _ => None,
        }
    }
}

/// True when any validation-only callback is enabled; such rows are flagged in
/// the timing CSV and rejected by the timed-campaign validator.
fn audit_callbacks_requested() -> bool {
    cfg!(feature = "noise-audit")
        || env_flag("CBS_NOISE_PROBE")
        || env_flag("FUSED_COMPARE_PRODUCT")
        || env_flag("CACHED_MULT_VERIFY_PBS")
        || env::var_os("FUSED_STEP_AUDIT_CSV").is_some()
}

/// The selector-box centering opt-in shifts every radix-4 selector plaintext by the public
/// phase q/16 so both extracted bits sit at the center of their accumulator boxes.  The shift
/// is only meaningful once the mask-dependent rounding bias of ordinary modulus switching is
/// removed, so the flag is rejected unless centered modulus switching is active on the
/// selector-lift path.  The lift is shared by the 4x4 and 8x8 product contracts, so the flag
/// applies to both.
fn validate_selector_box_centering(enabled: bool, centered_modulus_switch: bool) -> Result<()> {
    if enabled && !centered_modulus_switch {
        bail!("CBS_CENTER_SELECTOR_BOX=1 requires CBS_CENTERED_MS=1");
    }
    Ok(())
}

fn validate_run_shape(width_bits: usize, product_count: usize, trial_count: usize) -> Result<()> {
    if width_bits == 0 {
        bail!("width_bits must be positive");
    }
    if width_bits % 8 != 0 {
        bail!(
            "width_bits={width_bits} is unsupported; the released 4x4/8x8 contracts require a multiple of 8"
        );
    }
    if product_count != 1 {
        bail!("single-multiplication binary requires product count 1; got {product_count}");
    }
    if trial_count == 0 {
        bail!("trial_count must be positive");
    }
    Ok(())
}

fn run_trial(
    width_bits: usize,
    trial_idx: usize,
    variant: ProductVariant,
    keys: &KeySet,
    params: BenchParams,
    rng: &mut RngResources,
    plan: &ReductionPlan,
    operands: OperandSource,
) -> Result<()> {
    let lhs_clear = generated_radix_digits(operands, width_bits, trial_idx, 0);
    let rhs_clear = if let Some(scalar) = &plan.public_scalar {
        scalar.clone()
    } else if plan.squaring {
        lhs_clear.clone()
    } else {
        generated_radix_digits(operands, width_bits, trial_idx, 1)
    };
    let label = format!(
        "generated_w{width_bits}_single_trial{trial_idx}{}",
        variant.label_suffix()
    );

    let encrypt_started = Instant::now();
    let lhs_fresh = encrypt_radix_digits(&lhs_clear, &keys.harness, params, rng);
    let lhs = crate::normalize::refresh_inputs(&lhs_fresh, &keys.evaluator, params);
    let plaintext_rhs = crate::config::env_flag("CBS_CXP");
    let rhs = if plaintext_rhs {
        Vec::new()
    } else if plan.squaring {
        lhs.clone()
    } else {
        let fresh = encrypt_radix_digits(&rhs_clear, &keys.harness, params, rng);
        crate::normalize::refresh_inputs(&fresh, &keys.evaluator, params)
    };
    let encrypted = EncryptedOperands {
        lhs,
        rhs,
        squaring: plan.squaring,
        public_rhs: plaintext_rhs.then(|| rhs_clear.clone()),
        public_groups: plan.public_groups.clone(),
    };
    let encryption_ms = duration_ms(encrypt_started.elapsed());

    #[cfg(feature = "noise-audit")]
    let lift_audit_callback = |a: &[crate::product::LiftedSelectors], b: &[crate::product::LiftedSelectors]| {
        crate::product::audit_fused_steps(a,b,&lhs_clear,&rhs_clear,&keys.harness,params,&label)
    };
    #[cfg(feature = "noise-audit")]
    let lift_audit: Option<crate::product::LiftAudit<'_>> = if std::env::var_os("FUSED_STEP_AUDIT_CSV").is_some() {
        Some(&lift_audit_callback)
    } else { None };
    #[cfg(not(feature = "noise-audit"))]
    let lift_audit = None;
    let mut evaluation = evaluate_product(encrypted, &keys.evaluator, params, variant, lift_audit)?;
    let expected = expected_low_product_digits(&lhs_clear, &rhs_clear);

    // Measurement-only noise probe: the closure keeps the secret key inside
    // the harness while the evaluator reports probe points blindly.  Probe
    // runs are excluded from the released latency protocol.
    let noise_probe = NoiseProbe::from_env(&label, width_bits)?;
    let probe_closure;
    let probe_fn: Option<NoiseProbeFn> = if let Some(probe) = noise_probe.as_ref() {
        // Fill audit-only expected plaintext values from the clear operands
        // via the public routing provenance, so residuals are measured
        // against the true slot and slot crossings become visible.
        fill_expected_product_digits(&mut evaluation.columns, &lhs_clear, &rhs_clear, variant);
        if let Some(columns) = &mut evaluation.reference_columns {
            fill_expected_product_digits(columns, &lhs_clear, &rhs_clear, variant);
        }
        probe_closure = |stage: &str,
                         wave: usize,
                         column: usize,
                         event: usize,
                         bound: u64,
                         terms: usize,
                         unrefreshed: usize,
                         term_expected: Option<u64>,
                         ciphertext: &LweCiphertextOwned<u64>| {
            probe.record(
                &keys.harness,
                stage,
                wave,
                column,
                event,
                bound,
                terms,
                unrefreshed,
                term_expected,
                ciphertext,
            );
        };
        Some(&probe_closure)
    } else {
        None
    };
    if let Some(probe) = probe_fn {
        if let Some(columns) = &evaluation.reference_columns {
            for (q, column) in columns.iter().enumerate() {
                for (idx, term) in column.iter().enumerate() {
                    probe("product-reference", 0, q, idx, term.bound, 1, 1,
                          term.audit_expected(), &term.ciphertext);
                }
            }
        }
        for (column_idx, column) in evaluation.columns.iter().enumerate() {
            for (term_idx, term) in column.iter().enumerate() {
                probe(
                    "product",
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

    // This is the second half of the released evaluator wall-clock boundary.
    let normalize_started = Instant::now();
    let (normalized, normalization_stats) = normalize_height2(
        evaluation.columns,
        lhs_clear.len(),
        &keys.evaluator,
        params,
        probe_fn,
        plan,
    );
    let normalization_elapsed = normalize_started.elapsed();
    let total_ms = duration_ms(evaluation.elapsed + normalization_elapsed);

    let mut timings = evaluation.timings;
    timings.harness_encrypt_ms = encryption_ms;
    timings.normalization_ms = duration_ms(normalization_elapsed);
    timings.add_normalization(normalization_stats);
    timings.audit_callbacks |= audit_callbacks_requested() || probe_fn.is_some();

    let decrypt_started = Instant::now();
    let observed = decrypt_radix_digits(&normalized, 4, &keys.harness);
    timings.harness_decrypt_ms = duration_ms(decrypt_started.elapsed());
    let correct = observed == expected;
    println!(
        "cmux_lut_product_case label={} d={} row_bits=4 output_row_bits=4 total_ms={:.6} cbs_lifts={} cbs_ms={:.6} cmux_count={} cmux_ms={:.6} normalization_pbs={} normalization_ms={:.6} ok={}",
        label,
        lhs_clear.len(),
        total_ms,
        timings.cbs_lifts,
        timings.cbs_ms,
        timings.cmux_count,
        timings.ext_ms,
        timings.normalization_pbs,
        timings.normalization_ms,
        correct
    );

    append_timing_csv(
        &label, width_bits, lhs_clear.len(), total_ms, &timings, params, variant, correct,
    )?;
    if !correct {
        let mismatch = observed
            .iter()
            .zip(&expected)
            .position(|(actual, expected)| actual != expected)
            .unwrap_or(0);
        bail!(
            "{label}: encrypted product mismatch at radix digit {mismatch}: observed {}, expected {}",
            observed[mismatch],
            expected[mismatch]
        );
    }
    Ok(())
}

/// Derive the expected value of every routed local-product digit from the
/// clear operands via the public `(lhs_start, rhs_start, t)` provenance.
/// Audit-only: called when the noise probe is active.
fn fill_expected_product_digits(
    columns: &mut [Vec<crate::types::ProductDigitTerm>],
    lhs_clear: &[u64],
    rhs_clear: &[u64],
    variant: ProductVariant,
) {
    let chunk_digits = match variant {
        ProductVariant::Hybrid4x4 => 2usize,
        ProductVariant::Hybrid8x8 => 4usize,
    };
    let chunk_value = |digits: &[u64], start: usize| -> u64 {
        (0..chunk_digits)
            .map(|offset| digits[start + offset] << (2 * offset))
            .sum()
    };
    for column in columns.iter_mut() {
        for term in column.iter_mut() {
            if let Some((lhs_start, rhs_start, t)) = term.source {
                let capacity = crate::config::fused_products_per_group();
                let product = if capacity > 0 {
                    // Fused provenance is (native column, first pair index, output digit).
                    let q = lhs_start;
                    (rhs_start..(rhs_start + capacity).min(q + 1))
                        .map(|i| lhs_clear[i] * rhs_clear[q - i])
                        .sum()
                } else {
                    chunk_value(lhs_clear, lhs_start) * chunk_value(rhs_clear, rhs_start)
                };
                term.set_audit_expected(Some((product >> (2 * t)) & 3));
            }
        }
    }
}

/// Deterministic per-trial operands by default; `CBS_RANDOM_OPERANDS_SEED=<u64>`
/// (required by `CACHED_MULT_PATTERN=random`) selects seeded pseudorandom digits so
/// audit campaigns can vary inputs across trials and invocations.
fn generated_radix_digits(
    operands: OperandSource,
    width_bits: usize,
    trial_idx: usize,
    side: usize,
) -> Vec<u64> {
    let digit_count = width_bits.div_ceil(2);
    match operands {
        OperandSource::Zero => return vec![0; digit_count],
        OperandSource::Max => return vec![3; digit_count],
        OperandSource::Alternating => {
            return (0..digit_count)
                .map(|i| if (i + side) % 2 == 0 { 3 } else { 0 })
                .collect()
        }
        OperandSource::Random { .. } | OperandSource::Deterministic => {}
    }
    if let Some(seed) = operands.seed() {
        return (0..digit_count)
            .map(|idx| {
                splitmix64(
                    seed ^ (trial_idx as u64).wrapping_mul(0x9E3779B97F4A7C15)
                        ^ (side as u64).wrapping_mul(0xD1B54A32D192ED03)
                        ^ (idx as u64).wrapping_mul(0x2545F4914F6CDD1D),
                ) % 4
            })
            .collect();
    }
    (0..digit_count)
        .map(|idx| ((3 * trial_idx + 7 * idx + 11 * side + 1) % 4) as u64)
        .collect()
}

fn splitmix64(mut state: u64) -> u64 {
    state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn expected_low_product_digits(lhs: &[u64], rhs: &[u64]) -> Vec<u64> {
    let output_digits = lhs.len();
    let mut diagonal_sums = vec![0u64; output_digits + 1];
    for (lhs_idx, &lhs_digit) in lhs.iter().enumerate() {
        for (rhs_idx, &rhs_digit) in rhs.iter().enumerate() {
            let diagonal = lhs_idx + rhs_idx;
            if diagonal >= output_digits {
                continue;
            }
            let product = lhs_digit * rhs_digit;
            diagonal_sums[diagonal] += product % 4;
            diagonal_sums[diagonal + 1] += product / 4;
        }
    }

    let mut output = Vec::with_capacity(output_digits);
    let mut carry = 0u64;
    for diagonal in diagonal_sums.into_iter().take(output_digits) {
        let total = diagonal + carry;
        output.push(total % 4);
        carry = total / 4;
    }
    output
}

fn print_configuration(params: BenchParams, variant: ProductVariant) {
    println!(
        "cached_table={} shared_preprocessing={}",
        crate::config::cached_product(),
        crate::config::shared_preprocessing()
    );
    println!(
        "selector_lift_geometry centered_ms={} center_selector_box={}",
        u8::from(env_flag("CBS_CENTERED_MS")),
        u8::from(env_flag("CBS_CENTER_SELECTOR_BOX"))
    );
    println!(
        "base_params profile={} N={} k={} small_lwe={} lift_mode=direct-revhomtrace product_mode=cmux-lut product_contract={} normalizer_pbs=({}, {}) cbs_lift_pbs=({}, {}) normalizer_ks=({}, {}) cbs_lift_ks=({}, {})",
        params.profile_label,
        params.polynomial_size.0,
        params.glwe_dimension.0,
        params.lwe_dimension.0,
        variant.product_generator_label(),
        params.pbs_base_log.0,
        params.pbs_level.0,
        params.cbs_pbs_base_log.0,
        params.cbs_pbs_level.0,
        params.ks_base_log.0,
        params.ks_level.0,
        params.cbs_ks_base_log.0,
        params.cbs_ks_level.0
    );
}

fn print_direct_lift_parameters(keys: &KeySet) {
    let direct = &keys.evaluator.direct_lift;
    println!(
        "active_direct_lift_params auto=({}, {}) ss=({}, {}) cbs=({}, {}) log_lut_count={}",
        direct.auto_base_log.0,
        direct.auto_level.0,
        direct.ss_base_log.0,
        direct.ss_level.0,
        direct.cbs_base_log.0,
        direct.cbs_level.0,
        direct.log_lut_count.0
    );
}

fn parse_arg<T>(args: &[String], idx: usize, name: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    let value = args
        .get(idx)
        .with_context(|| format!("missing required argument {name}"))?;
    value
        .parse::<T>()
        .with_context(|| format!("failed to parse {name}={value}"))
}

fn parse_optional_arg<T>(args: &[String], idx: usize, default: T, name: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    match args.get(idx) {
        Some(value) => value
            .parse::<T>()
            .with_context(|| format!("failed to parse {name}={value}")),
        None => Ok(default),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        expected_low_product_digits, explicit_lift_centered_ms, generated_radix_digits,
        OperandSource,
    };

    #[test]
    fn generated_digits_match_the_released_single_product_sequence() {
        let legacy = OperandSource::parse(None, None).unwrap();
        assert_eq!(legacy, OperandSource::Deterministic);
        assert_eq!(
            generated_radix_digits(legacy, 16, 0, 0),
            vec![1, 0, 3, 2, 1, 0, 3, 2]
        );
        assert_eq!(
            generated_radix_digits(legacy, 16, 0, 1),
            vec![0, 3, 2, 1, 0, 3, 2, 1]
        );
    }

    #[test]
    fn random_pattern_requires_a_valid_seed() {
        assert!(OperandSource::parse(Some("random"), None).is_err());
        for bad in ["", "-1", "1.5", "18446744073709551616", "x"] {
            assert!(OperandSource::parse(Some("random"), Some(bad)).is_err(), "{bad}");
            assert!(OperandSource::parse(None, Some(bad)).is_err(), "{bad}");
        }
        assert!(OperandSource::parse(Some("typo"), Some("1")).is_err());
        let random = OperandSource::parse(Some("random"), Some("20260905")).unwrap();
        assert_eq!(random.seed(), Some(20260905));
        assert_eq!(random.pattern_label(), Some("random"));
        let implicit = OperandSource::parse(None, Some("20260905")).unwrap();
        // The legacy seeded path keeps its digit sequence.
        assert_eq!(generated_radix_digits(implicit, 64, 3, 1), generated_radix_digits(random, 64, 3, 1));
        assert_ne!(generated_radix_digits(random, 64, 0, 0), generated_radix_digits(random, 64, 1, 0));
    }

    #[test]
    fn lift_modulus_switch_mode_is_explicit() {
        assert!(explicit_lift_centered_ms(None).is_err());
        assert!(explicit_lift_centered_ms(Some("true")).is_err());
        assert!(explicit_lift_centered_ms(Some("")).is_err());
        assert!(explicit_lift_centered_ms(Some("1")).unwrap());
        assert!(!explicit_lift_centered_ms(Some("0")).unwrap());
    }

    #[test]
    fn clear_oracle_returns_the_low_half_product() {
        assert_eq!(expected_low_product_digits(&[3, 1], &[2, 3]), vec![2, 0]);
    }
}
