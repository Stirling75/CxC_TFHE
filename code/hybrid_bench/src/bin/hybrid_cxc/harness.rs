use crate::config::{validate_runtime_contract, BenchParams, ProductVariant};
use crate::keys::{
    decrypt_radix_digits, encrypt_radix_digits, generate_keys, KeySet, RngResources,
};
use crate::normalize::normalize_height2;
use crate::product::{evaluate_product, EncryptedOperands};
use crate::report::append_timing_csv;
use anyhow::{bail, Context, Result};
use std::env;
use std::time::Instant;

pub(crate) fn run() -> Result<()> {
    let args = env::args().collect::<Vec<_>>();
    let width_bits = parse_arg(&args, 1, "width_bits")?;
    let product_count = parse_optional_arg(&args, 2, 1usize, "product_count")?;
    let trial_count = parse_optional_arg(&args, 3, 1usize, "trial_count")?;
    validate_run_shape(width_bits, product_count, trial_count)?;
    validate_runtime_contract()?;

    let variant = ProductVariant::from_env()?;
    let params = BenchParams::from_env()?;
    print_configuration(params, variant);

    let mut rng = RngResources::new();
    let keygen = Instant::now();
    let keys = generate_keys(params, &mut rng);
    println!("keygen_ms={}", keygen.elapsed().as_millis());
    print_direct_lift_parameters(&keys);
    rng.keep_seeder_alive();

    for trial_idx in 0..trial_count {
        run_trial(width_bits, trial_idx, variant, &keys, params, &mut rng)?;
    }
    println!("ok=true");
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
) -> Result<()> {
    let lhs_clear = generated_radix_digits(width_bits, trial_idx, 0);
    let rhs_clear = generated_radix_digits(width_bits, trial_idx, 1);
    let label = format!(
        "generated_w{width_bits}_single_trial{trial_idx}{}",
        variant.label_suffix()
    );

    let encrypt_started = Instant::now();
    let encrypted = EncryptedOperands {
        lhs: encrypt_radix_digits(&lhs_clear, &keys.harness, params, rng),
        rhs: encrypt_radix_digits(&rhs_clear, &keys.harness, params, rng),
    };
    let encryption_ms = encrypt_started.elapsed().as_millis();

    let evaluation = evaluate_product(encrypted, &keys.evaluator, params, variant)?;
    let expected = expected_low_product_digits(&lhs_clear, &rhs_clear);

    // This is the second half of the released evaluator wall-clock boundary.
    let normalize_started = Instant::now();
    let (normalized, normalization_stats) =
        normalize_height2(evaluation.columns, lhs_clear.len(), &keys.evaluator, params);
    let normalization_elapsed = normalize_started.elapsed();
    let total_ms = (evaluation.elapsed + normalization_elapsed).as_millis();

    let mut timings = evaluation.timings;
    timings.harness_encrypt_ms = encryption_ms;
    timings.normalization_ms = normalization_elapsed.as_millis();
    timings.add_normalization(normalization_stats);

    let decrypt_started = Instant::now();
    let observed = decrypt_radix_digits(&normalized, 4, &keys.harness);
    timings.harness_decrypt_ms = decrypt_started.elapsed().as_millis();
    let correct = observed == expected;
    println!(
        "cmux_lut_product_case label={} d={} row_bits=4 output_row_bits=4 total_ms={} cbs_lifts={} cbs_ms={} cmux_count={} cmux_ms={} normalization_pbs={} normalization_ms={} ok={}",
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
    append_timing_csv(
        &label,
        width_bits,
        lhs_clear.len(),
        total_ms,
        &timings,
        params,
        variant,
    )?;
    Ok(())
}

fn generated_radix_digits(width_bits: usize, trial_idx: usize, side: usize) -> Vec<u64> {
    let digit_count = width_bits.div_ceil(2);
    (0..digit_count)
        .map(|idx| ((3 * trial_idx + 7 * idx + 11 * side + 1) % 4) as u64)
        .collect()
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
    println!("tfhe16_cbs_product_smoke=true");
    println!(
        "base_params profile={} N={} k={} small_lwe={} lift_mode=direct-revhomtrace product_mode=cmux-lut product_contract={} normalizer_pbs=({}, {}) cbs_lift_pbs=({}, {}) pfks=(15, 2) profile_cbs_default=(4, 4) normalizer_ks=({}, {}) cbs_lift_ks=({}, {})",
        params.profile_label,
        params.polynomial_size.0,
        params.glwe_dimension.0,
        params.lwe_dimension.0,
        variant.preset_label(),
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
    use super::{expected_low_product_digits, generated_radix_digits};

    #[test]
    fn generated_digits_match_the_released_single_product_sequence() {
        assert_eq!(
            generated_radix_digits(16, 0, 0),
            vec![1, 0, 3, 2, 1, 0, 3, 2]
        );
        assert_eq!(
            generated_radix_digits(16, 0, 1),
            vec![0, 3, 2, 1, 0, 3, 2, 1]
        );
    }

    #[test]
    fn clear_oracle_returns_the_low_half_product() {
        assert_eq!(expected_low_product_digits(&[3, 1], &[2, 3]), vec![2, 0]);
    }
}
