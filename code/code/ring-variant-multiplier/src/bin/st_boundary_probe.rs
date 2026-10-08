//! Tests the literal post-PBS small-key CC2 boundary in revised 2026/810.
//! This isolates key switching, with no PBS/trace/CMux error at its input.
use anyhow::{ensure, Result};
use clap::Parser;
use serde_json::json;
use std::path::PathBuf;
use tfhe::core_crypto::prelude::*;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 64)]
    samples_per_digit: usize,
    #[arg(long, default_value_t = 2)]
    ks_base_log: usize,
    #[arg(long, default_value_t = 7)]
    ks_level: usize,
    #[arg(long, default_value_t = 0.0)]
    key_noise_sigma: f64,
    #[arg(long)]
    pbs_input: bool,
    #[arg(long, default_value_t = 0.0)]
    glwe_noise_sigma: f64,
    #[arg(long)]
    output: PathBuf,
}

fn decode(phase: u64, delta_log: u32) -> u64 {
    phase.wrapping_add(1u64 << (delta_log - 1)) >> delta_log
}

fn emission_accumulator() -> GlweCiphertextOwned<u64> {
    let mut body: Vec<u64> = (0..1024).map(|i| ((i / 256) as u64) << 52).collect();
    for value in &mut body[..128] { *value = value.wrapping_neg(); }
    body.rotate_left(128);
    allocate_and_trivially_encrypt_new_glwe_ciphertext(
        GlweSize(3), &PlaintextList::from_container(body), CiphertextModulus::new_native())
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.samples_per_digit > 0, "positive sample count required");
    ensure!(args.ks_base_log > 0 && args.ks_level > 0
        && args.ks_base_log * args.ks_level < 64, "signed decomposition requires fewer than 64 bits");
    ensure!(args.key_noise_sigma.is_finite() && args.key_noise_sigma >= 0.0,
        "invalid Gaussian standard deviation");
    ensure!(args.glwe_noise_sigma.is_finite() && args.glwe_noise_sigma >= 0.0,
        "invalid GLWE Gaussian standard deviation");
    ensure!(!args.output.exists(), "use a fresh output directory");
    std::fs::create_dir_all(&args.output)?;
    let mut seeder = new_seeder();
    let mut secret_rng = SecretRandomGenerator::<DefaultRandomGenerator>::new(seeder.seed());
    let mut enc_rng = EncryptionRandomGenerator::<DefaultRandomGenerator>::new(
        seeder.seed(), seeder.as_mut());
    let input_key = allocate_and_generate_new_binary_lwe_secret_key(LweDimension(2048), &mut secret_rng);
    let output_key = allocate_and_generate_new_binary_lwe_secret_key(LweDimension(768), &mut secret_rng);
    let modulus = CiphertextModulus::new_native();
    let zero_noise = DynamicDistribution::new_gaussian_from_std_dev(StandardDev(0.0));
    let key_noise = DynamicDistribution::new_gaussian_from_std_dev(StandardDev(args.key_noise_sigma));
    let key = allocate_and_generate_new_lwe_keyswitch_key(
        &input_key, &output_key, DecompositionBaseLog(args.ks_base_log),
        DecompositionLevelCount(args.ks_level), key_noise, modulus, &mut enc_rng);
    let bsk = if args.pbs_input {
        let glwe_key = GlweSecretKey::from_container(input_key.as_ref().to_vec(), PolynomialSize(1024));
        let standard = allocate_and_generate_new_lwe_bootstrap_key(
            &output_key, &glwe_key, DecompositionBaseLog(15), DecompositionLevelCount(2),
            DynamicDistribution::new_gaussian_from_std_dev(StandardDev(args.glwe_noise_sigma)),
            modulus, &mut enc_rng);
        let mut fourier = FourierLweBootstrapKey::new(LweDimension(768), GlweSize(3),
            PolynomialSize(1024), DecompositionBaseLog(15), DecompositionLevelCount(2));
        convert_standard_lwe_bootstrap_key_to_fourier(&standard, &mut fourier);
        Some(fourier)
    } else { None };
    let accumulator = emission_accumulator();
    let mut writer = csv::Writer::from_path(args.output.join("samples.csv"))?;
    writer.write_record(["digit", "trial", "input_phase", "output_phase", "signed_error",
        "decoded_CC2", "cc2_ok", "scaled_lift_phase_error", "predicted_rounding_error"])?;
    let mut correct = 0;
    let mut correct_before_ks = 0;
    let mut squares = 0.0;
    let mut sum = 0.0;
    for digit in 0..4u64 {
        for trial in 0..args.samples_per_digit {
            let expected = digit << 52;
            let input = if let Some(bsk) = &bsk {
                let narrow = allocate_and_encrypt_new_lwe_ciphertext(
                    &output_key, Plaintext(digit << 61), zero_noise, modulus, &mut enc_rng);
                let mut extracted = LweCiphertext::new(0u64, LweSize(2049), modulus);
                programmable_bootstrap_lwe_ciphertext(&narrow, &mut extracted, &accumulator, bsk);
                extracted
            } else {
                allocate_and_encrypt_new_lwe_ciphertext(
                    &input_key, Plaintext(expected), zero_noise, modulus, &mut enc_rng)
            };
            let before = decrypt_lwe_ciphertext(&input_key, &input).0;
            if !args.pbs_input { ensure!(before == expected, "zero-noise input is not exact"); }
            correct_before_ks += usize::from(decode(before, 52) == digit);
            let decomposer = SignedDecomposer::<u64>::new(
                DecompositionBaseLog(args.ks_base_log), DecompositionLevelCount(args.ks_level));
            let rounding = input.get_mask().as_ref().iter().zip(input_key.as_ref())
                .fold(0u64, |sum, (&a, &s)| sum.wrapping_add(
                    a.wrapping_sub(decomposer.closest_representable(a)).wrapping_mul(s)));
            let mut output = LweCiphertext::new(0u64, LweSize(769), modulus);
            keyswitch_lwe_ciphertext(&key, &input, &mut output);
            let phase = decrypt_lwe_ciphertext(&output_key, &output).0;
            let error = phase.wrapping_sub(expected) as i64;
            if args.key_noise_sigma == 0.0 {
                ensure!(phase.wrapping_sub(before) == rounding, "key-switch rounding identity failed");
            }
            let decoded = decode(phase, 52);
            let ok = decoded == digit;
            correct += usize::from(ok);
            sum += error as f64;
            squares += (error as f64).powi(2);
            writer.write_record([digit.to_string(), trial.to_string(), before.to_string(),
                phase.to_string(), error.to_string(), decoded.to_string(), ok.to_string(),
                (error.wrapping_mul(1024)).to_string(), (rounding as i64).to_string()])?;
        }
    }
    writer.flush()?;
    let count = 4 * args.samples_per_digit;
    let rms = (squares / count as f64).sqrt();
    let report = json!({
        "experiment": "literal revised-ST CC2 output-key-switch precision probe",
        "not_a_multiplier_benchmark": true,
        "source": "2026/810 revision 2026-07-30, Sections 2/3.1 and Table 2",
        "input_lwe_dimension": 2048, "output_lwe_dimension": 768,
        "ks_base_log": args.ks_base_log, "ks_level": args.ks_level,
        "key_noise_sigma": args.key_noise_sigma,
        "pbs_input": args.pbs_input, "pbs_base_log": 15, "pbs_level": 2,
        "glwe_noise_sigma": args.glwe_noise_sigma, "before_ks_correct": correct_before_ks,
        "input_noise_sigma": 0, "secret_distribution": "binary",
        "encoding_delta_log2": 52,
        "input_preparation": if args.pbs_input {"actual identity PBS from padded two-bit input to CC2"} else {"exact noiseless phase with uniform LWE mask"},
        "tfhe_rs": "1.6.1", "samples": count, "cc2_correct": correct,
        "cc2_incorrect": count - correct, "mean_error": sum / count as f64,
        "rms_error": rms, "rms_in_message_steps": rms / (2.0f64).powi(52),
        "scope": "Does not reproduce any unpublished author-side correction or alternate boundary. Zero key noise, when requested, isolates decomposition rounding; it is not a secure parameter set."
    });
    std::fs::write(args.output.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
    println!("CC2 correct before KS {correct_before_ks}/{count}, after KS {correct}/{count}; RMS error / step = {:.4}", rms / (2.0f64).powi(52));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cc2_decoding_uses_the_full_padded_domain_not_mod_four() {
        for digit in 0..4 {
            assert_eq!(decode(digit << 52, 52), digit);
            assert_eq!(decode((digit << 52) + (1 << 51) - 1, 52), digit);
        }
        assert_eq!(decode(4 << 52, 52), 4);
        assert_eq!(decode(0u64.wrapping_sub(1 << 52), 52), 4095);
    }
}
