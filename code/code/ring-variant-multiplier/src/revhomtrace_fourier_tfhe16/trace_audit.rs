//! Paired exact-integer/FFT audit. Never compiled into the evaluator binary.
use super::*;
use std::path::Path;
use tfhe::core_crypto::algorithms::polynomial_algorithms::polynomial_wrapping_add_mul_assign;

fn phase(c: &GlweCiphertextOwned<u64>, secret: &GlweSecretKeyOwned<u64>) -> Vec<u64> {
    let mut out = PlaintextList::new(0, PlaintextCount(c.polynomial_size().0));
    decrypt_glwe_ciphertext(secret, c, &mut out);
    out.into_container()
}

fn product(a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut out = Polynomial::new(0, PolynomialSize(a.len()));
    polynomial_wrapping_add_mul_assign(
        &mut out,
        &Polynomial::from_container(a),
        &Polynomial::from_container(b),
    );
    out.into_container()
}

fn exact_auto(
    input: &GlweCiphertextOwned<u64>,
    output: &mut GlweCiphertextOwned<u64>,
    key: &GlweKeyswitchKeyOwned<u64>,
) {
    let row_len = input.polynomial_size().0 * output.glwe_size().0;
    let block_len = row_len * key.decomposition_level_count().0;
    let reversed: Vec<_> = key
        .as_ref()
        .chunks_exact(block_len)
        .flat_map(|block| block.chunks_exact(row_len).rev().flatten().copied())
        .collect();
    let mut ordered = key.clone();
    ordered.as_mut().copy_from_slice(&reversed);
    tfhe::core_crypto::algorithms::keyswitch_glwe_ciphertext(&ordered, input, output);
    // This repository's keys encrypt -sigma(S) and its FFT KS adds them.
    // The library also stores levels in reverse order and subtracts the key:
    // negate its output and restore 2b after adapting the row order above.
    for value in output.as_mut() {
        *value = value.wrapping_neg();
    }
    for (b, &original) in output
        .get_mut_body()
        .as_mut()
        .iter_mut()
        .zip(input.get_body().as_ref())
    {
        *b = b.wrapping_add(original.wrapping_mul(2));
    }
}

#[test]
fn exact_adapter_matches_signed_polynomial_decomposition() {
    let n = 16;
    let level = DecompositionLevelCount(9);
    let base = DecompositionBaseLog(4);
    let zero = || {
        GlweCiphertext::new(
            0u64,
            GlweSize(2),
            PolynomialSize(n),
            CiphertextModulus::new_native(),
        )
    };
    let mut key = GlweKeyswitchKey::new(
        0u64,
        base,
        level,
        GlweDimension(1),
        GlweDimension(1),
        PolynomialSize(n),
        CiphertextModulus::new_native(),
    );
    for (i, value) in key.as_mut().iter_mut().enumerate() {
        *value = (i as u64 + 1).wrapping_mul(0x9e3779b97f4a7c15);
    }
    let mut input = zero();
    for (i, value) in input.as_mut().iter_mut().enumerate() {
        *value = (i as u64 + 7).wrapping_mul(0xd1b54a32d192ed03);
    }
    let mut expected = input.clone();
    expected.get_mut_mask().as_mut().fill(0);
    let decomposer = SignedDecomposer::new(base, level);
    let mut digits = vec![vec![0u64; n]; level.0];
    for (i, &value) in input.get_mask().as_ref().iter().enumerate() {
        for (j, digit) in decomposer.decompose(value).enumerate() {
            digits[j][i] = digit.value();
        }
    }
    for (j, digit) in digits.iter().enumerate() {
        let row = &key.as_ref()[(level.0 - 1 - j) * 2 * n..(level.0 - j) * 2 * n];
        for (component, key_poly) in row.chunks_exact(n).enumerate() {
            let contribution = product(digit, key_poly);
            for (out, value) in expected.as_mut()[component * n..(component + 1) * n]
                .iter_mut()
                .zip(contribution)
            {
                *out = out.wrapping_add(value);
            }
        }
    }
    let mut actual = zero();
    exact_auto(&input, &mut actual, &key);
    assert_eq!(actual.as_ref(), expected.as_ref());
}

fn audit(n: usize, trials: usize, directory: &Path) {
    std::fs::create_dir_all(directory).unwrap();
    let mut seeder = new_seeder();
    let mut srng = SecretRandomGenerator::<DefaultRandomGenerator>::new(seeder.seed());
    let mut rng =
        EncryptionRandomGenerator::<DefaultRandomGenerator>::new(seeder.seed(), seeder.as_mut());
    let noise = DynamicDistribution::new_gaussian_from_std_dev(StandardDev(
        8.095547030480233e-30_f64.sqrt(),
    ));
    let sk = allocate_and_generate_new_binary_glwe_secret_key(
        GlweDimension(1),
        PolynomialSize(n),
        &mut srng,
    );
    let base = DecompositionBaseLog(9);
    let level = DecompositionLevelCount(5);
    let mut keys = Vec::new();
    for r in 1..=n.ilog2() {
        let exponent = (1usize << r) + 1;
        let sigma_sk = GlweSecretKey::from_container(
            eval_x_k_owned(sk.as_polynomial_list().get(0), exponent).into_container(),
            PolynomialSize(n),
        );
        let standard =
            generate_revhomtrace_glwe_keyswitch_key(&sigma_sk, &sk, base, level, noise, &mut rng);
        let mut fourier = FourierGlweKeyswitchKey::new(
            GlweSize(2),
            GlweSize(2),
            PolynomialSize(n),
            base,
            level,
            FftType::Vanilla,
        );
        convert_standard_glwe_keyswitch_key_to_fourier(&standard, &mut fourier);
        let mut split = FourierGlweKeyswitchKey::new(
            GlweSize(2),
            GlweSize(2),
            PolynomialSize(n),
            base,
            level,
            FftType::Split(40),
        );
        convert_standard_glwe_keyswitch_key_to_fourier(&standard, &mut split);
        keys.push((exponent, standard, fourier, split));
    }
    let mut local = csv::Writer::from_path(directory.join("local.csv")).unwrap();
    local
        .write_record([
            "trial",
            "stage",
            "stride",
            "coefficient",
            "delta_a",
            "delta_b",
            "phase_fft",
            "phase_crypto",
            "split_delta_a",
            "split_delta_b",
            "split_phase_fft",
        ])
        .unwrap();
    let mut final_rows = csv::Writer::from_path(directory.join("final.csv")).unwrap();
    final_rows
        .write_record([
            "trial",
            "coefficient",
            "trace_error",
            "trace_fft",
            "trace_other",
            "post_error",
            "post_fft",
            "post_other",
        ])
        .unwrap();
    let zero = || {
        GlweCiphertext::new(
            0,
            GlweSize(2),
            PolynomialSize(n),
            CiphertextModulus::new_native(),
        )
    };
    for trial in 0..trials {
        let mut state = zero();
        encrypt_glwe_ciphertext(
            &sk,
            &mut state,
            &PlaintextList::new(0, PlaintextCount(n)),
            noise,
            &mut rng,
        );
        let input_constant = phase(&state, &sk)[0];
        let mut fft_total = vec![0u64; n];
        for (j, (exponent, standard, fourier, split)) in keys.iter().enumerate() {
            glwe_preprocessing_moddown_1bit(&mut state);
            let mut sigma = zero();
            for (mut out, input) in sigma
                .as_mut_polynomial_list()
                .iter_mut()
                .zip(state.as_polynomial_list().iter())
            {
                out.as_mut()
                    .copy_from_slice(eval_x_k_owned(input, *exponent).as_ref());
            }
            let mut exact = zero();
            exact_auto(&sigma, &mut exact, standard);
            let mut floating = zero();
            super::super::fourier_glwe_keyswitch::keyswitch_glwe_ciphertext(
                fourier,
                &sigma,
                &mut floating,
            );
            let mut delta = zero();
            for ((d, &a), &b) in delta
                .as_mut()
                .iter_mut()
                .zip(floating.as_ref())
                .zip(exact.as_ref())
            {
                *d = a.wrapping_sub(b);
            }
            let fft_phase = phase(&delta, &sk);
            let mut split_delta = zero();
            super::super::fourier_glwe_keyswitch::keyswitch_glwe_ciphertext(
                split,
                &sigma,
                &mut split_delta,
            );
            for (d, &x) in split_delta.as_mut().iter_mut().zip(exact.as_ref()) {
                *d = d.wrapping_sub(x);
            }
            let split_phase = phase(&split_delta, &sk);
            let exact_phase = phase(&exact, &sk);
            let ideal = eval_x_k_owned(
                Polynomial::from_container(phase(&state, &sk).as_slice()),
                *exponent,
            )
            .into_container();
            let stride = n >> (j + 1);
            for i in 0..n {
                let crypto = exact_phase[i].wrapping_sub(ideal[i]);
                assert!(
                    (crypto as i64).unsigned_abs() < (1u64 << 48),
                    "exact KS sign/order mismatch"
                );
                local
                    .write_record([
                        trial.to_string(),
                        (j + 1).to_string(),
                        stride.to_string(),
                        i.to_string(),
                        (delta.as_ref()[i] as i64).to_string(),
                        (delta.as_ref()[n + i] as i64).to_string(),
                        (fft_phase[i] as i64).to_string(),
                        (crypto as i64).to_string(),
                        (split_delta.as_ref()[i] as i64).to_string(),
                        (split_delta.as_ref()[n + i] as i64).to_string(),
                        (split_phase[i] as i64).to_string(),
                    ])
                    .unwrap();
                if i % stride == 0 {
                    fft_total[i] = fft_total[i].wrapping_add(fft_phase[i]);
                }
            }
            glwe_ciphertext_add_assign(&mut state, &floating);
        }
        let mut error = phase(&state, &sk);
        error[0] = error[0].wrapping_sub(input_constant);
        let other: Vec<_> = error
            .iter()
            .zip(&fft_total)
            .map(|(&a, &b)| a.wrapping_sub(b))
            .collect();
        let s = sk.as_ref();
        let post = product(s, &error);
        let post_fft = product(s, &fft_total);
        let post_other = product(s, &other);
        for i in 0..n {
            assert_eq!(post[i], post_fft[i].wrapping_add(post_other[i]));
            assert!(
                (error[i] as i64).unsigned_abs() < (1u64 << 48),
                "trace correctness mismatch"
            );
            final_rows
                .write_record([
                    trial.to_string(),
                    i.to_string(),
                    (error[i] as i64).to_string(),
                    (fft_total[i] as i64).to_string(),
                    (other[i] as i64).to_string(),
                    (post[i] as i64).to_string(),
                    (post_fft[i] as i64).to_string(),
                    (post_other[i] as i64).to_string(),
                ])
                .unwrap();
        }
    }
    local.flush().unwrap();
    final_rows.flush().unwrap();
}

#[test]
#[ignore = "writes secret-key-assisted diagnostic observations; set TRACE_FFT_AUDIT_OUT"]
fn paired_trace_fft_audit() {
    let out = std::env::var("TRACE_FFT_AUDIT_OUT").expect("fresh output directory required");
    let n: usize = std::env::var("TRACE_FFT_AUDIT_N")
        .unwrap_or("2048".into())
        .parse()
        .unwrap();
    let trials: usize = std::env::var("TRACE_FFT_AUDIT_TRIALS")
        .unwrap_or("3".into())
        .parse()
        .unwrap();
    assert!(n.is_power_of_two() && n >= 2 && trials > 0);
    assert!(
        !Path::new(&out).exists(),
        "do not overwrite an earlier audit"
    );
    audit(n, trials, Path::new(&out));
}
