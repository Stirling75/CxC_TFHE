use crate::config::{
    centered_normalizer_modulus_switch, decode_scalar, encoding_delta, BenchParams, ProductVariant,
};
use crate::{revhomtrace_fourier_tfhe16, revhomtrace_lift_tfhe16};
use dyn_stack::{PodBuffer, PodStack};
use std::collections::HashMap;
use tfhe::core_crypto::fft_impl::fft64::crypto::bootstrap as fft_bootstrap;
use tfhe::core_crypto::prelude::*;
use tfhe::integer::ServerKey as IntegerServerKey;
use tfhe::shortint::atomic_pattern::{AtomicPatternServerKey, StandardAtomicPatternServerKey};
use tfhe::shortint::ciphertext::{MaxDegree, MaxNoiseLevel};
use tfhe::shortint::parameters::{CarryModulus, MessageModulus};
use tfhe::shortint::server_key::{ModulusSwitchConfiguration, ShortintBootstrappingKey};
use tfhe::shortint::PBSOrder;

pub(crate) struct HarnessSecretKeys {
    big_lwe_sk: LweSecretKeyOwned<u64>,
}

pub(crate) struct DirectLiftKeys {
    pub(crate) auto_keys: HashMap<usize, revhomtrace_fourier_tfhe16::AutomorphKeyOwned>,
    pub(crate) ss_key: revhomtrace_lift_tfhe16::FourierGgswListOwned,
    pub(crate) auto_base_log: DecompositionBaseLog,
    pub(crate) auto_level: DecompositionLevelCount,
    pub(crate) ss_base_log: DecompositionBaseLog,
    pub(crate) ss_level: DecompositionLevelCount,
    pub(crate) cbs_base_log: DecompositionBaseLog,
    pub(crate) cbs_level: DecompositionLevelCount,
    pub(crate) log_lut_count: LutCountLog,
}

pub(crate) struct EvaluationKeys {
    pub(crate) normalizer_radix_sks: IntegerServerKey,
    pub(crate) cbs_fourier_bsk: FourierLweBootstrapKeyOwned,
    pub(crate) cbs_big_to_small_ksk: LweKeyswitchKeyOwned<u64>,
    pub(crate) direct_lift: DirectLiftKeys,
}

impl EvaluationKeys {
    fn normalizer_atomic_pattern(&self) -> &StandardAtomicPatternServerKey {
        let shortint_sks: &tfhe::shortint::ServerKey = self.normalizer_radix_sks.as_ref();
        let AtomicPatternServerKey::Standard(pattern) = &shortint_sks.atomic_pattern else {
            panic!("hybrid normalizer requires the standard TFHE atomic pattern");
        };
        pattern
    }

    pub(crate) fn normalizer_fourier_bsk(&self) -> &FourierLweBootstrapKeyOwned {
        let ShortintBootstrappingKey::Classic { bsk, .. } =
            &self.normalizer_atomic_pattern().bootstrapping_key
        else {
            panic!("hybrid normalizer requires a classic Fourier bootstrap key");
        };
        bsk
    }

    pub(crate) fn normalizer_big_to_small_ksk(&self) -> &LweKeyswitchKeyOwned<u64> {
        &self.normalizer_atomic_pattern().key_switching_key
    }
}

pub(crate) struct KeySet {
    pub(crate) harness: HarnessSecretKeys,
    pub(crate) evaluator: EvaluationKeys,
}

pub(crate) struct RngResources {
    seeder: Box<dyn Seeder>,
    pub(crate) encryption: EncryptionRandomGenerator<DefaultRandomGenerator>,
    secret: SecretRandomGenerator<DefaultRandomGenerator>,
}

impl RngResources {
    pub(crate) fn new() -> Self {
        let mut seeder = new_seeder();
        let encryption = EncryptionRandomGenerator::new(seeder.seed(), seeder.as_mut());
        let secret = SecretRandomGenerator::new(seeder.seed());
        Self {
            seeder,
            encryption,
            secret,
        }
    }

    pub(crate) fn keep_seeder_alive(&self) {
        let _ = &self.seeder;
    }
}

pub(crate) fn generate_keys(
    params: BenchParams,
    variant: ProductVariant,
    rng: &mut RngResources,
) -> KeySet {
    let normalizer_params = params.normalizer_params();
    let normalizer_glwe_sk = allocate_and_generate_new_binary_glwe_secret_key(
        normalizer_params.glwe_dimension,
        normalizer_params.polynomial_size,
        &mut rng.secret,
    );
    let small_lwe_sk =
        allocate_and_generate_new_binary_lwe_secret_key(params.lwe_dimension, &mut rng.secret);
    let big_lwe_sk = normalizer_glwe_sk.clone().into_lwe_secret_key();
    // The even/odd option is the natural degree-two subring embedding.
    // Its extracted LWE mask is permuted back before column normalization.
    let product_secret = if params.even_odd_secret {
        crate::ring_key::even_odd_secret(big_lwe_sk.as_ref())
    } else { big_lwe_sk.as_ref().to_vec() };
    let glwe_sk = GlweSecretKey::from_container(
        product_secret, params.polynomial_size,
    );
    assert_eq!(glwe_sk.glwe_dimension(), params.glwe_dimension);

    let normalizer_fourier_bsk = generate_fourier_bootstrap_key(
        &small_lwe_sk,
        &normalizer_glwe_sk,
        params.pbs_base_log,
        params.pbs_level,
        normalizer_params,
        rng,
    );
    let cbs_fourier_bsk = generate_fourier_bootstrap_key(
        &small_lwe_sk,
        &glwe_sk,
        params.cbs_pbs_base_log,
        params.cbs_pbs_level,
        params,
        rng,
    );

    let normalizer_big_to_small_ksk = allocate_and_generate_new_lwe_keyswitch_key(
        &big_lwe_sk,
        &small_lwe_sk,
        params.ks_base_log,
        params.ks_level,
        params.lwe_noise_distribution,
        params.ciphertext_modulus,
        &mut rng.encryption,
    );
    let normalizer_radix_sks =
        build_normalizer_server_key(normalizer_big_to_small_ksk, normalizer_fourier_bsk);
    let cbs_big_to_small_ksk = allocate_and_generate_new_lwe_keyswitch_key(
        &big_lwe_sk,
        &small_lwe_sk,
        params.cbs_ks_base_log,
        params.cbs_ks_level,
        params.lwe_noise_distribution,
        params.ciphertext_modulus,
        &mut rng.encryption,
    );
    let direct_lift = generate_direct_lift_keys(&glwe_sk, params, variant, rng);

    KeySet {
        harness: HarnessSecretKeys { big_lwe_sk },
        evaluator: EvaluationKeys {
            normalizer_radix_sks,
            cbs_fourier_bsk,
            cbs_big_to_small_ksk,
            direct_lift,
        },
    }
}

fn generate_fourier_bootstrap_key(
    small_lwe_sk: &LweSecretKeyOwned<u64>,
    glwe_sk: &GlweSecretKeyOwned<u64>,
    base_log: DecompositionBaseLog,
    level: DecompositionLevelCount,
    params: BenchParams,
    rng: &mut RngResources,
) -> FourierLweBootstrapKeyOwned {
    let standard = allocate_and_generate_new_lwe_bootstrap_key(
        small_lwe_sk,
        glwe_sk,
        base_log,
        level,
        params.glwe_noise_distribution,
        params.ciphertext_modulus,
        &mut rng.encryption,
    );
    let fft = Fft::new(params.polynomial_size);
    let fft = fft.as_view();
    let mut fourier = FourierLweBootstrapKey::new(
        params.lwe_dimension,
        params.glwe_dimension.to_glwe_size(),
        params.polynomial_size,
        base_log,
        level,
    );
    let mut memory = PodBuffer::try_new(fft_bootstrap::fill_with_forward_fourier_scratch(fft))
        .expect("allocate bootstrap-key Fourier scratch");
    fourier.as_mut_view().fill_with_forward_fourier(
        standard.as_view(),
        fft,
        &mut PodStack::new(&mut memory),
    );
    fourier
}

fn build_normalizer_server_key(
    key_switching_key: LweKeyswitchKeyOwned<u64>,
    bsk: FourierLweBootstrapKeyOwned,
) -> IntegerServerKey {
    let message_modulus = MessageModulus(4);
    let carry_modulus = CarryModulus(4);
    let atomic_pattern = StandardAtomicPatternServerKey::from_raw_parts(
        key_switching_key,
        ShortintBootstrappingKey::Classic {
            bsk,
            modulus_switch_noise_reduction_key: if centered_normalizer_modulus_switch() {
                ModulusSwitchConfiguration::CenteredMeanNoiseReduction
            } else {
                ModulusSwitchConfiguration::Standard
            },
        },
        PBSOrder::KeyswitchBootstrap,
    );
    let shortint = tfhe::shortint::ServerKey::from_raw_parts(
        AtomicPatternServerKey::Standard(atomic_pattern),
        message_modulus,
        carry_modulus,
        MaxDegree::from_msg_carry_modulus(message_modulus, carry_modulus),
        MaxNoiseLevel::from_msg_carry_modulus(message_modulus, carry_modulus),
    );
    IntegerServerKey::new_radix_server_key_from_shortint(shortint)
}

fn generate_direct_lift_keys(
    glwe_sk: &GlweSecretKeyOwned<u64>,
    params: BenchParams,
    variant: ProductVariant,
    rng: &mut RngResources,
) -> DirectLiftKeys {
    let (auto_base_log, auto_level, ss_base_log, ss_level, cbs_base_log, cbs_level, log_lut_count) =
        params.direct_lift_decomposition(variant);
    let auto_keys = revhomtrace_fourier_tfhe16::gen_all_auto_keys(
        auto_base_log,
        auto_level,
        params.auto_fft.fft_type(),
        glwe_sk,
        params.glwe_noise_distribution,
        &mut rng.encryption,
    );
    for key in auto_keys.values() {
        let actual = key.fft_type();
        assert_eq!(actual.num_split(), params.auto_fft.fft_type().num_split());
        assert_eq!(actual.split_base_log(), params.auto_fft.fft_type().split_base_log());
    }
    let ss_key = revhomtrace_lift_tfhe16::generate_scheme_switching_key_standard(
        glwe_sk,
        ss_base_log,
        ss_level,
        params.glwe_noise_distribution,
        params.ciphertext_modulus,
        &mut rng.encryption,
    );
    DirectLiftKeys {
        auto_keys,
        ss_key,
        auto_base_log,
        auto_level,
        ss_base_log,
        ss_level,
        cbs_base_log,
        cbs_level,
        log_lut_count,
    }
}

pub(crate) fn encrypt_radix_digits(
    digits: &[u64],
    secrets: &HarnessSecretKeys,
    params: BenchParams,
    rng: &mut RngResources,
) -> Vec<LweCiphertextOwned<u64>> {
    digits
        .iter()
        .map(|&digit| {
            assert!(digit < 4);
            let mut lwe = LweCiphertext::new(
                0u64,
                secrets.big_lwe_sk.lwe_dimension().to_lwe_size(),
                params.ciphertext_modulus,
            );
            encrypt_lwe_ciphertext(
                &secrets.big_lwe_sk,
                &mut lwe,
                Plaintext(digit.wrapping_mul(encoding_delta(4))),
                params.glwe_noise_distribution,
                &mut rng.encryption,
            );
            lwe
        })
        .collect()
}

/// Raw decrypted phase for the measurement-only noise probe.  The secret key
/// never leaves this module; callers receive only the plaintext-plus-error
/// value of a ciphertext they already own.  Compiled only for noise-audit
/// builds so release binaries carry no decryption path.
#[cfg(feature = "noise-audit")]
pub(crate) fn raw_phase(ciphertext: &LweCiphertextOwned<u64>, secrets: &HarnessSecretKeys) -> u64 {
    decrypt_lwe_ciphertext(&secrets.big_lwe_sk, ciphertext).0
}

#[cfg(feature = "noise-audit")]
pub(crate) fn glwe_phase_at(
    c: &GlweCiphertextOwned<u64>,
    index: usize,
    secrets: &HarnessSecretKeys,
) -> u64 {
    let n = c.polynomial_size().0;
    assert!(index < n);
    let mut phase = c.get_body().as_ref()[index];
    for (mask, secret) in c
        .get_mask()
        .as_ref()
        .chunks_exact(n)
        .zip(secrets.big_lwe_sk.as_ref().chunks_exact(n))
    {
        for j in 0..n {
            if j <= index {
                phase = phase.wrapping_sub(mask[j].wrapping_mul(secret[index - j]));
            } else {
                phase = phase.wrapping_add(mask[j].wrapping_mul(secret[n + index - j]));
            }
        }
    }
    phase
}

#[cfg(feature = "noise-audit")]
pub(crate) fn ggsw_row_error_at(
    c: &GlweCiphertextOwned<u64>,
    index: usize,
    row: usize,
    bit: u64,
    factor: u64,
    secrets: &HarnessSecretKeys,
) -> i64 {
    let n = c.polynomial_size().0;
    let expected = if row < c.glwe_size().0 - 1 {
        secrets.big_lwe_sk.as_ref()[row * n + index]
            .wrapping_mul(bit)
            .wrapping_mul(factor)
            .wrapping_neg()
    } else if index == 0 {
        bit.wrapping_mul(factor)
    } else {
        0
    };
    glwe_phase_at(c, index, secrets).wrapping_sub(expected) as i64
}

#[cfg(feature = "noise-audit")]
pub(crate) fn negative_secret_product(
    input: &[u64],
    row: usize,
    secrets: &HarnessSecretKeys,
) -> Vec<u64> {
    let n = input.len();
    let mut out = vec![0u64; n];
    for (i, &s) in secrets.big_lwe_sk.as_ref()[row * n..(row + 1) * n]
        .iter()
        .enumerate()
    {
        if s == 0 {
            continue;
        }
        for (j, &x) in input.iter().enumerate() {
            if i + j < n {
                out[i + j] = out[i + j].wrapping_sub(x);
            } else {
                out[i + j - n] = out[i + j - n].wrapping_add(x);
            }
        }
    }
    out
}

#[cfg(all(test, feature = "noise-audit"))]
mod phase_tests {
    use super::*;
    #[test]
    fn secret_product_matches_mask_only_phase() {
        let secrets = HarnessSecretKeys {
            big_lwe_sk: LweSecretKey::from_container(vec![1, 0, 1, 1]),
        };
        let input = [2u64, (-3i64) as u64, 5, 7];
        let c = GlweCiphertext::from_container(
            [input.as_slice(), &[0u64; 4]].concat(),
            PolynomialSize(4),
            CiphertextModulus::new_native(),
        );
        assert_eq!(
            negative_secret_product(&input, 0, &secrets),
            (0..4)
                .map(|i| glwe_phase_at(&c, i, &secrets))
                .collect::<Vec<_>>()
        );
    }
    #[test]
    fn coefficient_phase_matches_sample_extraction() {
        for k in 1..=2 {
            let n = 32;
            let secrets = HarnessSecretKeys {
                big_lwe_sk: LweSecretKey::from_container(
                    (0..k * n)
                        .map(|i| ((i * i + 3 * i + 7) % 5 < 2) as u64)
                        .collect::<Vec<_>>(),
                ),
            };
            let c = GlweCiphertext::from_container(
                (0..(k + 1) * n)
                    .map(|i| (i as u64).wrapping_mul(0x9e3779b97f4a7c15))
                    .collect::<Vec<_>>(),
                PolynomialSize(n),
                CiphertextModulus::new_native(),
            );
            for i in 0..n {
                let mut lwe =
                    LweCiphertext::new(0, LweSize(k * n + 1), CiphertextModulus::new_native());
                extract_lwe_sample_from_glwe_ciphertext(&c, &mut lwe, MonomialDegree(i));
                assert_eq!(glwe_phase_at(&c, i, &secrets), raw_phase(&lwe, &secrets));
            }
        }
    }
}

pub(crate) fn decrypt_radix_digits(
    digits: &[LweCiphertextOwned<u64>],
    output_bits: usize,
    secrets: &HarnessSecretKeys,
) -> Vec<u64> {
    digits
        .iter()
        .map(|digit| {
            decode_scalar(
                decrypt_lwe_ciphertext(&secrets.big_lwe_sk, digit).0,
                output_bits,
            )
        })
        .collect()
}
