use crate::config::{centered_normalizer_modulus_switch, decode_scalar, BenchParams};
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

pub(crate) fn generate_keys(params: BenchParams, rng: &mut RngResources) -> KeySet {
    let glwe_sk = allocate_and_generate_new_binary_glwe_secret_key(
        params.glwe_dimension,
        params.polynomial_size,
        &mut rng.secret,
    );
    let small_lwe_sk =
        allocate_and_generate_new_binary_lwe_secret_key(params.lwe_dimension, &mut rng.secret);
    let big_lwe_sk = glwe_sk.clone().into_lwe_secret_key();

    let normalizer_fourier_bsk = generate_fourier_bootstrap_key(
        &small_lwe_sk,
        &glwe_sk,
        params.pbs_base_log,
        params.pbs_level,
        params,
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
    let direct_lift = generate_direct_lift_keys(&glwe_sk, params, rng);

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
    rng: &mut RngResources,
) -> DirectLiftKeys {
    let (auto_base_log, auto_level, ss_base_log, ss_level, cbs_base_log, cbs_level, log_lut_count) =
        params.direct_lift_decomposition();
    let auto_keys = revhomtrace_fourier_tfhe16::gen_all_auto_keys(
        auto_base_log,
        auto_level,
        revhomtrace_fourier_tfhe16::FftType::Vanilla,
        glwe_sk,
        params.glwe_noise_distribution,
        &mut rng.encryption,
    );
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
                Plaintext(digit << (u64::BITS as usize - 2)),
                params.glwe_noise_distribution,
                &mut rng.encryption,
            );
            lwe
        })
        .collect()
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
