use anyhow::{bail, Context, Result};
use std::env;
use tfhe::core_crypto::prelude::*;
use tfhe::shortint::parameters::{
    current_params::V1_6_PARAM_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M128, ClassicPBSParameters,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AutoFft {
    Vanilla,
    Split40,
}

impl AutoFft {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "vanilla" => Ok(Self::Vanilla),
            "split40" => Ok(Self::Split40),
            _ => bail!("DIRECT_AUTO_FFT must be vanilla or split40; got {value}"),
        }
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Vanilla => "vanilla",
            Self::Split40 => "split40",
        }
    }

    pub(crate) fn fft_type(self) -> crate::revhomtrace_fourier_tfhe16::FftType {
        match self {
            Self::Vanilla => crate::revhomtrace_fourier_tfhe16::FftType::Vanilla,
            Self::Split40 => crate::revhomtrace_fourier_tfhe16::FftType::Split(40),
        }
    }
}

pub(crate) fn cached_product() -> bool {
    env_flag("CBS_CACHED_PRODUCT")
}

pub(crate) fn fused_products_per_group() -> usize {
    env_usize("FUSED_PRODUCTS_PER_GROUP", 0)
}

pub(crate) fn shared_preprocessing() -> bool {
    env_flag("CBS_SHARED_PREPROCESSING")
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProductVariant {
    Hybrid4x4,
    Hybrid8x8,
}

impl ProductVariant {
    pub(crate) fn from_env() -> Result<Self> {
        let chunk4x4 = env_flag("CBS_CHUNK4X4_TREE_PRODUCT");
        let chunk8x8 =
            env_flag("CBS_CHUNK8X8_DIRECT_PRODUCT") || env_flag("CBS_CHUNK4X4_SPLIT_PRODUCT");

        match (chunk4x4, chunk8x8) {
            (true, false) => Ok(Self::Hybrid4x4),
            (false, true) => Ok(Self::Hybrid8x8),
            (true, true) => bail!(
                "CBS_CHUNK4X4_TREE_PRODUCT and CBS_CHUNK8X8_DIRECT_PRODUCT are mutually exclusive"
            ),
            (false, false) => match env::var("CBS_FAILURE_PRESET").as_deref() {
                Ok("hybrid-4x4") => Ok(Self::Hybrid4x4),
                Ok("hybrid-8x8") => Ok(Self::Hybrid8x8),
                _ => bail!(
                    "select the released product contract with CBS_FAILURE_PRESET=hybrid-4x4 or hybrid-8x8"
                ),
            },
        }
    }

    pub(crate) fn preset_label(self) -> &'static str {
        match self {
            Self::Hybrid4x4 => "hybrid-4x4",
            Self::Hybrid8x8 => "hybrid-8x8",
        }
    }

    pub(crate) fn product_generator_label(self) -> &'static str {
        if fused_products_per_group() > 0 {
            return "CBS-CMUX-PRODUCT-SUM";
        }
        if cached_product() {
            return "CBS-CMUX-8X8-CACHED";
        }
        match (self, parallel_cmux_cells()) {
            (Self::Hybrid4x4, true) => "CBS-CMUX-CHUNK4X4-TREE-PAR",
            (Self::Hybrid4x4, false) => "CBS-CMUX-CHUNK4X4-TREE",
            (Self::Hybrid8x8, true) => "CBS-CMUX-8X8-DIRECT-PAR",
            (Self::Hybrid8x8, false) => "CBS-CMUX-8X8-DIRECT",
        }
    }

    pub(crate) fn output_contract_label(self) -> &'static str {
        if fused_products_per_group() > 0 {
            return "BOUNDED-PRODUCT-SUM-DIGITS";
        }
        if cached_product() {
            return "CHUNK8X8-CACHED-DIGITS";
        }
        match self {
            Self::Hybrid4x4 => "CHUNK4X4-TREE-DIGITS",
            Self::Hybrid8x8 => "CHUNK8X8-DIRECT-DIGITS",
        }
    }

    pub(crate) fn label_suffix(self) -> &'static str {
        match self {
            Self::Hybrid4x4 => "",
            Self::Hybrid8x8 => "_p8",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct BenchParams {
    pub(crate) profile_label: &'static str,
    pub(crate) auto_fft: AutoFft,
    pub(crate) even_odd_secret: bool,
    pub(crate) lwe_dimension: LweDimension,
    pub(crate) glwe_dimension: GlweDimension,
    pub(crate) polynomial_size: PolynomialSize,
    pub(crate) normalizer_glwe_dimension: GlweDimension,
    pub(crate) normalizer_polynomial_size: PolynomialSize,
    pub(crate) lwe_noise_distribution: DynamicDistribution<u64>,
    pub(crate) glwe_noise_distribution: DynamicDistribution<u64>,
    pub(crate) pbs_base_log: DecompositionBaseLog,
    pub(crate) pbs_level: DecompositionLevelCount,
    pub(crate) cbs_pbs_base_log: DecompositionBaseLog,
    pub(crate) cbs_pbs_level: DecompositionLevelCount,
    pub(crate) ks_base_log: DecompositionBaseLog,
    pub(crate) ks_level: DecompositionLevelCount,
    pub(crate) cbs_ks_base_log: DecompositionBaseLog,
    pub(crate) cbs_ks_level: DecompositionLevelCount,
    pub(crate) ciphertext_modulus: CiphertextModulus<u64>,
}

impl BenchParams {
    pub(crate) fn from_env(variant: ProductVariant) -> Result<Self> {
        let profile = env::var("CBS_PARAM_PROFILE").unwrap_or_else(|_| "m2c2-gaussian".to_string());
        if profile != "m2c2" && profile != "m2c2-gaussian" {
            bail!(
                "the released hybrid multiplication presets require CBS_PARAM_PROFILE=m2c2-gaussian; got {profile}"
            );
        }
        let mut params = Self::from_classic_pbs(
            V1_6_PARAM_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M128,
            variant,
        );
        params.auto_fft = AutoFft::parse(&env::var("DIRECT_AUTO_FFT").unwrap_or_else(|_| "vanilla".into()))?;
        match env::var("CBS_RING_SHAPE").as_deref() {
            Ok("1024x2") => {
                params.glwe_dimension = GlweDimension(2);
                params.polynomial_size = PolynomialSize(1024);
            }
            Ok("2048x1") | Err(_) => {}
            Ok(value) => bail!("CBS_RING_SHAPE must be 2048x1 or 1024x2; got {value}"),
        }
        params.even_odd_secret = match env::var("CBS_SECRET_LAYOUT").as_deref() {
            Ok("even-odd") => true,
            Ok("contiguous") | Err(_) => false,
            Ok(value) => bail!("CBS_SECRET_LAYOUT must be contiguous or even-odd; got {value}"),
        };
        if params.even_odd_secret && params.polynomial_size.0 != 1024 {
            bail!("even-odd secret layout requires CBS_RING_SHAPE=1024x2");
        }
        let group = fused_products_per_group();
        if group > 0 {
            let digits = ((9 * group).ilog2() as usize + 2) / 2;
            if digits * (9 * group + 1) > params.polynomial_size.0 {
                bail!("product-sum group {group} does not fit N={}", params.polynomial_size.0);
            }
        }
        if params.polynomial_size.0 != 2048 && group == 0 {
            bail!("1024x2 currently supports the product-sum algorithm only");
        }
        Ok(params)
    }

    pub(crate) fn normalizer_params(self) -> Self {
        Self {
            glwe_dimension: self.normalizer_glwe_dimension,
            polynomial_size: self.normalizer_polynomial_size,
            even_odd_secret: false,
            ..self
        }
    }

    fn from_classic_pbs(params: ClassicPBSParameters, variant: ProductVariant) -> Self {
        let (lift_pbs_base_log, lift_pbs_level) = match variant {
            ProductVariant::Hybrid4x4 => (12, 3),
            ProductVariant::Hybrid8x8 => (9, 4),
        };
        let pbs_base_log = env_usize("PBS_BASE_LOG", params.pbs_base_log.0);
        let pbs_level = env_usize("PBS_LEVEL", params.pbs_level.0);
        let ks_base_log = env_usize("KS_BASE_LOG", 2);
        let ks_level = env_usize("KS_LEVEL", 8);
        Self {
            profile_label: "m2c2-gaussian",
            auto_fft: AutoFft::Vanilla,
            even_odd_secret: false,
            lwe_dimension: params.lwe_dimension,
            glwe_dimension: params.glwe_dimension,
            polynomial_size: params.polynomial_size,
            normalizer_glwe_dimension: params.glwe_dimension,
            normalizer_polynomial_size: params.polynomial_size,
            lwe_noise_distribution: params.lwe_noise_distribution,
            glwe_noise_distribution: params.glwe_noise_distribution,
            pbs_base_log: DecompositionBaseLog(pbs_base_log),
            pbs_level: DecompositionLevelCount(pbs_level),
            cbs_pbs_base_log: DecompositionBaseLog(env_usize_any(
                &["CBS_LIFT_PBS_BASE_LOG", "CBS_PBS_BASE_LOG"],
                lift_pbs_base_log,
            )),
            cbs_pbs_level: DecompositionLevelCount(env_usize_any(
                &["CBS_LIFT_PBS_LEVEL", "CBS_PBS_LEVEL"],
                lift_pbs_level,
            )),
            ks_base_log: DecompositionBaseLog(ks_base_log),
            ks_level: DecompositionLevelCount(ks_level),
            cbs_ks_base_log: DecompositionBaseLog(env_usize_any(
                &["CBS_LIFT_KS_BASE_LOG", "CBS_KS_BASE_LOG"],
                7,
            )),
            cbs_ks_level: DecompositionLevelCount(env_usize_any(
                &["CBS_LIFT_KS_LEVEL", "CBS_KS_LEVEL"],
                2,
            )),
            ciphertext_modulus: params.ciphertext_modulus,
        }
    }

    pub(crate) fn direct_lift_decomposition(
        self,
        variant: ProductVariant,
    ) -> (
        DecompositionBaseLog,
        DecompositionLevelCount,
        DecompositionBaseLog,
        DecompositionLevelCount,
        DecompositionBaseLog,
        DecompositionLevelCount,
        LutCountLog,
    ) {
        let (cbs_base_log, cbs_level) = match variant {
            ProductVariant::Hybrid4x4 => (4, 5),
            ProductVariant::Hybrid8x8 => (3, 6),
        };
        (
            DecompositionBaseLog(env_usize("DIRECT_AUTO_BASE_LOG", 9)),
            DecompositionLevelCount(env_usize("DIRECT_AUTO_LEVEL", 5)),
            DecompositionBaseLog(env_usize("DIRECT_SS_BASE_LOG", 13)),
            DecompositionLevelCount(env_usize("DIRECT_SS_LEVEL", 3)),
            DecompositionBaseLog(env_usize("DIRECT_CBS_BASE_LOG", cbs_base_log)),
            DecompositionLevelCount(env_usize("DIRECT_CBS_LEVEL", cbs_level)),
            LutCountLog(env_usize("DIRECT_LOG_LUT_COUNT", 2)),
        )
    }
}

pub(crate) fn validate_runtime_contract() -> Result<()> {
    if let Ok(mode) = env::var("CBS_PRODUCT_MODE") {
        if mode != "cmux-lut" && mode != "lut" {
            bail!("the paper binary supports only CBS_PRODUCT_MODE=cmux-lut; got {mode}");
        }
    }
    if let Ok(mode) = env::var("CBS_DIGIT_LIFT") {
        if mode != "direct-revhomtrace" {
            bail!("the paper binary supports only CBS_DIGIT_LIFT=direct-revhomtrace; got {mode}");
        }
    }
    for (name, expected) in [
        ("CMUX_ROW_BITS", 4usize),
        ("CBS_OUTPUT_ROW_BITS", 4usize),
        ("CBS_FINAL_OUTPUT_ROW_BITS", 4usize),
        ("CBS_CHUNK8X8_PREFIX_BITS", 8usize),
        ("CBS_CHUNK4X4_SPLIT_PREFIX_BITS", 8usize),
    ] {
        if let Ok(value) = env::var(name) {
            let parsed = value
                .parse::<usize>()
                .with_context(|| format!("failed to parse {name}={value}"))?;
            if parsed != expected {
                bail!("the released paper path requires {name}={expected}; got {parsed}");
            }
        }
    }
    Ok(())
}

pub(crate) fn env_flag(name: &str) -> bool {
    matches!(
        env::var(name).as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
}

pub(crate) fn env_flag_or(name: &str, default: bool) -> bool {
    match env::var(name) {
        Ok(value) => matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"),
        Err(_) => default,
    }
}

pub(crate) fn env_usize(name: &str, default: usize) -> usize {
    match env::var(name) {
        Ok(value) => value.parse::<usize>()
            .unwrap_or_else(|_| panic!("{name} must be an unsigned integer; got {value}")),
        Err(env::VarError::NotPresent) => default,
        Err(error) => panic!("invalid {name}: {error}"),
    }
}

pub(crate) fn env_usize_any(names: &[&str], default: usize) -> usize {
    names.iter().find(|name| env::var_os(name).is_some())
        .map(|name| env_usize(name, default)).unwrap_or(default)
}

pub(crate) fn parallel_digit_lift() -> bool {
    env_flag_or("CBS_PARALLEL_DIGIT_LIFT", true)
}

pub(crate) fn parallel_cmux_cells() -> bool {
    env_flag_or("CBS_PARALLEL_CMUX_CELLS", true)
}

pub(crate) fn centered_normalizer_modulus_switch() -> bool {
    env_flag_or("CBS_NORMALIZER_CENTERED_MS", true)
}

pub(crate) fn encoding_delta(bits: usize) -> u64 {
    1u64 << (63 - bits)
}

pub(crate) fn decode_scalar(value: u64, bits: usize) -> u64 {
    let delta = encoding_delta(bits);
    ((value.wrapping_add(delta / 2)) / delta) % (1u64 << bits)
}

#[cfg(test)]
mod fft_tests {
    use super::AutoFft;

    #[test]
    fn automorphism_backend_is_explicit_and_checked() {
        let vanilla = AutoFft::parse("vanilla").unwrap();
        let split = AutoFft::parse("split40").unwrap();
        assert_eq!(vanilla.fft_type().num_split(), 1);
        assert_eq!(split.fft_type().num_split(), 2);
        assert_eq!(split.fft_type().split_base_log(), 40);
        assert_eq!(split.label(), "split40");
        for value in ["", "split", "split0", "split64", "typo"] {
            assert!(AutoFft::parse(value).is_err());
        }
    }
}

/// `CBS_PHASE_THREADS=1`: run a phase on at most as many threads as it has
/// independent jobs. Narrow products leave threads idle in the selector
/// lift and the reduction rounds, where extra workers only add scheduling
/// overhead and share physical cores; one pool per size is kept for reuse.
pub(crate) fn with_phase_threads<R: Send>(useful: usize, f: impl FnOnce() -> R + Send) -> R {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};
    let total = rayon::current_num_threads();
    let threads = useful.max(1).min(total);
    if !env_flag("CBS_PHASE_THREADS") || threads >= total {
        return f();
    }
    static POOLS: OnceLock<Mutex<HashMap<usize, Arc<rayon::ThreadPool>>>> = OnceLock::new();
    let pool = POOLS
        .get_or_init(Default::default)
        .lock()
        .expect("phase pool registry")
        .entry(threads)
        .or_insert_with(|| {
            Arc::new(rayon::ThreadPoolBuilder::new().num_threads(threads).build().expect("phase pool"))
        })
        .clone();
    pool.install(f)
}
