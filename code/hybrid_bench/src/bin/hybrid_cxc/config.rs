use anyhow::{bail, Context, Result};
use std::env;
use tfhe::core_crypto::prelude::*;
use tfhe::shortint::parameters::{
    current_params::V1_6_PARAM_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M128, ClassicPBSParameters,
};

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
        match (self, parallel_cmux_cells()) {
            (Self::Hybrid4x4, true) => "CBS-CMUX-CHUNK4X4-TREE-PAR",
            (Self::Hybrid4x4, false) => "CBS-CMUX-CHUNK4X4-TREE",
            (Self::Hybrid8x8, true) => "CBS-CMUX-8X8-DIRECT-PAR",
            (Self::Hybrid8x8, false) => "CBS-CMUX-8X8-DIRECT",
        }
    }

    pub(crate) fn output_contract_label(self) -> &'static str {
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
    pub(crate) lwe_dimension: LweDimension,
    pub(crate) glwe_dimension: GlweDimension,
    pub(crate) polynomial_size: PolynomialSize,
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
    pub(crate) fn from_env() -> Result<Self> {
        let profile = env::var("CBS_PARAM_PROFILE").unwrap_or_else(|_| "m2c2-gaussian".to_string());
        if profile != "m2c2" && profile != "m2c2-gaussian" {
            bail!(
                "the released hybrid multiplication presets require CBS_PARAM_PROFILE=m2c2-gaussian; got {profile}"
            );
        }
        Ok(Self::from_classic_pbs(
            V1_6_PARAM_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M128,
        ))
    }

    fn from_classic_pbs(params: ClassicPBSParameters) -> Self {
        let pbs_base_log = env_usize("PBS_BASE_LOG", params.pbs_base_log.0);
        let pbs_level = env_usize("PBS_LEVEL", params.pbs_level.0);
        let ks_base_log = env_usize("KS_BASE_LOG", params.ks_base_log.0);
        let ks_level = env_usize("KS_LEVEL", params.ks_level.0);
        Self {
            profile_label: "m2c2-gaussian",
            lwe_dimension: params.lwe_dimension,
            glwe_dimension: params.glwe_dimension,
            polynomial_size: params.polynomial_size,
            lwe_noise_distribution: params.lwe_noise_distribution,
            glwe_noise_distribution: params.glwe_noise_distribution,
            pbs_base_log: DecompositionBaseLog(pbs_base_log),
            pbs_level: DecompositionLevelCount(pbs_level),
            cbs_pbs_base_log: DecompositionBaseLog(env_usize_any(
                &["CBS_LIFT_PBS_BASE_LOG", "CBS_PBS_BASE_LOG"],
                pbs_base_log,
            )),
            cbs_pbs_level: DecompositionLevelCount(env_usize_any(
                &["CBS_LIFT_PBS_LEVEL", "CBS_PBS_LEVEL"],
                pbs_level,
            )),
            ks_base_log: DecompositionBaseLog(ks_base_log),
            ks_level: DecompositionLevelCount(ks_level),
            cbs_ks_base_log: DecompositionBaseLog(env_usize_any(
                &["CBS_LIFT_KS_BASE_LOG", "CBS_KS_BASE_LOG"],
                ks_base_log,
            )),
            cbs_ks_level: DecompositionLevelCount(env_usize_any(
                &["CBS_LIFT_KS_LEVEL", "CBS_KS_LEVEL"],
                ks_level,
            )),
            ciphertext_modulus: params.ciphertext_modulus,
        }
    }

    pub(crate) fn direct_lift_decomposition(
        self,
    ) -> (
        DecompositionBaseLog,
        DecompositionLevelCount,
        DecompositionBaseLog,
        DecompositionLevelCount,
        DecompositionBaseLog,
        DecompositionLevelCount,
        LutCountLog,
    ) {
        (
            DecompositionBaseLog(env_usize("DIRECT_AUTO_BASE_LOG", 7)),
            DecompositionLevelCount(env_usize("DIRECT_AUTO_LEVEL", 7)),
            DecompositionBaseLog(env_usize("DIRECT_SS_BASE_LOG", 17)),
            DecompositionLevelCount(env_usize("DIRECT_SS_LEVEL", 2)),
            DecompositionBaseLog(env_usize("DIRECT_CBS_BASE_LOG", 4)),
            DecompositionLevelCount(env_usize("DIRECT_CBS_LEVEL", 4)),
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
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

pub(crate) fn env_usize_any(names: &[&str], default: usize) -> usize {
    names
        .iter()
        .find_map(|name| env::var(name).ok()?.parse::<usize>().ok())
        .unwrap_or(default)
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
