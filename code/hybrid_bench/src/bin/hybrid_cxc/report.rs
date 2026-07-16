use crate::config::{env_flag, env_usize, BenchParams, ProductVariant};
use crate::types::Timings;
use anyhow::{Context, Result};
use csv::WriterBuilder;
use std::env;
use std::fs::{create_dir_all, OpenOptions};
use std::path::Path;

const TIMING_HEADER: &[&str] = &[
    "case_type",
    "label",
    "lift_mode",
    "digit_lift_schedule",
    "normalizer_mode",
    "token_aware_normalizer",
    "rayon_threads",
    "width_bits",
    "digit_count",
    "product_count",
    "total_ms",
    "harness_encrypt_ms",
    "selector_extract_ms",
    "cbs_ms",
    "ext_ms",
    "diagonal_extract_ms",
    "normalization_ms",
    "harness_decrypt_ms",
    "cbs_lifts",
    "cmux_count",
    "normalization_pbs",
    "normalization_rounds",
    "normalization_max_chunks",
    "normalization_max_column_height",
    "normalization_max_column_bound",
    "ok",
    "pgen",
    "nkernel",
    "nsched",
    "output_contract",
    "reuse_model",
    "row_bits",
    "output_row_bits",
    "failure_preset",
    "param_profile",
    "pbs_base_log",
    "pbs_level",
    "ks_base_log",
    "ks_level",
    "direct_auto_base_log",
    "direct_auto_level",
    "direct_ss_base_log",
    "direct_ss_level",
    "direct_cbs_base_log",
    "direct_cbs_level",
    "normalizer_chunk_cap",
    "normalizer_centered_ms",
    "failure_target_bits",
    "failure_expected_worst_log2",
    "failure_expected_acceptance_log2",
    "failure_expected_union_log2",
    "phase_timer_kind",
    "product_generation_ms",
    "post_product_ms",
    "harness_overhead_ms",
    "measured_phase_ms",
    "product_generation_pct",
    "post_product_pct",
    "cbs_lift_pbs_base_log",
    "cbs_lift_pbs_level",
    "cbs_lift_ks_base_log",
    "cbs_lift_ks_level",
    "trial_index",
    "chunk8x8_prefix_bits",
    "height2_normalizer",
    "refresh_digit",
    "lift_centered_ms",
    "direct_log_lut_count",
];

pub(crate) fn append_timing_csv(
    label: &str,
    width_bits: usize,
    digit_count: usize,
    total_ms: u128,
    timings: &Timings,
    params: BenchParams,
    variant: ProductVariant,
) -> Result<()> {
    let Ok(path) = env::var("CBS_TIMING_CSV") else {
        return Ok(());
    };
    if let Some(parent) = Path::new(&path).parent() {
        if !parent.as_os_str().is_empty() {
            create_dir_all(parent)
                .with_context(|| format!("failed to create timing CSV directory for {path}"))?;
        }
    }
    let write_header = std::fs::metadata(&path)
        .map(|metadata| metadata.len() == 0)
        .unwrap_or(true);
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("failed to open CBS_TIMING_CSV={path}"))?;
    let mut writer = WriterBuilder::new().has_headers(false).from_writer(file);
    if write_header {
        writer.write_record(TIMING_HEADER)?;
    }

    let product_generation_ms = timings.product_generation_ms();
    let post_product_ms = timings.post_product_ms();
    let harness_overhead_ms = timings.harness_overhead_ms();
    let measured_phase_ms = product_generation_ms + post_product_ms;
    let product_generation_pct = percentage(product_generation_ms, measured_phase_ms);
    let post_product_pct = percentage(post_product_ms, measured_phase_ms);
    let pbs_base_log = env_or("PBS_BASE_LOG", "profile");
    let pbs_level = env_or("PBS_LEVEL", "profile");
    let ks_base_log = env_or("KS_BASE_LOG", "profile");
    let ks_level = env_or("KS_LEVEL", "profile");
    let worst_failure = env_or("CBS_FAILURE_EXPECTED_WORST_LOG2", "");

    let row = vec![
        "cmux_lut_product".to_string(),
        label.to_string(),
        "direct-revhomtrace".to_string(),
        digit_lift_schedule_label().to_string(),
        "height2-parallel-add-token".to_string(),
        "1".to_string(),
        rayon::current_num_threads().to_string(),
        width_bits.to_string(),
        digit_count.to_string(),
        "1".to_string(),
        total_ms.to_string(),
        timings.harness_encrypt_ms.to_string(),
        timings.selector_extract_ms.to_string(),
        timings.cbs_ms.to_string(),
        timings.ext_ms.to_string(),
        timings.diagonal_extract_ms.to_string(),
        timings.normalization_ms.to_string(),
        timings.harness_decrypt_ms.to_string(),
        timings.cbs_lifts.to_string(),
        timings.cmux_count.to_string(),
        timings.normalization_pbs.to_string(),
        timings.normalization_rounds.to_string(),
        timings.normalization_max_chunks.to_string(),
        timings.normalization_max_column_height.to_string(),
        timings.normalization_max_column_bound.to_string(),
        "true".to_string(),
        variant.product_generator_label().to_string(),
        "K-LOWCARRY1".to_string(),
        "S-HEIGHT2-TOK".to_string(),
        variant.output_contract_label().to_string(),
        reuse_model_label().to_string(),
        "4".to_string(),
        "4".to_string(),
        env_or("CBS_FAILURE_PRESET", variant.preset_label()),
        env_or("CBS_PARAM_PROFILE", params.profile_label),
        pbs_base_log.clone(),
        pbs_level.clone(),
        ks_base_log.clone(),
        ks_level.clone(),
        env_or("DIRECT_AUTO_BASE_LOG", "profile"),
        env_or("DIRECT_AUTO_LEVEL", "profile"),
        env_or("DIRECT_SS_BASE_LOG", "profile"),
        env_or("DIRECT_SS_LEVEL", "profile"),
        env_or("DIRECT_CBS_BASE_LOG", "profile"),
        env_or("DIRECT_CBS_LEVEL", "profile"),
        env_usize("CBS_NORMALIZER_CHUNK_CAP", 15).to_string(),
        bool_digit(crate::config::centered_normalizer_modulus_switch()),
        env_or("CBS_FAILURE_TARGET_BITS", ""),
        worst_failure.clone(),
        env_or("CBS_FAILURE_EXPECTED_ACCEPTANCE_LOG2", &worst_failure),
        env_or("CBS_FAILURE_EXPECTED_UNION_LOG2", &worst_failure),
        "evaluator-wall-clock".to_string(),
        product_generation_ms.to_string(),
        post_product_ms.to_string(),
        harness_overhead_ms.to_string(),
        measured_phase_ms.to_string(),
        format!("{product_generation_pct:.2}"),
        format!("{post_product_pct:.2}"),
        env_or("CBS_LIFT_PBS_BASE_LOG", &pbs_base_log),
        env_or("CBS_LIFT_PBS_LEVEL", &pbs_level),
        env_or("CBS_LIFT_KS_BASE_LOG", &ks_base_log),
        env_or("CBS_LIFT_KS_LEVEL", &ks_level),
        trial_index(label),
        "8".to_string(),
        "1".to_string(),
        "1".to_string(),
        env_or("CBS_CENTERED_MS", "0"),
        env_or("DIRECT_LOG_LUT_COUNT", "2"),
    ];
    debug_assert_eq!(row.len(), TIMING_HEADER.len());
    writer.write_record(row)?;
    writer.flush()?;
    Ok(())
}

fn env_or(name: &str, default: &str) -> String {
    env::var(name)
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| default.to_string())
}

fn bool_digit(value: bool) -> String {
    if value { "1" } else { "0" }.to_string()
}

fn digit_lift_schedule_label() -> &'static str {
    if env_flag("CBS_PARALLEL_PRODUCT_LIFT") {
        "parallel-product-digits"
    } else if crate::config::parallel_digit_lift() {
        "parallel-digits"
    } else {
        "serial-digits"
    }
}

fn reuse_model_label() -> &'static str {
    if env_flag("CBS_REUSE_LEFT_LIFTS") || env_flag("CBS_SHARED_LEFT") {
        "shared-left"
    } else {
        "none"
    }
}

fn percentage(part: u128, total: u128) -> f64 {
    if total == 0 {
        0.0
    } else {
        100.0 * part as f64 / total as f64
    }
}

fn trial_index(label: &str) -> String {
    label
        .split("_trial")
        .nth(1)
        .map(|suffix| {
            suffix
                .chars()
                .take_while(char::is_ascii_digit)
                .collect::<String>()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::TIMING_HEADER;

    const RELEASED_HEADER: &str = "case_type,label,lift_mode,digit_lift_schedule,normalizer_mode,token_aware_normalizer,rayon_threads,width_bits,digit_count,product_count,total_ms,harness_encrypt_ms,selector_extract_ms,cbs_ms,ext_ms,diagonal_extract_ms,normalization_ms,harness_decrypt_ms,cbs_lifts,cmux_count,normalization_pbs,normalization_rounds,normalization_max_chunks,normalization_max_column_height,normalization_max_column_bound,ok,pgen,nkernel,nsched,output_contract,reuse_model,row_bits,output_row_bits,failure_preset,param_profile,pbs_base_log,pbs_level,ks_base_log,ks_level,direct_auto_base_log,direct_auto_level,direct_ss_base_log,direct_ss_level,direct_cbs_base_log,direct_cbs_level,normalizer_chunk_cap,normalizer_centered_ms,failure_target_bits,failure_expected_worst_log2,failure_expected_acceptance_log2,failure_expected_union_log2,phase_timer_kind,product_generation_ms,post_product_ms,harness_overhead_ms,measured_phase_ms,product_generation_pct,post_product_pct,cbs_lift_pbs_base_log,cbs_lift_pbs_level,cbs_lift_ks_base_log,cbs_lift_ks_level,trial_index,chunk8x8_prefix_bits,height2_normalizer,refresh_digit,lift_centered_ms,direct_log_lut_count";

    #[test]
    fn timing_header_matches_the_released_schema() {
        assert_eq!(TIMING_HEADER.len(), 68);
        assert_eq!(TIMING_HEADER.join(","), RELEASED_HEADER);
    }
}
