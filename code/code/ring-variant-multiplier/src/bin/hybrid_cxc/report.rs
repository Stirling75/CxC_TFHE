use crate::config::{env_flag, env_usize, BenchParams, ProductVariant};
use crate::types::Timings;
use crate::keys::EvaluationKeys;
use anyhow::{Context, Result};
use csv::WriterBuilder;
use std::env;
use std::fs::{create_dir_all, OpenOptions};
use std::path::Path;

pub(crate) fn write_parameters(
    params: BenchParams,
    keys: &EvaluationKeys,
    operands: crate::harness::OperandSource,
) -> Result<()> {
    let Ok(path) = env::var("CBS_TIMING_CSV") else { return Ok(()); };
    let directory = Path::new(&path).parent().unwrap_or(Path::new("."));
    create_dir_all(directory)?;
    let bsk = keys.normalizer_fourier_bsk();
    let ksk = keys.normalizer_big_to_small_ksk();
    let lift = &keys.cbs_fourier_bsk;
    let lift_ks = &keys.cbs_big_to_small_ksk;
    let direct = &keys.direct_lift;
    let record = serde_json::json!({
        "tfhe_rs": "1.6.1", "input": "PBS-refreshed radix-4", "input_scale": 8,
        "message_modulus": 4, "carry_modulus": 4,
        "ciphertext_modulus": params.ciphertext_modulus,
        "lwe_noise_distribution": params.lwe_noise_distribution,
        "glwe_noise_distribution": params.glwe_noise_distribution,
        "lwe_dimension": bsk.input_lwe_dimension().0,
        "normalizer_ring": [bsk.polynomial_size().0, bsk.glwe_size().0 - 1],
        "cbs_ring": [lift.polynomial_size().0, lift.glwe_size().0 - 1],
        "pbs": [bsk.decomposition_base_log().0, bsk.decomposition_level_count().0],
        "ks": [ksk.decomposition_base_log().0, ksk.decomposition_level_count().0],
        "lift_pbs": [lift.decomposition_base_log().0, lift.decomposition_level_count().0],
        "lift_ks": [lift_ks.decomposition_base_log().0, lift_ks.decomposition_level_count().0],
        "auto": [direct.auto_base_log.0, direct.auto_level.0],
        "ss": [direct.ss_base_log.0, direct.ss_level.0],
        "cbs": [direct.cbs_base_log.0, direct.cbs_level.0],
        "auto_fft": params.auto_fft.label(), "log_lut_count": direct.log_lut_count.0,
        "whole_multiplier_log2_failure": null,
        // Plaintext operand provenance: CACHED_MULT_PATTERN and the parsed
        // CBS_RANDOM_OPERANDS_SEED (null when no seeded generator is used).
        "operand_pattern": operands.pattern_label(),
        "random_operands_seed": operands.seed()
    });
    std::fs::write(directory.join("parameters.json"), serde_json::to_vec_pretty(&record)?)?;
    Ok(())
}

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
    "center_selector_box",
    "cached_product_table",
    "shared_preprocessing",
    "squaring",
    "reduction_key_switches",
    "warmup",
    "fused_products_per_group",
    "fused_kernel",
    "direct_external_products",
    "external_product_equivalents",
    "prefix_cache_entries",
    "prefix_cache_hits",
    "direct_auto_fft",
    "cbs_polynomial_size",
    "cbs_glwe_dimension",
    "normalizer_polynomial_size",
    "normalizer_glwe_dimension",
    "input_lwe_dimension",
    "cbs_secret_layout",
    "normalization_reduction_pbs",
    "normalization_final_pbs",
    "audit_callbacks",
];

pub(crate) fn append_timing_csv(
    label: &str,
    width_bits: usize,
    digit_count: usize,
    total_ms: f64,
    timings: &Timings,
    params: BenchParams,
    variant: ProductVariant,
    correct: bool,
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
    let pbs_base_log = params.pbs_base_log.0.to_string();
    let pbs_level = params.pbs_level.0.to_string();
    let ks_base_log = params.ks_base_log.0.to_string();
    let ks_level = params.ks_level.0.to_string();
    let (auto_base, auto_level, ss_base, ss_level, cbs_base, cbs_level, _) =
        params.direct_lift_decomposition(variant);
    let worst_failure = env_or("CBS_FAILURE_EXPECTED_WORST_LOG2", "");

    let row = vec![
        "cmux_lut_product".to_string(),
        label.to_string(),
        "direct-revhomtrace".to_string(),
        digit_lift_schedule_label().to_string(),
        if crate::config::cached_product() {
            "height2-cache-aware"
        } else {
            "height2-bound-only"
        }
        .to_string(),
        "1".to_string(),
        rayon::current_num_threads().to_string(),
        width_bits.to_string(),
        digit_count.to_string(),
        "1".to_string(),
        ms(total_ms),
        ms(timings.harness_encrypt_ms),
        ms(timings.selector_extract_ms),
        ms(timings.cbs_ms),
        ms(timings.ext_ms),
        ms(timings.diagonal_extract_ms),
        ms(timings.normalization_ms),
        ms(timings.harness_decrypt_ms),
        timings.cbs_lifts.to_string(),
        timings.cmux_count.to_string(),
        timings.normalization_pbs.to_string(),
        timings.normalization_rounds.to_string(),
        timings.normalization_max_chunks.to_string(),
        timings.normalization_max_column_height.to_string(),
        timings.normalization_max_column_bound.to_string(),
        correct.to_string(),
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
        auto_base.0.to_string(),
        auto_level.0.to_string(),
        ss_base.0.to_string(),
        ss_level.0.to_string(),
        cbs_base.0.to_string(),
        cbs_level.0.to_string(),
        env_usize("CBS_NORMALIZER_CHUNK_CAP", 15).to_string(),
        bool_digit(crate::config::centered_normalizer_modulus_switch()),
        env_or("CBS_FAILURE_TARGET_BITS", ""),
        worst_failure.clone(),
        env_or("CBS_FAILURE_EXPECTED_ACCEPTANCE_LOG2", &worst_failure),
        env_or("CBS_FAILURE_EXPECTED_UNION_LOG2", &worst_failure),
        "evaluator-wall-clock".to_string(),
        ms(product_generation_ms),
        ms(post_product_ms),
        ms(harness_overhead_ms),
        ms(measured_phase_ms),
        format!("{product_generation_pct:.2}"),
        format!("{post_product_pct:.2}"),
        params.cbs_pbs_base_log.0.to_string(),
        params.cbs_pbs_level.0.to_string(),
        params.cbs_ks_base_log.0.to_string(),
        params.cbs_ks_level.0.to_string(),
        trial_index(label),
        "8".to_string(),
        "1".to_string(),
        "1".to_string(),
        env_or("CBS_CENTERED_MS", "0"),
        env_or("DIRECT_LOG_LUT_COUNT", "2"),
        env_or("CBS_CENTER_SELECTOR_BOX", "0"),
        bool_digit(crate::config::cached_product()),
        bool_digit(crate::config::shared_preprocessing()),
        bool_digit(env_flag("CACHED_MULT_SQUARING")),
        timings.normalization_key_switches.to_string(),
        bool_digit(
            trial_index(label).parse::<usize>().unwrap_or(0) < env_usize("CACHED_MULT_WARMUP", 0),
        ),
        crate::config::fused_products_per_group().to_string(),
        env_or("FUSED_KERNEL", "tree"),
        timings.direct_external_products.to_string(),
        (timings.cmux_count + timings.direct_external_products).to_string(),
        timings.prefix_cache_entries.to_string(),
        timings.prefix_cache_hits.to_string(),
        params.auto_fft.label().to_string(),
        params.polynomial_size.0.to_string(),
        params.glwe_dimension.0.to_string(),
        params.normalizer_polynomial_size.0.to_string(),
        params.normalizer_glwe_dimension.0.to_string(),
        params.lwe_dimension.0.to_string(),
        if params.even_odd_secret { "even-odd" } else { "contiguous" }.to_string(),
        timings.normalization_reduction_pbs.to_string(),
        timings.normalization_final_pbs.to_string(),
        bool_digit(timings.audit_callbacks),
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

fn percentage(part: f64, total: f64) -> f64 {
    if total <= 0.0 {
        0.0
    } else {
        100.0 * part / total
    }
}

/// Fractional milliseconds with nanosecond resolution.
fn ms(value: f64) -> String {
    format!("{value:.6}")
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

    // The revised artifact appends one provenance column, `center_selector_box`, to the
    // 68-column schema of the original submission so every CSV row records whether the
    // selector-box centering fix was active.
    const RELEASED_HEADER: &str = "case_type,label,lift_mode,digit_lift_schedule,normalizer_mode,token_aware_normalizer,rayon_threads,width_bits,digit_count,product_count,total_ms,harness_encrypt_ms,selector_extract_ms,cbs_ms,ext_ms,diagonal_extract_ms,normalization_ms,harness_decrypt_ms,cbs_lifts,cmux_count,normalization_pbs,normalization_rounds,normalization_max_chunks,normalization_max_column_height,normalization_max_column_bound,ok,pgen,nkernel,nsched,output_contract,reuse_model,row_bits,output_row_bits,failure_preset,param_profile,pbs_base_log,pbs_level,ks_base_log,ks_level,direct_auto_base_log,direct_auto_level,direct_ss_base_log,direct_ss_level,direct_cbs_base_log,direct_cbs_level,normalizer_chunk_cap,normalizer_centered_ms,failure_target_bits,failure_expected_worst_log2,failure_expected_acceptance_log2,failure_expected_union_log2,phase_timer_kind,product_generation_ms,post_product_ms,harness_overhead_ms,measured_phase_ms,product_generation_pct,post_product_pct,cbs_lift_pbs_base_log,cbs_lift_pbs_level,cbs_lift_ks_base_log,cbs_lift_ks_level,trial_index,chunk8x8_prefix_bits,height2_normalizer,refresh_digit,lift_centered_ms,direct_log_lut_count,center_selector_box";

    #[test]
    fn timing_header_matches_the_released_schema() {
        assert_eq!(TIMING_HEADER.len(), 90);
        assert_eq!(TIMING_HEADER[86], "cbs_secret_layout");
        assert_eq!(TIMING_HEADER[87..], ["normalization_reduction_pbs", "normalization_final_pbs", "audit_callbacks"]);
        assert_eq!(TIMING_HEADER[80], "direct_auto_fft");
        assert_eq!(TIMING_HEADER[..69].join(","), RELEASED_HEADER);
    }

    #[test]
    fn failed_result_is_written_as_false() {
        let path = std::env::temp_dir().join(format!("hybrid-failed-row-{}.csv", std::process::id()));
        assert!(!path.exists());
        std::env::set_var("CBS_TIMING_CSV", &path);
        let variant = crate::config::ProductVariant::Hybrid8x8;
        let params = crate::config::BenchParams::from_env(variant).unwrap();
        super::append_timing_csv("test_trial0", 16, 8, 1.0, &crate::types::Timings::default(),
                                 params, variant, false).unwrap();
        std::env::remove_var("CBS_TIMING_CSV");
        let mut reader = csv::Reader::from_path(&path).unwrap();
        let header = reader.headers().unwrap().clone();
        let ok = header.iter().position(|name| name == "ok").unwrap();
        let rows = reader.records().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(&rows[0][ok], "false");
        std::fs::remove_file(path).unwrap();
    }
}
