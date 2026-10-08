use anyhow::{ensure, Result};
use clap::Parser;
use rayon::prelude::*;
use serde_json::json;
use std::path::PathBuf;
use std::time::Instant;
use tfhe::integer::bigint::U256;
use tfhe::integer::{gen_keys_radix, IntegerCiphertext, RadixCiphertext};
use tfhe::shortint::parameters::current_params::V1_6_PARAM_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M128;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    width: usize,
    #[arg(long, default_value_t = 1)]
    threads: usize,
    #[arg(long, default_value_t = 3)]
    repetitions: usize,
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    #[arg(long, default_value_t = 20260905)]
    seed: u64,
    /// Multiply by the second operand as a public scalar (scalar_mul_parallelized).
    #[arg(long)]
    scalar: bool,
    /// Fixed public scalar for every trial, as radix-4 digits, least significant
    /// first (pairs the comparison with the scalars of the CxP product-sum runs).
    #[arg(long)]
    scalar_digits: Option<String>,
    /// Key-switching decomposition (base log, level) replacing that of the
    /// default set, e.g. 2,8 to meet a whole-multiplication target.
    #[arg(long)]
    ks: Option<String>,
    #[arg(long)]
    output: PathBuf,
}

fn splitmix64(mut state: u64) -> u64 {
    state = state.wrapping_add(0x9E3779B97F4A7C15);
    let z = (state ^ (state >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn operand(width: usize, trial: usize, side: usize, seed: u64) -> U256 {
    let mut words = [0u64; 4];
    for digit in 0..width / 2 {
        let value = splitmix64(seed
            ^ (trial as u64).wrapping_mul(0x9E3779B97F4A7C15)
            ^ (side as u64).wrapping_mul(0xD1B54A32D192ED03)
            ^ (digit as u64).wrapping_mul(0x2545F4914F6CDD1D)) % 4;
        words[digit / 32] |= value << (2 * (digit % 32));
    }
    U256::from(words)
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!([16, 32, 64, 128, 256].contains(&args.width), "unsupported width");
    ensure!(args.threads > 0 && args.repetitions > 0, "positive counts required");
    ensure!(!args.output.exists(), "use a fresh output directory");
    let fixed_scalar = match &args.scalar_digits {
        Some(digits) => {
            ensure!(args.scalar && digits.len() == args.width / 2, "scalar digits need --scalar and W/2 digits");
            let mut words = [0u64; 4];
            for (digit, c) in digits.chars().enumerate() {
                let value = c.to_digit(4).ok_or_else(|| anyhow::anyhow!("radix-4 digits only"))? as u64;
                words[digit / 32] |= value << (2 * (digit % 32));
            }
            Some(U256::from(words))
        }
        None => None,
    };
    std::fs::create_dir_all(&args.output)?;
    rayon::ThreadPoolBuilder::new().num_threads(args.threads).build_global()?;
    let mut params = V1_6_PARAM_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M128;
    if let Some(ks) = &args.ks {
        let ks = ks.split(',').map(str::parse::<usize>).collect::<Result<Vec<_>, _>>()?;
        ensure!(ks.len() == 2, "--ks takes base_log,level");
        params.ks_base_log = tfhe::shortint::parameters::DecompositionBaseLog(ks[0]);
        params.ks_level = tfhe::shortint::parameters::DecompositionLevelCount(ks[1]);
    }
    let metadata = json!({
        "method": if args.scalar { "TFHE-rs scalar_mul_parallelized" } else { "TFHE-rs mul_parallelized" },
        "tfhe_rs": "1.6.1", "scalar": args.scalar,
        "parameter": if args.ks.is_some() { "V1_6_PARAM_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M128 with replaced KS decomposition" } else { "V1_6_PARAM_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M128" },
        "parameters": params, "primitive_log2_p_fail": params.log2_p_fail,
        "lwe_dimension": params.lwe_dimension.0,
        "glwe_dimension": params.glwe_dimension.0,
        "polynomial_size": params.polynomial_size.0,
        "pbs_base_log": params.pbs_base_log.0, "pbs_level": params.pbs_level.0,
        "ks_base_log": params.ks_base_log.0, "ks_level": params.ks_level.0,
        "lwe_noise": format!("{:?}", params.lwe_noise_distribution),
        "glwe_noise": format!("{:?}", params.glwe_noise_distribution),
        "width": args.width, "threads": args.threads, "seed": args.seed,
        "repetitions": args.repetitions, "warmup": args.warmup,
        "scalar_digits": args.scalar_digits,
        "input": "identity-PBS-refreshed clean radix-4 blocks; preparation excluded",
        "output": "clean radix-4 XY modulo 2^W", "whole_multiplier_log2_failure": null
    });
    std::fs::write(args.output.join("parameters.json"), serde_json::to_vec_pretty(&metadata)?)?;
    let (client, server) = gen_keys_radix(params, args.width / 2);
    let shortint: &tfhe::shortint::ServerKey = server.as_ref();
    let identity = shortint.generate_lookup_table(|x| x % 4);
    let mask = U256::MAX >> (256 - args.width) as u32;
    let mut csv = csv::Writer::from_path(args.output.join("timings.csv"))?;
    csv.write_record(["method", "width", "threads", "trial", "warmup", "total_seconds", "pbs_count", "ok"])?;
    for trial in 0..args.warmup + args.repetitions {
        let x = operand(args.width, trial, 0, args.seed);
        let y = fixed_scalar.unwrap_or_else(|| operand(args.width, trial, 1, args.seed));
        let prepare = |value| {
            let encrypted = client.encrypt(value);
            RadixCiphertext::from(encrypted.blocks().par_iter()
                .map(|block| shortint.apply_lookup_table(block, &identity)).collect::<Vec<_>>())
        };
        let lhs = prepare(x);
        let rhs = if args.scalar { None } else { Some(prepare(y)) };
        tfhe::shortint::server_key::reset_pbs_count();
        let started = Instant::now();
        let product = match &rhs {
            Some(rhs) => server.mul_parallelized(&lhs, rhs),
            None => server.scalar_mul_parallelized(&lhs, y),
        };
        let seconds = started.elapsed().as_secs_f64();
        let clear: U256 = client.decrypt(&product);
        let ok = product.block_carries_are_empty() && clear == (x * y) & mask;
        csv.write_record([(if args.scalar { "tfhe-rs-1.6.1-gaussian-scalar" } else { "tfhe-rs-1.6.1-gaussian" }).to_string(), args.width.to_string(),
            args.threads.to_string(), trial.to_string(), (trial < args.warmup).to_string(),
            seconds.to_string(), tfhe::shortint::server_key::get_pbs_count().to_string(), ok.to_string()])?;
        csv.flush()?;
        ensure!(ok, "incorrect encrypted product; failed trial retained");
        println!("trial={trial} warmup={} seconds={seconds:.6} ok={ok}", trial < args.warmup);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn operands_match_the_hybrid_digit_generator() {
        let seed = 20260905;
        let value = operand(256, 2, 1, seed);
        for i in 0..128 {
            let expected = splitmix64(seed ^ 2u64.wrapping_mul(0x9E3779B97F4A7C15)
                ^ 0xD1B54A32D192ED03 ^ (i as u64).wrapping_mul(0x2545F4914F6CDD1D)) % 4;
            assert_eq!((value >> (2 * i) as u32) & U256::from(3u64), U256::from(expected));
        }
    }
}
