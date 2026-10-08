use super::{crypto, evaluate, plan::Plan, parameters::Parameters};
use anyhow::{ensure, Result};
use clap::{Parser, ValueEnum};
use serde_json::json;
use std::path::PathBuf;
use std::time::Instant;
use tfhe::core_crypto::prelude::*;
use tfhe::core_crypto::commons::numeric::CastFrom;
use tfhe::integer::bigint::U256;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Pattern { Random, Zero, Max, Alternating }

#[derive(Parser)]
struct Args {
    #[arg(long)] width: usize,
    #[arg(long, default_value_t = 1)] threads: usize,
    #[arg(long, default_value_t = 3)] repetitions: usize,
    #[arg(long, default_value_t = 1)] warmup: usize,
    #[arg(long, default_value_t = 20260905)] seed: u64,
    #[arg(long, value_enum, default_value_t = Pattern::Random)] pattern: Pattern,
    #[arg(long, default_value_t = 52)] boundary_delta_log2: u32,
    #[arg(long)] allow_alternate_boundary: bool,
    #[arg(long)] refined_noise_assumption: bool,
    #[arg(long)] split_trace_fft: bool,
    #[arg(long)] cc2_big_key: bool,
    #[arg(long)] parameters: Option<PathBuf>,
    #[arg(long)] allow_unverified_parameters: bool,
    #[arg(long)] inspect: bool,
    #[arg(long)] output: PathBuf,
}

fn splitmix64(mut state: u64) -> u64 {
    state = state.wrapping_add(0x9E3779B97F4A7C15);
    let z = (state ^ (state >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn operand(args: &Args, trial: usize, side: usize) -> U256 {
    let mut words = [0u64; 4];
    for digit in 0..args.width / 2 {
        let value = match args.pattern {
            Pattern::Random => splitmix64(args.seed
                ^ (trial as u64).wrapping_mul(0x9E3779B97F4A7C15)
                ^ (side as u64).wrapping_mul(0xD1B54A32D192ED03)
                ^ (digit as u64).wrapping_mul(0x2545F4914F6CDD1D)) % 4,
            Pattern::Zero => 0, Pattern::Max => 3,
            Pattern::Alternating => 1 + (digit + side) as u64 % 2,
        };
        words[digit / 32] |= value << (2 * (digit % 32));
    }
    U256::from(words)
}

fn digit(value: U256, index: usize, mask: u64) -> u64 {
    u64::cast_from(value >> index as u32) & mask
}

fn encrypt(value: U256, args: &Args, secret: &crypto::Secrets, keys: &crypto::EvaluationKeys) -> Vec<crypto::Lwe> {
    let mut seeder = new_seeder();
    let mut rng = EncryptionRandomGenerator::<DefaultRandomGenerator>::new(seeder.seed(), seeder.as_mut());
    (0..args.width / 2).map(|i| {
        if args.parameters.is_some() || keys.params.cc2_big_key {
            // Prepare genuine closed PBS outputs (and, for small-key CC2, their final KS) outside the timer.
            let raw = allocate_and_encrypt_new_lwe_ciphertext(&secret.big,
                Plaintext(digit(value, 2 * i, 3) << 61),
                DynamicDistribution::new_gaussian_from_std_dev(StandardDev(keys.params.glwe_sigma)),
                crypto::modulus(), &mut rng);
            crypto::emit(keys, &raw, args.boundary_delta_log2).1
        } else {
            allocate_and_encrypt_new_lwe_ciphertext(&secret.small,
                Plaintext(digit(value, 2 * i, 3) << args.boundary_delta_log2),
                DynamicDistribution::new_gaussian_from_std_dev(StandardDev(keys.params.lwe_sigma)),
                crypto::modulus(), &mut rng)
        }
    }).collect()
}

fn decode(key: &LweSecretKeyOwned<u64>, input: &crypto::Lwe, delta_log: u32) -> u64 {
    decrypt_lwe_ciphertext(key, input).0.wrapping_add(1 << (delta_log - 1)) >> delta_log
}

fn expected_terms(plan: &Plan, x: U256, y: U256) -> Vec<u64> {
    let mut values = vec![0; plan.value_count];
    for (i, job) in plan.products.iter().enumerate() {
        values[i] = ((digit(x, 8 * job.left, 255) * digit(y, 8 * job.right, 255)) >> job.bit) & 1;
    }
    for layer in &plan.layers {
        for job in &layer.jobs {
            let sum: u64 = job.inputs.iter().map(|&i| values[i]).sum();
            values[job.parity] = sum % 2;
            if let Some(id) = job.carry { values[id] = sum / 2; }
        }
    }
    values
}

pub fn run() -> Result<()> {
    let args = Args::parse();
    ensure!([16, 32, 64, 128, 256].contains(&args.width), "unsupported width");
    ensure!(args.threads > 0 && args.repetitions > 0, "positive counts required");
    let params = if let Some(path) = &args.parameters {
        ensure!(args.allow_unverified_parameters, "candidate parameters require --allow-unverified-parameters");
        ensure!(!args.refined_noise_assumption && !args.split_trace_fft, "candidate JSON defines the noise and FFT settings");
        ensure!(args.boundary_delta_log2 == 52, "retuned comparison must retain CC2 at delta=2^52");
        serde_json::from_slice::<Parameters>(&std::fs::read(path)?)?
    } else {
        ensure!(args.refined_noise_assumption, "explicitly acknowledge the unconfirmed Refined noise values");
        Parameters::reported(args.split_trace_fft)
    };
    let mut params = params;
    params.cc2_big_key |= args.cc2_big_key;
    params.validate()?;
    ensure!([52, 61].contains(&args.boundary_delta_log2), "supported boundary scales are 52 and 61");
    ensure!(args.boundary_delta_log2 == 52 || args.allow_alternate_boundary,
        "delta=2^61 changes the paper's CC2 boundary; use --allow-alternate-boundary explicitly");
    ensure!(!args.output.exists(), "use a fresh output directory");
    std::fs::create_dir_all(&args.output)?;
    rayon::ThreadPoolBuilder::new().num_threads(args.threads).build_global()?;
    std::env::set_var("CBS_CENTERED_MS", "1");
    std::env::set_var("CBS_CENTER_SELECTOR_BOX", "1");
    let plan = Plan::new(args.width);
    std::fs::write(args.output.join("plan.json"), serde_json::to_vec_pretty(&plan)?)?;
    let metadata = json!({
        "method": "independent revised-ST reconstruction", "author_implementation": false,
        "publication_ready": false, "whole_multiplier_failure_bound": null,
        "paper_revision": "2026-07-30", "tfhe_rs_version": "1.6.1",
        "paper_tfhe_rs_version": "1.7.0", "width": args.width, "threads": args.threads,
        "seed": args.seed, "pattern": format!("{:?}", args.pattern),
        "repetitions": args.repetitions, "warmup": args.warmup, "inspect": args.inspect,
        "parameter_set": params,
        "retuned_candidate": args.parameters.is_some(),
        "lwe_dimension": params.lwe_dimension, "N": params.polynomial_size, "k": params.glwe_dimension,
        "pbs": params.pbs, "ks": params.ks, "trace": params.auto, "ss": params.ss,
        "ggsw": params.cbs, "theta_input": 2,
        "theta_terminal": params.terminal_lut_count_log,
        "trace_fft": if params.split_trace_fft { "split40" } else { "vanilla" },
        "lwe_sigma": params.lwe_sigma, "glwe_sigma": params.glwe_sigma,
        "noise_source": if args.parameters.is_some() { "independently selected candidate; see parameter_set" }
            else { "explicit Refined assumption; see ../shokri-revised-comparison/noise_assumption_refined.json" },
        "boundary_delta_log2": args.boundary_delta_log2,
        "paper_CC2_boundary": args.boundary_delta_log2 == 52,
        "cc2_key": if params.cc2_big_key { "extracted big key, pre-PBS key switching" } else { "small key, closing key switch" },
        "input_preparation": if args.parameters.is_some() {
            "identity PBS and closing KS produce small-key CC2 inputs outside timing"
        } else { "fresh small-key Gaussian encryptions; excluded from timing; not asserted to meet the paper's bootstrapped variance contract" },
        "grouped_selector_box_centering": "q/16; adopted from local audited lift, not confirmed author configuration",
        "scope": "Diagnostic reconstruction, including final PBS and closing KS. Inspection is outside stage timers but inside total_s; inspected total_s is not a benchmark latency. No author speedup or target-128 claim.",
        "total_s_includes_inspection": args.inspect,
        "counts": plan.counts_for_packing(params.polynomial_size, params.terminal_lut_count_log, params.cc2_big_key)
    });
    std::fs::write(args.output.join("parameters.json"), serde_json::to_vec_pretty(&metadata)?)?;
    let (secret, keys) = crypto::generate_keys(&params);
    let mask = U256::MAX >> (256 - args.width) as u32;
    let mut csv = csv::Writer::from_path(args.output.join("timings.csv"))?;
    csv.write_record(["trial", "warmup", "lift_s", "product_s", "reduction_s", "terminal_s", "emission_s",
        "total_s", "phase_sum_s", "product_errors", "reduction_errors", "pre_emission_errors",
        "post_pbs_errors", "output_errors", "ok", "counts"])?;
    for trial in 0..args.warmup + args.repetitions {
        let x = operand(&args, trial, 0);
        let y = operand(&args, trial, 1);
        let expected = (x * y) & mask;
        let lhs = encrypt(x, &args, &secret, &keys);
        let rhs = encrypt(y, &args, &secret, &keys);
        for (value, inputs) in [(x, &lhs), (y, &rhs)] {
            ensure!(inputs.iter().enumerate().all(|(i, c)|
                decode(if params.cc2_big_key { &secret.big } else { &secret.small }, c, args.boundary_delta_log2)
                    == digit(value, 2 * i, 3)),
                "input preparation failed before multiplication");
        }
        keys.counts.reset();
        let started = Instant::now();
        let clock = Instant::now();
        let (ls, rs) = rayon::join(|| evaluate::input_selectors(&keys, &lhs, args.boundary_delta_log2),
                                   || evaluate::input_selectors(&keys, &rhs, args.boundary_delta_log2));
        let lift_s = clock.elapsed().as_secs_f64();
        let clock = Instant::now();
        let mut values = evaluate::products(&keys, &plan, &ls, &rs);
        let product_s = clock.elapsed().as_secs_f64();
        let reference = args.inspect.then(|| expected_terms(&plan, x, y));
        let product_errors = reference.as_ref().map(|r| (0..plan.products.len())
            .filter(|&id| decode(&secret.big, values[id].as_ref().unwrap(), 61) != r[id]).count());
        let clock = Instant::now();
        evaluate::reduce(&keys, &plan, &mut values);
        let reduction_s = clock.elapsed().as_secs_f64();
        let reduction_errors = reference.as_ref().map(|r| plan.final_columns.iter().flatten()
            .filter(|&&id| decode(&secret.big, values[id].as_ref().unwrap(), 61) != r[id]).count());
        let clock = Instant::now();
        let padded = evaluate::terminal(&keys, &plan, &values);
        let terminal_s = clock.elapsed().as_secs_f64();
        let pre_errors = if args.inspect { Some(padded.iter().enumerate()
            .filter(|(i, c)| decode(&secret.big, c, 61) != digit(expected, 2 * i, 3)).count()) } else { None };
        let clock = Instant::now();
        let emitted = evaluate::emission(&keys, &padded, args.boundary_delta_log2);
        let emission_s = clock.elapsed().as_secs_f64();
        let total_s = started.elapsed().as_secs_f64();
        let post_errors = emitted.iter().enumerate().filter(|(i, (big, _))|
            decode(&secret.big, big, args.boundary_delta_log2) != digit(expected, 2 * i, 3)).count();
        let output_errors = emitted.iter().enumerate().filter(|(i, (_, small))|
            decode(if params.cc2_big_key { &secret.big } else { &secret.small }, small, args.boundary_delta_log2)
                != digit(expected, 2 * i, 3)).count();
        let outputs_ok = output_errors == 0 && post_errors == 0
            && [product_errors, reduction_errors, pre_errors].iter().all(|e| e.unwrap_or(0) == 0);
        let counts = keys.counts.snapshot();
        let expected_counts = plan.counts_for_packing(params.polynomial_size, params.terminal_lut_count_log, params.cc2_big_key);
        let counts_ok = counts["blind_rotations"] == expected_counts["total_blind_rotations"]
            && counts["key_switches"] == expected_counts["key_switches"]
            && counts["conversions"] == expected_counts["ggsw_conversions"]
            && counts["cmux"].as_u64().unwrap() == expected_counts["product_cmux"].as_u64().unwrap()
                + expected_counts["terminal_cmux"].as_u64().unwrap();
        let ok = outputs_ok && counts_ok;
        let phase_sum_s = lift_s + product_s + reduction_s + terminal_s + emission_s;
        csv.write_record([trial.to_string(), (trial < args.warmup).to_string(), lift_s.to_string(),
            product_s.to_string(), reduction_s.to_string(), terminal_s.to_string(), emission_s.to_string(),
            total_s.to_string(), phase_sum_s.to_string(), product_errors.map(|n| n.to_string()).unwrap_or_default(),
            reduction_errors.map(|n| n.to_string()).unwrap_or_default(), pre_errors.map(|n| n.to_string()).unwrap_or_default(),
            post_errors.to_string(), output_errors.to_string(), ok.to_string(), counts.to_string()])?;
        csv.flush()?;
        println!("trial={trial} total={total_s:.6}s product_errors={product_errors:?} reduction_errors={reduction_errors:?} pre_emission_errors={pre_errors:?} post_pbs_errors={post_errors} output_errors={output_errors}");
        ensure!(ok, "correctness/count check failed; diagnostic row preserved, not a valid latency result");
    }
    Ok(())
}
