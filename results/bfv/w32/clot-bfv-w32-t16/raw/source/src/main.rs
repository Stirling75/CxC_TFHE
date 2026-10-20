mod arithmetic;
mod crypto;

use anyhow::{bail, ensure, Result};
use arithmetic::{convolution, precision, NATIVE_DELTA};
use clap::Parser;
use crypto::{Client, Lwe, Parameters};
use num_bigint::BigUint;
use rand::{Rng, SeedableRng};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};
use tfhe::core_crypto::prelude::*;

#[derive(Parser, Debug)]
#[command(about = "Independent CLOT-style radix multiplier with conditional research parameters")]
struct Args {
    #[arg(long, default_value = "deep-encoding-diagnostic", value_parser = ["st-diagnostic", "precision-diagnostic", "deep-encoding-diagnostic"])]
    preset: String,
    #[arg(long, conflicts_with = "preset")]
    parameters: Option<PathBuf>,
    #[arg(long)]
    coefficient_padding: bool,
    #[arg(long, num_args = 1.., default_values_t = [8usize])]
    widths: Vec<usize>,
    #[arg(long, num_args = 1.., default_values = ["random"])]
    patterns: Vec<String>,
    #[arg(long, default_value_t = 1)]
    repetitions: usize,
    #[arg(long, default_value_t = 0)]
    warmup: usize,
    #[arg(long, default_value_t = 1)]
    threads: usize,
    #[arg(long)]
    verify_serial: bool,
    #[arg(long, conflicts_with_all = ["kernel_only", "plan_only"])]
    benchmark: bool,
    #[arg(long, default_value_t = 20260918)]
    seed: u64,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    kernel_only: bool,
    #[arg(long)]
    plan_only: bool,
}

fn integer(digits: &[u64]) -> BigUint {
    digits
        .iter()
        .rev()
        .fold(BigUint::from(0u8), |x, d| (x << 2usize) + d)
}

fn operands(d: usize, pattern: &str, rng: &mut impl Rng) -> (Vec<u64>, Vec<u64>) {
    match pattern {
        "zero" => (vec![0; d], (0..d).map(|_| rng.gen_range(0..4)).collect()),
        "one" => {
            let mut a = vec![0; d];
            a[0] = 1;
            (a, vec![3; d])
        }
        "max" => {
            let a = vec![3; d];
            let mut b = a.clone();
            b[0] = 2;
            (a, b)
        }
        "alternating" => (
            (0..d).map(|i| if i % 2 == 0 { 3 } else { 0 }).collect(),
            (0..d).map(|i| if i % 2 == 1 { 3 } else { 0 }).collect(),
        ),
        "impulse" => {
            let mut a = vec![0; d];
            a[d - 1] = 3;
            (a, vec![3; d])
        }
        "random" => (
            (0..d).map(|_| rng.gen_range(0..4)).collect(),
            (0..d).map(|_| rng.gen_range(0..4)).collect(),
        ),
        _ => unreachable!(),
    }
}

// Decryption is confined to this harness. Neither kernel nor finish sees a key
// or a plaintext, and their elapsed times exclude these stage checks.
fn check(client: &Client, ciphertexts: &[Lwe], expected: &[u64], delta: u64) -> Value {
    assert_eq!(ciphertexts.len(), expected.len());
    let plaintext_modulus = (1u128 << 64) / delta as u128;
    let mut max_ratio = 0.0f64;
    let mut errors = Vec::new();
    let mut decoded_digits = Vec::new();
    for (i, (ct, &m)) in ciphertexts.iter().zip(expected).enumerate() {
        let phase = decrypt_lwe_ciphertext(&client.big, ct).0;
        let decoded =
            (((phase as u128 + delta as u128 / 2) / delta as u128) % plaintext_modulus) as u64;
        decoded_digits.push(decoded);
        let error = phase.wrapping_sub(m.wrapping_mul(delta)) as i64;
        max_ratio = max_ratio.max((error as f64).abs() / (delta as f64 / 2.0));
        if decoded != m {
            errors.push(json!({"index": i, "expected": m, "decoded": decoded}));
        }
    }
    json!({"checked": expected.len(), "correct": errors.is_empty(), "errors": errors,
        "decoded_digits": decoded_digits,
        "max_abs_phase_error_over_decoding_margin": max_ratio,
        "is_failure_probability_bound": false})
}

fn save(path: &Path, value: &Value) -> Result<()> {
    let temporary = path.with_extension("json.tmp");
    fs::write(&temporary, serde_json::to_string_pretty(value)? + "\n")?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.repetitions > 0, "repetitions must be positive");
    ensure!(args.threads > 0, "threads must be positive");
    ensure!(
        !args.verify_serial || (args.threads > 1 && !args.kernel_only),
        "--verify-serial requires multiple threads and the full path"
    );
    let trials = args
        .repetitions
        .checked_add(args.warmup)
        .ok_or_else(|| anyhow::anyhow!("too many trials"))?;
    ensure!(
        args.widths
            .iter()
            .all(|w| [8, 16, 32, 64, 128, 256].contains(w)),
        "supported widths: 8,16,32,64,128,256"
    );
    ensure!(
        args.patterns.iter().all(
            |p| ["zero", "one", "max", "alternating", "impulse", "random"].contains(&p.as_str())
        ),
        "invalid pattern"
    );
    for values in [
        &args
            .widths
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>(),
        &args.patterns,
    ] {
        ensure!(
            values
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len()
                == values.len(),
            "duplicate cases"
        );
    }
    fs::create_dir(&args.output)?;
    let params = if let Some(path) = &args.parameters {
        serde_json::from_slice::<Parameters>(&fs::read(path)?)?
    } else {
        match args.preset.as_str() {
            "precision-diagnostic" => Parameters::precise_encoding(),
            "deep-encoding-diagnostic" => Parameters::deep_encoding(),
            _ => Parameters::default(),
        }
    };
    params.validate()?;
    let plans: Vec<_> = args
        .widths
        .iter()
        .map(|&w| {
            let (input, bits, norm) = params.pbs_counts(w / 2);
            let p = precision(w / 2) + usize::from(args.coefficient_padding);
            let graph = params
                .degree_aware
                .then(|| arithmetic::normalizer_plan(w / 2, params.normalizer_lut_domain));
            let expected_ks = input
                + bits
                + if args.threads > 1 {
                    graph.as_ref().map_or(norm, |g| g.ks_count())
                } else {
                    norm
                };
            json!({"width": w, "radix": 4, "digits": w / 2,
        "max_convolution_coefficient": 9 * w / 2, "encoding_precision": p,
        "plaintext_modulus": 1u64 << p, "polynomial_degree": w - 2,
        "no_negacyclic_wrap": w - 2 < params.polynomial_size,
        "pbs_input_rescale": input, "pbs_bit_extraction": bits, "pbs_normalization": norm,
        "pbs_total": input + bits + norm, "ks_total": expected_ks,
        "normalizer_dependency_stages": graph.as_ref().map(|g| g.stages.len()),
        "normalizer_tasks": graph.as_ref().map(|g| g.ks_count()),
        "cbs": 0, "standalone_cmux": 0})
        })
        .collect();
    let mut manifest = json!({"method": "independent-clot-style-convolution-probe", "tfhe_version": "1.6.1",
        "preset": if args.parameters.is_some() { "external-parameters" } else { &args.preset },
        "parameter_file": args.parameters, "coefficient_padding": args.coefficient_padding,
        "parameters": params, "plans": plans, "patterns": args.patterns,
        "repetitions": args.repetitions, "warmup": args.warmup, "verify_serial": args.verify_serial,
        "plaintext_seed": args.seed, "key_randomness": "OS seeder, fresh per process",
        "kernel_only": args.kernel_only, "plan_only": args.plan_only,
        "benchmark": args.benchmark,
        "timing_contract": if args.benchmark { "continuous-evaluator-wall-clock-v1" }
            else { "diagnostic-stage-sum-v1" },
        "whole_multiplier_approved": false, "security_estimate_for_new_key_set": null,
        "failure_bound": null, "competitive_benchmark": false,
        "input_contract": "bootstrapped big-key radix-4, delta=2^59",
        "output_contract": "same-key radix-4, delta=2^59, lower W bits (full mode only)",
        "normalizer": if params.degree_aware && args.threads > 1 { "dependency-graph execution of the serial public-degree partition; shared low/carry KS" }
            else if params.degree_aware { "public-degree-bounded sequential reduction and carry propagation" }
            else { "sequential four-to-two reduction and carry propagation; reference implementation" },
        "tensor_backend": "TFHE-rs exact u128 polynomial product modulo 2^128, centered lifts",
        "sources": ["https://eprint.iacr.org/2021/729.pdf", "https://www.iacr.org/archive/asiacrypt2021/130900334/130900334.pdf"]});
    let compiled_sources = [
        ("src/main.rs", include_str!("main.rs")),
        ("src/crypto.rs", include_str!("crypto.rs")),
        ("src/arithmetic.rs", include_str!("arithmetic.rs")),
        ("Cargo.toml", include_str!("../Cargo.toml")),
        ("Cargo.lock", include_str!("../Cargo.lock")),
    ];
    for (name, contents) in compiled_sources {
        let path = args.output.join("source").join(name);
        fs::create_dir_all(path.parent().unwrap())?;
        fs::write(path, contents)?;
    }
    let source_hashes: std::collections::BTreeMap<_, _> = compiled_sources
        .into_iter()
        .map(|(name, contents)| (name, blake3::hash(contents.as_bytes()).to_hex().to_string()))
        .collect();
    manifest["compiled_source_blake3"] = serde_json::to_value(source_hashes)?;
    manifest["host"] = json!({"os": std::env::consts::OS, "architecture": std::env::consts::ARCH,
        "evaluation_threads": args.threads, "cpu_affinity_set_by_adapter": false,
        "external_affinity_record": "campaign run.json, when launched by the campaign runner"});
    manifest["executable_blake3"] = json!(blake3::hash(&fs::read(std::env::current_exe()?)?)
        .to_hex()
        .to_string());
    save(&args.output.join("manifest.json"), &manifest)?;
    if args.plan_only {
        println!("{}", serde_json::to_string_pretty(&manifest)?);
        return Ok(());
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build()?;
    ensure!(
        pool.current_num_threads() == args.threads,
        "thread pool mismatch"
    );
    let serial_pool = if args.verify_serial {
        Some(rayon::ThreadPoolBuilder::new().num_threads(1).build()?)
    } else {
        None
    };
    let start = Instant::now();
    let (client, evaluator) = crypto::generate_keys(params);
    manifest["key_generation_seconds"] = json!(start.elapsed().as_secs_f64());
    manifest["evaluation_key_payload_bytes"] = evaluator.key_bytes();
    save(&args.output.join("manifest.json"), &manifest)?;
    let mut completed = 0;
    for &width in &args.widths {
        for pattern in &args.patterns {
            for repetition in 0..trials {
                let seed = args.seed ^ ((width as u64) << 32) ^ repetition as u64;
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                let (a, b) = operands(width / 2, pattern, &mut rng);
                let p = precision(width / 2) + usize::from(args.coefficient_padding);
                eprintln!("W={width}, pattern={pattern}, repetition={repetition}, precision={p}");
                let file = args
                    .output
                    .join(format!("w{width}-{pattern}-r{repetition}.json"));
                let mut row = json!({"width": width, "pattern": pattern, "repetition": repetition,
                    "warmup": repetition < args.warmup, "threads": args.threads,
                    "a_hex": integer(&a).to_str_radix(16), "b_hex": integer(&b).to_str_radix(16),
                    "precision": p, "whole_multiplier_approved": false, "status": "preparing-inputs"});
                save(&file, &row)?;
                let now = Instant::now();
                let ca = evaluator.prepare_inputs(&client, &a);
                let cb = evaluator.prepare_inputs(&client, &b);
                row["input_preparation_seconds_excluded"] = json!(now.elapsed().as_secs_f64());
                row["native_input_a"] = check(&client, &ca, &a, NATIVE_DELTA);
                row["native_input_b"] = check(&client, &cb, &b, NATIVE_DELTA);
                if row["native_input_a"]["correct"] != true
                    || row["native_input_b"]["correct"] != true
                {
                    row["status"] = json!("failed-input-preparation");
                    save(&file, &row)?;
                    bail!("input failure: {}", file.display());
                }
                let (mut kernel, timed_output) = if args.benchmark {
                    let now = Instant::now();
                    let evaluated = pool.install(|| {
                        let mut kernel = evaluator.kernel(&ca, &cb, p);
                        let output = evaluator.finish(&mut kernel, p);
                        (kernel, Some(output))
                    });
                    row["total_seconds"] = json!(now.elapsed().as_secs_f64());
                    evaluated
                } else {
                    (pool.install(|| evaluator.kernel(&ca, &cb, p)), None)
                };
                let delta = 1u64 << (64 - p);
                row["rescaled_a"] = check(&client, &kernel.rescaled_a, &a, delta);
                row["rescaled_b"] = check(&client, &kernel.rescaled_b, &b, delta);
                for (name, ct, expected) in [
                    ("packed_a", &kernel.packed_a, &a),
                    ("packed_b", &kernel.packed_b, &b),
                ] {
                    let cts: Vec<_> = (0..expected.len())
                        .map(|i| evaluator.extract(ct, i))
                        .collect();
                    row[name] = check(&client, &cts, expected, delta);
                }
                let c = convolution(&a, &b);
                row["convolution"] = check(&client, &kernel.coefficients, &c[..width / 2], delta);
                row["timings_seconds"] = serde_json::to_value(&kernel.timings)?;
                row["counts"] = serde_json::to_value(&kernel.counts)?;
                let kernel_ok = [
                    "rescaled_a",
                    "rescaled_b",
                    "packed_a",
                    "packed_b",
                    "convolution",
                ]
                .iter()
                .all(|s| row[s]["correct"] == true);
                row["status"] = json!(if kernel_ok {
                    "kernel-correct"
                } else {
                    "failed-kernel"
                });
                save(&file, &row)?;
                ensure!(kernel_ok, "kernel failure: {}", file.display());
                if !args.kernel_only {
                    let output = timed_output
                        .unwrap_or_else(|| pool.install(|| evaluator.finish(&mut kernel, p)));
                    let (input_pbs, bit_pbs, norm_pbs) = evaluator.p.pbs_counts(width / 2);
                    ensure!(
                        kernel.counts.pbs == input_pbs + bit_pbs + norm_pbs,
                        "public PBS count mismatch"
                    );
                    let expected_ks = input_pbs
                        + bit_pbs
                        + if evaluator.p.degree_aware && args.threads > 1 {
                            arithmetic::normalizer_plan(
                                width / 2,
                                evaluator.p.normalizer_lut_domain,
                            )
                            .ks_count()
                        } else {
                            norm_pbs
                        };
                    ensure!(kernel.counts.ks == expected_ks, "public KS count mismatch");
                    let product = (integer(&a) * integer(&b)) % (BigUint::from(1u8) << width);
                    let expected: Vec<_> = (0..width / 2)
                        .map(|i| {
                            ((&product >> (2 * i)) % BigUint::from(4u8))
                                .to_u64_digits()
                                .first()
                                .copied()
                                .unwrap_or(0)
                        })
                        .collect();
                    row["output"] = check(&client, &output, &expected, NATIVE_DELTA);
                    row["expected_hex"] = json!(product.to_str_radix(16));
                    row["timings_seconds"] = serde_json::to_value(&kernel.timings)?;
                    row["counts"] = serde_json::to_value(&kernel.counts)?;
                    let ok = row["output"]["correct"] == true;
                    row["status"] = json!(if ok {
                        "full-path-correct"
                    } else {
                        "failed-output"
                    });
                    save(&file, &row)?;
                    ensure!(ok, "output failure: {}", file.display());
                    if let Some(serial_pool) = &serial_pool {
                        eprintln!("  checking identical inputs against serial execution");
                        let (reference, serial_output) = serial_pool.install(|| {
                            let mut reference = evaluator.kernel(&ca, &cb, p);
                            let output = evaluator.finish(&mut reference, p);
                            (reference, output)
                        });
                        let same_lwes = |a: &[Lwe], b: &[Lwe]| {
                            a.len() == b.len()
                                && a.iter().zip(b).all(|(x, y)| x.as_ref() == y.as_ref())
                        };
                        let same = same_lwes(&kernel.rescaled_a, &reference.rescaled_a)
                            && same_lwes(&kernel.rescaled_b, &reference.rescaled_b)
                            && kernel.packed_a.as_ref() == reference.packed_a.as_ref()
                            && kernel.packed_b.as_ref() == reference.packed_b.as_ref()
                            && same_lwes(&kernel.coefficients, &reference.coefficients)
                            && same_lwes(&output, &serial_output)
                            && kernel.counts.pbs == reference.counts.pbs;
                        row["serial_equivalence"] = json!({"ciphertexts_identical": same,
                            "same_keys_and_inputs": true, "reference_counts": reference.counts,
                            "reference_timings_seconds_excluded": reference.timings});
                        if !same {
                            row["status"] = json!("failed-serial-equivalence");
                        }
                        save(&file, &row)?;
                        ensure!(
                            same,
                            "serial/parallel ciphertext mismatch: {}",
                            file.display()
                        );
                    }
                }
                completed += 1;
                println!("{}: {}", file.display(), row["status"]);
            }
        }
    }
    save(
        &args.output.join("completed.json"),
        &json!({"cases": completed, "kernel_only": args.kernel_only,
        "threads": args.threads, "warmup_per_configuration": args.warmup,
        "whole_multiplier_approved": false, "competitive_benchmark": false}),
    )?;
    Ok(())
}
