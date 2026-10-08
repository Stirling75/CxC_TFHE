//! Our reimplementation of the radix multiplier of Bernard, Carouge, Joye,
//! Orfila and Tap (ePrint 2026/2310, Section 4.2, "Ours (w/ shift)"):
//! multi-value partial products and folds with the common factors of their
//! Table 4, a public noise-aware folding schedule (heaviest rule, balanced
//! column order; computed offline by `mvb-mul/schedule.py`), and the
//! unchanged TFHE-rs carry propagation. The schedule is executed tick by
//! tick: a tick holds at most T blind rotations that run in parallel.
use anyhow::{bail, ensure, Context, Result};
use clap::Parser;
use dyn_stack::{PodBuffer, PodStack};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::json;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;
use tfhe::core_crypto::prelude::*;
use tfhe::integer::bigint::U256;
use tfhe::integer::{gen_keys_radix, IntegerCiphertext, RadixCiphertext};
use tfhe::shortint::atomic_pattern::AtomicPatternServerKey;
use tfhe::shortint::ciphertext::{Degree, NoiseLevel};
use tfhe::shortint::parameters::current_params::V1_6_PARAM_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M128;
use tfhe::shortint::parameters::{DecompositionBaseLog, DecompositionLevelCount, DynamicDistribution, LweDimension, StandardDev};
use tfhe::shortint::server_key::ShortintBootstrappingKey;
use tfhe::shortint::Ciphertext;

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
    /// Offline schedule for this width and thread count (schedule.py).
    #[arg(long)]
    schedule: PathBuf,
    /// Key-switching decomposition (base log, level); the paper set PG,12.8 uses 4,4.
    #[arg(long, default_value = "4,4")]
    ks: String,
    /// Multiply an operand by itself (one ciphertext on both sides).
    #[arg(long)]
    identical: bool,
    #[arg(long)]
    output: PathBuf,
    /// Noise probe (not timed): record the error of every blind-rotation input
    /// before and after the key switch to <output>/probe.csv.
    #[arg(long)]
    probe: bool,
}

#[derive(Deserialize)]
struct Op {
    kind: String,
    #[serde(default)]
    i: usize,
    #[serde(default)]
    j: usize,
    #[serde(default, rename = "in")]
    inputs: Vec<u64>,
    #[serde(default)]
    in_types: Vec<String>,
    out: Vec<(usize, String, u64)>,
}

#[derive(Deserialize)]
struct Schedule {
    width: usize,
    threads: usize,
    phi: f64,
    ticks: Vec<Vec<Op>>,
    #[serde(rename = "final")]
    final_columns: Vec<Vec<(u64, String)>>,
}

const N: usize = 2048;
const DELTA: u64 = 1 << 59; // q/32: four message bits and the padding bit

/// Common factor and extraction filters at N = 2048 (ePrint 2026/2310, Table 4,
/// "Ours (w/ shift)"; witnesses checked in mvb-review/mvb_foundation.py).
/// Each output is sum_j w_j * SampleExtract_j(ACC) + shift * Delta.
struct Family {
    accumulator: GlweCiphertextOwned<u64>,
    outputs: [(&'static [(usize, i64)], u64); 2],
}

const FOLD_LOW: &[(usize, i64)] = &[(1984, -1), (1856, -1), (1728, -1), (1600, 2), (64, -1)];
const FOLD_HIGH: &[(usize, i64)] = &[(64, 1)];
const PRODUCT_LOW: &[(usize, i64)] = &[(1856, 1), (1728, -1), (1472, 1), (1344, -1), (1088, 2),
    (832, 1), (704, -1), (576, 2), (448, -1), (320, 1), (64, 2)];
const PRODUCT_HIGH: &[(usize, i64)] = &[(1984, -1), (1600, -1), (1216, -1), (832, -1), (192, -1)];

fn family(body: impl Fn(usize) -> i64, outputs: [(&'static [(usize, i64)], u64); 2]) -> Family {
    let coefficients: Vec<u64> = (0..N).map(|i| body(i).wrapping_mul(DELTA as i64) as u64).collect();
    let accumulator = allocate_and_trivially_encrypt_new_glwe_ciphertext(
        GlweSize(2), &PlaintextList::from_container(coefficients), CiphertextModulus::new_native());
    Family { accumulator, outputs }
}

struct Engine<'a> {
    ksk: &'a LweKeyswitchKeyOwned<u64>,
    bsk: &'a FourierLweBootstrapKeyOwned,
    fold: Family,
    product: Family,
}

thread_local! {
    static SCRATCH: RefCell<Option<(Fft, PodBuffer)>> = const { RefCell::new(None) };
}

impl Engine<'_> {
    /// Key switch, centered modulus switch (as TFHE-rs for CenteredMeanNoiseReduction),
    /// one blind rotation of the common factor, and the two filtered outputs.
    fn multi_value(&self, input: &LweCiphertextOwned<u64>, fam: &Family) -> [LweCiphertextOwned<u64>; 2] {
        let mut small = LweCiphertext::new(0u64, self.ksk.output_key_lwe_dimension().to_lwe_size(),
            CiphertextModulus::new_native());
        keyswitch_lwe_ciphertext(self.ksk, input, &mut small);
        let switched = lwe_ciphertext_centered_binary_modulus_switch(small.as_view(),
            PolynomialSize(N).to_blind_rotation_input_modulus_log());
        let mut acc = fam.accumulator.clone();
        SCRATCH.with(|cell| {
            let mut cell = cell.borrow_mut();
            let (fft, memory) = cell.get_or_insert_with(|| {
                let fft = Fft::new(PolynomialSize(N));
                let memory = PodBuffer::try_new(blind_rotate_assign_mem_optimized_requirement::<u64>(
                    GlweSize(2), PolynomialSize(N), fft.as_view())).expect("scratch");
                (fft, memory)
            });
            blind_rotate_assign_mem_optimized(&switched, &mut acc, self.bsk, fft.as_view(),
                &mut PodStack::new(memory));
        });
        let size = self.bsk.output_lwe_dimension().to_lwe_size();
        let mut samples: HashMap<usize, LweCiphertextOwned<u64>> = HashMap::new();
        fam.outputs.map(|(filter, shift)| {
            let mut out = LweCiphertext::new(0u64, size, CiphertextModulus::new_native());
            for &(index, weight) in filter {
                let sample = samples.entry(index).or_insert_with(|| {
                    let mut s = LweCiphertext::new(0u64, size, CiphertextModulus::new_native());
                    extract_lwe_sample_from_glwe_ciphertext(&acc, &mut s, MonomialDegree(index));
                    s
                });
                for (dst, &src) in out.as_mut().iter_mut().zip(sample.as_ref()) {
                    *dst = dst.wrapping_add(src.wrapping_mul(weight as u64));
                }
            }
            *out.get_mut_body().data = out.get_body().data.wrapping_add(shift * DELTA);
            out
        })
    }
}

fn splitmix64(mut state: u64) -> u64 {
    state = state.wrapping_add(0x9E3779B97F4A7C15);
    let z = (state ^ (state >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Same operand generator as tfhe_radix_baseline and the hybrid runner.
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
    ensure!(!args.output.exists(), "use a fresh output directory");
    let schedule: Schedule = serde_json::from_slice(&std::fs::read(&args.schedule)
        .with_context(|| format!("read {}", args.schedule.display()))?)?;
    ensure!(schedule.width == args.width && schedule.threads == args.threads,
        "schedule is for W={} T={}", schedule.width, schedule.threads);
    std::fs::create_dir_all(&args.output)?;
    rayon::ThreadPoolBuilder::new().num_threads(args.threads).build_global()?;

    // PG,12.8 of ePrint 2026/2310 Table 1: n = 930 with the LWE noise of the
    // TFHE-rs 2_3 set, the GLWE side and PBS gadget of the default 2_2 set.
    let ks: Vec<usize> = args.ks.split(',').map(str::parse).collect::<Result<_, _>>()?;
    ensure!(ks.len() == 2, "--ks takes base_log,level");
    let mut params = V1_6_PARAM_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M128;
    params.lwe_dimension = LweDimension(930);
    params.lwe_noise_distribution = DynamicDistribution::new_gaussian_from_std_dev(StandardDev(6.782362904013915e-07));
    params.ks_base_log = DecompositionBaseLog(ks[0]);
    params.ks_level = DecompositionLevelCount(ks[1]);
    ensure!(params.polynomial_size.0 == N && params.glwe_dimension.0 == 1, "N=2048, k=1 expected");

    let metadata = json!({
        "method": "Bernard et al. (ePrint 2026/2310) radix multiplier, our reimplementation",
        "tfhe_rs": "1.6.1", "parameter": "PG,12.8 (n=930, PBS 23,1)", "parameters": params,
        "lwe_dimension": params.lwe_dimension.0, "ks_base_log": ks[0], "ks_level": ks[1],
        "phi_max": schedule.phi, "rule": "heaviest", "order": "balanced",
        "width": args.width, "threads": args.threads, "seed": args.seed, "identical": args.identical,
        "repetitions": args.repetitions, "warmup": args.warmup,
        "ticks": schedule.ticks.len(),
        "blind_rotations": schedule.ticks.iter().map(Vec::len).sum::<usize>(),
        "input": "identity-PBS-refreshed clean radix-4 blocks; preparation excluded",
        "output": "clean radix-4 XY modulo 2^W (TFHE-rs full_propagate_parallelized)",
    });
    std::fs::write(args.output.join("parameters.json"), serde_json::to_vec_pretty(&metadata)?)?;

    let nb = args.width / 2;
    let (client, server) = gen_keys_radix(params, nb);
    let shortint: &tfhe::shortint::ServerKey = server.as_ref();
    let AtomicPatternServerKey::Standard(ap) = &shortint.atomic_pattern else { bail!("standard AP expected") };
    let ShortintBootstrappingKey::Classic { bsk, .. } = &ap.bootstrapping_key else { bail!("classic PBS expected") };
    let engine = Engine {
        ksk: &ap.key_switching_key,
        bsk,
        fold: family(|i| (i / (N / 4)) as i64 - 1, [(FOLD_LOW, 1), (FOLD_HIGH, 1)]),
        product: family(|i| (i >= 13 * N / 16) as i64, [(PRODUCT_LOW, 0), (PRODUCT_HIGH, 1)]),
    };
    let identity = shortint.generate_lookup_table(|x| x % 4);
    let message = shortint.generate_lookup_table(|x| x % 4);
    let product_low = shortint.generate_lookup_table_bivariate(|x, y| (x * y) % 4);
    let block = |lwe: LweCiphertextOwned<u64>, degree: u64, level: u64| {
        let mut ct = shortint.create_trivial(0);
        ct.ct = lwe;
        ct.degree = Degree::new(degree);
        ct.set_noise_level(NoiseLevel::NOMINAL * level, shortint.max_noise_level);
        ct
    };
    let mask = U256::MAX >> (256 - args.width) as u32;
    let shortint_client: &tfhe::shortint::ClientKey = client.as_ref().as_ref();
    let tfhe::shortint::client_key::atomic_pattern::AtomicPatternClientKey::Standard(client_ap) =
        &shortint_client.atomic_pattern else { bail!("standard AP expected") };
    let large_key = client_ap.large_lwe_secret_key();
    let small_key = client_ap.small_lwe_secret_key();
    // Signed distance of the phase to the nearest multiple of Delta, and the value.
    let error = |phase: u64| -> (i64, u64) {
        let value = phase.wrapping_add(DELTA / 2) / DELTA;
        (phase.wrapping_sub(value.wrapping_mul(DELTA)) as i64, value % 32)
    };
    let mut probe = if args.probe {
        let mut w = csv::Writer::from_path(args.output.join("probe.csv"))?;
        w.write_record(["trial", "kind", "in_types", "value", "pre_ks_error", "post_ks_error"])?;
        Some(w)
    } else { None };
    let mut csv = csv::Writer::from_path(args.output.join("timings.csv"))?;
    csv.write_record(["method", "width", "threads", "trial", "warmup", "total_seconds",
        "folding_seconds", "propagation_seconds", "blind_rotations", "propagation_pbs", "ok"])?;
    for trial in 0..args.warmup + args.repetitions {
        let x = operand(args.width, trial, 0, args.seed);
        let y = if args.identical { x } else { operand(args.width, trial, 1, args.seed) };
        let prepare = |value| -> Vec<Ciphertext> {
            let encrypted: RadixCiphertext = client.encrypt(value);
            encrypted.blocks().par_iter().map(|b| shortint.apply_lookup_table(b, &identity)).collect()
        };
        let lhs = prepare(x);
        let rhs = if args.identical { lhs.clone() } else { prepare(y) };

        let started = Instant::now();
        let mut terms: HashMap<u64, LweCiphertextOwned<u64>> = HashMap::new();
        let mut brs = 0usize;
        let trace = std::env::var("MVB_TRACE").is_ok();
        for tick in &schedule.ticks {
            if let Some(w) = probe.as_mut() {
                for op in tick {
                    let (input, types) = match op.kind.as_str() {
                        "pmvb" | "plsb" => {
                            let mut input = lhs[op.i].ct.clone();
                            for v in input.as_mut().iter_mut() { *v = v.wrapping_mul(4); }
                            lwe_ciphertext_add_assign(&mut input, &rhs[op.j].ct);
                            (input, if op.i == op.j && args.identical { "XX" } else { "XY" }.to_string())
                        }
                        _ => {
                            let mut sum = terms[&op.inputs[0]].clone();
                            for id in &op.inputs[1..] { lwe_ciphertext_add_assign(&mut sum, &terms[id]); }
                            (sum, String::new())
                        }
                    };
                    let types = if types.is_empty() { op.in_types.join("+") } else { types };
                    let (pre, value) = error(decrypt_lwe_ciphertext(&large_key, &input).0);
                    let mut small = LweCiphertext::new(0u64, engine.ksk.output_key_lwe_dimension().to_lwe_size(),
                        CiphertextModulus::new_native());
                    keyswitch_lwe_ciphertext(engine.ksk, &input, &mut small);
                    let (post, _) = error(decrypt_lwe_ciphertext(&small_key, &small).0);
                    w.write_record([trial.to_string(), op.kind.clone(), types, value.to_string(),
                        pre.to_string(), post.to_string()])?;
                }
            }
            let tick_started = Instant::now();
            let produced: Vec<Vec<(u64, LweCiphertextOwned<u64>)>> = tick.par_iter().map(|op| {
                match op.kind.as_str() {
                    "pmvb" | "plsb" => {
                        // TFHE-rs bivariate input: lhs * message_modulus + rhs.
                        let mut input = lhs[op.i].ct.clone();
                        for v in input.as_mut().iter_mut() { *v = v.wrapping_mul(4); }
                        lwe_ciphertext_add_assign(&mut input, &rhs[op.j].ct);
                        if op.kind == "pmvb" {
                            let [lo, hi] = engine.multi_value(&input, &engine.product);
                            vec![(op.out[0].2, lo), (op.out[1].2, hi)]
                        } else {
                            let r = shortint.unchecked_apply_lookup_table_bivariate(&lhs[op.i], &rhs[op.j], &product_low);
                            vec![(op.out[0].2, r.ct)]
                        }
                    }
                    "fmvb" | "fsingle" => {
                        let mut sum = terms[&op.inputs[0]].clone();
                        for id in &op.inputs[1..] { lwe_ciphertext_add_assign(&mut sum, &terms[id]); }
                        if op.kind == "fmvb" {
                            let [lo, hi] = engine.multi_value(&sum, &engine.fold);
                            vec![(op.out[0].2, lo), (op.out[1].2, hi)]
                        } else {
                            let r = shortint.apply_lookup_table(&block(sum, 15, op.inputs.len() as u64), &message);
                            vec![(op.out[0].2, r.ct)]
                        }
                    }
                    other => panic!("unknown op {other}"),
                }
            }).collect();
            brs += tick.len();
            if trace {
                eprintln!("tick {} ops {} kind {} ms {:.2}", brs, tick.len(), tick[0].kind,
                    tick_started.elapsed().as_secs_f64() * 1e3);
            }
            for op in tick { for id in &op.inputs { terms.remove(id); } }
            for (id, ct) in produced.into_iter().flatten() { terms.insert(id, ct); }
        }
        let folded = Instant::now();
        let blocks: Vec<Ciphertext> = schedule.final_columns.iter().map(|column| {
            if column.is_empty() {
                return shortint.create_trivial(0);
            }
            let mut sum = terms[&column[0].0].clone();
            for (id, _) in &column[1..] { lwe_ciphertext_add_assign(&mut sum, &terms[id]); }
            block(sum, 3 * column.len() as u64, column.len() as u64)
        }).collect();
        let mut product = RadixCiphertext::from(blocks);
        tfhe::shortint::server_key::reset_pbs_count();
        server.full_propagate_parallelized(&mut product);
        let finished = Instant::now();
        let propagation_pbs = tfhe::shortint::server_key::get_pbs_count();
        let seconds = (finished - started).as_secs_f64();
        let clear: U256 = client.decrypt(&product);
        let ok = product.block_carries_are_empty() && clear == (x * y) & mask;
        csv.write_record([format!("bernard-mvb{}", if args.identical { "-identical" } else { "" }),
            args.width.to_string(), args.threads.to_string(), trial.to_string(),
            (trial < args.warmup).to_string(), seconds.to_string(),
            (folded - started).as_secs_f64().to_string(), (finished - folded).as_secs_f64().to_string(),
            brs.to_string(), propagation_pbs.to_string(), ok.to_string()])?;
        csv.flush()?;
        ensure!(ok, "incorrect encrypted product; failed trial retained");
        println!("trial={trial} warmup={} seconds={seconds:.6} brs={brs} prop_pbs={propagation_pbs} ok={ok}",
            trial < args.warmup);
    }
    Ok(())
}
