mod backend;
mod circuits;
mod inputs;

use anyhow::{ensure, Result};
use backend::{Preset, ShortintGates};
use clap::{Parser, ValueEnum};
use inputs::Pattern;
use rayon::prelude::*;
use serde::Serialize;
use serde_json::json;
use std::{
    fs::{self, File},
    path::PathBuf,
    time::Instant,
};
use tfhe::shortint::{
    gen_keys,
    server_key::{get_pbs_count, reset_pbs_count},
    ClientKey,
};

#[derive(Clone, Copy, Debug, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
enum Method {
    Morshed,
    Trifan,
    TrifanPruned,
}
impl Method {
    fn name(self) -> &'static str {
        match self {
            Self::Morshed => "morshed",
            Self::Trifan => "trifan",
            Self::TrifanPruned => "trifan-pruned",
        }
    }
    fn output_width(self, width: usize) -> usize {
        match self {
            Self::Morshed => 2 * width,
            Self::Trifan | Self::TrifanPruned => width,
        }
    }
}

#[derive(Parser, Serialize)]
#[command(about = "Local shortint reconstruction checks, not original-author timings")]
struct Args {
    #[arg(long, value_enum, default_value = "m1c1-gaussian")]
    preset: Preset,
    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        default_value = "morshed,trifan"
    )]
    methods: Vec<Method>,
    #[arg(long, value_delimiter = ',', default_value = "16")]
    widths: Vec<usize>,
    #[arg(long, value_delimiter = ',', default_value = "4")]
    threads: Vec<usize>,
    #[arg(long, value_enum, value_delimiter = ',', default_value = "random")]
    patterns: Vec<Pattern>,
    #[arg(long, default_value_t = 1)]
    repetitions: usize,
    #[arg(long, default_value_t = 0)]
    warmup: usize,
    #[arg(long, default_value_t = 20260908)]
    seed: u64,
    #[arg(long, help = "Use fresh encrypted bits instead of bootstrapped bits")]
    fresh_inputs: bool,
    #[arg(
        long,
        help = "Acknowledge that no whole-multiplier failure bound is certified"
    )]
    allow_unverified_parameters: bool,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Serialize)]
struct Row {
    method: &'static str,
    preset: Preset,
    width: usize,
    output_width: usize,
    threads: usize,
    pattern: Pattern,
    seed: u64,
    trial: usize,
    warmup: bool,
    input_state: &'static str,
    input_pbs: u64,
    logical_lut_calls: u64,
    trivial_lut_calls: u64,
    real_pbs: u64,
    // TFHE-rs pbs-stats counter; it includes trivial LUT evaluations.
    library_pbs_count: u64,
    total_s: f64,
    x_hex: String,
    y_hex: String,
    expected_hex: String,
    output_hex: String,
    ok: bool,
}

fn trial(
    args: &Args,
    ck: &ClientKey,
    gates: &ShortintGates,
    method: Method,
    width: usize,
    pattern: Pattern,
    trial: usize,
) -> Row {
    let threads = rayon::current_num_threads();
    let (x, y) = inputs::operands(width, args.seed, trial, pattern);
    reset_pbs_count();
    let cx: Vec<_> = x
        .par_iter()
        .map(|b| gates.encrypt_bit(ck, *b, !args.fresh_inputs))
        .collect();
    let cy: Vec<_> = y
        .par_iter()
        .map(|b| gates.encrypt_bit(ck, *b, !args.fresh_inputs))
        .collect();
    let input_pbs = get_pbs_count();
    gates.reset_calls();
    reset_pbs_count();
    let start = Instant::now();
    let product = match method {
        Method::Morshed => circuits::morshed(gates, &cx, &cy, threads),
        Method::Trifan => circuits::trifan(gates, &cx, &cy),
        Method::TrifanPruned => circuits::trifan_pruned(gates, &cx, &cy),
    };
    let total_s = start.elapsed().as_secs_f64();
    let library_pbs_count = get_pbs_count();
    let counts = gates.counts();
    let output_width = method.output_width(width);
    let digits: Vec<_> = product
        .iter()
        .map(|b| ck.decrypt_message_and_carry(b))
        .collect();
    let clean = digits.iter().all(|b| *b <= 1);
    let bits: Vec<_> = digits.iter().map(|b| *b as u8).collect();
    let output = inputs::value(&bits);
    let expected = inputs::expected(&x, &y, output_width);
    let counts_match = counts.logical == circuits::logical_pbs(method.name(), width, threads)
        && counts.trivial + counts.real == counts.logical
        && counts.real > 0
        && library_pbs_count == counts.logical;
    let ok = clean && output == expected && product.len() == output_width && counts_match;
    Row {
        method: method.name(),
        preset: args.preset,
        width,
        output_width,
        threads,
        pattern,
        seed: args.seed,
        trial,
        warmup: trial < args.warmup,
        input_state: if args.fresh_inputs {
            "fresh"
        } else {
            "bootstrapped"
        },
        input_pbs,
        logical_lut_calls: counts.logical,
        trivial_lut_calls: counts.trivial,
        real_pbs: counts.real,
        library_pbs_count,
        total_s,
        x_hex: inputs::value(&x).to_str_radix(16),
        y_hex: inputs::value(&y).to_str_radix(16),
        expected_hex: expected.to_str_radix(16),
        output_hex: output.to_str_radix(16),
        ok,
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.allow_unverified_parameters, "Pass --allow-unverified-parameters for diagnostic checks; whole-multiplier analysis remains open.");
    ensure!(args.repetitions > 0 && args.warmup.checked_add(args.repetitions).is_some());
    ensure!(
        args.widths.iter().all(|w| (1..=256).contains(w)),
        "width must be 1..256"
    );
    ensure!(
        args.threads.iter().all(|t| (1..=64).contains(t)),
        "threads must be 1..64"
    );
    fs::create_dir(&args.output)?;
    let (constant, parameters) = args.preset.parameters();
    let metadata = json!({
        "arguments": args, "tfhe_rs": "1.7.0", "parameter_constant": constant,
        "parameters": parameters, "primitive_log2_p_fail": parameters.log2_p_fail,
        "failure_unit": "library primitive; not re-derived for this multiplier",
        "whole_multiplier_approved": false,
        "timing_scope": "native bit-array multiplication; keygen, LUT setup and input preparation excluded",
        "implementation": "our shortint circuit port; not the author binary",
        "os": std::env::consts::OS, "arch": std::env::consts::ARCH,
    });
    serde_json::to_writer_pretty(
        File::create(args.output.join("parameters.json"))?,
        &metadata,
    )?;
    eprintln!("Generating keys: {constant}");
    let (ck, sk) = gen_keys(parameters);
    let gates = ShortintGates::new(sk);
    let preflight_pool = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    preflight_pool.install(|| gates.preflight(&ck))?;
    fs::write(
        args.output.join("preflight.json"),
        "{\"encrypted_full_adder_triples\":8,\"both_adders_correct\":true}\n",
    )?;
    let mut csv = csv::Writer::from_path(args.output.join("timings.csv"))?;
    for &threads in &args.threads {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()?;
        for &width in &args.widths {
            for &pattern in &args.patterns {
                for &method in &args.methods {
                    for repetition in 0..args.warmup + args.repetitions {
                        let row = pool.install(|| {
                            trial(&args, &ck, &gates, method, width, pattern, repetition)
                        });
                        csv.serialize(&row)?;
                        csv.flush()?;
                        println!(
                            "{} W={} T={} {:?} trial={} {:.6}s real_PBS={} trivial={} logical={} correct={}",
                            row.method,
                            width,
                            threads,
                            pattern,
                            repetition,
                            row.total_s,
                            row.real_pbs,
                            row.trivial_lut_calls,
                            row.logical_lut_calls,
                            row.ok
                        );
                        ensure!(
                            row.ok,
                            "Incorrect product/count; failed trial retained in timings.csv"
                        );
                    }
                }
            }
        }
    }
    Ok(())
}
