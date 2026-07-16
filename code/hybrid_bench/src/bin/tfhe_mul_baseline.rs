use anyhow::{bail, Context, Result};
use clap::Parser;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;
use tfhe::integer::{gen_keys_radix, U256};
use tfhe::shortint::parameters::current_params::{
    V1_6_PARAM_MESSAGE_1_CARRY_1_KS_PBS_GAUSSIAN_2M128,
    V1_6_PARAM_MESSAGE_1_CARRY_1_KS_PBS_TUNIFORM_2M128,
    V1_6_PARAM_MESSAGE_1_CARRY_2_KS_PBS_GAUSSIAN_2M128,
    V1_6_PARAM_MESSAGE_1_CARRY_3_KS_PBS_GAUSSIAN_2M128,
    V1_6_PARAM_MESSAGE_1_CARRY_4_KS_PBS_GAUSSIAN_2M128,
    V1_6_PARAM_MESSAGE_2_CARRY_3_KS_PBS_GAUSSIAN_2M128,
    V1_6_PARAM_MESSAGE_2_CARRY_4_KS_PBS_GAUSSIAN_2M128,
    V1_6_PARAM_MESSAGE_3_CARRY_3_KS_PBS_TUNIFORM_2M128,
    V1_6_PARAM_MESSAGE_3_CARRY_4_KS_PBS_GAUSSIAN_2M128,
    V1_6_PARAM_MESSAGE_4_CARRY_4_KS_PBS_GAUSSIAN_2M128,
    V1_6_PARAM_MESSAGE_4_CARRY_4_KS_PBS_TUNIFORM_2M128,
};
use tfhe::shortint::parameters::{
    ClassicPBSParameters, PARAM_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M128,
    PARAM_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128, PARAM_MESSAGE_3_CARRY_3_KS_PBS_GAUSSIAN_2M128,
};
use tfhe::{get_pbs_count, reset_pbs_count};

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Single-multiplication TFHE-rs radix baseline"
)]
struct Cli {
    #[arg(long, default_value = "m2c2-gaussian")]
    params: String,

    #[arg(long, default_value = "16,32")]
    widths: String,

    #[arg(long, default_value = "mul-default")]
    ops: String,

    #[arg(long, default_value_t = 3)]
    reps: usize,

    #[arg(long, default_value_t = 1)]
    warmups: usize,

    #[arg(long)]
    threads: Option<usize>,

    #[arg(long, default_value_t = 0x5446_4845)]
    seed: u64,

    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug)]
struct ParamSpec {
    name: &'static str,
    message_bits: u32,
    carry_bits: u32,
    params: ClassicPBSParameters,
}

#[derive(Debug, Serialize)]
struct BenchRow {
    tfhe_version: &'static str,
    param: &'static str,
    message_bits: u32,
    carry_bits: u32,
    width_bits: usize,
    capacity_bits: usize,
    radix_blocks: usize,
    rayon_threads: usize,
    seed: u64,
    op: &'static str,
    rep: usize,
    chain_len: usize,
    add_terms: usize,
    reducer_degree_capacity: u64,
    elapsed_ns: u128,
    pbs_count: u64,
    phase_timer_kind: &'static str,
    product_elapsed_ns: Option<u128>,
    normalization_elapsed_ns: Option<u128>,
    product_pbs_model: u64,
    normalization_pbs_model: u64,
    term_pbs: u64,
    reduction_pbs: u64,
    final_propagate_pbs: u64,
    lazy_terms_total: u64,
    peak_lazy_terms: u64,
    peak_column_height: u64,
    clear_lhs: u64,
    clear_rhs: u64,
    clear_scalar: u64,
    dirty_expected: u64,
    decrypted: u64,
    expected: u64,
    clear_lhs_hex: String,
    clear_rhs_hex: String,
    dirty_expected_hex: String,
    decrypted_hex: String,
    expected_hex: String,
    ok: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if let Some(threads) = cli.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .context("failed to configure Rayon global thread pool")?;
    }

    let ops = parse_csv(&cli.ops);
    if ops.iter().any(|op| op != "mul-default") {
        bail!("tfhe_mul_baseline only supports --ops mul-default");
    }

    let params = parse_csv(&cli.params)
        .into_iter()
        .map(|name| parse_param(&name))
        .collect::<Result<Vec<_>>>()?;
    let widths = parse_csv(&cli.widths)
        .into_iter()
        .map(|value| parse_width(&value))
        .collect::<Result<Vec<_>>>()?;

    let mut writer = make_row_writer(cli.out.as_ref())?;
    for param in params {
        for width_bits in widths.iter().copied() {
            run_mul_suite(&cli, param, width_bits, &mut writer)?;
        }
    }
    writer.flush()?;
    Ok(())
}

fn run_mul_suite(
    cli: &Cli,
    param: ParamSpec,
    width_bits: usize,
    writer: &mut csv::Writer<Box<dyn Write>>,
) -> Result<()> {
    let radix_blocks = width_bits.div_ceil(param.message_bits as usize);
    let capacity_bits = radix_blocks * param.message_bits as usize;
    if capacity_bits > U256::BITS as usize {
        bail!(
            "width={width_bits} with param={} needs {capacity_bits} radix capacity bits; verifier supports at most {} bits",
            param.name,
            U256::BITS
        );
    }

    let mask = mask_for_bits(capacity_bits);
    let mut rng = SmallRng::seed_from_u64(cli.seed ^ width_bits as u64 ^ param.message_bits as u64);
    let (clear_lhs, mut clear_rhs) = operands_for_mask(&mut rng, mask);
    if clear_rhs == U256::ZERO {
        clear_rhs = U256::ONE;
    }
    let expected = (clear_lhs * clear_rhs) & mask;

    eprintln!(
        "keygen baseline param={} width={} blocks={} threads={}",
        param.name,
        width_bits,
        radix_blocks,
        rayon::current_num_threads()
    );
    let keygen_start = Instant::now();
    let (cks, sks) = gen_keys_radix(param.params, radix_blocks);
    eprintln!("keygen_done elapsed={:?}", keygen_start.elapsed());

    let lhs = cks.encrypt(clear_lhs);
    let rhs = cks.encrypt(clear_rhs);
    for _ in 0..cli.warmups {
        let product = sks.mul_parallelized(&lhs, &rhs);
        let _: U256 = cks.decrypt(&product);
    }

    for rep in 0..cli.reps {
        reset_pbs_count();
        let start = Instant::now();
        let product = sks.mul_parallelized(&lhs, &rhs);
        let elapsed_ns = start.elapsed().as_nanos();
        let pbs_count = get_pbs_count();
        let product_pbs = (radix_blocks * radix_blocks) as u64;
        let normalization_pbs = pbs_count.saturating_sub(product_pbs);
        let decrypted: U256 = cks.decrypt(&product);
        let ok = decrypted == expected;
        let row = BenchRow {
            tfhe_version: "tfhe-1.6.1",
            param: param.name,
            message_bits: param.message_bits,
            carry_bits: param.carry_bits,
            width_bits,
            capacity_bits,
            radix_blocks,
            rayon_threads: rayon::current_num_threads(),
            seed: cli.seed,
            op: "mul-default",
            rep,
            chain_len: 1,
            add_terms: 0,
            reducer_degree_capacity: 0,
            elapsed_ns,
            pbs_count,
            phase_timer_kind: "black_box_total_only",
            product_elapsed_ns: None,
            normalization_elapsed_ns: None,
            product_pbs_model: product_pbs,
            normalization_pbs_model: normalization_pbs,
            term_pbs: product_pbs,
            reduction_pbs: normalization_pbs,
            final_propagate_pbs: 0,
            lazy_terms_total: 0,
            peak_lazy_terms: 0,
            peak_column_height: 0,
            clear_lhs: u256_low_u64(clear_lhs),
            clear_rhs: u256_low_u64(clear_rhs),
            clear_scalar: 0,
            dirty_expected: 0,
            decrypted: u256_low_u64(decrypted),
            expected: u256_low_u64(expected),
            clear_lhs_hex: u256_hex(clear_lhs, capacity_bits),
            clear_rhs_hex: u256_hex(clear_rhs, capacity_bits),
            dirty_expected_hex: "0x0".to_string(),
            decrypted_hex: u256_hex(decrypted, capacity_bits),
            expected_hex: u256_hex(expected, capacity_bits),
            ok,
        };
        writer.serialize(&row)?;
        writer.flush()?;
        eprintln!(
            "baseline_row_done param={} width={} threads={} rep={} elapsed_ms={} pbs={} ok={}",
            param.name,
            width_bits,
            rayon::current_num_threads(),
            rep,
            elapsed_ns / 1_000_000,
            pbs_count,
            ok
        );
    }
    Ok(())
}

fn parse_csv(input: &str) -> Vec<String> {
    input
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn parse_param(name: &str) -> Result<ParamSpec> {
    match name {
        "m1c1-tuniform" => Ok(ParamSpec {
            name: "m1c1-tuniform",
            message_bits: 1,
            carry_bits: 1,
            params: V1_6_PARAM_MESSAGE_1_CARRY_1_KS_PBS_TUNIFORM_2M128,
        }),
        "m2c2" | "m2c2-tuniform" => Ok(ParamSpec {
            name: "m2c2-tuniform",
            message_bits: 2,
            carry_bits: 2,
            params: PARAM_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128,
        }),
        "m3c3-tuniform" => Ok(ParamSpec {
            name: "m3c3-tuniform",
            message_bits: 3,
            carry_bits: 3,
            params: V1_6_PARAM_MESSAGE_3_CARRY_3_KS_PBS_TUNIFORM_2M128,
        }),
        "m4c4-tuniform" => Ok(ParamSpec {
            name: "m4c4-tuniform",
            message_bits: 4,
            carry_bits: 4,
            params: V1_6_PARAM_MESSAGE_4_CARRY_4_KS_PBS_TUNIFORM_2M128,
        }),
        "m1c1-gaussian" => Ok(ParamSpec {
            name: "m1c1-gaussian",
            message_bits: 1,
            carry_bits: 1,
            params: V1_6_PARAM_MESSAGE_1_CARRY_1_KS_PBS_GAUSSIAN_2M128,
        }),
        "m1c2-gaussian" => Ok(ParamSpec {
            name: "m1c2-gaussian",
            message_bits: 1,
            carry_bits: 2,
            params: V1_6_PARAM_MESSAGE_1_CARRY_2_KS_PBS_GAUSSIAN_2M128,
        }),
        "m1c3-gaussian" => Ok(ParamSpec {
            name: "m1c3-gaussian",
            message_bits: 1,
            carry_bits: 3,
            params: V1_6_PARAM_MESSAGE_1_CARRY_3_KS_PBS_GAUSSIAN_2M128,
        }),
        "m1c4-gaussian" => Ok(ParamSpec {
            name: "m1c4-gaussian",
            message_bits: 1,
            carry_bits: 4,
            params: V1_6_PARAM_MESSAGE_1_CARRY_4_KS_PBS_GAUSSIAN_2M128,
        }),
        "m2c2-gaussian" => Ok(ParamSpec {
            name: "m2c2-gaussian",
            message_bits: 2,
            carry_bits: 2,
            params: PARAM_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M128,
        }),
        "m2c3-gaussian" => Ok(ParamSpec {
            name: "m2c3-gaussian",
            message_bits: 2,
            carry_bits: 3,
            params: V1_6_PARAM_MESSAGE_2_CARRY_3_KS_PBS_GAUSSIAN_2M128,
        }),
        "m2c4-gaussian" => Ok(ParamSpec {
            name: "m2c4-gaussian",
            message_bits: 2,
            carry_bits: 4,
            params: V1_6_PARAM_MESSAGE_2_CARRY_4_KS_PBS_GAUSSIAN_2M128,
        }),
        "m3c3-gaussian" => Ok(ParamSpec {
            name: "m3c3-gaussian",
            message_bits: 3,
            carry_bits: 3,
            params: PARAM_MESSAGE_3_CARRY_3_KS_PBS_GAUSSIAN_2M128,
        }),
        "m3c4-gaussian" => Ok(ParamSpec {
            name: "m3c4-gaussian",
            message_bits: 3,
            carry_bits: 4,
            params: V1_6_PARAM_MESSAGE_3_CARRY_4_KS_PBS_GAUSSIAN_2M128,
        }),
        "m4c4-gaussian" => Ok(ParamSpec {
            name: "m4c4-gaussian",
            message_bits: 4,
            carry_bits: 4,
            params: V1_6_PARAM_MESSAGE_4_CARRY_4_KS_PBS_GAUSSIAN_2M128,
        }),
        other => bail!("unknown parameter alias '{other}'"),
    }
}

fn parse_width(input: &str) -> Result<usize> {
    let width = input
        .parse::<usize>()
        .with_context(|| format!("invalid width '{input}'"))?;
    if width == 0 || width > U256::BITS as usize {
        bail!("width must be in 1..={}, got {width}", U256::BITS);
    }
    Ok(width)
}

fn operands_for_mask(rng: &mut SmallRng, mask: U256) -> (U256, U256) {
    (random_u256(rng) & mask, random_u256(rng) & mask)
}

fn mask_for_bits(bits: usize) -> U256 {
    if bits == 0 {
        U256::ZERO
    } else if bits >= U256::BITS as usize {
        U256::MAX
    } else {
        (U256::ONE << bits) - U256::ONE
    }
}

fn random_u256(rng: &mut SmallRng) -> U256 {
    U256::from([
        rng.gen::<u64>(),
        rng.gen::<u64>(),
        rng.gen::<u64>(),
        rng.gen::<u64>(),
    ])
}

fn u256_low_u64(value: U256) -> u64 {
    let mut bytes = [0_u8; 32];
    value.copy_to_le_byte_slice(&mut bytes);
    u64::from_le_bytes(bytes[..8].try_into().expect("low word slice has length 8"))
}

fn u256_hex(value: U256, bits: usize) -> String {
    let byte_len = bits.div_ceil(8).clamp(1, 32);
    let mut bytes = [0_u8; 32];
    value.copy_to_be_byte_slice(&mut bytes);
    let mut out = String::with_capacity(2 + 2 * byte_len);
    out.push_str("0x");
    for byte in &bytes[32 - byte_len..] {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to a String cannot fail");
    }
    out
}

fn make_row_writer(path: Option<&PathBuf>) -> Result<csv::Writer<Box<dyn Write>>> {
    match path {
        Some(path) => {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("failed to create {}", parent.display()))?;
            }
            let file = std::fs::File::create(path)
                .with_context(|| format!("failed to open {}", path.display()))?;
            Ok(csv::Writer::from_writer(Box::new(file)))
        }
        None => Ok(csv::Writer::from_writer(Box::new(std::io::stdout()))),
    }
}
