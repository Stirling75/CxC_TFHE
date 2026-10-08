use parmesan::cloudovo::multiplication::mul_impl;
use parmesan::params::PAR_TFHE_V0_5__M4_C0;
use parmesan::userovo::encryption::{parm_decrypt_to_vec, parm_encrypt_from_vec};
use parmesan::{ParmesanCloudovo, PrivKeySet, PubKeySet};
use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

fn decode(digits: &[i32]) -> Result<i128, Box<dyn Error>> {
    if digits.len() > 126 || digits.iter().any(|v| !(-1..=1).contains(v)) {
        return Err("invalid or oversized redundant-binary output".into());
    }
    Ok(digits.iter().enumerate().map(|(i, d)| i128::from(*d) * (1i128 << i)).sum())
}

// Signed-binary digits, least significant first: '-' = -1, '0' = 0, '+' = +1.
// Any out-of-range digit is written as '?' so the checker rejects it.
fn compact(digits: &[i32]) -> String {
    digits
        .iter()
        .map(|d| match d {
            -1 => '-',
            0 => '0',
            1 => '+',
            _ => '?',
        })
        .collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if !(6..=7).contains(&args.len()) {
        return Err("usage: parmesan-benchmark WIDTH RAYON_THREADS REPS WARMUP OUTPUT_DIRECTORY [SEED]".into());
    }
    let width: usize = args[1].parse()?;
    let threads: usize = args[2].parse()?;
    let repetitions: usize = args[3].parse()?;
    let warmup: usize = args[4].parse()?;
    let seed: u64 = args.get(6).map(|s| s.parse()).transpose()?.unwrap_or(20260907);
    if ![16, 32].contains(&width) || threads == 0 || repetitions == 0 {
        return Err("supported widths are 16 and 32; positive thread/repetition counts required".into());
    }
    let output = Path::new(&args[5]);
    fs::create_dir(output)?;
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global()?;
    let params = &PAR_TFHE_V0_5__M4_C0;
    let (client_key, server_key) = tfhe::shortint::gen_keys(params.concrete_pars);
    let private = PrivKeySet { client_key, server_key };
    let public = PubKeySet { server_key: &private.server_key };
    let cloud = ParmesanCloudovo::new(params, &public);
    fs::write(output.join("parameters.json"), serde_json::to_vec_pretty(&serde_json::json!({
        "parameters": params.concrete_pars, "tfhe_rs": "0.5.4",
        "width": width, "threads": threads, "seed": seed,
        "repetitions": repetitions, "warmup": warmup,
        "whole_multiplier_log2_failure": null
    }))?)?;
    fs::write(output.join("parameters.txt"), format!(
        "upstream=a5254f8ccb38d52ebb993d2d8c3c96f2930e9c20\ntfhe=0.5.4\nparams={params:?}\ninput=unsigned binary digits\noutput=full redundant-binary product\noutput_digits_lsb_first=-:-1,0:0,+:+1\noperands=splitmix(seed^(2*trial)), splitmix(seed^(2*trial+1)), masked to width\nkey_generation=fresh; excluded from timing\nseed={seed}\nrayon_threads={threads}\nscoped_os_threads=additional; not bounded by Rayon\nwhole_multiplier_failure=not derived\n"))?;
    let mut csv = OpenOptions::new().write(true).create_new(true).open(output.join("timings.csv"))?;
    writeln!(csv, "method,width,rayon_threads,trial,warmup,x,y,total_seconds,output_digits,ok,output_value,output_digits_lsb_first")?;
    let mask = (1u64 << width) - 1;
    for trial in 0..warmup + repetitions {
        let x = splitmix(seed ^ (trial as u64).wrapping_mul(2)) & mask;
        let y = splitmix(seed ^ (trial as u64).wrapping_mul(2).wrapping_add(1)) & mask;
        let bits = |v: u64| (0..width).map(|i| ((v >> i) & 1) as i32).collect();
        let lhs = parm_encrypt_from_vec(&private, &bits(x))?;
        let rhs = parm_encrypt_from_vec(&private, &bits(y))?;
        let start = Instant::now();
        let product = mul_impl(&cloud, &lhs, &rhs)?;
        let seconds = start.elapsed().as_secs_f64();
        let digits = parm_decrypt_to_vec(&private, &product)?;
        let decoded = decode(&digits);
        let ok = decoded.as_ref().is_ok_and(|value| *value == i128::from(x) * i128::from(y));
        let value = decoded.as_ref().map(ToString::to_string).unwrap_or_default();
        writeln!(csv, "parmesan-native,{width},{threads},{trial},{},{x},{y},{seconds:.9},{},{ok},{value},{}",
                 trial < warmup, digits.len(), compact(&digits))?;
        csv.flush()?;
        if !ok { return Err("incorrect encrypted full product".into()); }
        println!("width={width} trial={trial} seconds={seconds:.6} correct={ok}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn wide_signed_digits_are_not_truncated_to_i64() {
        let mut digits = vec![0; 65];
        digits[64] = 1;
        digits[0] = -1;
        assert_eq!(decode(&digits).unwrap(), (1i128 << 64) - 1);
        assert!(decode(&[2]).is_err());
    }

    #[test]
    fn compact_digits_are_lsb_first() {
        assert_eq!(compact(&[-1, 0, 1, 2]), "-0+?");
    }

    // Pinned vectors shared with parmesan_check.py (test_parmesan_check.py).
    #[test]
    fn operand_derivation_is_pinned() {
        assert_eq!(splitmix(0), 0xe220a8397b1dcdaf);
        assert_eq!(splitmix(20260907), 0xad2ea8a771202a78);
        assert_eq!(splitmix(20260907 ^ 1), 0x2b7e1f89419061d4);
    }
}
