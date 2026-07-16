use super::{
    evaluator::cxc_mul,
    types::{delta, EvalContext, Limb, Lwe, Stats, KAPPA, LIMB_BITS, LIMB_CHUNKS},
};
use rand::Rng;
use refined_tfhe_lhe::{
    gen_all_auto_keys, generate_scheme_switching_key, get_val_and_abs_err,
    int_lhe_instance::SHOKRI_TSOUTSOS_TABLE1, keygen_pbs,
};
use std::{env, time::Instant};
use tfhe::core_crypto::prelude::*;

pub(crate) fn run() {
    if env::args().nth(1).as_deref() == Some("--parameters") {
        print_parameters();
        return;
    }
    let (width_bits, x_arg, y_arg) = parse_args();
    let limbs = width_bits / LIMB_BITS;
    let parallel = env_bool("ST_PARALLEL", false);
    let mut clear_rng = rand::rng();
    let x_limbs = x_arg.unwrap_or_else(|| random_limbs(limbs, &mut clear_rng));
    let y_limbs = y_arg.unwrap_or_else(|| random_limbs(limbs, &mut clear_rng));
    let expected_limbs = mul_limbs_mod(&x_limbs, &y_limbs, limbs);

    let param = *SHOKRI_TSOUTSOS_TABLE1;
    let mut boxed_seeder = new_seeder();
    let seeder = boxed_seeder.as_mut();
    let mut secret_generator =
        SecretRandomGenerator::<ActivatedRandomGenerator>::new(seeder.seed());
    let mut encryption_generator =
        EncryptionRandomGenerator::<ActivatedRandomGenerator>::new(seeder.seed(), seeder);

    let (lwe_sk, glwe_sk, lwe_sk_after_ks, bsk, _ksk) = keygen_pbs(
        param.lwe_dimension(),
        param.glwe_dimension(),
        param.polynomial_size(),
        param.lwe_modular_std_dev(),
        param.glwe_modular_std_dev(),
        param.pbs_base_log(),
        param.pbs_level(),
        param.ks_base_log(),
        param.ks_level(),
        &mut secret_generator,
        &mut encryption_generator,
    );

    let ksk = allocate_and_generate_new_lwe_keyswitch_key(
        &lwe_sk,
        &lwe_sk_after_ks,
        param.ks_base_log(),
        param.ks_level(),
        param.lwe_modular_std_dev(),
        param.ciphertext_modulus(),
        &mut encryption_generator,
    );

    let auto_keys = gen_all_auto_keys(
        param.auto_base_log(),
        param.auto_level(),
        param.fft_type_auto(),
        &glwe_sk,
        param.glwe_modular_std_dev(),
        &mut encryption_generator,
    );
    let ss_key = generate_scheme_switching_key(
        &glwe_sk,
        param.ss_base_log(),
        param.ss_level(),
        param.glwe_modular_std_dev(),
        param.ciphertext_modulus(),
        &mut encryption_generator,
    );

    let ctx = EvalContext {
        ksk: &ksk,
        bsk: bsk.as_view(),
        auto_keys: &auto_keys,
        ss_key: ss_key.as_view(),
        output_lwe_size: lwe_sk.lwe_dimension().to_lwe_size(),
        glwe_size: param.glwe_dimension().to_glwe_size(),
        polynomial_size: param.polynomial_size(),
        cbs_base_log: param.cbs_base_log(),
        cbs_level: param.cbs_level(),
        log_lut_count: param.log_lut_count(),
        ciphertext_modulus: param.ciphertext_modulus(),
    };

    let x_blocks = encrypt_integer(
        &x_limbs,
        &lwe_sk,
        param.lwe_modular_std_dev(),
        param.ciphertext_modulus(),
        &mut encryption_generator,
    );
    let y_blocks = encrypt_integer(
        &y_limbs,
        &lwe_sk,
        param.lwe_modular_std_dev(),
        param.ciphertext_modulus(),
        &mut encryption_generator,
    );
    let zero_block = encrypt_limb(
        0,
        &lwe_sk,
        param.lwe_modular_std_dev(),
        param.ciphertext_modulus(),
        &mut encryption_generator,
    );
    let zero_carry = encrypt_lwe_value(
        0,
        1,
        &lwe_sk,
        param.lwe_modular_std_dev(),
        param.ciphertext_modulus(),
        &mut encryption_generator,
    );

    let mut stats = Stats::default();
    let start = Instant::now();
    let result_blocks = cxc_mul(
        &ctx,
        &x_blocks,
        &y_blocks,
        &zero_block,
        &zero_carry,
        parallel,
        &mut stats,
    );
    let elapsed = start.elapsed();

    let got_limbs = decrypt_integer(&result_blocks, &lwe_sk);
    let x_hex = format_limbs_hex(&x_limbs, width_bits);
    let y_hex = format_limbs_hex(&y_limbs, width_bits);
    let expected_hex = format_limbs_hex(&expected_limbs, width_bits);
    let got_hex = format_limbs_hex(&got_limbs, width_bits);
    println!("width={width_bits}");
    println!("x=0x{x_hex}, y=0x{y_hex}");
    println!("expected=0x{expected_hex}, got=0x{got_hex}");
    println!("elapsed={elapsed:?}");
    stats.print(elapsed);
    assert_eq!(got_limbs, expected_limbs);
}

fn print_parameters() {
    let param = *SHOKRI_TSOUTSOS_TABLE1;
    println!(
        "parameter=SHOKRI_TSOUTSOS_TABLE1 lwe_dimension={} glwe_dimension={} polynomial_size={} \
         pbs=({},{}) key_switch=({},{}) automorphism=({},{}) scheme_switch=({},{}) \
         circuit_bootstrap=({},{}) log_lut_count={} fft_type={:?} centered_ms=true vartheta=1 tau=2 \
         reported_log2_p_fail=-157.25 lwe_std_dev={:.17e} glwe_std_dev={:.17e}",
        param.lwe_dimension().0,
        param.glwe_dimension().0,
        param.polynomial_size().0,
        param.pbs_base_log().0,
        param.pbs_level().0,
        param.ks_base_log().0,
        param.ks_level().0,
        param.auto_base_log().0,
        param.auto_level().0,
        param.ss_base_log().0,
        param.ss_level().0,
        param.cbs_base_log().0,
        param.cbs_level().0,
        param.log_lut_count().0,
        param.fft_type_auto(),
        param.lwe_modular_std_dev().0,
        param.glwe_modular_std_dev().0,
    );
}

fn env_bool(name: &str, default: bool) -> bool {
    match env::var(name) {
        Ok(value) => matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"),
        Err(_) => default,
    }
}

fn parse_args() -> (usize, Option<Vec<usize>>, Option<Vec<usize>>) {
    let mut args = env::args().skip(1);
    let width = args
        .next()
        .map(|arg| arg.parse::<usize>().expect("width must be an integer"))
        .unwrap_or(32);
    assert!(width > 0, "width must be positive");
    assert_eq!(
        width % LIMB_BITS,
        0,
        "width must be a multiple of {LIMB_BITS}"
    );
    assert!(width <= 256, "supported widths do not exceed 256 bits");
    let limbs = width / LIMB_BITS;
    let x = args.next().map(|arg| parse_clear_arg(&arg, limbs));
    let y = args.next().map(|arg| parse_clear_arg(&arg, limbs));
    assert!(
        args.next().is_none(),
        "usage: shokri_tsoutsos_cxc [width] [x] [y]"
    );
    (width, x, y)
}

fn parse_clear_arg(arg: &str, limbs: usize) -> Vec<usize> {
    if let Some(raw) = arg.strip_prefix("0x").or_else(|| arg.strip_prefix("0X")) {
        parse_hex_limbs(raw, limbs)
    } else {
        parse_decimal_limbs(arg, limbs)
    }
}

fn parse_hex_limbs(raw: &str, limbs: usize) -> Vec<usize> {
    assert!(
        raw.chars().all(|ch| ch.is_ascii_hexdigit()),
        "hex integer arguments must contain only hex digits"
    );
    let digits = raw.trim_start_matches('0');
    let digits = if digits.is_empty() { "0" } else { digits };
    let mut out = vec![0usize; limbs];
    let mut end = digits.len();
    for limb in out.iter_mut() {
        if end == 0 {
            break;
        }
        let start = end.saturating_sub(4);
        *limb = usize::from_str_radix(&digits[start..end], 16)
            .expect("hex integer arguments must contain only hex digits");
        end = start;
    }
    assert!(
        digits[..end].chars().all(|ch| ch == '0'),
        "hex integer does not fit in the selected width"
    );
    out
}

fn random_limbs(limbs: usize, rng: &mut impl Rng) -> Vec<usize> {
    (0..limbs)
        .map(|_| rng.random_range(0..(1usize << LIMB_BITS)))
        .collect()
}

fn parse_decimal_limbs(raw: &str, limbs: usize) -> Vec<usize> {
    assert!(
        !raw.is_empty(),
        "decimal integer argument must not be empty"
    );
    assert!(
        raw.chars().all(|ch| ch.is_ascii_digit()),
        "decimal integer arguments must contain only decimal digits; use 0x... for hexadecimal"
    );
    let mut out = vec![0usize; limbs];
    for ch in raw.chars() {
        let digit = ch.to_digit(10).unwrap() as u32;
        let mut carry = digit;
        for limb in &mut out {
            let value = (*limb as u32) * 10 + carry;
            *limb = (value & 0xffff) as usize;
            carry = value >> LIMB_BITS;
        }
        assert!(
            carry == 0,
            "decimal integer does not fit in the selected width"
        );
    }
    out
}

fn mul_limbs_mod(lhs: &[usize], rhs: &[usize], limbs: usize) -> Vec<usize> {
    assert!(lhs.len() >= limbs && rhs.len() >= limbs);
    let mut acc = vec![0u128; limbs + 1];
    for (i, &x) in lhs.iter().enumerate().take(limbs) {
        for (j, &y) in rhs.iter().enumerate().take(limbs) {
            let q = i + j;
            if q < limbs {
                acc[q] += ((x & 0xffff) as u128) * ((y & 0xffff) as u128);
            }
        }
    }
    let base = 1u128 << LIMB_BITS;
    for q in 0..limbs {
        let carry = acc[q] >> LIMB_BITS;
        acc[q] &= base - 1;
        acc[q + 1] += carry;
    }
    acc.into_iter()
        .take(limbs)
        .map(|value| value as usize)
        .collect()
}

fn format_limbs_hex(limbs: &[usize], width_bits: usize) -> String {
    let limb_count = width_bits / LIMB_BITS;
    let mut out = String::with_capacity(width_bits / 4);
    for idx in (0..limb_count).rev() {
        out.push_str(&format!(
            "{:04x}",
            limbs.get(idx).copied().unwrap_or(0) & 0xffff
        ));
    }
    out
}

fn encrypt_limb<G: ByteRandomGenerator>(
    value: usize,
    lwe_sk: &LweSecretKeyOwned<u64>,
    noise: impl DispersionParameter + Copy,
    ciphertext_modulus: CiphertextModulus<u64>,
    generator: &mut EncryptionRandomGenerator<G>,
) -> Limb {
    (0..LIMB_CHUNKS)
        .map(|chunk| {
            encrypt_lwe_value(
                (value >> (KAPPA * chunk)) & 0b11,
                KAPPA,
                lwe_sk,
                noise,
                ciphertext_modulus,
                generator,
            )
        })
        .collect()
}

fn encrypt_integer<G: ByteRandomGenerator>(
    limbs: &[usize],
    lwe_sk: &LweSecretKeyOwned<u64>,
    noise: impl DispersionParameter + Copy,
    ciphertext_modulus: CiphertextModulus<u64>,
    generator: &mut EncryptionRandomGenerator<G>,
) -> Vec<Limb> {
    limbs
        .iter()
        .copied()
        .map(|limb| encrypt_limb(limb, lwe_sk, noise, ciphertext_modulus, generator))
        .collect()
}

fn encrypt_lwe_value<G: ByteRandomGenerator>(
    value: usize,
    bits: usize,
    lwe_sk: &LweSecretKeyOwned<u64>,
    noise: impl DispersionParameter,
    ciphertext_modulus: CiphertextModulus<u64>,
    generator: &mut EncryptionRandomGenerator<G>,
) -> Lwe {
    allocate_and_encrypt_new_lwe_ciphertext(
        lwe_sk,
        Plaintext((value as u64) * delta(bits)),
        noise,
        ciphertext_modulus,
        generator,
    )
}

fn decrypt_limb(block: &Limb, lwe_sk: &LweSecretKeyOwned<u64>) -> usize {
    block
        .iter()
        .enumerate()
        .map(|(chunk, lwe)| {
            let (value, _err) = get_val_and_abs_err(lwe_sk, lwe, 0u64, delta(KAPPA));
            ((value as usize) & 0b11) << (KAPPA * chunk)
        })
        .sum()
}

fn decrypt_integer(blocks: &[Limb], lwe_sk: &LweSecretKeyOwned<u64>) -> Vec<usize> {
    blocks
        .iter()
        .map(|block| decrypt_limb(block, lwe_sk))
        .collect()
}
