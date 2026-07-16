use super::types::{
    delta, EvalContext, Limb, Lwe, HALF_BITS, HALF_CHUNKS, KAPPA, LIMB_BITS, LIMB_CHUNKS,
    VP16_INPUT_BITS, VP17_INPUT_BITS,
};
#[cfg(feature = "multithread")]
use rayon::prelude::*;
use refined_tfhe_lhe::{blind_rotate_for_msb, convert_to_ggsw_after_blind_rotate_revtrace};
use std::collections::HashMap;
use tfhe::core_crypto::{
    fft_impl::fft64::{
        c64,
        crypto::wop_pbs::{vertical_packing, vertical_packing_scratch},
    },
    prelude::*,
};

#[derive(Default)]
pub(crate) struct LutCache {
    mul8x8: HashMap<usize, PolynomialListOwned<u64>>,
    add_sum_half: HashMap<usize, PolynomialListOwned<u64>>,
    add_carry: Option<PolynomialListOwned<u64>>,
    add_sum_half_no_carry: HashMap<usize, PolynomialListOwned<u64>>,
    add_carry_no_carry: Option<PolynomialListOwned<u64>>,
}

pub(crate) fn combine_half_bits(
    ctx: &EvalContext<'_>,
    x_bits: &GgswCiphertextListOwned<u64>,
    x_high: bool,
    y_bits: &GgswCiphertextListOwned<u64>,
    y_high: bool,
) -> FourierGgswCiphertextList<Vec<c64>> {
    assert_eq!(x_bits.ggsw_ciphertext_count().0, LIMB_BITS);
    assert_eq!(y_bits.ggsw_ciphertext_count().0, LIMB_BITS);
    let mut pair_bits = new_ggsw_list(ctx, VP16_INPUT_BITS);
    copy_half_bits(&mut pair_bits, 0, x_bits, x_high);
    copy_half_bits(&mut pair_bits, HALF_BITS, y_bits, y_high);
    fourier_from_ggsw(ctx, &pair_bits)
}

fn copy_half_bits(
    dst: &mut GgswCiphertextListOwned<u64>,
    dst_base: usize,
    src: &GgswCiphertextListOwned<u64>,
    src_high: bool,
) {
    let src_base = if src_high { HALF_BITS } else { 0 };
    for chunk_idx in 0..HALF_CHUNKS {
        for i in 0..KAPPA {
            let src_offset = src_base + KAPPA * chunk_idx + i;
            let dst_offset = dst_base + KAPPA * chunk_idx + i;
            copy_ggsw(
                dst,
                VP16_INPUT_BITS - 1 - dst_offset,
                src,
                LIMB_BITS - 1 - src_offset,
            );
        }
    }
}

pub(crate) fn lift_limb_bits(
    ctx: &EvalContext<'_>,
    block: &[Lwe],
    parallel: bool,
) -> GgswCiphertextListOwned<u64> {
    assert_eq!(block.len(), LIMB_CHUNKS);
    let mut ggsw_bits = new_ggsw_list(ctx, LIMB_BITS);

    let lifted_chunks = lift_lwe_chunks(ctx, block, KAPPA, parallel);
    for (chunk_idx, chunk_bits) in lifted_chunks.iter().enumerate() {
        for i in 0..KAPPA {
            copy_ggsw(
                &mut ggsw_bits,
                LIMB_BITS - 1 - (KAPPA * chunk_idx + i),
                &chunk_bits,
                i,
            );
        }
    }

    ggsw_bits
}

pub(crate) fn cbs_lift_add_half(
    ctx: &EvalContext<'_>,
    x_half: &[Lwe],
    y_half: &[Lwe],
    carry: &Lwe,
    parallel: bool,
) -> FourierGgswCiphertextList<Vec<c64>> {
    assert_eq!(x_half.len(), HALF_CHUNKS);
    assert_eq!(y_half.len(), HALF_CHUNKS);
    let mut ggsw_bits = new_ggsw_list(ctx, VP17_INPUT_BITS);

    let (x_chunks, y_chunks) = lift_two_lwe_chunk_slices(ctx, x_half, y_half, KAPPA, parallel);
    for (chunk_idx, chunk_bits) in x_chunks.iter().enumerate() {
        for i in 0..KAPPA {
            copy_ggsw(
                &mut ggsw_bits,
                VP17_INPUT_BITS - 1 - (KAPPA * chunk_idx + i),
                &chunk_bits,
                i,
            );
        }
    }
    for (chunk_idx, chunk_bits) in y_chunks.iter().enumerate() {
        for i in 0..KAPPA {
            copy_ggsw(
                &mut ggsw_bits,
                VP17_INPUT_BITS - 1 - (HALF_BITS + KAPPA * chunk_idx + i),
                &chunk_bits,
                i,
            );
        }
    }

    let carry_bit = cbs_lift_lwe(ctx, carry, 1);
    copy_ggsw(&mut ggsw_bits, 0, &carry_bit, 0);
    fourier_from_ggsw(ctx, &ggsw_bits)
}

pub(crate) fn cbs_lift_add_half_no_carry(
    ctx: &EvalContext<'_>,
    x_half: &[Lwe],
    y_half: &[Lwe],
    parallel: bool,
) -> FourierGgswCiphertextList<Vec<c64>> {
    assert_eq!(x_half.len(), HALF_CHUNKS);
    assert_eq!(y_half.len(), HALF_CHUNKS);
    let mut ggsw_bits = new_ggsw_list(ctx, 2 * HALF_BITS);

    let (x_chunks, y_chunks) = lift_two_lwe_chunk_slices(ctx, x_half, y_half, KAPPA, parallel);
    for (chunk_idx, chunk_bits) in x_chunks.iter().enumerate() {
        for i in 0..KAPPA {
            copy_ggsw(
                &mut ggsw_bits,
                2 * HALF_BITS - 1 - (KAPPA * chunk_idx + i),
                &chunk_bits,
                i,
            );
        }
    }
    for (chunk_idx, chunk_bits) in y_chunks.iter().enumerate() {
        for i in 0..KAPPA {
            copy_ggsw(
                &mut ggsw_bits,
                2 * HALF_BITS - 1 - (HALF_BITS + KAPPA * chunk_idx + i),
                &chunk_bits,
                i,
            );
        }
    }

    fourier_from_ggsw(ctx, &ggsw_bits)
}

fn lift_lwe_chunks(
    ctx: &EvalContext<'_>,
    chunks: &[Lwe],
    bits: usize,
    parallel: bool,
) -> Vec<GgswCiphertextListOwned<u64>> {
    #[cfg(feature = "multithread")]
    if parallel {
        return chunks
            .par_iter()
            .map(|chunk| cbs_lift_lwe(ctx, chunk, bits))
            .collect();
    }

    let _ = parallel;
    chunks
        .iter()
        .map(|chunk| cbs_lift_lwe(ctx, chunk, bits))
        .collect()
}

fn lift_two_lwe_chunk_slices(
    ctx: &EvalContext<'_>,
    lhs: &[Lwe],
    rhs: &[Lwe],
    bits: usize,
    parallel: bool,
) -> (
    Vec<GgswCiphertextListOwned<u64>>,
    Vec<GgswCiphertextListOwned<u64>>,
) {
    #[cfg(feature = "multithread")]
    if parallel {
        return rayon::join(
            || lift_lwe_chunks(ctx, lhs, bits, true),
            || lift_lwe_chunks(ctx, rhs, bits, true),
        );
    }

    (
        lift_lwe_chunks(ctx, lhs, bits, false),
        lift_lwe_chunks(ctx, rhs, bits, false),
    )
}

fn cbs_lift_lwe(ctx: &EvalContext<'_>, input: &Lwe, bits: usize) -> GgswCiphertextListOwned<u64> {
    let mut lwe_extract = LweCiphertext::new(0u64, input.lwe_size(), ctx.ciphertext_modulus);
    lwe_ciphertext_cleartext_mul(&mut lwe_extract, input, Cleartext(1u64));

    let mut lwe_extract_ks =
        LweCiphertext::new(0u64, ctx.ksk.output_lwe_size(), ctx.ciphertext_modulus);
    keyswitch_lwe_ciphertext(ctx.ksk, &lwe_extract, &mut lwe_extract_ks);
    apply_centered_ms_compensation(&mut lwe_extract_ks, ctx.polynomial_size, ctx.log_lut_count);

    let mut acc_glev = GlweCiphertextList::new(
        0u64,
        ctx.glwe_size,
        ctx.polynomial_size,
        GlweCiphertextCount(ctx.cbs_level.0),
        ctx.ciphertext_modulus,
    );
    blind_rotate_for_msb(
        &lwe_extract_ks,
        &mut acc_glev,
        ctx.bsk,
        ctx.log_lut_count,
        ctx.cbs_base_log,
        ctx.cbs_level,
        bits,
        ctx.ciphertext_modulus,
    );

    let mut out = new_ggsw_list(ctx, bits);
    for (i, mut ggsw) in out.iter_mut().enumerate() {
        convert_to_ggsw_after_blind_rotate_revtrace(
            &acc_glev,
            &mut ggsw,
            bits - i - 1,
            ctx.auto_keys,
            ctx.ss_key,
            ctx.ciphertext_modulus,
        );
    }
    out
}

pub(crate) fn vp_mul8x8(
    ctx: &EvalContext<'_>,
    pair_bits: &FourierGgswCiphertextList<Vec<c64>>,
    parallel: bool,
    lut_cache: &mut LutCache,
) -> Limb {
    for chunk in 0..LIMB_CHUNKS {
        lut_cache.mul8x8.entry(chunk).or_insert_with(|| {
            make_vp_lut(VP16_INPUT_BITS, ctx.polynomial_size, |masked_input| {
                let (x, y) = unmask_mul8x8(masked_input);
                let product = x * y;
                (((product >> (KAPPA * chunk)) & 0b11) as u64) * delta(KAPPA)
            })
        });
    }

    #[cfg(feature = "multithread")]
    if parallel {
        return (0..LIMB_CHUNKS)
            .into_par_iter()
            .map(|chunk| {
                let lut = lut_cache.mul8x8.get(&chunk).unwrap();
                vp_eval(ctx, pair_bits, VP16_INPUT_BITS, lut)
            })
            .collect();
    }

    let _ = parallel;
    (0..LIMB_CHUNKS)
        .map(|chunk| {
            let lut = lut_cache.mul8x8.get(&chunk).unwrap();
            vp_eval(ctx, pair_bits, VP16_INPUT_BITS, lut)
        })
        .collect()
}

pub(crate) fn vp_add_sum_half_no_carry(
    ctx: &EvalContext<'_>,
    input_bits: &FourierGgswCiphertextList<Vec<c64>>,
    parallel: bool,
    lut_cache: &mut LutCache,
) -> Vec<Lwe> {
    for chunk in 0..HALF_CHUNKS {
        lut_cache
            .add_sum_half_no_carry
            .entry(chunk)
            .or_insert_with(|| {
                make_vp_lut(2 * HALF_BITS, ctx.polynomial_size, |masked_input| {
                    let (x, y) = unmask_add16(masked_input);
                    let sum = (x + y) & 0xff;
                    (((sum >> (KAPPA * chunk)) & 0b11) as u64) * delta(KAPPA)
                })
            });
    }

    #[cfg(feature = "multithread")]
    if parallel {
        return (0..HALF_CHUNKS)
            .into_par_iter()
            .map(|chunk| {
                let lut = lut_cache.add_sum_half_no_carry.get(&chunk).unwrap();
                vp_eval(ctx, input_bits, 2 * HALF_BITS, lut)
            })
            .collect();
    }

    let _ = parallel;
    (0..HALF_CHUNKS)
        .map(|chunk| {
            let lut = lut_cache.add_sum_half_no_carry.get(&chunk).unwrap();
            vp_eval(ctx, input_bits, 2 * HALF_BITS, lut)
        })
        .collect()
}

pub(crate) fn vp_add_sum_half(
    ctx: &EvalContext<'_>,
    input_bits: &FourierGgswCiphertextList<Vec<c64>>,
    parallel: bool,
    lut_cache: &mut LutCache,
) -> Vec<Lwe> {
    for chunk in 0..HALF_CHUNKS {
        lut_cache.add_sum_half.entry(chunk).or_insert_with(|| {
            make_vp_lut(VP17_INPUT_BITS, ctx.polynomial_size, |masked_input| {
                let (x, y, carry) = unmask_add17(masked_input);
                let sum = (x + y + carry) & 0xff;
                (((sum >> (KAPPA * chunk)) & 0b11) as u64) * delta(KAPPA)
            })
        });
    }

    #[cfg(feature = "multithread")]
    if parallel {
        return (0..HALF_CHUNKS)
            .into_par_iter()
            .map(|chunk| {
                let lut = lut_cache.add_sum_half.get(&chunk).unwrap();
                vp_eval(ctx, input_bits, VP17_INPUT_BITS, lut)
            })
            .collect();
    }

    let _ = parallel;
    (0..HALF_CHUNKS)
        .map(|chunk| {
            let lut = lut_cache.add_sum_half.get(&chunk).unwrap();
            vp_eval(ctx, input_bits, VP17_INPUT_BITS, lut)
        })
        .collect()
}

pub(crate) fn vp_add_carry_no_carry(
    ctx: &EvalContext<'_>,
    input_bits: &FourierGgswCiphertextList<Vec<c64>>,
    lut_cache: &mut LutCache,
) -> Lwe {
    let lut = lut_cache.add_carry_no_carry.get_or_insert_with(|| {
        make_vp_lut(2 * HALF_BITS, ctx.polynomial_size, |masked_input| {
            let (x, y) = unmask_add16(masked_input);
            (((x + y) >> HALF_BITS) as u64) * delta(1)
        })
    });
    vp_eval(ctx, input_bits, 2 * HALF_BITS, lut)
}

pub(crate) fn vp_add_carry(
    ctx: &EvalContext<'_>,
    input_bits: &FourierGgswCiphertextList<Vec<c64>>,
    lut_cache: &mut LutCache,
) -> Lwe {
    let lut = lut_cache.add_carry.get_or_insert_with(|| {
        make_vp_lut(VP17_INPUT_BITS, ctx.polynomial_size, |masked_input| {
            let (x, y, carry) = unmask_add17(masked_input);
            (((x + y + carry) >> HALF_BITS) as u64) * delta(1)
        })
    });
    vp_eval(ctx, input_bits, VP17_INPUT_BITS, lut)
}

fn vp_eval(
    ctx: &EvalContext<'_>,
    ggsw_bits: &FourierGgswCiphertextList<Vec<c64>>,
    input_bits: usize,
    lut: &PolynomialListOwned<u64>,
) -> Lwe {
    let fft = Fft::new(ctx.polynomial_size);
    let fft = fft.as_view();
    let mut out = LweCiphertext::new(0u64, ctx.output_lwe_size, ctx.ciphertext_modulus);
    let mut buffer = ComputationBuffers::new();
    buffer.resize(
        vertical_packing_scratch::<u64>(
            ctx.glwe_size,
            ctx.polynomial_size,
            PolynomialCount((1usize << input_bits) / ctx.polynomial_size.0),
            input_bits,
            fft,
        )
        .unwrap()
        .unaligned_bytes_required(),
    );
    vertical_packing(
        lut.as_view(),
        out.as_mut_view(),
        ggsw_bits.as_view(),
        fft,
        buffer.stack(),
    );
    out
}

fn make_vp_lut(
    input_bits: usize,
    polynomial_size: PolynomialSize,
    f: impl Fn(usize) -> u64,
) -> PolynomialListOwned<u64> {
    let mut lut = PolynomialList::new(
        0u64,
        polynomial_size,
        PolynomialCount((1usize << input_bits) / polynomial_size.0),
    );
    for (cmux_idx, mut poly) in lut.iter_mut().enumerate() {
        for (i, value) in poly.iter_mut().enumerate() {
            *value = f(cmux_idx * polynomial_size.0 + i);
        }
    }
    lut
}

fn apply_centered_ms_compensation(
    lwe: &mut Lwe,
    polynomial_size: PolynomialSize,
    log_lut_count: LutCountLog,
) {
    let log_modulus = polynomial_size.0.ilog2() as usize + 1 - log_lut_count.0;
    let (body, mask) = lwe.as_mut().split_last_mut().unwrap();
    *body = body.wrapping_add(centered_binary_ms_correction(mask, log_modulus));
}

fn centered_binary_ms_correction(mask: &[u64], log_modulus: usize) -> u64 {
    let mut half_rounding_errors = 0u64;
    let mut doubled_halving_errors = 0i128;

    for &mask_element in mask {
        let error = modulus_switch_round(mask_element, log_modulus).wrapping_sub(mask_element);
        let signed_error = if error >= (1u64 << 63) {
            error as i128 - (1i128 << 64)
        } else {
            error as i128
        };
        let half_error = signed_error / 2;
        half_rounding_errors = half_rounding_errors.wrapping_add(half_error as i64 as u64);
        doubled_halving_errors += 2 * half_error - signed_error;
    }

    let halving_errors = (doubled_halving_errors / 2) as i64 as u64;
    let half_case = 1u64 << (u64::BITS as usize - log_modulus - 1);
    half_rounding_errors
        .wrapping_sub(halving_errors)
        .wrapping_sub(half_case)
}

fn modulus_switch_round(input: u64, log_modulus: usize) -> u64 {
    assert!(log_modulus < u64::BITS as usize);
    input.wrapping_add(1u64 << (u64::BITS as usize - log_modulus - 1))
        >> (u64::BITS as usize - log_modulus)
        << (u64::BITS as usize - log_modulus)
}

fn new_ggsw_list(ctx: &EvalContext<'_>, count: usize) -> GgswCiphertextListOwned<u64> {
    GgswCiphertextList::new(
        0u64,
        ctx.glwe_size,
        ctx.polynomial_size,
        ctx.cbs_base_log,
        ctx.cbs_level,
        GgswCiphertextCount(count),
        ctx.ciphertext_modulus,
    )
}

fn fourier_from_ggsw(
    ctx: &EvalContext<'_>,
    ggsw: &GgswCiphertextListOwned<u64>,
) -> FourierGgswCiphertextList<Vec<c64>> {
    let count = ggsw.ggsw_ciphertext_count().0;
    let mut fourier = FourierGgswCiphertextList::new(
        vec![
            c64::default();
            count
                * ctx.polynomial_size.to_fourier_polynomial_size().0
                * ctx.glwe_size.0
                * ctx.glwe_size.0
                * ctx.cbs_level.0
        ],
        count,
        ctx.glwe_size,
        ctx.polynomial_size,
        ctx.cbs_base_log,
        ctx.cbs_level,
    );
    for (mut fourier_ggsw, ggsw) in fourier.as_mut_view().into_ggsw_iter().zip(ggsw.iter()) {
        convert_standard_ggsw_ciphertext_to_fourier(&ggsw, &mut fourier_ggsw);
    }
    fourier
}

fn copy_ggsw(
    dst_list: &mut GgswCiphertextListOwned<u64>,
    dst_idx: usize,
    src_list: &GgswCiphertextListOwned<u64>,
    src_idx: usize,
) {
    let mut dst = dst_list.get_mut(dst_idx);
    let src = src_list.get(src_idx);
    dst.as_mut().copy_from_slice(src.as_ref());
}

fn unmask_add17(masked_input: usize) -> (usize, usize, usize) {
    let x = unmask_by_2bit_chunks(masked_input & 0xff, HALF_BITS);
    let y = unmask_by_2bit_chunks((masked_input >> HALF_BITS) & 0xff, HALF_BITS);
    let carry = (masked_input >> (2 * HALF_BITS)) & 1;
    (x, y, carry)
}

fn unmask_add16(masked_input: usize) -> (usize, usize) {
    let x = unmask_by_2bit_chunks(masked_input & 0xff, HALF_BITS);
    let y = unmask_by_2bit_chunks((masked_input >> HALF_BITS) & 0xff, HALF_BITS);
    (x, y)
}

fn unmask_mul8x8(masked_input: usize) -> (usize, usize) {
    let x = unmask_by_2bit_chunks(masked_input & 0xff, HALF_BITS);
    let y = unmask_by_2bit_chunks((masked_input >> HALF_BITS) & 0xff, HALF_BITS);
    (x, y)
}

fn unmask_by_2bit_chunks(masked: usize, bits: usize) -> usize {
    let mut out = 0usize;
    for offset in (0..bits).step_by(KAPPA) {
        let chunk = (masked >> offset) & 0b11;
        let msb = (chunk >> 1) & 1;
        let lsb = (chunk & 1) ^ msb;
        out |= (lsb | (msb << 1)) << offset;
    }
    out
}
