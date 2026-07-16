use super::{
    lookup::{
        cbs_lift_add_half, cbs_lift_add_half_no_carry, combine_half_bits, lift_limb_bits,
        vp_add_carry, vp_add_carry_no_carry, vp_add_sum_half, vp_add_sum_half_no_carry, vp_mul8x8,
        LutCache,
    },
    types::{EvalContext, Limb, Lwe, Stats, HALF_CHUNKS},
};
#[cfg(feature = "multithread")]
use rayon::prelude::*;
use std::time::Instant;
use tfhe::core_crypto::prelude::*;

pub(crate) fn cxc_mul(
    ctx: &EvalContext<'_>,
    x: &[Limb],
    y: &[Limb],
    zero: &Limb,
    zero_carry: &Lwe,
    parallel: bool,
    stats: &mut Stats,
) -> Vec<Limb> {
    assert_eq!(x.len(), y.len());
    let t = x.len();
    let x_bits = lift_operand_bits(ctx, x, parallel, stats);
    let y_bits = lift_operand_bits(ctx, y, parallel, stats);
    let mut out = (0..t).map(|_| clone_limb(zero)).collect::<Vec<_>>();
    let mut lut_cache = LutCache::default();

    for j in 0..t {
        for i in 0..t {
            let k = i + j;
            if k >= t {
                continue;
            }

            let (low, high) = partial_mul16_cxc(
                ctx,
                &x_bits[i],
                &y_bits[j],
                zero,
                zero_carry,
                parallel,
                k + 1 < t,
                &mut lut_cache,
                stats,
            );

            let emit_carry = k + 1 < t;
            let (sum, mut carry) = add16_cxc_emit(
                ctx,
                &out[k],
                &low,
                zero_carry,
                false,
                emit_carry,
                zero_carry,
                parallel,
                &mut lut_cache,
                stats,
            );
            out[k] = sum;
            let mut carry_is_known_zero = !emit_carry;

            if k + 1 < t {
                let emit_carry = k + 2 < t;
                let carry_in = if carry_is_known_zero {
                    zero_carry
                } else {
                    &carry
                };
                let (sum, next_carry) = add16_cxc_emit(
                    ctx,
                    &out[k + 1],
                    &high,
                    carry_in,
                    carry_is_known_zero,
                    emit_carry,
                    zero_carry,
                    parallel,
                    &mut lut_cache,
                    stats,
                );
                out[k + 1] = sum;
                carry = next_carry;
                carry_is_known_zero = !emit_carry;

                if !carry_is_known_zero {
                    for (limb_idx, limb) in out.iter_mut().enumerate().skip(k + 2) {
                        let emit_carry = limb_idx + 1 < t;
                        let (sum, next_carry) = add16_cxc_emit(
                            ctx,
                            limb,
                            zero,
                            &carry,
                            false,
                            emit_carry,
                            zero_carry,
                            parallel,
                            &mut lut_cache,
                            stats,
                        );
                        *limb = sum;
                        carry = next_carry;
                        if !emit_carry {
                            break;
                        }
                    }
                }
            }
        }
    }

    out
}

fn lift_operand_bits(
    ctx: &EvalContext<'_>,
    blocks: &[Limb],
    parallel: bool,
    stats: &mut Stats,
) -> Vec<GgswCiphertextListOwned<u64>> {
    #[cfg(feature = "multithread")]
    if parallel {
        let start = Instant::now();
        let lifted = blocks
            .par_iter()
            .map(|limb| lift_limb_bits(ctx, limb, parallel))
            .collect::<Vec<_>>();
        stats.initial_lift_calls += blocks.len();
        stats.initial_lift_time += start.elapsed();
        return lifted;
    }

    blocks
        .iter()
        .map(|limb| {
            let start = Instant::now();
            let lifted = lift_limb_bits(ctx, limb, parallel);
            stats.initial_lift_calls += 1;
            stats.initial_lift_time += start.elapsed();
            lifted
        })
        .collect()
}

fn partial_mul16_cxc(
    ctx: &EvalContext<'_>,
    x_bits: &GgswCiphertextListOwned<u64>,
    y_bits: &GgswCiphertextListOwned<u64>,
    zero: &Limb,
    zero_carry: &Lwe,
    parallel: bool,
    need_high: bool,
    lut_cache: &mut LutCache,
    stats: &mut Stats,
) -> (Limb, Limb) {
    let p00 = vp_byte_product(
        ctx, x_bits, false, y_bits, false, parallel, lut_cache, stats,
    );
    let p01 = vp_byte_product(ctx, x_bits, false, y_bits, true, parallel, lut_cache, stats);
    let p10 = vp_byte_product(ctx, x_bits, true, y_bits, false, parallel, lut_cache, stats);
    let p01_lo = shift_left8_low(&p01, zero);
    let p10_lo = shift_left8_low(&p10, zero);

    let (low_a, carry_a) = add16_cxc_emit(
        ctx, &p00, &p01_lo, zero_carry, true, need_high, zero_carry, parallel, lut_cache, stats,
    );
    let (low, carry_b) = add16_cxc_emit(
        ctx, &low_a, &p10_lo, zero_carry, true, need_high, zero_carry, parallel, lut_cache, stats,
    );
    if !need_high {
        return (low, clone_limb(zero));
    }

    let p11 = vp_byte_product(ctx, x_bits, true, y_bits, true, parallel, lut_cache, stats);
    let p01_hi = shift_right8_high(&p01, zero);
    let p10_hi = shift_right8_high(&p10, zero);
    let (high_a, _) = add16_cxc_emit(
        ctx, &p11, &p01_hi, &carry_a, false, false, zero_carry, parallel, lut_cache, stats,
    );
    let (high, _) = add16_cxc_emit(
        ctx, &high_a, &p10_hi, &carry_b, false, false, zero_carry, parallel, lut_cache, stats,
    );

    (low, high)
}

fn vp_byte_product(
    ctx: &EvalContext<'_>,
    x_bits: &GgswCiphertextListOwned<u64>,
    x_high: bool,
    y_bits: &GgswCiphertextListOwned<u64>,
    y_high: bool,
    parallel: bool,
    lut_cache: &mut LutCache,
    stats: &mut Stats,
) -> Limb {
    let start = Instant::now();
    let pair_bits = combine_half_bits(ctx, x_bits, x_high, y_bits, y_high);
    let product = vp_mul8x8(ctx, &pair_bits, parallel, lut_cache);
    stats.product_lookups += 1;
    stats.product_vp_time += start.elapsed();
    product
}

fn shift_left8_low(block: &Limb, zero: &Limb) -> Limb {
    zero[..HALF_CHUNKS]
        .iter()
        .cloned()
        .chain(block[..HALF_CHUNKS].iter().cloned())
        .collect()
}

fn shift_right8_high(block: &Limb, zero: &Limb) -> Limb {
    block[HALF_CHUNKS..]
        .iter()
        .cloned()
        .chain(zero[..HALF_CHUNKS].iter().cloned())
        .collect()
}

fn add16_cxc_emit(
    ctx: &EvalContext<'_>,
    x: &Limb,
    y: &Limb,
    carry_in: &Lwe,
    carry_in_is_known_zero: bool,
    emit_carry: bool,
    zero_carry: &Lwe,
    parallel: bool,
    lut_cache: &mut LutCache,
    stats: &mut Stats,
) -> (Limb, Lwe) {
    stats.add16_calls += 1;
    let add_start = Instant::now();

    let start = Instant::now();
    let low_bits = if carry_in_is_known_zero {
        stats.add16_lwe_lifts += 2 * HALF_CHUNKS;
        cbs_lift_add_half_no_carry(ctx, &x[..HALF_CHUNKS], &y[..HALF_CHUNKS], parallel)
    } else {
        stats.add16_lwe_lifts += 2 * HALF_CHUNKS + 1;
        cbs_lift_add_half(
            ctx,
            &x[..HALF_CHUNKS],
            &y[..HALF_CHUNKS],
            carry_in,
            parallel,
        )
    };
    stats.add16_low_lift_time += start.elapsed();

    let start = Instant::now();
    let z_lo = if carry_in_is_known_zero {
        vp_add_sum_half_no_carry(ctx, &low_bits, parallel, lut_cache)
    } else {
        vp_add_sum_half(ctx, &low_bits, parallel, lut_cache)
    };
    stats.add16_sum_vp_outputs += HALF_CHUNKS;
    stats.add16_sum_vp_time += start.elapsed();

    let start = Instant::now();
    let c1 = if carry_in_is_known_zero {
        vp_add_carry_no_carry(ctx, &low_bits, lut_cache)
    } else {
        vp_add_carry(ctx, &low_bits, lut_cache)
    };
    stats.add16_carry_vp_outputs += 1;
    stats.add16_carry_vp_time += start.elapsed();

    let start = Instant::now();
    stats.add16_lwe_lifts += 2 * HALF_CHUNKS + 1;
    let high_bits = cbs_lift_add_half(ctx, &x[HALF_CHUNKS..], &y[HALF_CHUNKS..], &c1, parallel);
    stats.add16_high_lift_time += start.elapsed();

    let start = Instant::now();
    let z_hi = vp_add_sum_half(ctx, &high_bits, parallel, lut_cache);
    stats.add16_sum_vp_outputs += HALF_CHUNKS;
    stats.add16_sum_vp_time += start.elapsed();

    let carry_out = if emit_carry {
        let start = Instant::now();
        let carry_out = vp_add_carry(ctx, &high_bits, lut_cache);
        stats.add16_carry_vp_outputs += 1;
        stats.add16_carry_vp_time += start.elapsed();
        carry_out
    } else {
        stats.discarded_top_carries += 1;
        zero_carry.clone()
    };

    stats.add16_time += add_start.elapsed();

    ([z_lo, z_hi].concat(), carry_out)
}

fn clone_limb(block: &Limb) -> Limb {
    block.iter().cloned().collect()
}
