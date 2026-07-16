use aligned_vec::ABox;
use refined_tfhe_lhe::AutomorphKey;
use std::{collections::HashMap, time::Duration};
use tfhe::core_crypto::{
    fft_impl::fft64::{
        c64,
        crypto::{bootstrap::FourierLweBootstrapKeyView, ggsw::FourierGgswCiphertextListView},
    },
    prelude::*,
};

pub(crate) const KAPPA: usize = 2;
pub(crate) const LIMB_BITS: usize = 16;
pub(crate) const HALF_BITS: usize = 8;
pub(crate) const LIMB_CHUNKS: usize = LIMB_BITS / KAPPA;
pub(crate) const HALF_CHUNKS: usize = HALF_BITS / KAPPA;
pub(crate) const VP16_INPUT_BITS: usize = 16;
pub(crate) const VP17_INPUT_BITS: usize = 17;

pub(crate) type Lwe = LweCiphertextOwned<u64>;
pub(crate) type Limb = Vec<Lwe>;

pub(crate) struct EvalContext<'a> {
    pub(crate) ksk: &'a LweKeyswitchKeyOwned<u64>,
    pub(crate) bsk: FourierLweBootstrapKeyView<'a>,
    pub(crate) auto_keys: &'a HashMap<usize, AutomorphKey<ABox<[c64]>>>,
    pub(crate) ss_key: FourierGgswCiphertextListView<'a>,
    pub(crate) output_lwe_size: LweSize,
    pub(crate) glwe_size: GlweSize,
    pub(crate) polynomial_size: PolynomialSize,
    pub(crate) cbs_base_log: DecompositionBaseLog,
    pub(crate) cbs_level: DecompositionLevelCount,
    pub(crate) log_lut_count: LutCountLog,
    pub(crate) ciphertext_modulus: CiphertextModulus<u64>,
}

#[derive(Default)]
pub(crate) struct Stats {
    pub(crate) initial_lift_calls: usize,
    pub(crate) initial_lift_time: Duration,
    pub(crate) product_lookups: usize,
    pub(crate) product_vp_time: Duration,
    pub(crate) add16_calls: usize,
    pub(crate) add16_lwe_lifts: usize,
    pub(crate) add16_time: Duration,
    pub(crate) add16_low_lift_time: Duration,
    pub(crate) add16_high_lift_time: Duration,
    pub(crate) add16_sum_vp_outputs: usize,
    pub(crate) add16_sum_vp_time: Duration,
    pub(crate) add16_carry_vp_outputs: usize,
    pub(crate) add16_carry_vp_time: Duration,
    pub(crate) discarded_top_carries: usize,
}

impl Stats {
    pub(crate) fn print(&self, elapsed: Duration) {
        let add16_lift_time = self.add16_low_lift_time + self.add16_high_lift_time;
        let add16_vp_time = self.add16_sum_vp_time + self.add16_carry_vp_time;
        println!("stats:");
        println!(
            "  initial CBS lift: {} limbs, {} 2-bit LWE lifts, {:.3} ms",
            self.initial_lift_calls,
            self.initial_lift_calls * LIMB_CHUNKS,
            ms(self.initial_lift_time)
        );
        println!(
            "  product VP8x8: {} byte-product groups, {} VP outputs, {:.3} ms",
            self.product_lookups,
            self.product_lookups * LIMB_CHUNKS,
            ms(self.product_vp_time)
        );
        println!(
            "  Add16CXC total: {} calls, {:.3} ms",
            self.add16_calls,
            ms(self.add16_time)
        );
        println!(
            "    Add16 CBS lift: {} half-add lifts, {} LWE lifts, {:.3} ms",
            self.add16_calls * 2,
            self.add16_lwe_lifts,
            ms(add16_lift_time)
        );
        println!(
            "    Add16 VP sum: {} VP outputs, {:.3} ms",
            self.add16_sum_vp_outputs,
            ms(self.add16_sum_vp_time)
        );
        println!(
            "    Add16 VP carry: {} VP outputs, {:.3} ms",
            self.add16_carry_vp_outputs,
            ms(self.add16_carry_vp_time)
        );
        println!(
            "    Add16 VP total: {} VP outputs, {:.3} ms",
            self.add16_sum_vp_outputs + self.add16_carry_vp_outputs,
            ms(add16_vp_time)
        );
        println!("  truncated carry outputs: {}", self.discarded_top_carries);
        println!("  measured elapsed: {:.3} ms", ms(elapsed));
    }
}

pub(crate) fn delta(bits: usize) -> u64 {
    1u64 << (u64::BITS as usize - bits)
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}
