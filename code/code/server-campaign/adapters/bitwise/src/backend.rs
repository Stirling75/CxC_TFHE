use crate::circuits::{morshed_adder, Gates};
use anyhow::{ensure, Result};
use clap::ValueEnum;
use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};
use tfhe::shortint::{
    parameters::{
        v1_7::{
            V1_7_PARAM_MESSAGE_1_CARRY_1_KS_PBS_GAUSSIAN_2M128,
            V1_7_PARAM_MESSAGE_1_CARRY_1_KS_PBS_TUNIFORM_2M128,
            V1_7_PARAM_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M128,
        },
        ClassicPBSParameters, Degree,
    },
    server_key::LookupTableOwned,
    Ciphertext, ClientKey, ServerKey,
};

#[derive(Clone, Copy, Debug, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Preset {
    M1c1Gaussian,
    M1c1Tuniform,
    M2c2GaussianControl,
}

impl Preset {
    pub fn parameters(self) -> (&'static str, ClassicPBSParameters) {
        match self {
            Self::M1c1Gaussian => (
                "V1_7_PARAM_MESSAGE_1_CARRY_1_KS_PBS_GAUSSIAN_2M128",
                V1_7_PARAM_MESSAGE_1_CARRY_1_KS_PBS_GAUSSIAN_2M128,
            ),
            Self::M1c1Tuniform => (
                "V1_7_PARAM_MESSAGE_1_CARRY_1_KS_PBS_TUNIFORM_2M128",
                V1_7_PARAM_MESSAGE_1_CARRY_1_KS_PBS_TUNIFORM_2M128,
            ),
            Self::M2c2GaussianControl => (
                "V1_7_PARAM_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M128",
                V1_7_PARAM_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M128,
            ),
        }
    }
}

pub struct ShortintGates {
    pub sk: ServerKey,
    and: LookupTableOwned,
    xor: LookupTableOwned,
    parity: LookupTableOwned,
    carry: LookupTableOwned,
    // Every adapter LUT call (the circuit's logical count).
    calls: AtomicU64,
    // Calls whose LUT input is a trivial (noiseless, zero-mask) ciphertext.
    // TFHE-rs evaluates these in the clear (trivial_pbs_assign), without a
    // blind rotation, but its pbs-stats PBS_COUNT still increments for them.
    trivial: AtomicU64,
    // Calls with a non-trivial input, i.e. real keyswitch + blind rotations.
    real: AtomicU64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LutCounts {
    pub logical: u64,
    pub trivial: u64,
    pub real: u64,
}

impl ShortintGates {
    pub fn new(sk: ServerKey) -> Self {
        Self {
            and: sk.generate_lookup_table(|s| (s & 1) & ((s >> 1) & 1)),
            xor: sk.generate_lookup_table(|s| (s & 1) ^ ((s >> 1) & 1)),
            parity: sk.generate_lookup_table(|s| s & 1),
            carry: sk.generate_lookup_table(|s| (s >> 1) & 1),
            sk,
            calls: AtomicU64::new(0),
            trivial: AtomicU64::new(0),
            real: AtomicU64::new(0),
        }
    }

    fn lookup(&self, input: &Ciphertext, lut: &LookupTableOwned) -> Ciphertext {
        self.sk
            .max_noise_level
            .validate(input.noise_level())
            .expect("PBS input noise capacity");
        assert!(input.degree.get() <= 3, "bit-gate LUT input domain");
        self.calls.fetch_add(1, Ordering::Relaxed);
        if input.is_trivial() {
            self.trivial.fetch_add(1, Ordering::Relaxed);
        } else {
            self.real.fetch_add(1, Ordering::Relaxed);
        }
        let result = self.sk.apply_lookup_table(input, lut);
        assert!(result.degree.get() <= 1);
        result
    }

    fn packed_pair(&self, a: &Ciphertext, b: &Ciphertext) -> Ciphertext {
        let twice_b = self
            .sk
            .checked_scalar_mul(b, 2)
            .expect("binary gate scalar capacity");
        self.sk
            .checked_add(a, &twice_b)
            .expect("binary gate addition capacity")
    }

    fn triple_sum(&self, a: &Ciphertext, b: &Ciphertext, c: &Ciphertext) -> Ciphertext {
        let pair = self
            .sk
            .checked_add(a, b)
            .expect("ternary gate pair capacity");
        self.sk
            .checked_add(&pair, c)
            .expect("ternary gate sum capacity")
    }

    pub fn reset_calls(&self) {
        self.calls.store(0, Ordering::Relaxed);
        self.trivial.store(0, Ordering::Relaxed);
        self.real.store(0, Ordering::Relaxed);
    }
    pub fn counts(&self) -> LutCounts {
        LutCounts {
            logical: self.calls.load(Ordering::Relaxed),
            trivial: self.trivial.load(Ordering::Relaxed),
            real: self.real.load(Ordering::Relaxed),
        }
    }

    pub fn encrypt_bit(&self, ck: &ClientKey, bit: u8, bootstrap: bool) -> Ciphertext {
        assert!(bit <= 1);
        let mut ct = ck.encrypt(bit as u64);
        // The input contract is a bit, including for the m2c2 control.
        ct.degree = Degree::new(1);
        if bootstrap {
            self.sk.apply_lookup_table(&ct, &self.parity)
        } else {
            ct
        }
    }

    pub fn preflight(&self, ck: &ClientKey) -> Result<()> {
        for a in 0..2 {
            for b in 0..2 {
                for c in 0..2 {
                    let ca = self.encrypt_bit(ck, a, true);
                    let cb = self.encrypt_bit(ck, b, true);
                    let cc = self.encrypt_bit(ck, c, true);
                    ensure!(ck.decrypt_message_and_carry(&self.and(&ca, &cb)) == (a & b) as u64);
                    ensure!(ck.decrypt_message_and_carry(&self.xor(&ca, &cb)) == (a ^ b) as u64);
                    ensure!(
                        ck.decrypt_message_and_carry(&self.parity3(&ca, &cb, &cc))
                            == ((a + b + c) & 1) as u64
                    );
                    for (s, k) in [
                        self.sum_carry(&ca, &cb, &cc),
                        morshed_adder(self, &ca, &cb, &cc),
                    ] {
                        let total = (a + b + c) as u64;
                        ensure!(
                            ck.decrypt_message_and_carry(&s) == total % 2,
                            "sum truth table"
                        );
                        ensure!(
                            ck.decrypt_message_and_carry(&k) == total / 2,
                            "carry truth table"
                        );
                    }
                }
            }
        }
        Ok(())
    }
}

impl Gates for ShortintGates {
    type Bit = Ciphertext;
    fn zero(&self) -> Ciphertext {
        self.sk.create_trivial(0)
    }
    fn and(&self, a: &Ciphertext, b: &Ciphertext) -> Ciphertext {
        self.lookup(&self.packed_pair(a, b), &self.and)
    }
    fn xor(&self, a: &Ciphertext, b: &Ciphertext) -> Ciphertext {
        self.lookup(&self.packed_pair(a, b), &self.xor)
    }
    fn parity3(&self, a: &Ciphertext, b: &Ciphertext, c: &Ciphertext) -> Ciphertext {
        self.lookup(&self.triple_sum(a, b, c), &self.parity)
    }
    fn sum_carry(
        &self,
        a: &Ciphertext,
        b: &Ciphertext,
        c: &Ciphertext,
    ) -> (Ciphertext, Ciphertext) {
        let sum = self.triple_sum(a, b, c);
        (
            self.lookup(&sum, &self.parity),
            self.lookup(&sum, &self.carry),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preset_contracts() {
        for preset in [Preset::M1c1Gaussian, Preset::M1c1Tuniform] {
            let (_, p) = preset.parameters();
            assert_eq!((p.message_modulus.0, p.carry_modulus.0), (2, 2));
            assert!(p.log2_p_fail < -128.0);
            assert_eq!((p.polynomial_size.0, p.glwe_dimension.0), (512, 4));
        }
    }
}
