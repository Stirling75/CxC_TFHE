use clap::ValueEnum;
use num_bigint::BigUint;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::Serialize;

#[derive(Clone, Copy, Debug, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Pattern {
    Random,
    Zero,
    Max,
    Alternating,
    CarryChain,
}

pub fn operands(width: usize, seed: u64, trial: usize, pattern: Pattern) -> (Vec<u8>, Vec<u8>) {
    let mut rng = StdRng::seed_from_u64(
        seed.wrapping_add(trial as u64)
            .wrapping_add((width as u64) << 32),
    );
    let mut x: Vec<_> = (0..width).map(|_| rng.gen_range(0..2)).collect();
    let mut y: Vec<_> = (0..width).map(|_| rng.gen_range(0..2)).collect();
    match pattern {
        Pattern::Random => {}
        Pattern::Zero => x.fill(0),
        Pattern::Max => {
            x.fill(1);
            y.fill(1);
        }
        Pattern::Alternating => {
            for j in 0..width {
                x[j] = (j % 2) as u8;
                y[j] = 1 - x[j];
            }
        }
        Pattern::CarryChain => {
            x.fill(1);
            y.fill(0);
            y[0] = 1;
            if width > 1 {
                y[1] = 1;
            }
        }
    }
    (x, y)
}

pub fn value(bits: &[u8]) -> BigUint {
    bits.iter()
        .rev()
        .fold(BigUint::from(0u8), |a, b| (a << 1) + b)
}

pub fn expected(x: &[u8], y: &[u8], output_width: usize) -> BigUint {
    let mask = (BigUint::from(1u8) << output_width) - 1u8;
    (value(x) * value(y)) & mask
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_product_checker_does_not_discard_high_bits() {
        let (x, y) = operands(256, 0, 0, Pattern::Max);
        assert_eq!(expected(&x, &y, 256), BigUint::from(1u8));
        assert!(expected(&x, &y, 512).bits() > 256);
    }

    // Pinned vectors shared with bitwise_check.py's Python port (test_bitwise_check.py).
    #[test]
    fn operand_derivation_is_pinned() {
        for (width, seed, trial, x_hex, y_hex) in [
            (16, 20260908, 0, "83ff", "7d14"),
            (16, 20260908, 3, "77e6", "c147"),
            (7, 0, 0, "22", "77"),
            (256, u64::MAX, 5,
             "8d345b771757ad0660c5ced50345e89f56dcf4c8350013b839a94aa5640e7451",
             "625a903c20bcbf03b486291cec4b490ecbc76b4118866f2b4b66fd3b89f528b7"),
        ] {
            let (x, y) = operands(width, seed, trial, Pattern::Random);
            assert_eq!(value(&x).to_str_radix(16), x_hex);
            assert_eq!(value(&y).to_str_radix(16), y_hex);
        }
    }
}
