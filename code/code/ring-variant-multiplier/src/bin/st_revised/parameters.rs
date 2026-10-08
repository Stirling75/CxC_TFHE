use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Parameters {
    pub name: String,
    pub lwe_dimension: usize,
    pub polynomial_size: usize,
    pub glwe_dimension: usize,
    pub lwe_sigma: f64,
    pub glwe_sigma: f64,
    pub pbs: [usize; 2],
    pub ks: [usize; 2],
    pub auto: [usize; 2],
    pub ss: [usize; 2],
    pub cbs: [usize; 2],
    #[serde(default = "default_terminal_lut_count_log")]
    pub terminal_lut_count_log: usize,
    pub split_trace_fft: bool,
    /// CC2 chunks stay under the extracted (big) key and every bootstrap is
    /// preceded by a key switch ("pre-PBS key switching"); otherwise CC2 is
    /// key-switched back to the small key after emission.
    #[serde(default)]
    pub cc2_big_key: bool,
}

fn default_terminal_lut_count_log() -> usize { 2 }

impl Parameters {
    pub fn reported(split_trace_fft: bool) -> Self {
        Self { name: "reported-decompositions-refined-noise-assumption".into(),
            lwe_dimension: 768, polynomial_size: 1024, glwe_dimension: 2,
            lwe_sigma: 8.763872947670246e-6, glwe_sigma: 9.25119974676756e-16,
            pbs: [15, 2], ks: [2, 7], auto: [12, 3], ss: [17, 2], cbs: [4, 4],
            terminal_lut_count_log: 2, split_trace_fft, cc2_big_key: false }
    }

    pub fn validate(&self) -> Result<()> {
        ensure!([1024, 2048].contains(&self.polynomial_size), "unsupported ring size");
        ensure!(self.glwe_dimension > 0 && self.glwe_dimension <= 2, "unsupported GLWE dimension");
        ensure!(self.lwe_dimension > 0 && self.lwe_dimension < self.polynomial_size * self.glwe_dimension,
            "small and extracted keys must have distinct supported dimensions");
        for sigma in [self.lwe_sigma, self.glwe_sigma] {
            ensure!(sigma.is_finite() && sigma > 0.0 && sigma < 1.0, "invalid noise standard deviation");
        }
        for [base, levels] in [self.pbs, self.ks, self.auto, self.ss, self.cbs] {
            ensure!(base > 0 && levels > 0 && base < 64 && levels < 64 && base * levels < 64,
                "invalid native signed decomposition");
        }
        ensure!(self.cbs[1] == 4, "terminal implementation currently supports four CBS levels");
        ensure!([1, 2].contains(&self.terminal_lut_count_log), "terminal LUT packing supports two or four lanes");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reported_parameters_are_preserved() {
        let p = Parameters::reported(false);
        p.validate().unwrap();
        assert_eq!((p.polynomial_size, p.glwe_dimension, p.lwe_dimension), (1024, 2, 768));
        assert_eq!(p.ks, [2, 7]);
    }
    #[test]
    fn reject_incompatible_cbs_and_key_dimensions() {
        let mut p = Parameters::reported(false);
        p.cbs[1] = 5;
        assert!(p.validate().is_err());
        p.cbs[1] = 4;
        p.lwe_dimension = 2048;
        assert!(p.validate().is_err());
    }
}
