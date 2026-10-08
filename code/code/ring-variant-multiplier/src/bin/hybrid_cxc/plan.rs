use crate::config::{cached_product, env_flag, BenchParams, ProductVariant};
use crate::types::BoundedLweTerm;
use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize)]
pub(crate) struct ReductionPlan {
    pub(crate) width: usize,
    pub(crate) chunk_bits: usize,
    pub(crate) mode: String,
    pub(crate) squaring: bool,
    pub(crate) log2_failure: Option<f64>,
    pub(crate) env: BTreeMap<String, String>,
    pub(crate) waves: Vec<Vec<PlannedColumn>>,
    pub(crate) final_columns: Vec<Vec<PublicTerm>>,
    /// Ciphertext-plaintext product: public radix-4 digits of the second operand
    /// and the scalar-aware groups of encrypted digit indices per column.
    #[serde(default)]
    pub(crate) public_scalar: Option<Vec<u64>>,
    #[serde(default)]
    pub(crate) public_groups: Option<Vec<Vec<Vec<usize>>>>,
    /// Carry-only reduction: the final addition reads unrefreshed (linear) digits directly.
    #[serde(default)]
    pub(crate) final_direct: bool,
}

#[derive(Deserialize)]
pub(crate) struct PlannedColumn {
    pub(crate) terms: Vec<PublicTerm>,
    pub(crate) groups: Vec<Vec<usize>>,
    /// Per group: keep the digit linear (sum - 4*carry) instead of bootstrapping it.
    #[serde(default)]
    pub(crate) linear: Vec<bool>,
    /// Per group: digit and carry from one multi-value bootstrapping.
    #[serde(default)]
    pub(crate) mvb: Vec<bool>,
}

#[derive(Deserialize, Debug, PartialEq)]
pub(crate) struct PublicTerm {
    bound: u64,
    refreshed: bool,
    source: Option<(usize, usize, usize)>,
}

impl PublicTerm {
    fn of(term: &BoundedLweTerm) -> Self {
        Self {
            bound: term.bound,
            refreshed: term.refreshed,
            source: term.source,
        }
    }
}

impl ReductionPlan {
    pub(crate) fn load(width: usize, params: BenchParams, variant: ProductVariant) -> Result<Self> {
        let path =
            std::env::var("CACHED_MULT_PLAN").context("use run.py to derive a public plan")?;
        let plan: Self = serde_json::from_slice(&std::fs::read(path)?)?;
        ensure!(
            plan.width == width && [16, 32, 64, 128, 256].contains(&width),
            "plan width mismatch"
        );
        ensure!(
            (plan.chunk_bits == 8 && variant == ProductVariant::Hybrid8x8)
                || (plan.chunk_bits == 4 && variant == ProductVariant::Hybrid4x4 && plan.mode == "baseline"),
            "product variant does not match the public plan"
        );
        ensure!(
            ["cached", "baseline", "fused"].contains(&plan.mode.as_str()),
            "unknown plan mode"
        );
        ensure!(
            cached_product() == (plan.mode == "cached"),
            "plan mode mismatch"
        );
        ensure!(
            (crate::config::fused_products_per_group() > 0) == (plan.mode == "fused"),
            "product-sum plan mismatch"
        );
        ensure!(
            plan.public_scalar.is_some() == plan.public_groups.is_some()
                && plan.public_scalar.as_ref().map_or(true, |s| s.len() == width / 2 && s.iter().all(|&b| b < 4))
                && plan.public_groups.as_ref().map_or(true, |g| g.len() == width / 2),
            "malformed public scalar plan"
        );
        ensure!(
            plan.public_scalar.is_none() || env_flag("CBS_CXP"),
            "a scalar-aware plan requires CBS_CXP=1"
        );
        ensure!(!plan.squaring, "this experiment evaluates general multiplication only");
        ensure!(
            plan.squaring == env_flag("CACHED_MULT_SQUARING"),
            "operand mode mismatch"
        );
        ensure!(plan.log2_failure.is_none(), "selector covariance audit: bound must remain unclaimed");
        ensure!(env_flag("FUSED_EXPLORATORY"), "research-only experiment");
        // The shared lift defaults to ordinary modulus switching; the analysed
        // campaign lift must state its mode explicitly in the public plan.
        ensure!(
            matches!(plan.env.get("CBS_CENTERED_MS").map(String::as_str), Some("0" | "1")),
            "the public plan must set CBS_CENTERED_MS explicitly (0 or 1)"
        );
        for (key, value) in &plan.env {
            ensure!(
                std::env::var(key).as_deref() == Ok(value.as_str()),
                "parameter mismatch: {key}"
            );
        }
        ensure!(
            params.lwe_dimension.0 == 866
                && [(1, 2048), (2, 1024)].contains(&(
                    params.glwe_dimension.0, params.polynomial_size.0))
                && params.normalizer_glwe_dimension.0 == 1
                && params.normalizer_polynomial_size.0 == 2048,
            "unmodeled ciphertext dimensions"
        );
        Ok(plan)
    }

    pub(crate) fn finish(&self, columns: &[Vec<BoundedLweTerm>], waves: usize) {
        assert_eq!(waves, self.waves.len(), "missing reduction wave");
        assert_eq!(columns.len(), self.final_columns.len());
        for (column, expected) in columns.iter().zip(&self.final_columns) {
            assert_eq!(
                column.iter().map(PublicTerm::of).collect::<Vec<_>>(),
                *expected,
                "final row does not match the Gaussian plan"
            );
        }
    }
}

impl PlannedColumn {
    pub(crate) fn is_linear(&self, group: usize) -> bool {
        self.linear.get(group).copied().unwrap_or(false)
    }

    pub(crate) fn is_mvb(&self, group: usize) -> bool {
        self.mvb.get(group).copied().unwrap_or(false)
    }

    pub(crate) fn partition(
        &self,
        current: Vec<BoundedLweTerm>,
        cap: u64,
    ) -> Vec<Vec<BoundedLweTerm>> {
        assert_eq!(
            current.iter().map(PublicTerm::of).collect::<Vec<_>>(),
            self.terms,
            "column provenance does not match the Gaussian plan"
        );
        let mut slots: Vec<_> = current.into_iter().map(Some).collect();
        let groups: Vec<Vec<_>> = self
            .groups
            .iter()
            .map(|group| {
                assert!(!group.is_empty(), "empty planned group");
                let terms: Vec<_> = group
                    .iter()
                    .map(|&idx| {
                        slots
                            .get_mut(idx)
                            .expect("planned index out of range")
                            .take()
                            .expect("planned term used twice")
                    })
                    .collect();
                assert!(terms.iter().map(|t| t.bound).sum::<u64>() <= cap);
                terms
            })
            .collect();
        assert!(slots.iter().all(Option::is_none), "plan dropped a term");
        groups
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tfhe::core_crypto::prelude::*;

    fn term() -> BoundedLweTerm {
        BoundedLweTerm {
            ciphertext: LweCiphertext::new(0u64, LweSize(2), CiphertextModulus::new_native()),
            bound: 3,
            refreshed: false,
            source: Some((0, 0, 0)),
            #[cfg(feature = "noise-audit")]
            expected: None,
        }
    }

    #[test]
    #[should_panic(expected = "used twice")]
    fn rejects_duplicate_indices() {
        let plan = PlannedColumn {
            terms: vec![PublicTerm::of(&term())],
            groups: vec![vec![0, 0]],
            linear: vec![],
            mvb: vec![],
        };
        plan.partition(vec![term()], 15);
    }

    #[test]
    #[should_panic(expected = "dropped")]
    fn rejects_missing_terms() {
        let plan = PlannedColumn {
            terms: vec![PublicTerm::of(&term())],
            groups: vec![],
            linear: vec![],
            mvb: vec![],
        };
        plan.partition(vec![term()], 15);
    }
}
