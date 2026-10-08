use tfhe::core_crypto::prelude::*;

#[derive(Clone)]
pub(crate) struct BoundedLweTerm {
    pub(crate) ciphertext: LweCiphertextOwned<u64>,
    pub(crate) bound: u64,
    pub(crate) refreshed: bool,
    /// Public routing provenance `(lhs_start_digit, rhs_start_digit, t)` of a
    /// local-product digit; used by the noise-audit harness to derive the
    /// expected plaintext value.  Always public information.
    pub(crate) source: Option<(usize, usize, usize)>,
    /// Audit-only expected plaintext value. This field is absent from normal
    /// benchmark binaries.
    #[cfg(feature = "noise-audit")]
    pub(crate) expected: Option<u64>,
}

impl BoundedLweTerm {
    pub(crate) fn audit_expected(&self) -> Option<u64> {
        #[cfg(feature = "noise-audit")]
        {
            self.expected
        }
        #[cfg(not(feature = "noise-audit"))]
        {
            None
        }
    }

    pub(crate) fn set_audit_expected(&mut self, expected: Option<u64>) {
        #[cfg(feature = "noise-audit")]
        {
            self.expected = expected;
        }
        #[cfg(not(feature = "noise-audit"))]
        {
            let _ = expected;
        }
    }
}

pub(crate) type ProductDigitTerm = BoundedLweTerm;

/// Wall-clock duration in fractional milliseconds (sub-millisecond precision).
pub(crate) fn duration_ms(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

#[derive(Default)]
pub(crate) struct Timings {
    pub(crate) harness_encrypt_ms: f64,
    pub(crate) selector_extract_ms: f64,
    pub(crate) cbs_ms: f64,
    pub(crate) ext_ms: f64,
    pub(crate) diagonal_extract_ms: f64,
    pub(crate) normalization_ms: f64,
    pub(crate) harness_decrypt_ms: f64,
    pub(crate) cbs_lifts: usize,
    pub(crate) cmux_count: usize,
    pub(crate) direct_external_products: usize,
    pub(crate) prefix_cache_entries: usize,
    pub(crate) prefix_cache_hits: usize,
    pub(crate) normalization_pbs: usize,
    pub(crate) normalization_reduction_pbs: usize,
    pub(crate) normalization_final_pbs: usize,
    pub(crate) normalization_rounds: usize,
    pub(crate) normalization_max_chunks: usize,
    pub(crate) normalization_max_column_height: usize,
    pub(crate) normalization_max_column_bound: u64,
    pub(crate) normalization_key_switches: usize,
    pub(crate) product_generation_elapsed_ms: f64,
    /// True when a validation/audit callback (noise probe, paired product,
    /// step audit, PBS re-verification) ran; such rows are not latency samples.
    pub(crate) audit_callbacks: bool,
}

impl Timings {
    pub(crate) fn add_normalization(&mut self, stats: NormalizationStats) {
        self.normalization_key_switches += stats.key_switches;
        self.normalization_pbs += stats.reduction_pbs + stats.final_pbs;
        self.normalization_reduction_pbs += stats.reduction_pbs;
        self.normalization_final_pbs += stats.final_pbs;
        self.normalization_rounds += stats.rounds;
        self.normalization_max_chunks = self.normalization_max_chunks.max(stats.max_chunks);
        self.normalization_max_column_height = self
            .normalization_max_column_height
            .max(stats.max_column_height);
        self.normalization_max_column_bound = self
            .normalization_max_column_bound
            .max(stats.max_column_bound);
    }

    pub(crate) fn product_generation_ms(&self) -> f64 {
        self.product_generation_elapsed_ms
    }

    pub(crate) fn post_product_ms(&self) -> f64 {
        self.normalization_ms
    }

    pub(crate) fn harness_overhead_ms(&self) -> f64 {
        self.harness_encrypt_ms + self.harness_decrypt_ms
    }
}

#[derive(Default, Clone, Copy)]
pub(crate) struct NormalizationStats {
    pub(crate) key_switches: usize,
    /// Executed reduction-wave blind rotations (digit plus retained carries).
    pub(crate) reduction_pbs: usize,
    /// Library PBS counter delta of the final row refresh and radix addition.
    pub(crate) final_pbs: usize,
    pub(crate) rounds: usize,
    pub(crate) max_chunks: usize,
    pub(crate) max_column_height: usize,
    pub(crate) max_column_bound: u64,
}
