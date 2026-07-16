use tfhe::core_crypto::prelude::*;

#[derive(Clone)]
pub(crate) struct BoundedLweTerm {
    pub(crate) ciphertext: LweCiphertextOwned<u64>,
    pub(crate) bound: u64,
    pub(crate) refreshed: bool,
}

pub(crate) type ProductDigitTerm = BoundedLweTerm;

#[derive(Default)]
pub(crate) struct Timings {
    pub(crate) harness_encrypt_ms: u128,
    pub(crate) selector_extract_ms: u128,
    pub(crate) cbs_ms: u128,
    pub(crate) ext_ms: u128,
    pub(crate) diagonal_extract_ms: u128,
    pub(crate) normalization_ms: u128,
    pub(crate) harness_decrypt_ms: u128,
    pub(crate) cbs_lifts: usize,
    pub(crate) cmux_count: usize,
    pub(crate) normalization_pbs: usize,
    pub(crate) normalization_rounds: usize,
    pub(crate) normalization_max_chunks: usize,
    pub(crate) normalization_max_column_height: usize,
    pub(crate) normalization_max_column_bound: u64,
}

impl Timings {
    pub(crate) fn add_normalization(&mut self, stats: NormalizationStats) {
        self.normalization_pbs += stats.pbs;
        self.normalization_rounds += stats.rounds;
        self.normalization_max_chunks = self.normalization_max_chunks.max(stats.max_chunks);
        self.normalization_max_column_height = self
            .normalization_max_column_height
            .max(stats.max_column_height);
        self.normalization_max_column_bound = self
            .normalization_max_column_bound
            .max(stats.max_column_bound);
    }

    pub(crate) fn product_generation_ms(&self) -> u128 {
        self.selector_extract_ms + self.cbs_ms + self.ext_ms + self.diagonal_extract_ms
    }

    pub(crate) fn post_product_ms(&self) -> u128 {
        self.normalization_ms
    }

    pub(crate) fn harness_overhead_ms(&self) -> u128 {
        self.harness_encrypt_ms + self.harness_decrypt_ms
    }
}

#[derive(Default, Clone, Copy)]
pub(crate) struct NormalizationStats {
    pub(crate) pbs: usize,
    pub(crate) rounds: usize,
    pub(crate) max_chunks: usize,
    pub(crate) max_column_height: usize,
    pub(crate) max_column_bound: u64,
}
