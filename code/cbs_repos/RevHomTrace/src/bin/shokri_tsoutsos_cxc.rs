//! CxC multiplication following Section 3.3 of ePrint 2026/810.
//!
//! The evaluator uses 16-bit limbs, four 8-by-8 VP lookups per limb product,
//! and the Add16CXC carry pipeline. The benchmark uses the dimensions and
//! decomposition parameters from Table 1 of the paper.

#[path = "shokri_tsoutsos_cxc/evaluator.rs"]
mod evaluator;
#[path = "shokri_tsoutsos_cxc/harness.rs"]
mod harness;
#[path = "shokri_tsoutsos_cxc/lookup.rs"]
mod lookup;
#[path = "shokri_tsoutsos_cxc/types.rs"]
mod types;

fn main() {
    harness::run();
}
