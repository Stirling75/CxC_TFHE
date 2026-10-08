//! Experimental cached-table multiplier and its unchanged product baseline.
//!
//! The harness owns encryption, decryption, and the clear correctness oracle.
//! Evaluator modules receive only ciphertexts and public evaluation keys.

#[path = "hybrid_cxc/config.rs"]
mod config;
#[path = "hybrid_cxc/harness.rs"]
mod harness;
#[path = "hybrid_cxc/keys.rs"]
mod keys;
#[path = "hybrid_cxc/normalize.rs"]
mod normalize;
#[path = "hybrid_cxc/plan.rs"]
mod plan;
#[path = "hybrid_cxc/probe.rs"]
mod probe;
#[path = "hybrid_cxc/product.rs"]
mod product;
#[path = "hybrid_cxc/report.rs"]
mod report;
#[path = "hybrid_cxc/ring_key.rs"]
mod ring_key;
#[path = "hybrid_cxc/types.rs"]
mod types;

#[path = "../revhomtrace_fourier_tfhe16/mod.rs"]
mod revhomtrace_fourier_tfhe16;
#[path = "../revhomtrace_lift_tfhe16.rs"]
mod revhomtrace_lift_tfhe16;

fn main() -> anyhow::Result<()> {
    harness::run()
}
