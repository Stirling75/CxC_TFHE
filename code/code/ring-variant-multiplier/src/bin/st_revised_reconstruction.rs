//! Independent July-30 ST construction, not an author-code reproduction.
#[path = "st_revised/crypto.rs"]
mod crypto;
#[path = "st_revised/plan.rs"]
mod plan;
#[path = "st_revised/evaluate.rs"]
mod evaluate;
#[path = "st_revised/harness.rs"]
mod harness;
#[path = "st_revised/parameters.rs"]
mod parameters;
#[path = "../revhomtrace_fourier_tfhe16/mod.rs"]
mod revhomtrace_fourier_tfhe16;
#[path = "../revhomtrace_lift_tfhe16.rs"]
mod revhomtrace_lift_tfhe16;

fn main() -> anyhow::Result<()> { harness::run() }
