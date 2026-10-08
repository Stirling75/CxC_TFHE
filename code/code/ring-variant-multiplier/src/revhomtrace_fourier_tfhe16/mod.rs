//! Minimal RevHomTrace Fourier-GLWE keyswitch/automorphism layer ported to
//! TFHE-rs 1.6.1 core types.
//!
//! This module intentionally keeps the RevHomTrace structure visible: a
//! standard TFHE-rs GLWE keyswitch key is converted into a Fourier GLev list,
//! then used by the automorphism trace in the direct CBS lift.

pub mod automorphism;
pub mod fourier_glev_ciphertext;
pub mod fourier_glwe_ciphertext;
pub mod fourier_glwe_keyswitch;
pub mod fourier_poly_mult;

pub use automorphism::*;
pub use fourier_glev_ciphertext::*;
pub use fourier_glwe_ciphertext::*;
pub use fourier_glwe_keyswitch::*;
pub use fourier_poly_mult::*;
