use crate::{
    automorphism::*, fourier_glwe_keyswitch::*, glwe_keyswitch::*, mod_switch::*,
    mod_switch_rev::*, utils::*,
};
use aligned_vec::ABox;
use std::collections::HashMap;
use tfhe::core_crypto::{fft_impl::fft64::c64, prelude::*};

pub fn trace_assign_with_glwe_preprocess<Scalar, ContMut>(
    glwe_in: &mut GlweCiphertext<ContMut>,
    auto_keys: &HashMap<usize, AutomorphKey<ABox<[c64]>>>,
) where
    Scalar: UnsignedTorus + Sync + Send,
    ContMut: ContainerMut<Element = Scalar>,
{
    glwe_preprocessing_assign(glwe_in);
    trace_partial_assign(glwe_in, auto_keys, 1);
}

pub fn revtrace_assign<Scalar, ContMut>(
    glwe_in: &mut GlweCiphertext<ContMut>,
    auto_keys: &HashMap<usize, AutomorphKey<ABox<[c64]>>>,
) where
    Scalar: UnsignedTorus + Sync + Send,
    ContMut: ContainerMut<Element = Scalar>,
{
    revtrace_partial_assign(glwe_in, auto_keys, 1);
}

pub fn revtrace_partial_assign<Scalar, Cont>(
    input: &mut GlweCiphertext<Cont>,
    auto_keys: &HashMap<usize, AutomorphKey<ABox<[c64]>>>,
    n: usize,
) where
    Scalar: UnsignedTorus,
    Cont: ContainerMut<Element = Scalar>,
{
    let glwe_size = input.glwe_size();
    let polynomial_size = input.polynomial_size();
    let ciphertext_modulus = input.ciphertext_modulus();

    assert!(polynomial_size.0 % n == 0);

    let mut buf =
        GlweCiphertextOwned::new(Scalar::ZERO, glwe_size, polynomial_size, ciphertext_modulus);
    let mut out: GlweCiphertext<Vec<Scalar>> =
        GlweCiphertext::new(Scalar::ZERO, glwe_size, polynomial_size, ciphertext_modulus);
    glwe_ciphertext_clone_from(&mut out, input);

    let log_polynomial_size = polynomial_size.0.ilog2() as usize;
    // glwe_preprocessing_assign(&mut out);
    // glwe_preprocessing_moddown_1bit(&mut out);
    let log_n = n.ilog2() as usize;
    for i in (1..=(log_polynomial_size - log_n)).rev() {
        let k = polynomial_size.0 / (1 << (i - 1)) + 1;
        let auto_key = auto_keys.get(&k).unwrap();
        glwe_preprocessing_moddown_1bit(&mut out);
        auto_key.auto(&mut buf, &out);
        glwe_ciphertext_add_assign(&mut out, &buf);
    }
    // glwe_ciphertext_clone_from(&mut buf, &out);
    // glwe_ciphertext_add_assign(&mut out, &buf);

    glwe_ciphertext_clone_from(input, &out);
}
