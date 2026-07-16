use crate::{
    automorphism::*, automorphism_rev::*, convert_lwe_to_glwe_const, glwe_conv::*,
    keyswitch_glwe_ciphertext, mod_switch::*, mod_switch_rev::*, utils::*, FourierGlweKeyswitchKey,
};
use aligned_vec::ABox;
use std::collections::HashMap;
use tfhe::core_crypto::{
    algorithms::slice_algorithms::slice_wrapping_opposite_assign, fft_impl::fft64::c64, prelude::*,
};

pub fn convert_lwe_to_glwe_by_revtrace<Scalar, InputCont, OutputCont>(
    input: &LweCiphertext<InputCont>,
    output: &mut GlweCiphertext<OutputCont>,
    auto_keys: &HashMap<usize, AutomorphKey<ABox<[c64]>>>,
) where
    Scalar: UnsignedTorus,
    InputCont: Container<Element = Scalar>,
    OutputCont: ContainerMut<Element = Scalar>,
{
    assert_eq!(input.ciphertext_modulus(), output.ciphertext_modulus());
    assert!(
        input.ciphertext_modulus().is_native_modulus(),
        "only native ciphertext modulus is supported"
    );

    let lwe_size = input.lwe_size();
    let lwe_dimension = lwe_size.to_lwe_dimension();
    let glwe_size = output.glwe_size();
    let glwe_dimension = glwe_size.to_glwe_dimension();
    let polynomial_size = output.polynomial_size();
    let ciphertext_modulus = input.ciphertext_modulus();

    assert_eq!(lwe_dimension.0, glwe_dimension.0 * polynomial_size.0);

    // Pre-processing
    // let mut buf = LweCiphertext::new(Scalar::ZERO, lwe_size, ciphertext_modulus);
    // lwe_preprocessing(input, &mut buf, polynomial_size);

    // LWEtoGLWEConst
    convert_lwe_to_glwe_const(&input, output);

    // Clear coefficients except the constant
    revtrace_assign(output, auto_keys);
}

pub fn convert_lwe_to_glwe_by_revtrace_high_prec<Scalar, InputCont, OutputCont>(
    input: &LweCiphertext<InputCont>,
    output: &mut GlweCiphertext<OutputCont>,
    glwe_ksk_to_large: &FourierGlweKeyswitchKey<ABox<[c64]>>,
    glwe_ksk_from_large: &FourierGlweKeyswitchKey<ABox<[c64]>>,
    auto_keys: &HashMap<usize, AutomorphKey<ABox<[c64]>>>,
) where
    Scalar: UnsignedTorus,
    InputCont: Container<Element = Scalar>,
    OutputCont: ContainerMut<Element = Scalar>,
{
    assert_eq!(input.ciphertext_modulus(), output.ciphertext_modulus());
    assert!(
        input.ciphertext_modulus().is_native_modulus(),
        "only native ciphertext modulus is supported"
    );
    assert_eq!(
        glwe_ksk_to_large.input_glwe_size(),
        glwe_ksk_from_large.output_glwe_size()
    );
    assert_eq!(
        glwe_ksk_to_large.output_glwe_size(),
        glwe_ksk_from_large.input_glwe_size()
    );

    let lwe_size = input.lwe_size();
    let lwe_dimension = lwe_size.to_lwe_dimension();
    let glwe_size = output.glwe_size();
    let glwe_dimension = glwe_size.to_glwe_dimension();
    let large_glwe_size = glwe_ksk_to_large.output_glwe_size();
    let polynomial_size = output.polynomial_size();
    let ciphertext_modulus = input.ciphertext_modulus();

    assert_eq!(lwe_dimension.0, glwe_dimension.0 * polynomial_size.0);

    // LWEtoGLWEConst
    let mut buf = GlweCiphertext::new(Scalar::ZERO, glwe_size, polynomial_size, ciphertext_modulus);
    convert_lwe_to_glwe_const(&input, &mut buf);

    // GLWE KS to Large
    let mut buf_large = GlweCiphertext::new(
        Scalar::ZERO,
        large_glwe_size,
        polynomial_size,
        ciphertext_modulus,
    );
    keyswitch_glwe_ciphertext(glwe_ksk_to_large, &buf, &mut buf_large);

    // Pre-processing
    // glwe_preprocessing_assign(&mut buf_large);

    // Clear coefficients except the constant
    revtrace_assign(&mut buf_large, auto_keys);

    // GLWE KS from Large
    keyswitch_glwe_ciphertext(glwe_ksk_from_large, &buf_large, output);
}

pub fn convert_lwes_to_glwe_by_revtrace<Scalar, InputCont, OutputCont>(
    input: &LweCiphertextList<InputCont>,
    output: &mut GlweCiphertext<OutputCont>,
    auto_keys: &HashMap<usize, AutomorphKey<ABox<[c64]>>>,
) where
    Scalar: UnsignedTorus,
    InputCont: Container<Element = Scalar>,
    OutputCont: ContainerMut<Element = Scalar>,
{
    assert_eq!(input.ciphertext_modulus(), output.ciphertext_modulus());
    assert!(
        input.ciphertext_modulus().is_native_modulus(),
        "only native ciphertext modulus is supported"
    );

    let lwe_size = input.lwe_size();
    let lwe_dimension = lwe_size.to_lwe_dimension();
    let glwe_size = output.glwe_size();
    let glwe_dimension = glwe_size.to_glwe_dimension();
    let polynomial_size = output.polynomial_size();
    let ciphertext_modulus = input.ciphertext_modulus();

    assert_eq!(lwe_dimension.0, glwe_dimension.0 * polynomial_size.0);

    let lwe_count = input.lwe_ciphertext_count().0;
    // let mut buf = LweCiphertext::new(Scalar::ZERO, lwe_size, ciphertext_modulus);
    let mut input_glwes = GlweCiphertextList::new(
        Scalar::ZERO,
        glwe_size,
        polynomial_size,
        GlweCiphertextCount(lwe_count),
        ciphertext_modulus,
    );

    for (input_lwe, mut input_glwe) in input.iter().zip(input_glwes.iter_mut()) {
        // lwe_preprocessing(&input_lwe, &mut buf, polynomial_size);
        convert_lwe_to_glwe_const(&input_lwe, &mut input_glwe);
    }

    let mut buf = pack_lwes_rev(&input_glwes, auto_keys);
    revtrace_partial_assign(&mut buf, auto_keys, lwe_count);
    glwe_ciphertext_clone_from(output, &buf);
}

fn pack_lwes_rev<Scalar, Cont>(
    input: &GlweCiphertextList<Cont>,
    auto_keys: &HashMap<usize, AutomorphKey<ABox<[c64]>>>,
) -> GlweCiphertextOwned<Scalar>
where
    Scalar: UnsignedTorus,
    Cont: Container<Element = Scalar>,
{
    let glwe_size = input.glwe_size();
    let polynomial_size = input.polynomial_size();
    let ciphertext_modulus = input.ciphertext_modulus();

    let mut output =
        GlweCiphertext::new(Scalar::ZERO, glwe_size, polynomial_size, ciphertext_modulus);

    let lwe_count = input.glwe_ciphertext_count().0;
    if lwe_count == 1 {
        glwe_ciphertext_clone_from(&mut output, &input.get(0));
    } else {
        assert_eq!(lwe_count % 2, 0);

        let half_lwe_count = lwe_count / 2;
        let mut input_even = GlweCiphertextList::new(
            Scalar::ZERO,
            glwe_size,
            polynomial_size,
            GlweCiphertextCount(half_lwe_count),
            ciphertext_modulus,
        );
        let mut input_odd = GlweCiphertextList::new(
            Scalar::ZERO,
            glwe_size,
            polynomial_size,
            GlweCiphertextCount(half_lwe_count),
            ciphertext_modulus,
        );

        for (i, (mut lwe_even, mut lwe_odd)) in
            input_even.iter_mut().zip(input_odd.iter_mut()).enumerate()
        {
            glwe_ciphertext_clone_from(&mut lwe_even, &input.get(2 * i));
            glwe_ciphertext_clone_from(&mut lwe_odd, &input.get(2 * i + 1));
        }

        let output_even = pack_lwes_rev(&input_even, auto_keys);
        let output_odd = pack_lwes_rev(&input_odd, auto_keys);

        // let mut buf = GlweCiphertext::new(Scalar::ZERO, glwe_size, polynomial_size, ciphertext_modulus);
        let mut buf_even =
            GlweCiphertext::new(Scalar::ZERO, glwe_size, polynomial_size, ciphertext_modulus);
        let mut buf_odd =
            GlweCiphertext::new(Scalar::ZERO, glwe_size, polynomial_size, ciphertext_modulus);
        glwe_ciphertext_clone_from(&mut buf_odd, &output_odd);
        glwe_ciphertext_monic_monomial_mul_assign(
            &mut buf_odd,
            MonomialDegree(polynomial_size.0 / lwe_count),
        );
        glwe_ciphertext_clone_from(&mut buf_even, &output_even);
        glwe_ciphertext_sub_assign(&mut buf_even, &buf_odd);
        glwe_preprocessing_moddown_1bit(&mut buf_even);
        // glwe_preprocessing_moddown_1bit(&mut buf_odd);
        // glwe_preprocessing_moddown_1bit(&mut buf_even);
        let auto_key = auto_keys.get(&(lwe_count + 1)).unwrap();
        auto_key.auto(&mut output, &buf_even);
        // auto_key.auto(&mut buf, &buf_odd);
        glwe_ciphertext_add_assign(&mut output, &buf_even);
        glwe_ciphertext_add_assign(&mut output, &buf_odd);
    }

    output
}

pub fn convert_lwes_to_glwe_by_trace_with_preprocessing_high_prec<Scalar, InputCont, OutputCont>(
    input: &LweCiphertextList<InputCont>,
    output: &mut GlweCiphertext<OutputCont>,
    glwe_ksk_to_large: &FourierGlweKeyswitchKey<ABox<[c64]>>,
    glwe_ksk_from_large: &FourierGlweKeyswitchKey<ABox<[c64]>>,
    auto_keys: &HashMap<usize, AutomorphKey<ABox<[c64]>>>,
) where
    Scalar: UnsignedTorus,
    InputCont: Container<Element = Scalar>,
    OutputCont: ContainerMut<Element = Scalar>,
{
    assert_eq!(input.ciphertext_modulus(), output.ciphertext_modulus());
    assert!(
        input.ciphertext_modulus().is_native_modulus(),
        "only native ciphertext modulus is supported"
    );

    assert_eq!(
        glwe_ksk_to_large.input_glwe_size(),
        glwe_ksk_from_large.output_glwe_size()
    );
    assert_eq!(
        glwe_ksk_to_large.output_glwe_size(),
        glwe_ksk_from_large.input_glwe_size()
    );

    let lwe_size = input.lwe_size();
    let lwe_dimension = lwe_size.to_lwe_dimension();
    let glwe_size = output.glwe_size();
    let glwe_dimension = glwe_size.to_glwe_dimension();
    let large_glwe_size = glwe_ksk_to_large.output_glwe_size();
    let polynomial_size = output.polynomial_size();
    let ciphertext_modulus = input.ciphertext_modulus();

    assert_eq!(lwe_dimension.0, glwe_dimension.0 * polynomial_size.0);

    let lwe_count = input.lwe_ciphertext_count().0;
    let mut buf = LweCiphertext::new(Scalar::ZERO, lwe_size, ciphertext_modulus);
    let mut inter_glwe =
        GlweCiphertext::new(Scalar::ZERO, glwe_size, polynomial_size, ciphertext_modulus);
    let mut large_input_glwes = GlweCiphertextList::new(
        Scalar::ZERO,
        large_glwe_size,
        polynomial_size,
        GlweCiphertextCount(lwe_count),
        ciphertext_modulus,
    );
    for (input_lwe, mut large_input_glwe) in input.iter().zip(large_input_glwes.iter_mut()) {
        // lwe_preprocessing(&input_lwe, &mut buf, polynomial_size);
        convert_lwe_to_glwe_const(&input_lwe, &mut inter_glwe);
        // glwe_preprocessing_assign(&mut inter_glwe);
        keyswitch_glwe_ciphertext(glwe_ksk_to_large, &inter_glwe, &mut large_input_glwe);
        glwe_preprocessing_assign(&mut large_input_glwe);
    }

    let mut buf = pack_lwes(&large_input_glwes, auto_keys);
    trace_partial_assign(&mut buf, auto_keys, lwe_count);
    keyswitch_glwe_ciphertext(glwe_ksk_from_large, &buf, output);
    // glwe_ciphertext_clone_from(output, &buf);
}

fn pack_lwes<Scalar, Cont>(
    input: &GlweCiphertextList<Cont>,
    auto_keys: &HashMap<usize, AutomorphKey<ABox<[c64]>>>,
) -> GlweCiphertextOwned<Scalar>
where
    Scalar: UnsignedTorus,
    Cont: Container<Element = Scalar>,
{
    let glwe_size = input.glwe_size();
    let polynomial_size = input.polynomial_size();
    let ciphertext_modulus = input.ciphertext_modulus();

    let mut output =
        GlweCiphertext::new(Scalar::ZERO, glwe_size, polynomial_size, ciphertext_modulus);

    let lwe_count = input.glwe_ciphertext_count().0;
    if lwe_count == 1 {
        glwe_ciphertext_clone_from(&mut output, &input.get(0));
    } else {
        assert_eq!(lwe_count % 2, 0);

        let half_lwe_count = lwe_count / 2;
        let mut input_even = GlweCiphertextList::new(
            Scalar::ZERO,
            glwe_size,
            polynomial_size,
            GlweCiphertextCount(half_lwe_count),
            ciphertext_modulus,
        );
        let mut input_odd = GlweCiphertextList::new(
            Scalar::ZERO,
            glwe_size,
            polynomial_size,
            GlweCiphertextCount(half_lwe_count),
            ciphertext_modulus,
        );

        for (i, (mut lwe_even, mut lwe_odd)) in
            input_even.iter_mut().zip(input_odd.iter_mut()).enumerate()
        {
            glwe_ciphertext_clone_from(&mut lwe_even, &input.get(2 * i));
            glwe_ciphertext_clone_from(&mut lwe_odd, &input.get(2 * i + 1));
        }

        let output_even = pack_lwes(&input_even, auto_keys);
        let output_odd = pack_lwes(&input_odd, auto_keys);

        let mut buf =
            GlweCiphertext::new(Scalar::ZERO, glwe_size, polynomial_size, ciphertext_modulus);
        glwe_ciphertext_sub_assign(&mut buf, &output_odd);
        glwe_ciphertext_monic_monomial_mul_assign(
            &mut buf,
            MonomialDegree(polynomial_size.0 / lwe_count),
        );
        glwe_ciphertext_add_assign(&mut buf, &output_even);
        let auto_key = auto_keys.get(&(lwe_count + 1)).unwrap();
        auto_key.auto(&mut output, &buf);

        glwe_ciphertext_clone_from(&mut buf, &output_odd);
        glwe_ciphertext_monic_monomial_mul_assign(
            &mut buf,
            MonomialDegree(polynomial_size.0 / lwe_count),
        );
        glwe_ciphertext_add_assign(&mut buf, &output_even);

        glwe_ciphertext_add_assign(&mut output, &buf);
    }

    output
}
