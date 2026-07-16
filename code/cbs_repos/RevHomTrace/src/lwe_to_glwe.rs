// use tfhe::core_crypto::algorithms::polynomial_algorithms::{polynomial_list_wrapping_sub_mul_assign};
use tfhe::core_crypto::algorithms::slice_algorithms::*;
use tfhe::core_crypto::commons::math::decomposition::SignedDecomposer;
use tfhe::core_crypto::commons::parameters::*;
use tfhe::core_crypto::commons::traits::*;
use tfhe::core_crypto::entities::*;
// use rayon::prelude::*;
use crate::{encrypt_glev_ciphertext, GlevCiphertextList};
use itertools::Itertools;
use tfhe::core_crypto::commons::traits::*;
use tfhe::core_crypto::entities::*;
use tfhe::core_crypto::{
    algorithms::slice_algorithms::slice_wrapping_opposite_assign,
    prelude::{polynomial_algorithms::*, *},
};

pub fn polynomial_list_wrapping_sub_mul_assign<Scalar, InputCont, OutputCont, PolyCont>(
    output_poly_list: &mut PolynomialList<OutputCont>,
    input_poly_list: &PolynomialList<InputCont>,
    scalar_poly: &Polynomial<PolyCont>,
) where
    Scalar: UnsignedInteger,
    OutputCont: ContainerMut<Element = Scalar>,
    InputCont: Container<Element = Scalar>,
    PolyCont: Container<Element = Scalar>,
{
    assert_eq!(
        output_poly_list.polynomial_size(),
        input_poly_list.polynomial_size()
    );
    assert_eq!(
        output_poly_list.polynomial_count(),
        input_poly_list.polynomial_count()
    );
    for (mut output_poly, input_poly) in output_poly_list.iter_mut().zip_eq(input_poly_list.iter())
    {
        polynomial_wrapping_sub_mul_assign(&mut output_poly, &input_poly, scalar_poly)
    }
}

pub fn polynomial_list_wrapping_add_mul_assign<Scalar, InputCont, OutputCont, PolyCont>(
    output_poly_list: &mut PolynomialList<OutputCont>,
    input_poly_list: &PolynomialList<InputCont>,
    scalar_poly: &Polynomial<PolyCont>,
) where
    Scalar: UnsignedInteger,
    OutputCont: ContainerMut<Element = Scalar>,
    InputCont: Container<Element = Scalar>,
    PolyCont: Container<Element = Scalar>,
{
    assert_eq!(
        output_poly_list.polynomial_size(),
        input_poly_list.polynomial_size()
    );
    assert_eq!(
        output_poly_list.polynomial_count(),
        input_poly_list.polynomial_count()
    );
    for (mut output_poly, input_poly) in output_poly_list.iter_mut().zip_eq(input_poly_list.iter())
    {
        polynomial_wrapping_add_mul_assign(&mut output_poly, &input_poly, scalar_poly)
    }
}

pub fn keyswitch_lwe_ciphertext_list_and_pack_in_glwe_ciphertext_ms18<
    Scalar,
    KeyCont,
    InputCont,
    OutputCont,
>(
    lwe_pksk: &LwePackingKeyswitchKey<KeyCont>,
    input_lwe_ciphertext: &LweCiphertextList<InputCont>,
    output_glwe_ciphertext: &mut GlweCiphertext<OutputCont>,
) where
    Scalar: UnsignedInteger,
    KeyCont: Container<Element = Scalar>,
    InputCont: Container<Element = Scalar>,
    OutputCont: ContainerMut<Element = Scalar>,
{
    assert!(
        lwe_pksk.input_key_lwe_dimension() == input_lwe_ciphertext.lwe_size().to_lwe_dimension(),
        "Mismatched input LweDimension. \
        LwePackingKeyswitchKey input LweDimension: {:?}, input LweCiphertext LweDimension {:?}.",
        lwe_pksk.input_key_lwe_dimension(),
        input_lwe_ciphertext.lwe_size().to_lwe_dimension()
    );
    assert!(
        lwe_pksk.output_key_glwe_dimension()
            == output_glwe_ciphertext.glwe_size().to_glwe_dimension(),
        "Mismatched output GlweDimension. \
        LwePackingKeyswitchKey output GlweDimension: {:?}, \
        output GlweCiphertext GlweDimension {:?}.",
        lwe_pksk.output_key_glwe_dimension(),
        output_glwe_ciphertext.glwe_size().to_glwe_dimension()
    );
    assert!(
        lwe_pksk.output_key_polynomial_size() == output_glwe_ciphertext.polynomial_size(),
        "Mismatched output PolynomialSize. \
        LwePackingKeyswitchKey output PolynomialSize: {:?}, \
        output GlweCiphertext PolynomialSize {:?}.",
        lwe_pksk.output_key_polynomial_size(),
        output_glwe_ciphertext.polynomial_size()
    );

    assert!(
        lwe_pksk.ciphertext_modulus() == input_lwe_ciphertext.ciphertext_modulus(),
        "Mismatched CiphertextModulus. \
        LwePackingKeyswitchKey CiphertextModulus: {:?}, input LweCiphertext CiphertextModulus {:?}.",
        lwe_pksk.ciphertext_modulus(),
        input_lwe_ciphertext.ciphertext_modulus()
    );
    assert!(
        lwe_pksk.ciphertext_modulus() == output_glwe_ciphertext.ciphertext_modulus(),
        "Mismatched CiphertextModulus. \
        LwePackingKeyswitchKey CiphertextModulus: {:?}, \
        output LweCiphertext CiphertextModulus {:?}.",
        lwe_pksk.ciphertext_modulus(),
        output_glwe_ciphertext.ciphertext_modulus()
    );
    assert!(
        input_lwe_ciphertext
            .ciphertext_modulus()
            .is_compatible_with_native_modulus(),
        "This operation currently only supports power of 2 moduli"
    );

    assert!(
        input_lwe_ciphertext.lwe_ciphertext_count().0 <= output_glwe_ciphertext.polynomial_size().0
    );

    output_glwe_ciphertext.as_mut().fill(Scalar::ZERO);
    let mut buffer = GlweCiphertext::new(
        Scalar::ZERO,
        output_glwe_ciphertext.glwe_size(),
        output_glwe_ciphertext.polynomial_size(),
        output_glwe_ciphertext.ciphertext_modulus(),
    );
    let m = input_lwe_ciphertext.lwe_ciphertext_count().0 as usize;
    let poly_size = lwe_pksk.output_key_polynomial_size().0 as usize;
    assert!(m <= poly_size);

    // 2. Clear output
    // output_glwe_ciphertext.as_mut().fill(Scalar::ZERO);

    // 3. Prepare buffer and decomposer
    let mut buffer = GlweCiphertext::new(
        Scalar::ZERO,
        output_glwe_ciphertext.glwe_size(),
        output_glwe_ciphertext.polynomial_size(),
        output_glwe_ciphertext.ciphertext_modulus(),
    );
    buffer.as_mut().fill(Scalar::ZERO);
    let decomposer = SignedDecomposer::new(
        lwe_pksk.decomposition_base_log(),
        lwe_pksk.decomposition_level_count(),
    );
    let levels = lwe_pksk.decomposition_level_count().0 as usize;
    // let glev_list = lwe_pksk.as_polynomial_list();

    for (j, lwe_ct) in input_lwe_ciphertext.iter().enumerate() {
        // buffer.get_mut_body().as_mut()[j] = *lwe_ct.get_body().data;
        *output_glwe_ciphertext
            .get_mut_body()
            .as_mut()
            .get_mut(j)
            .unwrap() = *lwe_ct.get_body().data;
    }

    // 4. For each LWE ciphertext j
    for (j, keyswitch_key_block) in lwe_pksk.iter().enumerate() {
        // a) Reset buffer
        // buffer.as_mut().fill(Scalar::ZERO);

        // b) Write body term b_j
        // let lwe_ct = input_lwe_ciphertext.get_ciphertext(j);
        // buffer.get_mut_body().as_mut()[0] = *lwe_ct.get_body().data;

        // c) Build decomposition polynomial-list for mask coefficients a_{j,i}
        let decomp_level_count = lwe_pksk.decomposition_level_count().0 as usize;
        let mut decomp_poly_list = tfhe::core_crypto::entities::PolynomialList::new(
            Scalar::ZERO,
            lwe_pksk.output_key_polynomial_size(),
            PolynomialCount(levels),
        );
        for (i, lwe_ct) in input_lwe_ciphertext.iter().enumerate() {
            let a_ij = lwe_ct.get_mask().as_ref()[j];
            let decomposition = decomposer.decompose(a_ij);
            for (lvl, piece) in decomposition.into_iter().enumerate() {
                *decomp_poly_list.get_mut(lvl).as_mut().get_mut(i).unwrap() = piece.value();
            }
        }

        // d) For each level ℓ, multiply decomp_poly_list[ℓ] × GLev(ℓ) and accumulate
        // for (l, decomp_poly) in decomp_poly_list
        //     .iter().enumerate()
        // {
        let mut tmp = GlweCiphertext::new(
            Scalar::ZERO,
            buffer.glwe_size(),
            buffer.polynomial_size(),
            buffer.ciphertext_modulus(),
        );
        // let mut keyswitch_key_block_iter = keyswitch_key_block.iter();

        for (level_key_ciphertext, decomp_poly) in
            keyswitch_key_block.iter().zip(decomp_poly_list.iter())
        {
            let mut tmp = GlweCiphertext::new(
                Scalar::ZERO,
                output_glwe_ciphertext.glwe_size(),
                output_glwe_ciphertext.polynomial_size(),
                output_glwe_ciphertext.ciphertext_modulus(),
            );

            polynomial_list_wrapping_add_mul_assign(
                &mut tmp.as_mut_polynomial_list(),
                &level_key_ciphertext.as_polynomial_list(),
                &decomp_poly,
            );

            glwe_ciphertext_sub_assign(output_glwe_ciphertext, &tmp);
        }
        // for decomp_poly in decomp_poly_list.iter()
        // {
        //     let mut tmp = GlweCiphertext::new(
        //         Scalar::ZERO,
        //         buffer.glwe_size(),
        //         buffer.polynomial_size(),
        //         buffer.ciphertext_modulus(),
        //     );
        //     for level_key_ciphertext in keyswitch_key_block.iter() {
        //         polynomial_list_wrapping_add_mul_assign(
        //             &mut tmp.as_mut_polynomial_list(),
        //             &level_key_ciphertext.as_polynomial_list(),
        //             &decomp_poly
        //         );
        //     }
        //     glwe_ciphertext_sub_assign(output_glwe_ciphertext, &tmp);
        // }
    }
}
