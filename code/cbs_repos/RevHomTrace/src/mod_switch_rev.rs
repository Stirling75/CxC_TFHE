use crate::mod_switch::*;
use tfhe::core_crypto::prelude::*;
pub fn glwe_preprocessing_given<Scalar, ContMut>(
    input: &mut GlweCiphertext<ContMut>,
    moddown_size: PolynomialSize,
) where
    Scalar: UnsignedInteger,
    ContMut: ContainerMut<Element = Scalar>,
{
    assert!(
        input.ciphertext_modulus().is_native_modulus(),
        "input ciphertext modulus is not native"
    );

    let polynomial_size = input.polynomial_size();

    assert!(Scalar::BITS > moddown_size.0.ilog2() as usize);

    let log_small_q = Scalar::BITS - moddown_size.0.ilog2() as usize;
    let small_ciphertext_modulus =
        CiphertextModulus::<Scalar>::try_new_power_of_2(log_small_q).unwrap();
    let glwe_size = input.glwe_size();

    let mut buf = GlweCiphertext::new(
        Scalar::ZERO,
        glwe_size,
        polynomial_size,
        small_ciphertext_modulus,
    );

    glwe_ciphertext_mod_switch_from_native_to_non_native_power_of_two(&input, &mut buf);
    glwe_ciphertext_mod_raise_from_non_native_power_of_two_to_native(&buf, input);
}

pub fn glwe_preprocessing_moddown_1bit<Scalar, ContMut>(input: &mut GlweCiphertext<ContMut>)
where
    Scalar: UnsignedInteger,
    ContMut: ContainerMut<Element = Scalar>,
{
    assert!(
        input.ciphertext_modulus().is_native_modulus(),
        "input ciphertext modulus is not native"
    );

    let polynomial_size = input.polynomial_size();

    // assert!(
    //     Scalar::BITS > polynomial_size.0.ilog2() as usize
    // );

    let log_small_q = Scalar::BITS - 1 as usize;
    let small_ciphertext_modulus =
        CiphertextModulus::<Scalar>::try_new_power_of_2(log_small_q).unwrap();
    let glwe_size = input.glwe_size();
    // println!("BITS : {}, log_small_q : {}, small_cipher_mod : {}", Scalar::BITS, log_small_q, small_ciphertext_modulus);

    let mut buf = GlweCiphertext::new(
        Scalar::ZERO,
        glwe_size,
        polynomial_size,
        small_ciphertext_modulus,
    );

    glwe_ciphertext_mod_switch_from_native_to_non_native_power_of_two(&input, &mut buf);
    glwe_ciphertext_mod_raise_from_non_native_power_of_two_to_native(&buf, input);
}
