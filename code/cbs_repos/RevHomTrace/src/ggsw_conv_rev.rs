use crate::{
    automorphism::*, automorphism_rev::*, ggsw_conv::*, glwe_conv::*, glwe_preprocessing_assign,
    keyswitch_glwe_ciphertext, pbs::*, utils::*, FourierGlweKeyswitchKey,
};
use aligned_vec::{ABox, CACHELINE_ALIGN};
use std::collections::HashMap;
use tfhe::core_crypto::{
    fft_impl::fft64::{
        c64,
        crypto::{bootstrap::FourierLweBootstrapKeyView, ggsw::FourierGgswCiphertextListView},
    },
    prelude::{polynomial_algorithms::*, *},
};

pub fn convert_to_ggsw_after_blind_rotate_revtrace<Scalar, InputCont, OutputCont>(
    glev_in: &GlweCiphertextList<InputCont>,
    ggsw_out: &mut GgswCiphertext<OutputCont>,
    bit_idx_from_msb: usize,
    auto_keys: &HashMap<usize, AutomorphKey<ABox<[c64]>>>,
    ss_key: FourierGgswCiphertextListView,
    ciphertext_modulus: CiphertextModulus<Scalar>,
) where
    Scalar: UnsignedTorus,
    InputCont: Container<Element = Scalar>,
    OutputCont: ContainerMut<Element = Scalar>,
{
    assert!(
        bit_idx_from_msb <= 2,
        "Multi-bit extraction is supported for at most 3 bits"
    );

    assert_eq!(glev_in.polynomial_size(), ggsw_out.polynomial_size());
    assert_eq!(glev_in.glwe_size(), ggsw_out.glwe_size());
    assert_eq!(glev_in.polynomial_size(), ss_key.polynomial_size());
    assert_eq!(glev_in.glwe_size(), ss_key.glwe_size());

    let glwe_size = glev_in.glwe_size();
    let polynomial_size = glev_in.polynomial_size();

    let cbs_level = ggsw_out.decomposition_level_count();
    let cbs_base_log = ggsw_out.decomposition_base_log();

    let large_lwe_dimension = LweDimension(glwe_size.to_glwe_dimension().0 * polynomial_size.0);
    let mut buf_lwe = LweCiphertext::new(
        Scalar::ZERO,
        large_lwe_dimension.to_lwe_size(),
        ciphertext_modulus,
    );

    let mut glev_out = GlweCiphertextList::new(
        Scalar::ZERO,
        glwe_size,
        polynomial_size,
        GlweCiphertextCount(cbs_level.0),
        ciphertext_modulus,
    );
    for (k, (mut glwe_out, glwe_in)) in glev_out.iter_mut().zip(glev_in.iter()).enumerate() {
        let cur_level = k + 1;
        let log_scale = Scalar::BITS - cur_level * cbs_base_log.0;

        if bit_idx_from_msb == 0 {
            extract_lwe_sample_from_glwe_ciphertext(&glwe_in, &mut buf_lwe, MonomialDegree(0));
            lwe_ciphertext_plaintext_add_assign(
                &mut buf_lwe,
                Plaintext(Scalar::ONE << (log_scale - 1)),
            );
        } else if bit_idx_from_msb == 1 {
            glwe_ciphertext_monic_monomial_mul(
                &mut glwe_out,
                &glwe_in,
                MonomialDegree(polynomial_size.0 / 2),
            );

            extract_lwe_sample_from_glwe_ciphertext(&glwe_out, &mut buf_lwe, MonomialDegree(0));
            lwe_ciphertext_opposite_assign(&mut buf_lwe);
            lwe_ciphertext_plaintext_add_assign(
                &mut buf_lwe,
                Plaintext(Scalar::ONE << (log_scale - 1)),
            );
        } else {
            // bit_idx_from_msb == 2
            glwe_ciphertext_monic_monomial_mul(
                &mut glwe_out,
                &glwe_in,
                MonomialDegree(polynomial_size.0 / 4),
            );

            let mut buf_glwe1 =
                GlweCiphertext::new(Scalar::ZERO, glwe_size, polynomial_size, ciphertext_modulus);
            let mut buf_glwe2 =
                GlweCiphertext::new(Scalar::ZERO, glwe_size, polynomial_size, ciphertext_modulus);

            glwe_ciphertext_monic_monomial_mul(
                &mut buf_glwe1,
                &glwe_out,
                MonomialDegree(polynomial_size.0 / 4),
            );
            glwe_ciphertext_monic_monomial_mul(
                &mut buf_glwe2,
                &glwe_out,
                MonomialDegree(polynomial_size.0 / 2),
            );

            glwe_ciphertext_sub_assign(&mut glwe_out, &buf_glwe1);
            glwe_ciphertext_add_assign(&mut glwe_out, &buf_glwe2);

            extract_lwe_sample_from_glwe_ciphertext(&glwe_out, &mut buf_lwe, MonomialDegree(0));
            lwe_ciphertext_opposite_assign(&mut buf_lwe);
            lwe_ciphertext_plaintext_add_assign(
                &mut buf_lwe,
                Plaintext(Scalar::ONE << (log_scale - 1)),
            );
        }

        // lwe_preprocessing_assign(&mut buf_lwe, polynomial_size);
        // lwe_preprocessing_moddown_1bit_assign(&mut buf_lwe);
        convert_lwe_to_glwe_const(&buf_lwe, &mut glwe_out);
        // trace_assign(&mut glwe_out, &auto_keys);
        revtrace_assign(&mut glwe_out, &auto_keys)
    }

    switch_scheme(&glev_out, ggsw_out, ss_key);
}
