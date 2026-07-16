//! TFHE-rs 1.6.1 support code for RevHomTrace-style CBS lifting.
//!
//! RevHomTrace was originally written against older TFHE-rs/core-crypto types.
//! This module keeps the same mathematical route, but rewrites the helper layer
//! with TFHE-rs 1.6.1 containers:
//!
//! ```text
//! LWE digit -> identity-LUT GLev -> scheme switch -> GGSW digit
//! ```
//!
//! It also exposes the Refined/RevHomTrace multi-bit WOPBS conversion shape:
//!
//! ```text
//! LWE -> blind-rotate MSB accumulator -> per-bit GGSW selectors -> CMUX update
//! ```
//!
//! Secret keys are intentionally not part of these evaluator routines.

#![allow(dead_code)]

use std::{collections::HashMap, env};

use aligned_vec::CACHELINE_ALIGN;
use dyn_stack::{PodBuffer, PodStack};
use tfhe::core_crypto::algorithms::polynomial_algorithms::*;
use tfhe::core_crypto::algorithms::slice_algorithms::slice_wrapping_opposite_assign;
use tfhe::core_crypto::commons::math::random::{Distribution, Uniform};
use tfhe::core_crypto::fft_impl::fft64::crypto::{
    bootstrap::{self as fft_bootstrap, FourierLweBootstrapKeyView},
    ggsw::{
        add_external_product_assign, add_external_product_assign_scratch, FourierGgswCiphertext,
        FourierGgswCiphertextList, FourierGgswCiphertextListView,
    },
};
use tfhe::core_crypto::fft_impl::fft64::math::fft::Fft;
use tfhe::core_crypto::prelude::*;

use crate::revhomtrace_fourier_tfhe16;

pub type FourierGgswListOwned = FourierGgswCiphertextList<Vec<tfhe_fft::c64>>;
pub type FourierGgswOwned = FourierGgswCiphertext<Vec<tfhe_fft::c64>>;

pub struct StandardAutomorphismKey {
    ksk: GlweKeyswitchKeyOwned<u64>,
    auto_k: usize,
}

impl StandardAutomorphismKey {
    pub fn generate<NoiseDistribution, Gen>(
        glwe_secret_key: &GlweSecretKeyOwned<u64>,
        auto_k: usize,
        decomp_base_log: DecompositionBaseLog,
        decomp_level_count: DecompositionLevelCount,
        noise_distribution: NoiseDistribution,
        ciphertext_modulus: CiphertextModulus<u64>,
        generator: &mut EncryptionRandomGenerator<Gen>,
    ) -> Self
    where
        NoiseDistribution: Distribution,
        u64: Encryptable<Uniform, NoiseDistribution>,
        Gen: ByteRandomGenerator,
    {
        let mut before_poly_list = PolynomialList::new(
            0u64,
            glwe_secret_key.polynomial_size(),
            PolynomialCount(glwe_secret_key.glwe_dimension().0),
        );

        for (mut before_poly, after_poly) in before_poly_list
            .iter_mut()
            .zip(glwe_secret_key.as_polynomial_list().iter())
        {
            eval_x_k_into(&mut before_poly, &after_poly, auto_k);
        }

        let before_key = GlweSecretKey::from_container(
            before_poly_list.into_container(),
            glwe_secret_key.polynomial_size(),
        );
        let ksk = allocate_and_generate_new_glwe_keyswitch_key(
            &before_key,
            glwe_secret_key,
            decomp_base_log,
            decomp_level_count,
            noise_distribution,
            ciphertext_modulus,
            generator,
        );

        Self { ksk, auto_k }
    }

    pub fn apply<InputCont, OutputCont>(
        &self,
        output: &mut GlweCiphertext<OutputCont>,
        input: &GlweCiphertext<InputCont>,
    ) where
        InputCont: Container<Element = u64>,
        OutputCont: ContainerMut<Element = u64>,
    {
        let mut powered = GlweCiphertext::new(
            0u64,
            input.glwe_size(),
            input.polynomial_size(),
            input.ciphertext_modulus(),
        );

        for (mut powered_poly, input_poly) in powered
            .as_mut_polynomial_list()
            .iter_mut()
            .zip(input.as_polynomial_list().iter())
        {
            eval_x_k_into(&mut powered_poly, &input_poly, self.auto_k);
        }

        keyswitch_glwe_ciphertext(&self.ksk, &powered, output);
    }
}

pub fn generate_standard_automorphism_keys<NoiseDistribution, Gen>(
    glwe_secret_key: &GlweSecretKeyOwned<u64>,
    decomp_base_log: DecompositionBaseLog,
    decomp_level_count: DecompositionLevelCount,
    noise_distribution: NoiseDistribution,
    ciphertext_modulus: CiphertextModulus<u64>,
    generator: &mut EncryptionRandomGenerator<Gen>,
) -> HashMap<usize, StandardAutomorphismKey>
where
    NoiseDistribution: Distribution + Clone,
    u64: Encryptable<Uniform, NoiseDistribution>,
    Gen: ByteRandomGenerator,
{
    let n = glwe_secret_key.polynomial_size().0;
    let mut out = HashMap::new();
    for i in 1..=n.ilog2() as usize {
        let k = n / (1usize << (i - 1)) + 1;
        out.insert(
            k,
            StandardAutomorphismKey::generate(
                glwe_secret_key,
                k,
                decomp_base_log,
                decomp_level_count,
                noise_distribution.clone(),
                ciphertext_modulus,
                generator,
            ),
        );
    }
    out
}

pub fn revtrace_assign_standard<ContMut>(
    input: &mut GlweCiphertext<ContMut>,
    auto_keys: &HashMap<usize, StandardAutomorphismKey>,
) where
    ContMut: ContainerMut<Element = u64>,
{
    let glwe_size = input.glwe_size();
    let polynomial_size = input.polynomial_size();
    let ciphertext_modulus = input.ciphertext_modulus();

    let mut buf = GlweCiphertext::new(0u64, glwe_size, polynomial_size, ciphertext_modulus);
    let mut out = GlweCiphertext::new(0u64, glwe_size, polynomial_size, ciphertext_modulus);
    clone_glwe(&mut out, input);

    let log_n = polynomial_size.0.ilog2() as usize;
    for i in (1..=log_n).rev() {
        let k = polynomial_size.0 / (1usize << (i - 1)) + 1;
        glwe_preprocessing_moddown_1bit(&mut out);
        auto_keys
            .get(&k)
            .expect("missing RevHomTrace automorphism key")
            .apply(&mut buf, &out);
        glwe_ciphertext_add_assign(&mut out, &buf);
    }

    clone_glwe(input, &out);
}

pub fn generate_scheme_switching_key_standard<NoiseDistribution, Gen>(
    glwe_secret_key: &GlweSecretKeyOwned<u64>,
    ss_base_log: DecompositionBaseLog,
    ss_level: DecompositionLevelCount,
    noise_distribution: NoiseDistribution,
    ciphertext_modulus: CiphertextModulus<u64>,
    generator: &mut EncryptionRandomGenerator<Gen>,
) -> FourierGgswListOwned
where
    NoiseDistribution: Distribution + Clone,
    u64: Encryptable<Uniform, NoiseDistribution>,
    Gen: ByteRandomGenerator,
{
    let glwe_dimension = glwe_secret_key.glwe_dimension();
    let glwe_size = glwe_dimension.to_glwe_size();
    let polynomial_size = glwe_secret_key.polynomial_size();

    let mut ggsw_key = GgswCiphertextList::new(
        0u64,
        glwe_size,
        polynomial_size,
        ss_base_log,
        ss_level,
        GgswCiphertextCount(glwe_dimension.0),
        ciphertext_modulus,
    );

    for mut ggsw in ggsw_key.iter_mut() {
        encrypt_constant_ggsw_ciphertext(
            glwe_secret_key,
            &mut ggsw,
            Cleartext(0u64),
            noise_distribution.clone(),
            generator,
        );
    }

    let sk_poly_list = glwe_secret_key.as_polynomial_list();
    for (i, mut ggsw) in ggsw_key.iter_mut().enumerate() {
        let sk_poly = sk_poly_list.get(i);
        for (row, mut glwe) in ggsw.as_mut_glwe_list().iter_mut().enumerate() {
            let level_matrix = row / glwe_size.0;
            let decomp_level = ss_level.0 - level_matrix;
            let log_scale = u64::BITS as usize - decomp_level * ss_base_log.0;

            let mut buf = Polynomial::new(0u64, polynomial_size);
            for (elem, sk) in buf.iter_mut().zip(sk_poly.iter()) {
                *elem = sk.wrapping_neg() << log_scale;
            }

            let col = row % glwe_size.0;
            if col < glwe_dimension.0 {
                let mut mask = glwe.get_mut_mask();
                let mut mask_poly_list = mask.as_mut_polynomial_list();
                let mut mask_poly = mask_poly_list.get_mut(col);
                polynomial_wrapping_add_assign(&mut mask_poly, &buf);
            } else {
                let mut body = glwe.get_mut_body();
                let mut body_poly = body.as_mut_polynomial();
                polynomial_wrapping_add_assign(&mut body_poly, &buf);
            }
        }
    }

    let fourier_len = glwe_dimension.0
        * polynomial_size.to_fourier_polynomial_size().0
        * glwe_size.0
        * glwe_size.0
        * ss_level.0;
    let mut fourier_key = FourierGgswCiphertextList::new(
        vec![tfhe_fft::c64::default(); fourier_len],
        glwe_dimension.0,
        glwe_size,
        polynomial_size,
        ss_base_log,
        ss_level,
    );

    for (mut fourier_ggsw, ggsw) in fourier_key
        .as_mut_view()
        .into_ggsw_iter()
        .zip(ggsw_key.iter())
    {
        convert_standard_ggsw_ciphertext_to_fourier(&ggsw, &mut fourier_ggsw);
    }

    fourier_key
}

pub fn switch_scheme_standard<OutputCont>(
    glev: &GlweCiphertextListOwned<u64>,
    ggsw: &mut GgswCiphertext<OutputCont>,
    ss_key: FourierGgswCiphertextListView<'_>,
) where
    OutputCont: ContainerMut<Element = u64>,
{
    assert_eq!(glev.ciphertext_modulus(), ggsw.ciphertext_modulus());
    assert_eq!(glev.polynomial_size(), ggsw.polynomial_size());
    assert_eq!(glev.polynomial_size(), ss_key.polynomial_size());
    assert_eq!(glev.glwe_size(), ggsw.glwe_size());
    assert_eq!(glev.glwe_size(), ss_key.glwe_size());
    assert_eq!(
        glev.glwe_ciphertext_count().0,
        ggsw.decomposition_level_count().0
    );

    ggsw.as_mut().fill(0u64);

    let glwe_size = glev.glwe_size();
    let glwe_dimension = glwe_size.to_glwe_dimension();
    let fft = Fft::new(glev.polynomial_size());
    let fft = fft.as_view();
    let scratch =
        add_external_product_assign_scratch::<u64>(glwe_size, glev.polynomial_size(), fft);

    let glev_count = glev.glwe_ciphertext_count().0;
    for (col, mut glwe_list) in ggsw
        .as_mut_glwe_list()
        .chunks_exact_mut(glwe_size.0)
        .enumerate()
    {
        let glwe_bit = glev.get(glev_count - 1 - col);
        let (mut glwe_mask_list, mut glwe_body_list) = glwe_list.split_at_mut(glwe_dimension.0);

        for (mut glwe_mask, ss_key_ggsw) in glwe_mask_list.iter_mut().zip(ss_key.into_ggsw_iter()) {
            let mut mem = PodBuffer::try_new(scratch).expect("allocate scheme-switch scratch");
            add_external_product_assign::<u64>(
                glwe_mask.as_mut_view(),
                ss_key_ggsw,
                glwe_bit.as_view(),
                fft,
                &mut PodStack::new(&mut mem),
            );
        }
        let mut body = glwe_body_list.get_mut(0);
        clone_glwe(&mut body, &glwe_bit);
    }
}

pub struct DecoupledSchemeSwitchingKey {
    source_glwe_dimension: GlweDimension,
    target_glwe_dimension: GlweDimension,
    polynomial_size: PolynomialSize,
    ss_base_log: DecompositionBaseLog,
    ss_level: DecompositionLevelCount,
    ciphertext_modulus: CiphertextModulus<u64>,
    ss0: Vec<GlweCiphertextListOwned<u64>>,
    ss1: Vec<GlweCiphertextListOwned<u64>>,
}

impl DecoupledSchemeSwitchingKey {
    pub fn source_glwe_dimension(&self) -> GlweDimension {
        self.source_glwe_dimension
    }

    pub fn target_glwe_dimension(&self) -> GlweDimension {
        self.target_glwe_dimension
    }

    pub fn polynomial_size(&self) -> PolynomialSize {
        self.polynomial_size
    }

    pub fn ss_base_log(&self) -> DecompositionBaseLog {
        self.ss_base_log
    }

    pub fn ss_level(&self) -> DecompositionLevelCount {
        self.ss_level
    }
}

pub fn generate_decoupled_scheme_switching_key<NoiseDistribution, Gen>(
    source_glwe_secret_key: &GlweSecretKeyOwned<u64>,
    target_glwe_secret_key: &GlweSecretKeyOwned<u64>,
    ss_base_log: DecompositionBaseLog,
    ss_level: DecompositionLevelCount,
    noise_distribution: NoiseDistribution,
    ciphertext_modulus: CiphertextModulus<u64>,
    generator: &mut EncryptionRandomGenerator<Gen>,
) -> DecoupledSchemeSwitchingKey
where
    NoiseDistribution: Distribution + Clone,
    u64: Encryptable<Uniform, NoiseDistribution>,
    Gen: ByteRandomGenerator,
{
    assert_eq!(
        source_glwe_secret_key.polynomial_size(),
        target_glwe_secret_key.polynomial_size()
    );
    assert_eq!(
        target_glwe_secret_key.glwe_dimension(),
        GlweDimension(1),
        "DecSS supports target rank k'=1"
    );

    let source_glwe_dimension = source_glwe_secret_key.glwe_dimension();
    let target_glwe_dimension = target_glwe_secret_key.glwe_dimension();
    let polynomial_size = source_glwe_secret_key.polynomial_size();
    let target_poly_list = target_glwe_secret_key.as_polynomial_list();
    let target_poly = target_poly_list.get(0);

    let mut ss0 = Vec::with_capacity(source_glwe_dimension.0);
    let mut ss1 = Vec::with_capacity(source_glwe_dimension.0);

    for source_poly in source_glwe_secret_key.as_polynomial_list().iter() {
        let source_owned = source_poly.as_ref().to_vec();
        let mut product_poly = vec![0u64; polynomial_size.0];
        negacyclic_polynomial_mul_wrapping_u64(
            &mut product_poly,
            &source_owned,
            target_poly.as_ref(),
        );

        ss0.push(encrypt_glev_plaintext_polynomial(
            target_glwe_secret_key,
            &source_owned,
            ss_base_log,
            ss_level,
            noise_distribution.clone(),
            ciphertext_modulus,
            generator,
        ));
        ss1.push(encrypt_glev_plaintext_polynomial(
            target_glwe_secret_key,
            &product_poly,
            ss_base_log,
            ss_level,
            noise_distribution.clone(),
            ciphertext_modulus,
            generator,
        ));
    }

    DecoupledSchemeSwitchingKey {
        source_glwe_dimension,
        target_glwe_dimension,
        polynomial_size,
        ss_base_log,
        ss_level,
        ciphertext_modulus,
        ss0,
        ss1,
    }
}

pub fn switch_scheme_decoupled_rank1<OutputCont>(
    glev: &GlweCiphertextListOwned<u64>,
    ggsw: &mut GgswCiphertext<OutputCont>,
    ss_key: &DecoupledSchemeSwitchingKey,
) where
    OutputCont: ContainerMut<Element = u64>,
{
    assert_eq!(glev.ciphertext_modulus(), ggsw.ciphertext_modulus());
    assert_eq!(glev.ciphertext_modulus(), ss_key.ciphertext_modulus);
    assert_eq!(glev.polynomial_size(), ggsw.polynomial_size());
    assert_eq!(glev.polynomial_size(), ss_key.polynomial_size);
    assert_eq!(
        glev.glwe_size().to_glwe_dimension(),
        ss_key.source_glwe_dimension
    );
    assert_eq!(
        ggsw.glwe_size().to_glwe_dimension(),
        ss_key.target_glwe_dimension
    );
    assert_eq!(
        ss_key.target_glwe_dimension,
        GlweDimension(1),
        "DecSS supports target rank k'=1"
    );
    assert_eq!(
        glev.glwe_ciphertext_count().0,
        ggsw.decomposition_level_count().0
    );

    ggsw.as_mut().fill(0u64);

    let source_glwe_dimension = ss_key.source_glwe_dimension.0;
    let target_glwe_size = ss_key.target_glwe_dimension.to_glwe_size();
    let glev_count = glev.glwe_ciphertext_count().0;

    for (level_idx, mut glwe_list) in ggsw
        .as_mut_glwe_list()
        .chunks_exact_mut(target_glwe_size.0)
        .enumerate()
    {
        let source_glwe = glev.get(glev_count - 1 - level_idx);
        let source_mask = source_glwe.get_mask();
        let source_body = source_glwe.get_body();
        let source_body_poly = source_body.as_polynomial();

        {
            let mut target_mask_row = glwe_list.get_mut(0);
            let mut target_mask = target_mask_row.get_mut_mask();
            let mut target_mask_poly_list = target_mask.as_mut_polynomial_list();
            let mut target_mask_poly = target_mask_poly_list.get_mut(0);
            target_mask_poly
                .as_mut()
                .clone_from_slice(source_body_poly.as_ref());

            for (i, source_mask_poly) in source_mask.as_polynomial_list().iter().enumerate() {
                add_decomposed_polynomial_glev_to_glwe(
                    &mut target_mask_row,
                    &source_mask_poly,
                    &ss_key.ss1[i],
                    false,
                    ss_key.ss_base_log,
                    ss_key.ss_level,
                );
            }
        }

        {
            let mut target_body_row = glwe_list.get_mut(1);
            let mut target_body = target_body_row.get_mut_body();
            let mut target_body_poly = target_body.as_mut_polynomial();
            target_body_poly
                .as_mut()
                .clone_from_slice(source_body_poly.as_ref());

            for (i, source_mask_poly) in source_mask
                .as_polynomial_list()
                .iter()
                .take(source_glwe_dimension)
                .enumerate()
            {
                add_decomposed_polynomial_glev_to_glwe(
                    &mut target_body_row,
                    &source_mask_poly,
                    &ss_key.ss0[i],
                    true,
                    ss_key.ss_base_log,
                    ss_key.ss_level,
                );
            }
        }
    }
}

fn encrypt_glev_plaintext_polynomial<NoiseDistribution, Gen>(
    glwe_secret_key: &GlweSecretKeyOwned<u64>,
    plaintext_poly: &[u64],
    base_log: DecompositionBaseLog,
    level_count: DecompositionLevelCount,
    noise_distribution: NoiseDistribution,
    ciphertext_modulus: CiphertextModulus<u64>,
    generator: &mut EncryptionRandomGenerator<Gen>,
) -> GlweCiphertextListOwned<u64>
where
    NoiseDistribution: Distribution + Clone,
    u64: Encryptable<Uniform, NoiseDistribution>,
    Gen: ByteRandomGenerator,
{
    let polynomial_size = glwe_secret_key.polynomial_size();
    assert_eq!(plaintext_poly.len(), polynomial_size.0);

    let mut glev = GlweCiphertextList::new(
        0u64,
        glwe_secret_key.glwe_dimension().to_glwe_size(),
        polynomial_size,
        GlweCiphertextCount(level_count.0),
        ciphertext_modulus,
    );

    for (level_idx, mut glwe) in glev.iter_mut().enumerate() {
        let level = level_idx + 1;
        let log_scale = u64::BITS as usize - level * base_log.0;
        let scaled = PlaintextList::from_container(
            plaintext_poly
                .iter()
                .map(|value| value.wrapping_shl(log_scale as u32))
                .collect::<Vec<_>>(),
        );
        encrypt_glwe_ciphertext(
            glwe_secret_key,
            &mut glwe,
            &scaled,
            noise_distribution.clone(),
            generator,
        );
    }

    glev
}

fn add_decomposed_polynomial_glev_to_glwe<OutputCont, PolyCont, KeyCont>(
    output: &mut GlweCiphertext<OutputCont>,
    polynomial: &Polynomial<PolyCont>,
    glev_key: &GlweCiphertextList<KeyCont>,
    subtract: bool,
    base_log: DecompositionBaseLog,
    level_count: DecompositionLevelCount,
) where
    OutputCont: ContainerMut<Element = u64>,
    PolyCont: Container<Element = u64>,
    KeyCont: Container<Element = u64>,
{
    assert_eq!(output.polynomial_size(), polynomial.polynomial_size());
    assert_eq!(output.polynomial_size(), glev_key.polynomial_size());
    assert_eq!(output.glwe_size(), glev_key.glwe_size());
    assert_eq!(glev_key.glwe_ciphertext_count().0, level_count.0);

    let polynomial_size = polynomial.polynomial_size();
    let decomposer = SignedDecomposer::new(base_log, level_count);
    let mut decomposed_polys =
        PolynomialList::new(0u64, polynomial_size, PolynomialCount(level_count.0));

    for (coeff_idx, value) in polynomial.iter().enumerate() {
        for (level_idx, decomp_value) in decomposer.decompose(*value).enumerate() {
            decomposed_polys
                .get_mut(level_idx)
                .as_mut()
                .get_mut(coeff_idx)
                .map(|slot| *slot = decomp_value.value());
        }
    }

    for (level_idx, decomp_poly) in decomposed_polys.iter().enumerate() {
        let key_glwe = glev_key.get(level_count.0 - 1 - level_idx);
        glwe_polynomial_mul_accumulate(output, &decomp_poly, &key_glwe, subtract);
    }
}

fn glwe_polynomial_mul_accumulate<OutputCont, PolyCont, InputCont>(
    output: &mut GlweCiphertext<OutputCont>,
    polynomial: &Polynomial<PolyCont>,
    input: &GlweCiphertext<InputCont>,
    subtract: bool,
) where
    OutputCont: ContainerMut<Element = u64>,
    PolyCont: Container<Element = u64>,
    InputCont: Container<Element = u64>,
{
    assert_eq!(output.glwe_size(), input.glwe_size());
    assert_eq!(output.polynomial_size(), input.polynomial_size());
    assert_eq!(polynomial.polynomial_size(), input.polynomial_size());

    for (mut output_poly, input_poly) in output
        .as_mut_polynomial_list()
        .iter_mut()
        .zip(input.as_polynomial_list().iter())
    {
        let mut product = vec![0u64; input.polynomial_size().0];
        negacyclic_polynomial_mul_wrapping_u64(
            &mut product,
            polynomial.as_ref(),
            input_poly.as_ref(),
        );
        if subtract {
            for (dst, value) in output_poly.as_mut().iter_mut().zip(product) {
                *dst = dst.wrapping_sub(value);
            }
        } else {
            for (dst, value) in output_poly.as_mut().iter_mut().zip(product) {
                *dst = dst.wrapping_add(value);
            }
        }
    }
}

fn negacyclic_polynomial_mul_wrapping_u64(output: &mut [u64], lhs: &[u64], rhs: &[u64]) {
    assert_eq!(lhs.len(), rhs.len());
    assert_eq!(output.len(), lhs.len());
    output.fill(0u64);

    let n = lhs.len();
    for (i, &lhs_coeff) in lhs.iter().enumerate() {
        if lhs_coeff == 0 {
            continue;
        }
        for (j, &rhs_coeff) in rhs.iter().enumerate() {
            if rhs_coeff == 0 {
                continue;
            }
            let product = lhs_coeff.wrapping_mul(rhs_coeff);
            let degree = i + j;
            if degree < n {
                output[degree] = output[degree].wrapping_add(product);
            } else {
                output[degree - n] = output[degree - n].wrapping_sub(product);
            }
        }
    }
}

pub fn generate_accumulator<F>(
    polynomial_size: PolynomialSize,
    glwe_size: GlweSize,
    message_modulus: usize,
    ciphertext_modulus: CiphertextModulus<u64>,
    delta: u64,
    f: F,
) -> GlweCiphertextOwned<u64>
where
    F: Fn(u64) -> u64,
{
    let box_size = polynomial_size.0 / message_modulus;
    let mut accumulator = vec![0u64; polynomial_size.0];

    for i in 0..message_modulus {
        let start = i * box_size;
        for slot in &mut accumulator[start..start + box_size] {
            *slot = f(i as u64).wrapping_mul(delta);
        }
    }

    let half_box_size = box_size / 2;
    for slot in &mut accumulator[0..half_box_size] {
        *slot = slot.wrapping_neg();
    }
    accumulator.rotate_left(half_box_size);

    let accumulator_plaintext = PlaintextList::from_container(accumulator);
    allocate_and_trivially_encrypt_new_glwe_ciphertext(
        glwe_size,
        &accumulator_plaintext,
        ciphertext_modulus,
    )
}

pub fn lwe_to_glev_by_identity_lut(
    lwe: &LweCiphertextOwned<u64>,
    glev: &mut GlweCiphertextListOwned<u64>,
    bsk: FourierLweBootstrapKeyView<'_>,
    auto_keys: &HashMap<usize, revhomtrace_fourier_tfhe16::AutomorphKeyOwned>,
    cbs_base_log: DecompositionBaseLog,
    log_lut_count: LutCountLog,
    input_bits: usize,
) {
    let lut_count = 1usize << log_lut_count.0;
    let glwe_size = bsk.glwe_size();
    let polynomial_size = bsk.polynomial_size();
    let ciphertext_modulus = lwe.ciphertext_modulus();
    let large_lwe_dimension = bsk.output_lwe_dimension();
    let mut buf_lwe =
        LweCiphertext::new(0u64, large_lwe_dimension.to_lwe_size(), ciphertext_modulus);

    for (chunk_idx, mut glev_chunk) in glev.chunks_mut(lut_count).enumerate() {
        let mut packed_accumulator = vec![0u64; polynomial_size.0];
        for k in 0..glev_chunk.glwe_ciphertext_count().0 {
            let level = chunk_idx * lut_count + k + 1;
            let log_scale = u64::BITS as usize - level * cbs_base_log.0;
            let delta = 1u64 << log_scale;
            let lut = generate_accumulator(
                polynomial_size,
                glwe_size,
                1usize << input_bits,
                ciphertext_modulus,
                delta,
                |x| x,
            );
            let body_offset = (glwe_size.0 - 1) * polynomial_size.0;
            for idx in (k..polynomial_size.0).step_by(lut_count) {
                packed_accumulator[idx] = lut.as_ref()[body_offset + idx];
            }
        }

        let accumulator_plaintext = PlaintextList::from_container(packed_accumulator);
        let accumulator = allocate_and_trivially_encrypt_new_glwe_ciphertext(
            glwe_size,
            &accumulator_plaintext,
            ciphertext_modulus,
        );

        let fft = Fft::new(polynomial_size);
        let fft = fft.as_view();
        let mut mem = PodBuffer::try_new(fft_bootstrap::blind_rotate_assign_scratch::<u64>(
            glwe_size,
            polynomial_size,
            fft,
        ))
        .expect("allocate direct-lift blind-rotate scratch");
        let stack = &mut PodStack::new(&mut mem);
        let (local_accumulator_data, stack) =
            stack.collect_aligned(CACHELINE_ALIGN, accumulator.as_ref().iter().copied());
        let mut local_accumulator = GlweCiphertextMutView::from_container(
            &mut *local_accumulator_data,
            polynomial_size,
            ciphertext_modulus,
        );

        gen_blind_rotate_local_assign(
            bsk,
            local_accumulator.as_mut_view(),
            ModulusSwitchOffset(0),
            log_lut_count,
            lwe.as_ref(),
            fft,
            stack,
        );

        let mut buf_glwe =
            GlweCiphertext::new(0u64, glwe_size, polynomial_size, ciphertext_modulus);
        for (k, mut glwe) in glev_chunk.iter_mut().enumerate() {
            clone_glwe(&mut buf_glwe, &local_accumulator);
            glwe_monic_monomial_div_assign(&mut buf_glwe, MonomialDegree(k));
            extract_lwe_sample_from_glwe_ciphertext(&buf_glwe, &mut buf_lwe, MonomialDegree(0));
            convert_lwe_to_glwe_const(&buf_lwe, &mut glwe);
            revhomtrace_fourier_tfhe16::revtrace_assign(&mut glwe, auto_keys);
        }
    }
}

pub fn cbs_lift_lwe_to_glev_direct(
    lwe: &LweCiphertextOwned<u64>,
    bsk: FourierLweBootstrapKeyView<'_>,
    auto_keys: &HashMap<usize, revhomtrace_fourier_tfhe16::AutomorphKeyOwned>,
    cbs_base_log: DecompositionBaseLog,
    cbs_level: DecompositionLevelCount,
    log_lut_count: LutCountLog,
    input_bits: usize,
) -> GlweCiphertextListOwned<u64> {
    let glwe_size = bsk.glwe_size();
    let polynomial_size = bsk.polynomial_size();
    let ciphertext_modulus = lwe.ciphertext_modulus();

    let mut glev = GlweCiphertextList::new(
        0u64,
        glwe_size,
        polynomial_size,
        GlweCiphertextCount(cbs_level.0),
        ciphertext_modulus,
    );
    lwe_to_glev_by_identity_lut(
        lwe,
        &mut glev,
        bsk,
        auto_keys,
        cbs_base_log,
        log_lut_count,
        input_bits,
    );
    glev
}

pub fn cbs_lift_lwe_to_standard_ggsw_direct(
    lwe: &LweCiphertextOwned<u64>,
    bsk: FourierLweBootstrapKeyView<'_>,
    auto_keys: &HashMap<usize, revhomtrace_fourier_tfhe16::AutomorphKeyOwned>,
    ss_key: FourierGgswCiphertextListView<'_>,
    cbs_base_log: DecompositionBaseLog,
    cbs_level: DecompositionLevelCount,
    log_lut_count: LutCountLog,
    input_bits: usize,
) -> GgswCiphertextOwned<u64> {
    let glwe_size = bsk.glwe_size();
    let polynomial_size = bsk.polynomial_size();
    let ciphertext_modulus = lwe.ciphertext_modulus();
    let glev = cbs_lift_lwe_to_glev_direct(
        lwe,
        bsk,
        auto_keys,
        cbs_base_log,
        cbs_level,
        log_lut_count,
        input_bits,
    );

    let mut ggsw = GgswCiphertext::new(
        0u64,
        glwe_size,
        polynomial_size,
        cbs_base_log,
        cbs_level,
        ciphertext_modulus,
    );
    switch_scheme_standard(&glev, &mut ggsw, ss_key);
    ggsw
}

pub fn cbs_lift_lwe_to_fourier_ggsw_direct(
    lwe: &LweCiphertextOwned<u64>,
    bsk: FourierLweBootstrapKeyView<'_>,
    auto_keys: &HashMap<usize, revhomtrace_fourier_tfhe16::AutomorphKeyOwned>,
    ss_key: FourierGgswCiphertextListView<'_>,
    cbs_base_log: DecompositionBaseLog,
    cbs_level: DecompositionLevelCount,
    log_lut_count: LutCountLog,
    input_bits: usize,
) -> FourierGgswOwned {
    let glwe_size = bsk.glwe_size();
    let polynomial_size = bsk.polynomial_size();
    let ggsw = cbs_lift_lwe_to_standard_ggsw_direct(
        lwe,
        bsk,
        auto_keys,
        ss_key,
        cbs_base_log,
        cbs_level,
        log_lut_count,
        input_bits,
    );

    let fourier_len =
        polynomial_size.to_fourier_polynomial_size().0 * glwe_size.0 * glwe_size.0 * cbs_level.0;
    let mut fourier_ggsw = FourierGgswCiphertext::from_container(
        vec![tfhe_fft::c64::default(); fourier_len],
        glwe_size,
        polynomial_size,
        cbs_base_log,
        cbs_level,
    );
    let fft = Fft::new(polynomial_size);
    let fft = fft.as_view();
    let mut mem = PodBuffer::try_new(
        tfhe::core_crypto::fft_impl::fft64::crypto::ggsw::fill_with_forward_fourier_scratch(fft),
    )
    .expect("allocate direct-lift Fourier conversion scratch");
    fourier_ggsw.as_mut_view().fill_with_forward_fourier(
        ggsw.as_view(),
        fft,
        &mut PodStack::new(&mut mem),
    );
    fourier_ggsw
}

pub fn blind_rotate_for_msb_revtrace_tfhe16<OutputCont>(
    lwe_in: &LweCiphertextOwned<u64>,
    glev_out: &mut GlweCiphertextList<OutputCont>,
    bsk: FourierLweBootstrapKeyView<'_>,
    log_lut_count: LutCountLog,
    cbs_base_log: DecompositionBaseLog,
    cbs_level: DecompositionLevelCount,
    num_extract_bits: usize,
) where
    OutputCont: ContainerMut<Element = u64>,
{
    assert_eq!(lwe_in.lwe_size(), bsk.input_lwe_dimension().to_lwe_size());
    assert_eq!(glev_out.glwe_size(), bsk.glwe_size());
    assert_eq!(glev_out.polynomial_size(), bsk.polynomial_size());
    assert_eq!(glev_out.glwe_ciphertext_count().0, cbs_level.0);
    assert!(
        (1..=3).contains(&num_extract_bits),
        "RevHomTrace multi-bit extraction currently supports 1..=3 bits"
    );

    let polynomial_size = bsk.polynomial_size();
    let glwe_size = bsk.glwe_size();
    let ciphertext_modulus = lwe_in.ciphertext_modulus();
    let half_box_size = polynomial_size.0 / (2 << num_extract_bits);
    let lut_count = 1usize << log_lut_count.0;

    for (acc_idx, mut glev_chunk) in glev_out.chunks_mut(lut_count).enumerate() {
        let mut accumulator = (0..polynomial_size.0)
            .map(|i| {
                let k = i % lut_count;
                let log_scale = u64::BITS as usize - (acc_idx * lut_count + k + 1) * cbs_base_log.0;
                1u64.wrapping_neg() << (log_scale - 1)
            })
            .collect::<Vec<u64>>();

        for slot in &mut accumulator[0..half_box_size] {
            *slot = slot.wrapping_neg();
        }
        accumulator.rotate_left(half_box_size);

        let accumulator_plaintext = PlaintextList::from_container(accumulator);
        let accumulator = allocate_and_trivially_encrypt_new_glwe_ciphertext(
            glwe_size,
            &accumulator_plaintext,
            ciphertext_modulus,
        );

        let fft = Fft::new(polynomial_size);
        let fft = fft.as_view();
        let mut mem = PodBuffer::try_new(fft_bootstrap::blind_rotate_assign_scratch::<u64>(
            glwe_size,
            polynomial_size,
            fft,
        ))
        .expect("allocate MSB blind-rotate scratch");
        let stack = &mut PodStack::new(&mut mem);
        let (local_accumulator_data, stack) =
            stack.collect_aligned(CACHELINE_ALIGN, accumulator.as_ref().iter().copied());
        let mut local_accumulator = GlweCiphertextMutView::from_container(
            &mut *local_accumulator_data,
            polynomial_size,
            ciphertext_modulus,
        );

        gen_blind_rotate_local_assign(
            bsk,
            local_accumulator.as_mut_view(),
            ModulusSwitchOffset(0),
            log_lut_count,
            lwe_in.as_ref(),
            fft,
            stack,
        );

        for (i, mut glwe) in glev_chunk.iter_mut().enumerate() {
            clone_glwe(&mut glwe, &local_accumulator);
            glwe_monic_monomial_div_assign(&mut glwe, MonomialDegree(i));
        }
    }
}

pub fn convert_to_ggsw_after_blind_rotate_revtrace_tfhe16<InputCont, OutputCont>(
    glev_in: &GlweCiphertextList<InputCont>,
    ggsw_out: &mut GgswCiphertext<OutputCont>,
    bit_idx_from_msb: usize,
    auto_keys: &HashMap<usize, revhomtrace_fourier_tfhe16::AutomorphKeyOwned>,
    ss_key: FourierGgswCiphertextListView<'_>,
) where
    InputCont: Container<Element = u64>,
    OutputCont: ContainerMut<Element = u64>,
{
    assert!(
        bit_idx_from_msb <= 2,
        "RevHomTrace multi-bit extraction currently supports at most 3 bits"
    );
    assert_eq!(glev_in.polynomial_size(), ggsw_out.polynomial_size());
    assert_eq!(glev_in.glwe_size(), ggsw_out.glwe_size());
    assert_eq!(glev_in.polynomial_size(), ss_key.polynomial_size());
    assert_eq!(glev_in.glwe_size(), ss_key.glwe_size());

    let glwe_size = glev_in.glwe_size();
    let polynomial_size = glev_in.polynomial_size();
    let ciphertext_modulus = glev_in.ciphertext_modulus();
    let cbs_level = ggsw_out.decomposition_level_count();
    let cbs_base_log = ggsw_out.decomposition_base_log();
    let large_lwe_dimension = LweDimension(glwe_size.to_glwe_dimension().0 * polynomial_size.0);

    let mut buf_lwe =
        LweCiphertext::new(0u64, large_lwe_dimension.to_lwe_size(), ciphertext_modulus);
    let mut glev_out = GlweCiphertextList::new(
        0u64,
        glwe_size,
        polynomial_size,
        GlweCiphertextCount(cbs_level.0),
        ciphertext_modulus,
    );

    for (k, (mut glwe_out, glwe_in)) in glev_out.iter_mut().zip(glev_in.iter()).enumerate() {
        let cur_level = k + 1;
        let log_scale = u64::BITS as usize - cur_level * cbs_base_log.0;

        match bit_idx_from_msb {
            0 => {
                extract_lwe_sample_from_glwe_ciphertext(&glwe_in, &mut buf_lwe, MonomialDegree(0));
                lwe_ciphertext_plaintext_add_assign(
                    &mut buf_lwe,
                    Plaintext(1u64 << (log_scale - 1)),
                );
            }
            1 => {
                glwe_monic_monomial_mul_into(
                    &mut glwe_out,
                    &glwe_in,
                    MonomialDegree(polynomial_size.0 / 2),
                );
                extract_lwe_sample_from_glwe_ciphertext(&glwe_out, &mut buf_lwe, MonomialDegree(0));
                lwe_ciphertext_opposite_assign(&mut buf_lwe);
                lwe_ciphertext_plaintext_add_assign(
                    &mut buf_lwe,
                    Plaintext(1u64 << (log_scale - 1)),
                );
            }
            2 => {
                glwe_monic_monomial_mul_into(
                    &mut glwe_out,
                    &glwe_in,
                    MonomialDegree(polynomial_size.0 / 4),
                );

                let mut buf_glwe1 =
                    GlweCiphertext::new(0u64, glwe_size, polynomial_size, ciphertext_modulus);
                let mut buf_glwe2 =
                    GlweCiphertext::new(0u64, glwe_size, polynomial_size, ciphertext_modulus);
                glwe_monic_monomial_mul_into(
                    &mut buf_glwe1,
                    &glwe_out,
                    MonomialDegree(polynomial_size.0 / 4),
                );
                glwe_monic_monomial_mul_into(
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
                    Plaintext(1u64 << (log_scale - 1)),
                );
            }
            _ => unreachable!(),
        }

        convert_lwe_to_glwe_const(&buf_lwe, &mut glwe_out);
        revhomtrace_fourier_tfhe16::revtrace_assign(&mut glwe_out, auto_keys);
    }

    switch_scheme_standard(&glev_out, ggsw_out, ss_key);
}

pub fn improved_wopbs_multi_bits_revtrace_tfhe16<InputCont, OutputCont, FourierCont, KeyCont>(
    lwe_in: &LweCiphertext<InputCont>,
    ggsw_list_out: &mut GgswCiphertextList<OutputCont>,
    fourier_ggsw_list_out: &mut FourierGgswCiphertextList<FourierCont>,
    num_extract_bits: usize,
    ksk: &LweKeyswitchKey<KeyCont>,
    bsk: FourierLweBootstrapKeyView<'_>,
    auto_keys: &HashMap<usize, revhomtrace_fourier_tfhe16::AutomorphKeyOwned>,
    ss_key: FourierGgswCiphertextListView<'_>,
    log_lut_count: LutCountLog,
) where
    InputCont: Container<Element = u64>,
    OutputCont: ContainerMut<Element = u64>,
    FourierCont: ContainerMut<Element = tfhe_fft::c64>,
    KeyCont: Container<Element = u64>,
{
    assert_eq!(
        lwe_in.ciphertext_modulus(),
        ggsw_list_out.ciphertext_modulus()
    );
    assert_eq!(lwe_in.ciphertext_modulus(), ksk.ciphertext_modulus());
    assert!(lwe_in.ciphertext_modulus().is_native_modulus());
    assert_eq!(
        lwe_in.lwe_size(),
        ksk.input_key_lwe_dimension().to_lwe_size()
    );
    assert_eq!(ksk.output_key_lwe_dimension(), bsk.input_lwe_dimension());
    assert_eq!(bsk.polynomial_size(), ggsw_list_out.polynomial_size());
    assert_eq!(bsk.glwe_size(), ggsw_list_out.glwe_size());
    assert_eq!(
        ggsw_list_out.ggsw_ciphertext_count().0,
        fourier_ggsw_list_out.count()
    );
    assert!(
        (1..=3).contains(&num_extract_bits),
        "RevHomTrace multi-bit extraction currently supports 1..=3 bits"
    );

    let polynomial_size = ggsw_list_out.polynomial_size();
    let glwe_size = ggsw_list_out.glwe_size();
    let ciphertext_modulus = lwe_in.ciphertext_modulus();
    let log_modulus = ggsw_list_out.ggsw_ciphertext_count().0;
    assert_eq!(log_modulus % num_extract_bits, 0);
    assert!(polynomial_size.0 >= 1usize << num_extract_bits);

    let cbs_base_log = ggsw_list_out.decomposition_base_log();
    let cbs_level = ggsw_list_out.decomposition_level_count();
    let mut buf = LweCiphertext::from_container(lwe_in.as_ref().to_vec(), ciphertext_modulus);
    let mut fourier_iter = fourier_ggsw_list_out.as_mut_view().into_ggsw_iter();

    for (idx, mut ggsw_chunk) in ggsw_list_out.chunks_exact_mut(num_extract_bits).enumerate() {
        let mut lwe_extract = LweCiphertext::new(0u64, buf.lwe_size(), ciphertext_modulus);
        lwe_ciphertext_cleartext_mul(
            &mut lwe_extract,
            &buf,
            Cleartext(1u64 << (log_modulus - num_extract_bits * (idx + 1))),
        );

        let mut lwe_extract_ks =
            LweCiphertext::new(0u64, ksk.output_lwe_size(), ciphertext_modulus);
        keyswitch_lwe_ciphertext(ksk, &lwe_extract, &mut lwe_extract_ks);

        let mut acc_glev = GlweCiphertextList::new(
            0u64,
            glwe_size,
            polynomial_size,
            GlweCiphertextCount(cbs_level.0),
            ciphertext_modulus,
        );
        blind_rotate_for_msb_revtrace_tfhe16(
            &lwe_extract_ks,
            &mut acc_glev,
            bsk,
            log_lut_count,
            cbs_base_log,
            cbs_level,
            num_extract_bits,
        );

        let log_scale = u64::BITS as usize - log_modulus + idx * num_extract_bits;
        let acc_plaintext = PlaintextList::from_container(
            (0..polynomial_size.0)
                .map(|i| {
                    if i < (1usize << num_extract_bits) {
                        if (i >> (num_extract_bits - 1)) == 0 {
                            (i << log_scale) as u64
                        } else {
                            (((1usize << (num_extract_bits - 1))
                                + ((1usize << num_extract_bits) - 1 - i))
                                << log_scale) as u64
                        }
                    } else {
                        0u64
                    }
                })
                .collect::<Vec<u64>>(),
        );
        let acc_id = allocate_and_trivially_encrypt_new_glwe_ciphertext(
            glwe_size,
            &acc_plaintext,
            ciphertext_modulus,
        );
        let mut ct0 = GlweCiphertext::new(0u64, glwe_size, polynomial_size, ciphertext_modulus);
        let mut ct1 = GlweCiphertext::new(0u64, glwe_size, polynomial_size, ciphertext_modulus);
        clone_glwe(&mut ct0, &acc_id);

        for i in 0..num_extract_bits {
            let mut ggsw = ggsw_chunk.get_mut(i);
            convert_to_ggsw_after_blind_rotate_revtrace_tfhe16(
                &acc_glev,
                &mut ggsw,
                num_extract_bits - i - 1,
                auto_keys,
                ss_key,
            );

            let mut fourier_ggsw_out = fourier_iter.next().expect("missing Fourier GGSW output");
            convert_standard_ggsw_ciphertext_to_fourier(&ggsw, &mut fourier_ggsw_out);

            glwe_monic_monomial_div_into(&mut ct1, &ct0, MonomialDegree(1usize << i));
            cmux_assign::<u64, _, _, _>(&mut ct0, &mut ct1, &fourier_ggsw_out);
        }

        let mut lwe_extract = LweCiphertext::new(0u64, lwe_in.lwe_size(), ciphertext_modulus);
        extract_lwe_sample_from_glwe_ciphertext(&ct0, &mut lwe_extract, MonomialDegree(0));
        lwe_ciphertext_sub_assign(&mut buf, &lwe_extract);
    }
}

pub fn gen_blind_rotate_local_assign(
    bsk: FourierLweBootstrapKeyView<'_>,
    mut lut: GlweCiphertextMutView<'_, u64>,
    mod_switch_offset: ModulusSwitchOffset,
    log_lut_count: LutCountLog,
    lwe: &[u64],
    fft: tfhe::core_crypto::fft_impl::fft64::math::fft::FftView<'_>,
    stack: &mut PodStack,
) {
    let (lwe_body, lwe_mask) = lwe.split_last().unwrap();

    let polynomial_size = lut.polynomial_size();
    let ciphertext_modulus = lut.ciphertext_modulus();
    assert_eq!(lut.glwe_size(), bsk.glwe_size());
    assert!(ciphertext_modulus.is_compatible_with_native_modulus());
    let centered_ms = matches!(
        env::var("CBS_CENTERED_MS").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    );
    let body = if centered_ms {
        let log_modulus = polynomial_size.0.ilog2() as usize + 1 - log_lut_count.0;
        lwe_body.wrapping_add(centered_binary_ms_body_correction_to_add_u64(
            lwe_mask,
            log_modulus,
        ))
    } else {
        *lwe_body
    };
    let monomial_degree = MonomialDegree(fast_pbs_modulus_switch_u64(
        body,
        polynomial_size,
        mod_switch_offset,
        log_lut_count,
    ));

    for mut poly in lut.as_mut_polynomial_list().iter_mut() {
        let (tmp_poly, _) = stack.make_aligned_raw(poly.as_ref().len(), CACHELINE_ALIGN);
        let mut tmp_poly = Polynomial::from_container(&mut *tmp_poly);
        tmp_poly.as_mut().copy_from_slice(poly.as_ref());
        polynomial_wrapping_monic_monomial_div(&mut poly, &tmp_poly, monomial_degree);
    }

    let mut ct0 = lut;
    let mut ct1 = GlweCiphertext::new(0u64, bsk.glwe_size(), polynomial_size, ciphertext_modulus);

    for (lwe_mask_element, bootstrap_key_ggsw) in lwe_mask.iter().zip(bsk.into_ggsw_iter()) {
        if *lwe_mask_element != 0 {
            let monomial_degree = MonomialDegree(fast_pbs_modulus_switch_u64(
                *lwe_mask_element,
                polynomial_size,
                mod_switch_offset,
                log_lut_count,
            ));

            for (mut ct1_poly, ct0_poly) in ct1
                .as_mut_polynomial_list()
                .iter_mut()
                .zip(ct0.as_polynomial_list().iter())
            {
                polynomial_wrapping_monic_monomial_mul_and_subtract_u64(
                    &mut ct1_poly,
                    &ct0_poly,
                    monomial_degree,
                );
            }

            add_external_product_assign(
                ct0.as_mut_view(),
                bootstrap_key_ggsw,
                ct1.as_view(),
                fft,
                stack,
            );
        }
    }
}

pub fn convert_lwe_to_glwe_const<InputCont, OutputCont>(
    input: &LweCiphertext<InputCont>,
    output: &mut GlweCiphertext<OutputCont>,
) where
    InputCont: Container<Element = u64>,
    OutputCont: ContainerMut<Element = u64>,
{
    let lwe_dimension = input.lwe_size().to_lwe_dimension().0;
    let glwe_dimension = output.glwe_size().to_glwe_dimension().0;
    let polynomial_size = output.polynomial_size().0;

    assert_eq!(lwe_dimension, glwe_dimension * polynomial_size);
    assert_eq!(input.ciphertext_modulus(), output.ciphertext_modulus());

    let (lwe_mask, lwe_body) = input.get_mask_and_body();
    let (mut glwe_mask, mut glwe_body) = output.get_mut_mask_and_body();

    *glwe_body.as_mut().get_mut(0).unwrap() = *lwe_body.data;

    for (glwe_poly, lwe_poly) in glwe_mask
        .as_mut()
        .chunks_exact_mut(polynomial_size)
        .zip(lwe_mask.as_ref().chunks_exact(polynomial_size))
    {
        glwe_poly.clone_from_slice(lwe_poly);
        glwe_poly.reverse();
        slice_wrapping_opposite_assign(&mut glwe_poly[0..polynomial_size - 1]);
        glwe_poly.rotate_left(polynomial_size - 1);
    }
}

pub fn glwe_preprocessing_moddown_1bit<ContMut>(input: &mut GlweCiphertext<ContMut>)
where
    ContMut: ContainerMut<Element = u64>,
{
    assert!(input.ciphertext_modulus().is_native_modulus());

    let small_ciphertext_modulus =
        CiphertextModulus::<u64>::try_new_power_of_2(u64::BITS as usize - 1).unwrap();
    let mut buf = GlweCiphertext::new(
        0u64,
        input.glwe_size(),
        input.polynomial_size(),
        small_ciphertext_modulus,
    );

    let divisor = small_ciphertext_modulus.get_power_of_two_scaling_to_native_torus();
    for (src, dst) in input.as_ref().iter().zip(buf.as_mut().iter_mut()) {
        *dst = *src - (*src % divisor);
    }
    for (src, dst) in buf.as_ref().iter().zip(input.as_mut().iter_mut()) {
        *dst = src.wrapping_div(divisor);
    }
}

fn clone_glwe<OutputCont, InputCont>(
    output: &mut GlweCiphertext<OutputCont>,
    input: &GlweCiphertext<InputCont>,
) where
    OutputCont: ContainerMut<Element = u64>,
    InputCont: Container<Element = u64>,
{
    assert_eq!(output.glwe_size(), input.glwe_size());
    assert_eq!(output.polynomial_size(), input.polynomial_size());
    output.as_mut().clone_from_slice(input.as_ref());
}

fn glwe_monic_monomial_div_assign<ContMut>(
    glwe: &mut GlweCiphertext<ContMut>,
    monomial_degree: MonomialDegree,
) where
    ContMut: ContainerMut<Element = u64>,
{
    for mut poly in glwe.as_mut_polynomial_list().iter_mut() {
        polynomial_wrapping_monic_monomial_div_assign(&mut poly, monomial_degree);
    }
}

fn glwe_monic_monomial_mul_into<OutputCont, InputCont>(
    output: &mut GlweCiphertext<OutputCont>,
    input: &GlweCiphertext<InputCont>,
    monomial_degree: MonomialDegree,
) where
    OutputCont: ContainerMut<Element = u64>,
    InputCont: Container<Element = u64>,
{
    assert_eq!(output.glwe_size(), input.glwe_size());
    assert_eq!(output.polynomial_size(), input.polynomial_size());
    assert_eq!(output.ciphertext_modulus(), input.ciphertext_modulus());
    for (mut output_poly, input_poly) in output
        .as_mut_polynomial_list()
        .iter_mut()
        .zip(input.as_polynomial_list().iter())
    {
        polynomial_wrapping_monic_monomial_mul(&mut output_poly, &input_poly, monomial_degree);
    }
}

fn glwe_monic_monomial_div_into<OutputCont, InputCont>(
    output: &mut GlweCiphertext<OutputCont>,
    input: &GlweCiphertext<InputCont>,
    monomial_degree: MonomialDegree,
) where
    OutputCont: ContainerMut<Element = u64>,
    InputCont: Container<Element = u64>,
{
    assert_eq!(output.glwe_size(), input.glwe_size());
    assert_eq!(output.polynomial_size(), input.polynomial_size());
    assert_eq!(output.ciphertext_modulus(), input.ciphertext_modulus());
    for (mut output_poly, input_poly) in output
        .as_mut_polynomial_list()
        .iter_mut()
        .zip(input.as_polynomial_list().iter())
    {
        polynomial_wrapping_monic_monomial_div(&mut output_poly, &input_poly, monomial_degree);
    }
}

fn fast_pbs_modulus_switch_u64(
    input: u64,
    poly_size: PolynomialSize,
    offset: ModulusSwitchOffset,
    lut_count_log: LutCountLog,
) -> usize {
    let mut output = input << offset.0;
    output >>= u64::BITS as usize - poly_size.0.ilog2() as usize - 2 + lut_count_log.0;
    output = output.wrapping_add(1);
    output >>= 1;
    output <<= lut_count_log.0;
    output as usize
}

fn standard_modulus_switch_round_u64(input: u64, log_modulus: usize) -> u64 {
    assert!(log_modulus <= u64::BITS as usize);
    if log_modulus == u64::BITS as usize {
        return input;
    }
    input.wrapping_add(1u64 << (u64::BITS as usize - log_modulus - 1))
        >> (u64::BITS as usize - log_modulus)
        << (u64::BITS as usize - log_modulus)
}

fn centered_binary_ms_body_correction_to_add_u64(lwe_mask: &[u64], log_modulus: usize) -> u64 {
    let mut sum_half_mask_round_errors = 0u64;
    let mut sum_halving_errors_doubled = 0i128;

    for &mask_elem in lwe_mask {
        let error =
            standard_modulus_switch_round_u64(mask_elem, log_modulus).wrapping_sub(mask_elem);
        let signed_error = if error >= (1u64 << 63) {
            error as i128 - (1i128 << 64)
        } else {
            error as i128
        };
        let half_error = signed_error / 2;
        let halving_error_doubled = 2 * half_error - signed_error;
        sum_half_mask_round_errors =
            sum_half_mask_round_errors.wrapping_add(half_error as i64 as u64);
        sum_halving_errors_doubled += halving_error_doubled;
    }

    let sum_halving_errors = (sum_halving_errors_doubled / 2) as i64 as u64;
    let sum_half_mask_round_errors = sum_half_mask_round_errors.wrapping_sub(sum_halving_errors);
    let half_case = 1u64 << (u64::BITS as usize - log_modulus - 1);
    sum_half_mask_round_errors.wrapping_sub(half_case)
}

fn polynomial_wrapping_monic_monomial_mul_and_subtract_u64<OutputCont, InputCont>(
    output: &mut Polynomial<OutputCont>,
    input: &Polynomial<InputCont>,
    monomial_degree: MonomialDegree,
) where
    OutputCont: ContainerMut<Element = u64>,
    InputCont: Container<Element = u64>,
{
    assert_eq!(output.polynomial_size(), input.polynomial_size());

    fn copy_with_neg_and_subtract(dst: &mut [u64], src: &[u64], src_orig: &[u64]) {
        for ((dst, src), src_orig) in dst.iter_mut().zip(src).zip(src_orig) {
            *dst = src.wrapping_neg().wrapping_sub(*src_orig);
        }
    }

    fn copy_without_neg_and_subtract(dst: &mut [u64], src: &[u64], src_orig: &[u64]) {
        for ((dst, src), src_orig) in dst.iter_mut().zip(src).zip(src_orig) {
            *dst = src.wrapping_sub(*src_orig);
        }
    }

    let polynomial_size = output.polynomial_size().0;
    let remaining_degree = monomial_degree.0 % polynomial_size;
    let full_cycles_count = monomial_degree.0 / polynomial_size;

    if full_cycles_count % 2 == 0 {
        copy_with_neg_and_subtract(
            &mut output.as_mut()[..remaining_degree],
            &input.as_ref()[polynomial_size - remaining_degree..],
            &input.as_ref()[..remaining_degree],
        );
        copy_without_neg_and_subtract(
            &mut output.as_mut()[remaining_degree..],
            &input.as_ref()[..polynomial_size - remaining_degree],
            &input.as_ref()[remaining_degree..],
        );
    } else {
        copy_without_neg_and_subtract(
            &mut output.as_mut()[..remaining_degree],
            &input.as_ref()[polynomial_size - remaining_degree..],
            &input.as_ref()[..remaining_degree],
        );
        copy_with_neg_and_subtract(
            &mut output.as_mut()[remaining_degree..],
            &input.as_ref()[..polynomial_size - remaining_degree],
            &input.as_ref()[remaining_degree..],
        );
    }
}

fn eval_x_k_into<OutputCont, InputCont>(
    output: &mut Polynomial<OutputCont>,
    input: &Polynomial<InputCont>,
    k: usize,
) where
    OutputCont: ContainerMut<Element = u64>,
    InputCont: Container<Element = u64>,
{
    assert_eq!(k % 2, 1);
    assert!(input.polynomial_size().0.is_power_of_two());
    output.as_mut().fill(0u64);
    output.as_mut()[0] = input.as_ref()[0];
    for i in 1..input.polynomial_size().0 {
        let j = i * k % input.polynomial_size().0;
        let sign = if ((i * k) / input.polynomial_size().0) % 2 == 0 {
            1u64
        } else {
            u64::MAX
        };
        output.as_mut()[j] = sign.wrapping_mul(input.as_ref()[i]);
    }
}
