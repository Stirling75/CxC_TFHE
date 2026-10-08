use std::collections::HashMap;
use tfhe::core_crypto::algorithms::slice_algorithms::slice_wrapping_opposite_assign;
use tfhe::core_crypto::commons::math::random::{Distribution, Uniform};
use tfhe::core_crypto::{fft_impl::fft64::c64, prelude::*};

use crate::revhomtrace_lift_tfhe16::glwe_preprocessing_moddown_1bit;

use super::fourier_glwe_keyswitch::*;

#[cfg(all(test, feature = "noise-audit"))]
#[path = "trace_audit.rs"]
mod trace_audit;

pub type AutomorphKeyOwned = AutomorphKey<Vec<c64>>;

// The following codes generalize rlweExpand
// from https://github.com/KULeuven-COSIC/SortingHat
// to automorphism on arbitrary GLWE dimension
pub struct AutomorphKey<C: Container<Element = c64>> {
    ksk: FourierGlweKeyswitchKey<C>,
    decomp_base_log: DecompositionBaseLog,
    decomp_level_count: DecompositionLevelCount,
    glwe_dimension: GlweDimension,
    polynomial_size: PolynomialSize,
    auto_k: usize,
}

impl AutomorphKey<Vec<c64>> {
    pub fn allocate(
        decomp_base_log: DecompositionBaseLog,
        decomp_level_count: DecompositionLevelCount,
        glwe_dimension: GlweDimension,
        polynomial_size: PolynomialSize,
        auto_k: usize,
        fft_type: FftType,
    ) -> Self {
        let glwe_size = glwe_dimension.to_glwe_size();
        let ksk = FourierGlweKeyswitchKey::new(
            glwe_size,
            glwe_size,
            polynomial_size,
            decomp_base_log,
            decomp_level_count,
            fft_type,
        );
        AutomorphKey {
            ksk: ksk,
            decomp_base_log,
            decomp_level_count,
            glwe_dimension,
            polynomial_size,
            auto_k: auto_k,
        }
    }

    pub fn decomposition_base_log(&self) -> DecompositionBaseLog {
        self.decomp_base_log
    }

    pub fn decomposition_level_count(&self) -> DecompositionLevelCount {
        self.decomp_level_count
    }

    pub fn glwe_dimension(&self) -> GlweDimension {
        self.glwe_dimension
    }

    pub fn polynomial_size(&self) -> PolynomialSize {
        self.polynomial_size
    }

    pub fn fft_type(&self) -> FftType {
        self.ksk.fft_type()
    }

    /// Fill this object with the appropriate key switching key
    /// that is used for the automorphism operation
    /// where after_key is {S_i(X)} and before_key is computed as {S_i(X^k)}.
    fn fill_with_automorph_key<Scalar, NoiseDistribution, G>(
        &mut self,
        before_key: &mut GlweSecretKeyOwned<Scalar>,
        after_key: &GlweSecretKeyOwned<Scalar>,
        k: usize,
        noise_parameters: NoiseDistribution,
        generator: &mut EncryptionRandomGenerator<G>,
    ) where
        Scalar: Encryptable<Uniform, NoiseDistribution> + UnsignedTorus + Sync + Send,
        NoiseDistribution: Distribution + Clone,
        G: ByteRandomGenerator,
    {
        debug_assert!(self.glwe_dimension == before_key.glwe_dimension());
        debug_assert!(self.glwe_dimension == after_key.glwe_dimension());
        debug_assert!(self.polynomial_size == before_key.polynomial_size());
        debug_assert!(self.polynomial_size == after_key.polynomial_size());

        let mut before_poly_list = PolynomialList::new(
            Scalar::ZERO,
            self.polynomial_size,
            PolynomialCount(self.glwe_dimension.0),
        );
        for (mut before_poly, after_poly) in before_poly_list
            .iter_mut()
            .zip(after_key.as_polynomial_list().iter())
        {
            let out = eval_x_k_owned(after_poly.as_view(), k);
            before_poly.as_mut().clone_from_slice(out.as_ref());
        }
        *before_key =
            GlweSecretKey::from_container(before_poly_list.into_container(), self.polynomial_size);

        self.fill_with_keyswitch_key(before_key, after_key, noise_parameters, generator);
        self.auto_k = k;
    }

    /// Fill this object with the appropriate keyswitching key
    /// that transforms ciphertexts under before_key to ciphertexts under after_key.
    fn fill_with_keyswitch_key<Scalar, NoiseDistribution, G>(
        &mut self,
        before_key: &GlweSecretKeyOwned<Scalar>,
        after_key: &GlweSecretKeyOwned<Scalar>,
        noise_parameters: NoiseDistribution,
        generator: &mut EncryptionRandomGenerator<G>,
    ) where
        Scalar: Encryptable<Uniform, NoiseDistribution> + UnsignedTorus + Sync + Send,
        NoiseDistribution: Distribution + Clone,
        G: ByteRandomGenerator,
    {
        debug_assert!(self.glwe_dimension == before_key.glwe_dimension());
        debug_assert!(self.glwe_dimension == after_key.glwe_dimension());
        debug_assert!(self.polynomial_size == before_key.polynomial_size());
        debug_assert!(self.polynomial_size == after_key.polynomial_size());

        let decomp_level_count = self.decomp_level_count;
        let decomp_base_log = self.decomp_base_log;
        let standard_ksk = generate_revhomtrace_glwe_keyswitch_key(
            before_key,
            after_key,
            decomp_base_log,
            decomp_level_count,
            noise_parameters,
            generator,
        );
        convert_standard_glwe_keyswitch_key_to_fourier(&standard_ksk, &mut self.ksk);
    }

    fn keyswitch_ciphertext<Scalar, InputCont, OutputCont>(
        &self,
        after: &mut GlweCiphertext<OutputCont>,
        before: &GlweCiphertext<InputCont>,
    ) where
        Scalar: UnsignedTorus + Sync + Send,
        InputCont: Container<Element = Scalar>,
        OutputCont: ContainerMut<Element = Scalar>,
    {
        super::fourier_glwe_keyswitch::keyswitch_glwe_ciphertext(&self.ksk, before, after);
    }

    pub fn auto<Scalar, InputCont, OutputCont>(
        &self,
        after: &mut GlweCiphertext<OutputCont>,
        before: &GlweCiphertext<InputCont>,
    ) where
        Scalar: UnsignedTorus + Sync + Send,
        InputCont: Container<Element = Scalar>,
        OutputCont: ContainerMut<Element = Scalar>,
    {
        let mut before_power = GlweCiphertextOwned::new(
            Scalar::ZERO,
            before.glwe_size(),
            before.polynomial_size(),
            before.ciphertext_modulus(),
        );
        for (mut poly_power, poly) in before_power
            .as_mut_polynomial_list()
            .iter_mut()
            .zip(before.as_polynomial_list().iter())
        {
            poly_power
                .as_mut()
                .clone_from_slice(eval_x_k_owned(poly, self.auto_k).as_ref());
        }

        self.keyswitch_ciphertext(after, &before_power);
    }
}

fn generate_revhomtrace_glwe_keyswitch_key<Scalar, NoiseDistribution, G>(
    input_glwe_sk: &GlweSecretKeyOwned<Scalar>,
    output_glwe_sk: &GlweSecretKeyOwned<Scalar>,
    decomp_base_log: DecompositionBaseLog,
    decomp_level_count: DecompositionLevelCount,
    noise_distribution: NoiseDistribution,
    generator: &mut EncryptionRandomGenerator<G>,
) -> GlweKeyswitchKeyOwned<Scalar>
where
    Scalar: Encryptable<Uniform, NoiseDistribution> + UnsignedTorus,
    NoiseDistribution: Distribution + Clone,
    G: ByteRandomGenerator,
{
    let polynomial_size = input_glwe_sk.polynomial_size();
    let input_glwe_dimension = input_glwe_sk.glwe_dimension();
    let output_glwe_dimension = output_glwe_sk.glwe_dimension();
    let ciphertext_modulus = CiphertextModulus::new_native();

    let mut ksk = GlweKeyswitchKeyOwned::new(
        Scalar::ZERO,
        decomp_base_log,
        decomp_level_count,
        input_glwe_dimension,
        output_glwe_dimension,
        polynomial_size,
        ciphertext_modulus,
    );

    fill_revhomtrace_glwe_keyswitch_key(
        input_glwe_sk,
        output_glwe_sk,
        &mut ksk,
        noise_distribution,
        generator,
    );

    ksk
}

fn fill_revhomtrace_glwe_keyswitch_key<
    Scalar,
    NoiseDistribution,
    InputKeyCont,
    OutputKeyCont,
    KSKeyCont,
    G,
>(
    input_glwe_sk: &GlweSecretKey<InputKeyCont>,
    output_glwe_sk: &GlweSecretKey<OutputKeyCont>,
    ksk: &mut GlweKeyswitchKey<KSKeyCont>,
    noise_distribution: NoiseDistribution,
    generator: &mut EncryptionRandomGenerator<G>,
) where
    Scalar: Encryptable<Uniform, NoiseDistribution> + UnsignedTorus,
    NoiseDistribution: Distribution + Clone,
    InputKeyCont: Container<Element = Scalar>,
    OutputKeyCont: Container<Element = Scalar>,
    KSKeyCont: ContainerMut<Element = Scalar>,
    G: ByteRandomGenerator,
{
    assert_eq!(
        ksk.input_key_glwe_dimension(),
        input_glwe_sk.glwe_dimension()
    );
    assert_eq!(
        ksk.output_key_glwe_dimension(),
        output_glwe_sk.glwe_dimension()
    );
    assert_eq!(ksk.polynomial_size(), input_glwe_sk.polynomial_size());
    assert_eq!(ksk.polynomial_size(), output_glwe_sk.polynomial_size());
    assert!(ksk.ciphertext_modulus().is_native_modulus());

    let polynomial_size = ksk.polynomial_size();
    let decomp_base_log = ksk.decomposition_base_log();
    let decomp_level_count = ksk.decomposition_level_count();

    for (input_sk_poly, mut glev_glwe_list) in input_glwe_sk.as_polynomial_list().iter().zip(
        ksk.as_mut_glwe_ciphertext_list()
            .chunks_exact_mut(decomp_level_count.0),
    ) {
        let mut neg_sk_poly = PlaintextList::new(Scalar::ZERO, PlaintextCount(polynomial_size.0));
        neg_sk_poly
            .as_mut()
            .clone_from_slice(input_sk_poly.as_ref());
        slice_wrapping_opposite_assign(neg_sk_poly.as_mut());

        for (level_idx, mut glwe) in glev_glwe_list.iter_mut().enumerate() {
            let level = level_idx + 1;
            let log_scale = Scalar::BITS - level * decomp_base_log.0;
            let scaled_pt = PlaintextList::from_container(
                neg_sk_poly
                    .as_ref()
                    .iter()
                    .map(|pt| *pt << log_scale)
                    .collect::<Vec<Scalar>>(),
            );
            encrypt_glwe_ciphertext(
                output_glwe_sk,
                &mut glwe,
                &scaled_pt,
                noise_distribution.clone(),
                generator,
            );
        }
    }
}

pub fn gen_all_auto_keys<Scalar, NoiseDistribution, G>(
    decomp_base_log: DecompositionBaseLog,
    decomp_level: DecompositionLevelCount,
    fft_type: FftType,
    glwe_secret_key: &GlweSecretKeyOwned<Scalar>,
    noise_parameters: NoiseDistribution,
    generator: &mut EncryptionRandomGenerator<G>,
) -> HashMap<usize, AutomorphKeyOwned>
where
    Scalar: Encryptable<Uniform, NoiseDistribution> + UnsignedTorus + Sync + Send,
    NoiseDistribution: Distribution + Clone,
    G: ByteRandomGenerator,
{
    let glwe_dimension = glwe_secret_key.glwe_dimension();
    let polynomial_size = glwe_secret_key.polynomial_size();

    let mut hm = HashMap::new();
    for i in 1..=(polynomial_size.0).ilog2() as usize {
        let k = polynomial_size.0 / (1 << (i - 1)) + 1;
        let mut glwe_ksk = AutomorphKey::allocate(
            decomp_base_log,
            decomp_level,
            glwe_dimension,
            polynomial_size,
            i,
            fft_type,
        );
        let mut before_key = glwe_secret_key.clone();

        glwe_ksk.fill_with_automorph_key(
            &mut before_key,
            &glwe_secret_key,
            k,
            noise_parameters.clone(),
            generator,
        );
        hm.insert(k, glwe_ksk);
    }

    hm
}

fn eval_x_k_owned<Scalar>(poly: PolynomialView<'_, Scalar>, k: usize) -> PolynomialOwned<Scalar>
where
    Scalar: UnsignedTorus,
{
    let mut out = PolynomialOwned::new(Scalar::ZERO, poly.polynomial_size());
    assert_eq!(k % 2, 1);
    assert!(poly.polynomial_size().0.is_power_of_two());
    out.as_mut()[0] = poly.as_ref()[0];
    for i in 1..poly.polynomial_size().0 {
        let j = i * k % poly.polynomial_size().0;
        let sign = if ((i * k) / poly.polynomial_size().0) % 2 == 0 {
            Scalar::ONE
        } else {
            Scalar::MAX
        };
        out.as_mut()[j] = sign.wrapping_mul(poly.as_ref()[i]);
    }
    out
}

pub fn trace<Scalar, Cont>(
    glwe_in: &GlweCiphertext<Cont>,
    auto_keys: &HashMap<usize, AutomorphKeyOwned>,
) -> GlweCiphertextOwned<Scalar>
where
    Scalar: UnsignedTorus + Sync + Send,
    Cont: Container<Element = Scalar>,
{
    let mut out = GlweCiphertext::new(
        Scalar::ZERO,
        glwe_in.glwe_size(),
        glwe_in.polynomial_size(),
        glwe_in.ciphertext_modulus(),
    );
    out.as_mut().clone_from_slice(glwe_in.as_ref());
    trace_assign(&mut out, auto_keys);

    out
}

pub fn trace_assign<Scalar, ContMut>(
    glwe_in: &mut GlweCiphertext<ContMut>,
    auto_keys: &HashMap<usize, AutomorphKeyOwned>,
) where
    Scalar: UnsignedTorus + Sync + Send,
    ContMut: ContainerMut<Element = Scalar>,
{
    trace_partial_assign(glwe_in, auto_keys, 1);
}

pub fn revtrace_assign<ContMut>(
    glwe_in: &mut GlweCiphertext<ContMut>,
    auto_keys: &HashMap<usize, AutomorphKeyOwned>,
) where
    ContMut: ContainerMut<Element = u64>,
{
    revtrace_partial_assign(glwe_in, auto_keys, 1);
}

pub fn revtrace_partial_assign<Cont>(
    input: &mut GlweCiphertext<Cont>,
    auto_keys: &HashMap<usize, AutomorphKeyOwned>,
    n: usize,
) where
    Cont: ContainerMut<Element = u64>,
{
    let glwe_size = input.glwe_size();
    let polynomial_size = input.polynomial_size();
    let ciphertext_modulus = input.ciphertext_modulus();

    assert!(polynomial_size.0 % n == 0);

    let mut buf = GlweCiphertextOwned::new(0u64, glwe_size, polynomial_size, ciphertext_modulus);
    let mut out: GlweCiphertext<Vec<u64>> =
        GlweCiphertext::new(0u64, glwe_size, polynomial_size, ciphertext_modulus);
    out.as_mut().clone_from_slice(input.as_ref());

    let log_polynomial_size = polynomial_size.0.ilog2() as usize;
    let log_n = n.ilog2() as usize;
    for i in (1..=(log_polynomial_size - log_n)).rev() {
        let k = polynomial_size.0 / (1 << (i - 1)) + 1;
        let auto_key = auto_keys.get(&k).unwrap();
        glwe_preprocessing_moddown_1bit(&mut out);
        auto_key.auto(&mut buf, &out);
        glwe_ciphertext_add_assign(&mut out, &buf);
    }

    input.as_mut().clone_from_slice(out.as_ref());
}

pub fn trace_partial_assign<Scalar, Cont>(
    input: &mut GlweCiphertext<Cont>,
    auto_keys: &HashMap<usize, AutomorphKeyOwned>,
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
    out.as_mut().clone_from_slice(input.as_ref());

    let log_polynomial_size = polynomial_size.0.ilog2() as usize;
    let log_n = n.ilog2() as usize;
    for i in 1..=(log_polynomial_size - log_n) {
        let k = polynomial_size.0 / (1 << (i - 1)) + 1;
        let auto_key = auto_keys.get(&k).unwrap();
        auto_key.auto(&mut buf, &out);
        glwe_ciphertext_add_assign(&mut out, &buf);
    }

    input.as_mut().clone_from_slice(out.as_ref());
}
