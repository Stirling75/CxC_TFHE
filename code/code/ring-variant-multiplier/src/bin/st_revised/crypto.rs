use crate::{revhomtrace_fourier_tfhe16 as trace, revhomtrace_lift_tfhe16 as lift};
use dyn_stack::{PodBuffer, PodStack, StackReq};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use tfhe::core_crypto::fft_impl::fft64::crypto::bootstrap;
use tfhe::core_crypto::fft_impl::fft64::math::fft::FftView;
use tfhe::core_crypto::prelude::*;
use super::parameters::Parameters;

pub type Lwe = LweCiphertextOwned<u64>;
pub type Glwe = GlweCiphertextOwned<u64>;
pub type Selector = lift::FourierGgswOwned;
pub const BIT_DELTA: u64 = 1u64 << 61;

#[derive(Default)]
pub struct Counters {
    pub blind_rotations: AtomicUsize,
    pub key_switches: AtomicUsize,
    pub cmux: AtomicUsize,
    pub conversions: AtomicUsize,
}

impl Counters {
    pub fn reset(&self) {
        for c in [&self.blind_rotations, &self.key_switches, &self.cmux, &self.conversions] {
            c.store(0, Ordering::Relaxed);
        }
    }
    pub fn snapshot(&self) -> serde_json::Value {
        serde_json::json!({
            "blind_rotations": self.blind_rotations.load(Ordering::Relaxed),
            "key_switches": self.key_switches.load(Ordering::Relaxed),
            "cmux": self.cmux.load(Ordering::Relaxed),
            "conversions": self.conversions.load(Ordering::Relaxed)
        })
    }
}

pub struct EvaluationKeys {
    pub params: Parameters,
    pub bsk: FourierLweBootstrapKeyOwned,
    pub ksk: LweKeyswitchKeyOwned<u64>,
    pub auto: HashMap<usize, trace::AutomorphKeyOwned>,
    pub ss: lift::FourierGgswListOwned,
    pub counts: Counters,
}

pub struct Secrets {
    pub small: LweSecretKeyOwned<u64>,
    pub big: LweSecretKeyOwned<u64>,
}

pub fn generate_keys(params: &Parameters) -> (Secrets, EvaluationKeys) {
    let Parameters { lwe_dimension: small_n, polynomial_size: n, glwe_dimension: k,
        lwe_sigma, glwe_sigma, pbs, ks, auto: auto_params, ss: ss_params, .. } = *params;
    let mut seeder = new_seeder();
    let mut secret = SecretRandomGenerator::<DefaultRandomGenerator>::new(seeder.seed());
    let mut encryption = EncryptionRandomGenerator::<DefaultRandomGenerator>::new(seeder.seed(), seeder.as_mut());
    let small = allocate_and_generate_new_binary_lwe_secret_key(LweDimension(small_n), &mut secret);
    let glwe = allocate_and_generate_new_binary_glwe_secret_key(GlweDimension(k), PolynomialSize(n), &mut secret);
    let big = glwe.clone().into_lwe_secret_key();
    let glwe_noise = DynamicDistribution::new_gaussian_from_std_dev(StandardDev(glwe_sigma));
    let lwe_noise = DynamicDistribution::new_gaussian_from_std_dev(StandardDev(lwe_sigma));
    let standard = allocate_and_generate_new_lwe_bootstrap_key(&small, &glwe,
        DecompositionBaseLog(pbs[0]), DecompositionLevelCount(pbs[1]), glwe_noise, modulus(), &mut encryption);
    let mut bsk = FourierLweBootstrapKey::new(LweDimension(small_n), GlweSize(k + 1), PolynomialSize(n),
        DecompositionBaseLog(pbs[0]), DecompositionLevelCount(pbs[1]));
    convert_standard_lwe_bootstrap_key_to_fourier(&standard, &mut bsk);
    drop(standard);
    let ksk = allocate_and_generate_new_lwe_keyswitch_key(&big, &small,
        DecompositionBaseLog(ks[0]), DecompositionLevelCount(ks[1]), lwe_noise, modulus(), &mut encryption);
    let fft_type = if params.split_trace_fft { trace::FftType::Split(40) } else { trace::FftType::Vanilla };
    let auto = trace::gen_all_auto_keys(DecompositionBaseLog(auto_params[0]), DecompositionLevelCount(auto_params[1]),
        fft_type, &glwe, glwe_noise, &mut encryption);
    let ss = lift::generate_scheme_switching_key_standard(&glwe, DecompositionBaseLog(ss_params[0]),
        DecompositionLevelCount(ss_params[1]), glwe_noise, modulus(), &mut encryption);
    (Secrets { small, big }, EvaluationKeys { params: params.clone(), bsk, ksk, auto, ss, counts: Counters::default() })
}

pub fn modulus() -> CiphertextModulus<u64> { CiphertextModulus::new_native() }
pub fn zero_glwe(keys: &EvaluationKeys) -> Glwe {
    Glwe::new(0, keys.bsk.glwe_size(), keys.bsk.polynomial_size(), modulus())
}
pub fn zero_lwe(keys: &EvaluationKeys) -> Lwe {
    Lwe::new(0, LweSize(keys.params.glwe_dimension * keys.params.polynomial_size + 1), modulus())
}

pub fn extract(input: &Glwe, coefficient: usize) -> Lwe {
    let mut out = Lwe::new(0, LweSize(input.glwe_size().to_glwe_dimension().0 * input.polynomial_size().0 + 1), modulus());
    extract_lwe_sample_from_glwe_ciphertext(input, &mut out, MonomialDegree(coefficient));
    out
}

pub fn rotate_div(input: &Glwe, degree: usize) -> Glwe {
    use tfhe::core_crypto::algorithms::polynomial_algorithms::polynomial_wrapping_monic_monomial_div;
    let mut out = Glwe::new(0, input.glwe_size(), input.polynomial_size(), modulus());
    for (mut dst, src) in out.as_mut_polynomial_list().iter_mut().zip(input.as_polynomial_list().iter()) {
        polynomial_wrapping_monic_monomial_div(&mut dst, &src, MonomialDegree(degree));
    }
    out
}

struct Scratch {
    polynomial_size: usize,
    fft: Fft,
    bytes: usize,
    buffer: PodBuffer,
}

thread_local! {
    // Per-thread FFT handle and scratch memory reused across CMux, blind
    // rotation and Fourier conversion. The scratch is overwritten before it is
    // read, so reuse does not change any output. If the slot is already
    // borrowed on this thread (re-entrancy), a fresh buffer is used instead.
    static SCRATCH: RefCell<Option<Scratch>> = const { RefCell::new(None) };
}

fn with_scratch<R>(n: PolynomialSize, requirement: impl Fn(FftView<'_>) -> StackReq,
                   f: impl FnOnce(FftView<'_>, &mut PodStack) -> R) -> R {
    SCRATCH.with(|cell| match cell.try_borrow_mut() {
        Ok(mut slot) => {
            if slot.as_ref().is_none_or(|s| s.polynomial_size != n.0) {
                let fft = Fft::new(n);
                let req = requirement(fft.as_view());
                let buffer = PodBuffer::try_new(req).unwrap();
                *slot = Some(Scratch { polynomial_size: n.0, fft, bytes: req.unaligned_bytes_required(), buffer });
            }
            let scratch = slot.as_mut().unwrap();
            let req = requirement(scratch.fft.as_view());
            if req.unaligned_bytes_required() > scratch.bytes {
                scratch.buffer = PodBuffer::try_new(req).unwrap();
                scratch.bytes = req.unaligned_bytes_required();
            }
            let Scratch { fft, buffer, .. } = scratch;
            f(fft.as_view(), PodStack::new(buffer))
        }
        Err(_) => {
            let fft = Fft::new(n);
            let mut buffer = PodBuffer::try_new(requirement(fft.as_view())).unwrap();
            f(fft.as_view(), PodStack::new(&mut buffer))
        }
    })
}

pub fn cmux(keys: &EvaluationKeys, selector: &Selector, zero: &Glwe, one: &Glwe) -> Glwe {
    cmux_owned(keys, selector, zero.clone(), one.clone())
}

/// CMux that consumes its inputs; avoids two GLWE copies when the caller owns them.
pub fn cmux_owned(keys: &EvaluationKeys, selector: &Selector, zero: Glwe, one: Glwe) -> Glwe {
    let out = cmux_raw(selector, zero, one);
    keys.counts.cmux.fetch_add(1, Ordering::Relaxed);
    out
}

fn cmux_raw(selector: &Selector, mut zero: Glwe, mut one: Glwe) -> Glwe {
    with_scratch(selector.polynomial_size(),
        |fft| cmux_assign_mem_optimized_requirement::<u64>(selector.glwe_size(), selector.polynomial_size(), fft),
        |fft, stack| cmux_assign_mem_optimized(&mut zero, &mut one, selector, fft, stack));
    zero
}

pub fn small_input(keys: &EvaluationKeys, input: &Lwe) -> Lwe {
    if input.lwe_size().0 == keys.params.lwe_dimension + 1 { return input.clone(); }
    assert_eq!(input.lwe_size().0, keys.params.glwe_dimension * keys.params.polynomial_size + 1);
    let mut small = Lwe::new(0, LweSize(keys.params.lwe_dimension + 1), modulus());
    keyswitch_lwe_ciphertext(&keys.ksk, input, &mut small);
    keys.counts.key_switches.fetch_add(1, Ordering::Relaxed);
    small
}

fn packed_lut(keys: &EvaluationKeys, functions: &[(u64, fn(u64) -> u64)]) -> Glwe {
    let n = keys.params.polynomial_size;
    let size = keys.bsk.glwe_size();
    assert!(functions.len().is_power_of_two() && functions.len() <= 4);
    let tables: Vec<_> = functions.iter().map(|&(delta, f)| {
        lift::generate_accumulator(PolynomialSize(n), size, 4, modulus(), delta, f)
    }).collect();
    let body: Vec<_> = (0..n).map(|i| tables[i % functions.len()].get_body().as_ref()[i]).collect();
    allocate_and_trivially_encrypt_new_glwe_ciphertext(size,
        &PlaintextList::from_container(body), modulus())
}

fn blind_rotate(keys: &EvaluationKeys, small: &Lwe, accumulator: Glwe, theta: usize) -> Glwe {
    let out = blind_rotate_raw(&keys.bsk, small, accumulator, theta);
    keys.counts.blind_rotations.fetch_add(1, Ordering::Relaxed);
    out
}

fn blind_rotate_raw(bsk: &FourierLweBootstrapKeyOwned, small: &Lwe, mut accumulator: Glwe, theta: usize) -> Glwe {
    with_scratch(bsk.polynomial_size(),
        |fft| bootstrap::blind_rotate_assign_scratch::<u64>(bsk.glwe_size(), bsk.polynomial_size(), fft),
        |fft, stack| lift::gen_blind_rotate_local_assign(bsk.as_view(), accumulator.as_mut_view(),
            ModulusSwitchOffset(0), LutCountLog(theta), 0, small.as_ref(), fft, stack));
    accumulator
}

pub fn compress(keys: &EvaluationKeys, input: &Lwe) -> (Lwe, Lwe) {
    let small = small_input(keys, input);
    let accumulator = packed_lut(keys, &[(BIT_DELTA, |s| s % 2), (BIT_DELTA, |s| s / 2)]);
    let rotated = blind_rotate(keys, &small, accumulator, 1);
    (extract(&rotated, 0), extract(&rotated, 1))
}

pub fn emit(keys: &EvaluationKeys, input: &Lwe, delta_log2: u32) -> (Lwe, Lwe) {
    let small = small_input(keys, input);
    let rotated = blind_rotate(keys, &small, packed_lut(keys, &[(1 << delta_log2, |s| s)]), 0);
    let big = extract(&rotated, 0);
    if keys.params.cc2_big_key {
        return (big.clone(), big);
    }
    let small = small_input(keys, &big);
    (big, small)
}

fn fourier_ggsw(standard: &GgswCiphertextOwned<u64>) -> Selector {
    let n = standard.polynomial_size();
    let size = standard.glwe_size();
    let len = (n.0 / 2) * size.0 * size.0 * standard.decomposition_level_count().0;
    let mut out = Selector::from_container(vec![tfhe_fft::c64::default(); len],
        size, n, standard.decomposition_base_log(), standard.decomposition_level_count());
    with_scratch(n, |fft| tfhe::core_crypto::fft_impl::fft64::crypto::ggsw::fill_with_forward_fourier_scratch(fft),
        |fft, stack| out.as_mut_view().fill_with_forward_fourier(standard.as_view(), fft, stack));
    out
}

pub fn lift_grouped(keys: &EvaluationKeys, input: &Lwe, delta_log2: u32) -> [Selector; 2] {
    let n = keys.bsk.polynomial_size();
    let size = keys.bsk.glwe_size();
    let [base, levels] = keys.params.cbs;
    let small = if keys.params.cc2_big_key {
        let mut scaled = input.clone();
        for a in scaled.as_mut() { *a = a.wrapping_mul(1u64 << (62 - delta_log2)); }
        small_input(keys, &scaled)
    } else {
        let mut small = small_input(keys, input);
        for a in small.as_mut() { *a = a.wrapping_mul(1u64 << (62 - delta_log2)); }
        small
    };
    let mut glev = GlweCiphertextList::new(0u64, size, n,
        GlweCiphertextCount(levels), modulus());
    lift::blind_rotate_for_msb_revtrace_tfhe16(&small, &mut glev, keys.bsk.as_view(),
        LutCountLog(2), DecompositionBaseLog(base), DecompositionLevelCount(levels), 2);
    keys.counts.blind_rotations.fetch_add(1, Ordering::Relaxed);
    std::array::from_fn(|i| {
        let mut ggsw = GgswCiphertext::new(0, size, n,
            DecompositionBaseLog(base), DecompositionLevelCount(levels), modulus());
        lift::convert_to_ggsw_after_blind_rotate_revtrace_tfhe16(&glev, &mut ggsw,
            1 - i, &keys.auto, keys.ss.as_view());
        keys.counts.conversions.fetch_add(1, Ordering::Relaxed);
        fourier_ggsw(&ggsw)
    })
}

pub fn lift_binary(keys: &EvaluationKeys, input: &Lwe, high: bool) -> Selector {
    let n = keys.bsk.polynomial_size();
    let size = keys.bsk.glwe_size();
    let [base, levels] = keys.params.cbs;
    let small = small_input(keys, input);
    let f: fn(u64) -> u64 = if high { |s| s / 2 } else { |s| s % 2 };
    let functions: Vec<_> = (1..=levels).map(|level| (1u64 << (64 - base * level), f)).collect();
    let mut glev = GlweCiphertextList::new(0u64, size, n,
        GlweCiphertextCount(levels), modulus());
    let lanes = 1 << keys.params.terminal_lut_count_log;
    for (mut output_chunk, functions) in glev.chunks_mut(lanes).zip(functions.chunks(lanes)) {
        let rotated = blind_rotate(keys, &small, packed_lut(keys, functions), keys.params.terminal_lut_count_log);
        for (lane, mut glwe) in output_chunk.iter_mut().enumerate() {
            let lwe = extract(&rotated, lane);
            lift::convert_lwe_to_glwe_const(&lwe, &mut glwe);
            trace::revtrace_assign(&mut glwe, &keys.auto);
        }
    }
    let mut ggsw = GgswCiphertext::new(0, size, n,
        DecompositionBaseLog(base), DecompositionLevelCount(levels), modulus());
    lift::switch_scheme_standard(&glev, &mut ggsw, keys.ss.as_view());
    keys.counts.conversions.fetch_add(1, Ordering::Relaxed);
    fourier_ggsw(&ggsw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::SmallRng, Rng, SeedableRng};
    use tfhe::core_crypto::commons::math::decomposition::SignedDecomposer;

    fn random_glwe(rng: &mut SmallRng, size: GlweSize, n: PolynomialSize) -> Glwe {
        Glwe::from_container((0..size.0 * n.0).map(|_| rng.gen()).collect(), n, modulus())
    }

    fn random_selector(rng: &mut SmallRng, size: GlweSize, n: PolynomialSize) -> Selector {
        let (base, levels) = (DecompositionBaseLog(5), DecompositionLevelCount(4));
        let standard = GgswCiphertext::from_container(
            (0..levels.0 * size.0 * size.0 * n.0).map(|_| rng.gen()).collect::<Vec<u64>>(),
            size, n, base, modulus());
        fourier_ggsw(&standard)
    }

    #[test]
    fn scratch_reusing_cmux_is_bit_identical_to_library_cmux() {
        let mut rng = SmallRng::seed_from_u64(7);
        let (size, n) = (GlweSize(2), PolynomialSize(2048));
        for _ in 0..4 {
            let selector = random_selector(&mut rng, size, n);
            let (zero, one) = (random_glwe(&mut rng, size, n), random_glwe(&mut rng, size, n));
            let (mut expected, mut rhs) = (zero.clone(), one.clone());
            cmux_assign::<u64, _, _, _>(&mut expected, &mut rhs, &selector);
            // Run twice to exercise the reused (dirty) thread-local scratch.
            assert_eq!(cmux_raw(&selector, zero.clone(), one.clone()).as_ref(), expected.as_ref());
            assert_eq!(cmux_raw(&selector, zero, one).as_ref(), expected.as_ref());
        }
    }

    #[test]
    fn scratch_reusing_blind_rotation_is_bit_identical() {
        let mut rng = SmallRng::seed_from_u64(11);
        let (small_n, size, n) = (16, GlweSize(2), PolynomialSize(2048));
        let (base, levels) = (DecompositionBaseLog(11), DecompositionLevelCount(3));
        let standard = LweBootstrapKey::from_container(
            (0..small_n * levels.0 * size.0 * size.0 * n.0).map(|_| rng.gen()).collect::<Vec<u64>>(),
            size, n, base, levels, modulus());
        let mut bsk = FourierLweBootstrapKey::new(LweDimension(small_n), size, n, base, levels);
        convert_standard_lwe_bootstrap_key_to_fourier(&standard, &mut bsk);
        for theta in [0, 1] {
            let small = Lwe::from_container((0..=small_n).map(|_| rng.gen()).collect(), modulus());
            let lut = random_glwe(&mut rng, size, n);
            let fft = Fft::new(n);
            let mut memory = PodBuffer::try_new(bootstrap::blind_rotate_assign_scratch::<u64>(
                size, n, fft.as_view())).unwrap();
            let mut expected = lut.clone();
            lift::gen_blind_rotate_local_assign(bsk.as_view(), expected.as_mut_view(), ModulusSwitchOffset(0),
                LutCountLog(theta), 0, small.as_ref(), fft.as_view(), PodStack::new(&mut memory));
            assert_eq!(blind_rotate_raw(&bsk, &small, lut.clone(), theta).as_ref(), expected.as_ref());
            assert_eq!(blind_rotate_raw(&bsk, &small, lut, theta).as_ref(), expected.as_ref());
        }
    }

    fn digit_square_sum(base_log: usize, levels: usize, x: u64) -> u64 {
        SignedDecomposer::<u64>::new(DecompositionBaseLog(base_log), DecompositionLevelCount(levels))
            .decompose(x).map(|t| { let d = t.value() as i64; (d * d) as u64 }).sum()
    }

    /// Cross-checks st_gaussian.signed_digit_second_moments (Python) against the
    /// TFHE-rs 1.6.1 decomposer itself: exhaustively over every rounding class for
    /// small decompositions, and by sampling for the two candidate KS shapes.
    #[test]
    fn library_signed_digit_second_moments_match_python_model() {
        for (base_log, levels, exact) in [(1, 4, 1.4375), (2, 3, 4.0625), (3, 2, 10.75),
                                          (1, 8, 2.77734375), (4, 3, 64.015625)] {
            let rep = base_log * levels;
            let total: u64 = (0u64..1 << (rep + 1)).map(|t| digit_square_sum(base_log, levels, t << (63 - rep))).sum();
            assert_eq!(total as f64 / (1u64 << (rep + 1)) as f64, exact, "B=2^{base_log}, l={levels}");
        }
        let mut rng = SmallRng::seed_from_u64(2026);
        // Python exact values; the old (B^2+2)/12 model gives 11.5 and 44.0.
        for (base_log, levels, model) in [(1, 23, 7.777777791023254), (3, 8, 42.419753074645996)] {
            let samples = 1 << 20;
            let mean = (0..samples).map(|_| digit_square_sum(base_log, levels, rng.gen())).sum::<u64>() as f64 / samples as f64;
            assert!((mean / model - 1.0).abs() < 2e-3, "B=2^{base_log}: sampled {mean}, model {model}");
        }
    }
}
