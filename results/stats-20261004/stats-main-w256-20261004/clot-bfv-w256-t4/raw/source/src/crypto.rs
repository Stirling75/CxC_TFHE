use crate::arithmetic::{
    bits_needed, bounded_group_length, mul64, normalizer_plan, public_pbs_counts,
    range_aware_pbs_counts, tensor, NATIVE_DELTA,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use tfhe::core_crypto::prelude::*;

pub type Lwe = LweCiphertextOwned<u64>;
pub type Glwe = GlweCiphertextOwned<u64>;
pub fn modulus() -> CiphertextModulus<u64> {
    CiphertextModulus::new_native()
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Parameters {
    pub n: usize,
    pub polynomial_size: usize,
    pub lwe_sigma: f64,
    pub glwe_sigma: f64,
    pub pbs: [usize; 2],
    pub encoding_pbs: [usize; 2],
    pub ks: [usize; 2],
    pub packing: [usize; 2],
    pub relin: [usize; 2],
    pub input_lut_domain: usize,
    pub normalizer_lut_domain: usize,
    pub degree_aware: bool,
}

impl Default for Parameters {
    fn default() -> Self {
        // Base dimensions/noise/PBS/KS from the existing ST research candidate.
        // Packing and relinearization are NEW diagnostic choices, not certified.
        Self {
            n: 1152,
            polynomial_size: 2048,
            lwe_sigma: 1.4901161193847656e-8,
            glwe_sigma: 9.25119974676756e-16,
            pbs: [11, 3],
            encoding_pbs: [11, 3],
            ks: [3, 8],
            packing: [4, 15],
            relin: [4, 15],
            input_lut_domain: 16,
            normalizer_lut_domain: 16,
            degree_aware: false,
        }
    }
}

impl Parameters {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.n > 0 && [2048, 4096].contains(&self.polynomial_size),
            "unsupported key/ring dimensions"
        );
        for sigma in [self.lwe_sigma, self.glwe_sigma] {
            anyhow::ensure!(
                sigma.is_finite() && sigma > 0.0 && sigma < 1.0,
                "invalid noise"
            );
        }
        for [base, levels] in [
            self.pbs,
            self.encoding_pbs,
            self.ks,
            self.packing,
            self.relin,
        ] {
            anyhow::ensure!(
                base > 0 && levels > 0 && base * levels < 64,
                "invalid decomposition"
            );
        }
        anyhow::ensure!(
            [4, 8, 16].contains(&self.input_lut_domain),
            "invalid input domain"
        );
        anyhow::ensure!(
            [8, 16].contains(&self.normalizer_lut_domain),
            "invalid normalizer domain"
        );
        anyhow::ensure!(
            self.degree_aware || self.normalizer_lut_domain == 16,
            "smaller normalizer domain requires a bounded schedule"
        );
        Ok(())
    }

    pub fn pbs_counts(&self, digits: usize) -> (usize, usize, usize) {
        if self.degree_aware {
            range_aware_pbs_counts(digits, self.normalizer_lut_domain)
        } else {
            public_pbs_counts(digits)
        }
    }

    pub fn precise_encoding() -> Self {
        Self {
            encoding_pbs: [4, 15],
            ..Self::default()
        }
    }

    pub fn deep_encoding() -> Self {
        Self {
            encoding_pbs: [3, 20],
            ..Self::default()
        }
    }
}

pub struct Client {
    pub small: LweSecretKeyOwned<u64>,
    pub big: LweSecretKeyOwned<u64>,
}

pub struct Evaluator {
    pub p: Parameters,
    pub bsk: FourierLweBootstrapKeyOwned,
    pub encoding_bsk: Option<FourierLweBootstrapKeyOwned>,
    pub ksk: LweKeyswitchKeyOwned<u64>,
    pub packing: LwePackingKeyswitchKeyOwned<u64>,
    pub relin: Vec<Glwe>,
}

#[derive(Default, Serialize)]
pub struct Counts {
    pub pbs: usize,
    pub ks: usize,
    pub packed_lwes: usize,
    pub glwe_products: usize,
    pub extracted_coefficients: usize,
}

impl Counts {
    fn merge(&mut self, other: Self) {
        self.pbs += other.pbs;
        self.ks += other.ks;
        self.packed_lwes += other.packed_lwes;
        self.glwe_products += other.glwe_products;
        self.extracted_coefficients += other.extracted_coefficients;
    }
}

#[derive(Default, Serialize)]
pub struct Timings {
    pub input_rescale: f64,
    pub packing: f64,
    pub tensor: f64,
    pub relinearization: f64,
    pub extraction: f64,
    pub digit_decomposition: f64,
    pub normalization: f64,
}

pub struct Kernel {
    pub rescaled_a: Vec<Lwe>,
    pub rescaled_b: Vec<Lwe>,
    pub packed_a: Glwe,
    pub packed_b: Glwe,
    pub coefficients: Vec<Lwe>,
    pub timings: Timings,
    pub counts: Counts,
}

pub fn generate_keys(p: Parameters) -> (Client, Evaluator) {
    let mut seeder = new_seeder();
    let mut secret = SecretRandomGenerator::<DefaultRandomGenerator>::new(seeder.seed());
    let mut enc =
        EncryptionRandomGenerator::<DefaultRandomGenerator>::new(seeder.seed(), seeder.as_mut());
    let small = allocate_and_generate_new_binary_lwe_secret_key(LweDimension(p.n), &mut secret);
    let glwe = allocate_and_generate_new_binary_glwe_secret_key(
        GlweDimension(1),
        PolynomialSize(p.polynomial_size),
        &mut secret,
    );
    let big = glwe.clone().into_lwe_secret_key();
    let gn = DynamicDistribution::new_gaussian_from_std_dev(StandardDev(p.glwe_sigma));
    let ln = DynamicDistribution::new_gaussian_from_std_dev(StandardDev(p.lwe_sigma));
    eprintln!("Generating research PBS and key-switch keys");
    let standard = allocate_and_generate_new_lwe_bootstrap_key(
        &small,
        &glwe,
        DecompositionBaseLog(p.pbs[0]),
        DecompositionLevelCount(p.pbs[1]),
        gn,
        modulus(),
        &mut enc,
    );
    let mut bsk = FourierLweBootstrapKey::new(
        LweDimension(p.n),
        GlweSize(2),
        PolynomialSize(p.polynomial_size),
        DecompositionBaseLog(p.pbs[0]),
        DecompositionLevelCount(p.pbs[1]),
    );
    convert_standard_lwe_bootstrap_key_to_fourier(&standard, &mut bsk);
    drop(standard);
    let encoding_bsk = if p.encoding_pbs == p.pbs {
        None
    } else {
        eprintln!("Generating lower-output-noise encoding PBS key");
        let standard = allocate_and_generate_new_lwe_bootstrap_key(
            &small,
            &glwe,
            DecompositionBaseLog(p.encoding_pbs[0]),
            DecompositionLevelCount(p.encoding_pbs[1]),
            gn,
            modulus(),
            &mut enc,
        );
        let mut key = FourierLweBootstrapKey::new(
            LweDimension(p.n),
            GlweSize(2),
            PolynomialSize(p.polynomial_size),
            DecompositionBaseLog(p.encoding_pbs[0]),
            DecompositionLevelCount(p.encoding_pbs[1]),
        );
        convert_standard_lwe_bootstrap_key_to_fourier(&standard, &mut key);
        Some(key)
    };
    let ksk = allocate_and_generate_new_lwe_keyswitch_key(
        &big,
        &small,
        DecompositionBaseLog(p.ks[0]),
        DecompositionLevelCount(p.ks[1]),
        ln,
        modulus(),
        &mut enc,
    );
    eprintln!(
        "Generating packing key (about {} MiB)",
        p.polynomial_size * p.packing[1] * 2 * p.polynomial_size * 8 / (1 << 20)
    );
    let packing = allocate_and_generate_new_lwe_packing_keyswitch_key(
        &big,
        &glwe,
        DecompositionBaseLog(p.packing[0]),
        DecompositionLevelCount(p.packing[1]),
        gn,
        modulus(),
        &mut enc,
    );
    let square = mul64(glwe.as_ref(), glwe.as_ref());
    let relin = (1..=p.relin[1])
        .map(|level| {
            let weight = 1u64 << (64 - p.relin[0] * level);
            let pt = PlaintextList::from_container(
                square
                    .iter()
                    .map(|x| x.wrapping_mul(weight))
                    .collect::<Vec<_>>(),
            );
            let mut ct = Glwe::new(0, GlweSize(2), PolynomialSize(p.polynomial_size), modulus());
            encrypt_glwe_ciphertext(&glwe, &mut ct, &pt, gn, &mut enc);
            ct
        })
        .collect();
    (
        Client { small, big },
        Evaluator {
            p,
            bsk,
            encoding_bsk,
            ksk,
            packing,
            relin,
        },
    )
}

impl Evaluator {
    pub fn extract(&self, ct: &Glwe, index: usize) -> Lwe {
        let mut out = Lwe::new(0, LweSize(self.p.polynomial_size + 1), modulus());
        extract_lwe_sample_from_glwe_ciphertext(ct, &mut out, MonomialDegree(index));
        out
    }

    fn lut(&self, f: impl Fn(u64) -> u64) -> Glwe {
        self.lut_domain(16, f)
    }

    fn lut_domain(&self, domain: usize, f: impl Fn(u64) -> u64) -> Glwe {
        let n = self.p.polynomial_size;
        let box_size = n / domain;
        let mut body: Vec<u64> = (0..n).map(|i| f((i / box_size) as u64)).collect();
        for x in &mut body[..box_size / 2] {
            *x = x.wrapping_neg();
        }
        body.rotate_left(box_size / 2);
        allocate_and_trivially_encrypt_new_glwe_ciphertext(
            GlweSize(2),
            &PlaintextList::from_container(body),
            modulus(),
        )
    }

    pub fn bootstrap_small(&self, input: &Lwe, lut: &Glwe) -> Lwe {
        let mut out = Lwe::new(0, LweSize(self.p.polynomial_size + 1), modulus());
        programmable_bootstrap_lwe_ciphertext(input, &mut out, lut, &self.bsk);
        out
    }

    fn bootstrap(&self, input: &Lwe, lut: &Glwe, counts: &mut Counts) -> Lwe {
        let mut small = Lwe::new(0, LweSize(self.p.n + 1), modulus());
        keyswitch_lwe_ciphertext(&self.ksk, input, &mut small);
        counts.ks += 1;
        counts.pbs += 1;
        self.bootstrap_small(&small, lut)
    }

    fn bootstrap_encoding(&self, input: &Lwe, lut: &Glwe, counts: &mut Counts) -> Lwe {
        let mut scaled = input.clone();
        for value in scaled.as_mut() {
            *value = value.wrapping_mul((16 / self.p.input_lut_domain) as u64);
        }
        let mut small = Lwe::new(0, LweSize(self.p.n + 1), modulus());
        keyswitch_lwe_ciphertext(&self.ksk, &scaled, &mut small);
        counts.ks += 1;
        counts.pbs += 1;
        let mut out = Lwe::new(0, LweSize(self.p.polynomial_size + 1), modulus());
        let key = self.encoding_bsk.as_ref().unwrap_or(&self.bsk);
        programmable_bootstrap_lwe_ciphertext(&small, &mut out, lut, key);
        out
    }

    pub fn prepare_inputs(&self, client: &Client, digits: &[u64]) -> Vec<Lwe> {
        let mut seeder = new_seeder();
        let mut enc = EncryptionRandomGenerator::<DefaultRandomGenerator>::new(
            seeder.seed(),
            seeder.as_mut(),
        );
        let ln = DynamicDistribution::new_gaussian_from_std_dev(StandardDev(self.p.lwe_sigma));
        let lut = self.lut(|x| x * NATIVE_DELTA);
        digits
            .iter()
            .map(|&x| {
                let input = allocate_and_encrypt_new_lwe_ciphertext(
                    &client.small,
                    Plaintext(x * NATIVE_DELTA),
                    ln,
                    modulus(),
                    &mut enc,
                );
                self.bootstrap_small(&input, &lut)
            })
            .collect()
    }

    fn pack(&self, inputs: &[Lwe]) -> Glwe {
        let data: Vec<u64> = inputs
            .iter()
            .flat_map(|x| x.as_ref().iter().copied())
            .collect();
        let list =
            LweCiphertextList::from_container(data, LweSize(self.p.polynomial_size + 1), modulus());
        let mut out = Glwe::new(
            0,
            GlweSize(2),
            PolynomialSize(self.p.polynomial_size),
            modulus(),
        );
        if rayon::current_num_threads() > 1 {
            par_keyswitch_lwe_ciphertext_list_and_pack_in_glwe_ciphertext(
                &self.packing,
                &list,
                &mut out,
            );
        } else {
            keyswitch_lwe_ciphertext_list_and_pack_in_glwe_ciphertext(
                &self.packing,
                &list,
                &mut out,
            );
        }
        out
    }

    fn relinearize(&self, parts: [Vec<u64>; 3]) -> Glwe {
        let [t, a, b] = parts;
        let mut out = Glwe::from_container(
            [a, b].concat(),
            PolynomialSize(self.p.polynomial_size),
            modulus(),
        );
        let n = self.p.polynomial_size;
        let mut digits = vec![vec![0u64; n]; self.p.relin[1]];
        let decomp = SignedDecomposer::new(
            DecompositionBaseLog(self.p.relin[0]),
            DecompositionLevelCount(self.p.relin[1]),
        );
        for (i, &x) in t.iter().enumerate() {
            for term in decomp.decompose(x) {
                digits[term.level().0 - 1][i] = term.value();
            }
        }
        for (digit, key) in digits.iter().zip(&self.relin) {
            for (dst, src) in out.as_mut().chunks_mut(n).zip(key.as_ref().chunks(n)) {
                for (v, p) in dst.iter_mut().zip(mul64(digit, src)) {
                    *v = v.wrapping_add(p);
                }
            }
        }
        out
    }

    pub fn kernel(&self, a: &[Lwe], b: &[Lwe], p: usize) -> Kernel {
        assert_eq!(a.len(), b.len());
        assert!(2 * a.len() - 1 <= self.p.polynomial_size);
        let mut counts = Counts::default();
        let mut timings = Timings::default();
        let delta = 1u64 << (64 - p);
        let lut = self.lut_domain(self.p.input_lut_domain, |x| x * delta);
        let now = Instant::now();
        let (a, b) = if rayon::current_num_threads() > 1 {
            let encoded: Vec<_> = a
                .par_iter()
                .chain(b.par_iter())
                .map(|x| {
                    let mut local = Counts::default();
                    let output = self.bootstrap_encoding(x, &lut, &mut local);
                    (output, local)
                })
                .collect();
            let mut values = Vec::with_capacity(encoded.len());
            for (output, local) in encoded {
                values.push(output);
                counts.merge(local);
            }
            let second = values.split_off(a.len());
            (values, second)
        } else {
            (
                a.iter()
                    .map(|x| self.bootstrap_encoding(x, &lut, &mut counts))
                    .collect(),
                b.iter()
                    .map(|x| self.bootstrap_encoding(x, &lut, &mut counts))
                    .collect(),
            )
        };
        timings.input_rescale = now.elapsed().as_secs_f64();
        eprintln!("  input rescaling done");
        let now = Instant::now();
        let packed_a = self.pack(&a);
        let packed_b = self.pack(&b);
        timings.packing = now.elapsed().as_secs_f64();
        counts.packed_lwes = a.len() + b.len();
        eprintln!("  packing done");
        let now = Instant::now();
        let parts = tensor(packed_a.as_ref(), packed_b.as_ref(), 64 - p);
        timings.tensor = now.elapsed().as_secs_f64();
        let now = Instant::now();
        let product = self.relinearize(parts);
        timings.relinearization = now.elapsed().as_secs_f64();
        counts.glwe_products = 1;
        let now = Instant::now();
        let coefficients = (0..a.len()).map(|q| self.extract(&product, q)).collect();
        timings.extraction = now.elapsed().as_secs_f64();
        counts.extracted_coefficients = a.len();
        Kernel {
            rescaled_a: a,
            rescaled_b: b,
            packed_a,
            packed_b,
            coefficients,
            timings,
            counts,
        }
    }

    // LSB-first sign extraction. Scaling precedes key switching so its noise
    // is not multiplied by 2^(p-j-1). No plaintext or secret key is consulted.
    fn coefficient_digits(
        &self,
        input: &Lwe,
        p: usize,
        bits: usize,
        counts: &mut Counts,
    ) -> Vec<Lwe> {
        let mut residual = input.clone();
        let mut digits: Vec<Lwe> = Vec::new();
        for j in 0..bits {
            let weight = 1u64 << (64 - p + j);
            let base = weight.min(NATIVE_DELTA);
            let half = base / 2;
            let lut = allocate_and_trivially_encrypt_new_glwe_ciphertext(
                GlweSize(2),
                &PlaintextList::new(half.wrapping_neg(), PlaintextCount(self.p.polynomial_size)),
                modulus(),
            );
            let mut parity = residual.clone();
            for x in parity.as_mut() {
                *x = x.wrapping_mul(1u64 << (p - j - 1));
            }
            *parity.get_mut_body().data = parity.get_body().data.wrapping_add(1 << 62);
            let mut bit = self.bootstrap(&parity, &lut, counts);
            *bit.get_mut_body().data = bit.get_body().data.wrapping_add(half);
            for (r, &v) in residual.as_mut().iter_mut().zip(bit.as_ref()) {
                *r = r.wrapping_sub(v.wrapping_mul(weight / base));
            }
            for v in bit.as_mut() {
                *v = v.wrapping_mul(NATIVE_DELTA / base);
            }
            if j % 2 == 0 {
                digits.push(bit);
            } else {
                for (x, &y) in digits
                    .last_mut()
                    .unwrap()
                    .as_mut()
                    .iter_mut()
                    .zip(bit.as_ref())
                {
                    *x = x.wrapping_add(y.wrapping_mul(2));
                }
            }
        }
        digits
    }

    pub fn finish(&self, kernel: &mut Kernel, p: usize) -> Vec<Lwe> {
        if self.p.degree_aware {
            return self.finish_range_aware(kernel, p);
        }
        let d = kernel.coefficients.len();
        let mut columns: Vec<Vec<Lwe>> = (0..d).map(|_| Vec::new()).collect();
        let now = Instant::now();
        for (q, coefficient) in kernel.coefficients.iter().enumerate() {
            let bits = bits_needed(9 * (q + 1) as u64).min(2 * (d - q));
            for (j, digit) in self
                .coefficient_digits(coefficient, p, bits, &mut kernel.counts)
                .into_iter()
                .enumerate()
            {
                columns[q + j].push(digit);
            }
        }
        kernel.timings.digit_decomposition = now.elapsed().as_secs_f64();
        eprintln!("  coefficient decomposition done");
        let now = Instant::now();
        let low = self.lut(|x| (x % 4) * NATIVE_DELTA);
        let high = self.lut(|x| (x / 4) * NATIVE_DELTA);
        let mut output = Vec::with_capacity(d);
        for q in 0..d {
            while columns[q].len() > 4 {
                let terms: Vec<_> = columns[q].drain(..4).collect();
                let sum = self.sum(&terms);
                columns[q].push(self.bootstrap(&sum, &low, &mut kernel.counts));
                if q + 1 < d {
                    columns[q + 1].push(self.bootstrap(&sum, &high, &mut kernel.counts));
                }
            }
            let sum = self.sum(&columns[q]);
            output.push(self.bootstrap(&sum, &low, &mut kernel.counts));
            if q + 1 < d && columns[q].len() > 1 {
                columns[q + 1].push(self.bootstrap(&sum, &high, &mut kernel.counts));
            }
        }
        kernel.timings.normalization = now.elapsed().as_secs_f64();
        output
    }

    fn bootstrap_normalizer(&self, input: &Lwe, lut: &Glwe, counts: &mut Counts) -> Lwe {
        let mut scaled = input.clone();
        for value in scaled.as_mut() {
            *value = value.wrapping_mul((16 / self.p.normalizer_lut_domain) as u64);
        }
        self.bootstrap(&scaled, lut, counts)
    }

    fn finish_range_aware(&self, kernel: &mut Kernel, p: usize) -> Vec<Lwe> {
        if rayon::current_num_threads() > 1 {
            return self.finish_range_parallel(kernel, p);
        }
        let d = kernel.coefficients.len();
        let mut columns: Vec<Vec<Lwe>> = (0..d).map(|_| Vec::new()).collect();
        let mut degrees: Vec<Vec<usize>> = vec![Vec::new(); d];
        let now = Instant::now();
        for (q, coefficient) in kernel.coefficients.iter().enumerate() {
            let bound = 9 * (q + 1);
            let bits = bits_needed(bound as u64).min(2 * (d - q));
            for (j, digit) in self
                .coefficient_digits(coefficient, p, bits, &mut kernel.counts)
                .into_iter()
                .enumerate()
            {
                columns[q + j].push(digit);
                degrees[q + j].push((bound >> (2 * j)).min(3));
            }
        }
        kernel.timings.digit_decomposition = now.elapsed().as_secs_f64();
        eprintln!("  coefficient decomposition done");
        let now = Instant::now();
        let domain = self.p.normalizer_lut_domain;
        let low = self.lut_domain(domain, |x| (x % 4) * NATIVE_DELTA);
        let high = self.lut_domain(domain, |x| (x / 4) * NATIVE_DELTA);
        let mut output = Vec::with_capacity(d);
        for q in 0..d {
            while degrees[q].iter().sum::<usize>() >= domain {
                let take = bounded_group_length(&degrees[q], domain - 1);
                assert!(take >= 2);
                let total = degrees[q].drain(..take).sum::<usize>();
                let terms: Vec<_> = columns[q].drain(..take).collect();
                let sum = self.sum(&terms);
                columns[q].push(self.bootstrap_normalizer(&sum, &low, &mut kernel.counts));
                degrees[q].push(total.min(3));
                if q + 1 < d {
                    columns[q + 1].push(self.bootstrap_normalizer(&sum, &high, &mut kernel.counts));
                    degrees[q + 1].push(total / 4);
                }
            }
            let sum = self.sum(&columns[q]);
            output.push(self.bootstrap_normalizer(&sum, &low, &mut kernel.counts));
            let total = degrees[q].iter().sum::<usize>();
            if q + 1 < d && total >= 4 {
                columns[q + 1].push(self.bootstrap_normalizer(&sum, &high, &mut kernel.counts));
                degrees[q + 1].push(total / 4);
            }
        }
        kernel.timings.normalization = now.elapsed().as_secs_f64();
        output
    }

    fn finish_range_parallel(&self, kernel: &mut Kernel, p: usize) -> Vec<Lwe> {
        let d = kernel.coefficients.len();
        let now = Instant::now();
        let expanded: Vec<_> = kernel
            .coefficients
            .par_iter()
            .enumerate()
            .map(|(q, coefficient)| {
                let bits = bits_needed(9 * (q + 1) as u64).min(2 * (d - q));
                let mut counts = Counts::default();
                let digits = self.coefficient_digits(coefficient, p, bits, &mut counts);
                (digits, counts)
            })
            .collect();
        let mut columns: Vec<Vec<Lwe>> = vec![Vec::new(); d];
        for (q, (digits, counts)) in expanded.into_iter().enumerate() {
            kernel.counts.merge(counts);
            for (j, digit) in digits.into_iter().enumerate() {
                columns[q + j].push(digit);
            }
        }
        kernel.timings.digit_decomposition = now.elapsed().as_secs_f64();
        eprintln!("  coefficient decomposition done");
        let now = Instant::now();
        let domain = self.p.normalizer_lut_domain;
        let plan = normalizer_plan(d, domain);
        let low = self.lut_domain(domain, |x| (x % 4) * NATIVE_DELTA);
        let high = self.lut_domain(domain, |x| (x / 4) * NATIVE_DELTA);
        let mut values: Vec<Option<Lwe>> = columns.into_iter().flatten().map(Some).collect();
        assert_eq!(values.len(), plan.initial_count);
        values.resize_with(plan.value_count, || None);
        // Count executed operations; the caller compares them with the public plan.
        let executed_pbs = AtomicUsize::new(0);
        let executed_ks = AtomicUsize::new(0);
        for stage in &plan.stages {
            let results: Vec<_> = stage
                .par_iter()
                .map(|task| {
                    let mut sum = values[task.inputs[0]].as_ref().unwrap().clone();
                    for &input in &task.inputs[1..] {
                        for (x, &y) in sum
                            .as_mut()
                            .iter_mut()
                            .zip(values[input].as_ref().unwrap().as_ref())
                        {
                            *x = x.wrapping_add(y);
                        }
                    }
                    for value in sum.as_mut() {
                        *value = value.wrapping_mul((16 / domain) as u64);
                    }
                    // Low and carry already used identical deterministic KS in
                    // the serial path; reuse it without changing either ciphertext.
                    let mut small = Lwe::new(0, LweSize(self.p.n + 1), modulus());
                    keyswitch_lwe_ciphertext(&self.ksk, &sum, &mut small);
                    executed_ks.fetch_add(1, Ordering::Relaxed);
                    if task.carry.is_some() {
                        let (lo, hi) = rayon::join(
                            || self.bootstrap_small(&small, &low),
                            || self.bootstrap_small(&small, &high),
                        );
                        executed_pbs.fetch_add(2, Ordering::Relaxed);
                        (lo, Some(hi))
                    } else {
                        executed_pbs.fetch_add(1, Ordering::Relaxed);
                        (self.bootstrap_small(&small, &low), None)
                    }
                })
                .collect();
            for (task, (low, carry)) in stage.iter().zip(results) {
                values[task.low] = Some(low);
                if let Some(index) = task.carry {
                    values[index] = carry;
                }
            }
        }
        kernel.counts.pbs += executed_pbs.into_inner();
        kernel.counts.ks += executed_ks.into_inner();
        let output = plan
            .outputs
            .iter()
            .map(|&i| values[i].take().unwrap())
            .collect();
        kernel.timings.normalization = now.elapsed().as_secs_f64();
        output
    }

    fn sum(&self, terms: &[Lwe]) -> Lwe {
        assert!(!terms.is_empty());
        let mut sum = terms[0].clone();
        for t in &terms[1..] {
            for (x, &y) in sum.as_mut().iter_mut().zip(t.as_ref()) {
                *x = x.wrapping_add(y);
            }
        }
        sum
    }

    pub fn key_bytes(&self) -> serde_json::Value {
        serde_json::json!({"fourier_bsk": self.p.n * 4 * self.p.pbs[1] * (self.p.polynomial_size / 2) * 16,
            "encoding_fourier_bsk": if self.encoding_bsk.is_some() { self.p.n * 4 * self.p.encoding_pbs[1] * (self.p.polynomial_size / 2) * 16 } else { 0 },
            "ksk": self.ksk.as_ref().len() * 8, "packing": self.packing.as_ref().len() * 8,
            "relinearization": self.relin.iter().map(|k| k.as_ref().len() * 8).sum::<usize>()})
    }
}
