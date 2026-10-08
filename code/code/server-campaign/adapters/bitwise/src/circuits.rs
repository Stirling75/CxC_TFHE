use rayon::prelude::*;

pub trait Gates: Sync {
    type Bit: Clone + Send + Sync;
    fn zero(&self) -> Self::Bit;
    fn and(&self, a: &Self::Bit, b: &Self::Bit) -> Self::Bit;
    fn xor(&self, a: &Self::Bit, b: &Self::Bit) -> Self::Bit;
    fn parity3(&self, a: &Self::Bit, b: &Self::Bit, c: &Self::Bit) -> Self::Bit;
    fn sum_carry(&self, a: &Self::Bit, b: &Self::Bit, c: &Self::Bit) -> (Self::Bit, Self::Bit);
}

// Morshed's cpuParallel/Cipher.cpp::addBits: four XORs and one AND.
pub fn morshed_adder<G: Gates>(g: &G, a: &G::Bit, b: &G::Bit, c: &G::Bit) -> (G::Bit, G::Bit) {
    let t1 = g.xor(a, c);
    let t2 = g.xor(b, c);
    let sum = g.xor(a, &t2);
    let both = g.and(&t1, &t2);
    (sum, g.xor(c, &both))
}

fn ripple_add<G: Gates>(g: &G, a: &[G::Bit], b: &[G::Bit]) -> Vec<G::Bit> {
    assert_eq!(a.len(), b.len());
    let mut carry = g.zero();
    a.iter()
        .zip(b)
        .map(|(a, b)| {
            let (sum, next) = morshed_adder(g, a, b, &carry);
            carry = next;
            sum
        })
        .collect()
}

pub fn morshed<G: Gates>(g: &G, x: &[G::Bit], y: &[G::Bit], threads: usize) -> Vec<G::Bit> {
    let width = x.len();
    assert!(width > 0 && threads > 0);
    assert_eq!(width, y.len());
    let groups = threads.min(width);
    // Static partial-product assignment, followed by a deterministic reduction.
    // Each private accumulator has the full 2W width, not the author's fixed 32.
    let mut sums: Vec<_> = (0..groups)
        .into_par_iter()
        .map(|group| {
            let mut acc = vec![g.zero(); 2 * width];
            for i in group * width / groups..(group + 1) * width / groups {
                let mut partial = vec![g.zero(); 2 * width];
                for j in 0..width {
                    partial[i + j] = g.and(&x[j], &y[i]);
                }
                acc = ripple_add(g, &acc, &partial);
            }
            acc
        })
        .collect();
    while sums.len() > 1 {
        sums = sums
            .par_chunks(2)
            .map(|pair| {
                if pair.len() == 2 {
                    ripple_add(g, &pair[0], &pair[1])
                } else {
                    pair[0].clone()
                }
            })
            .collect();
    }
    sums.pop().unwrap()
}

pub fn trifan<G: Gates>(g: &G, x: &[G::Bit], y: &[G::Bit]) -> Vec<G::Bit> {
    let width = x.len();
    assert!(width > 0);
    assert_eq!(width, y.len());
    let mut row = vec![g.zero(); width];
    let mut carry = vec![g.zero(); width];
    let mut result = Vec::with_capacity(width);
    for bit in y {
        let partial: Vec<_> = x.par_iter().map(|a| g.and(a, bit)).collect();
        let reduced: Vec<_> = (0..width)
            .into_par_iter()
            .map(|j| g.sum_carry(&row[j], &partial[j], &carry[j]))
            .collect();
        result.push(reduced[0].0.clone());
        row = reduced.iter().skip(1).map(|v| v.0.clone()).collect();
        row.push(g.zero());
        carry = reduced.into_iter().map(|v| v.1).collect();
    }
    result
}

// At step i, only W-i columns can still affect the lower-W-bit result.
pub fn trifan_pruned<G: Gates>(g: &G, x: &[G::Bit], y: &[G::Bit]) -> Vec<G::Bit> {
    let width = x.len();
    assert!(width > 0);
    assert_eq!(width, y.len());
    let mut row = vec![g.zero(); width];
    let mut carry = row.clone();
    let mut result = Vec::with_capacity(width);
    for bit in y {
        let active = row.len();
        let reduced: Vec<_> = (0..active)
            .into_par_iter()
            .map(|j| {
                let partial = g.and(&x[j], bit);
                if j + 1 == active {
                    (g.parity3(&row[j], &partial, &carry[j]), None)
                } else {
                    let (sum, next) = g.sum_carry(&row[j], &partial, &carry[j]);
                    (sum, Some(next))
                }
            })
            .collect();
        result.push(reduced[0].0.clone());
        row = reduced.iter().skip(1).map(|v| v.0.clone()).collect();
        carry = reduced.into_iter().filter_map(|v| v.1).collect();
    }
    result
}

pub fn logical_pbs(method: &str, width: usize, threads: usize) -> u64 {
    match method {
        "trifan" => (3 * width * width) as u64,
        "trifan-pruned" => ((3 * width * width + width) / 2) as u64,
        "morshed" => (width * width + 10 * width * (width + threads.min(width) - 1)) as u64,
        _ => panic!("unknown circuit"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigUint;
    use std::sync::atomic::{AtomicU64, Ordering};

    #[derive(Default)]
    struct Plain(AtomicU64);
    impl Gates for Plain {
        type Bit = u8;
        fn zero(&self) -> u8 {
            0
        }
        fn and(&self, a: &u8, b: &u8) -> u8 {
            self.0.fetch_add(1, Ordering::Relaxed);
            a & b
        }
        fn xor(&self, a: &u8, b: &u8) -> u8 {
            self.0.fetch_add(1, Ordering::Relaxed);
            a ^ b
        }
        fn parity3(&self, a: &u8, b: &u8, c: &u8) -> u8 {
            self.0.fetch_add(1, Ordering::Relaxed);
            (a + b + c) & 1
        }
        fn sum_carry(&self, a: &u8, b: &u8, c: &u8) -> (u8, u8) {
            self.0.fetch_add(2, Ordering::Relaxed);
            let s = a + b + c;
            (s & 1, s >> 1)
        }
    }

    fn value(bits: &[u8]) -> BigUint {
        bits.iter()
            .rev()
            .fold(BigUint::from(0u8), |a, b| (a << 1) + b)
    }

    fn check(x: &[u8], y: &[u8], threads: usize) {
        let g = Plain::default();
        let expected = value(x) * value(y);
        assert_eq!(value(&morshed(&g, x, y, threads)), expected);
        assert_eq!(
            g.0.swap(0, Ordering::Relaxed),
            logical_pbs("morshed", x.len(), threads)
        );
        let mask = (BigUint::from(1u8) << x.len()) - 1u8;
        assert_eq!(value(&trifan(&g, x, y)), &expected & &mask);
        assert_eq!(
            g.0.swap(0, Ordering::Relaxed),
            logical_pbs("trifan", x.len(), threads)
        );
        assert_eq!(value(&trifan_pruned(&g, x, y)), expected & mask);
        assert_eq!(
            g.0.load(Ordering::Relaxed),
            logical_pbs("trifan-pruned", x.len(), threads)
        );
    }

    // Mirrors ShortintGates: a LUT input is trivial iff all summed inputs are.
    #[derive(Default)]
    struct Triviality {
        logical: AtomicU64,
        real: AtomicU64,
    }
    impl Triviality {
        fn lut(&self, inputs: &[bool], calls: u64) -> bool {
            let trivial = inputs.iter().all(|t| *t);
            self.logical.fetch_add(calls, Ordering::Relaxed);
            if !trivial {
                self.real.fetch_add(calls, Ordering::Relaxed);
            }
            trivial
        }
    }
    impl Gates for Triviality {
        type Bit = bool;
        fn zero(&self) -> bool {
            true
        }
        fn and(&self, a: &bool, b: &bool) -> bool {
            self.lut(&[*a, *b], 1)
        }
        fn xor(&self, a: &bool, b: &bool) -> bool {
            self.lut(&[*a, *b], 1)
        }
        fn parity3(&self, a: &bool, b: &bool, c: &bool) -> bool {
            self.lut(&[*a, *b, *c], 1)
        }
        fn sum_carry(&self, a: &bool, b: &bool, c: &bool) -> (bool, bool) {
            let t = self.lut(&[*a, *b, *c], 2);
            (t, t)
        }
    }

    // Pinned (logical, real) values shared with test_bitwise_check.py.
    #[test]
    fn real_pbs_split_is_pinned() {
        let inputs = vec![false; 16];
        for (method, threads, logical, real) in [
            ("morshed", 1, 2816, 2800),
            ("morshed", 4, 3296, 2757),
            ("trifan", 4, 768, 768),
            ("trifan-pruned", 4, 392, 392),
        ] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let g = Triviality::default();
            pool.install(|| match method {
                "morshed" => drop(morshed(&g, &inputs, &inputs, threads)),
                "trifan" => drop(trifan(&g, &inputs, &inputs)),
                _ => drop(trifan_pruned(&g, &inputs, &inputs)),
            });
            assert_eq!(g.logical.load(Ordering::Relaxed), logical);
            assert_eq!(logical, logical_pbs(method, 16, threads));
            assert_eq!(g.real.load(Ordering::Relaxed), real, "{method} T={threads}");
        }
    }

    #[test]
    fn all_small_products_and_full_adders() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        pool.install(|| {
            let g = Plain::default();
            for a in 0..2 {
                for b in 0..2 {
                    for c in 0..2 {
                        assert_eq!(morshed_adder(&g, &a, &b, &c), g.sum_carry(&a, &b, &c));
                    }
                }
            }
            for width in 1..=4 {
                for x in 0..1u32 << width {
                    for y in 0..1u32 << width {
                        let bits =
                            |v: u32| (0..width).map(|j| ((v >> j) & 1) as u8).collect::<Vec<_>>();
                        check(&bits(x), &bits(y), 1);
                    }
                }
            }
        });
    }

    #[test]
    fn wider_products_and_parallel_reduction() {
        for threads in [1, 2, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                for width in [8, 16, 32, 64, 128, 256] {
                    let ones = vec![1; width];
                    let alternating = (0..width).map(|j| (j % 2) as u8).collect::<Vec<_>>();
                    check(&ones, &ones, threads);
                    check(&ones, &alternating, threads);
                }
            });
        }
    }
}
