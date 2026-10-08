//! Public key-coordinate maps for the even/odd subring embedding.

pub(crate) fn even_odd_secret(secret: &[u64]) -> Vec<u64> {
    assert!(!secret.is_empty() && secret.len() % 2 == 0);
    secret.iter().step_by(2).chain(secret.iter().skip(1).step_by(2)).copied().collect()
}

pub(crate) fn restore_extracted_key(ciphertext: &mut [u64]) {
    assert!(ciphertext.len() >= 3 && ciphertext.len() % 2 == 1);
    let n = (ciphertext.len() - 1) / 2;
    let original = ciphertext[..2 * n].to_vec();
    for i in 0..n {
        ciphertext[2 * i] = original[i];
        ciphertext[2 * i + 1] = original[n + i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn multiply(a: &[u64], b: &[u64]) -> Vec<u64> {
        let n = a.len();
        let mut out = vec![0u64; n];
        for (i, &x) in a.iter().enumerate() {
            for (j, &y) in b.iter().enumerate() {
                let product = x.wrapping_mul(y);
                out[(i + j) % n] = if i + j < n {
                    out[(i + j) % n].wrapping_add(product)
                } else { out[(i + j) % n].wrapping_sub(product) };
            }
        }
        out
    }

    #[test]
    fn extracted_lwe_phase_is_preserved_exactly() {
        for n in [2, 4, 8, 1024] {
            let secret: Vec<_> = (0..2*n).map(|i| (i % 3 == 0) as u64).collect();
            let split = even_odd_secret(&secret);
            let mut lwe: Vec<_> = (0..2*n+1).map(|i| (i as u64).wrapping_mul(0xD1B54A32D192ED03)).collect();
            let dot = |a: &[u64], s: &[u64]| a.iter().zip(s).fold(0u64, |z, (&a, &s)| z.wrapping_add(a.wrapping_mul(s)));
            let phase = lwe[2*n].wrapping_sub(dot(&lwe, &split));
            let body = lwe[2*n];
            restore_extracted_key(&mut lwe);
            assert_eq!(body, lwe[2*n]);
            assert_eq!(phase, body.wrapping_sub(dot(&lwe, &secret)));
        }
    }

    #[test]
    fn even_rlwe_coefficients_form_the_two_component_glwe() {
        for n in [2, 4, 8, 16] {
            let secret: Vec<_> = (0..2*n).map(|i| (i % 3 == 1) as u64).collect();
            let mask: Vec<_> = (0..2*n).map(|i| (i as u64).wrapping_mul(0x9E3779B97F4A7C15)).collect();
            let ring_product = multiply(&mask, &secret);
            let s = even_odd_secret(&secret);
            let a = even_odd_secret(&mask);
            let mut y_a_odd = vec![0u64; n];
            y_a_odd[0] = a[2*n-1].wrapping_neg();
            y_a_odd[1..].copy_from_slice(&a[n..2*n-1]);
            let p0 = multiply(&a[..n], &s[..n]);
            let p1 = multiply(&y_a_odd, &s[n..]);
            for i in 0..n {
                assert_eq!(p0[i].wrapping_add(p1[i]), ring_product[2*i]);
            }
        }
    }
}
