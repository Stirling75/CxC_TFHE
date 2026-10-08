use super::crypto::{self, EvaluationKeys, Glwe, Lwe, Selector, BIT_DELTA};
use super::plan::Plan;
use rayon::prelude::*;
use tfhe::core_crypto::prelude::*;

pub type Values = Vec<Option<Lwe>>;

fn decode_refined_byte(bits: usize) -> usize {
    (0..4).map(|i| {
        let y0 = (bits >> (2 * i)) & 1;
        let y1 = (bits >> (2 * i + 1)) & 1;
        ((y0 ^ y1) + 2 * y1) << (2 * i)
    }).sum()
}

pub fn input_selectors(keys: &EvaluationKeys, inputs: &[Lwe], delta_log2: u32) -> Vec<[Selector; 2]> {
    inputs.par_iter().map(|c| crypto::lift_grouped(keys, c, delta_log2)).collect()
}

fn product_tables(keys: &EvaluationKeys) -> Vec<Vec<Glwe>> {
    let n = keys.params.polynomial_size;
    (0..16).map(|bit| {
        (0..65536 / n).map(|bank| {
            let body: Vec<_> = (0..n).map(|coefficient| {
                let code = bank * n + coefficient;
                let x = decode_refined_byte(code & 255);
                let y = decode_refined_byte(code >> 8);
                (((x * y) >> bit) & 1) as u64 * BIT_DELTA
            }).collect();
            allocate_and_trivially_encrypt_new_glwe_ciphertext(
                keys.bsk.glwe_size(), &PlaintextList::from_container(body), crypto::modulus())
        }).collect()
    }).collect()
}

pub fn products(keys: &EvaluationKeys, plan: &Plan, lhs: &[[Selector; 2]], rhs: &[[Selector; 2]]) -> Values {
    let tables = product_tables(keys);
    let low_bits = keys.params.polynomial_size.ilog2() as usize;
    let outputs: Vec<_> = plan.products.par_iter().map(|job| {
        let selector = |bit: usize| -> &Selector {
            if bit < 8 { &lhs[4 * job.left + bit / 2][bit % 2] }
            else { &rhs[4 * job.right + (bit - 8) / 2][bit % 2] }
        };
        // The tables are built inside the product timer (above). The first
        // CMux level reads the shared bank by reference instead of cloning
        // about 1 MiB per product bit; later levels consume owned GLWEs. The
        // CMux sequence and every output are unchanged.
        let table = &tables[job.bit];
        let half = table.len() / 2;
        let mut bank: Vec<Glwe> = (0..half)
            .map(|i| crypto::cmux(keys, selector(15), &table[i], &table[i + half])).collect();
        for bit in (low_bits..15).rev() {
            let half = bank.len() / 2;
            let upper = bank.split_off(half);
            bank = bank.into_iter().zip(upper)
                .map(|(zero, one)| crypto::cmux_owned(keys, selector(bit), zero, one)).collect();
        }
        assert_eq!(bank.len(), 1);
        let mut selected = bank.pop().unwrap();
        for bit in (0..low_bits).rev() {
            let rotated = crypto::rotate_div(&selected, 1 << bit);
            selected = crypto::cmux_owned(keys, selector(bit), selected, rotated);
        }
        crypto::extract(&selected, 0)
    }).collect();
    let mut values = vec![None; plan.value_count];
    for (id, out) in outputs.into_iter().enumerate() { values[id] = Some(out); }
    values
}

pub fn sum(keys: &EvaluationKeys, values: &Values, ids: &[usize]) -> Lwe {
    let mut out = crypto::zero_lwe(keys);
    for &id in ids { lwe_ciphertext_add_assign(&mut out, values[id].as_ref().unwrap()); }
    out
}

pub fn reduce(keys: &EvaluationKeys, plan: &Plan, values: &mut Values) {
    for layer in &plan.layers {
        let results: Vec<_> = layer.jobs.par_iter().map(|job| {
            crypto::compress(keys, &sum(keys, values, &job.inputs))
        }).collect();
        for (job, (lo, hi)) in layer.jobs.iter().zip(results) {
            values[job.parity] = Some(lo);
            if let Some(id) = job.carry { values[id] = Some(hi); }
        }
    }
}

fn add_output_bit(input: &Glwe, column: usize) -> Glwe {
    let mut out = input.clone();
    let mut body = out.get_mut_body();
    body.as_mut()[column / 2] = input.get_body().as_ref()[column / 2]
        .wrapping_add(1u64 << (61 + column % 2));
    out
}

fn suffix_step(keys: &EvaluationKeys, states: (Glwe, Glwe), lo: &Selector,
               previous_high: Option<&Selector>, column: usize) -> (Glwe, Glwe) {
    let (a, c) = states;
    let a_plus_bit = add_output_bit(&a, column);
    // The two CMux gates of each level are independent; run them in parallel.
    let (u0, u1) = rayon::join(|| crypto::cmux(keys, lo, &a, &a_plus_bit),
                               || crypto::cmux(keys, lo, &a_plus_bit, &c));
    if let Some(high) = previous_high {
        // U2 = C + lo*bit reuses U0. This gives four CMux gates in two
        // levels for both incoming-carry hypotheses, rather than a 3-input LUT.
        let mut u2 = u0.clone();
        glwe_ciphertext_add_assign(&mut u2, &c);
        glwe_ciphertext_sub_assign(&mut u2, &a);
        rayon::join(|| crypto::cmux(keys, high, &u0, &u1), || crypto::cmux_owned(keys, high, u1.clone(), u2))
    } else { (u0, u1) }
}

pub fn terminal(keys: &EvaluationKeys, plan: &Plan, values: &Values) -> Vec<Lwe> {
    // The high bit of column c feeds only column c+1. For the top column W-1
    // it would feed column W, outside the lower-W output, so it is not read.
    let reads: Vec<_> = plan.final_columns[8..].par_iter().enumerate().map(|(offset, ids)| {
        let small = crypto::small_input(keys, &sum(keys, values, ids));
        if ids.len() > 1 && offset + 9 < plan.width {
            let (lo, hi) = rayon::join(|| crypto::lift_binary(keys, &small, false),
                                      || crypto::lift_binary(keys, &small, true));
            (lo, Some(hi))
        } else { (crypto::lift_binary(keys, &small, false), None) }
    }).collect();
    let mut states = (crypto::zero_glwe(keys), crypto::zero_glwe(keys));
    for column in (8..plan.width).rev() {
        let previous_high = if column > 8 { reads[column - 9].1.as_ref() } else { None };
        states = suffix_step(keys, states, &reads[column - 8].0, previous_high, column);
    }
    (0..plan.width / 2).map(|block| {
        if block < 4 {
            let mut low = sum(keys, values, &plan.final_columns[2 * block]);
            let high = sum(keys, values, &plan.final_columns[2 * block + 1]);
            lwe_ciphertext_add_assign(&mut low, &high);
            lwe_ciphertext_add_assign(&mut low, &high);
            low
        } else { crypto::extract(&states.0, block) }
    }).collect()
}

pub fn emission(keys: &EvaluationKeys, inputs: &[Lwe], delta_log2: u32) -> Vec<(Lwe, Lwe)> {
    inputs.par_iter().map(|c| crypto::emit(keys, c, delta_log2)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn four_cmux_terminal_step_matches_all_successors() {
        let cmux = |b, a, c| if b { c } else { a };
        for a in [0i64, 32, 96] {
            for c in [0i64, 64, 128] {
                for bit in [1, 2, 4, 8] {
                    for lo in [false, true] {
                        for hi in [false, true] {
                            let u0 = cmux(lo, a, a + bit);
                            let u1 = cmux(lo, a + bit, c);
                            let u2 = u0 + c - a;
                            let out = [cmux(hi, u0, u1), cmux(hi, u1, u2)];
                            for incoming in 0..=1 {
                                let total = lo as usize + hi as usize + incoming;
                                assert_eq!(out[incoming], [a, c][total / 2] + (total % 2) as i64 * bit);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn product_table_matches_every_refined_pair() {
        for code in 0..65536 {
            let a = decode_refined_byte(code & 255);
            let b = decode_refined_byte(code >> 8);
            let decode = |raw: usize| (0..4).fold(0, |out, i| {
                let y = (raw >> (2 * i)) & 3;
                out | ([0, 1, 3, 2][y] << (2 * i))
            });
            assert_eq!(a * b, decode(code & 255) * decode(code >> 8));
        }
    }
}
