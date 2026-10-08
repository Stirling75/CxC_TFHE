use super::*;

struct Scratch {
    fft: Fft,
    memory: PodBuffer,
    delta: GlweCiphertextOwned<u64>,
    product: GlweCiphertextOwned<u64>,
    moved: GlweCiphertextOwned<u64>,
    work: Work,
}

impl Scratch {
    fn new(params: BenchParams) -> Self {
        let fft = Fft::new(params.polynomial_size);
        let requirement = cmux_assign_mem_optimized_requirement::<u64>(
            params.glwe_dimension.to_glwe_size(), params.polynomial_size, fft.as_view());
        Self { fft, memory: PodBuffer::try_new(requirement).expect("CMux scratch"),
            delta: zero_glwe(params), product: zero_glwe(params), moved: zero_glwe(params),
            work: Work::default() }
    }

    fn select(&mut self, zero: &mut GlweCiphertextOwned<u64>, one: &mut GlweCiphertextOwned<u64>,
              bits: &Selectors, bit: usize) {
        let selector = bits.as_view().into_ggsw_iter().nth(bit).expect("radix selector");
        cmux_assign_mem_optimized::<u64, _, _, _>(zero, one, &selector,
            self.fft.as_view(), &mut PodStack::new(&mut self.memory));
        self.work.cmux += 1;
        record_cmux();
    }

    fn left_prefix(&mut self, state: &GlweCiphertextOwned<u64>, a: &Selectors,
                   stride: usize) -> Vec<GlweCiphertextOwned<u64>> {
        let mut branches = vec![state.clone()];
        for b in 1..=3 {
            let shift = stride * b;
            monomial_div_into(&mut self.delta, state, MonomialDegree(shift));
            for (dst, &src) in self.delta.as_mut().iter_mut().zip(state.as_ref()) {
                *dst = dst.wrapping_sub(src);
            }
            self.product.as_mut().fill(0);
            let y0 = a.as_view().into_ggsw_iter().next().expect("low refined selector");
            add_external_product_assign_mem_optimized::<u64, _, _, _>(
                &mut self.product, &y0, &self.delta, self.fft.as_view(),
                &mut PodStack::new(&mut self.memory));
            self.work.direct += 1;

            // R=X^(-stride*b), a=y0+3*y1-2*y0*y1, E=y0*(R-I)C.
            // The y1 branches are C+E and R^3 C-R^2 E. Reuse E itself;
            // do not assume decomposition commutes with subtraction.
            let mut low = state.clone();
            for (dst, &src) in low.as_mut().iter_mut().zip(self.product.as_ref()) {
                *dst = dst.wrapping_add(src);
            }
            let mut high = state.clone();
            monomial_div_into(&mut high, state, MonomialDegree(3*shift));
            monomial_div_into(&mut self.moved, &self.product, MonomialDegree(2*shift));
            for (dst, &src) in high.as_mut().iter_mut().zip(self.moved.as_ref()) {
                *dst = dst.wrapping_sub(src);
            }
            self.select(&mut low, &mut high, a, 1);
            branches.push(low);
        }
        branches
    }

    fn right_query(&mut self, mut branches: Vec<GlweCiphertextOwned<u64>>,
                   b: &Selectors) -> GlweCiphertextOwned<u64> {
        // Binary index (y0,y1) orders the decoded radix values as 0,3,1,2.
        branches.swap(1, 3);
        branches.swap(2, 3);
        let (zero, one) = branches.split_at_mut(2);
        for (zero, one) in zero.iter_mut().zip(one) { self.select(zero, one, b, 0); }
        branches.truncate(2);
        let (zero, one) = branches.split_at_mut(1);
        self.select(&mut zero[0], &mut one[0], b, 1);
        branches.swap_remove(0)
    }
}

impl Scratch {
    /// One product with a public digit `b` of the second operand: only the
    /// candidate v=b of `left_prefix` is needed, i.e. one direct external
    /// product and one CMux on the selectors of the encrypted digit.
    fn public_step(&mut self, state: &GlweCiphertextOwned<u64>, a: &Selectors,
                   stride: usize, b: usize) -> GlweCiphertextOwned<u64> {
        if b == 0 { return state.clone(); }
        let shift = stride * b;
        monomial_div_into(&mut self.delta, state, MonomialDegree(shift));
        for (dst, &src) in self.delta.as_mut().iter_mut().zip(state.as_ref()) {
            *dst = dst.wrapping_sub(src);
        }
        self.product.as_mut().fill(0);
        let y0 = a.as_view().into_ggsw_iter().next().expect("low refined selector");
        add_external_product_assign_mem_optimized::<u64, _, _, _>(
            &mut self.product, &y0, &self.delta, self.fft.as_view(),
            &mut PodStack::new(&mut self.memory));
        self.work.direct += 1;
        let mut low = state.clone();
        for (dst, &src) in low.as_mut().iter_mut().zip(self.product.as_ref()) {
            *dst = dst.wrapping_add(src);
        }
        let mut high = state.clone();
        monomial_div_into(&mut high, state, MonomialDegree(3*shift));
        monomial_div_into(&mut self.moved, &self.product, MonomialDegree(2*shift));
        for (dst, &src) in high.as_mut().iter_mut().zip(self.moved.as_ref()) {
            *dst = dst.wrapping_sub(src);
        }
        self.select(&mut low, &mut high, a, 1);
        low
    }
}

/// Ciphertext-plaintext product-sum lookup: `scalar[j]` are the public radix-4
/// digits of the second operand. Groups, tables, and digit bounds are those of
/// the ciphertext-ciphertext kernel, so the same restoration plan applies.
pub(super) fn evaluate_public(lhs: &[Selectors], scalar: &[u64], params: BenchParams)
    -> Result<Evaluation> {
    let started = Instant::now();
    let d = lhs.len();
    let capacity = crate::config::fused_products_per_group();
    assert_eq!(d, scalar.len());
    assert!((1..=45).contains(&capacity));
    assert!(scalar.iter().all(|&b| b < 4));
    let tables: Vec<_> = (1..=capacity.min(d)).map(|r| public_table(r, params)).collect();
    let jobs: Vec<_> = (0..d).flat_map(|q| (0..=q).step_by(capacity)
        .map(move |offset| (q, offset, capacity.min(q+1-offset)))).collect();
    let output_size = params.glwe_dimension.to_equivalent_lwe_dimension(params.polynomial_size).to_lwe_size();
    let query = |scratch: &mut Scratch, (q, offset, length): (usize, usize, usize)| {
        scratch.work = Work::default();
        let stride = output_digits(length);
        let mut state = tables[length-1].clone();
        for i in offset..offset+length {
            state = scratch.public_step(&state, &lhs[i], stride, scalar[q-i] as usize);
        }
        let mut digits = extract_clean_digits(&state, q, (q, offset), stride, d, output_size, params);
        for (_, term) in &mut digits {
            term.bound = ((9*length) >> (2*term.source.unwrap().2)).min(3) as u64;
        }
        (digits, scratch.work)
    };
    let cells: Vec<_> = if parallel_cmux_cells() {
        jobs.into_par_iter().map_init(|| Scratch::new(params), query).collect()
    } else {
        let mut scratch = Scratch::new(params);
        jobs.into_iter().map(|job| query(&mut scratch, job)).collect()
    };
    let mut work = Work::default();
    let mut columns: Vec<Vec<_>> = (0..d).map(|_| Vec::new()).collect();
    for (cell, count) in cells {
        work.add(count);
        for (q, term) in cell { columns[q].push(term); }
    }
    Ok(Evaluation { columns, work, elapsed_ms: duration_ms(started.elapsed()) })
}

/// Scalar-aware ciphertext-plaintext lookup: the public plan groups the
/// products with nonzero scalar digits, and a group sum is bounded by
/// 3*sum(b), which decides its table, its digits, and their bounds.
pub(super) fn evaluate_scalar_aware(lhs: &[Selectors], scalar: &[u64], groups: &[Vec<Vec<usize>>],
                                    params: BenchParams) -> Result<Evaluation> {
    let started = Instant::now();
    let d = lhs.len();
    assert_eq!(d, scalar.len());
    assert_eq!(d, groups.len());
    let jobs: Vec<_> = groups.iter().enumerate()
        .flat_map(|(q, column)| column.iter().map(move |group| (q, group.as_slice()))).collect();
    for &(q, group) in &jobs {
        assert!(!group.is_empty() && group.iter().all(|&i| i <= q && scalar[q-i] != 0));
    }
    let output_size = params.glwe_dimension.to_equivalent_lwe_dimension(params.polynomial_size).to_lwe_size();
    let query = |scratch: &mut Scratch, (q, group): (usize, &[usize])| {
        scratch.work = Work::default();
        let maximum = 3 * group.iter().map(|&i| scalar[q-i] as usize).sum::<usize>();
        let stride = super::public_digits(maximum);
        let mut state = super::sum_table(maximum, params);
        for &i in group {
            state = scratch.public_step(&state, &lhs[i], stride, scalar[q-i] as usize);
        }
        let mut digits = extract_clean_digits(&state, q, (q, group[0]), stride, d, output_size, params);
        for (_, term) in &mut digits {
            term.bound = (maximum >> (2*term.source.unwrap().2)).min(3) as u64;
        }
        (digits, scratch.work)
    };
    let cells: Vec<_> = if parallel_cmux_cells() {
        jobs.into_par_iter().map_init(|| Scratch::new(params), query).collect()
    } else {
        let mut scratch = Scratch::new(params);
        jobs.into_iter().map(|job| query(&mut scratch, job)).collect()
    };
    let mut work = Work::default();
    let mut columns: Vec<Vec<_>> = (0..d).map(|_| Vec::new()).collect();
    for (cell, count) in cells {
        work.add(count);
        for (q, term) in cell { columns[q].push(term); }
    }
    Ok(Evaluation { columns, work, elapsed_ms: duration_ms(started.elapsed()) })
}

pub(super) fn evaluate(lhs: &[Selectors], rhs: &[Selectors], params: BenchParams,
                      use_cache: bool) -> Result<Evaluation> {
    let started = Instant::now();
    let d = lhs.len();
    let capacity = crate::config::fused_products_per_group();
    assert_eq!(d, rhs.len());
    assert!((1..=45).contains(&capacity));
    let tables: Vec<_> = (1..=capacity.min(d)).map(|r| public_table(r, params)).collect();
    let offsets: Vec<_> = (0..d).step_by(capacity).collect();
    let build = |scratch: &mut Scratch, &offset: &usize| {
        scratch.work = Work::default();
        let entry = if use_cache && offset+capacity < d {
            let prefix = scratch.left_prefix(&tables[capacity-1], &lhs[offset], output_digits(capacity));
            scratch.work.cache_entries = 1;
            Some(prefix)
        } else { None };
        (entry, scratch.work)
    };
    let entries: Vec<_> = if parallel_cmux_cells() {
        offsets.par_iter().map_init(|| Scratch::new(params), build).collect()
    } else {
        let mut scratch = Scratch::new(params);
        offsets.iter().map(|off| build(&mut scratch, off)).collect()
    };
    let mut work = Work::default();
    for (_, count) in &entries { work.add(*count); }
    let jobs: Vec<_> = (0..d).flat_map(|q| (0..=q).step_by(capacity)
        .map(move |offset| (q, offset, capacity.min(q+1-offset)))).collect();
    let output_size = params.glwe_dimension.to_equivalent_lwe_dimension(params.polynomial_size).to_lwe_size();
    let query = |scratch: &mut Scratch, (q, offset, length): (usize, usize, usize)| {
        scratch.work = Work::default();
        let stride = output_digits(length);
        let mut state = tables[length-1].clone();
        let mut first = offset;
        if length == capacity {
            if let Some(prefix) = &entries[offset/capacity].0 {
                state = scratch.right_query(prefix.clone(), &rhs[q-offset]);
                scratch.work.cache_hits += 1;
                first += 1;
            }
        }
        for i in first..offset+length {
            let prefix = scratch.left_prefix(&state, &lhs[i], stride);
            state = scratch.right_query(prefix, &rhs[q-i]);
        }
        let mut digits = extract_clean_digits(&state, q, (q, offset), stride, d, output_size, params);
        for (_, term) in &mut digits {
            term.bound = ((9*length) >> (2*term.source.unwrap().2)).min(3) as u64;
        }
        (digits, scratch.work)
    };
    let cells: Vec<_> = if parallel_cmux_cells() {
        jobs.into_par_iter().map_init(|| Scratch::new(params), query).collect()
    } else {
        let mut scratch = Scratch::new(params);
        jobs.into_iter().map(|job| query(&mut scratch, job)).collect()
    };
    let mut columns: Vec<Vec<_>> = (0..d).map(|_| Vec::new()).collect();
    for (cell, count) in cells {
        work.add(count);
        for (q, term) in cell { columns[q].push(term); }
    }
    Ok(Evaluation { columns, work, elapsed_ms: duration_ms(started.elapsed()) })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rotate(c: &[u64], shift: usize) -> Vec<u64> {
        (0..c.len()).map(|t| {
            let v = c[(t+shift)%c.len()];
            if ((t+shift)/c.len())%2 == 0 { v } else { v.wrapping_neg() }
        }).collect()
    }

    #[test]
    fn reused_prefix_identity_holds_for_every_refined_digit_and_rotation() {
        let c: Vec<_> = (0u64..32).map(|i| i.wrapping_mul(0x9e3779b97f4a7c15)).collect();
        for stride in 1..=6 {
            for y0 in 0..=1 {
                for y1 in 0..=1 {
                    for b in 1..=3 {
                        let shift = stride*b;
                        let r = rotate(&c, shift);
                        let e: Vec<_> = r.iter().zip(&c).map(|(&r,&c)|
                            if y0 == 1 { r.wrapping_sub(c) } else { 0 }).collect();
                        let r3 = rotate(&c, 3*shift);
                        let r2e = rotate(&e, 2*shift);
                        let result: Vec<_> = (0..c.len()).map(|i| if y1 == 0 {
                            c[i].wrapping_add(e[i])
                        } else { r3[i].wrapping_sub(r2e[i]) }).collect();
                        assert_eq!(result, rotate(&c, shift*decode_refined_digit(y0,y1)));
                    }
                }
            }
        }
    }
}
