//! Secret-key-assisted diagnostic, called only by the harness's audit closure.
use super::*;
use crate::keys::{ggsw_row_error_at, glwe_phase_at, negative_secret_product, HarnessSecretKeys};
use std::fs::OpenOptions;

pub(crate) fn audit(
    lhs: &[Selectors],
    rhs: &[Selectors],
    x: &[u64],
    y: &[u64],
    secrets: &HarnessSecretKeys,
    params: BenchParams,
    label: &str,
) -> Result<()> {
    let path = std::env::var("FUSED_STEP_AUDIT_CSV")?;
    audit_selector_rows(
        lhs,
        rhs,
        x,
        y,
        secrets,
        params,
        label,
        &format!("{path}.selectors.csv"),
    )?;
    let header = !std::path::Path::new(&path).exists();
    let mut writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(OpenOptions::new().create(true).append(true).open(path)?);
    if header {
        writer.write_record([
            "trial",
            "q",
            "offset",
            "length",
            "step",
            "level",
            "digit",
            "bit",
            "stage",
            "residual",
            "increment",
            "gadget",
            "key_fft",
        ])?;
    }
    let cap = crate::config::fused_products_per_group();
    assert!(cap > 0);
    let d = x.len();
    let mut diagonals = vec![cap.min(d) - 1, d - 1];
    diagonals.sort_unstable();
    diagonals.dedup();
    let mut scratch = Scratch::new(params);
    let (_, _, _, _, base, level, _) = params.direct_lift_decomposition(ProductVariant::Hybrid8x8);
    let decomposer = SignedDecomposer::<u64>::new(base, level);
    for q in diagonals {
        for offset in (0..=q).step_by(cap) {
            let length = cap.min(q + 1 - offset);
            let stride = output_digits(length);
            let total: usize = (offset..offset + length)
                .map(|i| (x[i] * y[q - i]) as usize)
                .sum();
            let mut state = public_table(length, params);
            let mut partial = 0;
            let mut previous = vec![0u64; stride];
            for step in 0..length {
                let i = offset + step;
                partial += (x[i] * y[q - i]) as usize;
                let bits = [
                    (x[i] & 1) ^ (x[i] >> 1),
                    x[i] >> 1,
                    (y[q - i] & 1) ^ (y[q - i] >> 1),
                    y[q - i] >> 1,
                ];
                for (index, node) in scratch.nodes.iter_mut().enumerate() {
                    monomial_div_into(node, &state, MonomialDegree(rotation(index, stride)));
                }
                let mut count = 16;
                let mut summed = vec![0u64; stride];
                for (level, (selectors, bit_index)) in [
                    (&lhs[i], 0),
                    (&lhs[i], 1),
                    (&rhs[q - i], 0),
                    (&rhs[q - i], 1),
                ]
                .into_iter()
                .enumerate()
                {
                    let half = count / 2;
                    let chosen_node = bits[level + 1..]
                        .iter()
                        .fold(0usize, |a, &b| 2 * a + b as usize);
                    let bit = bits[level];
                    let selector = selectors.as_view().into_ggsw_iter().nth(bit_index).unwrap();
                    let (zeros, ones) = scratch.nodes[..count].split_at_mut(half);
                    for (node, (zero, one)) in zeros.iter_mut().zip(ones).enumerate() {
                        let before = if node == chosen_node {
                            let chosen = if bit == 0 { &*zero } else { &*one };
                            let phases: Vec<_> = (0..stride)
                                .map(|t| {
                                    glwe_phase_at(chosen, stride * (total - partial) + t, secrets)
                                })
                                .collect();
                            let mut error = zero.clone();
                            for ((r, &a), &b) in error
                                .as_mut()
                                .iter_mut()
                                .zip(zero.as_ref())
                                .zip(one.as_ref())
                            {
                                let diff = b.wrapping_sub(a);
                                *r = decomposer.closest_representable(diff).wrapping_sub(diff);
                            }
                            let rounding: Vec<_> = (0..stride)
                                .map(|t| {
                                    if bit == 0 {
                                        0
                                    } else {
                                        glwe_phase_at(
                                            &error,
                                            stride * (total - partial) + t,
                                            secrets,
                                        )
                                    }
                                })
                                .collect();
                            Some((phases, rounding))
                        } else {
                            None
                        };
                        cmux_assign_mem_optimized::<u64, _, _, _>(
                            zero,
                            one,
                            &selector,
                            scratch.fft.as_view(),
                            &mut PodStack::new(&mut scratch.memory),
                        );
                        if let Some((phases, rounding)) = before {
                            for t in 0..stride {
                                let phase =
                                    glwe_phase_at(zero, stride * (total - partial) + t, secrets);
                                let increment = phase.wrapping_sub(phases[t]);
                                let key = increment.wrapping_sub(rounding[t]);
                                summed[t] = summed[t].wrapping_add(increment);
                                writer.write_record([
                                    label.to_owned(),
                                    q.to_string(),
                                    offset.to_string(),
                                    length.to_string(),
                                    step.to_string(),
                                    level.to_string(),
                                    t.to_string(),
                                    bit.to_string(),
                                    "gate".into(),
                                    "".into(),
                                    (increment as i64).to_string(),
                                    (rounding[t] as i64).to_string(),
                                    (key as i64).to_string(),
                                ])?;
                            }
                        }
                    }
                    count = half;
                }
                std::mem::swap(&mut state, &mut scratch.nodes[0]);
                for t in 0..stride {
                    let expected = ((total >> (2 * t)) & 3) as u64 * encoding_delta(4);
                    let residual = glwe_phase_at(&state, stride * (total - partial) + t, secrets)
                        .wrapping_sub(expected);
                    let increment = residual.wrapping_sub(previous[t]);
                    assert_eq!(
                        increment, summed[t],
                        "selected-path local errors must telescope"
                    );
                    previous[t] = residual;
                    writer.write_record([
                        label.to_owned(),
                        q.to_string(),
                        offset.to_string(),
                        length.to_string(),
                        step.to_string(),
                        "".into(),
                        t.to_string(),
                        "".into(),
                        "state".into(),
                        (residual as i64).to_string(),
                        (increment as i64).to_string(),
                        "".into(),
                        "".into(),
                    ])?;
                }
            }
        }
    }
    writer.flush()?;
    Ok(())
}

fn audit_selector_rows(
    lhs: &[Selectors],
    rhs: &[Selectors],
    x: &[u64],
    y: &[u64],
    secrets: &HarnessSecretKeys,
    params: BenchParams,
    label: &str,
    path: &str,
) -> Result<()> {
    use tfhe::core_crypto::fft_impl::fft64::math::polynomial::FourierPolynomialView;
    let header = !std::path::Path::new(path).exists();
    let mut writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(OpenOptions::new().create(true).append(true).open(path)?);
    if header {
        writer.write_record([
            "trial",
            "operand",
            "index",
            "bit_index",
            "level",
            "row",
            "coefficient",
            "error",
        ])?;
    }
    let split_path = format!("{path}.split.csv");
    let split_header = !std::path::Path::new(&split_path).exists();
    let mut split = csv::WriterBuilder::new().has_headers(false).from_writer(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(split_path)?,
    );
    if split_header {
        split.write_record([
            "trial",
            "operand",
            "index",
            "bit_index",
            "level",
            "row",
            "coefficient",
            "body_error",
            "mask_error",
            "propagated_body",
            "ss_rounding",
            "ss_key_fft",
        ])?;
    }
    let fft = Fft::new(params.polynomial_size);
    let mut memory = PodBuffer::try_new(fft.as_view().backward_scratch()).unwrap();
    let n = params.polynomial_size.0;
    let mut standard = zero_glwe(params);
    for (name, selectors, digits) in [("X", lhs, x), ("Y", rhs, y)] {
        for index in [0, x.len() / 2, x.len() - 1] {
            for (bit_index, selector) in selectors[index].as_view().into_ggsw_iter().enumerate() {
                let value = digits[index];
                let bit = if bit_index == 0 {
                    (value & 1) ^ (value >> 1)
                } else {
                    value >> 1
                };
                let base = selector.decomposition_base_log().0;
                for matrix in selector.into_levels() {
                    let level = matrix.decomposition_level().0;
                    let factor = 1u64 << (64 - base * level);
                    let mut rows = Vec::new();
                    for (row, entry) in matrix.into_rows().enumerate() {
                        for (out, input) in standard
                            .as_mut_polynomial_list()
                            .iter_mut()
                            .zip(entry.data().chunks_exact(n / 2))
                        {
                            fft.as_view().backward_as_torus(
                                out,
                                FourierPolynomialView { data: input },
                                &mut PodStack::new(&mut memory),
                            );
                        }
                        for coefficient in (0..n).step_by(n / 32) {
                            let error = ggsw_row_error_at(
                                &standard,
                                coefficient,
                                row,
                                bit,
                                factor,
                                secrets,
                            );
                            writer.write_record([
                                label.to_owned(),
                                name.into(),
                                index.to_string(),
                                bit_index.to_string(),
                                level.to_string(),
                                row.to_string(),
                                coefficient.to_string(),
                                error.to_string(),
                            ])?;
                        }
                        rows.push(standard.clone());
                    }
                    if index == 0 && (level == 1 || level == selector.decomposition_level_count().0)
                    {
                        let body = rows.last().unwrap();
                        let body_error: Vec<_> = (0..n)
                            .map(|j| {
                                ggsw_row_error_at(
                                    body,
                                    j,
                                    params.glwe_dimension.0,
                                    bit,
                                    factor,
                                    secrets,
                                ) as u64
                            })
                            .collect();
                        let (_, _, ss_base, ss_level, _, _, _) =
                            params.direct_lift_decomposition(ProductVariant::Hybrid8x8);
                        let decomposer = SignedDecomposer::<u64>::new(ss_base, ss_level);
                        let mut residual = body.clone();
                        for (r, &value) in residual.as_mut().iter_mut().zip(body.as_ref()) {
                            *r = decomposer.closest_representable(value).wrapping_sub(value);
                        }
                        let phase_rounding: Vec<_> = (0..n)
                            .map(|j| glwe_phase_at(&residual, j, secrets))
                            .collect();
                        for (row, mask) in rows[..params.glwe_dimension.0].iter().enumerate() {
                            let propagated = negative_secret_product(&body_error, row, secrets);
                            let rounding = negative_secret_product(&phase_rounding, row, secrets);
                            for j in 0..n {
                                let error = ggsw_row_error_at(mask, j, row, bit, factor, secrets);
                                let key = (error as u64)
                                    .wrapping_sub(propagated[j])
                                    .wrapping_sub(rounding[j])
                                    as i64;
                                split.write_record([
                                    label.to_owned(),
                                    name.into(),
                                    index.to_string(),
                                    bit_index.to_string(),
                                    level.to_string(),
                                    row.to_string(),
                                    j.to_string(),
                                    (body_error[j] as i64).to_string(),
                                    error.to_string(),
                                    (propagated[j] as i64).to_string(),
                                    (rounding[j] as i64).to_string(),
                                    key.to_string(),
                                ])?;
                            }
                        }
                    }
                }
            }
        }
    }
    writer.flush()?;
    split.flush()?;
    Ok(())
}
