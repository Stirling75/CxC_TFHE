use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct ProductBit {
    pub left: usize,
    pub right: usize,
    pub bit: usize,
    pub column: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct Compressor {
    pub inputs: Vec<usize>,
    pub parity: usize,
    pub carry: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct Layer {
    pub target: usize,
    pub jobs: Vec<Compressor>,
    pub after: Vec<Vec<usize>>,
}

#[derive(Debug, Serialize)]
pub struct Plan {
    pub width: usize,
    pub products: Vec<ProductBit>,
    pub initial: Vec<Vec<usize>>,
    pub layers: Vec<Layer>,
    pub final_columns: Vec<Vec<usize>>,
    pub value_count: usize,
}

impl Plan {
    pub fn new(width: usize) -> Self {
        assert!([16, 32, 64, 128, 256].contains(&width));
        let mut products = Vec::new();
        let mut columns = vec![Vec::new(); width];
        for left in 0..width / 8 {
            for right in 0..width / 8 - left {
                let offset = 8 * (left + right);
                for bit in 0..16.min(width - offset) {
                    columns[offset + bit].push(products.len());
                    products.push(ProductBit { left, right, bit, column: offset + bit });
                }
            }
        }
        let initial = columns.clone();
        let mut next_id = products.len();
        let mut targets = Vec::new();
        let mut target = 3;
        while target < columns.iter().map(Vec::len).max().unwrap() {
            targets.push(target);
            target = 3 * target / 2;
        }
        let mut layers = Vec::new();
        for target in targets.into_iter().rev() {
            let mut after = vec![Vec::new(); width];
            let mut jobs = Vec::new();
            for (c, old) in columns.iter().enumerate() {
                // Eq. (12): count new incoming carries, but consume old terms only.
                let excess = (old.len() + after[c].len()).saturating_sub(target);
                let mut cursor = 0;
                for arity in std::iter::repeat_n(3, excess / 2)
                    .chain(std::iter::repeat_n(2, excess % 2))
                {
                    let inputs = old[cursor..cursor + arity].to_vec();
                    cursor += arity;
                    let parity = next_id;
                    next_id += 1;
                    let carry = (c + 1 < width).then(|| {
                        let id = next_id;
                        next_id += 1;
                        after[c + 1].push(id);
                        id
                    });
                    after[c].push(parity);
                    jobs.push(Compressor { inputs, parity, carry });
                }
                after[c].extend_from_slice(&old[cursor..]);
                assert!(after[c].len() <= target);
            }
            columns = after.clone();
            layers.push(Layer { target, jobs, after });
        }
        assert!(columns.iter().all(|c| c.len() <= 3));
        assert!(columns[..8].iter().all(|c| c.len() == 1));
        Self { width, products, initial, layers, final_columns: columns, value_count: next_id }
    }

    pub fn counts(&self) -> serde_json::Value {
        self.counts_for_ring(1024)
    }

    pub fn counts_for_ring(&self, polynomial_size: usize) -> serde_json::Value {
        self.counts_for_packing(polynomial_size, 2, false)
    }

    pub fn counts_for_packing(&self, polynomial_size: usize, terminal_lut_count_log: usize,
                              cc2_big_key: bool) -> serde_json::Value {
        assert!([1024, 2048].contains(&polynomial_size));
        assert!([1, 2].contains(&terminal_lut_count_log));
        let br_per_terminal_cbs = 4 / (1 << terminal_lut_count_log);
        let cmux_per_bit = 65536 / polynomial_size - 1 + polynomial_size.ilog2() as usize;
        // One low-bit read per terminal column, plus a high-bit read when the
        // column holds more than one term. The top column's high bit would
        // only feed column W, outside the lower-W output, and is not read.
        let terminal_reads: usize = (8..self.width)
            .map(|c| 1 + usize::from(self.final_columns[c].len() > 1 && c + 1 < self.width)).sum();
        let terminal_cmux: usize = (8..self.width).map(|c| {
            if self.final_columns[c - 1].len() > 1 { 4 } else { 2 }
        }).sum();
        let compressors: usize = self.layers.iter().map(|l| l.jobs.len()).sum();
        serde_json::json!({
            "input_grouped_cbs": self.width, "terminal_binary_cbs": terminal_reads,
            "ggsw_conversions": 2 * self.width + terminal_reads,
            "dadda_compressors": compressors, "dadda_layers": self.layers.len(),
            "emission_pbs": self.width / 2, "product_bits": self.products.len(),
            "product_cmux": self.products.len() * cmux_per_bit, "terminal_cmux": terminal_cmux,
            "product_cmux_per_bit": cmux_per_bit,
            "total_blind_rotations": self.width + br_per_terminal_cbs * terminal_reads + compressors + self.width / 2,
            // One KS per compressor and per terminal column sum. With small-key
            // CC2: two per emission block (pre-PBS and closing), none at the
            // lift. With big-key CC2: one per emission block and one per lift.
            "key_switches": compressors + (self.width - 8)
                + if cc2_big_key { self.width / 2 + self.width } else { self.width },
            "br_per_terminal_cbs": br_per_terminal_cbs,
            "segments": 1,
            "terminal_note": "Independent four-CMux realization of the paper's two-level two-state step; not recovered author gate counts"
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn counts_match_independent_python_reference() {
        for (width, products, compressors, reads, layers) in [
            // Terminal reads exclude the unused top-column high read (one per width).
            (16, 32, 0, 15, 0), (64, 512, 332, 111, 5), (256, 8192, 7414, 495, 8),
        ] {
            let plan = Plan::new(width);
            assert_eq!(plan.products.len(), products);
            let counts = plan.counts();
            assert_eq!(counts["dadda_compressors"], compressors);
            assert_eq!(counts["terminal_binary_cbs"], reads);
            assert_eq!(plan.layers.len(), layers);
            assert_eq!(counts["key_switches"], compressors + 2 * width - 8);
            assert_eq!(counts["ggsw_conversions"], 2 * width + reads);
            assert!(plan.final_columns[width - 1].len() > 1);
            let mut available: std::collections::HashSet<_> = (0..products).collect();
            for layer in &plan.layers {
                let mut output = Vec::new();
                for job in &layer.jobs {
                    assert!(job.inputs.iter().all(|id| available.contains(id)));
                    output.push(job.parity);
                    output.extend(job.carry);
                }
                available.extend(output);
                assert!(layer.after.iter().all(|c| c.len() <= layer.target));
            }
        }
    }

    #[test]
    fn ring_packing_changes_only_product_cmux_count() {
        let plan = Plan::new(64);
        let small = plan.counts_for_ring(1024);
        let large = plan.counts_for_ring(2048);
        assert_eq!(large["product_cmux"], 512 * 42);
        for field in ["total_blind_rotations", "ggsw_conversions", "dadda_compressors", "terminal_cmux"] {
            assert_eq!(small[field], large[field]);
        }
        let narrow = plan.counts_for_packing(2048, 1, false);
        assert_eq!(narrow["total_blind_rotations"].as_u64().unwrap(),
            large["total_blind_rotations"].as_u64().unwrap() + large["terminal_binary_cbs"].as_u64().unwrap());
        assert_eq!(narrow["ggsw_conversions"], large["ggsw_conversions"]);
    }
}
