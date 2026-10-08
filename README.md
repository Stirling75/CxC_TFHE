# Extract Sums, Not Products: artifact

Code, measurements, and scripts accompanying the paper.

```
code/            benchmark runner, method catalogue, Rust sources
results/         measured timings, failure estimates, noise measurements, security estimates
paper-scripts/   regenerate the figure data and tables from results/
```

## Build and test

Linux x86-64, Python 3.9+, Rust 1.91.1+, a C toolchain. This copy omits the
vendored crates; fetch the locked crates once. The security estimates in
`results/security/` additionally need SageMath and the lattice estimator.

```sh
cd code
for m in code/ring-variant-multiplier code/server-campaign/adapters/bfv-style \
         code/server-campaign/adapters/parmesan code/server-campaign/adapters/bitwise; do
  cargo fetch --locked --manifest-path $m/Cargo.toml
done
sh run.sh build --methods hybrid-grouped-rev-ld tfhe-rs-ks28 --jobs 8
sh run.sh test
```

## Run

```sh
sh run.sh bench --research --methods hybrid-grouped-rev-ld tfhe-rs-ks28 \
    --widths 128 --threads 1 64 --repetitions 5 --warmup 1 --output ../results/my-run
```

Before timing, the runner computes the failure estimate of each case and stops
if a screened case misses 2^-128 (`--allow-failing-screen` overrides). After
each run, it checks the parameters, the method-specific operation counts, and
every decrypted product. `--dry-run` writes schedules and estimates without
timing. A thread budget T sets both the worker count and the CPU affinity.

| Paper | `--methods` |
| --- | --- |
| Ours, product-sum / chunk-product lookup | `hybrid-grouped-rev-ld` / `hybrid-cached-rev-ld` |
| Ours+MVB | `hybrid-grouped-rev-mvb`, `hybrid-cached-rev-mvb` |
| TFHE-rs, KS (2,8) / default | `tfhe-rs-ks28` / `tfhe-rs` |
| Shokri-Tsoutsos, n=688 / paper | `st-r2-n688` / `st-reported-r2` |
| Bernard et al., KS (3,6) / paper | `bernard-mvb-ks36` / `bernard-mvb` |
| BFV adaptation | `clot-bfv` |
| Trifan et al. / PARMESAN | `trifan` / `parmesan` |
| Table 8 controls | `hybrid-grouped-rev-no-cache-ld`, `hybrid-8x8-rev-ld`, `hybrid-8x8-rev-ld-sks`, `hybrid-4x4-rev-ld` |

## Results

| `results/` | Content | Paper |
| --- | --- | --- |
| `main/` | ours, TFHE-rs, Shokri-Tsoutsos paper set, BFV | Tables 1, 7, Fig. 5 |
| `mvb/`, `bernard/`, `st688/`, `trifan/`, `parmesan/` | the other methods | Tables 1, 7, Fig. 5 |
| `controls/`, `planner/` | phase breakdown and controls | Table 8 |
| `estimates/` | failure estimates of our configurations | Tables 5, 9 |
| `limit-sweep/` | limit L on linear digits | Fig. 4 |
| `noise/` | errors of decrypted intermediate ciphertexts | Table 6 |
| `security/` | lattice-estimator inputs and outputs | Table 3 |

Each case `<method>-w<W>-t<T>/` holds the raw timings and schedule of the
method and `run.json` (command, environment, estimate), and `summary.csv`
gives the mean and standard deviation per case without the warm-up. Repeated
files are stored once and referenced by `{"same_as": <relative path>}`. Paths
recorded inside the run records and `source-manifests/` refer to the original
campaign directories.

## Reproduce the paper

From the repository root:

```sh
cd paper-scripts && make check
```

This regenerates the figure data and Tables 6, 7, 9, S1, and S2, and our
estimates of Table 5, and compares them byte by byte with `expected/`.
`make sweep` recomputes the sweep of Figure 4 (several minutes), and
`scaled_estimate.py` the estimate with scaled variances of Section 6.2.

## Licenses

PARMESAN (`code/code/server-campaign/sources/parmesan`) is AGPL-3.0. The CBS
components derived from RevHomTrace and Refined TFHE keep their MIT notices in
`code/licenses/`. The PARMESAN sources keep their upstream comments, and
vendored crates keep their own licenses.
