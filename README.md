# Supplementary material: encrypted integer multiplication benchmarks

This archive holds the benchmark package for every method measured in the
paper, the measured results, and the scripts that regenerate the figure data,
the latency and failure tables, and the noise measurements of the paper.

```
code/            benchmark package (runner, method catalogue, Rust sources, vendored crates)
results/         measured timings, failure estimates, noise measurements, security estimates
paper-scripts/   generators of the paper's figure data and tables, reading ../results
```

## 1. Build

Requirements: Linux x86-64, Python 3.9+, Rust 1.91.1+ and a C toolchain.
All locked crate sources are in `code/vendor/`; `code/.cargo/config.toml`
selects them and enables offline mode, so no network access is needed.
Keep the hidden `.cargo/` directory. Run every command from `code/`:

```sh
cd code
sh run.sh doctor                                  # records CPU topology and memory
sh run.sh build --methods hybrid-grouped-rev-ld tfhe-rs-ks28 --jobs 8
sh run.sh test                                    # Python unit tests and cargo test
sh run.sh list                                    # every method in config/methods.json
```

`build` compiles only the crates the selected methods need into `code/build/`
(`--build-root` changes this). Building all methods takes several minutes.

## 2. Running a benchmark

```sh
sh run.sh smoke --research --methods hybrid-grouped-rev-ld --output results/smoke
sh run.sh bench --research --methods hybrid-grouped-rev-ld hybrid-cached-rev-ld tfhe-rs-ks28 \
    --widths 16 32 64 128 256 --threads 1 2 4 8 16 32 64 \
    --repetitions 5 --warmup 1 --output results/my-run
```

`--research` acknowledges that whole-multiplication failure bounds are model
estimates. For our configurations and the screened baselines (Shokri-Tsoutsos
retuned, BFV adaptation), the runner derives the public schedule, computes the
estimate before any measurement, and stops if it misses 2^-128 unless
`--allow-failing-screen` is given. For the other baselines the estimate is
recorded and labelled but does not gate the run. `--dry-run` resolves
schedules and failure estimates without running anything. Output directories
must be new.

Each case directory `<method>-w<W>-t<T>/` contains `raw/timings.csv` (one row
per trial, warm-up flagged), `raw/parameters.json` (parameters reported by the
binary), `plan.json` (the public restoration schedule, for our
configurations) and `run.json` (command, environment, CPU set, resolved
analysis, binary hash, peak RSS). After each run, the runner checks the
reported parameters against the catalogue, the operation counts of every
trial against the schedule (`scripts/results.py`,
`code/ring-variant-multiplier/run.py`), and that every product decrypts
correctly. `summary.csv` holds the mean and sample standard deviation over the
measured (non-warm-up) trials.

## 3. Method names

| Paper name | `--methods` key |
| --- | --- |
| Ours, product-sum lookup | `hybrid-grouped-rev-ld` |
| Ours, chunk-product lookup | `hybrid-cached-rev-ld` |
| Ours+MVB (product-sum / chunk-product) | `hybrid-grouped-rev-mvb` / `hybrid-cached-rev-mvb` |
| TFHE-rs radix, finer key switch (2,8) / default | `tfhe-rs-ks28` / `tfhe-rs` |
| Shokri-Tsoutsos, retuned (n=688) / paper parameters | `st-r2-n688` / `st-reported-r2` |
| Bernard et al., key switch (3,6) / paper set | `bernard-mvb-ks36` / `bernard-mvb` |
| BFV adaptation (CLOT tensor product) | `clot-bfv` |
| Trifan et al. | `trifan` |
| PARMESAN (W = 16, 32 only) | `parmesan` |

Controls of Table 8 (W=128, one thread): `hybrid-grouped-rev-no-cache-ld`
(product-sum lookup without row cache), `hybrid-8x8-rev-ld` (chunk-product
lookup without row cache), `hybrid-8x8-rev-ld-sks` (without row cache and
shared key switch), `hybrid-4x4-rev-ld` (per product, 4-bit chunks).
`st-r2-n704` is the earlier retuned Shokri-Tsoutsos set, compared with n=688
in the supplement.

## 4. Failure estimates

`scripts/cases.py:resolve(method, width)` attaches a whole-multiplication
log2 failure estimate to every case (`whole_product_log2_estimate` and
`failure_basis` in `run.json`):

- ours: the public schedule from `model/make_plan.py`; then
  `heterogeneous_screen.py` propagates the variances of Section 5.3 through
  selector lifting, lookup, reduction and final addition, and takes a union
  bound over all decryption events, for distinct (AB) and identical (A=B)
  operands;
- Shokri-Tsoutsos: `server-campaign/st_retune.py` (for the paper parameters,
  the analysis is recorded in `run.json` and the case is labelled
  unverifiable);
- BFV adaptation: `adapters/bfv-style/verify_candidate.py`; at W=256 the
  Gaussian estimate is replaced by a correlated-quotient estimate calibrated
  to measurements;
- TFHE-rs radix and Trifan et al.: per-PBS library label + log2(number of PBS);
- Bernard et al.: `model/bernard/failure.py` over the public folding schedule
  (`model/bernard/schedule.py`), worst case over the thread budgets;
- PARMESAN: Gaussian model of its preset, union bound over its PBS count.

`paper-scripts/estimates.py` recomputes the estimates of our configurations
(Tables 5 and 9, Table S3) into `out/data/failure-ours.csv`; the stored copy
with all per-family terms is `results/estimates-20261007/estimates.json`. It
first checks that the derived schedules equal the `plan.json` of the measured
runs. The estimates recorded in `run.json` of the timing campaigns predate a
refinement of the model (first candidates shared by full groups, margin of the
packed selector table) and differ from these by at most 0.1.

These are model estimates, not proofs; see Section 5 of the paper.

## 5. Measurement protocol

A thread budget T means `RAYON_NUM_THREADS=T` *and* Linux CPU affinity of the
benchmark process to the first T CPU IDs (`--cpu-ids` sets the order). Key
generation and input encryption are not timed. Each case runs one warm-up and
five measured multiplications on random operands (three at W=256 and for
Trifan et al., two for Trifan et al. at W=256, five for `st-r2-n688` at W=256
from 16 threads); every result is decrypted and checked. The campaigns of
October 4 to 7 each started only after the one-minute load average stayed
below 4 for five minutes. The measurements used a 32-core AMD Ryzen
Threadripper PRO 5975WX (64 logical CPUs); the recorded topology is in each
`campaign.json`.

## 6. Results and regeneration

| Directory in `results/` | Content | Paper |
| --- | --- | --- |
| `stats-20261004/stats-main-*` | ours, TFHE-rs, ST paper set, BFV; all W and T | Fig. 5, Tables 1, 7, S1-S2 |
| `stats-20261004/stats-controls-*` | controls, W=128, T=1 | Table 8 |
| `stats-20261004/stats-{trifan,parmesan}-*`, `w256-20261002` | Trifan et al., PARMESAN | Fig. 5, Tables S1-S2 |
| `mvb-20261005` | Ours+MVB | Fig. 5, Tables 1, 9 |
| `bernard-20261005` | Bernard et al., both key switches | Fig. 5, Tables S1-S2 |
| `st688-20261007` | retuned Shokri-Tsoutsos n=688; paired n=688 vs n=704 | Fig. 5, Tables S1-S2 |
| `estimates-20261007` | failure estimates of our configurations | Tables 5, 9, S3 |
| `limit-sweep-20261006` | sweep of the linear-digit limit L | Fig. 4, Table S4 |
| `noise-20261007` | errors of decrypted intermediate ciphertexts | Table 6 |
| `security-20261006` | lattice-estimator inputs and outputs | Section 6.2, Table S5 |
| `planner-20261006` | restoration PBS count of the 4-bit control | Table 8 |

`noise-20261007/` holds, as gzipped CSV, the errors of decrypted intermediate
ciphertexts: GGSW selector rows for the three parameter sets of Table 4 (W=16,
two multiplications), the reduction and final-addition inputs of product-sum
lookup (W=64, 128, 256, with the modeled variances in `model.json`), the
digits of Ours+MVB, and the blind-rotation inputs of Bernard et al. (paper
set).

Copies of the noise analysis, the schedule, and the source manifest that
repeat across thread budgets or campaigns are replaced by
`{"same_as": <relative path>}` pointers (`paper-scripts/dedupe_results.py`);
absolute build paths in `run.json` are replaced by `<pkg>`. Cases of methods
not used in the paper were removed from the campaigns, so a `campaign.json`
may list more methods than its directory holds. `paper-scripts/summaries.txt`
lists the summaries that feed the figures and tables, in precedence order. To
regenerate everything and compare with the data of the paper:

```sh
cd paper-scripts && make check     # or: sh regenerate.sh  (about one minute)
```

This writes `out/data/*.csv` (latency vs width and threads, the ratios plotted
in Figure 5, phase breakdown, restoration PBS counts, Figure 4, failure
estimates, the noise ratios of Table 6 and the excess kurtosis of Section 6.2), `out/threads-table.tex` (Table 7),
`out/appendix-latency.tex` (Tables S1-S2) and `out/mvb-table.tex` (Table 9),
and compares each with `expected/`; it exits with an error on any difference.
`make sweep` recomputes the limit sweep itself (several minutes).
`scaled_estimate.py` recomputes the estimate of Section 6.2 with the modeled
variances scaled by 0.61.

The Windows import libraries (`lib/`) of the vendored `windows_*` crates are
removed, since the code builds only on Linux; their checksums are updated.

## Licenses

PARMESAN (`code/code/server-campaign/sources/parmesan`) is AGPL-3.0; CBS
components derived from RevHomTrace and Refined TFHE keep their MIT notices in
`code/licenses/`; vendored crates keep their own license files.
