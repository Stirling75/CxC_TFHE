# Extract Sums, Not Products

This artifact contains the code, measurements, and reproduction scripts for
the paper **"Extract Sums, Not Products: Faster TFHE Integer Multiplication
under a Whole-Multiplication Failure Target."** It includes our multiplication
methods, the baselines used in the evaluation, and the scripts for estimating
failure probabilities and regenerating the paper's tables and figure data.

To explore the reported results without running new benchmarks, you can start
with [Reproducing the paper](#reproducing-the-paper). To build the code and run
your own experiments, follow the instructions below.

## What's included

| Directory | Contents |
| --- | --- |
| [`code/`](code/) | Benchmark runner, method catalogue, and Rust implementations |
| [`results/`](results/) | Recorded timings, failure estimates, noise measurements, and security estimates |
| [`paper-scripts/`](paper-scripts/) | Scripts for regenerating tables and figure data from the recorded results |

## Building and testing

This artifact was tested on a Linux x86-64 server with an AMD Ryzen
Threadripper PRO 5975WX (32 cores, 64 hardware threads), which was used for
the performance evaluation in the paper.

To build and run the benchmarks, please have Python 3.9 or later, Rust 1.91.1
or later, and a C toolchain installed. Regenerating the tables and figure data
from the included results requires only Python and Make. Recomputing the
security estimates additionally requires SageMath and the lattice estimator;
see [`results/security/README.md`](results/security/README.md) for details.

**First, download the dependencies.** This lightweight package does not
include vendored Rust crates, so an internet connection is needed for this
step. The lockfiles pin the dependency versions used by each implementation.
Starting from the repository root, run:

```sh
cd code
for m in code/ring-variant-multiplier code/server-campaign/adapters/bfv-style \
         code/server-campaign/adapters/parmesan code/server-campaign/adapters/bitwise; do
  cargo fetch --locked --manifest-path $m/Cargo.toml
done
```

**Next, build and test.** From the same `code/` directory, the following
commands build our product-sum multiplier and the TFHE-rs baseline, then run
the package's test suites:

```sh
sh run.sh build --methods hybrid-grouped-rev-ld tfhe-rs-ks28 --jobs 8
sh run.sh test
```

The build and test commands use the dependencies downloaded in the first
step and do not need an internet connection.

## Running benchmarks

The example below compares our product-sum multiplier with TFHE-rs at
128 bits, using one and 64 threads. Each case includes one warm-up followed
by five measured runs. Run it from `code/`, and adjust the thread counts to
the CPUs available on your machine:

```sh
sh run.sh bench --research --methods hybrid-grouped-rev-ld tfhe-rs-ks28 \
    --widths 128 --threads 1 64 --repetitions 5 --warmup 1 --output ../results/my-run
```

Please choose a new output directory for each experiment. Before timing, the
runner evaluates the failure screen for each case and stops if a screened case
misses the `2^-128` target. After each run, it checks the parameters,
method-specific operation counts, and every decrypted product. On Linux, the
thread budget sets both the worker count and CPU affinity.

The `--research` flag acknowledges that the failure estimates depend on the
analysis assumptions described in the paper. You can add `--dry-run` to
inspect the schedules and estimates without running the benchmarks.
For comparisons with configurations that miss the failure screen,
`--allow-failing-screen` permits the run and labels these cases in the results.

### Choosing a method

Use the identifiers below with `--methods` in both the build and benchmark
commands. You can also run `sh run.sh list` to see the full catalogue.

| Method in the paper | `--methods` identifier |
| --- | --- |
| Ours, product-sum / chunk-product lookup | `hybrid-grouped-rev-ld` / `hybrid-cached-rev-ld` |
| Ours+MVB | `hybrid-grouped-rev-mvb`, `hybrid-cached-rev-mvb` |
| TFHE-rs, KS (2,8) / default | `tfhe-rs-ks28` / `tfhe-rs` |
| Shokri-Tsoutsos, n=688 / paper | `st-r2-n688` / `st-reported-r2` |
| Bernard et al., KS (3,6) / paper | `bernard-mvb-ks36` / `bernard-mvb` |
| BFV adaptation | `clot-bfv` |
| Trifan et al. / PARMESAN | `trifan` / `parmesan` |
| Table 8 controls | `hybrid-grouped-rev-no-cache-ld`, `hybrid-8x8-rev-ld`, `hybrid-8x8-rev-ld-sks`, `hybrid-4x4-rev-ld` |

## Included results

The recorded results are organized by experiment. The following table maps
each directory to the corresponding tables and figures in the paper.

| Directory under `results/` | Contents | Paper |
| --- | --- | --- |
| `main/` | Our methods, TFHE-rs, Shokri-Tsoutsos with the paper's parameters, and the BFV adaptation | Tables 1, 7, Fig. 5 |
| `mvb/`, `bernard/`, `st688/`, `trifan/`, `parmesan/` | MVB variants and additional comparisons | Tables 1, 7, Fig. 5 |
| `controls/`, `planner/` | Phase breakdown and controls | Table 8 |
| `estimates/` | Failure estimates of our configurations | Tables 5, 9 |
| `limit-sweep/` | Sweep of the limit L on linear digits | Fig. 4 |
| `noise/` | Errors of decrypted intermediate ciphertexts | Table 6 |
| `security/` | Lattice-estimator inputs and outputs | Table 3 |

Each case directory, named `<method>-w<W>-t<T>/`, contains its raw timings
and schedule, together with a `run.json` record of the command, environment,
and failure estimate. The `summary.csv` files report the mean and standard
deviation for each case, excluding the warm-up.

To keep the package small, repeated files are stored once and referenced by
`{"same_as": "<relative path>"}`. Paths within the run records and
`source-manifests/` preserve the original experiment directory names.

## Reproducing the paper

You can regenerate the paper's tables and figure data directly from the
included results, without compiling the Rust code or rerunning the encrypted
multiplications. From the repository root, run:

```sh
cd paper-scripts && make check
```

This command regenerates the figure data, Tables 6, 7, 9, S1, and S2, and our
estimates in Table 5. It then compares the generated files byte for byte with
the reference files in [`paper-scripts/expected/`](paper-scripts/expected/).

From `paper-scripts/`, you can also run `make sweep` to recompute the sweep in
Figure 4 (this takes several minutes), or `python3 scaled_estimate.py` to
recompute the estimate with scaled variances discussed in Section 6.2.

## Licenses

The PARMESAN sources in [`code/code/server-campaign/sources/parmesan`](code/code/server-campaign/sources/parmesan/)
are licensed under AGPL-3.0 and retain their upstream comments. The CBS
components derived from RevHomTrace and Refined TFHE retain their MIT notices
in [`code/licenses/`](code/licenses/). Downloaded Rust dependencies remain
subject to their respective licenses.
