# BFV-Style Parallel Execution

September 18, 2026. Local optimization of the separately implemented BFV-style
baseline. The purpose is a stronger comparison, not preservation of an earlier
speed ranking. Neither the running server campaign nor the manuscript changes.

## What Changed

The n800/N2048 parameters, input/output scale, full convolution, bit extraction,
public normalizer partitions, and PBS count remain unchanged. The default
one-thread path retains the serial reference computation.

With `--threads` greater than one:

1. Re-encode both operands' input blocks concurrently, preserving their order.
2. Use TFHE-rs's parallel packing routine. Its reductions are exact wrapping
   integer additions; this does not change a floating-point accumulation order.
3. Decompose different convolution coefficients concurrently. The dependent
   bit-extraction loop inside each coefficient remains sequential.
4. Compile the original normalizer groups into a dependency graph. Within a
   stage, independent groups run concurrently. Every group's members and
   within-group summation order are retained.
5. Reuse each normalizer group's identical key-switch result for its low and
   carry PBS. The two PBS calls themselves can run concurrently. The old path
   already repeated the same deterministic KS with the same input and key;
   sharing it introduces no new ciphertext dependence.

All parallel work uses one explicitly sized Rayon pool. Nested jobs share
that pool rather than creating additional worker pools. The serial reference
check, when requested, uses a separate one-thread pool after measured work
has finished; the two evaluations do not run concurrently.

At W256, both paths perform 2521 PBS calls. The new execution reduces ordinary
KS calls from 2521 to 1972 by removing 549 repetitions. The normalizer graph
has 555 groups in 250 dependency stages. This remaining dependency depth means
that normalizer speedup cannot be assumed proportional to the thread count.

## Correctness And Failure Model

The graph reproduces the reference plaintext results in 1200 cases covering
six widths and both supported normalizer domains. A separate fixed-graph test
checks group membership, input order, output indices and dependency levels.

`--verify-serial` evaluates the same encrypted inputs under the same keys using
the one-thread reference, outside measured timings. It compares every
re-encoded input block, both packed GLWEs, every extracted convolution
coefficient and every final output LWE coefficient for exact equality, not
merely equality after decryption. The PBS counts must agree too.

Recorded equivalence checks: `results/parallel-equivalence-small-20260918`
contains six W8 input patterns, and `results/parallel-equivalence-w256-20260918`
contains one random W256 product under a second fresh key set. All seven pass
the byte-for-byte comparison and plaintext checks. At W256 the parallel
stage sum is 25.782 s; its same-key, same-input serial reference is 132.596 s.
This is a diagnostic pair, not a repeated speedup benchmark.

The Gaussian event calculation is unchanged because the per-event ciphertext
computation and public partitions are preserved. Reusing deterministic KS does
not remove either PBS decision from the union bound. The selected layout-aware
screen, including its existing FFT model, remains -145.516051 at W256.
All Gaussian, coefficient-independence, shared-key and numerical-model
qualifications from [the parameter search](PARAMETER_SEARCH_20260918.md) remain.
Faster execution does not strengthen that result into an unconditional proof.

## Measurement Protocol

Use one warm-up and three measured random products for each method at the same
width and worker count. These are preliminary Mac measurements, not the final
server study. Key generation, initial encryption/refresh and correctness checks
are excluded. BFV reports a sum of instrumented stages; the grouped method
reports its evaluator wall clock. Small BFV setup gaps remain uninstrumented.
No CPU affinity or frequency control is imposed.

Warm-up rows are retained and explicitly labeled, but excluded from the
reported arithmetic mean. Do not report a single warm-up run as a measured
sample, or compare eight-thread BFV against one-thread grouped multiplication.

## Measured Results

The eight-thread BFV grid is `results/parallel-t8-grid-20260918`; its independent
validation and summary are in `results/parallel-t8-grid-audit-20260918.json`.
Each width has one retained warm-up and three measured random products.

| W | BFV stage-sum mean (s) | Sample standard deviation (s) |
|---|---:|---:|
| 8 | 0.607 | 0.021 |
| 16 | 1.232 | 0.020 |
| 32 | 2.914 | 0.096 |
| 64 | 6.237 | 0.089 |
| 128 | 12.797 | 0.067 |
| 256 | 25.915 | 0.325 |

All 24 grid products pass. Combined with the seven equivalence cases, this
update validates 31 input cases under three fresh key sets, plus seven serial
reference evaluations of those same ciphertexts. All eight Rust tests and
21 Python tests pass, including warm-up exclusion and the earlier numerical
regressions. The noise report derived from the executed grid is
`results/parallel-t8-noise-20260918.json`; its model values are unchanged.

The grouped control in `results/grouped-t8-measured-20260918` uses the audited
binary and plan, eight workers, one warm-up, and three measured random products.
Its actual parameters/counts match the earlier audit and all four outputs pass.

| W256, eight threads | Measured seconds | Mean (s) | Sample SD (s) |
|---|---|---:|---:|
| BFV-style, parallel | 25.750, 26.289, 25.705 | 25.915 | 0.325 |
| Grouped hybrid | 9.492, 9.721, 9.688 | 9.634 | 0.124 |

The ratio of these local means is about 2.690 in favor of the grouped method.
This supports a comparison of these implementations and settings, not a claim
that every BFV-style construction is slower. Three repetitions on one Mac,
with the timing-contract caveats above, are not the final server evaluation.

The BFV W256 mean comprises 12.335 s input re-encoding, 1.666 s packing,
0.00449 s tensor/relinearization, 0.000217 s extraction, 4.409 s coefficient
decomposition and 7.499 s normalization. Input re-encoding remains the largest
stage, and the 250-stage carry dependency limits normalization parallelism.
No evaluation-key payload or parameter reduction was made in this update.

## Commands

Run from this directory with an existing offline Cargo cache. Each output path
must be new. Execute the performance commands sequentially, not concurrently.

```bash
cargo test --release --locked --offline
python3 -B -m unittest test_noise.py test_execution.py
python3 -B verify_candidate.py --output results/parallel-preflight-repeat.json

# Same-key, same-input byte-for-byte check, outside any benchmark comparison.
sh run.sh --parameters parameters/bfv-n800-range-aware.json --threads 8 \
  --verify-serial --widths 8 --patterns zero one max alternating impulse random \
  --output results/parallel-equivalence-small-repeat
sh run.sh --parameters parameters/bfv-n800-range-aware.json --threads 8 \
  --verify-serial --widths 256 --patterns random --seed 20260922 \
  --output results/parallel-equivalence-w256-repeat

# Eight-thread grid, including the W256 matched comparison.
sh run.sh --parameters parameters/bfv-n800-range-aware.json --threads 8 \
  --widths 8 16 32 64 128 256 --patterns random --warmup 1 --repetitions 3 --seed 20260923 \
  --output results/parallel-t8-repeat
python3 -B audit_run.py --run results/parallel-t8-repeat \
  --output results/parallel-t8-repeat-audit.json
python3 -B compare_grouped.py --threads 8 --warmup 1 --repetitions 3 \
  --seed 20260923 --output results/grouped-t8-repeat

```

`audit_run.py` checks completion, all requested trials, warm-up labels, runtime
parameters, plaintext-product records, every recorded correctness check, PBS
and KS counts, optional serial equivalence, finite stage times, and recomputed
conditional noise estimates. Its tests deliberately corrupt KS counts,
parameters, warm-up labels and serial-equivalence records to verify rejection.

## Remaining Optimization Questions

- The encoding BSK remains large and input re-encoding is expensive. A smaller
  key or different arithmetic backend requires a new numerical-error screen,
  not just a faster timing result.
- The current normalizer keeps the old partitions to preserve the failure
  model. A new carry/reduction structure must be analyzed separately.
- Bounded-length convolution may reduce the required encoding precision but
  increases product/decomposition/normalization work. This route remains
  unmeasured; neither the current timings nor this optimization settle it.
- Thread scaling, memory bandwidth, and evaluation-key payload still need
  server-side measurements before making a publication-level comparison.
