# BFV-Style Probe: Local Validation

September 18, 2026. Correctness diagnostics on macOS/aarch64, one evaluation
thread. These are not warmed-up server benchmarks or certified parameters.

This is the ORIGINAL n1152 diagnostic record. The later n800 candidate,
conditional model results and encrypted tests are reported separately in
[the parameter-search record](PARAMETER_SEARCH_20260918.md).

## Functional Checks

The original diagnostic candidate uses n=1152, N=2048, k=1, ordinary PBS (11,3),
input-encoding PBS (3,20), KS (3,8), packing (4,15), and relinearization (4,15).
The pairs denote (base_log, levels). Convolution encoding uses no padding bit;
native radix input/output retains its original scale 2^59. Full parameters are
in each run's manifest.

- Six Rust tests pass: exact tensor against signed BigInt arithmetic;
  rounding; a regression for premature reduction modulo q; radix convolution
  and carry restoration on 600 plaintext pairs through W=256; the PBS schedule;
  and all 2048 plaintext values of the sign-extraction algebra.
- `results/full-deep-20260918`: W=8 and W=256, each with `max` and `random`,
  four encrypted end-to-end products, all correct.
- `results/width-grid-20260918`: W=8,16,32,64,128,256, one random product each
  under a fresh key set, six encrypted end-to-end products, all correct.
- In total: ten products under two key sets for the final candidate. Three
  were 256-bit products. This small sample cannot measure rare failures.

`max` means A=2^W-1 and B=2^W-2, encrypted separately. The tests include native
input checks, rescaled and packed inputs, extracted convolution coefficients,
and every final radix digit against a BigUint product modulo 2^W. Decryption
does not occur inside the multiplication evaluator.

## Unoptimized Stage Times

One diagnostic observation per width in `width-grid-20260918`:

| W | Sum of measured stages (s) | Tensor + relinearization (ms) | Correct |
|---|---:|---:|---|
| 8 | 2.561 | 12.12 | yes |
| 16 | 5.405 | 11.93 | yes |
| 32 | 12.005 | 12.04 | yes |
| 64 | 26.882 | 12.34 | yes |
| 128 | 58.744 | 13.34 | yes |
| 256 | 122.412 | 12.61 | yes |

At W=256 this last trial spends 50.26 s in input re-encoding, 5.12 s in packing,
46.08 s in coefficient decomposition and 20.94 s in radix restoration. The
earlier two W=256 trials took 124.58 and 125.20 s by the same stage-sum measure.
Key generation, initial encryption/refresh, checking, and small harness/setup
overheads are excluded. Do not present this table as a competitive latency
comparison: the decomposition, keys and execution are not optimized.

The evaluation-key payload is 2,894,692,352 bytes (2.696 GiB). This is neither
serialized size nor peak process memory. The additional encoding Fourier BSK
alone accounts for 1440 MiB.

## Negative Trials Retained

| Run | Encoding PBS | W256 plaintext modulus | Kernel outcome |
|---|---|---:|---|
| `kernel-256-initial-20260918` | (11,3) | 4096 | 127/128 coefficients wrong |
| `full-precision-20260918` | (4,15) | 4096 | 42/128 coefficients wrong |
| `full-deep-20260918` | (3,20) | 2048 | both requested products correct |

The first two trials decoded the inputs correctly after re-encoding and packing.
Their convolution errors exceeded the decoding margin. The final trial changes
both encoding precision and BSK decomposition, not one isolated variable.
Cryptographic randomness differs between runs, so the counts are observations,
not exact outcomes guaranteed by rerunning the same command.

The last grid's `source/` contains the source and lock file embedded in its
binary. Earlier trials retain parameter records and, where available, source
fingerprints, but predate embedded source snapshots.

## Noise Status

The subsequent [literature-based screen](NOISE_ANALYSIS.md) does NOT meet
2^-128 for this candidate. In particular, a functional success above is not a
parameter approval. The code and numerical records keep approval false.
All nine Python analysis tests pass, including a cross-check of modeled event
counts against the encrypted width grid. The complete source passes Rust's
format check.

## Reproduction

Run from this adapter directory with a populated Cargo cache:

```bash
cargo test --release --locked --offline
sh run.sh --widths 8 256 --patterns max random --output results/full-deep-repeat
sh run.sh --widths 8 16 32 64 128 256 --patterns random --seed 20260919 \
  --output results/width-grid-repeat
python3 -B -m unittest test_noise.py
python3 -B analyze_noise.py --manifest results/width-grid-repeat/manifest.json \
  --output results/noise-screen-repeat.json
```

Each output path must be new. The frozen September 8 server package, its
method catalog and builder are unchanged; this adapter is not vendored into it.
