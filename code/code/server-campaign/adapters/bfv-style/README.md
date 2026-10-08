# BFV-Style Multiplication Probe

Research addition to the existing server-campaign working tree, September 18,
2026. This is an independent implementation, not author code or a reproduction
of CLOT's reported latency. It remains absent from the original September 8
server package. The September 21 package adds it as `clot-bfv`, preserving the
old methods. Do not alter a campaign that is already running.

## September 21 Server Integration

The campaign supplies the explicit n800 range-aware parameter file and
`--benchmark`. This mode measures one uninterrupted call from native-radix
inputs through restored radix output, before any intermediate decrypted checks.
The reported latency is `total_seconds`, not the historical stage sum.
Warm-up exclusion, output digits, operation counts, actual parameters and the
conditional Gaussian estimate are independently checked. Security-report hashes
are pinned. See [validation](SERVER_VALIDATION_20260921.md) and the packaged
top-level `README.md` for the benchmark commands.

## Retuned Research Candidate

[Parallel execution](OPTIMIZATION_20260918.md) adds a bounded Rayon pool,
independent input/coefficient work, exact parallel packing and dependency-graph
normalization. Parameters and PBS count are unchanged; duplicate low/carry KS
is shared. `--verify-serial` compares same-key ciphertexts byte for byte.
The default remains one thread.

The same-eight-thread W256 preliminary comparison, with one warm-up and three
measured runs, averages 25.915 s for BFV and 9.634 s for grouped hybrid. The
linked record contains all-width timings, 31 successful input cases and seven
same-key serial/parallel ciphertext equivalence checks. These are not final
server measurements or a claim about optimal BFV-style constructions.

[Parameter search and reproduction](PARAMETER_SEARCH_20260918.md) records the
new n800/N2048 candidate, its range-aware input/output conversions and separate
noise/security screens. The conditional layout-aware Gaussian model, including
the existing FFT term, gives log2 union -145.516051 at W=256. Conventional
classical estimates are 134.0332 bits for the small LWE and 129.1686 bits for
the GLWE's LWE proxy. Neither number is a new proof for the complete key set.

Ten encrypted products under two fresh key sets pass, covering every width
8..256 and three W256 trials. The 2521 executed PBS at W=256 match the model's
schedule. These are local
functional checks, not rare-failure measurements or final speed benchmarks.
Use the explicit parameter file; the legacy default is retained to reproduce
the earlier diagnostic, which did NOT pass the noise screen.

```bash
python3 -B verify_candidate.py --output results/candidate-preflight-repeat.json
sh run.sh --parameters parameters/bfv-n800-range-aware.json \
  --widths 8 16 32 64 128 256 --patterns random \
  --output results/candidate-repeat
```

## Scope

The full path accepts bootstrapped radix-4 LWE blocks at scale `2^59` and returns
the lower W bits as radix-4 blocks under the same extracted GLWE key and scale.
Supported widths are 8, 16, 32, 64, 128, and 256. The evaluator uses only public
evaluation keys and ciphertexts. Decryption and BigUint arithmetic occur only
in the checker, outside the measured stages.

1. PBS re-encodes each native block at the convolution scale.
2. Packing key switching places the blocks at consecutive GLWE coefficients.
3. One k=1 BFV-style tensor product, scale-and-round, and relinearization
   compute the polynomial convolution.
4. Sample extraction returns the lower W/2 convolution coefficients.
5. LSB-first sign bootstrapping decomposes each coefficient into radix digits.
6. Sequential reduction and carry propagation restore clean radix-4 output.
   The legacy default uses four-to-two groups; the new candidate bounds each
   group sum by seven. Neither is our optimized parallel normalizer.

See [construction and analysis](../../BFV_STYLE_CONSTRUCTION.md) for the
256-bit extension, correctness conditions, noise terms, and alternatives.
The [original local validation](VALIDATION_20260918.md) records ten successful
encrypted products and the earlier failures. The [literature-based noise
screen](NOISE_ANALYSIS.md) does not meet the target for that ORIGINAL diagnostic;
the retuned candidate and its limitations are recorded separately above.
Correctness tests alone must not be used to approve either candidate.

## Commands

Run from this directory. `run.sh` wraps `cargo run --release` and uses the
package-root `.cargo/config.toml`, so it builds offline from the vendored
crates. Each output directory must be new; create its parent first.
Inside a built server package, point `--output` outside `code/`, for example
`--output /path/outside/package/results/smoke`. Any file written under
`code/` changes the campaign source identity, and the runner then refuses the
existing build.

```bash
mkdir -p results
cargo test --release --locked --offline

# No cryptographic work: inspect all-width encoding/range plans.
sh run.sh --plan-only --widths 8 16 32 64 128 256 --output results/plan

# Small encrypted end-to-end check, including all conversion and normalization.
sh run.sh --widths 8 --patterns zero one max alternating impulse random \
  --output results/smoke

# Explicitly tests the kernel only, NOT a complete multiplication interface.
sh run.sh --kernel-only --widths 256 --patterns max random \
  --output results/kernel-256

# Full 256-bit encrypted multiplication, with carry-producing and random inputs.
sh run.sh --widths 256 --patterns max random --output results/full-256

# Conditional literature-based variance screen, with and without PBS FFT terms.
python3 -B -m unittest test_noise.py
python3 -B analyze_noise.py --manifest results/full-256/manifest.json \
  --output results/full-256-noise.json
```

The default `--preset deep-encoding-diagnostic` uses a separate (base_log=3,
levels=20) BSK only for input re-encoding. Input preparation and restoration
retain (11,3). Keys, dimensions and underlying noise distributions are unchanged.
The earlier `--preset st-diagnostic` uses (11,3) throughout; its W=256 kernel
check failed even though input rescaling and packing decoded correctly. Keep
that negative result rather than treating the original settings as usable.
An intermediate `precision-diagnostic` setting uses (4,15). The first two
W=256 attempts also reserved a padding bit for convolution coefficients and
both failed. The current default does not reserve this unnecessary bit:
the sign-based decomposition accepts the full plaintext interval. Use
`--coefficient-padding` to reproduce the earlier encoding choice explicitly.

```bash
# Negative diagnostic, expected to expose inadequate multiplication precision.
sh run.sh --preset st-diagnostic --coefficient-padding --kernel-only --widths 256 --patterns max \
  --output results/original-encoding-check
```

`--repetitions` defaults to one and `--warmup` to zero. Warm-up rows, when
requested, are retained separately and excluded by `audit_run.py`. These are
local diagnostic timings, not a final server benchmark. Plaintext inputs are seeded;
cryptographic keys and encryption
randomness come from the library's OS seeder. Re-running creates fresh keys.
Never use these diagnostic timings as the final performance comparison.

The locked dependencies include TFHE-rs 1.6.1, tfhe-csprng 0.9.0, and
tfhe-safe-serialize 0.1.0. Resolving the latter two to newer compatible-looking
versions caused a versionable-trait mismatch during initial compilation. Keep
the lock file. This standalone research directory needs a populated Cargo
cache. The September 21 server package vendors all four locked Rust manifests
and builds offline from its package root without fetching dependencies.

The packing key alone is 960 MiB, the ordinary Fourier BSK 216 MiB, and the
additional encoding Fourier BSK 1440 MiB for the default (3,20) setting
(1080 MiB for the intermediate (4,15) setting). Allow additional memory during key
generation. It uses exact standard-domain packing/relinearization and defaults
to one evaluation thread. `--threads` selects a bounded local worker pool;
the standalone adapter does not set affinity. In the server package the outer
campaign runner enforces Linux affinity and records the selected CPU IDs.

## Results And Limits

- `manifest.json`: compiled-source and binary fingerprints, exact parameters,
  input/output contracts, key payload, key-generation time, plaintext seed,
  and public width/PBS-count plans.
- `source/`: the source and Cargo lock embedded in the executable, retained
  when starting a new run. Early development trials predate this addition and
  have parameter records/fingerprints but not embedded source snapshots.
- One JSON file per width/pattern/repetition: operands, per-stage correctness,
  observed phase-error-to-margin ratios, operation counts and separated times.
- `completed.json`: successful end of all requested diagnostic cases.
- Failed checks retain their record and stop with a nonzero exit code. Kernel
  checkpoints are saved before the potentially longer decomposition stage.
- Phase-error ratios are decrypted observations, NOT estimated failure bounds.
- Input preparation (encryption plus initial refresh), key generation and
  checking are excluded from stage timings. Native-input rescaling, packing,
  multiplication, extraction, digit decomposition and normalization are included.
- For the legacy default, base dimensions, noise, PBS and ordinary KS follow the existing ST
  research candidate; the encoding BSK, packing and relinearization
  decompositions are new.
  That default's conditional whole-operation screen does not meet -128;
  it is not a certificate. Security for the complete evaluation-key set has
  NOT been certified. Every record keeps approval false.

The original prototype did not change the main campaign. The September 21
integration adds a new method without changing any prior method's source or
parameters. No manuscript or historical result is changed.
