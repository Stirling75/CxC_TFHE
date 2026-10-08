# CLOT-Based Adapter: Server Integration

September 21, 2026. No cryptographic parameter was changed in this integration.
The candidate remains `parameters/bfv-n800-range-aware.json`. The original
September 8 archive, server binaries and existing results are untouched.

## Construction And Analysis Checks

CLOT's extended November 2021 paper was reread: Algorithm 1 and Theorem 1,
pp. 9-11; quotient moments in Appendix C.1, pp. 55-56; packing in Appendix D,
pp. 62-63. Source PDF SHA-256:
`96d2b32894045375dd2f2dea5a01578b0163e39c19e88151cdec321ec1050dcd`.

The k=1 implementation uses centered lifts, polynomial products modulo 2^128,
scale-and-round followed by signed decomposition and positive relinearization
under the b-aS phase convention. Independent BigInt tests check the widened
tensor. Plaintext tests check convolution, truncation and both normalizer
schedules through W256. This radix-convolution/restoration path is our
adaptation, not CLOT author code or an optimality claim.

The noise screen retains the quotient/error contribution, separate filled and
empty packing-coefficient variances, residual error reuse during bit extraction,
scale-before-KS ordering, normalization decisions and final output decoding.
The layout-aware adaptation reduces to CLOT Eq. (1) for dense inputs in a unit
test. Its Gaussian/variance-additive and FFT approximations are unchanged.

| Width | log2 conditional whole-operation estimate |
| --- | ---: |
| 8 | -239.263166 |
| 16 | -237.262985 |
| 32 | -235.821723 |
| 64 | -234.416400 |
| 128 | -233.201263 |
| 256 | -145.516051 |

At W256 the plan has 2521 PBS calls and 2649 modeled events. With parallel
normalization, shared low/carry key switching reduces actual KS calls to 1972;
the serial reference has 2521. Duplicate decision events in the screen remain
conservative rather than claiming independence of the shared input.

Every preflight re-derives the noise values, checks the hashes of both retained
security reports, matches dimensions/noise distributions, and checks all eight
recorded attack estimates. Minima remain 134.0332 bits for small LWE and
129.1686 bits for the GLWE LWE proxy. This is verification of retained reports,
not a fresh Sage run or a proof for the complete secret-dependent key set.

## Measurement Fix

The earlier prototype summed separate phase timers and decrypted intermediate
states between the kernel and restoration. New `--benchmark` mode runs both
inside one continuous outer timer. All intermediate decrypted checks, BigUint
reference arithmetic and optional serial-equivalence work follow this timer.
Input generation/initial refresh and key generation remain outside it.

`total_seconds` is the comparison metric. Phase timers remain diagnostic and
must sum to no more than the outer measurement. The checker rejects legacy
stage-only records when called by the campaign and excludes warm-ups from the
mean. It validates every trial, actual output digits, exact input/output
contracts, parameters, PBS/KS counts and model/runtime agreement.

## Local Verification

- Eight Rust arithmetic tests pass.
- Twenty-seven Python model/execution tests pass in the research directory.
  Historical encrypted fixtures are optional in the source-only package;
  security reports and synthetic validator tests are included.
- The new continuous-timing mode passes ten fresh encrypted products at
  W16/32/64/128/256, T8: one warm-up and one measured random product per width.
  Raw data: `results/server-integration-grid-20260921`; independent audit:
  `results/server-integration-grid-audit-20260921.json` in the research tree.
- Two more products, W16 and W256 with A=2^W-1 and B=2^W-2, pass under
  another fresh key set. Re-evaluating each with one thread yields identical
  ciphertexts to the eight-thread execution. Data and audit are
  `results/server-integration-extremes-20260921` and
  `results/server-integration-extremes-audit-20260921.json`.
- A pre-release package passes all 18 runner tests, 17 shared-analysis tests,
  three public-plan tests, and its CLOT suite (18 passed, nine optional
  historical-fixture tests skipped), plus all eight CLOT Rust tests. A release
  build succeeds offline with an initially empty Cargo home. Linux x86-64
  dependency resolution also succeeds offline; this is not a Linux build.
- A dry run prepares 154 supported cases across all 16 methods, five widths,
  and thread budgets 1/8, explicitly skipping the six unsupported PARMESAN
  cells. All ten CLOT commands use the outer timer, five measurements and
  one warm-up. Ninety-four frozen-base code/license files match their hashes.
- The preflight is retained as `results/server-preflight-20260921.json` in the
  research tree. No empirical floor or measured-error fitting is used.

These are local functional checks, not final server latency measurements or
observations of a 2^-128 event. Linux compilation and affinity-backed timing
must be checked on the server. Follow the top-level `README.md` for build,
test, smoke and five-measurement commands. The candidate remains marked
`whole_multiplier_approved=false`: general independently prepared AB under
the documented conditional model, not identified-ciphertext squaring.
