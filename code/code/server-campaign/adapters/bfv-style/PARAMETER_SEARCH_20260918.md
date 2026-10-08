# BFV-Style Parameter Search

September 18, 2026. Independent CLOT-style implementation for general AB mod
2^W with clean TFHE radix-4 inputs and outputs. This is a local research
candidate, not CLOT author code or a final benchmark. The frozen server package
and manuscript are unchanged.

This record preserves the original one-thread timings. Subsequent
[parallel optimization](OPTIMIZATION_20260918.md) keeps these parameters and
PBS computations, reorders independent tasks, and shares duplicate KS work.
Its multi-thread timings and protocol are recorded separately.

## Selected Candidate

The selected file is `parameters/bfv-n800-range-aware.json`. Pairs below mean
`(base_log, levels)`, not `(base, levels)`.

| Item | Value |
|---|---|
| Ciphertext modulus | 2^64 |
| Small LWE dimension | 800 |
| GLWE dimension / polynomial size | 1 / 2048 |
| Small LWE error standard deviation, torus units | 2^-17 |
| GLWE error standard deviation, torus units | 9.25119974676756e-16 |
| Ordinary PBS | (15, 2) |
| Input-encoding PBS | (1, 43) |
| Ordinary KS | (1, 15) |
| Packing KS | (3, 14) |
| Relinearization | (12, 3) |
| Input-encoding LUT / normalizer LUT | 4 values / 8 values |
| Native radix / encoding scale | 4 / 2^59 |

The full convolution layout is unchanged: at W=256, 128 encrypted radix-4
digits are packed into each polynomial, the product has degree at most 254,
and its coefficients are at most 1152. The convolution plaintext modulus is
2048, without an extra padding bit. Output extraction retains the lower W bits.

### Two Construction Changes

This is not decomposition retuning alone. Keeping the original 16-value PBS
decisions at N=2048 leaves insufficient modulus-switching margin in this model.

1. A clean radix-4 input contains only 0,1,2,3. Publicly multiply its phase by
   four before ordinary KS, then use a four-value input-encoding LUT. This
   increases the decision spacing from q/32 to q/8. Input variance is multiplied
   by 16; the subsequently added KS error is not.
2. During output normalization, track public digit bounds and group terms only
   while their sum is at most seven. Publicly multiply the sum's phase by two
   before KS, then evaluate low/carry with eight-value LUTs. Their outputs
   retain the original q/32 scale. The incoming variance is multiplied by four,
   with fresh KS error added afterward. This is a smaller-capacity sequential
   reference normalizer, not the paper's optimized parallel normalizer.

At W=256 this uses 256 input-encoding PBS, 1161 bit-extraction PBS and 1104
normalization PBS: 2521 in total. Including 128 final decoding events gives
2649 modeled decisions. The original diagnostic used 1925 PBS. More PBS calls
are accepted here in exchange for wider decision margins and cheaper ordinary
PBS parameters.

## Conditional Failure Screen

The model and assumptions are in [NOISE_ANALYSIS.md](NOISE_ANALYSIS.md).
The selected estimates use the layout-aware CLOT C.1 adaptation, Gaussian and
variance-additive approximations, the existing FFT-error model, and a union
over every decision in the actual public schedule. No empirical noise floor
or fitted inflation factor is used. General XY is the scope; squaring is not
certified by this search.

| W | log2 whole-operation Gaussian union estimate, FFT included |
|---|---:|
| 8 | -239.263166 |
| 16 | -237.262985 |
| 32 | -235.821723 |
| 64 | -234.416400 |
| 128 | -233.201263 |
| 256 | -145.516051 |

These are conditional estimates, not measured failure rates. They are
recomputed by `verify_candidate.py`; no passing value is hard-coded into its
gate. The report retains `whole_multiplier_approved=false` to distinguish
the conditional screen from unconditional certification.

The layout matters. Treating every polynomial coefficient as having the same
variance as a filled packing slot gives a much looser dense-envelope estimate:
at W=256, -5.526208 with the FFT model and -52.235112 without it. The
layout-aware counterpart without FFT is -231.942320. Thus the selected result
must not be described as passing the dense envelope, or as independent of the
FFT model. The sparse calculation uses the known empty message slots and their
lower error variance; its reduction to CLOT's dense formula is unit-tested.
Coefficient-independence and shared-evaluation-key approximations remain
assumptions. A passing Gaussian model alone does not prove their tail behavior.

## Security Search

All rows use binary secrets, q=2^64, discrete Gaussian errors, unlimited samples,
and the full eight-attack lattice-estimator rather than its rough mode.
Estimator commit: `6019056011d10d7e9c30a0d5da2d2f729fbc2eec`, Sage 10.9.rc0.

| Small LWE candidate | Minimum estimated classical attack cost, log2 | Decision |
|---|---:|---|
| n=768, sigma=2^-16 | 135.4189 | Security passes; original restoration noise unsuitable |
| n=800, sigma=2^-18 | 127.5208 | Rejected |
| n=832, sigma=2^-19 | 126.4964 | Rejected |
| n=896, sigma=2^-21 | 124.7319 | Rejected |
| n=800, sigma=2^-17 | 134.0332 | Selected |
| n=832, sigma=2^-18 | 132.3086 | Alternative |

The unchanged N=2048 GLWE instance has minimum 129.168621 bits in the
previously evaluated kN-dimensional LWE proxy, recorded in
`../../results/st-retune-security-full-20260907.json`. The preflight matches
both dimensions and noise distributions exactly, requires all eight attack
results, and recomputes the minimum. This is not a ring-specific proof or a
new proof for the evaluation keys, including the encryption of S^2 used for
relinearization. The usual secret-dependent-key assumptions must remain stated.

## Search Coverage

`search_parameters.py` examines n/sigma pairs (800,2^-17), (832,2^-18),
(1088,2^-25), (1152,2^-26) at N=2048 and 4096. It constructs decomposition
frontiers for base logs 1..16 with positive levels and base_log*levels<64.
KS minimizes its modeled variance; relinearization is fixed at (12,3).

The eight dimension cases screen 41,600 combinations on those frontiers.
Necessary noise filters are followed by the full event schedule for candidates
ordered by an approximate work score. The search stops after twelve passing
candidates per dimension case; 36 candidates receive the complete screen.
This is a bounded search, not an exhaustive optimality claim or a latency
prediction. No N=4096 candidate passed within this search and numerical model.

The selected n800 candidate is rank two. Rank one uses packing (3,13) and
passes at -134.255929; one additional packing level improves the model margin
to -145.516051 at the cost of a larger packing key. The original n1152
diagnostic's functional successes did not imply a passing failure model.

## Reproduction

From this adapter directory, with an existing offline Cargo cache:

```bash
cargo test --release --locked --offline
python3 -B -m unittest test_noise.py
python3 -B verify_candidate.py --output results/candidate-preflight-repeat.json

sh run.sh --parameters parameters/bfv-n800-range-aware.json \
  --widths 8 16 32 64 128 256 --patterns random --seed 20260920 \
  --output results/retuned-grid-repeat
python3 -B analyze_noise.py --manifest results/retuned-grid-repeat/manifest.json \
  --output results/retuned-grid-repeat-noise.json

sh run.sh --parameters parameters/bfv-n800-range-aware.json \
  --widths 8 256 --patterns max random --seed 20260921 \
  --output results/retuned-extremes-repeat

python3 -B search_parameters.py --output results/parameter-search-repeat
```

All output paths must be new. Cryptographic keys are fresh per process; the
seed fixes plaintexts only. No ongoing server run is involved.

The complete lattice search is separately reproducible with Sage:

```bash
DOT_SAGE=./.sage-repeat sage -python ../../st_security.sage.py \
  --estimator <path-to>/lattice-estimator \
  --input parameters/security-search-additional.json \
  --output results/security-additional-repeat.json --full
```

Use `parameters/security-search-input.json` to repeat the first four candidates.
The estimator repository must be at the commit above. The numerical preflight
reads the retained security reports; it does not rerun Sage at every execution.

## Performance Scope

Times for this implementation include input re-encoding, packing, the BFV-style
kernel, coefficient extraction/decomposition, and clean-radix restoration.
They exclude key generation, initial input preparation and decrypted checks.
Evaluation uses one thread. These are untuned local diagnostics without
warm-up or statistical benchmarking, not final latency claims.

The first encrypted grid (`results/retuned-grid-20260918`) passes at all six
widths under one fresh key set:

| W | Sum of measured stages (s) | Tensor + relinearization (ms) | Executed PBS |
|---|---:|---:|---:|
| 8 | 2.956 | 4.678 | 30 |
| 16 | 6.343 | 4.485 | 82 |
| 32 | 13.656 | 4.568 | 200 |
| 64 | 29.135 | 4.473 | 484 |
| 128 | 63.032 | 4.638 | 1103 |
| 256 | 131.590 | 4.813 | 2521 |

`results/retuned-extremes-20260918` adds four successful products under a
second fresh key set: W8 and W256, each with `max` and `random`. Here `max`
means A=2^W-1 and B=2^W-2, not squaring. The two extra W256 stage sums are
132.552 s and 130.171 s. In total, all ten encrypted products pass, including
three W256 products. Neither the repeated trials nor the observed phase-error
ratios are used to fit the analytical failure screen.

The W256 stages are input re-encoding 75.206 s, packing 4.381 s, tensor and
relinearization 0.004813 s, sample extraction 0.000536 s, coefficient
decomposition 26.843 s, and normalization 25.156 s. The kernel accounts for
less than 0.01% of the stage sum. It would therefore be misleading to compare
its millisecond latency alone against an end-to-end radix multiplier.

Evaluation-key payload is 3,495,772,160 bytes (3.256 GiB), including a 2.100 GiB
encoding Fourier BSK and an 896 MiB packing key. This is neither serialized size
nor peak RSS. More precise input encoding costs both time and key memory.

`results/retuned-grid-noise-20260918.json` recomputes the conditional model from
the parameters and public plans embedded in this executed grid. Its 2649 W256
events include exactly the 2521 PBS calls observed in the encrypted execution.
The first extracted bit of the last retained convolution coefficient is the
worst modeled event; the bit-extraction family dominates the union. The
normalization family is -231.922216, while input re-encoding is -947.744025.

Seven Rust tests and fifteen Python tests pass. The Rust checks include 1200
additional plaintext cases for the range-aware schedule through W256; the
Python checks cover all-width screens, runtime counts, security-row matching,
and preservation of the old diagnostic's 24 numerical values. Functional tests
do not measure or validate a 2^-128 tail probability.
Runtime integer parameters must match exactly. Noise standard deviations allow
at most two binary64 ULPs for Rust/Python decimal parsing: the recorded GLWE
sigma differs by one ULP. The numerical report is recomputed from the executed
manifest, so that conversion is not hidden by using the input JSON alone.

A matching one-thread diagnostic of the audited grouped-hybrid executable can
be run after the BFV process has finished:

```bash
python3 -B compare_grouped.py --output results/grouped-w256-t1-repeat
```

The executed comparison is `results/grouped-w256-t1-20260918`. At W256,
one thread, zero warm-up and one random product, the grouped-hybrid evaluator
reports **50.958 s**, with correct output. Its runtime parameters and operation
counts match the earlier audited executable/plan; PBS normalization uses 1141
calls. The three BFV observations above take **130.171--132.552 s**. Thus this
local diagnostic favors the current grouped implementation, not the BFV probe.
The grouped result is one evaluator wall-clock measurement; BFV times are sums
of instrumented stages excluding small uninstrumented setup intervals. Neither
is a final statistical comparison, and the old eight-thread timings are not
used to compute a speedup.

The helper verifies the original executable hash and reuses its public plan,
while changing only the thread setting, plaintext seed and output paths. It
does not modify the binary, plan, frozen package or server. A one-shot
comparison cannot establish a general speed advantage. In particular, a fast
tensor kernel does not eliminate the radix conversion costs, and this simple
full-convolution/decomposition path is not necessarily the best BFV-style
construction.
