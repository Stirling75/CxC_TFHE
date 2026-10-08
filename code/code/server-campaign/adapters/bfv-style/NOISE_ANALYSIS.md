# Literature-Based Noise Screen

September 18, 2026. This note records the original diagnostic and the model
used for subsequent tuning. It does not modify a running benchmark.
The [later parameter search](PARAMETER_SEARCH_20260918.md) records a different
n800, range-aware candidate that passes the conditional layout-aware screen.
The historical numerical table below remains the n1152 default, not that candidate.

## Sources And Scope

- CLOT, [extended ASIACRYPT 2021 paper](https://eprint.iacr.org/2021/729.pdf):
  Theorem 1 / Eq. (1), p. 10; PBS formula, Appendix B, p. 44; binary S^2
  moments, Table 5, p. 48; tensor expansion and quotient moments, Appendix C.1,
  pp. 51-58; relinearization, C.3, pp. 59-62; packing, Appendix D, pp. 62-63.
- [TFHE-rs Handbook](https://github.com/zama-ai/tfhe-rs-handbook), local first
  edition v1.6: LWE KS Theorem 6 / Eq. (10), p. 26; packing Section 2.5.3,
  pp. 28-29; blind rotation Theorem 11 / Eq. (24), p. 36. This edition's
  integer multiplication section uses bivariate PBS, not BFV tensor products.
- The campaign's existing `cbs_variance_estimator.py` supplies the CLOT-style
  PBS and KS variance expressions, Gaussian tails, ordinary modulus-switching
  variance, and its Refined-style FFT estimate. No probe-derived floor is used.

Source fingerprints (SHA-256):

```text
CLOT PDF: 96d2b32894045375dd2f2dea5a01578b0163e39c19e88151cdec321ec1050dcd
Handbook PDF: c4519a8f71f6daf8695f375deabc484c8f675587d22bc745c70aac344c820356
primitive estimator: e2e12dad5a9a35ab2d1042fd7245ee91494ce0e4eacd8f8647e7514ea35ab91b
```

All variances below use integer ciphertext-modulus units, q=2^64. Only general
XY multiplication is modeled. Binary secret coefficients have E[S]=1/2 and
Var(S)=1/4. As in the source estimates, this screen uses Gaussian and
variance-additive approximations, including fresh-error treatment across
primitive calls. It is not a proof of joint distributions for shared evaluation
keys or of the Gaussian tails of products of errors. Nonzero rounding means
and conditional distributions are not newly certified here.

## Packing And Multiplication

For packing d LWE inputs of variance Venc and dimension n_in into one GLWE,
let Bp and lp be the packing decomposition base and level count. The model is

    R = n_in ((q^2/Bp^(2lp)-1)/24 + 1/16),
    K = d n_in lp (Bp^2+2)/12 sigma_packing,int^2,
    Vfill = Venc + R + K,     Vempty = K.

This is CLOT's signed-digit moment convention. The handbook gives a more
detailed decomposition correction epsilon(B,l); it supports the filled/empty
structure, but that correction is not silently substituted into the existing
CLOT primitive model.

For k=1 and equal encoding scale Delta, the tensor error before relinearization
has the following contributions (CLOT C.1):

    M1 E2 + M2 E1 + E1 E2 / Delta
    + (q/Delta)(E1 U2 + E2 U1) + tensor rounding.

U1,U2 are the quotient polynomials arising from centered lifts modulo q.
Dropping their contribution would badly underestimate this particular path.

Two calculations are provided:

1. `dense-envelope` applies Eq. (1) with every coefficient's variance replaced
   by Vfill and message coefficient magnitudes bounded by 3. It is an envelope
   inside the same stochastic approximation, not an assumption-free bound.
2. `layout-aware` uses the C.1 expansion with V_i=Vfill for i<d and Vempty
   otherwise. For output coefficient r, the two message/error terms use
   18 sum_{i<d} V_{r-i}, and the error/error term uses
   sum_i V_i V_{r-i}/Delta^2. Indices are modulo N; negacyclic signs vanish
   in these variance sums under the coefficient-independence approximation.
   The quotient term uses (q^2/Delta^2) E[U^2] 2 sum_i V_i, with CLOT C.1's
   approximation for E[U^2]. Tensor rounding and relinearization remain those
   of Eq. (1), using Table 5's binary S^2 moments.

The second calculation is our coefficient-sum adaptation, not CLOT's C.2
formula copied verbatim: C.2 explicitly specializes to coefficients containing
one product and excludes coefficients containing sums of products. A test
checks that the adaptation reduces exactly to Eq. (1) for dense inputs with
equal coefficient variances.

## Original Event Schedule

The script propagates these variances through the code's actual public schedule:

- Input re-encoding: native PBS-output noise, ordinary KS, and modulus switch.
- Bit extraction: public scaling of the residual BEFORE ordinary KS, followed
  by sign PBS with margin q/4. Previous extracted-bit errors remain in the
  residual; the model does not refresh that residual for free.
- Radix restoration: bit output rescaling, pairing, four-to-two reduction,
  and carry propagation. Repeated occurrences of an explicitly tracked error
  are combined by their amplitudes before computing variance.
- Final decoding: margin Delta_native/2, one event per output digit.

Separate PBS outputs are still treated as variance-additive under the stated
heuristic. Source tracking does not establish independence of different calls
sharing a BSK or of packing outputs sharing a key. The outer union does not
need event independence, but this does not validate the inner Gaussian model.

At W=256 there are 256 input re-encoding, 1161 bit-extraction, 508 normalizer
and 128 final-decoding events: 2053 in total. The first three counts match the
executed Rust PBS count of 1925. Convolution-decoding checks in the harness are
not additional runtime decisions and are not double-counted.

## Original Diagnostic Numbers

Input: `results/width-grid-20260918/manifest.json`, default (3,20) encoder.
Output: `results/noise-screen-20260918.json`, with every event and variance term.
The table reports raw log2 sums of Gaussian failure estimates. All figures
remain conditional on the approximations above, not measured failure rates.

| W | Layout-aware, no PBS FFT term | Layout-aware, with PBS FFT model | Dense envelope, with FFT |
|---|---:|---:|---:|
| 8 | -61.19 | -61.19 | -61.19 |
| 16 | -60.09 | -60.09 | -60.09 |
| 32 | -58.81 | -58.81 | -58.81 |
| 64 | -57.60 | -57.60 | -57.60 |
| 128 | -56.50 | -56.50 | -14.98 |
| 256 | -55.42 | -13.19 | +4.82 |

A positive raw union sum is capped at probability one; it is not a probability
greater than one. None of these candidate configurations passes -128.
Failure of a sufficient model screen does not establish that the true failure
probability is above 2^-128.

The no-FFT column is an optimistic sensitivity calculation, NOT the implemented
backend: PBS still uses FFT. Integer tensor, packing and relinearization do not.
The FFT term is the existing numerical-error model, not a new empirical floor
and not a certified bound for this unusually deep encoding BSK.

At W=256 with the layout-aware FFT model:

- Encoding PBS variance: 1.51e17 algebraic + 8.56e18 numerical-model contribution.
- Last convolution coefficient: quotient/error term 8.35e29; message/error
  term 2.01e22; relinearization key term 1.92e14. The quotient term dominates.
- Product sigma / decoding margin: 0.20288, a modeled standard deviation,
  not an observed maximum-error ratio.
- Input re-encoding event union: -57.00; bit extraction: -13.19; normalizer:
  -56.01. Final decoding is negligible in this screen.
- Even removing the PBS FFT term leaves the native 16-value PBS decisions
  around -65 per call at N=2048, hence about -55.42 for the complete schedule.

Thus there are two distinct tuning questions: the input-decision precision of
the native PBS path, and the small post-PBS error required before a BFV tensor
product. Neither is fixed merely by the observed correctness of ten trials.

## Subsequent Parameter Work

The subsequent search evaluates larger ordinary PBS rings and compatible
lower-precision boundary conversions to address the native decision margin.
The selected candidate uses the latter, plus a more precise input encoder.
Bounded-length convolutions remain an alternative: lowering convolution
precision reduces both quotient-error amplification and the extraction burden.
Changes must include their actual conversion costs and new key payloads.
Do not treat the present full-width, bit-extraction reference as the best
BFV-style algorithm or infer that BFV-style multiplication is inherently slow.

The old presets remain reproducible. The main campaign parameters, frozen
package and manuscript are unchanged. Parameter approval still requires
reviewing the approximations, numerical implementation and complete key set.

The original nine Python tests cover the dense-formula limit, sparse convolution
terms, packing variance, residual reuse, scale-before-KS order, FFT sensitivity,
event unions, and count agreement with the saved encrypted trials. The expanded
suite also checks the range-aware schedule, public scaling, original numerical
regressions, security preflight and the retuned encrypted grid.
