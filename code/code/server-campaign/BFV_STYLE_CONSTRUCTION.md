# BFV-Style Multiplication: Construction Before Benchmarking

September 18, 2026. General AB modulo 2^W, with TFHE radix input and output.
This note and the accompanying probe are research additions, not final paper
results or approved parameters.

Subsequent [parameter search](adapters/bfv-style/PARAMETER_SEARCH_20260918.md):
a separate n800/N2048 candidate with four-value input encoding and an
eight-value, public-bound normalizer passes the conditional layout-aware
Gaussian screen through W=256 (-145.516051 with the FFT model). Ten encrypted
products pass under two fresh key sets. The original n1152 diagnostic and its failed
noise screen remain historical evidence, not the selected parameters.

## 1. Primary-Source Reading

The requested CLOT reference is Chillotti, Ligier, Orfila and Tap, *Improved
Programmable Bootstrapping with Larger Precision and Efficient Arithmetic
Circuits for TFHE*, ASIACRYPT 2021. Read the
[conference version](https://www.iacr.org/archive/asiacrypt2021/130900334/130900334.pdf)
and [extended version](https://eprint.iacr.org/2021/729.pdf), especially Section
3.1, Algorithms 1 and 2, Section 3.1.2, Section 5.1.3, and Appendices C and D.

The relevant primitive is GLWE tensor multiplication, scale-and-round, and
relinearization. LWE multiplication adds packing key switching before this
primitive and sample extraction afterward. This is not necessarily a switch
to a separate BFV implementation. The paper also considers packing multiple
products or their sum. Its larger-integer discussion retains carries rather
than treating a modular local product as a full integer product.

The polynomial-convolution layout and the reference output restoration below
are our adaptation. Do not attribute this complete 256-bit implementation,
the selected parameters, or its measured timings to CLOT. The paper's
128-bit *security* table is not a certificate of 2^-128 *failure probability*
for this adapted multiplier.

## 2. Three Candidate Comparisons

| Route | Representation | Main question |
|---|---|---|
| Direct GLWE multiplication | Native modulus, packing KS, BFV-style tensor/relinearization | Can TFHE radix boundaries be preserved efficiently? |
| Separate BFV backend | Scheme conversion plus BFV evaluation plus conversion back | How much do the real conversion keys, noise and digit extraction cost? |
| Tiled direct multiplication | Several bounded-length convolutions, then radix reduction | Does less precision/noise compensate for more products? |

The first route is implemented as a correctness probe. This avoids silently
timing BFV encryption of plaintext inputs as though it were TFHE-to-BFV
conversion. Native BFV-only timings could be supplemental, but cannot replace
the matched-input/output comparison required for the revision.

## 3. Extending To 256 Bits

Let beta=4 and d=W/2. Write A=sum_i a_i beta^i and B=sum_i b_i beta^i,
with 0<=a_i,b_i<=3. Pack the *encrypted* digits as coefficients of

    A_poly(X) = sum_i a_i X^i,
    B_poly(X) = sum_i b_i X^i.

Their product has coefficient c_q=sum_{i+j=q} a_i b_j. If 2d-2<N, there is
no negacyclic wrap in the message polynomial. Extract only q<d because the
other coefficients contribute multiples of beta^d to AB. Do NOT reduce the
message polynomial modulo X^d+1: the negative wrap would corrupt the low part.

For W=256, d=128 and the product degree is at most 254. This fits easily in
N=2048. Every coefficient is at most 9d=1152. The current probe uses plaintext
modulus t=2048 and encoding Delta=q/t, with q=2^64. Thus one ring multiplication can
compute the convolution of a 256-bit pair without a 256-bit plaintext modulus.
This says nothing yet about the cost of packing and restoring radix output.

| W | d | Max coefficient | t (no coefficient padding) |
|---|---|---|---|
| 8 | 4 | 36 | 64 |
| 16 | 8 | 72 | 128 |
| 32 | 16 | 144 | 256 |
| 64 | 32 | 288 | 512 |
| 128 | 64 | 576 | 1024 |
| 256 | 128 | 1152 | 2048 |

Increasing the digit radix is not free. With beta=2^b, d=ceil(W/b), the
coefficient bound is d(beta-1)^2; the required precision grows accordingly.
For W=256: beta=2 gives bound 256, beta=4 gives 1152, beta=16 gives 14400,
and beta=256 gives 2080800. The native input is radix-4, so changing beta also
requires a genuine encrypted conversion. These alternatives are not timed yet.

## 4. Exact Tensor Arithmetic

For k=1 and ciphertexts (a_1,b_1),(a_2,b_2), compute

    T = round(a_1 a_2 / Delta) mod q,
    A = round((a_1 b_2 + b_1 a_2) / Delta) mod q,
    B = round(b_1 b_2 / Delta) mod q.

All products are negacyclic polynomial products. Lift ciphertext coefficients
to centered integers before multiplication. Products are computed modulo q^2
using u128, and are rounded only afterward. Reducing modulo q before division
would discard essential information. Since Delta is a power of two dividing q,
discarded multiples of q^2 become multiples of q after division, so wrapping
u128 accumulation is exact for the requested output. Rounding ties go upward;
the tests include negative and wrapped values.

The tensor phase is B-A*S+T*S^2. A GLev encryption of S^2 under S replaces
the quadratic-key term and returns an ordinary GLWE. The implementation uses
TFHE-rs signed decomposition and exact u64 polynomial arithmetic for this step.
The positive sign of this added relinearization ciphertext follows the TFHE
phase convention b-a*S; it must not be copied from an incompatible convention.

Key generation knows S, but the multiplication evaluator does not. As usual
with evaluation keys encrypting secret-dependent values, the complete key set
needs its security assumptions stated and assessed, not just an LWE dimension.

## 5. Matching The Input And Output

### Input encoding

The supplied blocks use Delta_native=q/32. At W=256 the convolution uses
Delta=q/2048. Dividing each coefficient of an LWE ciphertext by 64 is not an
automatic same-modulus encoding conversion: modular wrap terms matter. The
probe uses a real PBS to re-encode every digit, then packs the resulting big-key
LWE blocks. This conversion is included in the recorded stage time.

The initial N=2048 probe used the existing ST candidate's (11,3) PBS decomposition
also for this re-encoding. The W=8 full path passed, but the W=256 convolution
failed after correctly decoded rescaling and packing. Its retained record is
`adapters/bfv-style/results/kernel-256-initial-20260918/w256-max-r0.json`.
Being accurate enough to decode a digit is not the same as being sufficiently
quiet for a following BFV multiplication. The next diagnostic therefore changes
only the re-encoding BSK decomposition to (4,15), with a distinct key and its
time and memory included. That padded W=256 attempt reduced the error but
still failed: 42 of 128 coefficients decoded incorrectly in the saved trial.

The subsequent candidate uses (3,20) and removes the coefficient padding bit,
which sign-based extraction does not require. The native input/output PBS
domain still keeps its usual padding. Doubling Delta increases the decoding
margin and decreases the q/Delta multiplier in the core error expression.
This is a test candidate, not failure-certified tuning. All failed records are
retained; the two changes are not conflated as an isolated BSK-only comparison.

Input preparation separately encrypts and refreshes native-radix digits; it
is outside the operation timing. Thus the timed operation does not start from
plaintexts or from ciphertexts already prepared in the more favorable scale.

### Coefficient decomposition

An extracted c_q is an integer coefficient, not a clean radix digit. A single
large-domain ordinary PBS is not assumed to work just because the polynomial
has enough positions for all input digits. Lookup domain precision, modulus
switching error, and accumulated ciphertext noise are separate constraints.

The first implementation uses sign bootstrapping to extract bits from low to
high. This follows the same LSB-first shift/center/subtract structure as
TFHE-rs 1.6.1's core `fft64::crypto::wop_pbs::extract_bits`; our adapter
additionally constructs big-key native-scale digits for this output interface.
If a residual R encrypts 2^j u Delta, multiply it publicly by
2^(p-j-1), where t=2^p. Its phase is then (u mod 2)q/2, plus scaled error.
Add q/4 to center the two cases. A constant negacyclic LUT with value -h/2
returns -h/2 or +h/2; adding h/2 yields an encrypted bit at scale h.

Choose h=min(2^j Delta, Delta_native). Public integer multiplication then
produces both the bit contribution to subtract from R and the bit at native
scale. Subtracting removes that plaintext bit without dividing a ciphertext.
Crucially, scale the big-key input *before* ordinary key switching, so that the
new key-switch error is not amplified by the public factor.

Two such bits form one radix-4 digit. Digits beyond the W-bit output boundary
are never extracted. Each retained bit needs a PBS in this reference path;
this is not a claim of optimal CLOT digit decomposition or WoP-PBS performance.

At W=256, the public reference schedule uses 256 input re-encoding PBS,
1161 bit-extraction PBS and 508 restoration PBS, totaling 1925, plus 256
single-LWE packing contributions and one GLWE product. There are no CBS or
standalone CMux calls; the internal operations of each PBS are not zero-cost.
The code checks the executed PBS count against this public schedule.

### Radix restoration

Place the j-th digit of c_q in column q+j. Groups of four digits sum to at most
12, within the native 16-value padded PBS domain. Digit and carry PBS outputs
preserve that weighted sum. The reference reduces such groups and propagates
carries sequentially, discarding only those past the output boundary. Final
blocks are refreshed at the original scale and key, not merely decrypted and
re-encrypted. Parallel restoration is a later optimization.

Under correct primitive evaluations and correct bit decisions, this preserves
sum_q c_q beta^q = AB modulo beta^d and returns d digits in [0,3]. The condition
is essential: algebraic correctness is not a probabilistic correctness bound.

## 6. Noise Questions Before Parameter Approval

The core multiplication error is not only M_1*E_2 + M_2*E_1. Centered lifts
also create quotient polynomials U_1,U_2 from phase reduction modulo q. At equal
scales, the relevant terms include

    M_1 E_2 + M_2 E_1 + E_1 E_2 / Delta
    + (q/Delta)(E_1 U_2 + E_2 U_1)
    + tensor rounding + relinearization error.

These terms, and packing's filled/empty coefficient noise, must be accounted
for; see CLOT Theorem 1 and Appendices C/D. Increasing t=q/Delta both reduces
the decoding margin and amplifies a multiplication error contribution. Packing
more digits can also increase the noise. A wide convolution is not automatically
better than a tiled one.

For this adaptation the pending analysis must cover:

1. The input preparation and re-encoding noise actually produced by PBS.
2. Packing into all selected coefficients with shared evaluation keys.
3. Tensor rounding and the S^2 relinearization decomposition.
4. The scaled residual at each sign decision, including previously extracted
   bit errors reused in later residuals.
5. Ordinary key-switch noise, PBS modulus switching and numerical error.
6. Normalizer decisions and final radix decoding, for the complete schedule.

The implementation does not translate observed phase errors into a Gaussian
tail or a failure claim. It records observations only. The separate
[literature-based screen](adapters/bfv-style/NOISE_ANALYSIS.md) now applies
CLOT's model to the complete decision schedule, with an explicit sparse-layout
adaptation and FFT sensitivity. The diagnostic candidate does not meet -128
under that screen. Its stochastic and numerical assumptions still need review;
dependencies inside a decision must not be confused with independence between
outer events.

## 7. Benchmark Readiness

The initial correctness code is in [adapters/bfv-style](adapters/bfv-style/README.md).
Stage times, key sizes, checks, and exact parameter values are recorded. Missing
work before adding an approved main-campaign entry:

- Validate alternative keys and the new multiplication/restoration noise model.
- Compare full convolution with bounded tiles and more efficient digit extraction.
- Optimize standard-domain packing and relinearization without invalidating
  the exact reference, and add multicore scheduling without overlapping jobs.
- Match input preparation assumptions and common output contract with the
  other baselines; compare full latency and stage latency separately.
- Use a new versioned server package with the command guide and vendored
  dependencies, preserving the current running package byte-for-byte.

Until then, correctness success, a kernel-only result, or a native multiplication
latency must not be reported as a failure-matched end-to-end performance result.
