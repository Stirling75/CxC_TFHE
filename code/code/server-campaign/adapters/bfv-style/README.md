# BFV adaptation

Our implementation of the BFV-style multiplication of Chillotti et al.
(ASIACRYPT 2021) with TFHE-rs radix inputs and outputs: the input digits are
packed into one GLWE ciphertext, all local products are computed by one tensor
product with relinearization, and the result is converted back to radix digits
by digit extraction and carry restoration. Packing and conversion are timed.

- `parameters/bfv-n800-range-aware.json`: the parameter set used in the paper.
- `verify_candidate.py`: Gaussian failure estimate (Section 5.4 of the paper).
- `analyze_noise.py`, `audit_run.py`: noise accounting and run checks.

The runner (`code/run.sh`, method `clot-bfv`) builds and runs it.
