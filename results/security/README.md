# Security estimates (Section 6.2)

lattice-estimator commit 6019056011d10d7e9c30a0d5da2d2f729fbc2eec, `LWE.estimate` with
all attacks, binary secrets, discrete Gaussian errors, q = 2^64. GLWE keys are
estimated as LWE instances of dimension kN. Each line of `keys.txt` is
`n log2(sigma/q) tag`. Set `LATTICE_ESTIMATOR_PATH` in `est.sage.py`, then run

    sage est.sage.py <n> <log2sigma> <tag>

The `RESULT` line of each `.out` file is the minimum over all attacks.
