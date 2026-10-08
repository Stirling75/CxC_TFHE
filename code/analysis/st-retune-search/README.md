# Finer search for the retuned Shokri-Tsoutsos set

`search.py BASE.json` keeps n, N, k and both noise levels of `BASE.json` and
searches all decompositions for the cheapest set whose estimate is below
2^-128 at W = 16 to 256. Each `base-n*.json` uses the LWE noise for about
128-bit security (`results/security/` for the estimator). The
`out-*.txt` files are the results: no set meets the target for n <= 688 at
130-bit noise, and n = 688 with sigma = 2^-15.05 (128.05 bits) gives the
selected set `st-cc2-r2-n688`, with the same decompositions as n = 704.
