# Hybrid CxC multiplication

This artifact contains the implementations used for the ciphertext-ciphertext
multiplication experiments:

- `hybrid-4x4`: hybrid multiplication with 4-by-4-bit chunk products
- `hybrid-8x8`: hybrid multiplication with 8-by-8-bit chunk products
- `tfhe-rs`: TFHE-rs radix multiplication
- `shokri-tsoutsos`: reconstruction of the CxC multiplier in ePrint 2026/810

The benchmark measures one encrypted multiplication modulo `2^W`. Key
generation, encryption, and decryption are excluded from the reported latency.

## Requirements

- Linux or macOS
- Rust 1.95.0
- Python 3.9 or later
- a C/C++ compiler and standard build tools

## Running the artifact

Build all binaries:

```bash
./scripts/run.sh build
```

Run one encrypted 16-bit multiplication for every implementation:

```bash
./scripts/run.sh smoke all
```

Run the complete experiment:

```bash
BENCH_RUN_ID=paper ./scripts/run.sh bench all
```

The default matrix uses widths `16,32,64,128,256` and thread counts
`1,2,4,8,16,32,64`. The experiments use ten repetitions. 

Each implementation can also be run separately:

```bash
./scripts/run.sh bench hybrid-4x4
./scripts/run.sh bench hybrid-8x8
./scripts/run.sh bench tfhe-rs
./scripts/run.sh bench shokri-tsoutsos
```

The matrix and repetition count can be reduced for a shorter run:

```bash
BENCH_WIDTHS=64,128 \
BENCH_THREADS=1,8 \
BENCH_REPS=3 \
BENCH_RUN_ID=short \
./scripts/run.sh bench all
```

Results are written to `results/<BENCH_RUN_ID>/`. Each CSV row records one
successful encrypted multiplication. Repeating the same command with the same `BENCH_RUN_ID` skips completed configurations.

The parameter sets used by the four implementations are listed in `config/parameters.json`.
