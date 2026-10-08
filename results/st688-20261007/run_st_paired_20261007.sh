#!/bin/bash
# Paired check of st-r2-n704 and st-r2-n688 at high thread counts in one quiet session.
set -u
cd "$(dirname "$0")"
stage() { echo "=== $(date "+%F %T") $1 (load $(cut -d' ' -f1-3 /proc/loadavg))"; }
quiet() { local ok=0; while [ $ok -lt 5 ]; do if awk '{exit !($1 < 4)}' /proc/loadavg; then ok=$((ok+1)); else ok=0; fi; sleep 60; done; }
for W in 128 256; do
  for M in st-r2-n704 st-r2-n688; do
    quiet; stage "$M W=$W"
    sh run.sh bench --research --methods $M --widths $W --threads 16 32 64        --repetitions 5 --warmup 1 --output results/paired-$M-w$W-20261007
  done
done
stage done
