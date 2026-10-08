#!/bin/bash
# st-r2-n688 at 16-64 threads for W=16..64 in a quiet session (W=128/256 in run_st_paired_20261007.sh).
set -u
cd "$(dirname "$0")"
stage() { echo "=== $(date "+%F %T") $1 (load $(cut -d' ' -f1-3 /proc/loadavg))"; }
quiet() { local ok=0; while [ $ok -lt 5 ]; do if awk '{exit !($1 < 4)}' /proc/loadavg; then ok=$((ok+1)); else ok=0; fi; sleep 60; done; }
for W in 16 32 64; do
  quiet; stage "st-r2-n688 W=$W"
  sh run.sh bench --research --methods st-r2-n688 --widths $W --threads 16 32 64      --repetitions 5 --warmup 1 --output results/paired-st-r2-n688-w$W-20261007
done
stage done
