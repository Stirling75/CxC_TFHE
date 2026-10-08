#!/bin/bash
# Revision: tightest retuned Shokri-Tsoutsos set (n=688), same protocol as run_stats_20261004b.sh.
set -u
cd "$(dirname "$0")"
stage() { echo "=== $(date "+%F %T") $1 (load $(cut -d' ' -f1-3 /proc/loadavg))"; }
quiet() { local ok=0; while [ $ok -lt 5 ]; do if awk '{exit !($1 < 4)}' /proc/loadavg; then ok=$((ok+1)); else ok=0; fi; sleep 60; done; }
TS=(1 2 4 8 16 32 64)
quiet; stage build
sh run.sh build --methods st-r2-n688 --jobs 32 || exit 1
stage smoke
sh run.sh smoke --research --methods st-r2-n688 --output results/smoke-st688-20261007 || exit 1
for W in 16 32 64 128; do
  quiet; stage "st688 W=$W"
  sh run.sh bench --research --methods st-r2-n688 --widths $W --threads "${TS[@]}"      --repetitions 5 --warmup 1 --output results/stats-st688-w$W-20261007
done
quiet; stage "st688 W=256"
sh run.sh bench --research --methods st-r2-n688 --widths 256 --threads "${TS[@]}"    --repetitions 3 --warmup 1 --output results/stats-st688-w256-20261007
stage done
