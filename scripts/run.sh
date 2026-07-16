#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
bench="${root}/code/hybrid_bench"
revhomtrace="${root}/code/cbs_repos/RevHomTrace"

usage() {
  cat <<'USAGE'
Usage:
  ./scripts/run.sh build
  ./scripts/run.sh dry-run <target>
  ./scripts/run.sh smoke <target>
  ./scripts/run.sh bench <target>

Targets:
  all
  hybrid-4x4
  hybrid-8x8
  tfhe-rs
  shokri-tsoutsos

Controls:
  BENCH_WIDTHS             default: 16,32,64,128,256
  BENCH_THREADS            default: 1,2,4,8,16,32,64
  BENCH_REPS               override all method-specific repetition counts
  BENCH_REPS_HYBRID        default: 10
  BENCH_REPS_TFHE          default: 10
  BENCH_REPS_ST            default: 10
  BENCH_ST_PATTERNS        default: random
  BENCH_RUN_ID             default: run-<UTC timestamp>
  BENCH_OUTPUT_DIR         default: results/<BENCH_RUN_ID>
  BENCH_RESUME             default: 1
  BENCH_CONTINUE_ON_ERROR  default: 0
  BENCH_TIMEOUT_SEC        default: 0 (disabled)
USAGE
}

action="${1:-}"
target="${2:-}"

if [[ -z "${action}" || "${action}" == "-h" || "${action}" == "--help" ]]; then
  usage
  exit 0
fi

build_all() {
  (
    cd "${bench}"
    cargo build --release --locked \
      --bin hybrid_cxc \
      --bin tfhe_mul_baseline
  )
  (
    cd "${revhomtrace}"
    cargo build --release --locked --features multithread \
      --bin shokri_tsoutsos_cxc
  )
}

if [[ "${action}" == "build" ]]; then
  build_all
  exit 0
fi

case "${action}" in
  dry-run|smoke|bench) ;;
  *)
    usage >&2
    exit 2
    ;;
esac

case "${target}" in
  all|hybrid-4x4|hybrid-8x8|tfhe-rs|shokri-tsoutsos) ;;
  *)
    echo "unknown target: ${target:-<empty>}" >&2
    usage >&2
    exit 2
    ;;
esac

: "${BENCH_WIDTHS:=16,32,64,128,256}"
: "${BENCH_THREADS:=1,2,4,8,16,32,64}"
: "${BENCH_REPS_HYBRID:=${BENCH_REPS:-10}}"
: "${BENCH_REPS_TFHE:=${BENCH_REPS:-10}}"
: "${BENCH_REPS_ST:=${BENCH_REPS:-3}}"
: "${BENCH_ST_PATTERNS:=random}"
: "${BENCH_RUN_ID:=run-$(date -u +%Y%m%dT%H%M%SZ)}"
: "${BENCH_OUTPUT_DIR:=${root}/results/${BENCH_RUN_ID}}"
: "${BENCH_RESUME:=1}"
: "${BENCH_CONTINUE_ON_ERROR:=0}"
: "${BENCH_TIMEOUT_SEC:=0}"

if [[ ! "${BENCH_RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "BENCH_RUN_ID may contain only letters, digits, '.', '_', and '-'" >&2
  exit 2
fi

for value in "${BENCH_REPS_HYBRID}" "${BENCH_REPS_TFHE}" "${BENCH_REPS_ST}"; do
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "repetition counts must be positive integers" >&2
    exit 2
  fi
done

for value in "${BENCH_RESUME}" "${BENCH_CONTINUE_ON_ERROR}"; do
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "BENCH_RESUME and BENCH_CONTINUE_ON_ERROR must be 0 or 1" >&2
    exit 2
  fi
done

if [[ ! "${BENCH_TIMEOUT_SEC}" =~ ^[0-9]+$ ]]; then
  echo "BENCH_TIMEOUT_SEC must be zero or a positive integer" >&2
  exit 2
fi

widths="${BENCH_WIDTHS}"
threads="${BENCH_THREADS}"
hybrid_reps="${BENCH_REPS_HYBRID}"
tfhe_reps="${BENCH_REPS_TFHE}"
st_reps="${BENCH_REPS_ST}"
st_patterns="${BENCH_ST_PATTERNS}"
dry_run=0

if [[ "${action}" == "dry-run" ]]; then
  dry_run=1
  BENCH_OUTPUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/hybrid-cxc-dry-run.XXXXXX")"
  trap 'rm -rf "${BENCH_OUTPUT_DIR}"' EXIT
elif [[ "${action}" == "smoke" ]]; then
  widths="16"
  threads="1"
  hybrid_reps="1"
  tfhe_reps="1"
  st_reps="1"
  st_patterns="dense"
fi

if [[ "${BENCH_OUTPUT_DIR}" != /* ]]; then
  BENCH_OUTPUT_DIR="${PWD}/${BENCH_OUTPUT_DIR}"
fi
mkdir -p "${BENCH_OUTPUT_DIR}"
BENCH_OUTPUT_DIR="$(cd "${BENCH_OUTPUT_DIR}" && pwd)"

run_hybrid() {
  local label="$1"
  local runner_args=(
    "${label}"
    --widths "${widths}"
    --threads "${threads}"
    --repetitions "${hybrid_reps}"
    --output "${BENCH_OUTPUT_DIR}/${label}.csv"
    --failures "${BENCH_OUTPUT_DIR}/${label}.failures.tsv"
    --log-dir "${BENCH_OUTPUT_DIR}/logs/${label}"
    --stamp "${BENCH_RUN_ID}-${label}"
    --resume "${BENCH_RESUME}"
    --continue-on-error "${BENCH_CONTINUE_ON_ERROR}"
    --timeout-sec "${BENCH_TIMEOUT_SEC}"
  )
  if [[ "${dry_run}" == "1" ]]; then
    runner_args+=(--dry-run)
  fi
  (
    cd "${bench}"
    python3 scripts/run_hybrid_paper.py "${runner_args[@]}"
  )
}

run_tfhe_rs() {
  (
    cd "${bench}"
    BASELINE_WIDTH_LIST="${widths}" \
    BASELINE_THREAD_LIST="${threads}" \
    BASELINE_REPS="${tfhe_reps}" \
    BASELINE_WARMUPS="$([[ "${action}" == "smoke" ]] && echo 0 || echo 1)" \
    BASELINE_PARAMS="m2c2-gaussian" \
    BASELINE_STAMP="${BENCH_RUN_ID}-tfhe-rs" \
    BASELINE_OUT_DIR="${BENCH_OUTPUT_DIR}/tfhe-rs" \
    BASELINE_DRY_RUN="${dry_run}" \
    BASELINE_RESUME="${BENCH_RESUME}" \
    BASELINE_CONTINUE_ON_ERROR="${BENCH_CONTINUE_ON_ERROR}" \
    BASELINE_CASE_TIMEOUT_SEC="${BENCH_TIMEOUT_SEC}" \
      scripts/run_linux_tfhe_baseline_matrix.sh
  )
}

run_shokri_tsoutsos() {
  (
    cd "${bench}"
    ST_REPO="${revhomtrace}" \
    ST_WIDTHS="${widths}" \
    ST_THREADS="${threads}" \
    ST_REPS="${st_reps}" \
    ST_SEED="20260703" \
    ST_PATTERNS="${st_patterns}" \
    ST_OUTPUT_DIR="${BENCH_OUTPUT_DIR}/shokri-tsoutsos" \
    ST_DRY_RUN="${dry_run}" \
    ST_RESUME="${BENCH_RESUME}" \
    ST_TIMEOUT_SEC="${BENCH_TIMEOUT_SEC}" \
      scripts/run_shokri_tsoutsos_matrix.sh
  )
}

run_target() {
  case "$1" in
    hybrid-4x4) run_hybrid hybrid-4x4 ;;
    hybrid-8x8) run_hybrid hybrid-8x8 ;;
    tfhe-rs) run_tfhe_rs ;;
    shokri-tsoutsos) run_shokri_tsoutsos ;;
  esac
}

if [[ "${target}" == "all" ]]; then
  for item in tfhe-rs hybrid-4x4 hybrid-8x8 shokri-tsoutsos; do
    run_target "${item}"
  done
else
  run_target "${target}"
fi

if [[ "${dry_run}" == "0" ]]; then
  echo "results: ${BENCH_OUTPUT_DIR}"
fi
