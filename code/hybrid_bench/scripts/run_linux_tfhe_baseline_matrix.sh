#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

make_widths() {
  local out=()
  local start="${BASELINE_WIDTH_START:-16}"
  local end="${BASELINE_WIDTH_END:-256}"
  local step="${BASELINE_WIDTH_STEP:-16}"
  for ((width = start; width <= end; width += step)); do
    out+=("${width}")
  done
  local IFS=,
  echo "${out[*]}"
}

message_bits_for_param() {
  case "$1" in
    m1c*) echo 1 ;;
    m2c2|m2c*) echo 2 ;;
    m3c*) echo 3 ;;
    m4c*) echo 4 ;;
    *)
      echo "cannot infer message bits from parameter alias: $1" >&2
      exit 2
      ;;
  esac
}

filter_widths_for_param() {
  local param="$1"
  local widths_csv="$2"
  local msg_bits
  msg_bits="$(message_bits_for_param "${param}")"

  local keep=()
  local width
  IFS=',' read -r -a widths <<< "${widths_csv}"
  for width in "${widths[@]}"; do
    width="${width//[[:space:]]/}"
    [[ -n "${width}" ]] || continue
    local blocks=$(( (width + msg_bits - 1) / msg_bits ))
    local capacity=$(( blocks * msg_bits ))
    if (( capacity <= 256 )); then
      keep+=("${width}")
    else
      echo "skip baseline param=${param} width=${width}: radix capacity ${capacity} exceeds U256 checker" >&2
    fi
  done

  local IFS=,
  echo "${keep[*]}"
}

: "${BASELINE_WIDTH_LIST:=16,32,64,128,256}"
: "${BASELINE_THREAD_LIST:=1,8}"
: "${BASELINE_REPS:=10}"
: "${BASELINE_WARMUPS:=1}"
: "${BASELINE_OPS:=mul-default}"
: "${BASELINE_SEED:=1413896261}"
: "${BASELINE_PARAMS:=m2c2-tuniform,m2c2-gaussian}"
: "${BASELINE_STAMP:=linux-tfhe-baseline-mul-reps${BASELINE_REPS}}"
: "${BASELINE_OUT_DIR:=results/${BASELINE_STAMP}}"
: "${BASELINE_DRY_RUN:=0}"
: "${BASELINE_RESUME:=1}"
: "${BASELINE_CONTINUE_ON_ERROR:=1}"
: "${BASELINE_CASE_TIMEOUT_SEC:=0}"
: "${BASELINE_LOG_DIR:=${BASELINE_OUT_DIR}/logs}"
: "${BASELINE_FAILURES_TSV:=${BASELINE_OUT_DIR}/failures.tsv}"
: "${BASELINE_USE_CARGO:=auto}"
: "${MALLOC_ARENA_MAX:=2}"
: "${OMP_NUM_THREADS:=1}"
: "${OPENBLAS_NUM_THREADS:=1}"
: "${MKL_NUM_THREADS:=1}"
: "${BLIS_NUM_THREADS:=1}"

mkdir -p "${BASELINE_OUT_DIR}" "${BASELINE_LOG_DIR}"

manifest="${BASELINE_OUT_DIR}/manifest.tsv"
printf "kind\tparam\tthreads\twidths\tops\treps\twarmups\tseed\tcsv\tlog\tfailures\n" > "${manifest}"
if [[ ! -s "${BASELINE_FAILURES_TSV}" ]]; then
  printf "status\tparam\twidth\tthreads\telapsed_sec\tcsv\tlog\n" > "${BASELINE_FAILURES_TSV}"
fi

echo "TFHE-rs baseline matrix"
echo "  params=${BASELINE_PARAMS}"
echo "  widths=${BASELINE_WIDTH_LIST}"
echo "  threads=${BASELINE_THREAD_LIST}"
echo "  ops=${BASELINE_OPS}"
echo "  reps=${BASELINE_REPS}"
echo "  warmups=${BASELINE_WARMUPS}"
echo "  seed=${BASELINE_SEED}"
echo "  out_dir=${BASELINE_OUT_DIR}"
echo "  dry_run=${BASELINE_DRY_RUN}"
echo "  resume=${BASELINE_RESUME}"
echo "  continue_on_error=${BASELINE_CONTINUE_ON_ERROR}"
echo "  case_timeout_sec=${BASELINE_CASE_TIMEOUT_SEC}"
echo "  log_dir=${BASELINE_LOG_DIR}"
echo "  failures=${BASELINE_FAILURES_TSV}"
echo "  runner=${BASELINE_USE_CARGO}"
echo "  env_hygiene=MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX},OMP_NUM_THREADS=${OMP_NUM_THREADS},OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS},MKL_NUM_THREADS=${MKL_NUM_THREADS},BLIS_NUM_THREADS=${BLIS_NUM_THREADS}"

IFS=',' read -r -a params <<< "${BASELINE_PARAMS}"
IFS=',' read -r -a threads_list <<< "${BASELINE_THREAD_LIST}"

baseline_complete() {
  local csv="$1"
  local param="$2"
  local width="$3"
  local threads="$4"

  [[ "${BASELINE_RESUME}" == "1" && -s "${csv}" ]] || return 1
  python3 - "${csv}" "${param}" "${width}" "${threads}" "${BASELINE_REPS}" "${BASELINE_SEED}" <<'PY'
import csv
import sys

path, param, width, threads, reps, seed = sys.argv[1:]
needed = int(reps)
try:
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
except FileNotFoundError:
    sys.exit(1)

seen_reps = set()
for row in rows:
    if (
        row.get("param") == param
        and row.get("width_bits") == width
        and row.get("rayon_threads") == threads
        and row.get("seed") == seed
        and row.get("ok", "").lower() == "true"
    ):
        seen_reps.add(row.get("rep", ""))
expected_reps = {str(rep) for rep in range(needed)}
sys.exit(0 if expected_reps.issubset(seen_reps) else 1)
PY
}

exit_if_interrupted() {
  local status="$1"
  local what="$2"
  case "${status}" in
    130|143)
      echo "${what} interrupted status=${status}; stopping the matrix" >&2
      exit "${status}"
      ;;
  esac
}

for param in "${params[@]}"; do
  param="${param//[[:space:]]/}"
  [[ -n "${param}" ]] || continue

  safe_widths="$(filter_widths_for_param "${param}" "${BASELINE_WIDTH_LIST}")"
  if [[ -z "${safe_widths}" ]]; then
    echo "skip baseline param=${param}: no width survives capacity filter" >&2
    continue
  fi
  IFS=',' read -r -a safe_width_list <<< "${safe_widths}"

  for threads in "${threads_list[@]}"; do
    threads="${threads//[[:space:]]/}"
    [[ -n "${threads}" ]] || continue

    for width in "${safe_width_list[@]}"; do
      width="${width//[[:space:]]/}"
      [[ -n "${width}" ]] || continue

      out="${BASELINE_OUT_DIR}/tfhe-baseline-${param}-w${width}-t${threads}.csv"
      log="${BASELINE_LOG_DIR}/tfhe-baseline-${param}-w${width}-t${threads}.log"
      printf "tfhe\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${param}" "${threads}" "${width}" "${BASELINE_OPS}" \
        "${BASELINE_REPS}" "${BASELINE_WARMUPS}" "${BASELINE_SEED}" "${out}" "${log}" "${BASELINE_FAILURES_TSV}" >> "${manifest}"

      if baseline_complete "${out}" "${param}" "${width}" "${threads}"; then
        echo "skip complete baseline param=${param} width=${width} threads=${threads}"
        continue
      fi

      echo "run baseline param=${param} threads=${threads} width=${width}"
      if [[ "${BASELINE_DRY_RUN}" != "1" ]]; then
        runner=(cargo run --release --bin tfhe_mul_baseline --)
        if [[ "${BASELINE_USE_CARGO}" == "0" || "${BASELINE_USE_CARGO}" == "direct" ]]; then
          runner=(target/release/tfhe_mul_baseline)
        elif [[ "${BASELINE_USE_CARGO}" == "auto" \
          && -x target/release/tfhe_mul_baseline \
          && target/release/tfhe_mul_baseline -nt src/bin/tfhe_mul_baseline.rs \
          && target/release/tfhe_mul_baseline -nt Cargo.toml \
          && target/release/tfhe_mul_baseline -nt Cargo.lock ]]; then
          runner=(target/release/tfhe_mul_baseline)
        fi

        case_command=(
          env
          MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX}"
          OMP_NUM_THREADS="${OMP_NUM_THREADS}"
          OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}"
          MKL_NUM_THREADS="${MKL_NUM_THREADS}"
          BLIS_NUM_THREADS="${BLIS_NUM_THREADS}"
          "${runner[@]}"
          --params "${param}"
          --widths "${width}"
          --ops "${BASELINE_OPS}"
          --reps "${BASELINE_REPS}"
          --warmups "${BASELINE_WARMUPS}"
          --seed "${BASELINE_SEED}"
          --threads "${threads}"
          --out "${out}"
        )
        if [[ "${BASELINE_CASE_TIMEOUT_SEC}" != "0" ]]; then
          case_command=(
            python3 scripts/run_with_timeout.py
            "${BASELINE_CASE_TIMEOUT_SEC}" --
            "${case_command[@]}"
          )
        fi

        start_sec="$(date +%s)"
        status=0
        "${case_command[@]}" >"${log}" 2>&1 || status=$?
        elapsed_sec="$(( $(date +%s) - start_sec ))"
        exit_if_interrupted "${status}" "baseline"
        if [[ "${status}" != "0" ]]; then
          printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "${status}" "${param}" "${width}" "${threads}" "${elapsed_sec}" "${out}" "${log}" >> "${BASELINE_FAILURES_TSV}"
          echo "baseline failed status=${status} param=${param} width=${width} threads=${threads} elapsed=${elapsed_sec}s log=${log}" >&2
          tail -40 "${log}" >&2 || true
          if [[ "${BASELINE_CONTINUE_ON_ERROR}" != "1" ]]; then
            exit "${status}"
          fi
        else
          echo "baseline ok param=${param} width=${width} threads=${threads} elapsed=${elapsed_sec}s log=${log}"
        fi
      fi
    done
  done
done

echo "done baseline matrix"
echo "manifest=${manifest}"
