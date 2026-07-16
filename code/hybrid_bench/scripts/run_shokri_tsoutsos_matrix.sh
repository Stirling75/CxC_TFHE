#!/usr/bin/env bash
set -euo pipefail

bench_root="$(cd "$(dirname "$0")/.." && pwd)"
: "${ST_REPO:=${bench_root}/../cbs_repos/RevHomTrace}"
: "${ST_WIDTHS:=16,32,64,128,256}"
: "${ST_THREADS:=1,2,4,8,16,32,64}"
: "${ST_REPS:=3}"
: "${ST_PATTERNS:=random}"
: "${ST_SEED:=20260703}"
: "${ST_OUTPUT_DIR:=${bench_root}/results/shokri-tsoutsos}"
: "${ST_RESUME:=1}"
: "${ST_DRY_RUN:=0}"
: "${ST_TIMEOUT_SEC:=0}"
: "${CARGO_BUILD_JOBS:=1}"
: "${CARGO_INCREMENTAL:=0}"

if [[ ! "${ST_REPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ST_REPS must be a positive integer" >&2
  exit 2
fi
if [[ ! "${ST_TIMEOUT_SEC}" =~ ^[0-9]+$ ]]; then
  echo "ST_TIMEOUT_SEC must be a non-negative integer" >&2
  exit 2
fi

runner="${bench_root}/scripts/run_shokri_tsoutsos.py"
repo="$(cd "${ST_REPO}" && pwd)"

echo "Shokri--Tsoutsos CxC benchmark"
echo "  parameter=SHOKRI_TSOUTSOS_TABLE1"
echo "  widths=${ST_WIDTHS}"
echo "  threads=${ST_THREADS}"
echo "  patterns=${ST_PATTERNS}"
echo "  repetitions=${ST_REPS}"
echo "  output=${ST_OUTPUT_DIR}"

if [[ "${ST_DRY_RUN}" == "0" ]]; then
  mkdir -p "${ST_OUTPUT_DIR}"
  printf 'threads\tpath\n' > "${ST_OUTPUT_DIR}/manifest.tsv"
fi

IFS=',' read -r -a thread_values <<< "${ST_THREADS}"
for threads in "${thread_values[@]}"; do
  threads="${threads//[[:space:]]/}"
  if [[ ! "${threads}" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid thread count: ${threads}" >&2
    exit 2
  fi

  output="${ST_OUTPUT_DIR}/shokri-tsoutsos-cxc-t${threads}.csv"
  command=(
    python3 "${runner}"
    --repo "${repo}"
    --widths "${ST_WIDTHS}"
    --patterns "${ST_PATTERNS}"
    --trials "${ST_REPS}"
    --seed "${ST_SEED}"
    --timeout "${ST_TIMEOUT_SEC}"
    --out "${output}"
  )
  [[ "${ST_RESUME}" == "1" ]] && command+=(--resume)
  [[ "${ST_DRY_RUN}" == "1" ]] && command+=(--dry-run)

  echo "run threads=${threads}"
  env \
    ST_PARALLEL=1 \
    RAYON_NUM_THREADS="${threads}" \
    CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS}" \
    CARGO_INCREMENTAL="${CARGO_INCREMENTAL}" \
    "${command[@]}"

  if [[ "${ST_DRY_RUN}" == "0" ]]; then
    printf '%s\t%s\n' "${threads}" "${output}" >> "${ST_OUTPUT_DIR}/manifest.tsv"
  fi
done

if [[ "${ST_DRY_RUN}" == "0" ]]; then
  echo "manifest: ${ST_OUTPUT_DIR}/manifest.tsv"
fi
