#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

: "${SPLIT_TAG:?defina SPLIT_TAG=A|B|C}"
SPLIT_TAG="${SPLIT_TAG^^}"
case "${SPLIT_TAG}" in
  A|B|C) ;;
  *) echo "[error] SPLIT_TAG invalido: ${SPLIT_TAG}" >&2; exit 2 ;;
esac

: "${SCRATCH:?defina SCRATCH (ex.: /scratch/usuario)}"
VENV="${VENV:-${SCRATCH}/pids-repro}"
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/replication/runs}"
export HF_HOME="${HF_HOME:-${SCRATCH}/hf-cache}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ ! -f "${VENV}/bin/activate" ]]; then
  echo "[error] venv nao encontrado: ${VENV}" >&2
  echo "[hint] rode replication/slurm/setup_env_cluster.sh" >&2
  exit 2
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
PY="${VENV}/bin/python"

processed_file_for_split() {
  case "${SPLIT_TAG}" in
    A) echo "jailbreak_benchmarks_processed_v2.csv" ;;
    B) echo "jailbreak_benchmarks_processed_v2_splitB.csv" ;;
    C) echo "jailbreak_benchmarks_processed_v2_splitC.csv" ;;
  esac
}

preflight_common() {
  local processed
  processed="data/processed/$(processed_file_for_split)"
  if [[ ! -f "${REPO_ROOT}/${processed}" ]]; then
    echo "[error] arquivo ausente: ${processed}" >&2
    exit 2
  fi
  local oracle="experiments/results/metrics/metrics_ablation_split${SPLIT_TAG}.csv"
  if [[ ! -f "${REPO_ROOT}/${oracle}" ]]; then
    echo "[error] oraculo ausente: ${oracle}" >&2
    exit 2
  fi
  if ! "${PY}" -c "import pandas, sklearn, numpy, nbconvert, torch, transformers, sentence_transformers" >/dev/null 2>&1; then
    echo "[error] dependencias Python ausentes no venv ${VENV}" >&2
    exit 2
  fi
  mkdir -p "${RUNS_ROOT}/${SPLIT_TAG}" "${RUNS_ROOT}/logs"
  echo "[env] split=${SPLIT_TAG} venv=${VENV} hf_home=${HF_HOME} runs=${RUNS_ROOT}"
}

run_compare() {
  local run_dir="$1"
  local out="${run_dir}/comparison.csv"
  rm -f "${out}"
  "${PY}" "${REPO_ROOT}/replication/compare_results.py" \
    "${run_dir}/experiments/results/metrics" \
    --oracle "${REPO_ROOT}/experiments/results/metrics" \
    --out "${out}" || return $?
  "${PY}" - "${out}" <<'PY' || return $?
import sys

import pandas as pd

path = sys.argv[1]
frame = pd.read_csv(path)
if frame.empty:
    raise SystemExit(f"[error] comparison.csv vazio: {path}")
print(f"[compare] {len(frame)} pares em {path}")
PY
}

run_harness() {
  local label="$1"
  shift
  if [[ ! "${label}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "[error] label invalido: ${label}" >&2
    exit 2
  fi
  local family_runs="${RUNS_ROOT}/${SPLIT_TAG}/${label}"
  local status=0
  local before after new_dirs run_dir
  mkdir -p "${family_runs}"
  before="$(find "${family_runs}" -mindepth 1 -maxdepth 1 -type d | sort)"

  set +e
  "${PY}" "${REPO_ROOT}/replication/run_local.py" "${SPLIT_TAG}" \
    --label "${label}" --runs-dir "${family_runs}" --skip-pytest "$@"
  status=$?
  set -e

  after="$(find "${family_runs}" -mindepth 1 -maxdepth 1 -type d | sort)"
  new_dirs="$(comm -13 <(printf '%s\n' "${before}") \
    <(printf '%s\n' "${after}") | sed '/^$/d')"
  run_dir="$(printf '%s\n' "${new_dirs}" | tail -n 1)"
  if [[ -z "${run_dir}" ]]; then
    echo "[error] nenhum run dir novo em ${family_runs}" >&2
    [[ "${status}" -ne 0 ]] || status=1
    exit "${status}"
  fi
  if ! run_compare "${run_dir}"; then
    if [[ "${status}" -eq 0 ]]; then
      status=1
    fi
  fi
  exit "${status}"
}
