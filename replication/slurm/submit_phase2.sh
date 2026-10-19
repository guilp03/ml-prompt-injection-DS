#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

: "${SCRATCH:?defina SCRATCH (ex.: /scratch/usuario)}"
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/replication/runs}"
PARTITION_LIGHT="${PARTITION_LIGHT:-short-simple}"
PARTITION_TX="${PARTITION_TX:-short-complex}"
WALLTIME_LIGHT="${WALLTIME_LIGHT:-02:00:00}"
WALLTIME_TX="${WALLTIME_TX:-04:00:00}"
GRES_TX="${GRES_TX:-gpu:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-32G}"

if [[ "${RUNS_ROOT}" == *,* || "${RUNS_ROOT}" == *$'\n'* ]]; then
  echo "[error] RUNS_ROOT nao pode conter virgula ou nova linha" >&2
  exit 2
fi

SPLITS=("$@")
if [[ ${#SPLITS[@]} -eq 0 ]]; then
  SPLITS=(A B C)
fi

VALIDATED=()
for split in "${SPLITS[@]}"; do
  split="${split^^}"
  case "${split}" in
    A|B|C) ;;
    *) echo "[error] split invalido: ${split}" >&2; exit 2 ;;
  esac
  VALIDATED+=("${split}")
done
SPLITS=("${VALIDATED[@]}")

mkdir -p "${RUNS_ROOT}/logs"

for split in "${SPLITS[@]}"; do
  light_id="$(sbatch --parsable \
    --job-name="pids-light-${split}" \
    --partition="${PARTITION_LIGHT}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEM}" \
    --time="${WALLTIME_LIGHT}" \
    --export="ALL,SPLIT_TAG=${split},RUNS_ROOT=${RUNS_ROOT}" \
    --output="${RUNS_ROOT}/logs/%x-%j.out" \
    "${SCRIPT_DIR}/light.sh")"

  echo "[submit] split ${split}: light=${light_id}"

  if ! tx_id="$(sbatch --parsable \
    --job-name="pids-tx-${split}" \
    --partition="${PARTITION_TX}" \
    --gres="${GRES_TX}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEM}" \
    --time="${WALLTIME_TX}" \
    --export="ALL,SPLIT_TAG=${split},RUNS_ROOT=${RUNS_ROOT}" \
    --output="${RUNS_ROOT}/logs/%x-%j.out" \
    "${SCRIPT_DIR}/transformers.sh")"; then
    echo "[error] falha ao submeter tx do split ${split}; light=${light_id} ja foi submetido" >&2
    exit 1
  fi
  echo "[submit] split ${split}: tx=${tx_id}"
done

echo "[info] acompanhe com: squeue -u ${USER:-$(id -un)}"
echo "[info] agregue ao final: python ${REPO_ROOT}/replication/aggregate_phase2.py"
