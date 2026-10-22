#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -z "${SCRATCH:-}" ]]; then
  SCRATCH="${HOME:?HOME nao definido}"
  echo "[warn] SCRATCH vazio; usando HOME como base: ${SCRATCH}" >&2
fi
if [[ ! -d "${SCRATCH}" || ! -w "${SCRATCH}" ]]; then
  echo "[error] base nao gravavel: SCRATCH=${SCRATCH}" >&2
  echo "[hint] export SCRATCH=/caminho/gravavel (ex.: \$HOME ou \$PROJECTS)" >&2
  exit 2
fi
VENV="${VENV:-${SCRATCH}/pids-repro}"
HF_HOME="${HF_HOME:-${SCRATCH}/hf-cache}"
export SCRATCH VENV HF_HOME
if [[ -n "${PYTHON_MODULE:-}" ]]; then
  export PYTHON_MODULE
fi

RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/replication/runs}"
PARTITION_LIGHT="${PARTITION_LIGHT:-short-simple}"
PARTITION_TX="${PARTITION_TX:-short-complex}"
WALLTIME_LIGHT="${WALLTIME_LIGHT:-02:00:00}"
WALLTIME_TX="${WALLTIME_TX:-04:00:00}"
GRES_TX="${GRES_TX:-gpu:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-32G}"

for value in "${RUNS_ROOT}" "${SCRATCH}" "${VENV}" "${HF_HOME}"; do
  if [[ "${value}" == *,* || "${value}" == *$'\n'* ]]; then
    echo "[error] caminho com virgula ou nova linha nao pode ir no --export: ${value}" >&2
    exit 2
  fi
done

mem_to_gb() {
  local value="${1^^}"
  case "${value}" in
    *TB) echo $(( ${value%TB} * 1024 )) ;;
    *GB) echo $(( ${value%GB} )) ;;
    *MB) echo $(( ${value%MB} / 1024 )) ;;
    *T) echo $(( ${value%T} * 1024 )) ;;
    *G) echo $(( ${value%G} )) ;;
    *M) echo $(( ${value%M} / 1024 )) ;;
    *) echo 0 ;;
  esac
}

# Limites de QoS documentados na APUANA: simple=16 CPUs/64GB/2 GPUs,
# complex=48 CPUs/500GB/4 GPUs. Requisicoes acima disso ficam presas em PD.
check_qos() {
  local partition="$1" label="$2" max_cpus max_mem mem_gb
  case "${partition}" in
    *-simple) max_cpus=16; max_mem=64 ;;
    *-complex) max_cpus=48; max_mem=500 ;;
    *) return 0 ;;
  esac
  if (( CPUS_PER_TASK > max_cpus )); then
    echo "[error] ${label}: ${CPUS_PER_TASK} CPUs excede o limite de ${max_cpus} da particao ${partition}" >&2
    exit 2
  fi
  mem_gb="$(mem_to_gb "${MEM}")"
  if (( mem_gb > max_mem )); then
    echo "[error] ${label}: MEM=${MEM} excede o limite de ${max_mem}GB da particao ${partition}" >&2
    exit 2
  fi
}
check_qos "${PARTITION_LIGHT}" light
check_qos "${PARTITION_TX}" tx

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
    --export="ALL,SPLIT_TAG=${split},RUNS_ROOT=${RUNS_ROOT},SCRATCH=${SCRATCH},VENV=${VENV},HF_HOME=${HF_HOME}" \
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
    --export="ALL,SPLIT_TAG=${split},RUNS_ROOT=${RUNS_ROOT},SCRATCH=${SCRATCH},VENV=${VENV},HF_HOME=${HF_HOME}" \
    --output="${RUNS_ROOT}/logs/%x-%j.out" \
    "${SCRIPT_DIR}/transformers.sh")"; then
    echo "[error] falha ao submeter tx do split ${split}; light=${light_id} ja foi submetido" >&2
    exit 1
  fi
  echo "[submit] split ${split}: tx=${tx_id}"
done

echo "[info] acompanhe com: squeue -u ${USER:-$(id -un)}"
echo "[info] agregue ao final: python ${REPO_ROOT}/replication/aggregate_phase2.py"
