#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

: "${SCRATCH:?defina SCRATCH (ex.: /scratch/usuario)}"
VENV="${VENV:-${SCRATCH}/pids-repro}"

if command -v module >/dev/null 2>&1; then
  module load Python/3.10 || true
fi

if [[ ! -d "${VENV}" ]]; then
  "${PYTHON:-python3}" -m venv "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "${REPO_ROOT}/requirements.txt"

python - <<'PY'
import torch

print(f"[torch] {torch.__version__} cuda_available={torch.cuda.is_available()}")
PY

mkdir -p "${REPO_ROOT}/replication/runs"
python -m pip freeze > "${REPO_ROOT}/replication/runs/env-lock.txt"
echo "[ok] ambiente em ${VENV}"
echo "[ok] lock em ${REPO_ROOT}/replication/runs/env-lock.txt"
echo "[next] export HF_HOME=${SCRATCH}/hf-cache"
