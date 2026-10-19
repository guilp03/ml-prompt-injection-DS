#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

: "${SCRATCH:?defina SCRATCH (ex.: /scratch/usuario)}"
VENV="${VENV:-${SCRATCH}/pids-repro}"

if command -v module >/dev/null 2>&1; then
  if ! module load Python/3.10; then
    echo "[warn] falha ao carregar Python/3.10; usando ${PYTHON:-python3} do PATH" >&2
  fi
else
  echo "[warn] comando module ausente; usando ${PYTHON:-python3} do PATH" >&2
fi

PYTHON_BIN="${PYTHON:-python3}"
if ! "${PYTHON_BIN}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "[error] Python >= 3.10 necessario (${PYTHON_BIN})" >&2
  exit 2
fi

if [[ ! -f "${VENV}/bin/activate" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV}"
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
python -m pip freeze > "${REPO_ROOT}/replication/runs/env-lock.txt.tmp"
mv "${REPO_ROOT}/replication/runs/env-lock.txt.tmp" \
  "${REPO_ROOT}/replication/runs/env-lock.txt"
echo "[ok] ambiente em ${VENV}"
echo "[ok] lock em ${REPO_ROOT}/replication/runs/env-lock.txt"
echo "[next] export HF_HOME=${SCRATCH}/hf-cache"
