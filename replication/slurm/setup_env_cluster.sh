#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# A documentacao da APUANA usa "module load Python3.10" no guia de uso e
# "python/3.8" em exemplos antigos. PYTHON_MODULE permite fixar o nome exato;
# sem ele, tenta os nomes comuns e o primeiro python3.x achado em `module avail`.
load_python_module() {
  if ! command -v module >/dev/null 2>&1; then
    echo "[warn] comando module ausente; usando ${PYTHON:-python3} do PATH" >&2
    return 0
  fi
  if [[ -n "${PYTHON_MODULE:-}" ]]; then
    if module load "${PYTHON_MODULE}" >/dev/null 2>&1; then
      echo "[env] module ${PYTHON_MODULE} carregado"
      return 0
    fi
    echo "[error] PYTHON_MODULE=${PYTHON_MODULE} nao pode ser carregado" >&2
    exit 2
  fi
  local mod discovered
  local candidates=(Python/3.10 Python3.10 python/3.10 python3.10)
  discovered="$(module avail 2>&1 | tr ' ()' '\n' \
    | grep -iE '^python3?/3\.[0-9]+' | sort -V | tail -n1 || true)"
  if [[ -n "${discovered}" ]]; then
    candidates+=("${discovered}")
  fi
  for mod in "${candidates[@]}"; do
    if module load "${mod}" >/dev/null 2>&1; then
      echo "[env] module ${mod} carregado"
      return 0
    fi
  done
  echo "[warn] nenhum modulo Python 3.x encontrado; usando ${PYTHON:-python3}" >&2
  echo "[hint] veja os modulos disponiveis: module avail 2>&1 | grep -i python" >&2
  return 0
}
load_python_module

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

PYTHON_BIN="${PYTHON:-python3}"
if ! "${PYTHON_BIN}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "[error] Python >= 3.10 necessario (${PYTHON_BIN})" >&2
  exit 2
fi

# torch CUDA + dependencias ocupam alguns GB; avisa antes de encher a area.
if command -v df >/dev/null 2>&1; then
  avail_kb="$(df -Pk "${SCRATCH}" | awk 'NR == 2 {print $4}')"
  if [[ "${avail_kb:-0}" =~ ^[0-9]+$ ]] && (( avail_kb < 15 * 1024 * 1024 )); then
    echo "[warn] menos de 15 GB livres em ${SCRATCH} (${avail_kb} KB)" >&2
  fi
fi

if [[ ! -f "${VENV}/bin/activate" ]]; then
  if ! "${PYTHON_BIN}" -m venv "${VENV}"; then
    echo "[error] falha ao criar venv em ${VENV}" >&2
    exit 2
  fi
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

if ! python -m pip --version >/dev/null 2>&1; then
  echo "[warn] pip ausente no venv; tentando ensurepip" >&2
  python -m ensurepip --upgrade >/dev/null 2>&1 || true
fi
if ! python -m pip --version >/dev/null 2>&1; then
  echo "[error] pip ausente no venv ${VENV}" >&2
  echo "[info] python do venv: $(python -c 'import sys; print(sys.version.split()[0], sys.executable)' 2>/dev/null || echo '?')" >&2
  echo "[hint] carregue um modulo Python com pip (module avail 2>&1 | grep -i python)," >&2
  echo "[hint] exporte PYTHON_MODULE=<nome>, remova o venv quebrado e rode de novo:" >&2
  echo "[hint]   rm -rf \"${VENV}\" && bash replication/slurm/setup_env_cluster.sh" >&2
  exit 2
fi

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
