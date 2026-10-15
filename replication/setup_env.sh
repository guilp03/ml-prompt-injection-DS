#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=pids-repro
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if ! conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
  conda env create -f environment.yml
fi

conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
# torch build CPU (evita baixar wheel CUDA ~2GB em máquina sem GPU)
conda run -n "${ENV_NAME}" python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu torch
conda run -n "${ENV_NAME}" python -m pip install \
  -r replication/requirements-local.txt

conda run -n "${ENV_NAME}" python -c \
  "import torch, transformers, sentence_transformers, sklearn, pandas, xgboost; print('env ok', torch.__version__)"
