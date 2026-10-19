#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

preflight_common
"${PY}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("[error] CUDA indisponivel neste job")
print(f"[gpu] torch={torch.__version__} device={torch.cuda.get_device_name(0)}")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

run_harness tx --notebooks 07,08
