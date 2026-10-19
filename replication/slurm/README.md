# Runbook — Fase 2 no cluster APUANA

## 1. Pré-requisitos

- Acesso SSH à APUANA com VPN do CIn ativa.
- `$SCRATCH` definido e com espaço para repo, venv, cache HF e runs.
- Repositório no cluster (sugestão: `$SCRATCH/ml_PIDS`).
- Os dados processados dos splits A/B/C já vêm commitados no repo.

## 2. Ambiente (uma vez, no nó de login)

```bash
cd "$SCRATCH/ml_PIDS"
export SCRATCH="$SCRATCH"
bash replication/slurm/setup_env_cluster.sh
source "$SCRATCH/pids-repro/bin/activate"
python -m pytest -q
```

O script cria `$SCRATCH/pids-repro` com torch CUDA e grava
`replication/runs/env-lock.txt`.

## 3. Cache de modelos (uma vez, no nó de login)

```bash
export HF_HOME="$SCRATCH/hf-cache"
source "$SCRATCH/pids-repro/bin/activate"
python - <<'PY'
from transformers import AutoTokenizer
for name in (
    "BAAI/bge-small-en-v1.5",
    "roberta-base",
    "microsoft/deberta-base",
):
    AutoTokenizer.from_pretrained(name)
    print("ok", name)
PY
```

Se os nós de compute não tiverem egress, exporte antes de submeter:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## 4. Piloto (split A)

```bash
bash replication/slurm/submit_phase2.sh A
squeue -u "$USER"
```

Quando os dois jobs de A terminarem:

```bash
python replication/aggregate_phase2.py
```

## 5. Execução completa

```bash
bash replication/slurm/submit_phase2.sh A B C
squeue -u "$USER"
```

## 6. Coleta

```bash
ls replication/runs/*/*/*/comparison.csv
cat replication/runs/env-lock.txt
python replication/aggregate_phase2.py
```

Copie para `$PROJECTS` ou para a máquina local:

- `replication/phase2/phase2_comparison_all.csv`
- `replication/phase2/phase2_summary.md`
- logs em `replication/runs/logs/`
- opcional: tarball das métricas por run

O `$SCRATCH` é limpo após 90 dias.

## 7. Retomada e diagnóstico

- Re-submeta somente o job afetado: `bash replication/slurm/submit_phase2.sh B`
  submete os dois jobs de B; para um só, use `sbatch` diretamente com os
  recursos do `submit_phase2.sh`.
- Logs: `replication/runs/logs/pids-<familia>-<split>-<jobid>.out`.
- Falha de GPU: confira `nvidia-smi` no log e `torch.cuda.is_available()`.
- Falha de egress/cache: repita o passo 3 e use os `*_OFFLINE`.
- Walltime: aumente `WALLTIME_TX` ao re-submeter.
- Convergência: `squeue -u "$USER"`, `sacct -u "$USER" --format=JobID,JobName,State,Elapsed`.
