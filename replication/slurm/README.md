# Runbook — Fase 2 no cluster APUANA

## 1. Pré-requisitos

- Acesso SSH à APUANA com VPN do CIn ativa.
- `$SCRATCH` definido e com espaço para repo, venv, cache HF e runs.
- Repositório no cluster (sugestão: `$SCRATCH/ml_PIDS`).
- Os dados processados dos splits A/B/C já vêm commitados no repo.

## 2. Ambiente (uma vez, no nó de login)

```bash
export SCRATCH="${SCRATCH:?defina SCRATCH}"
cd "$SCRATCH/ml_PIDS"
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
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SentenceTransformer("BAAI/bge-small-en-v1.5")
for name in ("roberta-base", "microsoft/deberta-base"):
    AutoTokenizer.from_pretrained(name)
    AutoModelForSequenceClassification.from_pretrained(name)
print("cache ok")
PY
```

Isso baixa os pesos do sentence-transformer BGE e os checkpoints dos dois
backbones, de modo que jobs com `*_OFFLINE` tenham tudo o que precisam.

Se os nós de compute não tiverem egress, exporte antes de submeter:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Essas variáveis são repassadas aos jobs via `--export=ALL`; exporte-as no shell
de onde o `sbatch` é chamado.

## 4. Piloto (split A)

```bash
bash replication/slurm/submit_phase2.sh A
squeue -u "$USER"
```

Quando os dois jobs de A terminarem:

```bash
source "$SCRATCH/pids-repro/bin/activate"
python replication/aggregate_phase2.py
```

## 5. Execução completa

```bash
bash replication/slurm/submit_phase2.sh B C
squeue -u "$USER"
```

O split A já rodou no piloto; reenviá-lo criaria runs duplicados.

### Recursos configuráveis

```bash
PARTITION_LIGHT=short-simple PARTITION_TX=short-complex \
WALLTIME_LIGHT=02:00:00 WALLTIME_TX=04:00:00 \
GRES_TX=gpu:1 CPUS_PER_TASK=8 MEM=32G \
  bash replication/slurm/submit_phase2.sh B C
```

Valide os recursos antes do piloto com `sbatch --test-only` e as mesmas flags
do exemplo de retomada da seção 7.

## 6. Coleta

```bash
ls replication/runs/*/*/*/comparison.csv
cat replication/runs/env-lock.txt
source "$SCRATCH/pids-repro/bin/activate"
python replication/aggregate_phase2.py
```

Copie para `$PROJECTS` (se disponível) ou para a máquina local:

- `replication/phase2/phase2_comparison_all.csv`
- `replication/phase2/phase2_summary.md`
- `replication/runs/env-lock.txt`
- logs em `replication/runs/logs/`
- opcional: tarball das métricas por run

O `$SCRATCH` é limpo após 90 dias.

## 7. Retomada e diagnóstico

- Re-submeta somente o job afetado. Exemplo de um único job `tx` do split B,
  a partir da raiz do repo:

  ```bash
  export SCRATCH="${SCRATCH:?defina SCRATCH}"
  export HF_HOME="${HF_HOME:-$SCRATCH/hf-cache}"
  export SPLIT_TAG=B
  sbatch --job-name="pids-tx-B" \
    --partition=short-complex --gres=gpu:1 \
    --cpus-per-task=8 --mem=32G --time=04:00:00 \
    --export=ALL,SPLIT_TAG=B,RUNS_ROOT="$PWD/replication/runs" \
    --output="$PWD/replication/runs/logs/%x-%j.out" \
    replication/slurm/transformers.sh
  ```

  O mesmo padrão vale para `light.sh` (sem `--gres`); ajuste partição,
  walltime e GRES conforme necessário.
- Logs: `replication/runs/logs/pids-<familia>-<split>-<jobid>.out`.
- Falha de GPU: procure a linha `[gpu] ...` no log e, se disponível, rode
  `nvidia-smi` no nó; confira `torch.cuda.is_available()`.
- Falha de egress/cache: repita o passo 3 e use os `*_OFFLINE`.
- Walltime: aumente `WALLTIME_TX` ao re-submeter.
- Convergência: `squeue -u "$USER"`, `sacct -u "$USER" --format=JobID,JobName,State,Elapsed`.
