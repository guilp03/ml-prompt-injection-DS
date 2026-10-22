# Runbook — Fase 2 no cluster APUANA

Baseado na documentação oficial em <https://apuana.cin.ufpe.br/usage>
(consultada em 2026-10-22).

## 1. Pré-requisitos

- Login `cin.ufpe.br` com acesso habilitado e **VPN do CIn** ativa.
- Acesso SSH a `slurm-client1.cin.ufpe.br` (os nós `slurm-client1/2` são os
  nós de login; não execute notebooks neles).
- Chave SSH cadastrada no GitHub para clonar o repositório no cluster (ou use
  HTTPS com token). A documentação do CIn mostra `ssh-keygen -t rsa`; uma
  chave `ed25519` também funciona:
  `ssh-keygen -t ed25519 -C "glp3@apuana"` e cadastre a `.pub` no GitHub.

## 2. Descobrir as áreas de armazenamento

A documentação lista `$SCRATCH` como área de resultados temporários, mas a
variável pode não estar exportada no shell. **Não clone nada para `/ml_PIDS`**:
se `$SCRATCH` estiver vazio, a expansão vira a raiz do sistema.

```bash
echo "HOME=$HOME"
echo "SCRATCH=${SCRATCH-<VAZIO>}"
echo "PROJECTS=${PROJECTS-<VAZIO>}"
echo "ARCHIVE=${ARCHIVE-<VAZIO>}"
df -h "$HOME" ${SCRATCH:+"$SCRATCH"}
beegfs-ctl --cfgFile=/etc/beegfs/home.d/beegfs-client.conf \
  --getquota --uid "$USER"          # quota do $HOME
```

Se `$SCRATCH` estiver vazio, use `$HOME` como base (os scripts também fazem
esse fallback, com aviso):

```bash
export SCRATCH="$HOME"
```

| Área | Performance | Uso | Quota | Backup | Limpeza |
|---|---|---|---|---|---|
| `/network/datasets/` | alta | datasets brutos (leitura) | — | — | — |
| `$HOME` | baixa | código, libs, ambiente | 100 GB / 1000K arquivos | diário | não |
| `$SCRATCH` | alta | resultados temporários | sem quota | não | 90 dias sem uso |
| `$SLURM_TMPDIR` | altíssima | disco local do job | 4 TB | não | fim do job |
| `/network/projects/` | média | espaço compartilhado | 200 GB / 1000K | diário | não |
| `$ARCHIVE` | baixa | resultados de longo prazo | 500 GB | não | não |

`$ARCHIVE` só existe nos nós de login.

## 3. Transferir o repositório

```bash
mkdir -p "$HOME/Projects"
cd "$HOME/Projects"
git clone git@github.com:guilp03/ml-prompt-injection-DS.git ml_PIDS
cd ml_PIDS
git checkout repro-local-phase1
git log --oneline -1        # esperado: 6dcb352
```

Alternativa sem git no cluster (a partir da máquina local):

```bash
rsync --bwlimit=1000 -azP --exclude '.git' --exclude 'replication/runs' \
  /home/gl-pereira/Projects/ml_PIDS/ \
  glp3@slurm-client1.cin.ufpe.br:~/Projects/ml_PIDS/
```

## 4. Ambiente (uma vez, no nó de login)

A APUANA recomenda venv em `$HOME`; nosso script usa `$SCRATCH/pids-repro`
por default (ou `$HOME/pids-repro` se `SCRATCH` estiver vazio).

```bash
export SCRATCH="${SCRATCH:-$HOME}"       # base para venv e cache HF
export VENV="${VENV:-$SCRATCH/pids-repro}"
export HF_HOME="${HF_HOME:-$SCRATCH/hf-cache}"
cd "$HOME/Projects/ml_PIDS"

# o setup tenta Python/3.10, Python3.10 e python/3.10
bash replication/slurm/setup_env_cluster.sh
source "$VENV/bin/activate"
python -m pytest -q
```

O script grava `replication/runs/env-lock.txt`. Se quiser conferir os módulos
disponíveis:

```bash
module avail 2>&1 | grep -i python
```

Se o nome do módulo não for um dos que o script tenta, fixe-o explicitamente:

```bash
export PYTHON_MODULE=Python/3.10.13   # use o nome exato do `module avail`
rm -rf "$VENV"                       # remove venv criado sem pip
bash replication/slurm/setup_env_cluster.sh
```

O mesmo `PYTHON_MODULE` é repassado aos jobs pelo `submit_phase2.sh`. Se o
`python3` do PATH não tiver `ensurepip`, o setup tenta `python -m ensurepip`;
persistindo a falha, ele mostra o Python usado e pede para recriar o venv com
um módulo adequado.

## 5. Cache de modelos (uma vez, no nó de login)

```bash
export HF_HOME="${HF_HOME:-$SCRATCH/hf-cache}"
source "$VENV/bin/activate"
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

Se os nós de compute não tiverem egress, exporte antes de submeter (o
`--export=ALL` repassa para os jobs):

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Se você já tem cache em `~/hf` (ou `~/models`), aponte o `HF_HOME` para ele
antes de rodar o setup e a submissão; caso contrário, o default é
`$SCRATCH/hf-cache`.

## 6. Recursos e partições

| Partição | Interativa | Recursos | Tempo máx. | QoS (máx.) |
|---|---|---|---|---|
| `emergency` | não | node completo | 1 dia | — |
| `install` | sim | meio node | 30 min | — |
| `debug` | sim | node completo | 30 min | — |
| `short-simple` | não | meio node | 2 dias | 16 CPUs / 64 GB / 2 GPUs |
| `short-complex` | não | node completo | 2 dias | 48 CPUs / 500 GB / 4 GPUs |
| `long-simple` | não | meio node | 7 dias | 16 CPUs / 64 GB / 2 GPUs |
| `long-complex` | não | node completo | 7 dias | 48 CPUs / 500 GB / 4 GPUs |

Defaults do nosso `submit_phase2.sh`: `short-simple` (light), `short-complex`
(tx), 8 CPUs, 32 GB, 1 GPU — dentro dos limites. O script recusa CPUs/RAM acima
do QoS da partição para não deixar o job preso em `PD (QOSMax)`.

Teste interativo curto (somente em `install`/`debug`):

```bash
salloc --partition=install --gres=gpu:1 -c 8 --mem=16G --time=00:20:00
module load Python3.10
source "$VENV/bin/activate"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
exit
```

## 7. Piloto (split A)

Valide os recursos antes de submeter:

```bash
sbatch --test-only \
  --job-name=pids-tx-A --partition=short-complex --gres=gpu:1 \
  --cpus-per-task=8 --mem=32G --time=04:00:00 \
  --export=ALL,SPLIT_TAG=A replication/slurm/transformers.sh
```

Depois submeta os dois jobs do split A:

```bash
cd "$HOME/Projects/ml_PIDS"
bash replication/slurm/submit_phase2.sh A
squeue -u "$USER"
```

Quando os dois jobs terminarem:

```bash
source "$VENV/bin/activate"
python replication/aggregate_phase2.py
```

## 8. Execução completa

```bash
bash replication/slurm/submit_phase2.sh B C
squeue -u "$USER"
```

O split A já rodou no piloto; reenviá-lo criaria runs duplicados (o agregador
mantém o último, mas o custo é desnecessário).

Recursos configuráveis:

```bash
PARTITION_LIGHT=short-simple PARTITION_TX=short-complex \
WALLTIME_LIGHT=02:00:00 WALLTIME_TX=04:00:00 \
GRES_TX=gpu:1 CPUS_PER_TASK=8 MEM=32G \
  bash replication/slurm/submit_phase2.sh B C
```

## 9. Coleta

```bash
source "$VENV/bin/activate"
python replication/aggregate_phase2.py
ls replication/runs/*/*/*/comparison.csv
cat replication/runs/env-lock.txt
cat replication/phase2/phase2_summary.md
```

Guarde no `$ARCHIVE` (nó de login) antes da limpeza de 90 dias do `$SCRATCH`:

```bash
tar cJf "$ARCHIVE/pids-fase2-$(date +%Y%m%d).tar.xz" --xattrs \
  replication/phase2 replication/report-phase2.md replication/runs/logs \
  replication/runs/env-lock.txt
```

Copie para a máquina local (se preferir):

```bash
rsync --bwlimit=1000 -azP \
  glp3@slurm-client1.cin.ufpe.br:~/Projects/ml_PIDS/replication/phase2 \
  ~/Projects/ml_PIDS/replication/
```

## 10. Retomada e diagnóstico

- Re-submeta somente o job afetado. Exemplo de um único job `tx` do split B:

  ```bash
  export SCRATCH="${SCRATCH:-$HOME}"
  export VENV="${VENV:-$SCRATCH/pids-repro}"
  export HF_HOME="${HF_HOME:-$SCRATCH/hf-cache}"
  export SPLIT_TAG=B
  sbatch --job-name="pids-tx-B" \
    --partition=short-complex --gres=gpu:1 \
    --cpus-per-task=8 --mem=32G --time=04:00:00 \
    --export=ALL,SPLIT_TAG=B,RUNS_ROOT="$PWD/replication/runs",\
SCRATCH="$SCRATCH",VENV="$VENV",HF_HOME="$HF_HOME" \
    --output="$PWD/replication/runs/logs/%x-%j.out" \
    replication/slurm/transformers.sh
  ```

  O mesmo padrão vale para `light.sh` (sem `--gres`).
- Logs: `replication/runs/logs/pids-<familia>-<split>-<jobid>.out`.
- GPU: procure `[gpu] ...` no log; `nvidia-smi` dentro do job.
- `torch.cuda.is_available()` falso: confira a versão do driver com
  `nvidia-smi` e, se o wheel CUDA do `requirements.txt` for mais novo que o
  driver, reinstale um build compatível no venv, por exemplo:
  `pip install --index-url https://download.pytorch.org/whl/cu124 torch`.
- Egress/cache: repita o passo 5 e use `HF_HUB_OFFLINE=1` /
  `TRANSFORMERS_OFFLINE=1`.
- Walltime: aumente `WALLTIME_TX` ao re-submeter (máx. 2 dias em `short-*`).
- Estado dos jobs: `squeue -u "$USER"` e
  `sacct -u "$USER" --format=JobID,JobName,State,Elapsed,MaxRSS`.
- Job preso em `PD`: veja o motivo em `squeue -j <id> -o %r` (`QOSMax`,
  `Resources`, `Priority`, ...).
