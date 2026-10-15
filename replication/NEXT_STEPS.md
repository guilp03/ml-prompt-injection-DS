# Próximos passos — da Fase 1 (local) para a Fase 2 (cluster)

Documento de planejamento dos passos seguintes à reprodução local mínima
(Fase 1, branch `repro-local-phase1`). Referências: spec em
`docs/superpowers/specs/2026-10-15-replicacao-local-ai-jailbreak-framework-design.md`
e plano em `docs/superpowers/plans/2026-10-15-replicacao-local-ai-jailbreak-framework.md`.

## 0. Onde estamos

Concluído e verificado (Fase 1, split A, famílias leves):

- Repo de referência vendorizado; tooling de reprodução em `replication/`.
- Harness `run_local.py` (isolamento por run, `pytest` + notebooks 04→06) e
  comparador `compare_results.py` (diff contra o oráculo) com 21 testes.
- Resultado: conclusões qualitativas centrais reproduzidas; `auc_pr`/`roc_auc`
  100% dentro da tolerância; macro-F1 46/51; divergências no track
  `deployment_threshold` e em métricas de baixo FPR. Relatório em
  `replication/report.md`.

**Lacunas que a Fase 2 precisa fechar:**
1. Transformers fine-tuned (RoBERTa-base, DeBERTa-base) e híbrido
   RoBERTa+IBVS — notebooks `07` e `08` (a conclusão "RoBERTa+IBVS melhora
   low-FPR" não foi verificada localmente).
2. Splits `B` e `C` (repetibilidade) para as famílias 04→06.
3. Divergência sistemática observada no track `deployment_threshold`
   (macro-F1 +0.09–0.12 vs. oráculo) — investigar/calibrar (opcional).

---

## 1. Fase 2 — adaptação para a APUANA

### 1.1 Fatos da APUANA (CIn/UFPE) relevantes

- **Acesso:** login `cin.ufpe.br` habilitado + **VPN do CIn**.
- **Gerência de jobs:** SLURM (`sbatch`, `srun`, `salloc`, `squeue`).
- **Partições:** `emergency`, `install`/`debug` (interativas), `short-simple`,
  `short-complex` (até 2 dias), `long-simple`, `long-complex` (até 7 dias).
- **GPUs:** 11× RTX 3090 (24 GB), 5× A100 (80 GB), 1× L40S (48 GB),
  1× RTX 5090, 1× RTX Blackwell 6000.
- **Armazenamento:** `$HOME`, `$DATASETS`, `$SCRATCH` (auto-limpeza em 90 dias),
  `$PROJECTS`, `$ARCHIVE`, `$SLURM_TMPDIR` (disco local rápido, ~4 TB).
- **Python:** `module load Python/3.10` + `venv`/`conda`; containers
  Singularity/Apptainer disponíveis.

### 1.2 Insumos necessários (o que precisamos)

| Item | Status |
|---|---|
| Acesso SSH à APUANA + VPN do CIn funcionando | **precisa confirmar** |
| Destino para o repo/dados (`$PROJECTS` ou `$SCRATCH`) e quota | **precisa confirmar** |
| Internet/HuggingFace a partir dos nós de **compute** | **incógnita** (docs não afirmam; alguns clusters bloqueiam egress) |
| Escolha de partição/GPU e tempo de parede por job | definir (sugestão: `short-complex`, `--gres=gpu:1`) |
| Método de transferência (`git` ou `rsync`) | definir |

### 1.3 Passos concretos

**A. Transferir o repositório para o cluster.**
- Opção 1: `git push` da branch `repro-local-phase1` para um remoto e `git clone`
  no cluster (recomendado; preserva histórico e o oráculo commitado).
- Opção 2: `rsync -a --exclude '.git' --exclude 'replication/runs' <local>/ user@apuana:$SCRATCH/ml_PIDS/`.
- Destino sugerido: `$SCRATCH/ml_PIDS` (resultados são regeneráveis; `$SCRATCH`
  não é backup) ou `$PROJECTS` se houver quota.

**B. Criar o ambiente no cluster (com GPU).**
- Diferença da Fase 1: instalar **torch com CUDA** (não o build CPU).
  ```bash
  module load Python/3.10        # ou conda
  python -m venv $SCRATCH/pids-repro
  source $SCRATCH/pids-repro/bin/activate
  pip install -r requirements.txt          # inclui torch (build CUDA)
  # (não usar replication/requirements-local.txt, que exclui torch)
  python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
  ```
- Alternativa: adaptar `replication/setup_env.sh` para o cluster (sem o índice CPU).
  Registrar as versões efetivas para reprodutibilidade.

**C. Dados.**
- Os CSVs processados (splits A/B/C + 3 regimes OOD) estão **commitados no repo**,
  então os notebooks `04`–`08` **não precisam de HuggingFace**.
- Só o notebook `02_preprocessing` (não usado na reprodução) precisa dos loaders
  externos do HF (`r1char9/...`, `leolee99/NotInject`).
- Se quiser rodar `07`/`08`, os **checkpoints dos backbones** (`roberta-base`,
  `deberta-base`) e o `BAAI/bge-small-en-v1.5` são baixados do HF — depende de
  egress dos nós de compute. **Verificar cedo**; se bloqueado, pré-baixar no nó
  de login (com internet) e apontar `HF_HOME`/cache para `$SCRATCH`.

**D. Scripts SLURM.**
- Criar `replication/slurm/` com scripts `sbatch` que chamam o harness. Esboço:
  ```bash
  #!/bin/bash
  #SBATCH --job-name=pids-repro
  #SBATCH --partition=short-complex
  #SBATCH --gres=gpu:1
  #SBATCH --cpus-per-task=8
  #SBATCH --mem=32G
  #SBATCH --time=04:00:00
  #SBATCH --output=slurm-%j.log

  source $SCRATCH/pids-repro/bin/activate
  cd $SCRATCH/ml_PIDS
  SPLIT_TAG=A python replication/run_local.py A --smoke-transformers --skip-pytest
  ```
- Para os transformers (`07`/`08`), rodar o harness com `--smoke-transformers`
  em GPU (configs fixas `num_epochs=3`, `max_length=256`; custo baixo com ~700
  exemplos).
- Rodar split A/B/C para as famílias leves: `python replication/run_local.py {A,B,C}`.

**E. Ajustes no harness para o cluster (pequenos).**
- Permitir selecionar notebooks (ex.: `--notebooks 07,08`) para não re-rodar
  04–06 quando o objetivo é só os transformers.
- Torch CUDA no cluster: `run_local.py` não precisa mudar (usa `sys.executable`).
- Confirmar que o override de kernel (`--ExecutePreprocessor.kernel_name=python3`)
  funciona no env do cluster.

**F. Coletar e comparar.**
- Trazer os `replication/runs/<run_id>/experiments/results/` de volta (rsync) ou
  rodar `compare_results.py` **no próprio cluster** contra o oráculo commitado.
- Consolidar um relatório da Fase 2 (novo `replication/report-phase2.md`),
  cobrindo transformers, splits B/C e as três conclusões que faltam.

### 1.4 Critérios de sucesso da Fase 2

1. Notebooks `07`/`08` rodam em GPU e geram métricas para RoBERTa-base,
   DeBERTa-base e RoBERTa+IBVS.
2. Rodar `04`/`05`/`06` para os splits B e C (repetibilidade completa A/B/C).
3. Comparação vs. oráculo: `auc_pr`/`roc_auc` próximos (esperado 100% em ranking),
   macro-F1 majoritariamente dentro de ±0.05, transformers competitivos.
4. Conclusão "RoBERTa+IBVS melhora low-FPR" verificada (ou refutada
   documentadamente).
5. Relatório da Fase 2 com ambiente, comandos, tabela de comparação e desvios.

---

## 2. Follow-ups de robustez (revisão final da Fase 1)

Não-bloqueantes, recomendados antes de escalar:

1. **Comparador silencioso em oráculo ausente/vazio:** `compare_results.py`
   retorna 0 e grava CSV mesmo com 0 pares. Adicionar aviso/exit não-zero quando
   `--oracle` não existe ou não há pares.
2. **`except Exception: continue` amplo** em `load_metrics_dir`: restringir e/ou
   logar CSVs ilegíveis para não subestimar a comparação.
3. **Teste da "costura" harness→comparador:** um teste end-to-end que garanta que
   o caminho de saída do `run_local` é o mesmo que o `compare_results` consome.
4. (Opcional) Testes explícitos de `A`/`C` em `parse_split_tag_from_name`.

---

## 3. Decisões em aberto / incógnitas

- **Egress dos nós de compute da APUANA** (HuggingFace): verificar; define se
  basta rodar online ou se é preciso pré-cachear modelos/datasets.
- **Partição/GPU por job:** `short-complex` com 1× GPU deve bastar; A100 só se
  necessário.
- **Onde guardar resultados:** `$SCRATCH` (temporário) + cópia para o repo local.
- **Escopo do estudo threshold/calibração:** o artigo aponta calibração como
  gargalo; opcionalmente expandir a Fase 2 para um experimento de calibration
  (temperature scaling) — fora do escopo atual.
