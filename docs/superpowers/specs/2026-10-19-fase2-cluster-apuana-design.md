# Spec: Fase 2 — reprodução no cluster APUANA (transformers e splits B/C)

**Data:** 2026-10-19
**Autor:** sessão de brainstorming
**Status:** aguardando revisão do usuário
**Fase anterior:** `docs/superpowers/specs/2026-10-15-replicacao-local-ai-jailbreak-framework-design.md`

## 1. Objetivo

Concluir a reprodução do artigo *"Prompt Injection Detection is
Regime-Dependent: A Deployment-Aware Evaluation with Interpretable Structural
Signals"* no cluster APUANA (CIn/UFPE), fechando as lacunas deixadas pela Fase 1:

1. Executar os notebooks `07` (RoBERTa-base, DeBERTa-base) e `08`
   (RoBERTa + IBVS) em GPU.
2. Executar os notebooks `04`, `05` e `06` para os splits `B` e `C`,
   completando a repetibilidade `A/B/C`.
3. Comparar os resultados gerados contra o oráculo commitado em
   `experiments/results/` e documentar concordâncias e desvios.

O escopo é **reprodução completa** (critérios 1–5 de `replication/NEXT_STEPS.md`),
sem calibração/threshold scaling e sem refatorações de robustez além das
descritas neste documento.

## 2. Contexto

- A Fase 1 (branch `repro-local-phase1`) executou `04→06` no split `A` em CPU e
  reproduziu as conclusões qualitativas centrais. Relatório em
  `replication/report.md`.
- O harness `replication/run_local.py` isola cada execução em
  `replication/runs/<run_id>/`, nunca sobrescreve o oráculo e aceita
  `--skip-pytest`, `--dry-run`, `--runs-dir`, `--minimal-outputs` e
  `--smoke-transformers`.
- O comparador `replication/compare_results.py` gera `comparison.csv` contra o
  oráculo e retorna 0 mesmo quando não há pares; os scripts da Fase 2 devem
  detectar esse caso.
- Acesso à APUANA (SSH + VPN do CIn) já está funcionando. O entregável desta
  frente é **scripts + harness + runbook**, validados localmente; a execução
  real no cluster será feita pelo usuário.
- Dados processados dos splits A/B/C e os três regimes OOD estão commitados no
  repo. Apenas checkpoints/tokenizers do Hugging Face precisam de rede ou cache.

## 3. Decisões de escopo (confirmadas no brainstorming)

1. **Escopo:** reprodução completa (`07/08` + splits B/C).
2. **Acesso:** APUANA já disponível; o plano assume verificação prática dos
   fatos do cluster (partições, quotas, egress).
3. **Granularidade:** 6 jobs independentes — 2 famílias × 3 splits:
   `light_<split>` (`04→06`, CPU) e `tx_<split>` (`07→08`, GPU).
4. **Comparação:** executada no cluster; voltam para o repo apenas os resumos
   (`phase2_comparison_all.csv`, `phase2_summary.md`) e o relatório.
5. **Entregável desta frente:** implementar e validar o tooling localmente e
   documentar o runbook; a execução fica a cargo do usuário.

## 4. Arquitetura

O harness existente continua sendo o único executor. A camada nova é fina:
seleção de notebooks, identificação de run e wrappers SLURM.

```
replication/
├── run_local.py                        # estendido: --notebooks, --label
├── tests/test_run_local.py             # testes novos + regressão
├── aggregate_phase2.py                 # consolida comparison.csv dos 6 jobs
├── tests/test_aggregate_phase2.py
├── slurm/
│   ├── common.sh                       # paths, env, preflight, log
│   ├── setup_env_cluster.sh            # venv + torch CUDA + deps + freeze
│   ├── light.sh                        # job 04→05→06 de um split
│   ├── transformers.sh                 # job 07→08 de um split
│   ├── submit_phase2.sh                # submete 6 jobs
│   └── README.md                       # runbook operacional
├── phase2/
│   ├── phase2_comparison_all.csv       # gerado pelo agregador
│   └── phase2_summary.md               # gerado pelo agregador
├── report-phase2.md                    # relatório final preenchido
└── runs/<split>/<family>/<run_id>/     # runs isolados (fora do git)
```

Racional: manter o comportamento já testado do harness, evitar orquestrador
próprio para apenas 6 jobs e deixar o comparador como fonte única de verdade.

## 5. Componentes

### 5.1 `run_local.py` (extensão)

- Novo argumento `--notebooks`: lista separada por vírgula de números de
  notebook (`04,05,06` ou `07,08`). A resolução é feita pelo prefixo numérico
  contra `CORE_NOTEBOOKS + TRANSFORMER_NOTEBOOKS`; notebooks desconhecidos
  geram erro de argumento. Quando ausente, mantém-se o comportamento atual
  (`04→06`, mais `07/08` se `--smoke-transformers`).
- Novo argumento `--label`: sufixo sanitizado (`[A-Za-z0-9_-]`) anexado ao
  `run_id`, resultando em `<utc_timestamp>_<split>_<label>`. Sem `--label`, o
  formato atual é preservado.
- Nenhuma mudança no isolamento por cópia, no envio de `SPLIT_TAG`/
  `WRITE_MINIMAL_OUTPUTS` nem no kernel forçado (`python3`).

### 5.2 `aggregate_phase2.py`

- Entrada: `--runs-root` (default `replication/runs`), varrendo
  `runs/**/comparison.csv`.
- Extrai `split` e `family` do caminho (`runs/<split>/<family>/<run_id>/`)
  quando presentes e adiciona como colunas.
- Concatena, remove duplicatas por `(model, split, eval_track, ood_name,
  split_tag)` mantendo a última ocorrência e grava:
  - `replication/phase2/phase2_comparison_all.csv`
  - `replication/phase2/phase2_summary.md` com contagens dentro/fora da
    tolerância por métrica, por split e por família.
- Falha com exit não-zero se não encontrar nenhum `comparison.csv` ou nenhuma
  linha.

### 5.3 Scripts SLURM

**`common.sh`** (carregado via `source`, não executado diretamente):
- Define `REPO_ROOT`, `RUNS_ROOT`, `HF_HOME` e ativa o venv.
- Preflight: repo correto, Python do venv, arquivos processados dos splits
  A/B/C, oráculo presente, e (no job tx) `torch.cuda.is_available()`.
- Cria dirs de runs/logs e imprime ambiente + `nvidia-smi` quando aplicável.

**`setup_env_cluster.sh`** (executado uma vez no nó de login):
- `module load Python/3.10` (ou Python do sistema, se já for 3.10+).
- Cria `$SCRATCH/pids-repro` e instala `requirements.txt` com torch CUDA.
- Registra `pip freeze` em `replication/runs/env-lock.txt` e imprime versões de
  torch/CUDA.

**`light.sh`** e **`transformers.sh`**:
- Recebem `SPLIT_TAG` via `--export` ou como primeiro argumento.
- Chamam o harness com `--notebooks` e `--label`, apontando `--runs-dir` para
  `runs/<split>/<family>`.
- Capturam o exit code do harness, executam `compare_results.py` com
  `--oracle <repo>/experiments/results/metrics` e `--out <run_dir>/comparison.csv`,
  e então:
  - falham se `comparison.csv` não tiver linhas;
  - preservam o exit code original do harness caso ele tenha falhado.
- Usam `set -euo pipefail`; a captura de status é feita com `set +e` em torno
  apenas da chamada do harness.

**`submit_phase2.sh`**:
- Submete os 6 jobs sem dependências e imprime os IDs.
- Recursos configuráveis por variáveis de ambiente (defaults entre parênteses):
  `PARTITION_LIGHT` (`short-simple`), `PARTITION_TX` (`short-complex`),
  `WALLTIME_LIGHT` (`02:00:00`), `WALLTIME_TX` (`04:00:00`),
  `GRES_TX` (`gpu:1`), `CPUS_PER_TASK` (`8`), `MEM` (`32G`),
  `RUNS_ROOT` (`$SCRATCH/ml_PIDS/replication/runs`).
- Aceita lista de splits (`A B C` por default), permitindo piloto só com `A`.

### 5.4 Relatório

`replication/report-phase2.md` contém as seções: ambiente/versões, comandos,
tabela de comparação por família/split, conclusões (incluindo RoBERTa+IBVS e
low-FPR), desvios e limitações. É preenchido após a execução; esta frente cria
o template.

## 6. Fluxo de execução

Pré-requisitos no nó de login:

```bash
export SCRATCH=/path/do/scratch          # conforme a APUANA
git clone <repo> "$SCRATCH/ml_PIDS" && cd "$SCRATCH/ml_PIDS"
bash replication/slurm/setup_env_cluster.sh
export HF_HOME="$SCRATCH/hf-cache"
python -c "from transformers import AutoTokenizer; \
  [AutoTokenizer.from_pretrained(m) for m in \
   ('BAAI/bge-small-en-v1.5','roberta-base','microsoft/deberta-base')]"
```

- `04→06` não usam datasets do HF, mas o `05` baixa o `BAAI/bge-small-en-v1.5`;
  por isso o cache HF é necessário também para os jobs leves.
- Se os nós de compute não tiverem egress, usar o cache pré-populado com
  `HF_HUB_OFFLINE=1` (e `TRANSFORMERS_OFFLINE=1`).

Submissão e acompanhamento:

```bash
bash replication/slurm/submit_phase2.sh A B C
squeue -u "$USER"
# após concluir:
python replication/aggregate_phase2.py
```

Layout dos runs:

```
runs/<split>/light/<utc>_<split>_light/experiments/results/metrics/
runs/<split>/light/<utc>_<split>_light/comparison.csv
runs/<split>/tx/<utc>_<split>_tx/experiments/results/metrics/
runs/<split>/tx/<utc>_<split>_tx/comparison.csv
```

Ao final, copiar `replication/phase2/`, `replication/report-phase2.md` e, se
desejado, um tarball das métricas para `$PROJECTS` ou para a máquina local
antes da expiração de 90 dias do `$SCRATCH`.

## 7. Recursos SLURM

| Job | Partição | Recursos | Conteúdo | Walltime |
|---|---|---|---|---|
| `light_<split>` | `short-simple` | 8 CPU, 32G | notebooks `04,05,06` | 2h |
| `tx_<split>` | `short-complex` | 1 GPU, 8 CPU, 32G | notebooks `07,08` | 4h |

Os notebooks mantêm as configs fixas atuais (`num_epochs=3`, `max_length=256`,
`train_batch_size=8`). Nenhum hiperparâmetro é alterado nesta fase.

## 8. Tratamento de erros e retomada

- Preflight falha cedo com mensagem explícita (env, dados, oráculo, GPU).
- Falha de notebook não descarta resultados parciais: a comparação roda sobre o
  que foi gerado e o job termina com o exit code original.
- Retomada: re-submeter somente o job afetado; `run_id` com split+label e
  timestamp UTC evita colisão e preserva runs anteriores.
- Comparação vazia (0 linhas) é tratada como falha do job, nunca como sucesso
  silencioso.
- `--dry-run` do harness permite inspecionar comandos sem executar notebooks.

## 9. Testes e validação

Antes do cluster:

1. `pytest replication/tests` — regressão dos 21 testes + novos casos:
   - `--notebooks 04,05,06` e `07,08` selecionam e ordenam corretamente;
   - notebook desconhecido gera erro;
   - `--label` aparece no `run_id` e é sanitizado.
2. `pytest` para `aggregate_phase2.py` (concatenação, deduplicação, resumo,
   falha em entrada vazia).
3. `bash -n` em todos os scripts e `shellcheck` quando disponível.
4. `run_local.py --dry-run` para as duas famílias.

No cluster:

5. `sbatch --test-only` para validar partição/recursos.
6. Piloto com `submit_phase2.sh A` antes de submeter `B` e `C`.
7. Preflight manual: `torch.cuda.is_available()`, cache HF e oráculo.

## 10. Critérios de sucesso (Fase 2)

1. Os 6 jobs concluem; `07/08` geram métricas para `TRANSFORMER_ROBERTA_BASE`,
   `TRANSFORMER_DEBERTA_BASE` e as variantes RoBERTa+IBVS.
2. Splits `A/B/C` completos para `04→06` e `07→08` (repetibilidade completa).
3. Comparação contra o oráculo: `auc_pr`/`roc_auc` próximos (esperado próximo de
   100% em ranking), macro-F1 majoritariamente em `±0.05`; pares fora da
   tolerância documentados.
4. Conclusão "RoBERTa+IBVS melhora low-FPR" verificada ou refutada com
   evidência.
5. `report-phase2.md`, `phase2_comparison_all.csv` e `phase2_summary.md`
   presentes, com ambiente, comandos, tabela comparativa e desvios.

## 11. Riscos e mitigação

| Risco | Mitigação |
|---|---|
| Egress HF bloqueado nos nós de compute | pré-download no login, `HF_HOME` em `$SCRATCH`, modo offline |
| Walltime insuficiente para `07/08` | 4h de margem, piloto no split A, reagendamento via `WALLTIME_TX` |
| `$SCRATCH` expira em 90 dias | copiar resumos/relatório/tarball para `$PROJECTS`/local |
| Drift de versões de libs | `pip freeze` em `runs/env-lock.txt`; ambiente dedicado |
| GPU não alocada ou OOM | preflight de CUDA; 32G de RAM; batch fixo em 8 |
| Colisão de run dirs | `runs/<split>/<family>/<run_id>` + label no `run_id` |
| Comparador silencioso | guarda de `comparison.csv` vazio com exit não-zero |
| Partições/quotas diferentes do esperado | variáveis de ambiente no `submit_phase2.sh`; `--test-only` |

## 12. Fora de escopo

- Calibração/threshold scaling e qualquer mudança de método.
- Refatorações de robustez do `compare_results.py` além da guarda de vazio
  (ex.: restringir `except`, teste end-to-end completo).
- Reexecutar `02_preprocessing` ou depender dos loaders externos do HF.
- Alterar a árvore de referência ou os notebooks dos autores.
- Commitar runs brutos ou métricas por split no git.

## 13. Próximo passo

Após revisão e aprovação deste spec, invocar a skill `writing-plans` para
produzir o plano de implementação detalhado da Fase 2.
