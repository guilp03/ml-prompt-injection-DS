# Spec: Reprodução local mínima do framework de detecção de prompt injection

**Data:** 2026-10-15
**Autor:** sessão de brainstorming
**Status:** aguardando revisão do usuário

## 1. Objetivo

Reproduzir, de forma fiel, o trabalho do artigo *"Prompt Injection Detection is
Regime-Dependent: A Deployment-Aware Evaluation with Interpretable Structural
Signals"* (Akinrele & Gowda, arXiv:2605.26999; PMLR v348) usando como base o
repositório público dos autores:

- `https://github.com/TimiAkinrele/ai-jailbreak-framework`

**Meta de aceite escolhida pelo usuário:** reprodução **qualitativa** (confirmar
as conclusões estruturais do artigo), com **números batendo minimamente** (dentro
de tolerância) como verificação de sanidade.

Este spec cobre **apenas a Fase 1 (reprodução local mínima)**. A adaptação para o
cluster APUANA (CIn/UFPE) é a **Fase 2**, fora do escopo deste documento.

## 2. Contexto

- Diretório de trabalho: `/home/gl-pereira/Projects/ml_PIDS` (atualmente contém
  apenas o PDF do artigo e um resumo em markdown; **não é um repositório git**).
- Máquina local: sem GPU, 16 CPUs, ~9.6 GB RAM, Python 3.13 (miniconda).
- Cluster APUANA: SLURM, com GPUs (RTX 3090/A100/L40S), acesso via VPN do CIn.
  Usado somente na Fase 2.
- O repo de referência já inclui **dados processados** (`data/processed/*.csv`
  para splits A/B/C, contendo todos os splits ID e os 3 regimes OOD) e **todas
  as tabelas de resultado commitadas** em `experiments/results/`.
  → Os notebooks `04`–`08` **não precisam de internet nem de download do
  HuggingFace**; apenas `02_preprocessing` usa os loaders externos do HF.

## 3. Decisões de escopo (confirmadas no brainstorming)

1. **Estratégia:** adaptar o repo de referência (opção "híbrido/vendor"), não
   reimplementar do zero. O foco é reproduzir; mudanças de método vêm depois.
2. **Ordem:** primeiro reprodução local mínima para garantir que tudo funciona;
   depois adaptar para o cluster.
3. **Escopo local (abordagem A):** rodar as famílias leves em CPU — notebooks
   `04` (lexical + flags + IBVS v1/v2), `05` (semântico BGE) e `06` (híbridos
   semânticos). Transformers fine-tuned (`07`, `08`) ficam para a Fase 2, com
   smoke opcional local.
4. **Verificação:** comparar saídas geradas contra o oráculo (CSVs commitados).

## 4. Arquitetura

`ml_PIDS` passa a ser um repositório git. A árvore do repo de referência é
copiada **verbatim** (preservando atribuição e o oráculo). Todo o tooling nosso
fica isolado em `replication/`.

```
ml_PIDS/
├── README.md                 # nosso: objetivo, atribuição, como rodar
├── environment.yml           # env conda Python 3.10 + deps (torch CPU)
├── requirements.txt          # (do ref, verbatim)
├── data/                     # (do ref, verbatim) raw + processed
├── src/                      # (do ref, verbatim) módulos reutilizáveis
├── notebooks/                # (do ref, verbatim) 02..08
├── scripts/                  # (do ref, verbatim) utilitários
├── tests/                    # (do ref, verbatim) suíte pytest
├── experiments/results/      # ORÁCULO: resultados dos autores — NÃO sobrescrever
└── replication/              # SOMENTE nosso
    ├── runs/                 # cópias isoladas onde os notebooks executam
    ├── run_local.sh          # harness de execução
    ├── compare_results.py    # diff runs/ vs oráculo
    ├── report.md             # relatório de reprodução (atualizado por run)
    └── NOTES.md              # decisões de adaptação / desvios
```

**Racional:** nunca sobrescrever o oráculo (`experiments/results/`) nem misturar
nosso tooling ao código original. Toda saída de execução vai para
`replication/runs/<run_id>/`.

## 5. Componentes

### 5.1 Ambiente (`environment.yml` + README)

- Env conda dedicado com `python=3.10`.
- Dependências instaladas a partir de `requirements.txt`
  (`datasets`, `jupyter`, `langdetect`, `numpy`, `pandas`, `pytest`,
  `scikit-learn`, `scipy`, `sentence-transformers`, `torch`, `transformers`,
  `xgboost`), com **torch CPU**.
- Fixar versões dos pacotes críticos (torch, transformers,
  sentence-transformers, scikit-learn, numpy) para reprodutibilidade.
- Comandos de criação/ativação documentados no README.

### 5.2 Harness de execução (`replication/run_local.sh`)

Assinatura: `run_local.sh <SPLIT_TAG> [--smoke-transformers]`

Passos:
1. Copia a árvore do ref para `replication/runs/<run_id>/`, onde
   `run_id = <timestamp>_<SPLIT_TAG>`.
2. Executa `pytest` dentro da cópia.
3. Executa os notebooks `04`, `05`, `06` via
   `jupyter nbconvert --execute --inplace`, com as variáveis de ambiente
   `SPLIT_TAG=<tag>` e `WRITE_MINIMAL_OUTPUTS=1`, **a partir do diretório
   `notebooks/`** (os notebooks fazem `PROJECT_ROOT = Path("..")`).
4. Com `--smoke-transformers`: executa `07` e `08` com configuração reduzida
   (epochs/max_length pequenos) apenas para validar execução em CPU. Padrão:
   desligado.
5. Todas as saídas ficam em `<run>/experiments/results/`.

### 5.3 Verificação (`replication/compare_results.py`)

- Lê os CSVs gerados em `runs/<run_id>/experiments/results/metrics/` e os do
  oráculo em `experiments/results/metrics/`.
- Faz join por par **modelo × regime** (ID, OOD primary, hard-negative,
  std-injection).
- Calcula deltas de `macro_f1`, `roc_auc`, `auc_pr` e `tpr@1%/5%/10%`.
- Aplica tolerância configurável (default `±0.05`; opção `±0.02`).
- Emite `runs/<run_id>/comparison.csv` + resumo no terminal: por métrica, quantos
  pares modelo×regime dentro/fora da tolerância.
- Marca explicitamente modelos **não comparáveis localmente** (transformers
  `07`/`08`).

### 5.4 Relatório (`replication/report.md`)

Registra: ambiente/versões, comandos executados, tabela de comparação
(oráculo vs. reproduzido), desvios observados e verificação dos critérios de
sucesso. `NOTES.md` registra decisões de adaptação (paths, flags, atalhos).

## 6. Fluxo de dados

```
repo ref (data/processed/*.csv, splits A/B/C + 3 regimes OOD)
        │  copiado para
        ▼
replication/runs/<run_id>/
        │  notebooks 04/05/06 (SPLIT_TAG, WRITE_MINIMAL_OUTPUTS=1)
        ▼
replication/runs/<run_id>/experiments/results/metrics/*.csv
        │  comparado com
        ▼
experiments/results/metrics/*.csv   ← ORÁCULO (intocado)
        │
        ▼
replication/runs/<run_id>/comparison.csv + replication/report.md
```

## 7. Critérios de sucesso (Fase 1)

Considera-se a Fase 1 bem-sucedida quando:

1. **Mecânica:** `pytest` passa e os notebooks `04/05/06` executam sem erro para
   `SPLIT_TAG=A`.
2. **Numérico mínimo:** para as famílias comparáveis (TF-IDF, flags, IBVS v1/v2,
   BGE, híbridos semânticos), macro-F1 e ROC-AUC ficam dentro de `±0.05` do
   oráculo na **maioria** dos pares modelo×regime. Pares que estourarem são
   documentados no relatório.
3. **Qualitativo (gate real):** reproduzir as conclusões estruturais do artigo
   na porção coberta localmente, notadamente: TF-IDF competitivo no regime
   hard-negative; IBVS complementar em alguns regimes; divergência entre ranking
   e desempenho a baixo FPR. A conclusão envolvendo RoBERTa+IBVS fica
   **deferida à Fase 2**.
4. **Reprodutibilidade:** `replication/report.md` registra ambiente, comandos,
   tabela de comparação e desvios.

## 8. Riscos e tratamento

| Risco | Tratamento |
|---|---|
| Python 3.13 incompatível com deps | env conda 3.10 dedicado |
| Notebooks sobrescrevem o oráculo | executar sempre em cópia isolada |
| `Path("..")` exige CWD correto | `nbconvert` a partir de `notebooks/` |
| Deltas numéricos por seed/versão | tolerância configurável + documentar |
| `02_preprocessing` requer HF/internet | opcional; se rodar, em cópia descartável |
| CPU/RAM no smoke de transformers | config reduzida, off por padrão |
| Espaço em disco (torch/BGE) | verificar disponibilidade; BGE-small ~130 MB |

## 9. Fora de escopo (Fase 1)

- Adaptação para a APUANA (SLURM, containers, paths `$SCRATCH`/`$SLURM_TMPDIR`,
  env no cluster) — Fase 2.
- Rodar `02_preprocessing` como fonte oficial dos splits (mantemos os do ref;
  `02` é validação opcional).
- Estudo de calibração (temperature scaling) — trabalho futuro do artigo.

## 10. Próximo passo

Após revisão e aprovação deste spec, invocar a skill `writing-plans` para
produzir o plano de implementação detalhado da Fase 1.
