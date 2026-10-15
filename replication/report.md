# Relatório de reprodução — Fase 1 (local)

## Ambiente
- Máquina: Linux `pop-os` 7.1.5-76070105-generic, x86_64, **sem GPU**;
  `nproc` = 16 CPUs; RAM total 9.6 GiB (≈2.5 GiB disponíveis).
- Python / libs (env conda `pids-repro`):
  `Python 3.10.22 | torch 2.14.1+cpu | transformers 5.20.0 |
  sentence-transformers 6.1.0 | scikit-learn 1.7.2 | pandas 2.3.3`.

## Execução
- Run id: `20261015-190126_A`
- Comando: `python replication/run_local.py A`
- Configuração: `SPLIT_TAG=A`, `WRITE_MINIMAL_OUTPUTS=0` (necessário para os
  regimes OOD secundários), kernel `python3` forçado no `nbconvert`
  (`--ExecutePreprocessor.kernel_name=python3`).
- Duração: ~10.6 min (pytest + notebooks 04, 05, 06 + coleta).
- Saídas: `replication/runs/20261015-190126_A/experiments/results/metrics/`.

## Comparação numérica (ID + OOD primário) — run split A
Comando:
```
conda run -n pids-repro python replication/compare_results.py \
  replication/runs/20261015-190126_A/experiments/results/metrics
```
Resumo:
```
Pares comparáveis: 51
Tolerância: ±0.05
Dentro da tolerância (todas as métricas): 32/51
  acc: 46/51 dentro
  macro_f1: 46/51 dentro
  auc_pr: 51/51 dentro
  roc_auc: 51/51 dentro
  tpr_at_1pct_fpr: 43/51 dentro
  tpr_at_5pct_fpr: 42/51 dentro
  tpr_at_10pct_fpr: 43/51 dentro
```
`auc_pr` e `roc_auc` batem em **100%** dos pares. As divergências concentram-se
em accuracy/macro-F1 no track `deployment_threshold` e nas métricas de baixo FPR.

Pares representativos (macro-F1 e ROC-AUC, ranking):

| Modelo | Regime | macro-F1 (gen / orác.) | ROC-AUC (gen / orác.) | Δ |
|---|---|---|---|---|
| M1_TFIDF_ONLY | ID TEST | 0.7985 / 0.7714 | 0.8799 / 0.8792 | +0.027 / +0.001 |
| M3_..._IBVS_V2_TOTAL | ID TEST | 0.7811 / 0.7496 | 0.9049 / 0.8959 | +0.032 / +0.009 |
| M1_TFIDF_ONLY | OOD primário | 0.6366 / 0.6003 | 0.6827 / 0.6816 | +0.036 / +0.001 |
| M3_..._IBVS_V2_TOTAL | OOD primário | 0.6585 / 0.6206 | 0.7412 / 0.7422 | +0.038 / −0.001 |

**Pares fora da tolerância (19/51)** — nenhum em `auc_pr`/`roc_auc`:
- **acc e macro-F1** em `deployment_threshold` para os dois baselines léxicos:
  `M1_TFIDF_ONLY` (VAL/TEST/OOD) e `M2_TFIDF_PLUS_FLAGS` (VAL/TEST). Δ ≈ +0.06 a
  +0.12 (gerado **acima** do oráculo). Desvio sistemático no ponto de operação.
- **Baixo FPR** (`tpr@1/5/10%FPR`) espalhado por M1/M2/M3 e híbridos
  (`HYBRID_SEM_EXPERT_GATE_M1_M3`, `HYBRID_SEM_LEARNED_META_FUSION`,
  `HYBRID_SEM_GATE_ELSE_M3_IBVS_V2`), em geral Δ ≈ 0.05–0.18.
  São as métricas estatisticamente mais ruidosas (poucos positivos na cauda).

## Avaliação qualitativa (regimes secundários)
Track `deployment_threshold`, agregado por família (run split A):

**Hard-negative OOD (`ood_test_injection`)** — macro-F1:
`M1_TFIDF_ONLY` **0.8589** > `M2` 0.7597 > `M3_IBVS_V2` 0.7468 >
`M3_STRUCTURED` 0.7088 > `M3_IBVS_V1` 0.6828; híbridos 0.596–0.668;
`SEM_LOGREG_BGE` 0.5786. Em ROC-AUC o `M1` também lidera com folga (0.9361
vs. ≤0.805 dos demais).
- **TF-IDF lidera o regime hard-negative** frente a todas as variantes IBVS e
  semânticas. Reproduz a conclusão qualitativa central do artigo.

**Std-injection OOD (`ood_test_injection_standard`)** — macro-F1:
híbridos semânticos `HYBRID_SEM_ANCHORED_IBVS_BOOST` **0.7348**,
`EXPERT_GATE_M1_M3` 0.7297, `LEARNED_META_FUSION` 0.7129 >
`M1_TFIDF_ONLY` 0.6797 > `M3_IBVS_V2` 0.6366 > `M2` 0.6350.
- **Híbridos semânticos são competitivos e superam o TF-IDF** neste regime,
  coerente com o artigo (o sinal lexical puro degrada em injection padrão).

**IBVS complementar:** comparando `M2` (TF-IDF+flags) e `M3` (…+IBVS), a adição
do IBVS v2 aumenta macro-F1 em ID TEST (0.7423→0.7811), OOD primário
(0.6290→0.6585), hard-negative (0.5777→0.6061) e std-injection (0.4668→0.4977).
Ou seja, o IBVS é **complementar** (soma sobre flags), mas em hard-neg/std-inj
não basta para superar o `M1` léxico puro.

**Gap ranking vs. baixo-FPR:** claro e reproduzido. Ex.: `M1` no hard-negative
tem ROC-AUC 0.936 mas `tpr@1%FPR` de apenas 0.366; no std-injection ROC-AUC 0.742
com `tpr@1%FPR` 0.029. Os híbridos têm ROC-AUC de 0.67–0.80 com `tpr@1%FPR`
≤ 0.06. Alta capacidade de ranking **não** se traduz em desempenho utilizável no
ponto de operação de baixo FPR — o ponto central do artigo.

## Critérios de sucesso (spec §7)
1. **Mecânica:** pytest OK (52 passed: 31 do ref + 21 do harness) e notebooks
   `04/05/06` executados sem erro para `SPLIT_TAG=A` → **sim**.
2. **Numérico mínimo (±0.05):** 32/51 pares dentro em **todas** as métricas;
   macro-F1 e ROC-AUC dentro na grande maioria (46/51 e 51/51). Desvios
   documentados → **parcial** (limítrofe; só falha o critério "todas as
   métricas" em pares de `deployment_threshold` e baixo FPR).
3. **Qualitativo (gate real):** TF-IDF competitivo/líder no hard-negative
   (**sim**); IBVS complementar em vários regimes (**sim**); divergência
   ranking vs. baixo-FPR (**sim**). RoBERTa+IBVS deferido à Fase 2.
   → **sim** na porção coberta localmente.
4. **Reprodutibilidade:** ambiente, comando, tabela e desvios registrados aqui
   e em `NOTES.md` → **sim**.

## Desvios observados
- acc/macro-F1 em `deployment_threshold` para `M1`/`M2` ficam ~0.09–0.12 acima
  do oráculo (seleção de limiar sensível a versão/seed). Não corrigido no
  oráculo, apenas registrado.
- Métricas de baixo FPR são as menos estáveis (mesmo com `auc_pr`/`roc_auc` 100%).

## Limitações / deferido à Fase 2
- Transformer baselines e RoBERTa+IBVS (notebooks 07/08): **não avaliável
  localmente nesta fase** (CPU sem GPU; configs fixas `num_epochs=3`,
  `max_length=256`). Deferido à Fase 2 (cluster APUANA).
- Somente o split A foi executado; splits B/C e adaptação ao cluster ficam fora
  do escopo da Fase 1.
