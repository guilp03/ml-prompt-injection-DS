# Relatório de reprodução — Fase 2 (APUANA)

## Ambiente
Preencher com a saída de `replication/runs/env-lock.txt`, versão de torch/CUDA,
partição e GPU usadas.

## Comandos
Registrar os comandos de submissão, os job IDs e o walltime observado de cada job.

## Tabela de comparação
Inserir `replication/phase2/phase2_summary.md` e destacar, por split e família,
os pares fora de ±0.05 em `macro_f1` e as métricas de baixo FPR.

Confirmar que os seis jobs e todos os splits A/B/C concluíram. Além de
`macro_f1` e das métricas de baixo FPR, reportar a proximidade de
`auc_pr`/`roc_auc` em relação ao oráculo.

## Conclusões
Preencher com a verificação de:
- RoBERTa+IBVS melhora low-FPR?
- transformers superam baselines léxicos?
- TF-IDF segue competitivo no hard-negative?
- IBVS é complementar?

## Desvios
Listar divergências numéricas relevantes e hipóteses (versão, seed, threshold).

## Limitações
Registrar qualquer job que não concluiu e o impacto nos critérios de sucesso.
