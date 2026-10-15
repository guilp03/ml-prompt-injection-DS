# Reprodução local — Prompt Injection Detection is Regime-Dependent

Tooling de reprodução local (Fase 1) sobre o repositório de referência dos
autores: `https://github.com/TimiAkinrele/ai-jailbreak-framework` (commit `ec5cac98e942aa38c49018a364e5cb2a9ae138e1`).

## Ambiente

```bash
bash replication/setup_env.sh          # cria o env conda `pids-repro`
conda activate pids-repro
```

## Executar a reprodução local (split A)

```bash
conda run -n pids-repro python replication/run_local.py A
```

Isso copia a árvore do ref para `replication/runs/<run_id>/`, roda `pytest` e os
notebooks `04`, `05`, `06` com `WRITE_MINIMAL_OUTPUTS=0`. As saídas ficam em
`replication/runs/<run_id>/experiments/results/`. O oráculo em
`experiments/results/` nunca é tocado.

## Comparar com o oráculo

```bash
conda run -n pids-repro python replication/compare_results.py \
  replication/runs/<run_id>/experiments/results/metrics
```

Gera `comparison.csv` e imprime um resumo dentro/fora da tolerância.

## Estrutura

- `run_local.py` / `run_local.sh` — harness de execução (isolado por run).
- `compare_results.py` — diff de métricas geradas vs. oráculo.
- `report.md` — relatório de reprodução.
- `NOTES.md` — decisões de adaptação e desvios.
