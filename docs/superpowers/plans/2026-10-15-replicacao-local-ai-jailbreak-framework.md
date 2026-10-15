# Reprodução local mínima do framework de prompt injection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adaptar o repositório público `TimiAkinrele/ai-jailbreak-framework` em `ml_PIDS` e executar localmente as famílias leves (lexical, IBVS, semântico, híbridos) do split A, comparando os resultados contra as tabelas commitadas pelos autores (oráculo).

**Architecture:** O repo de referência é copiado **verbatim** para a raiz de `ml_PIDS` (preservando `data/`, `src/`, `notebooks/`, `tests/` e o oráculo em `experiments/results/`). Todo o tooling de reprodução fica isolado em `replication/`. Um harness em Python copia a árvore para `replication/runs/<run_id>/` (excluindo `experiments/`, `replication/`, `docs/`, `.git`), roda `pytest` e os notebooks `04`→`06`, e um comparador faz o diff dos CSVs de métricas gerados contra o oráculo.

**Tech Stack:** Python 3.10 (conda), torch CPU, transformers, sentence-transformers, scikit-learn, pandas, jupyter/nbconvert, pytest.

**Referência:** spec em `docs/superpowers/specs/2026-10-15-replicacao-local-ai-jailbreak-framework-design.md`.

---

## Estrutura de arquivos

| Arquivo | Responsabilidade |
|---|---|
| `environment.yml` | Definição do ambiente conda (Python 3.10 + pip). |
| `replication/requirements-local.txt` | Deps do ref **sem torch** (torch é instalado como build CPU). |
| `replication/setup_env.sh` | Cria o env e instala deps + torch CPU. |
| `replication/run_local.py` | Harness testável (copia árvore, roda pytest e notebooks). |
| `replication/run_local.sh` | Wrapper fino de `run_local.py`. |
| `replication/compare_results.py` | Diff dos CSVs gerados vs. oráculo. |
| `replication/__init__.py` | Torna `replication` um pacote importável nos testes. |
| `replication/tests/test_compare_results.py` | Testes de `compare_results.py`. |
| `replication/tests/test_run_local.py` | Testes de `run_local.py`. |
| `replication/runs/.gitignore` | Ignora o conteúdo de `runs/`. |
| `replication/README.md` | Objetivo, atribuição e comandos. |
| `replication/NOTES.md` | Decisões de adaptação / desvios. |
| `replication/report.md` | Relatório de reprodução (preenchido no Task 7). |

---

### Task 1: Vendorizar o repositório de referência

**Files:**
- Create (copiados): `README.md`, `.gitignore`, `requirements.txt`, `data/`, `src/`, `notebooks/`, `scripts/`, `tests/`, `experiments/`

- [ ] **Step 1: Clonar o repo de referência em diretório temporário**

```bash
rm -rf /tmp/ai-jailbreak-framework
git clone --depth 1 https://github.com/TimiAkinrele/ai-jailbreak-framework.git /tmp/ai-jailbreak-framework
```

Expected: clone concluído sem erro.

- [ ] **Step 2: Registrar o commit de referência (para atribuição)**

```bash
git -C /tmp/ai-jailbreak-framework rev-parse HEAD
```

Anote o hash retornado; ele será usado no `replication/NOTES.md` (Task 5).

- [ ] **Step 3: Copiar a árvore verbatim para a raiz do projeto**

```bash
rsync -a --exclude '.git' /tmp/ai-jailbreak-framework/ /home/gl-pereira/Projects/ml_PIDS/
```

(roda em `/home/gl-pereira/Projects/ml_PIDS` como destino)

- [ ] **Step 4: Verificar a cópia**

```bash
cd /home/gl-pereira/Projects/ml_PIDS
test -f src/features/ibvs.py \
  && test -d data/processed \
  && test -f experiments/results/metrics/results_table_v2_repeatability.csv \
  && test -f experiments/results/metrics/metrics_ablation_splitA.csv \
  && test -f requirements.txt \
  && echo "VENDO OK"
```

Expected: `VENDO OK`.

- [ ] **Step 5: Verificar contagem de arquivos vs. referência**

```bash
echo "ref:  $(find /tmp/ai-jailbreak-framework -type f -not -path '*/.git/*' | wc -l)"
echo "dest: $(find /home/gl-pereira/Projects/ml_PIDS -type f -not -path '*/.git/*' -not -path '*/replication/*' -not -path '*/docs/*' | wc -l)"
```

Expected: os dois números devem ser iguais (a única diferença de arquivos é o nosso `docs/` e `replication/`, ambos excluídos na contagem).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: vendoriza repo de referencia ai-jailbreak-framework"
```

---

### Task 2: Ambiente Python dedicado

**Files:**
- Create: `environment.yml`
- Create: `replication/requirements-local.txt`
- Create: `replication/setup_env.sh`

- [ ] **Step 1: Criar `replication/requirements-local.txt`**

Conteúdo (igual ao `requirements.txt` do ref **sem** a linha `torch`):

```
datasets
jupyter
langdetect
numpy
pandas
pytest
scikit-learn
scipy
sentence-transformers
transformers
xgboost
```

- [ ] **Step 2: Criar `environment.yml`**

```yaml
name: pids-repro
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pip
```

- [ ] **Step 3: Criar `replication/setup_env.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=pids-repro
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if ! conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
  conda env create -f environment.yml
fi

conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
# torch build CPU (evita baixar wheel CUDA ~2GB em máquina sem GPU)
conda run -n "${ENV_NAME}" python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu torch
conda run -n "${ENV_NAME}" python -m pip install \
  -r replication/requirements-local.txt

conda run -n "${ENV_NAME}" python -c \
  "import torch, transformers, sentence_transformers, sklearn, pandas, xgboost; print('env ok', torch.__version__)"
```

- [ ] **Step 4: Dar permissão de execução**

```bash
chmod +x replication/setup_env.sh
```

- [ ] **Step 5: Criar o ambiente (requer rede; ~5–10 min)**

```bash
bash replication/setup_env.sh
```

Expected: termina com `env ok <versao>`.

- [ ] **Step 6: Verificar a suíte de testes do ref no ambiente**

```bash
conda run -n pids-repro pytest -q
```

Expected: testes coletados; não deve haver erro de import. Se algum teste do ref falhar por versão, anote em `replication/NOTES.md` (Task 5) — isso não bloqueia a Fase 1.

- [ ] **Step 7: Commit**

```bash
git add environment.yml replication/requirements-local.txt replication/setup_env.sh
git commit -m "chore: ambiente conda pids-repro com torch CPU"
```

---

### Task 3: Comparador de resultados (`compare_results.py`) — TDD

**Files:**
- Create: `replication/__init__.py`
- Create: `replication/compare_results.py`
- Test: `replication/tests/test_compare_results.py`

- [ ] **Step 1: Criar `replication/__init__.py` vazio**

```bash
touch /home/gl-pereira/Projects/ml_PIDS/replication/__init__.py
```

- [ ] **Step 2: Escrever os testes que falham**

Criar `replication/tests/test_compare_results.py`:

```python
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from replication import compare_results as cr


def _write_metrics(dirpath: Path, model: str, split: str, eval_track: str,
                   macro_f1: float, roc_auc: float,
                   filename: str = "metrics_ablation_splitA.csv") -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "model": model, "split": split, "eval_track": eval_track,
        "acc": 0.8, "macro_f1": macro_f1, "roc_auc": roc_auc, "auc_pr": 0.9,
        "tpr_at_1pct_fpr": 0.1, "tpr_at_5pct_fpr": 0.2, "tpr_at_10pct_fpr": 0.3,
    }]).to_csv(dirpath / filename, index=False)


def test_parse_ood_from_name():
    assert cr.parse_ood_from_name(
        Path("metrics_ablation_splitA__ood-ood_test_injection.csv")
    ) == "ood_test_injection"
    assert cr.parse_ood_from_name(Path("metrics_ablation_splitA.csv")) is None


def test_load_metrics_dir_infers_ood_name_from_suffix(tmp_path):
    _write_metrics(tmp_path, "M1", "OOD", "ranking", 0.5, 0.6,
                   filename="metrics_ablation_splitA__ood-ood_test_injection.csv")
    df = cr.load_metrics_dir(tmp_path)
    assert (df["ood_name"] == "ood_test_injection").all()


def test_build_comparison_flags_within_tolerance(tmp_path):
    gen_dir = tmp_path / "gen"
    oracle_dir = tmp_path / "oracle"
    _write_metrics(gen_dir, "M1", "TEST", "ranking", 0.80, 0.90)
    _write_metrics(oracle_dir, "M1", "TEST", "ranking", 0.82, 0.99)
    comp = cr.build_comparison(
        cr.load_metrics_dir(gen_dir), cr.load_metrics_dir(oracle_dir), tolerance=0.05
    )
    assert len(comp) == 1
    row = comp.iloc[0]
    assert abs(float(row["macro_f1_delta"]) - (-0.02)) < 1e-9
    assert bool(row["macro_f1_within_tol"]) is True
    assert bool(row["roc_auc_within_tol"]) is False
    assert bool(row["all_within_tol"]) is False


def test_summarize_reports_counts(tmp_path):
    gen_dir = tmp_path / "gen"
    oracle_dir = tmp_path / "oracle"
    _write_metrics(gen_dir, "M1", "TEST", "ranking", 0.80, 0.90)
    _write_metrics(oracle_dir, "M1", "TEST", "ranking", 0.81, 0.91)
    comp = cr.build_comparison(
        cr.load_metrics_dir(gen_dir), cr.load_metrics_dir(oracle_dir), tolerance=0.05
    )
    text = cr.summarize(comp, 0.05)
    assert "Dentro da tolerância" in text
```

- [ ] **Step 3: Rodar os testes e confirmar falha**

```bash
conda run -n pids-repro pytest replication/tests/test_compare_results.py -q
```

Expected: FAIL com `ModuleNotFoundError: No module named 'replication.compare_results'` (módulo ainda não existe).

- [ ] **Step 4: Implementar `replication/compare_results.py`**

```python
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

METRIC_COLS = [
    "acc", "macro_f1", "auc_pr", "roc_auc",
    "tpr_at_1pct_fpr", "tpr_at_5pct_fpr", "tpr_at_10pct_fpr",
]
KEY_COLS = ["model", "split", "eval_track", "ood_name"]
SKIP_FILES = {"results_table_v2_repeatability.csv"}


def parse_ood_from_name(path: Path) -> str | None:
    match = re.search(r"__ood-([A-Za-z0-9_]+)", path.stem)
    return match.group(1) if match else None


def load_metrics_dir(root: Path) -> pd.DataFrame:
    frames = []
    for csv_path in sorted(root.rglob("*.csv")):
        if csv_path.name in SKIP_FILES:
            continue
        try:
            frame = pd.read_csv(csv_path)
        except Exception:
            continue
        if "model" not in frame.columns or "macro_f1" not in frame.columns:
            continue
        if "ood_name" not in frame.columns:
            parsed = parse_ood_from_name(csv_path)
            frame["ood_name"] = parsed if parsed else ""
        frame["source_file"] = csv_path.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=KEY_COLS + METRIC_COLS + ["source_file"])
    out = pd.concat(frames, ignore_index=True)
    for key in KEY_COLS:
        if key not in out.columns:
            out[key] = ""
        out[key] = out[key].fillna("").astype(str)
    out["model"] = out["model"].astype(str)
    out["split"] = out["split"].astype(str)
    return out


def build_comparison(gen: pd.DataFrame, oracle: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    if gen.empty or oracle.empty:
        return pd.DataFrame()
    gen = gen[KEY_COLS + [c for c in METRIC_COLS if c in gen.columns]]
    oracle = oracle[KEY_COLS + [c for c in METRIC_COLS if c in oracle.columns]]
    gen = gen.drop_duplicates(subset=KEY_COLS, keep="last")
    oracle = oracle.drop_duplicates(subset=KEY_COLS, keep="last")
    merged = gen.merge(oracle, on=KEY_COLS, how="inner", suffixes=("_gen", "_oracle"))
    rows = []
    for _, record in merged.iterrows():
        row = {key: record[key] for key in KEY_COLS}
        all_ok = True
        for metric in METRIC_COLS:
            gen_col, or_col = f"{metric}_gen", f"{metric}_oracle"
            if gen_col not in merged.columns or or_col not in merged.columns:
                continue
            if pd.isna(record[gen_col]) or pd.isna(record[or_col]):
                continue
            delta = float(record[gen_col]) - float(record[or_col])
            within = abs(delta) <= tolerance
            row[f"{metric}_gen"] = round(float(record[gen_col]), 4)
            row[f"{metric}_oracle"] = round(float(record[or_col]), 4)
            row[f"{metric}_delta"] = round(delta, 4)
            row[f"{metric}_within_tol"] = within
            all_ok = all_ok and within
        row["all_within_tol"] = all_ok
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(comp: pd.DataFrame, tolerance: float) -> str:
    if comp.empty:
        return "Nenhum par modelo×regime comparável encontrado."
    lines = [
        f"Pares comparáveis: {len(comp)}",
        f"Tolerância: ±{tolerance}",
        f"Dentro da tolerância (todas as métricas): "
        f"{int(comp['all_within_tol'].sum())}/{len(comp)}",
    ]
    for metric in METRIC_COLS:
        col = f"{metric}_within_tol"
        if col in comp.columns:
            valid = comp[col].dropna()
            if len(valid):
                lines.append(f"  {metric}: {int(valid.sum())}/{len(valid)} dentro")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compara métricas geradas vs. oráculo.")
    parser.add_argument("generated", help="Dir de métricas geradas (run).")
    parser.add_argument("--oracle", default="experiments/results/metrics")
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    gen = load_metrics_dir(Path(args.generated))
    oracle = load_metrics_dir(Path(args.oracle))
    comp = build_comparison(gen, oracle, args.tolerance)

    out_path = (
        Path(args.out)
        if args.out
        else Path(args.generated).resolve().parent.parent / "comparison.csv"
    )
    comp.to_csv(out_path, index=False)
    print(summarize(comp, args.tolerance))
    print(f"Detalhe: {out_path}")
    print(f"Gerados: {len(gen)} linhas | Oráculo: {len(oracle)} linhas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Rodar os testes e confirmar sucesso**

```bash
conda run -n pids-repro pytest replication/tests/test_compare_results.py -q
```

Expected: `4 passed`.

- [ ] **Step 6: Commit**

```bash
git add replication/__init__.py replication/compare_results.py replication/tests/test_compare_results.py
git commit -m "feat: comparador de metricas gerado vs oraculo (TDD)"
```

---

### Task 4: Harness de execução (`run_local.py`) — TDD

**Files:**
- Create: `replication/run_local.py`
- Create: `replication/run_local.sh`
- Create: `replication/runs/.gitignore`
- Test: `replication/tests/test_run_local.py`

- [ ] **Step 1: Escrever os testes que falham**

Criar `replication/tests/test_run_local.py`:

```python
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from replication import run_local as rl


def test_build_run_id_format():
    run_id = rl.build_run_id("a", now=0)
    assert run_id.endswith("_A")
    assert run_id.split("_")[0] == "19700101-000000"


def test_copy_reference_tree_excludes(tmp_path):
    src = tmp_path / "src"
    dest = tmp_path / "dest"
    (src / "data").mkdir(parents=True)
    (src / "data" / "x.csv").write_text("a")
    (src / "experiments" / "results").mkdir(parents=True)
    (src / "experiments" / "results" / "r.csv").write_text("b")
    (src / "replication").mkdir()
    (src / "replication" / "keep.txt").write_text("c")
    (src / ".git").mkdir()
    (src / ".git" / "HEAD").write_text("d")
    (src / "docs").mkdir()
    (src / "docs" / "s.md").write_text("e")

    rl.copy_reference_tree(src, dest)

    assert (dest / "data" / "x.csv").exists()
    assert not (dest / "experiments").exists()
    assert not (dest / "replication").exists()
    assert not (dest / ".git").exists()
    assert not (dest / "docs").exists()


def test_notebook_command():
    cmd = rl.notebook_command("04_ablation_study.ipynb")
    assert "nbconvert" in cmd
    assert cmd[-1] == "notebooks/04_ablation_study.ipynb"


def test_build_env_defaults_full_outputs():
    env = rl.build_env("A", minimal=False)
    assert env["SPLIT_TAG"] == "A"
    assert env["WRITE_MINIMAL_OUTPUTS"] == "0"

    env2 = rl.build_env("b", minimal=True)
    assert env2["SPLIT_TAG"] == "B"
    assert env2["WRITE_MINIMAL_OUTPUTS"] == "1"
```

- [ ] **Step 2: Rodar os testes e confirmar falha**

```bash
conda run -n pids-repro pytest replication/tests/test_run_local.py -q
```

Expected: FAIL com `ModuleNotFoundError: No module named 'replication.run_local'`.

- [ ] **Step 3: Implementar `replication/run_local.py`**

```python
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = REPO_ROOT / "replication" / "runs"

CORE_NOTEBOOKS = [
    "04_ablation_study.ipynb",
    "05_semantic_baseline.ipynb",
    "06_hybrid_routing.ipynb",
]
TRANSFORMER_NOTEBOOKS = [
    "07_transformer_baselines.ipynb",
    "08_roberta_ibvs_hybrid.ipynb",
]
EXCLUDED_TOP_LEVEL = {".git", "replication", "docs", "experiments"}


def build_run_id(split_tag: str, now: float | None = None) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(
        now if now is not None else time.time()))
    return f"{stamp}_{split_tag.strip().upper()}"


def copy_reference_tree(src_root: Path, dest_root: Path,
                        excluded: set[str] | None = None) -> None:
    excluded = EXCLUDED_TOP_LEVEL if excluded is None else excluded
    dest_root.mkdir(parents=True, exist_ok=True)
    for item in sorted(src_root.iterdir()):
        if item.name in excluded:
            continue
        target = dest_root / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def notebook_command(notebook: str) -> list[str]:
    return [
        "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
        f"notebooks/{notebook}",
    ]


def build_env(split_tag: str, minimal: bool) -> dict[str, str]:
    env = dict(os.environ)
    env["SPLIT_TAG"] = split_tag.strip().upper()
    env["WRITE_MINIMAL_OUTPUTS"] = "1" if minimal else "0"
    return env


def run_command(cmd: list[str], cwd: Path, env: dict[str, str],
                dry_run: bool = False) -> None:
    print(f"[run] (cwd={cwd}) {' '.join(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def run(args: argparse.Namespace) -> int:
    run_id = build_run_id(args.split_tag)
    dest = Path(args.runs_dir) / run_id
    print(f"Run id:  {run_id}")
    print(f"Run dir: {dest}")

    copy_reference_tree(REPO_ROOT, dest)
    env = build_env(args.split_tag, args.minimal_outputs)

    if not args.skip_pytest:
        run_command([sys.executable, "-m", "pytest", "-q"], dest, env, args.dry_run)

    notebooks = list(CORE_NOTEBOOKS)
    if args.smoke_transformers:
        notebooks += TRANSFORMER_NOTEBOOKS
    for notebook in notebooks:
        run_command(notebook_command(notebook), dest, env, args.dry_run)

    print(f"\nConcluído. Saídas em: {dest / 'experiments' / 'results'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Harness de reprodução local (Fase 1).")
    parser.add_argument("split_tag", help="A, B ou C")
    parser.add_argument("--smoke-transformers", action="store_true",
                        help="Também executar notebooks 07 e 08 (lento em CPU).")
    parser.add_argument("--minimal-outputs", action="store_true",
                        help="Usar WRITE_MINIMAL_OUTPUTS=1 (padrão: outputs completos).")
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    args = parser.parse_args(argv)
    args.split_tag = args.split_tag.strip().upper()
    if args.split_tag not in {"A", "B", "C"}:
        parser.error("split_tag deve ser A, B ou C")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Rodar os testes e confirmar sucesso**

```bash
conda run -n pids-repro pytest replication/tests/test_run_local.py -q
```

Expected: `4 passed`.

- [ ] **Step 5: Criar o wrapper `replication/run_local.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python "${SCRIPT_DIR}/run_local.py" "$@"
```

```bash
chmod +x /home/gl-pereira/Projects/ml_PIDS/replication/run_local.sh
```

- [ ] **Step 6: Criar `replication/runs/.gitignore`**

```
*
!.gitignore
```

- [ ] **Step 7: Verificar o harness com um dry-run**

```bash
conda run -n pids-repro python replication/run_local.py A --dry-run
```

Expected: imprime `Run id: ..._A`, o `Run dir`, e as linhas `[run] ... pytest -q` e `[run] ... notebooks/04_ablation_study.ipynb` etc., sem executar nada.

- [ ] **Step 8: Commit**

```bash
git add replication/run_local.py replication/run_local.sh replication/runs/.gitignore replication/tests/test_run_local.py
git commit -m "feat: harness de reproducao local com isolamento de runs (TDD)"
```

---

### Task 5: Documentação de replicação (README, NOTES, report)

**Files:**
- Create: `replication/README.md`
- Create: `replication/NOTES.md`
- Create: `replication/report.md`

- [ ] **Step 1: Criar `replication/README.md`**

Conteúdo (substituir `<HASH>` pelo hash registrado no Task 1 Step 2):

```markdown
# Reprodução local — Prompt Injection Detection is Regime-Dependent

Tooling de reprodução local (Fase 1) sobre o repositório de referência dos
autores: `https://github.com/TimiAkinrele/ai-jailbreak-framework` (commit `<HASH>`).

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
```

- [ ] **Step 2: Criar `replication/NOTES.md`**

Conteúdo:

```markdown
# Notas de adaptação

- Repo de referência: `TimiAkinrele/ai-jailbreak-framework` (commit `<HASH>`),
  copiado verbatim para a raiz deste repo como base da reprodução.
- Nosso tooling vive só em `replication/`; a árvore do ref não é editada.
- `experiments/results/` é o **oráculo** (tabelas commitadas pelos autores) e
  nunca é sobrescrito: cada execução roda numa cópia isolada em
  `replication/runs/<run_id>/`.
- Escopo local (Fase 1): notebooks `04` (lexical+IBVS), `05` (semântico BGE) e
  `06` (híbridos semânticos). Transformers (`07`,`08`) ficam para a Fase 2
  (cluster APUANA); smoke local é opcional e lento (configs `num_epochs=3`,
  `max_length=256` são fixas no código).
- Rodamos com `WRITE_MINIMAL_OUTPUTS=0` porque os regimes OOD hard-negative e
  std-injection (necessários à conclusão qualitativa central) só são gravados
  nesse modo.
- torch instalado como build **CPU** (máquina local sem GPU).
- `<desvios>`: registrar aqui diferenças de versão de libs, falhas de teste do
  ref, e qualquer divergência numérica relevante.
```

- [ ] **Step 3: Criar `replication/report.md` (esqueleto)**

```markdown
# Relatório de reprodução — Fase 1 (local)

## Ambiente
- Máquina: (preencher: sem GPU, 16 CPUs, RAM)
- Python / torch / transformers / sentence-transformers: (preencher)

## Execução
- Run id: (preencher)
- Comando: (preencher)
- Duração: (preencher)

## Comparação numérica (ID + OOD primário) — run split A
(inserir tabela de `comparison.csv` / saída do resumo)

## Avaliação qualitativa (regimes secundários)
- Hard-negative OOD: (TF-IDF competitivo? preencher)
- Std-injection OOD: (preencher)
- Gap ranking vs. baixo-FPR: (preencher)

## Critérios de sucesso (spec §7)
1. Mecânica (pytest + notebooks 04/05/06): (preencher)
2. Numérico mínimo (±0.05): (preencher)
3. Qualitativo: (preencher)
4. Reprodutibilidade: (preencher)

## Limitações / deferido à Fase 2
- Transformer baselines e RoBERTa+IBVS (notebooks 07/08).
```

- [ ] **Step 4: Commit**

```bash
git add replication/README.md replication/NOTES.md replication/report.md
git commit -m "docs: README, notas de adaptacao e esqueleto de relatorio"
```

---

### Task 6: Executar a Fase 1 (split A)

**Files:**
- Create (gerados): `replication/runs/<run_id>/experiments/results/**`

- [ ] **Step 1: Rodar o harness para o split A**

```bash
cd /home/gl-pereira/Projects/ml_PIDS
conda run -n pids-repro python replication/run_local.py A
```

Expected: `pytest` sem erro e os 3 notebooks executando de ponta a ponta; mensagens `Saved ...` dos notebooks; termina com `Concluído. Saídas em: ...`.

Se algum notebook falhar, inspecione o erro, corrija a causa (ex.: dependência faltando) e re-rode. Registre qualquer correção em `replication/NOTES.md`.

- [ ] **Step 2: Localizar o run id**

```bash
ls -1dt /home/gl-pereira/Projects/ml_PIDS/replication/runs/*_A | head -1
```

Anote o caminho (referido como `<RUN>` nos próximos passos).

- [ ] **Step 3: Confirmar os artefatos gerados**

```bash
ls "$(<RUN>)/experiments/results/metrics"
```

Expected: presentes ao menos `metrics_ablation_splitA.csv`,
`metrics_semantic_splitA.csv`, `metrics_hybrid_splitA.csv`, e arquivos
`*__ood-ood_test_injection*.csv` e `*__ood-ood_test_injection_standard*.csv`
(outputs completos).

- [ ] **Step 4: Nenhum commit de artefatos de run**

Os arquivos em `replication/runs/` são ignorados por `.gitignore`. Não commitar.

---

### Task 7: Comparar, preencher o relatório e avaliar critérios

**Files:**
- Modify: `replication/report.md`
- Create (gerado): `<RUN>/experiments/results/comparison.csv`

- [ ] **Step 1: Rodar a comparação contra o oráculo**

```bash
cd /home/gl-pereira/Projects/ml_PIDS
conda run -n pids-repro python replication/compare_results.py \
  "$(<RUN>)/experiments/results/metrics"
```

Expected: imprime `Pares comparáveis: N`, contagem dentro/fora da tolerância, e
grava `<RUN>/experiments/results/comparison.csv`.

- [ ] **Step 2: Inspecionar deltas fora da tolerância**

```bash
conda run -n pids-repro python - <<'PY'
import pandas as pd, glob, os
run = sorted(glob.glob("replication/runs/*_A"))[-1]
df = pd.read_csv(os.path.join(run, "experiments/results/comparison.csv"))
out = df[~df["all_within_tol"].astype(bool)]
print(out.to_string())
print("fora da tolerância:", len(out), "de", len(df))
PY
```

Anote quais pares/regimes estouraram para o relatório.

- [ ] **Step 3: Avaliação qualitativa dos regimes secundários**

```bash
conda run -n pids-repro python - <<'PY'
import pandas as pd, glob, os
run = sorted(glob.glob("replication/runs/*_A"))[-1]
base = os.path.join(run, "experiments/results/metrics")
for name in ["ood_test_injection", "ood_test_injection_standard"]:
    files = glob.glob(os.path.join(base, f"*__ood-{name}*.csv"))
    print("==", name, "->", len(files), "arquivos")
    for f in files[:3]:
        d = pd.read_csv(f)
        cols = [c for c in ["model","split","eval_track","macro_f1","roc_auc","tpr_at_1pct_fpr"] if c in d.columns]
        print(os.path.basename(f)); print(d[cols].to_string())
PY
```

Verifique: no hard-negative, o TF-IDF (`M1_TFIDF_ONLY`) lidera por macro-F1? Isso sustenta a conclusão qualitativa central.

- [ ] **Step 4: Preencher `replication/report.md`**

Preencher todas as seções `(preencher)` com: ambiente/versões, run id, duração,
tabela de comparação (resumo + lista de deltas fora da tolerância), achados
qualitativos dos regimes secundários, e o veredito de cada critério de sucesso
do spec §7 (cumprido / parcial / não).

- [ ] **Step 5: Commit**

```bash
git add replication/report.md
git commit -m "docs: relatorio de reproducao da Fase 1 (split A)"
```

---

## Self-review (preenchido pelo autor do plano)

**Cobertura do spec:**
- §4 arquitetura (cópia verbatim + `replication/` isolado) → Task 1, Task 4.
- §5.1 ambiente → Task 2.
- §5.2 harness → Task 4.
- §5.3 verificação → Task 3, Task 7.
- §5.4 relatório → Task 5, Task 7.
- §7 critérios → Task 7 Steps 3–4.
- §6 fluxo de dados → Task 6, Task 7.
- Fora de escopo (cluster, `02`, calibração): não há tarefas — correto.

**Consistência de tipos/nomes:**
- `build_run_id`, `copy_reference_tree`, `notebook_command`, `build_env`,
  `run_command`, `run`, `main` definidos na Task 4 e usados nos testes da Task 4.
- `parse_ood_from_name`, `load_metrics_dir`, `build_comparison`, `summarize`,
  `main` definidos na Task 3 e usados nos testes da Task 3.
- `KEY_COLS = [model, split, eval_track, ood_name]` idêntico em código e testes.
- Caminho de saída do comparador (`generated.parent.parent/comparison.csv`)
  compatível com o comando da Task 7 Step 1.

**Placeholders:** apenas marcadores `(preencher)` no relatório e `<HASH>` na
documentação, que são intencionalmente preenchidos durante a execução (Task 1
Step 2 e Task 7). Nenhum placeholder em código.
