# Fase 2 (APUANA) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adicionar ao repositório o tooling de execução e agregação da Fase 2, permitindo rodar `04→06` e `07→08` nos splits A/B/C via SLURM e comparar os resultados contra o oráculo.

**Architecture:** O harness existente (`replication/run_local.py`) continua sendo o único executor e ganha seleção de notebooks (`--notebooks`) e sufixo de run (`--label`). Uma camada fina em `replication/slurm/` chama o harness, executa `compare_results.py` e valida que a comparação não está vazia. Um agregador (`replication/aggregate_phase2.py`) consolida os `comparison.csv` dos 6 jobs em uma tabela e um resumo.

**Tech Stack:** Python 3.10, pandas, pytest, Bash, SLURM (`sbatch`), Jupyter `nbconvert`.

**Spec de referência:** `docs/superpowers/specs/2026-10-19-fase2-cluster-apuana-design.md`

---

## File Structure

Arquivos criados/modificados por este plano:

- Modify: `replication/run_local.py` — resolução de notebooks e label no `run_id`.
- Modify: `replication/tests/test_run_local.py` — testes das novas opções.
- Create: `replication/aggregate_phase2.py` — consolidação dos `comparison.csv`.
- Create: `replication/tests/test_aggregate_phase2.py` — testes do agregador.
- Create: `replication/slurm/common.sh` — preflight, ambiente, `run_harness`, `run_compare`.
- Create: `replication/slurm/setup_env_cluster.sh` — venv + torch CUDA + lock.
- Create: `replication/slurm/light.sh` — job `04→06`.
- Create: `replication/slurm/transformers.sh` — job `07→08`.
- Create: `replication/slurm/submit_phase2.sh` — submissão dos 6 jobs.
- Create: `replication/tests/test_slurm_scripts.py` — `bash -n` e bit executável.
- Create: `replication/slurm/README.md` — runbook operacional.
- Create: `replication/report-phase2.md` — template do relatório final.
- Modify: `replication/README.md` — ponteiro para o runbook da Fase 2.

Convenções: todos os comandos são executados a partir da raiz do repo
(`/home/gl-pereira/Projects/ml_PIDS`). Commits seguem o estilo existente
(`feat:`, `test:`, `docs:`, `chore:`).

---

### Task 1: Resolver de notebooks em `run_local.py`

**Files:**
- Modify: `replication/run_local.py`
- Test: `replication/tests/test_run_local.py`

- [ ] **Step 1: Escrever os testes que falham**

Adicione `import pytest` ao topo de `replication/tests/test_run_local.py`
(depois de `import sys`) e acrescente ao final do arquivo:

```python
def test_resolve_notebooks_by_number():
    assert rl.resolve_notebooks("04,05,06", False) == rl.CORE_NOTEBOOKS
    assert rl.resolve_notebooks("07,08", False) == rl.TRANSFORMER_NOTEBOOKS


def test_resolve_notebooks_accepts_full_names_and_dedupes():
    assert rl.resolve_notebooks(
        "07_transformer_baselines.ipynb,08", False
    ) == rl.TRANSFORMER_NOTEBOOKS
    assert rl.resolve_notebooks("04,04", False) == [rl.CORE_NOTEBOOKS[0]]


def test_resolve_notebooks_default_and_smoke():
    assert rl.resolve_notebooks(None, False) == rl.CORE_NOTEBOOKS
    assert rl.resolve_notebooks(None, True) == rl.ALL_NOTEBOOKS


def test_resolve_notebooks_rejects_unknown():
    with pytest.raises(ValueError, match="unknown notebook"):
        rl.resolve_notebooks("99", False)
    with pytest.raises(ValueError, match="empty"):
        rl.resolve_notebooks("04,,05", False)
```

- [ ] **Step 2: Rodar os testes para ver falhar**

Run: `pytest replication/tests/test_run_local.py -q`

Expected: FAIL com `AttributeError: module 'replication.run_local' has no attribute 'resolve_notebooks'`.

- [ ] **Step 3: Implementar o resolver**

Em `replication/run_local.py`, adicione `import re` junto aos imports e,
depois de `TRANSFORMER_NOTEBOOKS`, acrescente:

```python
ALL_NOTEBOOKS = CORE_NOTEBOOKS + TRANSFORMER_NOTEBOOKS
NOTEBOOK_BY_NUMBER = {name[:2]: name for name in ALL_NOTEBOOKS}


def resolve_notebooks(selection: str | None,
                      smoke_transformers: bool) -> list[str]:
    if selection is None:
        chosen = list(CORE_NOTEBOOKS)
        if smoke_transformers:
            chosen += TRANSFORMER_NOTEBOOKS
        return chosen

    chosen: list[str] = []
    for raw in selection.split(","):
        token = raw.strip()
        if not token:
            raise ValueError("--notebooks contains an empty entry")
        number = token[:2]
        name = NOTEBOOK_BY_NUMBER.get(number)
        if name is None or token not in {number, name}:
            raise ValueError(f"unknown notebook: {token}")
        if name not in chosen:
            chosen.append(name)
    return chosen
```

- [ ] **Step 4: Rodar os testes para ver passar**

Run: `pytest replication/tests/test_run_local.py -q`

Expected: PASS (todos os testes do arquivo).

- [ ] **Step 5: Commit**

```bash
git add replication/run_local.py replication/tests/test_run_local.py
git commit -m "feat: selecao de notebooks no harness de reproducao"
```

---

### Task 2: Label no `run_id`

**Files:**
- Modify: `replication/run_local.py`
- Test: `replication/tests/test_run_local.py`

- [ ] **Step 1: Escrever os testes que falham**

Acrescente ao final de `replication/tests/test_run_local.py`:

```python
def test_build_run_id_with_label():
    assert rl.build_run_id("a", now=0, label="light") == \
        "19700101-000000_A_light"


def test_build_run_id_rejects_invalid_label():
    with pytest.raises(ValueError, match="invalid label"):
        rl.build_run_id("A", label="bad label")
```

- [ ] **Step 2: Rodar os testes para ver falhar**

Run: `pytest replication/tests/test_run_local.py -q`

Expected: FAIL com `TypeError: build_run_id() got an unexpected keyword argument 'label'`.

- [ ] **Step 3: Implementar o label**

Substitua `build_run_id` em `replication/run_local.py` por:

```python
LABEL_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def build_run_id(split_tag: str, now: float | None = None,
                 label: str | None = None) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime(
        now if now is not None else time.time()))
    run_id = f"{stamp}_{split_tag.strip().upper()}"
    if label:
        cleaned = label.strip()
        if not LABEL_RE.fullmatch(cleaned):
            raise ValueError(f"invalid label: {label!r}")
        run_id = f"{run_id}_{cleaned}"
    return run_id
```

- [ ] **Step 4: Rodar os testes para ver passar**

Run: `pytest replication/tests/test_run_local.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add replication/run_local.py replication/tests/test_run_local.py
git commit -m "feat: label opcional no run_id do harness"
```

---

### Task 3: Ligar `--notebooks` e `--label` na CLI

**Files:**
- Modify: `replication/run_local.py`
- Test: `replication/tests/test_run_local.py`

- [ ] **Step 1: Escrever os testes que falham**

Substitua `_make_args` em `replication/tests/test_run_local.py` por:

```python
def _make_args(tmp_path, **overrides):
    defaults = dict(
        split_tag="A",
        smoke_transformers=False,
        minimal_outputs=False,
        skip_pytest=False,
        dry_run=True,
        runs_dir=str(tmp_path),
        notebooks=None,
        label=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)
```

E acrescente ao final do arquivo:

```python
def test_run_selects_only_requested_notebooks(tmp_path, monkeypatch):
    calls = _capture(monkeypatch)

    rl.run(_make_args(tmp_path, notebooks="07,08", label="tx"))

    notebooks = [c[0][-1] for c in calls if c[0][-1].endswith(".ipynb")]
    assert notebooks == [
        "notebooks/07_transformer_baselines.ipynb",
        "notebooks/08_roberta_ibvs_hybrid.ipynb",
    ]


def test_main_rejects_notebooks_with_smoke(monkeypatch):
    with pytest.raises(SystemExit):
        rl.main([
            "A", "--notebooks", "07,08", "--smoke-transformers", "--dry-run",
        ])
```

- [ ] **Step 2: Rodar os testes para ver falhar**

Run: `pytest replication/tests/test_run_local.py -q`

Expected: FAIL com `AttributeError: 'Namespace' object has no attribute 'notebooks'` em `run` e/ou `SystemExit` não levantado no teste da CLI.

- [ ] **Step 3: Implementar a integração na CLI**

Em `run`, substitua o cálculo do `run_id` e o bloco de notebooks por:

```python
def run(args: argparse.Namespace) -> int:
    run_id = build_run_id(args.split_tag, label=args.label)
    dest = Path(args.runs_dir) / run_id
    print(f"Run id:  {run_id}")
    print(f"Run dir: {dest}")

    if not args.dry_run:
        copy_reference_tree(REPO_ROOT, dest)
    env = build_env(args.split_tag, args.minimal_outputs)

    if not args.skip_pytest:
        run_command([sys.executable, "-m", "pytest", "-q"], dest, env, args.dry_run)

    notebooks = resolve_notebooks(args.notebooks, args.smoke_transformers)
    for notebook in notebooks:
        run_command(notebook_command(notebook), dest, env, args.dry_run)

    print(f"\nConcluído. Saídas em: {dest / 'experiments' / 'results'}")
    return 0
```

Em `main`, acrescente os argumentos depois de `--smoke-transformers`:

```python
    parser.add_argument("--notebooks", default=None,
                        help="Lista separada por vírgula (ex.: 04,05,06).")
    parser.add_argument("--label", default=None,
                        help="Sufixo do run_id (ex.: light, tx).")
```

E substitua o final de `main` (`return run(args)`) por:

```python
    if args.notebooks is not None and args.smoke_transformers:
        parser.error("use --notebooks ou --smoke-transformers, não ambos")
    try:
        return run(args)
    except ValueError as exc:
        parser.error(str(exc))
```

- [ ] **Step 4: Rodar os testes para ver passar**

Run: `pytest replication/tests/test_run_local.py -q`

Expected: PASS.

- [ ] **Step 5: Rodar a suíte de reprodução inteira**

Run: `pytest replication/tests -q`

Expected: PASS (todos os testes de `replication/tests`).

- [ ] **Step 6: Verificar o dry-run manualmente**

Run:

```bash
python replication/run_local.py A --notebooks 07,08 --label tx --dry-run \
  --runs-dir /tmp/pids-dry-run --skip-pytest
```

Expected: imprime `Run id: <timestamp>_A_tx` e dois comandos `nbconvert`,
para `07_transformer_baselines.ipynb` e `08_roberta_ibvs_hybrid.ipynb`.
Nenhum diretório é criado.

- [ ] **Step 7: Commit**

```bash
git add replication/run_local.py replication/tests/test_run_local.py
git commit -m "feat: opcoes --notebooks e --label no harness"
```

---

### Task 4: Agregador — carregamento e deduplicação

**Files:**
- Create: `replication/aggregate_phase2.py`
- Create: `replication/tests/test_aggregate_phase2.py`

- [ ] **Step 1: Escrever os testes que falham**

Crie `replication/tests/test_aggregate_phase2.py` com:

```python
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from replication import aggregate_phase2 as agg


def _write_comparison(path: Path, model: str, macro_f1: float,
                      split_tag: str = "A") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "model": model,
        "split": "ID TEST",
        "eval_track": "ranking",
        "ood_name": "",
        "split_tag": split_tag,
        "macro_f1_gen": macro_f1,
        "macro_f1_oracle": macro_f1,
        "macro_f1_within_tol": True,
        "all_within_tol": True,
    }]).to_csv(path, index=False)
    return path


def test_extract_split_family(tmp_path):
    path = tmp_path / "A" / "light" / "run1" / "comparison.csv"
    path.parent.mkdir(parents=True)
    path.touch()
    assert agg.extract_split_family(path, tmp_path) == ("A", "light")


def test_load_comparisons_adds_path_columns(tmp_path):
    _write_comparison(tmp_path / "A" / "light" / "run1" / "comparison.csv", "M1", 0.8)
    frame = agg.load_comparisons(tmp_path)
    assert len(frame) == 1
    assert frame.iloc[0]["run_split"] == "A"
    assert frame.iloc[0]["run_family"] == "light"
    assert frame.iloc[0]["source_comparison"] == "A/light/run1/comparison.csv"


def test_deduplicate_keeps_last(tmp_path):
    _write_comparison(tmp_path / "A" / "light" / "run1" / "comparison.csv", "M1", 0.7)
    _write_comparison(tmp_path / "A" / "light" / "run2" / "comparison.csv", "M1", 0.9)
    frame = agg.deduplicate(agg.load_comparisons(tmp_path))
    assert len(frame) == 1
    assert frame.iloc[0]["macro_f1_gen"] == 0.9
```

- [ ] **Step 2: Rodar os testes para ver falhar**

Run: `pytest replication/tests/test_aggregate_phase2.py -q`

Expected: FAIL com `ModuleNotFoundError: No module named 'replication.aggregate_phase2'`.

- [ ] **Step 3: Implementar carregamento e deduplicação**

Crie `replication/aggregate_phase2.py` com:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

KEY_COLS = ["model", "split", "eval_track", "ood_name", "split_tag"]
PATH_COLS = ["run_split", "run_family"]
METRIC_COLS = [
    "acc", "macro_f1", "auc_pr", "roc_auc",
    "tpr_at_1pct_fpr", "tpr_at_5pct_fpr", "tpr_at_10pct_fpr",
]


def extract_split_family(comparison_path: Path,
                         runs_root: Path) -> tuple[str, str]:
    try:
        rel = comparison_path.resolve().relative_to(runs_root.resolve())
    except ValueError:
        return "", ""
    parts = rel.parts
    if len(parts) < 3:
        return "", ""
    return parts[0], parts[1]


def load_comparisons(runs_root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(runs_root.rglob("comparison.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            print(f"[warn] ignorando {path}: {exc}", file=sys.stderr)
            continue
        if frame.empty:
            continue
        split, family = extract_split_family(path, runs_root)
        frame["run_split"] = split
        frame["run_family"] = family
        frame["source_comparison"] = str(path.relative_to(runs_root))
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=KEY_COLS + PATH_COLS + ["all_within_tol"])
    return pd.concat(frames, ignore_index=True)


def deduplicate(comp: pd.DataFrame) -> pd.DataFrame:
    if comp.empty:
        return comp
    return comp.drop_duplicates(
        subset=KEY_COLS + PATH_COLS, keep="last"
    ).reset_index(drop=True)
```

- [ ] **Step 4: Rodar os testes para ver passar**

Run: `pytest replication/tests/test_aggregate_phase2.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add replication/aggregate_phase2.py replication/tests/test_aggregate_phase2.py
git commit -m "feat: carregamento e deduplicacao de comparacoes da fase 2"
```

---

### Task 5: Agregador — resumo e CLI

**Files:**
- Modify: `replication/aggregate_phase2.py`
- Modify: `replication/tests/test_aggregate_phase2.py`

- [ ] **Step 1: Escrever os testes que falham**

Acrescente ao final de `replication/tests/test_aggregate_phase2.py`:

```python
def test_build_summary_contains_counts(tmp_path):
    _write_comparison(tmp_path / "A" / "light" / "run1" / "comparison.csv", "M1", 0.8)
    _write_comparison(tmp_path / "A" / "tx" / "run1" / "comparison.csv", "M2", 0.9)
    summary = agg.build_summary(agg.deduplicate(agg.load_comparisons(tmp_path)))
    assert "Pares comparáveis: 2" in summary
    assert "| A | 2 | 2 |" in summary


def test_main_writes_outputs(tmp_path):
    runs = tmp_path / "runs"
    out = tmp_path / "phase2"
    _write_comparison(runs / "A" / "light" / "run1" / "comparison.csv", "M1", 0.8)
    rc = agg.main(["--runs-root", str(runs), "--out-dir", str(out)])
    assert rc == 0
    assert (out / "phase2_comparison_all.csv").exists()
    assert (out / "phase2_summary.md").exists()
    table = pd.read_csv(out / "phase2_comparison_all.csv")
    assert set(["run_split", "run_family"]).issubset(table.columns)


def test_main_empty_returns_nonzero(tmp_path):
    rc = agg.main([
        "--runs-root", str(tmp_path / "empty"),
        "--out-dir", str(tmp_path / "phase2"),
    ])
    assert rc == 1
```

- [ ] **Step 2: Rodar os testes para ver falhar**

Run: `pytest replication/tests/test_aggregate_phase2.py -q`

Expected: FAIL com `AttributeError: module 'replication.aggregate_phase2' has no attribute 'build_summary'`.

- [ ] **Step 3: Implementar resumo e CLI**

Acrescente ao final de `replication/aggregate_phase2.py`:

```python
def build_summary(comp: pd.DataFrame) -> str:
    lines = ["# Resumo da Fase 2", "", f"Pares comparáveis: {len(comp)}", ""]
    if comp.empty:
        return "\n".join(lines)

    within = int(comp["all_within_tol"].sum())
    lines += [
        "## Geral", "",
        f"- Todos dentro da tolerância: {within}/{len(comp)}", "",
        "## Por métrica", "",
        "| Métrica | Dentro | Total |",
        "|---|---|---|",
    ]
    for metric in METRIC_COLS:
        col = f"{metric}_within_tol"
        if col not in comp.columns:
            continue
        valid = comp[col].dropna()
        lines.append(f"| {metric} | {int(valid.sum())} | {len(valid)} |")

    for title, col in (
        ("Por split da execução", "run_split"),
        ("Por família", "run_family"),
    ):
        lines += ["", f"## {title}", "", f"| {col} | Pares | Todos dentro |",
                  "|---|---|---|"]
        for value, group in comp.groupby(col, dropna=False):
            label = value if value else "-"
            lines.append(
                f"| {label} | {len(group)} | "
                f"{int(group['all_within_tol'].sum())} |"
            )

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Consolida comparison.csv da Fase 2.")
    parser.add_argument("--runs-root", default="replication/runs")
    parser.add_argument("--out-dir", default="replication/phase2")
    args = parser.parse_args(argv)

    comp = deduplicate(load_comparisons(Path(args.runs_root)))
    if comp.empty:
        print(
            f"[error] nenhum comparison.csv com linhas em {args.runs_root}",
            file=sys.stderr,
        )
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "phase2_comparison_all.csv"
    summary_path = out_dir / "phase2_summary.md"
    comp.to_csv(table_path, index=False)
    summary_path.write_text(build_summary(comp))

    print(f"Pares: {len(comp)}")
    print(f"Tabela: {table_path}")
    print(f"Resumo: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Rodar os testes para ver passar**

Run: `pytest replication/tests/test_aggregate_phase2.py -q`

Expected: PASS.

- [ ] **Step 5: Verificar a CLI manualmente**

Run: `python replication/aggregate_phase2.py --runs-root /tmp/nao-existe`

Expected: exit code 1 e mensagem `[error] nenhum comparison.csv com linhas em /tmp/nao-existe`.

- [ ] **Step 6: Commit**

```bash
git add replication/aggregate_phase2.py replication/tests/test_aggregate_phase2.py
git commit -m "feat: resumo e CLI do agregador da fase 2"
```

---

### Task 6: `common.sh`, setup do ambiente e teste de sintaxe

**Files:**
- Create: `replication/slurm/common.sh`
- Create: `replication/slurm/setup_env_cluster.sh`
- Create: `replication/tests/test_slurm_scripts.py`

- [ ] **Step 1: Escrever o teste que falha**

Crie `replication/tests/test_slurm_scripts.py` com:

```python
import subprocess
from pathlib import Path

SLURM_DIR = Path(__file__).resolve().parents[1] / "slurm"


def _scripts():
    return sorted(SLURM_DIR.glob("*.sh"))


def test_slurm_scripts_exist():
    assert _scripts(), f"nenhum script em {SLURM_DIR}"


def test_slurm_scripts_pass_bash_syntax_check():
    for script in _scripts():
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_slurm_scripts_are_executable():
    for script in _scripts():
        assert script.stat().st_mode & 0o111, f"{script} nao e executavel"
```

- [ ] **Step 2: Rodar o teste para ver falhar**

Run: `pytest replication/tests/test_slurm_scripts.py -q`

Expected: FAIL com `AssertionError: nenhum script em .../replication/slurm`.

- [ ] **Step 3: Criar `common.sh`**

Crie `replication/slurm/common.sh` com:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

: "${SPLIT_TAG:?defina SPLIT_TAG=A|B|C}"
SPLIT_TAG="${SPLIT_TAG^^}"
case "${SPLIT_TAG}" in
  A|B|C) ;;
  *) echo "[error] SPLIT_TAG invalido: ${SPLIT_TAG}" >&2; exit 2 ;;
esac

: "${SCRATCH:?defina SCRATCH (ex.: /scratch/usuario)}"
VENV="${VENV:-${SCRATCH}/pids-repro}"
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/replication/runs}"
export HF_HOME="${HF_HOME:-${SCRATCH}/hf-cache}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ ! -f "${VENV}/bin/activate" ]]; then
  echo "[error] venv nao encontrado: ${VENV}" >&2
  echo "[hint] rode replication/slurm/setup_env_cluster.sh" >&2
  exit 2
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
PY="${VENV}/bin/python"

processed_file_for_split() {
  case "${SPLIT_TAG}" in
    A) echo "jailbreak_benchmarks_processed_v2.csv" ;;
    B) echo "jailbreak_benchmarks_processed_v2_splitB.csv" ;;
    C) echo "jailbreak_benchmarks_processed_v2_splitC.csv" ;;
  esac
}

preflight_common() {
  local processed="data/processed/$(processed_file_for_split)"
  if [[ ! -f "${REPO_ROOT}/${processed}" ]]; then
    echo "[error] arquivo ausente: ${processed}" >&2
    exit 2
  fi
  if [[ ! -f "${REPO_ROOT}/experiments/results/metrics/metrics_ablation_splitA.csv" ]]; then
    echo "[error] oraculo ausente em experiments/results/metrics" >&2
    exit 2
  fi
  if ! "${PY}" -c "import pandas, sklearn, numpy" >/dev/null 2>&1; then
    echo "[error] dependencias Python ausentes no venv ${VENV}" >&2
    exit 2
  fi
  mkdir -p "${RUNS_ROOT}/${SPLIT_TAG}" "${RUNS_ROOT}/logs"
  echo "[env] split=${SPLIT_TAG} venv=${VENV} hf_home=${HF_HOME} runs=${RUNS_ROOT}"
}

run_compare() {
  local run_dir="$1"
  local out="${run_dir}/comparison.csv"
  "${PY}" "${REPO_ROOT}/replication/compare_results.py" \
    "${run_dir}/experiments/results/metrics" \
    --oracle "${REPO_ROOT}/experiments/results/metrics" \
    --out "${out}"
  "${PY}" - "${out}" <<'PY'
import sys

import pandas as pd

path = sys.argv[1]
frame = pd.read_csv(path)
if frame.empty:
    raise SystemExit(f"[error] comparison.csv vazio: {path}")
print(f"[compare] {len(frame)} pares em {path}")
PY
}

run_harness() {
  local label="$1"
  shift
  local family_runs="${RUNS_ROOT}/${SPLIT_TAG}/${label}"
  local status=0
  mkdir -p "${family_runs}"

  set +e
  "${PY}" "${REPO_ROOT}/replication/run_local.py" "${SPLIT_TAG}" \
    --label "${label}" --runs-dir "${family_runs}" --skip-pytest "$@"
  status=$?
  set -e

  local run_dir
  run_dir="$(find "${family_runs}" -mindepth 1 -maxdepth 1 -type d \
    | sort | tail -n 1)"
  if [[ -z "${run_dir}" ]]; then
    echo "[error] nenhum run dir em ${family_runs}" >&2
    exit 1
  fi
  if ! run_compare "${run_dir}"; then
    if [[ "${status}" -eq 0 ]]; then
      status=1
    fi
  fi
  exit "${status}"
}
```

- [ ] **Step 4: Criar `setup_env_cluster.sh`**

Crie `replication/slurm/setup_env_cluster.sh` com:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

: "${SCRATCH:?defina SCRATCH (ex.: /scratch/usuario)}"
VENV="${VENV:-${SCRATCH}/pids-repro}"

if command -v module >/dev/null 2>&1; then
  module load Python/3.10 || true
fi

if [[ ! -d "${VENV}" ]]; then
  "${PYTHON:-python3}" -m venv "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "${REPO_ROOT}/requirements.txt"

python - <<'PY'
import torch

print(f"[torch] {torch.__version__} cuda_available={torch.cuda.is_available()}")
PY

mkdir -p "${REPO_ROOT}/replication/runs"
python -m pip freeze > "${REPO_ROOT}/replication/runs/env-lock.txt"
echo "[ok] ambiente em ${VENV}"
echo "[ok] lock em ${REPO_ROOT}/replication/runs/env-lock.txt"
echo "[next] export HF_HOME=${SCRATCH}/hf-cache"
```

- [ ] **Step 5: Dar permissão de execução e rodar o teste**

Run:

```bash
chmod +x replication/slurm/common.sh replication/slurm/setup_env_cluster.sh
pytest replication/tests/test_slurm_scripts.py -q
```

Expected: PASS.

- [ ] **Step 6: Checar sintaxe com `bash -n`**

Run: `bash -n replication/slurm/common.sh replication/slurm/setup_env_cluster.sh`

Expected: sem saída e exit code 0.

- [ ] **Step 7: Commit**

```bash
git add replication/slurm/common.sh replication/slurm/setup_env_cluster.sh \
  replication/tests/test_slurm_scripts.py
git commit -m "feat: base dos scripts SLURM e setup do ambiente do cluster"
```

---

### Task 7: Scripts dos jobs `light` e `tx`

**Files:**
- Create: `replication/slurm/light.sh`
- Create: `replication/slurm/transformers.sh`

- [ ] **Step 1: Criar `light.sh`**

Crie `replication/slurm/light.sh` com:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

preflight_common
run_harness light --notebooks 04,05,06
```

- [ ] **Step 2: Criar `transformers.sh`**

Crie `replication/slurm/transformers.sh` com:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

preflight_common
"${PY}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("[error] CUDA indisponivel neste job")
print(f"[gpu] torch={torch.__version__} device={torch.cuda.get_device_name(0)}")
PY

run_harness tx --notebooks 07,08
```

- [ ] **Step 3: Dar permissão de execução e validar sintaxe**

Run:

```bash
chmod +x replication/slurm/light.sh replication/slurm/transformers.sh
pytest replication/tests/test_slurm_scripts.py -q
```

Expected: PASS (os dois scripts passam em `bash -n` e têm bit executável).

- [ ] **Step 4: Validar os comandos gerados sem executar notebooks**

Run:

```bash
SPLIT_TAG=A SCRATCH=/tmp/pids-scratch \
  python replication/run_local.py A --notebooks 04,05,06 --label light \
  --dry-run --runs-dir /tmp/pids-light --skip-pytest
```

Expected: exatamente três comandos `nbconvert`, na ordem `04`, `05`, `06`.

- [ ] **Step 5: Commit**

```bash
git add replication/slurm/light.sh replication/slurm/transformers.sh
git commit -m "feat: jobs SLURM leves e de transformers"
```

---

### Task 8: `submit_phase2.sh`

**Files:**
- Create: `replication/slurm/submit_phase2.sh`

- [ ] **Step 1: Criar o script**

Crie `replication/slurm/submit_phase2.sh` com:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

: "${SCRATCH:?defina SCRATCH (ex.: /scratch/usuario)}"
RUNS_ROOT="${RUNS_ROOT:-${REPO_ROOT}/replication/runs}"
PARTITION_LIGHT="${PARTITION_LIGHT:-short-simple}"
PARTITION_TX="${PARTITION_TX:-short-complex}"
WALLTIME_LIGHT="${WALLTIME_LIGHT:-02:00:00}"
WALLTIME_TX="${WALLTIME_TX:-04:00:00}"
GRES_TX="${GRES_TX:-gpu:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-32G}"

SPLITS=("$@")
if [[ ${#SPLITS[@]} -eq 0 ]]; then
  SPLITS=(A B C)
fi

mkdir -p "${RUNS_ROOT}/logs"

for split in "${SPLITS[@]}"; do
  split="${split^^}"
  case "${split}" in
    A|B|C) ;;
    *) echo "[error] split invalido: ${split}" >&2; exit 2 ;;
  esac

  light_id="$(sbatch --parsable \
    --job-name="pids-light-${split}" \
    --partition="${PARTITION_LIGHT}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEM}" \
    --time="${WALLTIME_LIGHT}" \
    --export="ALL,SPLIT_TAG=${split},RUNS_ROOT=${RUNS_ROOT}" \
    --output="${RUNS_ROOT}/logs/%x-%j.out" \
    "${SCRIPT_DIR}/light.sh")"

  tx_id="$(sbatch --parsable \
    --job-name="pids-tx-${split}" \
    --partition="${PARTITION_TX}" \
    --gres="${GRES_TX}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEM}" \
    --time="${WALLTIME_TX}" \
    --export="ALL,SPLIT_TAG=${split},RUNS_ROOT=${RUNS_ROOT}" \
    --output="${RUNS_ROOT}/logs/%x-%j.out" \
    "${SCRIPT_DIR}/transformers.sh")"

  echo "[submit] split ${split}: light=${light_id} tx=${tx_id}"
done

echo "[info] acompanhe com: squeue -u ${USER}"
echo "[info] agregue ao final: python ${REPO_ROOT}/replication/aggregate_phase2.py"
```

- [ ] **Step 2: Dar permissão e validar sintaxe**

Run:

```bash
chmod +x replication/slurm/submit_phase2.sh
pytest replication/tests/test_slurm_scripts.py -q
```

Expected: PASS.

- [ ] **Step 3: Verificar variáveis default sem submeter**

Run:

```bash
bash -n replication/slurm/submit_phase2.sh
command -v sbatch || echo "sbatch ausente (esperado fora do cluster)"
```

Expected: `bash -n` sem saída; fora do cluster, imprime `sbatch ausente`.

- [ ] **Step 4: Commit**

```bash
git add replication/slurm/submit_phase2.sh
git commit -m "feat: submissao dos seis jobs da fase 2"
```

---

### Task 9: Runbook, template de relatório e ponteiro no README

**Files:**
- Create: `replication/slurm/README.md`
- Create: `replication/report-phase2.md`
- Modify: `replication/README.md`

- [ ] **Step 1: Criar o runbook**

Crie `replication/slurm/README.md` com:

```markdown
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
```

- [ ] **Step 2: Criar o template do relatório**

Crie `replication/report-phase2.md` com:

```markdown
# Relatório de reprodução — Fase 2 (APUANA)

## Ambiente
Preencher com a saída de `replication/runs/env-lock.txt`, versão de torch/CUDA,
partição e GPU usadas.

## Comandos
Registrar os comandos de submissão, os job IDs e o walltime observado de cada job.

## Tabela de comparação
Inserir `replication/phase2/phase2_summary.md` e destacar, por split e família,
os pares fora de ±0.05 em `macro_f1` e as métricas de baixo FPR.

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
```

- [ ] **Step 3: Adicionar ponteiro no README de reprodução**

Em `replication/README.md`, depois da seção "Estrutura", acrescente:

```markdown
## Fase 2 (cluster APUANA)

O roteiro de execução no cluster está em `replication/slurm/README.md`.
O tooling adiciona `--notebooks`/`--label` ao harness, scripts SLURM para os
jobs `light` (`04→06`) e `tx` (`07→08`) e o agregador
`replication/aggregate_phase2.py`.
```

- [ ] **Step 4: Checar os arquivos**

Run:

```bash
ls -l replication/slurm/README.md replication/report-phase2.md
grep -n "Fase 2 (cluster APUANA)" replication/README.md
```

Expected: os dois arquivos existem; o `grep` encontra a seção no README.

- [ ] **Step 5: Commit**

```bash
git add replication/slurm/README.md replication/report-phase2.md \
  replication/README.md
git commit -m "docs: runbook e template de relatorio da fase 2"
```

---

### Task 10: Validação final

**Files:**
- Nenhum arquivo novo; validação de tudo que foi criado.

- [ ] **Step 1: Rodar toda a suíte de testes**

Run: `pytest replication/tests -q`

Expected: PASS (testes existentes + novos).

- [ ] **Step 2: Rodar `bash -n` em todos os scripts**

Run: `for f in replication/slurm/*.sh; do bash -n "$f"; done`

Expected: sem saída e exit code 0.

- [ ] **Step 3: Rodar `shellcheck` se estiver disponível**

Run:

```bash
if command -v shellcheck >/dev/null 2>&1; then
  shellcheck replication/slurm/*.sh
else
  echo "shellcheck ausente; bash -n ja validou a sintaxe"
fi
```

Expected: sem erros ou aviso de ausência do `shellcheck`.

- [ ] **Step 4: Validar os dois dry-runs**

Run:

```bash
python replication/run_local.py A --notebooks 04,05,06 --label light \
  --dry-run --runs-dir /tmp/pids-light --skip-pytest
python replication/run_local.py A --notebooks 07,08 --label tx \
  --dry-run --runs-dir /tmp/pids-tx --skip-pytest
```

Expected: o primeiro imprime três `nbconvert` (04, 05, 06); o segundo imprime
dois (07, 08). Nenhum arquivo é criado em `/tmp/pids-*`.

- [ ] **Step 5: Validar o agregador com dados sintéticos**

Run:

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

path = Path("/tmp/pids-agg/runs/A/light/run1/comparison.csv")
path.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame([{
    "model": "M1", "split": "ID TEST", "eval_track": "ranking",
    "ood_name": "", "split_tag": "A", "all_within_tol": True,
    "macro_f1_within_tol": True,
}]).to_csv(path, index=False)
PY
python replication/aggregate_phase2.py \
  --runs-root /tmp/pids-agg/runs --out-dir /tmp/pids-agg/phase2
cat /tmp/pids-agg/phase2/phase2_summary.md
```

Expected: `Pares: 1`; o sumário contém `Pares comparáveis: 1` e uma linha
`| A | 1 | 1 |`.

- [ ] **Step 6: Revisar o status do git e commitar o que faltar**

Run: `git status --short`

Expected: apenas arquivos intencionais. Se houver algo não commitado:

```bash
git add replication/
git commit -m "chore: validacao final do tooling da fase 2"
```

## Implementação — desvios do plano

Esta seção registra as mudanças feitas durante a execução em relação às tarefas
originais, todas motivadas por revisões de especificação/qualidade.

- **Task 3:** validação de `--notebooks`/`--label` movida para antes de qualquer efeito colateral; `main()` pré-valida e não captura mais `ValueError` de `run()`; testes de CLI cobrindo `unknown notebook`, `invalid label`, exclusão mútua e `_A_tx`.
- **Task 4:** `except` restrito a `OSError`/`ParserError`/`EmptyDataError`/`UnicodeDecodeError`; teste de CSV malformado.
- **Task 5:** guardas para `all_within_tol` e colunas-chave ausentes; resumo construído antes da escrita, UTF-8 explícito e `OSError` retornando 1; `load_comparisons` ignora arquivos fora de `runs/<A|B|C>/<light|tx>/<run_id>/comparison.csv`.
- **Task 6:** `run_compare` remove `comparison.csv` antigo e propaga falhas; `run_harness` valida label, seleciona apenas run dirs novos (snapshot antes/depois) e preserva o status; preflight usa o oráculo do split ativo e verifica `nbconvert`; lock do ambiente é atômico; `setup_env_cluster.sh` valida Python >= 3.10; testes comportamentais com stub de Python/venv.
- **Task 8:** splits são normalizados/validados antes de qualquer `sbatch`; o ID do job leve é impresso imediatamente; falha do job `tx` informa o leve já submetido; `RUNS_ROOT` com vírgula/newline é rejeitado.
- **Task 9:** o cache HF baixa os pesos do BGE e dos dois backbones (não só tokenizers); após o piloto, a execução completa usa `B C`; o venv é ativado antes da agregação; a coleta inclui `env-lock.txt`; `SCRATCH` é exportado; exemplo concreto de retry por `sbatch`.

Commits de implementação: `ca579de`, `755d98e`, `4c81fef`, `0c5dfa4`, `4a3cb17`, `b81ef69`, `9e6b741`, `956fbf6`, `79c55f1`, `5595ce3`, `831672e`, `61b2f52`, `8cbc941`, `9967aee`, `b7f06ba`, `4d4b1e7`, `8c2aad3`, `691c625`, `e4dc4ec`, `a1dabd5`, `fe7fbd0`, `db0b2f5`.
