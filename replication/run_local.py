from __future__ import annotations

import argparse
import os
import re
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
ALL_NOTEBOOKS = CORE_NOTEBOOKS + TRANSFORMER_NOTEBOOKS
NOTEBOOK_BY_NUMBER = {name[:2]: name for name in ALL_NOTEBOOKS}
EXCLUDED_TOP_LEVEL = {".git", "replication", "docs", "experiments"}


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
        sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
        "--execute", "--inplace",
        "--ExecutePreprocessor.kernel_name=python3",
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
    run_id = build_run_id(args.split_tag, label=args.label)
    notebooks = resolve_notebooks(args.notebooks, args.smoke_transformers)
    dest = Path(args.runs_dir) / run_id
    print(f"Run id:  {run_id}")
    print(f"Run dir: {dest}")

    if not args.dry_run:
        copy_reference_tree(REPO_ROOT, dest)
    env = build_env(args.split_tag, args.minimal_outputs)

    if not args.skip_pytest:
        run_command([sys.executable, "-m", "pytest", "-q"], dest, env, args.dry_run)

    for notebook in notebooks:
        run_command(notebook_command(notebook), dest, env, args.dry_run)

    print(f"\nConcluído. Saídas em: {dest / 'experiments' / 'results'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Harness de reprodução local (Fase 1).")
    parser.add_argument("split_tag", help="A, B ou C")
    parser.add_argument("--smoke-transformers", action="store_true",
                        help="Também executar notebooks 07 e 08 (lento em CPU).")
    parser.add_argument("--notebooks", default=None,
                        help="Lista separada por vírgula, por número ou nome "
                             "completo (ex.: 04,05,06); não combinar com "
                             "--smoke-transformers.")
    parser.add_argument("--label", default=None,
                        help="Sufixo do run_id (ex.: light, tx).")
    parser.add_argument("--minimal-outputs", action="store_true",
                        help="Usar WRITE_MINIMAL_OUTPUTS=1 (padrão: outputs completos).")
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    args = parser.parse_args(argv)
    args.split_tag = args.split_tag.strip().upper()
    if args.split_tag not in {"A", "B", "C"}:
        parser.error("split_tag deve ser A, B ou C")
    if args.notebooks is not None and args.smoke_transformers:
        parser.error("use --notebooks ou --smoke-transformers, não ambos")
    try:
        build_run_id(args.split_tag, label=args.label)
        resolve_notebooks(args.notebooks, args.smoke_transformers)
    except ValueError as exc:
        parser.error(str(exc))
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
