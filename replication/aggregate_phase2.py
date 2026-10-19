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
        except (OSError, pd.errors.ParserError,
                pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
            print(f"[warn] ignorando {path}: {exc}", file=sys.stderr)
            continue
        if frame.empty:
            continue
        missing = [col for col in KEY_COLS if col not in frame.columns]
        if missing:
            print(
                f"[warn] ignorando {path}: colunas ausentes {missing}",
                file=sys.stderr,
            )
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


def build_summary(comp: pd.DataFrame) -> str:
    lines = ["# Resumo da Fase 2", "", f"Pares comparáveis: {len(comp)}", ""]
    if comp.empty:
        return "\n".join(lines)

    lines += ["## Geral", ""]
    if "all_within_tol" in comp.columns:
        within = int(comp["all_within_tol"].sum())
        lines.append(f"- Todos dentro da tolerância: {within}/{len(comp)}")
    else:
        lines.append("- all_within_tol ausente")
    lines += [
        "", "## Por métrica", "",
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
            label = "-" if pd.isna(value) or value == "" else value
            group_within = (
                int(group["all_within_tol"].sum())
                if "all_within_tol" in group.columns else "n/d"
            )
            lines.append(
                f"| {label} | {len(group)} | {group_within} |"
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

    summary = build_summary(comp)

    out_dir = Path(args.out_dir)
    table_path = out_dir / "phase2_comparison_all.csv"
    summary_path = out_dir / "phase2_summary.md"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        comp.to_csv(table_path, index=False)
        summary_path.write_text(summary, encoding="utf-8")
    except OSError as exc:
        print(f"[error] falha ao escrever resultados: {exc}", file=sys.stderr)
        return 1

    print(f"Pares: {len(comp)}")
    print(f"Tabela: {table_path}")
    print(f"Resumo: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
