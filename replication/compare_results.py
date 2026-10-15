from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

METRIC_COLS = [
    "acc", "macro_f1", "auc_pr", "roc_auc",
    "tpr_at_1pct_fpr", "tpr_at_5pct_fpr", "tpr_at_10pct_fpr",
]
KEY_COLS = ["model", "split", "eval_track", "ood_name", "split_tag"]


def _is_skipped(path: Path) -> bool:
    return path.name.startswith("results_table_")


def parse_ood_from_name(path: Path) -> str | None:
    match = re.search(r"__ood-([A-Za-z0-9_]+)", path.stem)
    return match.group(1) if match else None


def parse_split_tag_from_name(path: Path) -> str:
    head = path.stem.split("__")[0]
    match = re.search(r"_split([A-Za-z])", head, re.IGNORECASE)
    return match.group(1).upper() if match else ""


def load_metrics_dir(root: Path) -> pd.DataFrame:
    frames = []
    for csv_path in sorted(root.rglob("*.csv")):
        if _is_skipped(csv_path):
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
        if "split_tag" not in frame.columns:
            frame["split_tag"] = parse_split_tag_from_name(csv_path)
        frame["source_file"] = csv_path.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=KEY_COLS + METRIC_COLS + ["source_file"])
    out = pd.concat(frames, ignore_index=True)
    for key in KEY_COLS:
        if key not in out.columns:
            out[key] = ""
        out[key] = out[key].fillna("").astype(str)
    return out


def _comparison_columns() -> list[str]:
    columns = list(KEY_COLS)
    for metric in METRIC_COLS:
        columns += [
            f"{metric}_gen", f"{metric}_oracle",
            f"{metric}_delta", f"{metric}_within_tol",
        ]
    columns.append("all_within_tol")
    return columns


def build_comparison(gen: pd.DataFrame, oracle: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    if gen.empty or oracle.empty:
        return pd.DataFrame(columns=_comparison_columns())
    gen = gen[KEY_COLS + [c for c in METRIC_COLS if c in gen.columns]]
    oracle = oracle[KEY_COLS + [c for c in METRIC_COLS if c in oracle.columns]]
    gen = gen.drop_duplicates(subset=KEY_COLS, keep="last")
    oracle = oracle.drop_duplicates(subset=KEY_COLS, keep="last")
    merged = gen.merge(oracle, on=KEY_COLS, how="inner", suffixes=("_gen", "_oracle"))
    rows = []
    for _, record in merged.iterrows():
        row = {key: record[key] for key in KEY_COLS}
        all_ok = True
        valid_count = 0
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
            valid_count += 1
        row["all_within_tol"] = valid_count > 0 and all_ok
        rows.append(row)
    return pd.DataFrame(rows, columns=_comparison_columns())


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
        else Path(args.generated).resolve().parent / "comparison.csv"
    )
    comp.to_csv(out_path, index=False)
    print(summarize(comp, args.tolerance))
    print(f"Detalhe: {out_path}")
    print(f"Gerados: {len(gen)} linhas | Oráculo: {len(oracle)} linhas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
