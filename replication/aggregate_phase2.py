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
