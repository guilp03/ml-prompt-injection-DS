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


def test_load_comparisons_skips_malformed_csv(tmp_path, capsys):
    _write_comparison(tmp_path / "A" / "light" / "run1" / "comparison.csv", "M1", 0.8)
    bad = tmp_path / "A" / "tx" / "run1" / "comparison.csv"
    bad.parent.mkdir(parents=True)
    bad.write_text('a,b\n"unterminated,2')

    frame = agg.load_comparisons(tmp_path)

    assert len(frame) == 1
    assert frame.iloc[0]["run_family"] == "light"
    assert "ignorando" in capsys.readouterr().err


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
