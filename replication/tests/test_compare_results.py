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


def test_parse_split_tag_from_name():
    assert cr.parse_split_tag_from_name(Path("metrics_ablation_splitB.csv")) == "B"
    assert cr.parse_split_tag_from_name(Path("metrics_ablation_splitb.csv")) == "B"
    assert cr.parse_split_tag_from_name(Path("metrics_ablation.csv")) == ""


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


def test_different_splits_are_not_confused(tmp_path):
    gen_dir = tmp_path / "gen"
    oracle_dir = tmp_path / "oracle"
    _write_metrics(gen_dir, "M1", "TEST", "ranking", 0.80, 0.90,
                   filename="metrics_ablation_splitA.csv")
    _write_metrics(oracle_dir, "M1", "TEST", "ranking", 0.80, 0.90,
                   filename="metrics_ablation_splitA.csv")
    _write_metrics(oracle_dir, "M1", "TEST", "ranking", 0.50, 0.90,
                   filename="metrics_ablation_splitC.csv")
    comp = cr.build_comparison(
        cr.load_metrics_dir(gen_dir), cr.load_metrics_dir(oracle_dir), tolerance=0.05
    )
    assert len(comp) == 1
    assert comp.iloc[0]["split_tag"] == "A"
    assert abs(float(comp.iloc[0]["macro_f1_delta"])) < 1e-9


def test_all_within_tol_false_when_no_valid_metric(tmp_path):
    gen_dir = tmp_path / "gen"
    oracle_dir = tmp_path / "oracle"
    for d in (gen_dir, oracle_dir):
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{
            "model": "M1", "split": "TEST", "eval_track": "ranking",
            "acc": float("nan"), "macro_f1": float("nan"), "roc_auc": float("nan"),
            "auc_pr": float("nan"), "tpr_at_1pct_fpr": float("nan"),
            "tpr_at_5pct_fpr": float("nan"), "tpr_at_10pct_fpr": float("nan"),
        }]).to_csv(d / "metrics_ablation_splitA.csv", index=False)
    comp = cr.build_comparison(
        cr.load_metrics_dir(gen_dir), cr.load_metrics_dir(oracle_dir), tolerance=0.05
    )
    assert len(comp) == 1
    assert bool(comp.iloc[0]["all_within_tol"]) is False


def test_build_comparison_empty_has_expected_columns():
    comp = cr.build_comparison(pd.DataFrame(), pd.DataFrame(), tolerance=0.05)
    assert "all_within_tol" in comp.columns
    assert "model" in comp.columns


def test_main_default_out_path_is_results_dir(tmp_path):
    run = tmp_path / "run"
    metrics_dir = run / "experiments" / "results" / "metrics"
    _write_metrics(metrics_dir, "M1", "TEST", "ranking", 0.80, 0.90)
    rc = cr.main([str(metrics_dir), "--oracle", str(metrics_dir)])
    assert rc == 0
    assert (run / "experiments" / "results" / "comparison.csv").exists()
