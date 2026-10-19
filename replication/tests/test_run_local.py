import argparse
import sys
import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from replication import run_local as rl


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


def _capture(monkeypatch):
    calls = []

    def fake_run_command(cmd, cwd, env, dry_run=False):
        calls.append((cmd, cwd, env))

    monkeypatch.setattr(rl, "run_command", fake_run_command)
    return calls


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
    assert "--ExecutePreprocessor.kernel_name=python3" in cmd
    assert cmd[-1] == "notebooks/04_ablation_study.ipynb"


def test_build_env_defaults_full_outputs():
    env = rl.build_env("A", minimal=False)
    assert env["SPLIT_TAG"] == "A"
    assert env["WRITE_MINIMAL_OUTPUTS"] == "0"

    env2 = rl.build_env("b", minimal=True)
    assert env2["SPLIT_TAG"] == "B"
    assert env2["WRITE_MINIMAL_OUTPUTS"] == "1"


def test_run_dry_run_does_not_copy_and_orders_commands(tmp_path, monkeypatch):
    calls = _capture(monkeypatch)

    rc = rl.run(_make_args(tmp_path))

    assert rc == 0
    assert not any(tmp_path.iterdir())

    cmds = [c[0] for c in calls]
    assert any("pytest" in cmd for cmd in cmds)
    notebooks = [cmd[-1] for cmd in cmds if cmd[-1].endswith(".ipynb")]
    assert notebooks == [
        "notebooks/04_ablation_study.ipynb",
        "notebooks/05_semantic_baseline.ipynb",
        "notebooks/06_hybrid_routing.ipynb",
    ]


def test_run_skip_pytest_omits_pytest_command(tmp_path, monkeypatch):
    calls = _capture(monkeypatch)

    rl.run(_make_args(tmp_path, skip_pytest=True))

    cmds = [c[0] for c in calls]
    assert not any("pytest" in cmd for cmd in cmds)


def test_run_smoke_transformers_appends_notebooks(tmp_path, monkeypatch):
    calls = _capture(monkeypatch)

    rl.run(_make_args(tmp_path, smoke_transformers=True))

    notebooks = [c[0][-1] for c in calls if c[0][-1].endswith(".ipynb")]
    assert notebooks[-2:] == [
        "notebooks/07_transformer_baselines.ipynb",
        "notebooks/08_roberta_ibvs_hybrid.ipynb",
    ]


def test_run_passes_correct_env(tmp_path, monkeypatch):
    calls = _capture(monkeypatch)

    rl.run(_make_args(tmp_path, split_tag="b", minimal_outputs=True))

    assert calls
    for _cmd, _cwd, env in calls:
        assert env["SPLIT_TAG"] == "B"
        assert env["WRITE_MINIMAL_OUTPUTS"] == "1"


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


def test_build_run_id_with_label():
    assert rl.build_run_id("a", now=0, label="light") == \
        "19700101-000000_A_light"


def test_build_run_id_rejects_invalid_label():
    with pytest.raises(ValueError, match="invalid label"):
        rl.build_run_id("A", label="bad label")


def test_run_selects_only_requested_notebooks(tmp_path, monkeypatch, capsys):
    calls = _capture(monkeypatch)

    rc = rl.run(_make_args(tmp_path, notebooks="07,08", label="tx"))

    assert rc == 0
    assert "_A_tx" in capsys.readouterr().out
    notebooks = [c[0][-1] for c in calls if c[0][-1].endswith(".ipynb")]
    assert notebooks == [
        "notebooks/07_transformer_baselines.ipynb",
        "notebooks/08_roberta_ibvs_hybrid.ipynb",
    ]


def test_main_rejects_notebooks_with_smoke(capsys):
    with pytest.raises(SystemExit) as exc:
        rl.main([
            "A", "--notebooks", "07,08", "--smoke-transformers", "--dry-run",
        ])
    assert exc.value.code == 2
    assert "use --notebooks ou --smoke-transformers" in capsys.readouterr().err


def test_main_rejects_unknown_notebook(capsys):
    with pytest.raises(SystemExit) as exc:
        rl.main(["A", "--notebooks", "99", "--dry-run"])
    assert exc.value.code == 2
    assert "unknown notebook" in capsys.readouterr().err


def test_main_rejects_invalid_label(capsys):
    with pytest.raises(SystemExit) as exc:
        rl.main(["A", "--label", "bad label", "--dry-run"])
    assert exc.value.code == 2
    assert "invalid label" in capsys.readouterr().err
