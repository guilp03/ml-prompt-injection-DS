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
