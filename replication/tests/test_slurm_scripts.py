import os
import subprocess
from pathlib import Path

SLURM_DIR = Path(__file__).resolve().parents[1] / "slurm"
REPO_ROOT = Path(__file__).resolve().parents[2]
COMMON = REPO_ROOT / "replication" / "slurm" / "common.sh"


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


def _write_stub_python(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("""#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == "-c" ]]; then exit 0; fi
if [[ "$*" == *"run_local.py"* ]]; then
  runs_dir=""
  prev=""
  for arg in "$@"; do
    if [[ "$prev" == "--runs-dir" ]]; then runs_dir="$arg"; fi
    prev="$arg"
  done
  if [[ "${STUB_HARNESS_STATUS:-0}" != "0" ]]; then
    if [[ "${STUB_CREATE_ON_FAILURE:-0}" == "1" ]]; then
      mkdir -p "$runs_dir/run-${STUB_RUN_ID:-1}"
    fi
    exit "${STUB_HARNESS_STATUS}"
  fi
  mkdir -p "$runs_dir/run-${STUB_RUN_ID:-1}"
  exit 0
fi
if [[ "$*" == *"compare_results.py"* ]]; then
  out=""
  prev=""
  for arg in "$@"; do
    if [[ "$prev" == "--out" ]]; then out="$arg"; fi
    prev="$arg"
  done
  if [[ "${STUB_COMPARE_STATUS:-0}" != "0" ]]; then exit "${STUB_COMPARE_STATUS}"; fi
  if [[ "${STUB_WRITE_COMPARISON:-1}" == "1" ]]; then
    mkdir -p "$(dirname "$out")"
    printf 'model\\nM1\\n' > "$out"
  fi
  exit 0
fi
if [[ "$1" == "-" ]]; then
  path="$2"
  if [[ ! -s "$path" ]]; then exit 1; fi
  lines="$(wc -l < "$path")"
  if [[ "$lines" -le 1 ]]; then exit 1; fi
  exit 0
fi
exit 0
""", encoding="utf-8")
    path.chmod(0o755)


def _run_common(tmp_path, body, **env):
    venv = tmp_path / "venv"
    _write_stub_python(venv / "bin" / "python")
    (venv / "bin" / "activate").write_text("", encoding="utf-8")
    full_env = {
        **os.environ,
        "SPLIT_TAG": "A",
        "SCRATCH": str(tmp_path),
        "VENV": str(venv),
        "RUNS_ROOT": str(tmp_path / "runs"),
        **env,
    }
    return subprocess.run(
        ["bash", "-c", f'source "{COMMON}"\n{body}\n'],
        text=True, capture_output=True, env=full_env,
    )


def test_run_compare_propagates_compare_failure_with_stale_output(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "comparison.csv").write_text("model\nM1\n", encoding="utf-8")
    result = _run_common(
        tmp_path, f'run_compare "{run_dir}"', STUB_COMPARE_STATUS="7",
    )
    assert result.returncode == 7
    assert not (run_dir / "comparison.csv").exists()


def test_run_compare_rejects_empty_output(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result = _run_common(
        tmp_path, f'run_compare "{run_dir}"', STUB_WRITE_COMPARISON="0",
    )
    assert result.returncode != 0


def test_run_harness_preserves_failure_status(tmp_path):
    result = _run_common(
        tmp_path, "run_harness light --notebooks 04",
        STUB_HARNESS_STATUS="7",
    )
    assert result.returncode == 7


def test_run_harness_preserves_status_when_compare_succeeds(tmp_path):
    result = _run_common(
        tmp_path, "run_harness light --notebooks 04",
        STUB_HARNESS_STATUS="5", STUB_CREATE_ON_FAILURE="1",
    )
    assert result.returncode == 5


def test_run_harness_success_runs_compare(tmp_path):
    result = _run_common(
        tmp_path, "run_harness light --notebooks 04",
        STUB_HARNESS_STATUS="0", STUB_WRITE_COMPARISON="1",
    )
    run_dir = tmp_path / "runs" / "A" / "light" / "run-1"
    assert result.returncode == 0
    assert (run_dir / "comparison.csv").exists()


def test_run_harness_fails_when_compare_fails(tmp_path):
    result = _run_common(
        tmp_path, "run_harness light --notebooks 04",
        STUB_HARNESS_STATUS="0", STUB_COMPARE_STATUS="7",
    )
    assert result.returncode != 0


def test_run_harness_does_not_compare_stale_run(tmp_path):
    stale = tmp_path / "runs" / "A" / "light" / "run-old"
    stale.mkdir(parents=True)
    marker = stale / "comparison.csv"
    marker.write_text("model\nSTALE\n", encoding="utf-8")
    result = _run_common(
        tmp_path, "run_harness light --notebooks 04",
        STUB_HARNESS_STATUS="2",
    )
    assert result.returncode == 2
    assert marker.read_text(encoding="utf-8") == "model\nSTALE\n"
