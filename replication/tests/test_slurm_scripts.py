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
