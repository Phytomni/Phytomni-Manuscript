import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_wrapper_does_not_sync_or_choose_an_unrelated_interpreter():
    source = (ROOT / "reproduce.sh").read_text()
    assert "exec python -m scripts.reproduce_cli" in source
    assert "exec uv" not in source
    assert 'exec python3' not in source
    assert 'export PHYTOMNI_SAVE' not in source
    assert 'export PYTHONPATH' not in source
    assert 'sys.prefix' in source and 'sys.base_prefix' in source


def invoke(*args):
    environment = os.environ.copy()
    environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment["PATH"]
    return subprocess.run(["bash", "reproduce.sh", *args], cwd=ROOT,
                          env=environment, capture_output=True, text=True, timeout=20)


def test_wrapper_no_arguments_is_help_not_execution():
    before = set((ROOT / "logs/reproduce").glob("*"))
    result = invoke()
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert set((ROOT / "logs/reproduce").glob("*")) == before


def test_wrapper_rejects_retired_smoke_flag_before_execution():
    before = set((ROOT / "logs/reproduce").glob("*"))
    result = invoke("--check")
    assert result.returncode == 2, result.stdout + result.stderr
    assert set((ROOT / "logs/reproduce").glob("*")) == before
