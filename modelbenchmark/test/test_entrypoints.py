"""Cross-platform launcher entry points.

POSIX uses the bash launcher ``./modelbench``; Windows uses ``modelbench.cmd``
(and the equivalent ``modelbench.ps1``).  Both forms must reach the same
``modelbench.interfaces.cli`` package, forward arguments verbatim and propagate
the exit code.  ``setup`` is delegated to the stdlib-only ``tool/bootstrap.py``.

The CLI tests are skipped when the local ``.venv`` is missing, exactly as
before the cross-platform port.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
IS_WINDOWS = os.name == "nt"
POSIX_LAUNCHER = ROOT / "modelbench"
CMD_LAUNCHER = ROOT / "modelbench.cmd"
PS1_LAUNCHER = ROOT / "modelbench.ps1"
VENV_PYTHON = (
    ROOT / ".venv" / "Scripts" / "python.exe"
    if IS_WINDOWS
    else ROOT / ".venv" / "bin" / "python"
)
VENV_READY = VENV_PYTHON.is_file()


def _launcher_command(args):
    """Build the platform-native launcher invocation for the current OS."""
    if IS_WINDOWS:
        return [
            os.environ.get("COMSPEC", "cmd.exe"),
            "/c",
            str(CMD_LAUNCHER),
            *args,
        ]
    return [str(POSIX_LAUNCHER), *args]


def _run_launcher(args, timeout=120, env=None):
    return subprocess.run(
        _launcher_command(args),
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def test_posix_launcher_exists_and_is_executable():
    if IS_WINDOWS:
        pytest.skip("POSIX launcher is not used on Windows")
    assert POSIX_LAUNCHER.is_file()
    assert os.access(POSIX_LAUNCHER, os.X_OK)


def test_windows_launchers_are_self_locating_and_forward_to_the_same_cli():
    """cmd/ps1 must resolve the repo from the script, not the current directory."""
    assert CMD_LAUNCHER.is_file()
    assert PS1_LAUNCHER.is_file()

    cmd = CMD_LAUNCHER.read_text(encoding="utf-8")
    assert "%~dp0" in cmd
    assert "tool\\bootstrap.py" in cmd
    assert ".venv\\Scripts\\python.exe" in cmd
    assert "modelbench.interfaces.cli" in cmd
    assert "MB_PYTHON" in cmd
    assert "py -3.12" in cmd
    # A missing venv must be an explicit setup hint with exit code 2.
    assert "environment not ready: run modelbench.cmd setup" in cmd
    assert "exit /b 2" in cmd

    ps1 = PS1_LAUNCHER.read_text(encoding="utf-8")
    assert "$PSScriptRoot" in ps1
    assert "tool\\bootstrap.py" in ps1
    assert ".venv\\Scripts\\python.exe" in ps1
    assert "modelbench.interfaces.cli" in ps1
    assert "MB_PYTHON" in ps1
    assert "$LASTEXITCODE" in ps1
    assert "environment not ready: run modelbench.ps1 setup" in ps1
    # The PowerShell wrapper is a launcher only: no git or other side effects.
    assert not re.search(r"\bgit\b", ps1, re.IGNORECASE)


@pytest.mark.skipif(not VENV_READY, reason="requires project .venv")
def test_unified_launcher_help_and_version():
    for args in (["--version"], ["--help"], ["web", "--help"], ["analyze", "--help"]):
        proc = _run_launcher(args, timeout=30)
        assert proc.returncode == 0, (args, proc.stderr)
        assert "modelbench" in (proc.stdout + proc.stderr)


@pytest.mark.skipif(
    not VENV_READY
    or not (ROOT / "dataset" / "test" / "performance" / "testset.jsonl").is_file(),
    reason="requires project .venv and dataset/test/performance/testset.jsonl on disk",
)
def test_validate_offline_reports_each_performance_scenario_and_total():
    """validate must check every perf_jsonl slice (not only the first dataset name)."""
    proc = _run_launcher(["validate", "--offline"], timeout=180)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    for scenario in (
        "knowledge_qa",
        "instruction_following",
        "long_context",
        "math_logic",
        "coding",
        "general_agent",
        "search_agent",
        "multilingual",
    ):
        assert f"[ok] dataset perf_jsonl[{scenario}]:" in out
    assert "[ok] total: 240 samples across 8 dataset selection(s)" in out


@pytest.mark.skipif(not VENV_READY, reason="requires project .venv")
def test_run_dry_run_shows_performance_plan_and_estimate():
    proc = _run_launcher(["run", "--dry-run", "--models", "minicpm5-1b"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    assert "mode: performance" in out
    assert "scenarios (8):" in out
    assert "knowledge_qa=perf_jsonl[knowledge_qa]" in out
    if (ROOT / "dataset" / "test" / "performance" / "testset.jsonl").is_file():
        assert "estimated measured requests: ~240" in out
        assert "estimated cells: ~24" in out


def test_setup_dry_run_plans_without_installing(tmp_path):
    """`setup` must delegate to tool/bootstrap.py on every platform.

    The dry run is non-destructive: it only resolves an interpreter and prints
    the commands it would run.
    """
    venv = tmp_path / "setup-dry-run-venv"
    env = dict(os.environ)
    # Force the launcher to select the interpreter running this test suite so
    # the test does not depend on a system python3.12 being installed.
    env["PYTHON"] = sys.executable
    env["MB_PYTHON"] = sys.executable
    proc = _run_launcher(["setup", "--dry-run", "--venv", str(venv)], timeout=180, env=env)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    assert "dry-run complete" in out
    assert "-m pip install" in out
    assert ".[dev]" in out
    assert not venv.exists(), "setup --dry-run must not create the venv"
