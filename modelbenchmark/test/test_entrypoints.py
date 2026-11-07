"""The project has one executable entry point: ./modelbench."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "modelbench"


@pytest.mark.skipif(not (ROOT / ".venv").is_dir(), reason="requires project .venv")
def test_unified_launcher_help_and_version():
    for args in (["--version"], ["--help"], ["web", "--help"], ["analyze", "--help"]):
        proc = subprocess.run(
            [str(LAUNCHER), *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, (args, proc.stderr)
        assert "modelbench" in (proc.stdout + proc.stderr)


@pytest.mark.skipif(
    not (ROOT / ".venv").is_dir()
    or not (ROOT / "dataset" / "test" / "performance" / "testset.jsonl").is_file(),
    reason="requires project .venv and dataset/test/performance/testset.jsonl on disk",
)
def test_validate_offline_reports_each_performance_scenario_and_total():
    """validate must check every perf_jsonl slice (not only the first dataset name)."""
    proc = subprocess.run(
        [str(LAUNCHER), "validate", "--offline"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
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
        "chinese_long_context",
    ):
        assert f"[ok] dataset perf_jsonl[{scenario}]:" in out
    assert "[ok] total: 480 samples across 8 dataset selection(s)" in out


@pytest.mark.skipif(not (ROOT / ".venv").is_dir(), reason="requires project .venv")
def test_run_dry_run_shows_performance_plan_and_estimate():
    proc = subprocess.run(
        [str(LAUNCHER), "run", "--dry-run", "--models", "minicpm5-1b"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    assert "mode: performance" in out
    assert "scenarios (8):" in out
    assert "knowledge_qa=perf_jsonl[knowledge_qa]" in out
    if (ROOT / "dataset" / "test" / "performance" / "testset.jsonl").is_file():
        assert "estimated measured requests: ~480" in out
        assert "estimated cells: ~24" in out
