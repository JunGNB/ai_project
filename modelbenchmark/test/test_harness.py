"""Tests for the AI engineering harness (task board + gate checks)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from harness import gate, taskctl


def test_task_board_lifecycle(tmp_path, monkeypatch):
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    monkeypatch.setenv("MB_TASKS_DIR", str(tasks))
    monkeypatch.setenv("MB_STATE_DIR", str(tmp_path / "state"))
    report = tmp_path / "latest.json"
    report.write_text(json.dumps({"summary": {"ok": True}, "git_commit": "deadbeef"}), encoding="utf-8")
    monkeypatch.setenv("MB_GATE_REPORT", str(report))

    assert taskctl.main(["new", "--title", "demo task", "--priority", "P1",
                         "--scope", "src/", "--acceptance", "gate passes"]) == 0
    task_path = tasks / "T-001.json"
    assert task_path.is_file()
    task = json.loads(task_path.read_text(encoding="utf-8"))
    assert task["status"] == "pending" and task["owner"] is None

    assert taskctl.main(["claim", "T-001", "--owner", "tester"]) == 0
    task = json.loads(task_path.read_text(encoding="utf-8"))
    assert task["status"] == "in_progress" and task["owner"] == "tester"

    assert taskctl.main(["complete", "T-001", "--note", "done by test", "--gate"]) == 0
    task = json.loads(task_path.read_text(encoding="utf-8"))
    assert task["status"] == "done" and task["notes"][-1]["note"] == "done by test"

    assert taskctl.main(["reopen", "T-001"]) == 0
    task = json.loads(task_path.read_text(encoding="utf-8"))
    assert task["status"] == "pending" and task["owner"] is None

    journal = (tmp_path / "state" / "journal.jsonl").read_text(encoding="utf-8")
    actions = [json.loads(line)["action"] for line in journal.splitlines()]
    assert actions == ["new", "claim", "complete", "reopen"]


def test_task_board_rejects_invalid_transition(tmp_path, monkeypatch):
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    monkeypatch.setenv("MB_TASKS_DIR", str(tasks))
    monkeypatch.setenv("MB_STATE_DIR", str(tmp_path / "state"))
    assert taskctl.main(["new", "--title", "demo", "--acceptance", "x"]) == 0
    assert taskctl.main(["claim", "T-001", "--owner", "a"]) == 0
    with pytest.raises(SystemExit) as exc:
        taskctl.main(["claim", "T-001", "--owner", "b"])
    assert exc.value.code == 1


def test_gate_read_only_checks_pass():
    ctx = gate.Context(root=gate.ROOT, py=sys.executable)
    for func in (gate.check_docs_links, gate.check_secrets, gate.check_hygiene):
        result = func(ctx)
        assert result.ok, f"{func.__name__}: {result.detail}"


def test_gate_registry_has_performance_checks():
    names = {check.name for check in gate.CHECKS}
    assert {"imports", "compile", "tests", "validate", "perf_plan",
            "docs_links", "secrets", "hygiene"} <= names
    assert "matrix_plans" not in names
    assert not hasattr(gate, "EXPECTED_PLANS")


def test_gate_live_summary_resolves_batch_layout(tmp_path):
    """--live must find output/<batch>_performance_<model>/summary.json."""
    import json

    output = tmp_path / "output"
    run_dir = output / "gate-live-1_performance_minicpm5-1b"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text('{"mode": "performance"}', encoding="utf-8")
    batches = output / "batches"
    batches.mkdir()
    (batches / "gate-live-1.json").write_text(
        json.dumps({"model_runs": [{"run_dir": str(run_dir)}]}), encoding="utf-8"
    )
    assert gate.resolve_live_run_dir(tmp_path, "gate-live-1") == run_dir

    # Glob fallback (no batch metadata) and legacy layout still resolve.
    (output / "gate-live-2" / "summary.json").parent.mkdir(parents=True)
    (output / "gate-live-2" / "summary.json").write_text("{}", encoding="utf-8")
    assert gate.resolve_live_run_dir(tmp_path, "gate-live-2") == output / "gate-live-2"
    glob_dir = output / "gate-live-3_performance_m"
    glob_dir.mkdir()
    (glob_dir / "summary.json").write_text("{}", encoding="utf-8")
    assert gate.resolve_live_run_dir(tmp_path, "gate-live-3") == glob_dir
    assert gate.resolve_live_run_dir(tmp_path, "gate-live-missing") is None


def test_gate_report_written_to_custom_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(gate, "REPORT_LATEST", tmp_path / "latest.json")
    report = gate.run_checks(
        gate.Context(root=gate.ROOT, py=sys.executable),
        [check for check in gate.CHECKS if check.name == "hygiene"],
    )
    path = gate.write_report(report)
    assert path.is_file() and (tmp_path / "latest.json").is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["summary"]["total"] == 1 and data["summary"]["ok"] is True
    assert data["checks"][0]["name"] == "hygiene"
