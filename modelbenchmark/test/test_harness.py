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
    report.write_text(
        json.dumps(
            {
                "summary": {"ok": True, "total": 10, "passed": 10},
                "git_commit": "deadbeef",
                "checks": [
                    {"name": name, "ok": True}
                    for name in sorted(taskctl.REQUIRED_GATE_CHECKS)
                ],
            }
        ),
        encoding="utf-8",
    )
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
            "docs_links", "secrets", "hygiene", "live_smoke"} <= names
    assert "matrix_plans" not in names
    assert not hasattr(gate, "EXPECTED_PLANS")
    # live_smoke is opt-in only (needs a real oMLX server).
    assert "live_smoke" not in {check.name for check in gate.DEFAULT_CHECKS}


# --------------------------------------------------------------------------- #
# gate live-smoke resolution (schema 3.0 flat layout: runs/<run_id>/)
# --------------------------------------------------------------------------- #

RUN_ID = "gate-live-1__performance__minicpm5-1b"


def _write_index(output: Path, run_path: str, run_id: str = RUN_ID) -> None:
    batch_id = run_id.split("__")[0]
    payload = {
        "schema_version": "2.0",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "n_batches": 1,
        "n_runs": 1,
        "n_blobs": {"prompts": 0, "bytes": 0},
        "totals": {"bytes": 0, "files": 0},
        "batches": [
            {
                "batch_id": batch_id,
                "created_at": "2026-01-01T00:00:00+00:00",
                "path": f"batches/{batch_id}.json",
                "methods": ["performance"],
                "models": [],
                "run_ids": [run_id],
                "n_models": 1,
                "n_samples": 1,
                "duration_s": 1.0,
            }
        ],
        "runs": [
            {
                "run_id": run_id,
                "run_path": run_path,
                "batch_id": batch_id,
                "batch_index": 0,
                "batch_models_total": 1,
                "method": "performance",
                "mode": "performance",
                "model": "minicpm5-1b",
                "model_id": "minicpm5-1b",
                "model_slug": "minicpm5-1b",
                "model_size_gb": 1.0,
                "created_at": "2026-01-01T00:00:00+00:00",
                "duration_s": 1.0,
                "status": "ok",
                "cancelled": False,
                "models_selected": ["minicpm5-1b"],
                "models_measured": ["minicpm5-1b"],
                "n_models_selected": 1,
                "n_models_measured": 1,
                "n_skipped": 0,
                "skipped": [],
                "memory_errors": 0,
                "n_samples": 1,
                "n_records": 1,
                "n_cells": 1,
                "datasets": ["mmlu"],
                "bytes_total": 1,
                "bytes_total_human": "1 B",
                "files": {},
                "legacy": False,
            }
        ],
    }
    (output / "index.json").write_text(json.dumps(payload), encoding="utf-8")


def _make_run_dir(root: Path, *, batch_id: str = "gate-live-1") -> Path:
    output = root / "output"
    # Flat layout: the directory name is the run id.
    run_dir = output / "runs" / f"{batch_id}__performance__minicpm5-1b"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "schema_version": "3.0",
                "run_id": f"{batch_id}__performance__minicpm5-1b",
                "batch_id": batch_id,
                "method": "performance",
                "model": {
                    "id": "minicpm5-1b",
                    "alias": "minicpm5-1b",
                    "model_slug": "minicpm5-1b",
                },
                "created_at": "2026-01-01T00:00:00+00:00",
                "env": {"hardware": {"chip": "M4"}, "omlx": {}, "model_env": {}},
                "outcome": {"status": "ok", "n_measured": 1, "n_selected": 1,
                            "n_records": 1, "n_samples": 1, "skipped": []},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "aggregates.json").write_text(
        json.dumps({"schema_version": "3.0", "overall": [], "cells": []}),
        encoding="utf-8",
    )
    record = {
        "phase": "measured",
        "status": "ok",
        "cache": {"cleared_before_request": True, "cached_tokens": 0},
        "metrics": {"input_context_tokens": 100},
        "score": None,
    }
    (run_dir / "records.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    return run_dir


def test_resolve_live_run_dir_prefers_index(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    output = tmp_path / "output"
    _write_index(output, run_dir.relative_to(output).as_posix())
    assert gate.resolve_live_run_dir(tmp_path, "gate-live-1") == run_dir


def test_resolve_live_run_dir_globs_when_index_missing(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    output = tmp_path / "output"
    # No index.json at all: the run was just written and the catalog is stale.
    assert gate.resolve_live_run_dir(tmp_path, "gate-live-1") == run_dir
    # A full single-segment run id resolves through the same fallback.
    assert gate.resolve_live_run_dir(
        tmp_path, "gate-live-1__performance__minicpm5-1b"
    ) == run_dir
    assert gate.resolve_live_run_dir(tmp_path, "gate-live-missing") is None


def test_resolve_live_run_dir_ignores_retired_layouts(tmp_path):
    # 1) schema 2.x bundle at output/<run_id>/ (pre-nesting layout)
    legacy = tmp_path / "output" / "gate-live-legacy"
    legacy.mkdir(parents=True)
    (legacy / "summary.json").write_text("{}", encoding="utf-8")
    assert gate.resolve_live_run_dir(tmp_path, "gate-live-legacy") is None
    # 2) the earlier nested schema-3.0 layout runs/<YYYY-MM>/<batch>/<method>/<slug>/
    nested = (
        tmp_path
        / "output"
        / "runs"
        / "2026-11"
        / "gate-live-nested"
        / "performance"
        / "minicpm5-1b"
    )
    nested.mkdir(parents=True)
    (nested / "run.json").write_text(
        json.dumps({"run_id": "gate-live-nested__performance__minicpm5-1b",
                    "batch_id": "gate-live-nested"}),
        encoding="utf-8",
    )
    assert gate.resolve_live_run_dir(tmp_path, "gate-live-nested") is None


# --------------------------------------------------------------------------- #
# gate live-smoke evidence (no live oMLX needed)
# --------------------------------------------------------------------------- #


def test_live_smoke_evidence_accepts_schema_3_bundle(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    evidence = gate._live_smoke_evidence(run_dir)
    assert evidence["ok"] is True, evidence["detail"]
    assert evidence["data"]["shape_ok"] is True
    assert evidence["data"]["env_ok"] is True
    assert gate._live_smoke_evidence(None)["ok"] is False
    assert gate._live_smoke_evidence(tmp_path / "nowhere")["ok"] is False


def test_live_smoke_evidence_rejects_shape_and_contract_violations(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    # 1) outcome.n_measured no longer matches the measured records
    run_json = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    run_json["outcome"]["n_measured"] = 7
    (run_dir / "run.json").write_text(json.dumps(run_json), encoding="utf-8")
    evidence = gate._live_smoke_evidence(run_dir)
    assert evidence["ok"] is False and evidence["data"]["shape_ok"] is False

    # 2) aggregates.json lost its overall list
    run_json["outcome"]["n_measured"] = 1
    (run_dir / "run.json").write_text(json.dumps(run_json), encoding="utf-8")
    (run_dir / "aggregates.json").write_text(json.dumps({}), encoding="utf-8")
    evidence = gate._live_smoke_evidence(run_dir)
    assert evidence["ok"] is False and evidence["data"]["shape_ok"] is False

    # 3) environment snapshot missing from run.json
    (run_dir / "aggregates.json").write_text(
        json.dumps({"overall": []}), encoding="utf-8"
    )
    run_json["env"] = {}
    (run_dir / "run.json").write_text(json.dumps(run_json), encoding="utf-8")
    evidence = gate._live_smoke_evidence(run_dir)
    assert evidence["ok"] is False and evidence["data"]["env_ok"] is False

    # 4) a cold-cache violation (cache hit on a measured request)
    run_json["env"] = {"hardware": {}}
    (run_dir / "run.json").write_text(json.dumps(run_json), encoding="utf-8")
    bad = {
        "phase": "measured",
        "status": "ok",
        "cache": {"cleared_before_request": True, "cached_tokens": 42},
        "metrics": {"input_context_tokens": 100},
        "score": None,
    }
    (run_dir / "records.jsonl").write_text(json.dumps(bad) + "\n", encoding="utf-8")
    evidence = gate._live_smoke_evidence(run_dir)
    assert evidence["ok"] is False

    # 5) a scored record (performance mode never scores)
    ok_record = {
        "phase": "measured",
        "status": "ok",
        "cache": {"cleared_before_request": True, "cached_tokens": 0},
        "metrics": {"input_context_tokens": 100},
        "score": {"correct": True},
    }
    (run_dir / "records.jsonl").write_text(
        json.dumps(ok_record) + "\n", encoding="utf-8"
    )
    evidence = gate._live_smoke_evidence(run_dir)
    assert evidence["ok"] is False


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


def test_complete_gate_rejects_partial_report(tmp_path, monkeypatch):
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    monkeypatch.setenv("MB_TASKS_DIR", str(tasks))
    monkeypatch.setenv("MB_STATE_DIR", str(tmp_path / "state"))
    report = tmp_path / "latest.json"
    report.write_text(
        json.dumps(
            {
                "summary": {"ok": True, "total": 2, "passed": 2},
                "checks": [{"name": "docs_links", "ok": True}, {"name": "docs_contract", "ok": True}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MB_GATE_REPORT", str(report))
    assert taskctl.main(["new", "--title", "demo", "--priority", "P1"]) == 0
    assert taskctl.main(["claim", "T-001", "--owner", "tester"]) == 0
    with pytest.raises(SystemExit) as excinfo:
        taskctl.main(["complete", "T-001", "--note", "partial gate", "--gate"])
    assert excinfo.value.code == 1
