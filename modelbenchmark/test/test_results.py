"""Tests for run catalog, analysis projections and web result helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modelbench.adapters.config.testsets import TestSet as TSet
from modelbench.adapters.config.testsets import TestSetScenario as TScen
from modelbench.adapters.storage import catalog
from modelbench.adapters.storage.writer import RunWriter
from modelbench.domain.models import (
    ChatMessage,
    PerfMetrics,
    Phase,
    RawResponse,
    RecordStatus,
    RequestSpec,
    RunManifest,
    SampleRecord,
)
from modelbench.interfaces.web import server as web


def _record(run_id: str = "r1", sample_id: str = "s1") -> SampleRecord:
    return SampleRecord(
        run_id=run_id,
        model="m1",
        scenario="knowledge_qa",
        dataset="mmlu",
        sample_id=sample_id,
        repeat=0,
        phase=Phase.MEASURED,
        status=RecordStatus.OK,
        request=RequestSpec(
            model="m1",
            messages=[ChatMessage(role="user", content="hi")],
            max_tokens=16,
            actual_prompt_tokens=100,
        ),
        response=RawResponse(text="x"),
        metrics=PerfMetrics(
            ttft_ms=100.0,
            gen_tps=20.0,
            peak_rss_mb=1024.0,
            peak_memory_gb=1.0,
            input_context_tokens=100,
            input_context_chars=200,
        ),
    )


def _write_bundle(root: Path, run_id: str = "run-a", legacy: bool = False) -> Path:
    writer = RunWriter(root, run_id=run_id)
    manifest = RunManifest(
        run_id=run_id,
        created_at="2026-01-01T00:00:00Z",
        project_version="0.3.0",
        models={"selected": [{"id": "m1", "alias": "m1"}]},
    )
    summary = {
        "schema_version": "1.0" if legacy else "2.1",
        "run_id": run_id,
        "mode": "performance",
        "generated_at": "2026-01-01T00:00:00Z",
        "n_records": 2,
        "n_samples": 2,
        "n_cells": 1,
        "overall": [] if legacy else [
            {"model": "m1", "n_samples": 2, "n_valid": 2, "perf": {"ttft_ms": {"mean": 100.0}}}
        ],
        "cells": [
            {"model": "m1", "dataset": "mmlu", "n_samples": 2, "n_repeats": 1, "n_valid": 2}
        ],
        "samples": [] if legacy else [{"model": "m1", "dataset": "mmlu", "sample_id": "s1"}],
        "skipped": [],
        "duration_s": 1.0,
    }
    writer.initialize(manifest, {"upload_fields": {"chip_name": "M4"}}, {"selected": []}, {})
    records = [_record(run_id, "s1"), _record(run_id, "s2")]
    for record in records:
        writer.write_record(record)
    writer.finalize(records, perf={"max_tokens": 128}, run_config={"repeats": 1})
    if legacy:
        (Path(root) / run_id / "summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
    return Path(root) / run_id


def test_writer_writes_analysis_and_catalog(tmp_path):
    run_dir = _write_bundle(tmp_path, "run-a")
    for name in ("run.json", "samples_long.csv", "environment.json", "README.md"):
        assert (run_dir / "analysis" / name).is_file(), name
    index_path = tmp_path / "index.json"
    assert index_path.is_file()
    index = json.loads(index_path.read_text())
    entry = index["runs"][0]
    assert entry["run_id"] == "run-a"
    assert entry["models_selected"] == ["m1"]
    assert entry["models_measured"] == ["m1"]
    assert entry["n_samples"] == 2
    assert entry["analysis"]
    long_rows = (run_dir / "analysis" / "samples_long.csv").read_text().splitlines()
    assert len(long_rows) == 1 + 2 * 12  # two samples x 12 metrics
    run_meta = json.loads((run_dir / "analysis" / "run.json").read_text())
    assert run_meta["run_id"] == "run-a"
    env = json.loads((run_dir / "analysis" / "environment.json").read_text())
    assert env["hardware"]["upload_fields"]["chip_name"] == "M4"


def test_catalog_rebuild_and_legacy_entry(tmp_path):
    _write_bundle(tmp_path, "run-new")
    _write_bundle(tmp_path, "run-old", legacy=True)
    index = catalog.rebuild_index(tmp_path)
    assert index["n_runs"] == 2
    entries = {entry["run_id"]: entry for entry in index["runs"]}
    assert entries["run-new"]["legacy"] is False
    assert entries["run-old"]["legacy"] is True
    assert entries["run-old"]["models_measured"] == ["m1"]
    # rebuild from disk equals the on-disk index
    assert catalog.load_index(tmp_path)["n_runs"] == 2
    catalog.upsert_run(tmp_path, tmp_path / "run-new",
                       json.loads((tmp_path / "run-new" / "summary.json").read_text()),
                       json.loads((tmp_path / "run-new" / "manifest.json").read_text()))
    assert catalog.load_index(tmp_path)["n_runs"] == 2


def test_resolve_run_file_rejects_traversal(tmp_path):
    run_dir = _write_bundle(tmp_path, "run-a")
    assert catalog.resolve_run_file(run_dir, "summary.json").is_file()
    with pytest.raises(ValueError):
        catalog.resolve_run_file(run_dir, "../manifest.json")
    with pytest.raises(FileNotFoundError):
        catalog.resolve_run_file(run_dir, "missing.txt")


def test_run_file_listing_includes_analysis(tmp_path):
    run_dir = _write_bundle(tmp_path, "run-a")
    names = {item["name"] for item in catalog.run_file_listing(run_dir)}
    assert {"summary.json", "samples.csv", "report.md", "records.jsonl"} <= names
    assert "analysis/samples_long.csv" in names


def test_scenario_samples_endpoint_uses_cache(monkeypatch, tmp_path):
    from modelbench.adapters.config.settings import load_config

    cfg = load_config()
    cfg.datasets.raw_dir = str(tmp_path / "raw")
    cfg.datasets.processed_dir = str(tmp_path / "processed")
    cfg.scenarios["knowledge_qa"].dataset = "synthetic"
    cfg.scenarios["knowledge_qa"].params = {"dataset_params": {"count": 3}}
    testset = TSet(
        name="synthetic_set",
        scenarios={"knowledge_qa": TScen()},
    )
    monkeypatch.setattr(web, "load_config", lambda: cfg)
    monkeypatch.setattr(web, "load_testsets", lambda *args, **kwargs: {"synthetic_set": testset})
    web._SCENARIO_SAMPLE_CACHE.clear()
    first = web._scenario_samples("synthetic_set", "knowledge_qa", 0, 2)
    assert first["total"] == 3
    assert first["returned"] == 2
    assert first["samples"][0]["sample_id"] == "synthetic-0000"
    second = web._scenario_samples("synthetic_set", "knowledge_qa", 2, 10)
    assert [sample["sample_id"] for sample in second["samples"]] == ["synthetic-0002"]
    with pytest.raises(ValueError):
        web._scenario_samples("synthetic_set", "missing", 0, 1)


def test_open_in_file_manager_uses_platform_command(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)

    monkeypatch.setattr(web.subprocess, "run", fake_run)
    monkeypatch.setattr(web.sys, "platform", "darwin")
    web._open_in_file_manager(Path("/tmp/run"))
    assert calls == [["open", "/tmp/run"]]


def test_web_run_detail_and_catalog(tmp_path, monkeypatch):
    _write_bundle(tmp_path, "run-a")
    monkeypatch.setattr(web, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(web, "_CATALOG_CACHE", {"at": 0.0, "data": None})
    detail = web._run_detail("run-a")
    assert detail["summary"]["run_id"] == "run-a"
    assert detail["files"]
    assert detail["models_measured"] == ["m1"]
    overview = web._server_overview()
    assert "strategies" not in overview
    assert overview["catalog"]["n_runs"] == 1
    assert overview["runs"][0]["run_id"] == "run-a"


def test_schema_2_empty_run_is_not_legacy(tmp_path):
    run_dir = tmp_path / "empty-run"
    writer = RunWriter(tmp_path, run_id="empty-run")
    manifest = RunManifest(
        run_id="empty-run",
        created_at="2026-01-01T00:00:00Z",
        project_version="0.3.0",
        models={"selected": [{"id": "m1", "alias": "m1"}]},
    )
    writer.initialize(manifest, {}, {"selected": []}, {})
    writer.finalize([], perf={"max_tokens": 128}, run_config={"repeats": 1})
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["schema_version"] == "2.1"
    assert summary["overall"] == []
    entry = catalog.run_entry(run_dir, summary)
    assert entry["legacy"] is False
    assert entry["models_measured"] == []


def test_ensure_index_picks_up_new_runs(tmp_path):
    _write_bundle(tmp_path, "run-a")
    assert catalog.ensure_index(tmp_path)["n_runs"] == 1
    _write_bundle(tmp_path, "run-b")
    # make the new run look newer than the index even on coarse mtimes
    import os
    import time

    later = time.time() + 5
    os.utime(tmp_path / "run-b", (later, later))
    index = catalog.ensure_index(tmp_path)
    assert index["n_runs"] == 2
    assert {entry["run_id"] for entry in index["runs"]} == {"run-a", "run-b"}


def test_web_detail_surfaces_warnings(tmp_path, monkeypatch):
    run_dir = tmp_path / "run-warn"
    writer = RunWriter(tmp_path, run_id="run-warn")
    manifest = RunManifest(
        run_id="run-warn",
        created_at="2026-01-01T00:00:00Z",
        project_version="0.3.0",
    )
    writer.initialize(manifest, {}, {"selected": []}, {})
    writer.finalize([], perf={}, run_config={}, extra={"warnings": ["a warning"]})
    monkeypatch.setattr(web, "OUTPUT_DIR", tmp_path)
    detail = web._run_detail("run-warn")
    assert detail["warnings"] == ["a warning"]


def test_sample_worker_preview_and_large_isolation(monkeypatch, tmp_path):
    from modelbench.interfaces.web import sample_worker

    preview = sample_worker.build_preview(
        str(tmp_path / "raw"), str(tmp_path / "processed"), "synthetic", {"count": 2}
    )
    assert preview["total"] == 2
    assert [row["sample_id"] for row in preview["samples"]] == ["synthetic-0000", "synthetic-0001"]

    calls = []
    monkeypatch.setattr(web, "_dataset_raw_bytes", lambda raw_dir, dataset: 999 * 1024 * 1024)
    monkeypatch.setattr(
        web,
        "_preview_via_worker",
        lambda raw, processed, dataset, params, testset_dir=None: calls.append(dataset)
        or {
            "dataset": dataset,
            "total": 0,
            "samples": [],
        },
    )
    result = web._build_preview(
        str(tmp_path / "raw"), str(tmp_path / "processed"), "big", {}
    )
    assert calls == ["big"]
    assert result["samples"] == []
    assert result["dataset"] == "big"


def test_excel_sanitizes_formula_like_text():
    """Model output is untrusted; Excel must not evaluate it as a formula."""
    from modelbench.adapters.storage.excel_report import _sanitize_excel_text

    assert _sanitize_excel_text("=1+1") == "'=1+1"
    assert _sanitize_excel_text("+SUM(A1)") == "'+SUM(A1)"
    assert _sanitize_excel_text("-3") == "'-3"
    assert _sanitize_excel_text("@cmd") == "'@cmd"
    assert _sanitize_excel_text("normal answer") == "normal answer"
    assert "\x07" not in _sanitize_excel_text("a\x07b")
