"""Tests for the per-model batch layout (time + method + model)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modelbench.adapters.config.settings import load_config
from modelbench.application.batch import BatchRunner
from modelbench.domain.models import RunManifest


class _FakeOrchestrator:
    seen_models = []

    def __init__(self, cfg, client, writer, *, progress_cb=None, cancel_event=None):
        self.cfg = cfg
        self.writer = writer
        self.progress_cb = progress_cb
        _FakeOrchestrator.seen_models.append(list(cfg.selection.models))

    def run(self):
        manifest = RunManifest(
            run_id=self.writer.run_id,
            created_at="2026-01-01T00:00:00Z",
            project_version="test",
            batch=self.writer.batch,
            models={"selected": [{"id": self.cfg.selection.models[0]}]},
        )
        self.writer.initialize(manifest, {}, {"selected": []}, {})
        summary = self.writer.finalize([], perf={"max_tokens": 128}, run_config={"repeats": 1})
        summary.update({"n_samples": 1, "n_records": 1, "duration_s": 0.1})
        if self.progress_cb:
            self.progress_cb("model_start", {"model": self.cfg.selection.models[0], "index": 1, "total": 1})
            self.progress_cb("model_done", {"model": self.cfg.selection.models[0]})
        return summary


def test_batch_runner_splits_each_model_into_own_bundle(tmp_path):
    cfg = load_config()
    cfg.selection.models = ["minicpm5-1b", "qwen3.5-9b"]
    cfg.selection.scenarios = ["knowledge_qa"]
    events = []
    _FakeOrchestrator.seen_models = []
    runner = BatchRunner(
        cfg,
        client=None,
        output_root=tmp_path,
        batch_id="20260101T000000Z",
        orchestrator_cls=_FakeOrchestrator,
        progress_cb=lambda event, payload: events.append(event),
    )
    result = runner.run()
    assert result["batch_id"] == "20260101T000000Z"
    assert result["method"] == "performance"
    assert result["n_models_tested"] == 2
    assert _FakeOrchestrator.seen_models == [["MiniCPM5-1B"], ["Qwen3.5-9B-MLX-4bit"]]

    expected = [
        "20260101T000000Z_performance_minicpm5-1b",
        "20260101T000000Z_performance_qwen3.5-9b",
    ]
    for run_id in expected:
        run_dir = tmp_path / run_id
        assert run_dir.is_dir()
        summary = json.loads((run_dir / "summary.json").read_text())
        assert summary["batch"]["batch_id"] == "20260101T000000Z"
        assert summary["batch"]["method"] == "performance"
        assert summary["batch"]["run_id"] == run_id
        assert (run_dir / "analysis" / "run.json").is_file()
    assert (tmp_path / "batches" / "20260101T000000Z.json").is_file()
    assert events.count("batch_model_start") == 2
    assert events.count("batch_model_done") == 2

    analysis = result.get("result_analysis") or {}
    workbook_path = Path(analysis.get("workbook") or "")
    assert workbook_path.is_file()
    assert "result_analysis" in analysis["dir"]
    from openpyxl import load_workbook

    wb = load_workbook(workbook_path)
    assert wb.sheetnames == ["硬件配置", "模型性能平均", "详细数据"]
    detail_headers = [cell.value for cell in wb["详细数据"][1]]
    for required in (
        "测试模型", "样本来源", "样本所属场景", "API 输入内容",
        "输入 prompt token 大小", "输出结果", "输出 token 大小", "峰值内存占用(GB)",
    ):
        assert required in detail_headers
    assert wb["模型性能平均"].max_row == 3  # header + two models
    # Regenerate from the catalog without re-running the benchmark.
    from modelbench.adapters.storage.excel_report import generate_for_batch

    regenerated = generate_for_batch(tmp_path, "20260101T000000Z")
    assert Path(regenerated["workbook"]).is_file()
    assert (Path(regenerated["dir"]) / "batch.json").is_file()

    index = json.loads((tmp_path / "index.json").read_text())
    assert index["n_runs"] == 2
    assert index["n_batches"] == 1
    batch = index["batches"][0]
    assert batch["batch_id"] == "20260101T000000Z"
    assert batch["method"] == "performance"
    assert batch["n_models"] == 2
    # In this fake run there are no measured records, so the catalog falls back
    # to the batch label (alias); real runs expose the canonical model id.
    assert {run["model"] for run in batch["models"]} == {"minicpm5-1b", "qwen3.5-9b"}


def test_batch_runner_rejects_existing_directories(tmp_path):
    cfg = load_config()
    cfg.selection.models = ["minicpm5-1b"]
    cfg.selection.scenarios = ["knowledge_qa"]
    (tmp_path / "dup_performance_minicpm5-1b").mkdir()
    runner = BatchRunner(
        cfg,
        client=None,
        output_root=tmp_path,
        batch_id="dup",
        orchestrator_cls=_FakeOrchestrator,
    )
    try:
        runner.run()
    except FileExistsError as exc:
        assert "already exist" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected FileExistsError")


def test_excel_analysis_sanitizes_control_characters(tmp_path):
    """Control chars in prompt/output text must not abort Excel generation.

    Regression: a real run's long prompt contained BEL/VT, openpyxl raised
    IllegalCharacterError, BatchRunner caught it and left an empty
    ``output/result_analysis/<batch>`` directory with no summary.xlsx.
    """
    from openpyxl import load_workbook
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

    from modelbench.adapters.storage.excel_report import write_batch_excel

    run_dir = tmp_path / "ctl_performance_minicpm5-1b"
    run_dir.mkdir()
    record = {
        "phase": "measured",
        "model": "MiniCPM5-1B",
        "dataset": "dureader",
        "scenario": "search_agent",
        "sample_id": "sample-1",
        "repeat": 0,
        "request": {
            "messages": [
                {"role": "user", "content": "bad\x07 bell \x0b vt \x00 nul text"},
            ],
            "truncated": False,
        },
        "response": {"text": "answer\x01with\x0ccontrol\x1fchars"},
        "metrics": {
            "input_context_tokens": 10,
            "input_context_chars": 20,
            "completion_tokens": 4,
            "ttft_ms": 1.0,
            "gen_tps": 2.0,
            "peak_memory_gb": 1.0,
        },
        "status": "ok",
        "timestamp": "2026-01-01T00:00:00Z",
    }
    (run_dir / "records.jsonl").write_text(
        json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    batch = {
        "batch_id": "ctl",
        "method": "performance",
        "created_at": "2026-01-01T00:00:00Z",
        "models_selected": ["MiniCPM5-1B"],
        "n_models_tested": 1,
        "duration_s": 1.0,
        "model_runs": [
            {
                "model": "MiniCPM5-1B",
                "model_id": "MiniCPM5-1B",
                "run_dir": str(run_dir),
                "n_samples": 1,
            }
        ],
    }
    result = write_batch_excel(tmp_path, batch)
    workbook_path = Path(result["workbook"])
    assert workbook_path.is_file()
    workbook = load_workbook(workbook_path)
    assert workbook["详细数据"].max_row >= 2
    for ws in workbook.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                if isinstance(cell.value, str):
                    assert not ILLEGAL_CHARACTERS_RE.search(cell.value)
    # Sanitized text still carries the readable parts.
    values = [
        cell.value
        for row in workbook["详细数据"].iter_rows()
        for cell in row
        if isinstance(cell.value, str)
    ]
    joined = "\n".join(values)
    assert "bad bell" in joined and "vt" in joined and "answer" in joined


def test_run_lock_prevents_concurrent_benchmark_processes(tmp_path):
    """AGENTS #10: a second process must fail fast instead of evicting models."""
    from modelbench.application.locking import RunLock, RunLockError

    with RunLock(tmp_path):
        with pytest.raises(RunLockError, match="another benchmark run is active"):
            with RunLock(tmp_path):
                pass  # pragma: no cover - must not be reached
    # Released: a new run can acquire the same output root.
    with RunLock(tmp_path):
        pass


def test_batch_runner_refuses_when_another_run_holds_the_lock(tmp_path):
    from modelbench.application.locking import RunLock, RunLockError

    cfg = load_config()
    cfg.selection.models = ["minicpm5-1b"]
    cfg.selection.scenarios = ["knowledge_qa"]
    runner = BatchRunner(
        cfg,
        client=None,
        output_root=tmp_path,
        batch_id="locked",
        orchestrator_cls=_FakeOrchestrator,
    )
    with RunLock(tmp_path):
        with pytest.raises(RunLockError):
            runner.run()
