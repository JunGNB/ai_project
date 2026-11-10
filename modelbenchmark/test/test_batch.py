"""Tests for the batch layout (schema 3.0) and the Excel/aggregate export layer.

Covered here:

* ``BatchRunner`` writes ``output/runs/<YYYY-MM>/<batch_id>/`` with one run
  directory per model and a single ``batch.json`` for the batch;
* Excel export reads only schema-3.0 artifacts (``batch.json`` + ``run.json`` +
  ``aggregates.json`` + ``records.jsonl`` + prompt blobs) and lands in
  ``output/exports/<batch_id>.xlsx`` — the prompt text is resolved from the blob
  pool and still head/tail clipped at Excel's cell limit;
* cross-run comparison reads ``aggregates.json`` and the analysis CLI accepts a
  run directory or a run id.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modelbench.adapters.config.settings import load_config
from modelbench.adapters.storage import layout
from modelbench.adapters.storage.blobstore import PromptBlobStore
from modelbench.application.batch import BatchRunner
from modelbench.domain.models import RunManifest


class _FakeOrchestrator:
    """Stand-in for ``Orchestrator``: writes run.json + empty aggregates."""

    seen_models = []

    def __init__(self, cfg, client, writer, *, progress_cb=None, cancel_event=None):
        self.cfg = cfg
        self.writer = writer
        self.progress_cb = progress_cb
        _FakeOrchestrator.seen_models.append(list(cfg.selection.models))

    def run(self):
        model_id = self.cfg.selection.models[0]
        manifest = RunManifest(
            run_id=self.writer.run_id,
            created_at="2026-01-01T00:00:00Z",
            project_version="test",
            env={
                "hardware": {
                    "chip": "Apple M4 Pro",
                    "hostname": "test-host",
                    "arch": "arm64",
                    "upload_fields": {"chip_name": "Apple M4 Pro", "memory_gb": 48},
                    "cpu": {"logical_cores": 14, "physical_cores": 14},
                    "memory": {"total_gb": 48},
                    "python": {"version": "3.12.0"},
                },
                "omlx": {"version": "9.9.9", "feature_flags": []},
            },
        )
        self.writer.initialize(
            manifest,
            None,
            {
                "selected": [{"id": model_id}],
                "models": [{"id": model_id, "estimated_size": 1073741824}],
            },
            None,
        )
        summary = self.writer.finalize(
            [], perf={"max_tokens": 128}, run_config={"repeats": 1}
        )
        summary.update({"n_samples": 0, "n_records": 0, "duration_s": 0.1})
        return summary


def _batch_runner(tmp_path, batch_id, models=None, **kwargs):
    cfg = load_config()
    cfg.selection.models = list(models or ["minicpm5-1b", "qwen3.5-9b"])
    cfg.selection.scenarios = ["knowledge_qa"]
    return BatchRunner(
        cfg,
        client=None,
        output_root=tmp_path,
        batch_id=batch_id,
        orchestrator_cls=_FakeOrchestrator,
        **kwargs,
    )


def test_batch_runner_splits_each_model_into_own_run(tmp_path):
    events = []
    _FakeOrchestrator.seen_models = []
    runner = _batch_runner(
        tmp_path,
        "20260101T000000Z",
        progress_cb=lambda event, payload: events.append(event),
    )
    result = runner.run()
    assert result["schema_version"] == "3.0"
    assert result["batch_id"] == "20260101T000000Z"
    assert result["method"] == "performance"
    assert result["n_models_selected"] == 2
    assert result["n_models_measured"] == 2
    assert _FakeOrchestrator.seen_models == [["MiniCPM5-1B"], ["Qwen3.5-9B-MLX-4bit"]]

    # Flat layout: runs/<run_id>/ (one level), batches/<batch_id>.json.
    batch_path = Path(result["batch_file"])
    assert batch_path == layout.batch_path(tmp_path, "20260101T000000Z")
    assert batch_path.is_file()
    assert Path(result["batch_dir"]) == layout.batches_dir(tmp_path)
    assert str(tmp_path) not in batch_path.read_text(encoding="utf-8")
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    assert batch["schema_version"] == "3.0"
    assert batch["batch_id"] == "20260101T000000Z"
    assert batch["method"] == "performance"
    assert len(batch["model_runs"]) == 2

    expected_run_ids = [
        layout.make_run_id("20260101T000000Z", "performance", "minicpm5-1b"),
        layout.make_run_id("20260101T000000Z", "performance", "qwen3.5-9b"),
    ]
    assert [entry["run_id"] for entry in batch["model_runs"]] == expected_run_ids
    for entry in batch["model_runs"]:
        # run_path is relative to the output root (the batch file stays movable).
        assert entry["run_path"] == f"runs/{entry['run_id']}"
        assert "run_dir" not in entry
        assert "summary_path" not in entry
        run_dir = tmp_path / entry["run_path"]
        assert run_dir == layout.run_dir(tmp_path, entry["run_id"])
        assert run_dir.parent == layout.runs_dir(tmp_path)
        assert run_dir.is_dir()
        run = json.loads((run_dir / layout.RUN_JSON).read_text(encoding="utf-8"))
        assert run["run_id"] == entry["run_id"]
        assert run["batch_id"] == "20260101T000000Z"
        assert run["method"] == "performance"
        assert run["env"]["hardware"]["chip"] == "Apple M4 Pro"
        assert run["env"]["model_env"], "model_env snapshot must be recorded once per run"
        for name in (
            layout.MANIFEST_JSON,
            layout.SAMPLES_CSV,
            layout.AGGREGATES_JSON,
            layout.REPORT_MD,
        ):
            assert (run_dir / name).is_file(), name
        # records.jsonl exists once at least one record was written; these fake
        # runs measure nothing, so only the derived files are guaranteed.
        assert entry["n_records"] == 0
        assert (run_dir / layout.RECORDS).is_file() is False
        aggregates = json.loads(
            (run_dir / layout.AGGREGATES_JSON).read_text(encoding="utf-8")
        )
        assert aggregates["run_id"] == entry["run_id"]
        assert isinstance(aggregates["overall"], list)

    # One flat directory per run and nothing else under runs/.
    run_dirs = sorted(path.name for path in layout.runs_dir(tmp_path).iterdir())
    assert run_dirs == sorted(expected_run_ids)
    assert not (tmp_path / layout.RUNS / "2026-11").exists()  # no YYYY-MM bucket
    assert not (tmp_path / layout.RUNS / "performance").exists()  # no method level
    assert not (tmp_path / "20260101T000000Z_performance_minicpm5-1b").exists()
    # Excel is generated on demand, never as a batch side effect.
    assert not (tmp_path / "result_analysis").exists()
    assert not list((tmp_path / layout.EXPORTS).glob("*.xlsx"))

    # The batch runner's own return value is enough to render the workbook.
    from openpyxl import load_workbook

    from modelbench.adapters.storage.excel_report import (
        generate_all,
        generate_for_batch,
        write_batch_excel,
    )

    direct = write_batch_excel(tmp_path, batch_path, result)
    assert direct == tmp_path / layout.EXPORTS / "20260101T000000Z.xlsx"
    assert direct.is_file()

    # Regenerate the workbook for the existing batch without re-running anything.

    assert generate_all(tmp_path) == [direct]
    workbook_path = generate_for_batch(tmp_path, "20260101T000000Z")
    assert workbook_path == direct
    assert workbook_path == tmp_path / layout.EXPORTS / "20260101T000000Z.xlsx"
    assert workbook_path.is_file()
    assert not (tmp_path / "result_analysis").exists()
    workbook = load_workbook(workbook_path)
    assert workbook.sheetnames == ["硬件配置", "模型性能平均", "详细数据"]
    averages_headers = [cell.value for cell in workbook["模型性能平均"][1]]
    assert averages_headers[:4] == ["测试模型", "模型ID", "模型大小(GB)", "上下文档位"]
    # two empty runs -> one status row each
    assert workbook["模型性能平均"].max_row == 3
    assert workbook["模型性能平均"].cell(row=2, column=3).value == 1.0
    assert events.count("batch_model_start") == 2
    assert events.count("batch_model_done") == 2


# --------------------------------------------------------------------------- #
# Excel export: schema-3.0 data sources
# --------------------------------------------------------------------------- #


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_batch_tree(tmp_path: Path, *, long_prompt_chars: int = 40_000):
    """A complete schema-3.0 batch tree with one model, one long prompt."""
    batch_id = "ctl-20260101"
    run_id = layout.make_run_id(batch_id, "performance", "minicpm5-1b")
    batch_path = layout.batch_path(tmp_path, batch_id)
    run_dir = layout.run_dir(tmp_path, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    store = PromptBlobStore(tmp_path)
    long_text = "HEAD-" + ("x" * long_prompt_chars) + "-TAIL"
    long_sha = store.put_messages(
        [
            {"role": "system", "content": "你是助手"},
            {"role": "user", "content": long_text},
        ]
    )
    _write_json(
        batch_path,
        {
            "schema_version": "3.0",
            "batch_id": batch_id,
            "source": "cli",
            "method": "performance",
            "created_at": "2026-01-01T00:00:00+00:00",
            "finished_at": "2026-01-01T00:01:00+00:00",
            "models_selected": ["minicpm5-1b"],
            "models_measured": ["minicpm5-1b"],
            "model_runs": [
                {
                    "model": "minicpm5-1b",
                    "model_id": "MiniCPM5-1B",
                    "model_slug": "minicpm5-1b",
                    "method": "performance",
                    "run_id": run_id,
                    "run_path": f"runs/{run_id}",
                    "status": "ok",
                    "n_samples": 1,
                    "duration_s": 1.5,
                }
            ],
            "inputs": {"testset": {"path": "dataset/test/performance/testset.jsonl"}},
        },
    )
    _write_json(
        run_dir / layout.RUN_JSON,
        {
            "schema_version": "3.0",
            "run_id": run_id,
            "batch_id": batch_id,
            "method": "performance",
            "model": {
                "id": "MiniCPM5-1B",
                "alias": "minicpm5-1b",
                "model_slug": "minicpm5-1b",
            },
            "created_at": "2026-01-01T00:00:00+00:00",
            "finished_at": "2026-01-01T00:01:00+00:00",
            "duration_s": 1.5,
            "git": {"commit": "deadbeef", "dirty": False},
            "config": {
                "perf": {"max_tokens": 128, "decode_guard": True},
                "run": {"repeats": 1, "cache_mode": "cold", "thinking_mode": False},
                "scenarios": ["knowledge_qa"],
                "context_windows": [1024],
            },
            "env": {
                "hardware": {
                    "chip": "Apple M4 Pro",
                    "hostname": "test-host",
                    "arch": "arm64",
                    "upload_fields": {
                        "chip_name": "Apple M4 Pro",
                        "chip_variant": "M4 Pro",
                        "memory_gb": 48,
                        "gpu_cores": 20,
                        "os_version": "macOS 15.1",
                        "omlx_version": "0.2.0",
                    },
                    "cpu": {"logical_cores": 14, "physical_cores": 14},
                    "memory": {"total_gb": 48, "available_gb": 30},
                    "python": {"version": "3.12.0"},
                },
                "omlx": {
                    "version": "0.2.0",
                    "feature_flags": [],
                    "default_config": True,
                    "global_settings": {"memory": {"max_size_gb": 40}},
                },
                "model_env": {
                    "0123456789abcdef": {
                        "id": "MiniCPM5-1B",
                        "engine_type": "vlm",
                        "quantization": "oQ4",
                        "estimated_size": 5285234950,
                        "context_length": 131072,
                        "settings": {},
                        "feature_flags": {},
                    }
                },
            },
            "outcome": {
                "status": "ok",
                "cancelled": False,
                "memory_errors": 0,
                "n_selected": 2,
                "n_measured": 1,
                "n_samples": 1,
                "skipped": [{"scenario": "s", "sample_id": "x", "reason": "dataset missing"}],
            },
        },
    )
    _write_json(
        run_dir / layout.AGGREGATES_JSON,
        {
            "schema_version": "3.0",
            "run_id": run_id,
            "generated_at": "2026-01-01T00:01:00+00:00",
            "n_records": 2,
            "n_samples": 1,
            "n_cells": 1,
            "overall": [
                {
                    "model": "MiniCPM5-1B",
                    "context_length": 1024,
                    "n_datasets": 1,
                    "n_samples": 1,
                    "n_valid": 1,
                    "n_error": 0,
                    "n_length_capped": 0,
                    "perf": {"ttft_ms": {"n": 1, "mean": 12.5, "median": 12.5, "p95": 12.5}},
                    "peak_memory_gb_max": 4.2,
                }
            ],
            "cells": [],
        },
    )
    records = [
        # measured sample: prompt body only via prompt_sha256
        {
            "phase": "measured",
            "model": "MiniCPM5-1B",
            "dataset": "dureader",
            "scenario": "search_agent",
            "sample_id": "sample-1",
            "repeat": 0,
            "request": {
                "prompt_sha256": long_sha,
                "prompt_chars": len(long_text),
                "truncated": False,
            },
            "response": {"text": "answer\x01with\x0ccontrol\x1fchars", "finish_reason": "stop"},
            "metrics": {
                "input_context_tokens": 1100,
                "input_context_chars": len(long_text),
                "completion_tokens": 4,
                "ttft_ms": 12.5,
                "gen_tps": 20.0,
                "peak_memory_gb": 4.2,
            },
            "status": "ok",
            "timestamp": "2026-01-01T00:00:30+00:00",
        },
        # warmup sample is excluded from the details sheet
        {
            "phase": "warmup",
            "dataset": "dureader",
            "scenario": "search_agent",
            "sample_id": "sample-0",
            "repeat": 0,
            "request": {},
            "response": {},
            "metrics": {},
            "status": "ok",
        },
        # measured sample whose blob is gone: the cell must say so, not be blank
        {
            "phase": "measured",
            "model": "MiniCPM5-1B",
            "dataset": "gsm8k",
            "scenario": "knowledge_qa",
            "sample_id": "sample-2",
            "repeat": 0,
            "request": {"prompt_sha256": "deadbeef" * 8, "truncated": True},
            "response": {"text": "=1+1", "finish_reason": "length"},
            "metrics": {"input_context_tokens": 10, "input_context_chars": 20},
            "status": "ok",
            "timestamp": "2026-01-01T00:00:40+00:00",
        },
    ]
    (run_dir / layout.RECORDS).write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    return batch_id, batch_path, run_dir, long_text


def test_excel_export_reads_schema3_sources_and_clips_long_prompts(tmp_path):
    from openpyxl import load_workbook
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

    from modelbench.adapters.storage.excel_report import (
        _MAX_CELL_CHARS,
        write_batch_excel,
    )

    batch_id, batch_path, run_dir, long_text = _make_batch_tree(tmp_path)
    assert run_dir == layout.run_dir(tmp_path, layout.make_run_id(batch_id, "performance", "minicpm5-1b"))
    workbook_path = write_batch_excel(tmp_path, batch_path)
    assert workbook_path == tmp_path / layout.EXPORTS / f"{batch_id}.xlsx"
    assert workbook_path.is_file()
    # Only the workbook: no batch.json / README.md copies, no result_analysis/.
    assert [path.name for path in workbook_path.parent.iterdir()] == [workbook_path.name]
    assert not (tmp_path / "result_analysis").exists()

    workbook = load_workbook(workbook_path)
    assert workbook.sheetnames == ["硬件配置", "模型性能平均", "详细数据"]

    averages = workbook["模型性能平均"]
    assert [cell.value for cell in averages[1]][:4] == [
        "测试模型", "模型ID", "模型大小(GB)", "上下文档位",
    ]
    assert averages.max_row == 2
    assert averages.cell(row=2, column=1).value == "minicpm5-1b"
    assert averages.cell(row=2, column=2).value == "MiniCPM5-1B"
    assert averages.cell(row=2, column=3).value == 4.92
    assert averages.cell(row=2, column=4).value == "1K"
    assert averages.cell(row=2, column=20).value == 4.2  # 峰值内存最大
    assert averages.cell(row=2, column=26).value == 1  # 跳过数

    hardware = workbook["硬件配置"]
    hardware_rows = {
        row[0].value: row[1].value
        for row in hardware.iter_rows(min_row=2, max_col=2)
        if row[0].value is not None
    }
    assert hardware_rows["批次 ID"] == batch_id
    assert hardware_rows["芯片"] == "Apple M4 Pro"
    assert hardware_rows["oMLX 版本"] == "0.2.0"
    assert hardware_rows["git commit"] == "deadbeef"

    details = workbook["详细数据"]
    detail_headers = [cell.value for cell in details[1]]
    for required in (
        "测试模型", "样本来源", "样本所属场景", "API 输入内容",
        "输入 prompt token 大小", "输出结果", "输出 token 大小", "峰值内存占用(GB)",
    ):
        assert required in detail_headers
    # header + 2 measured records (warmup filtered out)
    assert details.max_row == 3

    input_cell = details.cell(row=2, column=detail_headers.index("API 输入内容") + 1).value
    # The prompt body came from the blob store, clipped with the original length.
    assert len(input_cell) <= _MAX_CELL_CHARS
    assert "HEAD-" in input_cell and "-TAIL" in input_cell
    assert "内容已截断" in input_cell
    expected_len = (
        len("[system]\n你是助手") + 2 + len(f"[user]\n{long_text}")
    )
    assert f"原长度 {expected_len} 字符" in input_cell

    missing_cell = details.cell(row=3, column=detail_headers.index("API 输入内容") + 1).value
    assert "prompt blob 缺失" in missing_cell
    # Formula-injection guard on model output is preserved.
    output_cell = details.cell(row=3, column=detail_headers.index("输出结果") + 1).value
    assert output_cell.startswith("'=")
    # Control characters were stripped from every cell.
    for ws in workbook.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                if isinstance(cell.value, str):
                    assert not ILLEGAL_CHARACTERS_RE.search(cell.value)

    # Deleting exports/ is safe: regenerate from the batch tree.
    workbook_path.unlink()
    from modelbench.adapters.storage.excel_report import generate_for_batch

    regenerated = generate_for_batch(tmp_path, batch_id)
    assert regenerated.is_file()
    assert regenerated == workbook_path


def test_excel_export_handles_batch_without_runs(tmp_path):
    """A batch whose models never produced a run still yields a valid workbook."""
    from openpyxl import load_workbook

    from modelbench.adapters.storage.excel_report import write_batch_excel

    batch_path = layout.batch_path(tmp_path, "empty")
    _write_json(
        batch_path,
        {
            "schema_version": "3.0",
            "batch_id": "empty",
            "method": "performance",
            "created_at": "2026-01-01T00:00:00+00:00",
            "models_selected": [],
            "model_runs": [],
        },
    )
    workbook_path = write_batch_excel(tmp_path, batch_path)
    workbook = load_workbook(workbook_path)
    assert workbook.sheetnames == ["硬件配置", "模型性能平均", "详细数据"]
    assert workbook["模型性能平均"].cell(row=2, column=1).value == "无模型结果"
    assert workbook["详细数据"].cell(row=2, column=1).value == "无逐样本记录"


def test_excel_export_falls_back_to_runs_by_batch_id(tmp_path):
    """An incomplete batch file (no model_runs) still exports its runs.

    The flat tree has no shared batch directory, so the fallback identifies runs
    by their own ``run.json.batch_id``.
    """
    from openpyxl import load_workbook

    from modelbench.adapters.storage.excel_report import write_batch_excel

    batch_id = "orphan"
    run_id = layout.make_run_id(batch_id, "performance", "m1")
    run_dir = layout.run_dir(tmp_path, run_id)
    run_dir.mkdir(parents=True)
    _write_json(
        run_dir / layout.RUN_JSON,
        {
            "schema_version": "3.0",
            "run_id": run_id,
            "batch_id": batch_id,
            "method": "performance",
            "model": {"id": "M1", "alias": "m1", "model_slug": "m1"},
            "created_at": "2026-01-01T00:00:00+00:00",
            "env": {},
            "config": {},
            "outcome": {"status": "ok"},
        },
    )
    _write_json(
        run_dir / layout.AGGREGATES_JSON,
        {
            "schema_version": "3.0",
            "run_id": run_id,
            "overall": [
                {"model": "M1", "context_length": 1024, "n_samples": 1, "perf": {}}
            ],
            "cells": [],
        },
    )
    batch_path = layout.batch_path(tmp_path, batch_id)
    _write_json(
        batch_path,
        {
            "schema_version": "3.0",
            "batch_id": batch_id,
            "method": "performance",
            "created_at": "2026-01-01T00:00:00+00:00",
            "models_selected": ["m1"],
            "model_runs": [],
        },
    )
    workbook = load_workbook(write_batch_excel(tmp_path, batch_path))
    averages = workbook["模型性能平均"]
    assert averages.max_row == 2
    assert averages.cell(row=2, column=2).value == "M1"


def test_excel_export_rejects_missing_or_directory_batch_path(tmp_path):
    from modelbench.adapters.storage.excel_report import write_batch_excel

    # Missing batch metadata file (batches/<batch_id>.json).
    with pytest.raises(FileNotFoundError):
        write_batch_excel(tmp_path, layout.batch_path(tmp_path, "nope"))
    # A directory is a wiring bug in the flat layout: fail loudly, do not guess.
    (tmp_path / layout.BATCHES).mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="must be a file"):
        write_batch_excel(tmp_path, layout.batches_dir(tmp_path))


# --------------------------------------------------------------------------- #
# aggregates comparison / analysis CLI
# --------------------------------------------------------------------------- #


def _make_aggregates_run(tmp_path: Path, run_id: str, *, ttft: float) -> Path:
    run_dir = layout.run_dir(tmp_path, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / layout.AGGREGATES_JSON,
        {
            "schema_version": "3.0",
            "run_id": run_id,
            "generated_at": "2026-11-01T00:00:00+00:00",
            "overall": [
                {
                    "model": "MiniCPM5-1B",
                    "context_length": 1024,
                    "n_datasets": 1,
                    "n_samples": 5,
                    "perf": {"ttft_ms": {"n": 5, "mean": ttft}},
                }
            ],
            "cells": [],
        },
    )
    return run_dir


def test_load_aggregates_and_compare_read_aggregates_json(tmp_path):
    from modelbench.interfaces.analysis.compare import (
        compare_summaries,
        load_aggregates,
        load_summary,
    )

    a = _make_aggregates_run(tmp_path, "batchA__performance__m", ttft=10.0)
    b = _make_aggregates_run(tmp_path, "batchB__performance__m", ttft=12.0)

    doc = load_aggregates(a)
    assert doc["run_id"] == "batchA__performance__m"
    assert load_summary(a) == doc

    table = compare_summaries([doc, load_aggregates(b)], metrics=["ttft_ms"])
    assert "| 模型 | 档位 |" in table
    assert "batchA__performance__m:ttft_ms" in table
    assert "batchB__performance__m:ttft_ms" in table
    assert "Δ(batchB__performance__m-batchA__performance__m):ttft_ms" in table
    assert "| MiniCPM5-1B | 1K | 10.000 | 12.000 | 2.000 |" in table

    # Accepting paths directly keeps the old convenience working.
    assert "10.000" in compare_summaries([a, b], metrics=["ttft_ms"])

    with pytest.raises(FileNotFoundError):
        load_aggregates(tmp_path / "nope")


def test_analysis_cli_accepts_run_directory_or_run_id(tmp_path, capsys, monkeypatch):
    from modelbench.adapters.storage import catalog
    from modelbench.interfaces.analysis import cli as analysis_cli

    run_a = _make_aggregates_run(tmp_path, "batchA__performance__m", ttft=10.0)
    run_b = _make_aggregates_run(tmp_path, "batchB__performance__m", ttft=12.0)

    assert analysis_cli.main([str(run_a), str(run_b), "--metrics", "ttft_ms"]) == 0
    out = capsys.readouterr().out
    assert "batchA__performance__m:ttft_ms" in out

    # run ids resolve through catalog.resolve_run against the output root.
    monkeypatch.setattr(
        catalog,
        "resolve_run",
        lambda output_root, run_id: {"batchA__performance__m": run_a, "batchB__performance__m": run_b}[run_id],
    )
    assert (
        analysis_cli.main(
            [
                "batchA__performance__m",
                "batchB__performance__m",
                "--metrics",
                "ttft_ms",
                "--output-dir",
                str(tmp_path),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert [doc["run_id"] for doc in payload] == [
        "batchA__performance__m",
        "batchB__performance__m",
    ]

    assert analysis_cli.main(["missing-run", "--output-dir", str(tmp_path)]) == 1
    assert "ERROR" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# locking
# --------------------------------------------------------------------------- #


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


def test_batch_runner_refuses_existing_batch_file(tmp_path):
    runner = _batch_runner(tmp_path, "dup")
    first = runner.run()
    assert Path(first["batch_file"]).is_file()
    assert Path(first["batch_file"]) == layout.batch_path(tmp_path, "dup")

    with pytest.raises(FileExistsError, match="already exist"):
        _batch_runner(tmp_path, "dup").run()


def test_batch_runner_refuses_when_another_run_holds_the_lock(tmp_path):
    from modelbench.application.locking import RunLock, RunLockError

    runner = _batch_runner(tmp_path, "locked", models=["minicpm5-1b"])
    with RunLock(tmp_path):
        with pytest.raises(RunLockError):
            runner.run()
