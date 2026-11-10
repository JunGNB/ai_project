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
    # The persisted batch.json keeps relative paths only (stays movable);
    # absolute run/export paths live only in the in-memory return value.
    persisted_text = batch_path.read_text(encoding="utf-8")
    for key in ("batch_file", "batch_dir", "export_file", "run_dir"):
        assert f'"{key}"' not in persisted_text, key
    assert str(tmp_path) not in persisted_text
    assert json.loads(persisted_text)["exports"] == {
        "xlsx": "exports/20260101T000000Z.xlsx"
    }
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
    # T-068: the Excel summary is a default deliverable of every finished
    # batch, generated automatically into exports/ (no result_analysis/ ever).
    assert not (tmp_path / "result_analysis").exists()
    auto_exports = sorted((tmp_path / layout.EXPORTS).glob("*.xlsx"))
    assert [p.name for p in auto_exports] == ["20260101T000000Z.xlsx"]
    # The run result carries the export path and the movable batch.json
    # records the fact with a relative name only.
    assert Path(result["export_file"]) == auto_exports[0]
    batch_json = json.loads(batch_path.read_text(encoding="utf-8"))
    assert batch_json["exports"] == {"xlsx": "exports/20260101T000000Z.xlsx"}
    assert "batch_file" not in batch_json
    assert "export_file" not in batch_json

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
    # two empty runs -> one status row each (+ the one-line calibre note)
    assert workbook["模型性能平均"].max_row == 4
    assert workbook["模型性能平均"].cell(row=2, column=3).value == 1.0
    assert events.count("batch_model_start") == 2
    assert events.count("batch_model_done") == 2


# --------------------------------------------------------------------------- #
# Excel export: schema-3.0 data sources
# --------------------------------------------------------------------------- #


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_batch_tree(
    tmp_path: Path,
    *,
    long_prompt_chars: int = 40_000,
    dual_caliber: bool = False,
):
    """A complete schema-3.0 batch tree with one model, one long prompt.

    Measured records carry ``request.context_meta.isl_tolerance`` (actual 1100
    vs the 1K target 1024 -> in band).  ``dual_caliber=True`` adds a second
    measured record whose actual ISL is out of band (1144 tokens, no
    ``context_meta`` so the export's ±10% fallback decides) plus an 8K overall
    row without matching records, so the dual-caliber export columns have both
    a real in-band vs all-sample difference and an empty bucket to leave blank.
    """
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
    short_sha = store.put_messages([{"role": "user", "content": "出带样本的问题"}])
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
    overall_rows = [
        {
            "model": "MiniCPM5-1B",
            "context_length": 1024,
            "n_datasets": 1,
            "n_samples": 1,
            "n_valid": 1,
            "n_error": 0,
            "n_length_capped": 0,
            "n_isl_in_band": 1,
            "n_isl_out_of_band": 0,
            # In-band calibre: the mean uses only in-band samples (AGENTS #13).
            "perf": {"ttft_ms": {"n": 1, "mean": 12.5, "median": 12.5, "p95": 12.5}},
            "peak_memory_gb_max": 4.2,
        }
    ]
    in_band_metrics = {
        "input_context_tokens": 1100,
        "input_context_chars": len(long_text),
        "completion_tokens": 4,
        "ttft_ms": 12.5,
        "gen_tps": 20.0,
        "peak_memory_gb": 4.2,
    }
    if dual_caliber:
        # in-band leg of the pair: ttft 10.0 while the out-of-band record has
        # 30.0 -> all-sample mean 20.0, relative shift +100%.
        overall_rows[0].update(
            {
                "n_samples": 2,
                "n_valid": 2,
                "n_isl_in_band": 1,
                "n_isl_out_of_band": 1,
                "perf": {
                    "ttft_ms": {"n": 1, "mean": 10.0, "median": 10.0, "p95": 10.0},
                    "gen_tps": {"n": 1, "mean": 20.0, "median": 20.0, "p95": 20.0},
                    "e2e_ms": {"n": 1, "mean": 100.0, "median": 100.0, "p95": 100.0},
                },
            }
        )
        # A bucket with aggregates but no records: all-sample columns stay blank.
        overall_rows.append(
            {
                "model": "MiniCPM5-1B",
                "context_length": 8192,
                "n_datasets": 1,
                "n_samples": 1,
                "n_valid": 1,
                "n_error": 0,
                "n_length_capped": 0,
                "n_isl_in_band": 1,
                "n_isl_out_of_band": 0,
                "perf": {"ttft_ms": {"n": 1, "mean": 50.0, "median": 50.0, "p95": 50.0}},
                "peak_memory_gb_max": 5.0,
            }
        )
        in_band_metrics.update({"ttft_ms": 10.0, "e2e_ms": 100.0})
    records = [
        # measured sample: prompt body only via prompt_sha256; ISL in band
        # (actual 1100 inside 1024 +/- 10% = 921..1126)
        {
            "phase": "measured",
            "model": "MiniCPM5-1B",
            "dataset": "dureader",
            "scenario": "search_agent",
            "sample_id": "sample-1",
            "repeat": 0,
            "context_length": 1024,
            "request": {
                "prompt_sha256": long_sha,
                "prompt_chars": len(long_text),
                "truncated": False,
                "context_meta": {
                    "isl_tolerance": {
                        "target": 1024,
                        "low": 921,
                        "high": 1126,
                        "actual": 1100,
                        "ok": True,
                    }
                },
            },
            "response": {"text": "answer\x01with\x0ccontrol\x1fchars", "finish_reason": "stop"},
            "metrics": in_band_metrics,
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
    if dual_caliber:
        records.append(
            # measured but ISL out of band (1144 > 1126): no context_meta, so
            # the export falls back to the ±10% band check on the record
            {
                "phase": "measured",
                "model": "MiniCPM5-1B",
                "dataset": "dureader",
                "scenario": "search_agent",
                "sample_id": "sample-3",
                "repeat": 0,
                "context_length": 1024,
                "request": {
                    "prompt_sha256": short_sha,
                    "prompt_chars": 7,
                    "truncated": False,
                    "tokenized_prompt_tokens": 1144,
                },
                "response": {"text": "出带样本的长回答", "finish_reason": "stop"},
                "metrics": {
                    "input_context_tokens": 1144,
                    "input_context_chars": 7,
                    "completion_tokens": 6,
                    "ttft_ms": 30.0,
                    "gen_tps": 40.0,
                    "e2e_ms": 300.0,
                    "peak_memory_gb": 4.4,
                },
                "status": "ok",
                "timestamp": "2026-01-01T00:00:50+00:00",
            }
        )
    _write_json(
        run_dir / layout.AGGREGATES_JSON,
        {
            "schema_version": "3.0",
            "run_id": run_id,
            "generated_at": "2026-01-01T00:01:00+00:00",
            "n_records": len(records),
            "n_samples": len({r.get("sample_id") for r in records if r.get("phase") == "measured"}),
            "n_cells": 1,
            "overall": overall_rows,
            "cells": [],
        },
    )
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
    # header + one data row + the one-line calibre note
    assert averages.max_row == 3
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


def test_excel_details_sheet_annotates_isl_band_and_deviation(tmp_path):
    """T-064: the details sheet annotates each sample's ISL band membership.

    ``ISL入带`` mirrors ``metrics.isl_band`` (authoritative
    ``request.context_meta.isl_tolerance.ok``, ±10% fallback on the actual
    token count, legacy records without any basis kept as in band);
    ``ISL偏差%`` is ``(actual - target) / target * 100`` and stays blank when
    the record carries no target/actual evidence.
    """
    from openpyxl import load_workbook

    from modelbench.adapters.storage.excel_report import write_batch_excel

    batch_id, batch_path, run_dir, _ = _make_batch_tree(tmp_path)
    workbook = load_workbook(write_batch_excel(tmp_path, batch_path))
    details = workbook["详细数据"]
    headers = [cell.value for cell in details[1]]
    assert "ISL入带" in headers
    assert "ISL偏差%" in headers

    def sample_row(sample_id: str) -> dict:
        id_col = headers.index("样本ID")
        for row in details.iter_rows(min_row=2):
            if row[id_col].value == sample_id:
                return {headers[i]: cell.value for i, cell in enumerate(row)}
        raise AssertionError(f"sample row not found: {sample_id}")

    # In band: actual 1100 vs the 1K target 1024 -> +7.4%.
    in_band = sample_row("sample-1")
    assert in_band["ISL入带"] is True
    assert in_band["ISL偏差%"] == 7.4
    # No isl_tolerance, no context_length: kept like a legacy record; the
    # deviation cannot be computed and stays an empty cell, not a fake 0.
    legacy = sample_row("sample-2")
    assert legacy["ISL入带"] is True
    assert legacy["ISL偏差%"] is None

    # Single in-band sample: the all-sample calibre equals the in-band mean,
    # so the out-of-band TTFT shift renders as an exact 0.0%.
    averages = workbook["模型性能平均"]
    avg_headers = [cell.value for cell in averages[1]]
    col = {name: index + 1 for index, name in enumerate(avg_headers)}
    assert averages.cell(row=2, column=col["TTFT均值(ms)"]).value == 12.5
    assert averages.cell(row=2, column=col["TG TPS均值"]).value == 20.0
    assert averages.cell(row=2, column=col["E2E均值(ms)"]).value is None
    assert averages.cell(row=2, column=col["TTFT均值(ms)（±10%过滤）"]).value == 12.5
    assert averages.cell(row=2, column=col["出带TTFT波动范围"]).value == "0.0%"


def test_excel_averages_sheet_dual_caliber_and_out_of_band_shift(tmp_path):
    """T-064: all-sample means + out-of-band TTFT shift next to the in-band ones.

    One in-band record (ttft 10.0) and one out-of-band record (ttft 30.0, ISL
    1144) share the 1K bucket: the in-band mean stays 10.0 (AGENTS #13), the
    all-sample mean is 20.0 and the relative shift is +100.0%.  An 8K overall
    row without records leaves the all-sample columns blank.
    """
    from openpyxl import load_workbook

    from modelbench.adapters.storage.excel_report import write_batch_excel

    batch_id, batch_path, run_dir, _ = _make_batch_tree(tmp_path, dual_caliber=True)
    workbook = load_workbook(write_batch_excel(tmp_path, batch_path))
    averages = workbook["模型性能平均"]
    headers = [cell.value for cell in averages[1]]
    for name in (
        "TTFT均值(ms)",
        "TG TPS均值",
        "E2E均值(ms)",
        "TTFT均值(ms)（±10%过滤）",
        "TG TPS均值（±10%过滤）",
        "E2E均值(ms)（±10%过滤）",
        "出带TTFT波动范围",
    ):
        assert name in headers
    # header + 1K row + 8K row + the one-line calibre note (no extra sheet).
    assert averages.max_row == 4
    col = {name: index + 1 for index, name in enumerate(headers)}

    # 1K row: in-band mean untouched, all-sample means include the out-of-band
    # record, and the TTFT shift shows how much the out-of-band leg moves it.
    assert averages.cell(row=2, column=col["TTFT均值(ms)"]).value == 20.0
    assert averages.cell(row=2, column=col["TTFT均值(ms)（±10%过滤）"]).value == 10.0
    assert averages.cell(row=2, column=col["TG TPS均值"]).value == 30.0
    assert averages.cell(row=2, column=col["TG TPS均值（±10%过滤）"]).value == 20.0
    assert averages.cell(row=2, column=col["E2E均值(ms)"]).value == 200.0
    assert averages.cell(row=2, column=col["E2E均值(ms)（±10%过滤）"]).value == 100.0
    assert averages.cell(row=2, column=col["出带TTFT波动范围"]).value == "+100.0%"
    assert averages.cell(row=2, column=col["ISL入带数"]).value == 1
    assert averages.cell(row=2, column=col["ISL出带数"]).value == 1

    # 8K overall row without matching records: blank, never a fabricated 0.
    for name in (
        "TTFT均值(ms)",
        "TG TPS均值",
        "E2E均值(ms)",
        "出带TTFT波动范围",
    ):
        assert averages.cell(row=3, column=col[name]).value is None
    # The in-band calibre of that row comes from the aggregates (never blank).
    assert averages.cell(row=3, column=col["TTFT均值(ms)（±10%过滤）"]).value == 50.0

    # The calibre note is one short trailing line, not an explanation sheet.
    note_text = averages.cell(row=4, column=1).value
    assert isinstance(note_text, str)
    assert note_text.startswith("注：")
    assert "全样本" in note_text and "入带" in note_text
    assert workbook.sheetnames == ["硬件配置", "模型性能平均", "详细数据"]

    # The out-of-band record itself is annotated on the details sheet: the
    # ±10% fallback (no context_meta) marks it out of band at +11.7%.
    details = workbook["详细数据"]
    detail_headers = [cell.value for cell in details[1]]
    id_col = detail_headers.index("样本ID")
    for row in details.iter_rows(min_row=2):
        if row[id_col].value == "sample-3":
            values = {detail_headers[i]: cell.value for i, cell in enumerate(row)}
            assert values["ISL入带"] is False
            assert values["ISL偏差%"] == 11.7
            break
    else:
        raise AssertionError("out-of-band sample row not found")


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
    # header + one overall row + the one-line calibre note
    assert averages.max_row == 3
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


def test_batch_excel_export_failure_warns_but_never_fails_the_run(
    tmp_path, monkeypatch, capsys
):
    """T-068: a broken Excel export must not fail the benchmark run itself.

    The failure is loud (warning in batch.json + stderr, AGENTS #9) and the
    batch summary still returns; ``exports`` records the miss so the Web/CLI
    can point at ``modelbench export-analysis`` for regeneration.
    """
    from modelbench.adapters.storage import excel_report

    def broken(*args, **kwargs):
        raise RuntimeError("openpyxl exploded")

    monkeypatch.setattr(excel_report, "write_batch_excel", broken)
    events = []
    runner = _batch_runner(
        tmp_path,
        "boom",
        models=["minicpm5-1b"],
        progress_cb=lambda event, payload: events.append(event),
    )
    result = runner.run()  # must not raise
    assert result["batch_id"] == "boom"
    assert "export_file" not in result
    assert result["exports"] is None
    assert any(
        warning.startswith("excel export failed") for warning in result["warnings"]
    )
    assert "excel export failed" in capsys.readouterr().err
    batch_json = json.loads(
        layout.batch_path(tmp_path, "boom").read_text(encoding="utf-8")
    )
    assert any(
        warning.startswith("excel export failed") for warning in batch_json["warnings"]
    )
    assert "exports" not in batch_json or batch_json["exports"] is None
    assert not list((tmp_path / layout.EXPORTS).glob("*.xlsx"))
    assert events.count("batch_done") == 1
