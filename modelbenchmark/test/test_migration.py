"""Migration tests: schema-2.1 legacy bundles -> schema-3.0 flat runs.

Contract (``tool/migrate/legacy_v2_to_v3.py``):

- migrated records carry ``schema_version="3.0"`` (the legacy version moves to
  ``sample_meta.migrated_from_schema``);
- prompt bodies are externalized into ``output/blobs/prompts/``;
- legacy ``summary.json`` means are reproduced by ``aggregates.json``;
- the flat layout is written under ``output/runs/<batch>__performance__<slug>``.

The fixture builds the smallest possible legacy bundle in ``tmp_path``; the
real archived bundles under ``output/archive/legacy-v2/`` are never touched.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "tool" / "migrate") not in sys.path:
    sys.path.insert(0, str(ROOT / "tool" / "migrate"))

from legacy_v2_to_v3 import migrate  # noqa: E402


def _legacy_record(schema: str, sample_id: str, ttft_ms: float) -> dict:
    """One schema-2.1 record: prompt inline, metrics flat, no env snapshot."""
    return {
        "schema_version": schema,
        "run_id": "web-testbatch-0001_performance_test-model-1b",
        "timestamp": "2026-11-01T00:00:00+00:00",
        "model": "TestModel-1B",
        "model_config_snapshot": {"id": "TestModel-1B", "loaded": True},
        "scenario": "knowledge_qa",
        "context_length": 1024,
        "dataset": "longbench_v2:Single-Document QA",
        "sample_id": sample_id,
        "repeat": 0,
        "phase": "measured",
        "status": "ok",
        "error": None,
        "request": {
            "model": "TestModel-1B",
            "messages": [
                {"role": "user", "content": "这是一条用于迁移测试的 1K 知识问答样本。"}
            ],
            "max_tokens": 50,
            "temperature": 0.0,
            "top_p": None,
            "seed": None,
            "stream": True,
            "tools": None,
            "extra": {},
            "chat_template_kwargs": None,
            "target_context_tokens": 1024,
            "actual_prompt_tokens": 1013,
            "truncated": False,
            "padded": False,
            "context_meta": {},
        },
        "response": {
            "text": "总结。",
            "reasoning_text": "",
            "tool_calls": [],
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": 1013,
                "completion_tokens": 30,
                "total_tokens": 1043,
            },
            "num_chunks": 30,
            "first_token_ts": 0.25,
            "first_visible_ts": 0.25,
            "last_token_ts": 0.5,
            "ttft_provider_reported": None,
        },
        "metrics": {
            "ttft_ms": ttft_ms,
            "ttft_visible_ms": ttft_ms,
            "tpot_ms": 7.0,
            "e2e_ms": 250.0,
            "gen_tps": 120.0,
            "prefill_tps": 3000.0,
            "total_tps": 130.0,
            "input_context_tokens": 1013,
            "input_context_chars": 400,
            "peak_cpu_pct": 90.0,
            "mean_cpu_pct": 80.0,
            "cpu_seconds": 0.2,
            "peak_rss_mb": 1200.0,
            "peak_memory_gb": 1.2,
            "cpu_samples": 5,
            "cpu_sample_window_ms": 50,
            "prompt_tokens": 1013,
            "completion_tokens": 30,
            "cached_tokens": 0,
        },
        "score": None,
        "cache": {
            "mode": "cold",
            "cleared_before_request": True,
            "clear_latency_ms": 10.0,
            "cached_tokens": 0,
            "polluted": False,
            "note": "hot+ssd cleared",
        },
        "sample_meta": {
            "dataset_meta": {"workload": "longbench_v2"},
            "target_context_tokens": 1024,
        },
    }


def _write_legacy_bundle(archive: Path, records: list) -> Path:
    run_dir = archive / "web-testbatch-0001_performance_test-model-1b"
    run_dir.mkdir(parents=True)
    (run_dir / "records.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "2.1",
                "run_id": run_dir.name,
                "created_at": "2026-11-01T00:00:00+00:00",
                "project_version": "0.2.9",
                "mode": "performance",
                "config_snapshot": {"perf": {"max_tokens": 50}, "run": {"repeats": 1}},
                "hardware": {"chip": "TestChip"},
                "omlx": {"version": "test"},
                "server": {},
                "models": {},
                "scenarios": ["knowledge_qa"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_dir / "hardware.json").write_text('{"chip": "TestChip"}', encoding="utf-8")
    (run_dir / "omlx.json").write_text('{"version": "test"}', encoding="utf-8")
    (run_dir / "models.json").write_text('{"loaded": {}}', encoding="utf-8")
    means = {
        m["metrics"]["ttft_ms"] for m in records if m.get("phase") == "measured"
    }
    legacy_mean = sum(means) / len(means) if len(means) > 1 else next(iter(means))
    n = len([m for m in records if m.get("phase") == "measured"])
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "2.1",
                "run_id": run_dir.name,
                "n_records": len(records),
                "n_samples": n,
                "n_cells": 1,
                "duration_s": 12.5,
                "overall": [
                    {
                        "model": "TestModel-1B",
                        "context_length": 1024,
                        "n_samples": n,
                        "n_valid": n,
                        "perf": {
                            "ttft_ms": {"n": n, "mean": legacy_mean},
                        },
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (archive / "batches").mkdir(exist_ok=True)
    (archive / "batches" / "web-testbatch-0001.json").write_text(
        json.dumps(
            {
                "batch_id": "testbatch-0001",
                "method": "performance",
                "created_at": "2026-11-01T00:00:00+00:00",
                "models_tested": ["TestModel-1B"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return run_dir


def test_migration_emits_schema3_and_provenance(tmp_path):
    """Migration output: schema_version=3.0 + migrated_from_schema non-empty."""
    archive = tmp_path / "legacy-v2"
    records = [
        _legacy_record("2.1", "perf-knowledge_qa-1k-000", 100.0),
        _legacy_record("2.1", "perf-knowledge_qa-1k-001", 200.0),
    ]
    _write_legacy_bundle(archive, records)

    out_root = tmp_path / "output"
    result = migrate(archive, out_root)

    assert result["runs"], "migration produced no runs"
    run_rel = result["runs"][0]["run_path"]
    assert run_rel == "runs/testbatch-0001__performance__test-model-1b"
    run_dir = out_root / run_rel

    migrated = [
        json.loads(line)
        for line in (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(migrated) == 2
    for rec in migrated:
        assert rec["schema_version"] == "3.0"
        prov = (rec.get("sample_meta") or {}).get("migrated_from_schema")
        assert prov, "migrated_from_schema must be non-empty"
        assert prov == "2.1"
        # Schema-3.0: prompt body externalized, only the digest stays inline.
        assert not (rec["request"].get("messages"))
        assert rec["request"]["prompt_sha256"]


def test_migration_reproduces_legacy_aggregate_means(tmp_path):
    """Fidelity: aggregates.json overall means equal legacy summary.json means."""
    archive = tmp_path / "legacy-v2"
    records = [
        _legacy_record("2.1", "perf-knowledge_qa-1k-000", 100.0),
        _legacy_record("2.1", "perf-knowledge_qa-1k-001", 200.0),
    ]
    _write_legacy_bundle(archive, records)

    out_root = tmp_path / "output"
    migrate(archive, out_root)
    run_dir = out_root / "runs" / "testbatch-0001__performance__test-model-1b"
    agg = json.loads((run_dir / "aggregates.json").read_text(encoding="utf-8"))
    overall = {
        (row["model"], row["context_length"]): row for row in agg.get("overall", [])
    }
    row = overall[("TestModel-1B", 1024)]
    assert row["perf"]["ttft_ms"]["mean"] == 150.0
    assert row["perf"]["ttft_ms"]["n"] == 2


def test_migration_batch_metadata_is_schema3(tmp_path):
    """Batch metadata file is written with schema_version 3.0."""
    archive = tmp_path / "legacy-v2"
    records = [_legacy_record("2.1", "perf-knowledge_qa-1k-000", 100.0)]
    _write_legacy_bundle(archive, records)

    out_root = tmp_path / "output"
    migrate(archive, out_root)
    batch = json.loads(
        (out_root / "batches" / "testbatch-0001.json").read_text(encoding="utf-8")
    )
    assert batch["schema_version"] == "3.0"
    assert batch["model_runs"][0]["run_id"] == (
        "testbatch-0001__performance__test-model-1b"
    )


def test_migration_missing_schema_defaults_to_21(tmp_path):
    """Records without schema_version fall back to the 2.1 provenance marker."""
    archive = tmp_path / "legacy-v2"
    records = [_legacy_record(None, "perf-knowledge_qa-1k-000", 100.0)]
    _write_legacy_bundle(archive, records)

    out_root = tmp_path / "output"
    migrate(archive, out_root)
    run_dir = out_root / "runs" / "testbatch-0001__performance__test-model-1b"
    rec = json.loads(
        (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert rec["schema_version"] == "3.0"
    assert rec["sample_meta"]["migrated_from_schema"] == "2.1"
