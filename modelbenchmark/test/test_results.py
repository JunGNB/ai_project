"""Tests for the schema-3.0 catalog, storage tools and CLI plumbing.

These cover the flat output layout (``runs/<run_id>/`` + ``batches/<id>.json``),
the index (``output/index.json`` schema 2.0), the derived-artifact tools
(``rebuild``/``verify``/``gc``/``pack``) and the CLI surface that exposes them.
Bundles are produced by the real ``RunWriter`` so the tests exercise the same
files the product writes instead of hand-rolled JSON.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from modelbench.adapters.config.testsets import TestSet as TSet
from modelbench.adapters.config.testsets import TestSetScenario as TScen
from modelbench.adapters.storage import catalog, layout, tools
from modelbench.adapters.storage.writer import RunWriter
from modelbench.domain.models import (
    CacheState,
    ChatMessage,
    PerfMetrics,
    Phase,
    RawResponse,
    RecordStatus,
    RequestSpec,
    RunManifest,
    SampleRecord,
    Usage,
)

CREATED_AT = "2026-01-01T00:00:00+00:00"
BATCH_ID = "20260101T000000Z-ab12"


# --------------------------------------------------------------------------- #
# fixtures: real schema-3.0 bundles (flat layout)
# --------------------------------------------------------------------------- #


def _record(
    run_id: str,
    model_id: str,
    sample_id: str,
    *,
    scenario: str = "knowledge_qa",
    dataset: str = "mmlu",
    context_length: int = 1024,
    status: RecordStatus = RecordStatus.OK,
    completion_tokens: int = 20,
    ttft_ms: float = 100.0,
) -> SampleRecord:
    return SampleRecord(
        run_id=run_id,
        model=model_id,
        scenario=scenario,
        dataset=dataset,
        context_length=context_length,
        sample_id=sample_id,
        repeat=0,
        phase=Phase.MEASURED,
        status=status,
        request=RequestSpec(
            model=model_id,
            # Deliberately independent from the model: the same sample must map
            # to one prompt blob even across models (the whole point of the pool).
            messages=[ChatMessage(role="user", content="prompt " * 10 + sample_id)],
            max_tokens=16,
            actual_prompt_tokens=100,
        ),
        response=RawResponse(
            text="hello world output",
            finish_reason="length",
            usage=Usage(prompt_tokens=100, completion_tokens=completion_tokens),
        ),
        metrics=PerfMetrics(
            ttft_ms=ttft_ms,
            ttft_visible_ms=ttft_ms,
            tpot_ms=10.0,
            gen_tps=20.0,
            e2e_ms=1000.0,
            prefill_tps=900.0,
            total_tps=120.0,
            input_context_tokens=100,
            input_context_chars=200,
            completion_tokens=completion_tokens,
            cached_tokens=0,
            peak_rss_mb=1024.0,
            peak_memory_gb=1.0,
            peak_cpu_pct=120.0,
            mean_cpu_pct=40.0,
        ),
        cache=CacheState(cleared_before_request=True, cached_tokens=0),
    )


def _sample_specs(count: int = 2):
    specs = [
        ("s1", "knowledge_qa", "mmlu", 1024),
        ("s2", "knowledge_qa", "mmlu", 1024),
        ("s3", "math_logic", "gsm8k", 8192),
        ("s4", "coding", "longbench_v2", 32768),
    ]
    return specs[:count]


def _write_batch_file(
    batch_file: Path,
    batch_id: str,
    models: list,
    created_at: str = CREATED_AT,
) -> dict:
    """Write ``batches/<batch_id>.json`` exactly as the flat layout expects."""
    model_runs = [
        {
            "model": item["model_id"],
            "model_id": item["model_id"],
            "model_slug": item["slug"],
            "method": "performance",
            "run_id": item["run_id"],
            # Relative to output/; ``run_dir`` no longer exists in batch.json.
            "run_path": f"{layout.RUNS}/{item['run_id']}",
            "model_size_gb": 5.2,
            "status": "ok",
            "n_records": item["n_records"],
            "n_samples": item["n_samples"],
            "duration_s": 1.0,
        }
        for item in models
    ]
    payload = {
        "schema_version": "3.0",
        "batch_id": batch_id,
        "source": "cli",
        "method": "performance",
        "created_at": created_at,
        "finished_at": created_at,
        "models_selected": [item["model_id"] for item in models],
        "models_measured": [item["model_id"] for item in models],
        "n_models_selected": len(models),
        "n_models_measured": len(models),
        "model_runs": model_runs,
        "inputs": {
            "testset": {
                "path": "dataset/test/performance/testset.jsonl",
                "sha256": "a" * 64,
                "rows": 4,
                "context_windows": [1024, 8192, 32768],
                "decode_guard_version": 1,
            }
        },
        "config": {"perf": {"max_tokens": 16}, "run": {"repeats": 1}},
        "retention": {"pinned": False, "keep_until": None},
        "provenance": {"git_commit": None, "host": {}},
        "warnings": [],
        "errors": [],
    }
    batch_file.parent.mkdir(parents=True, exist_ok=True)
    batch_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def _build_batch(
    root: Path,
    *,
    batch_id: str = BATCH_ID,
    models: tuple = (("m1", "m1"),),
    sample_count: int = 2,
    write_batch_json: bool = True,
    with_log: bool = True,
    finalize: bool = True,
) -> dict:
    """Write a complete schema-3.0 batch under ``root`` using ``RunWriter``."""
    root = Path(root)
    built = {}
    for model_id, slug in models:
        writer = RunWriter(
            root,
            batch_id=batch_id,
            method="performance",
            model_slug=slug,
            created_at=CREATED_AT,
            model={"id": model_id, "alias": slug, "model_slug": slug},
        )
        manifest = RunManifest(
            run_id=writer.run_id,
            batch_id=batch_id,
            method="performance",
            model={"id": model_id, "alias": slug, "model_slug": slug},
            project_version="0.3.0",
            created_at=CREATED_AT,
            config={
                "perf": {"max_tokens": 16, "samples_per_dataset": sample_count},
                "run": {"repeats": 1, "cache_mode": "cold", "warmup_requests": 0},
            },
            env={
                "hardware": {
                    "chip": "M4",
                    "chip_name": "M4",
                    "memory_gb": 24,
                    "gpu_cores": 10,
                    "os_version": "macOS",
                },
                "omlx": {"omlx_version": "1.0", "feature_flags": [], "models": {}},
                "model_env": {},
            },
        )
        writer.initialize(manifest)
        writer.register_model_env(
            {
                "id": model_id,
                "engine_type": "vlm",
                "quantization": "oQ4",
                "estimated_size": 5612040826,
                "settings": {},
            }
        )
        if with_log:
            writer.log_path.write_text("run log line\n", encoding="utf-8")
        records = [
            _record(
                writer.run_id,
                model_id,
                sample_id,
                scenario=scenario,
                dataset=dataset,
                context_length=context_length,
            )
            for sample_id, scenario, dataset, context_length in _sample_specs(sample_count)
        ]
        for record in records:
            writer.write_record(record)
        built[slug] = {
            "writer": writer,
            "records": records,
            "model_id": model_id,
            "run_dir": writer.run_dir,
        }
    batch_file = layout.batch_path(root, batch_id)
    if write_batch_json:
        _write_batch_file(
            batch_file,
            batch_id,
            [
                {
                    "model_id": item["model_id"],
                    "slug": slug,
                    "run_id": item["writer"].run_id,
                    "n_records": len(item["records"]),
                    "n_samples": len(item["records"]),
                }
                for slug, item in built.items()
            ],
        )
    if finalize:
        for slug, item in built.items():
            item["writer"].finalize(
                item["records"],
                perf={"max_tokens": 16},
                run_config={"repeats": 1},
                outcome={"duration_s": 1.0},
            )
    return {
        "root": root,
        "batch_id": batch_id,
        "batch_file": batch_file,
        **built,
    }


# --------------------------------------------------------------------------- #
# catalog: flat layout + index schema 2.0
# --------------------------------------------------------------------------- #


def test_writer_bundle_is_flat_and_indexed(tmp_path):
    built = _build_batch(tmp_path)
    run_dir = built["m1"]["writer"].run_dir
    run_id = built["m1"]["writer"].run_id
    # Flat layout: runs/<run_id>/ and batches/<batch_id>.json, no month dirs.
    assert run_dir.parent == tmp_path / layout.RUNS
    assert run_dir.name == run_id
    assert built["batch_file"] == tmp_path / layout.BATCHES / f"{BATCH_ID}.json"
    assert not (tmp_path / layout.RUNS / "2026-01").exists()
    expected = {
        "run.json",
        "MANIFEST.json",
        "records.jsonl",
        "samples.csv",
        "aggregates.json",
        "report.md",
        "logs/run.log",
    }
    present = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    assert expected <= present

    index = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert index["schema_version"] == "2.0"
    assert index["n_runs"] == 1 and index["n_batches"] == 1
    assert index["n_blobs"]["prompts"] == 2  # two unique sample prompts
    entry = index["runs"][0]
    for key in (
        "run_id",
        "run_path",
        "batch_id",
        "method",
        "mode",
        "model",
        "model_id",
        "model_size_gb",
        "created_at",
        "duration_s",
        "status",
        "cancelled",
        "models_selected",
        "models_measured",
        "n_models_selected",
        "n_models_measured",
        "n_skipped",
        "skipped",
        "memory_errors",
        "n_samples",
        "n_records",
        "n_cells",
        "datasets",
        "bytes_total",
        "bytes_total_human",
        "files",
        "legacy",
    ):
        assert key in entry, key
    assert entry["run_id"] == run_id
    assert entry["run_path"] == f"{layout.RUNS}/{run_id}"
    assert entry["batch_id"] == BATCH_ID
    assert entry["model"] == "m1"
    assert entry["model_size_gb"] == pytest.approx(5.2)
    assert entry["n_samples"] == 2 and entry["n_records"] == 2
    assert entry["legacy"] is False
    assert entry["files"]["records.jsonl"]["sha256"]
    assert entry["files"]["records.jsonl"]["bytes"] > 0
    assert entry["datasets"] == ["mmlu"]
    batch = index["batches"][0]
    assert batch["path"] == f"{layout.BATCHES}/{BATCH_ID}.json"
    assert batch["models"][0]["run_path"] == f"{layout.RUNS}/{run_id}"


def test_index_stores_only_paths_relative_to_output_root(tmp_path):
    _build_batch(tmp_path)
    text = (tmp_path / "index.json").read_text(encoding="utf-8")
    assert str(tmp_path.resolve()) not in text
    assert "output_dir" not in text
    index = catalog.load_index(tmp_path)
    for entry in index["runs"]:
        assert entry["run_path"] == f"{layout.RUNS}/{entry['run_id']}"
    for batch in index["batches"]:
        assert batch["path"] == f"{layout.BATCHES}/{batch['batch_id']}.json"

    # The whole output tree can be moved; the index keeps resolving.
    moved = tmp_path / "moved"
    shutil.copytree(tmp_path, moved)
    index_moved = catalog.load_index(moved)
    run_id = index_moved["runs"][0]["run_id"]
    assert catalog.resolve_run(moved, run_id).is_dir()
    assert catalog.resolve_run(moved, run_id) == moved / layout.RUNS / run_id


def test_index_rebuild_is_stable_except_generated_at(tmp_path):
    _build_batch(tmp_path)
    before = catalog.load_index(tmp_path)
    (tmp_path / "index.json").unlink()
    after = catalog.rebuild_index(tmp_path)
    assert after["schema_version"] == before["schema_version"]
    assert after["runs"] == before["runs"]
    assert after["batches"] == before["batches"]
    assert after["n_blobs"] == before["n_blobs"]


def test_batches_are_grouped_from_batch_files(tmp_path):
    built = _build_batch(tmp_path, models=(("m1", "m1"), ("m2", "m2")))
    index = catalog.rebuild_index(tmp_path)
    assert index["n_runs"] == 2 and index["n_batches"] == 1
    batch = index["batches"][0]
    assert batch["batch_id"] == BATCH_ID
    assert batch["path"] == f"{layout.BATCHES}/{BATCH_ID}.json"
    assert batch["methods"] == ["performance"]
    assert batch["n_models"] == 2
    assert {item["run_id"] for item in batch["models"]} == {
        built["m1"]["writer"].run_id,
        built["m2"]["writer"].run_id,
    }
    by_id = {entry["run_id"]: entry for entry in index["runs"]}
    assert by_id[built["m1"]["writer"].run_id]["batch_index"] in (0, 1)
    assert by_id[built["m1"]["writer"].run_id]["batch_models_total"] == 2
    # Same sample prompts across both models => still one blob per prompt.
    assert index["n_blobs"]["prompts"] == 2
    # Members resolve through batch.json instead of directory nesting.
    members = catalog.run_dirs_in_batch(tmp_path, catalog.load_batch(built["batch_file"]))
    assert {path.name for path in members} == {
        built["m1"]["writer"].run_id,
        built["m2"]["writer"].run_id,
    }


def test_upsert_run_then_upsert_batch(tmp_path):
    # Finalize a run *before* the batch file exists: writers call upsert_run on
    # finalize while BatchRunner only writes batches/<id>.json at the very end.
    built = _build_batch(tmp_path, batch_id=BATCH_ID, write_batch_json=False)
    index = catalog.load_index(tmp_path)
    assert index["n_runs"] == 1
    assert index["runs"][0]["batch_id"] == BATCH_ID
    assert index["runs"][0]["batch_index"] is None
    assert index["batches"][0]["path"] is None

    batch_file = built["batch_file"]
    _write_batch_file(
        batch_file,
        BATCH_ID,
        [
            {
                "model_id": "m1",
                "slug": "m1",
                "run_id": built["m1"]["writer"].run_id,
                "n_records": 2,
                "n_samples": 2,
            }
        ],
    )
    catalog.upsert_batch(tmp_path, batch_file)
    index = catalog.load_index(tmp_path)
    assert index["n_runs"] == 1
    assert index["runs"][0]["batch_index"] == 0
    assert index["batches"][0]["n_models"] == 1
    assert index["batches"][0]["path"] == f"{layout.BATCHES}/{BATCH_ID}.json"


def test_run_without_aggregates_is_still_catalogued(tmp_path):
    built = _build_batch(tmp_path, write_batch_json=True, finalize=False)
    index = catalog.rebuild_index(tmp_path)
    assert index["n_runs"] == 1
    entry = index["runs"][0]
    assert entry["run_id"] == built["m1"]["writer"].run_id
    assert entry["run_path"] == f"{layout.RUNS}/{built['m1']['writer'].run_id}"
    assert entry["n_samples"] is None
    assert entry["n_records"] is None
    # Without run.json.outcome the batch metadata is the only status source.
    assert entry["status"] == "ok"
    assert entry["files"] == {}


def test_index_ignores_archived_legacy_bundles(tmp_path):
    """Schema 2.x bundles under archive/ are GC roots but never indexed."""
    legacy = tmp_path / "archive" / "legacy-v2" / "web-old_performance_m1"
    legacy.mkdir(parents=True)
    (legacy / "summary.json").write_text(
        json.dumps({"schema_version": "2.1"}), encoding="utf-8"
    )
    index = catalog.rebuild_index(tmp_path)
    assert index["n_runs"] == 0 and index["batches"] == []

    _build_batch(tmp_path)
    index = catalog.rebuild_index(tmp_path)
    assert index["n_runs"] == 1
    assert all("legacy-v2" not in entry["run_path"] for entry in index["runs"])


def test_resolve_run_and_batch_helpers(tmp_path):
    built = _build_batch(tmp_path)
    run_id = built["m1"]["writer"].run_id
    assert catalog.resolve_run(tmp_path, run_id) == built["m1"]["writer"].run_dir
    with pytest.raises(KeyError):
        catalog.resolve_run(tmp_path, "20260101T000000Z-ab12__performance__nope")
    with pytest.raises(ValueError):
        catalog.resolve_run(tmp_path, "../" + run_id)
    batch = catalog.load_batch(built["batch_file"])
    assert batch["batch_id"] == BATCH_ID
    assert catalog.load_batch(tmp_path / "missing.json") is None
    assert catalog.find_batch_file(tmp_path, BATCH_ID) == built["batch_file"]
    assert catalog.find_batch_file(tmp_path, "unknown") is None
    assert (
        catalog.batch_file_for_run(tmp_path, built["m1"]["writer"].run_dir)
        == built["batch_file"]
    )
    assert [path.name for path in catalog.iter_run_dirs(tmp_path)] == [run_id]
    assert catalog.iter_batch_files(tmp_path) == [built["batch_file"]]


def test_viewable_files_and_run_file_listing(tmp_path):
    built = _build_batch(tmp_path)
    run_dir = built["m1"]["writer"].run_dir
    assert catalog.VIEWABLE_FILES == (
        "run.json",
        "MANIFEST.json",
        "records.jsonl",
        "samples.csv",
        "aggregates.json",
        "report.md",
        "logs/run.log",
    )
    listing = {item["name"]: item for item in catalog.run_file_listing(run_dir)}
    assert set(listing) == set(catalog.VIEWABLE_FILES)
    assert listing["records.jsonl"]["hidden"] is True
    assert listing["run.json"]["hidden"] is False
    assert catalog.resolve_run_file(run_dir, "run.json").is_file()
    with pytest.raises(ValueError):
        catalog.resolve_run_file(run_dir, "../run.json")
    with pytest.raises(ValueError):
        catalog.resolve_run_file(run_dir, "raw/whatever.json")
    (run_dir / "report.md").unlink()
    with pytest.raises(FileNotFoundError):
        catalog.resolve_run_file(run_dir, "report.md")


def test_load_records_restores_messages_placeholder(tmp_path):
    built = _build_batch(tmp_path, models=(("m1", "m1"), ("m2", "m2")))
    run_dir = built["m1"]["writer"].run_dir
    records = catalog.load_records(run_dir)
    assert len(records) == 2
    assert all(record.request.messages == [] for record in records)
    assert all(record.request.prompt_sha256 for record in records)
    # Prompt bodies are deduplicated across models by content address.
    assert len(catalog.prompt_sha256s(run_dir)) == 2
    assert set(catalog.prompt_sha256s(run_dir)) == set(
        catalog.prompt_sha256s(built["m2"]["writer"].run_dir)
    )


# --------------------------------------------------------------------------- #
# tools: rebuild / verify
# --------------------------------------------------------------------------- #


def test_rebuild_restores_derived_files(tmp_path):
    built = _build_batch(tmp_path)
    run_dir = built["m1"]["writer"].run_dir
    original_samples = (run_dir / "samples.csv").read_text(encoding="utf-8")
    original_aggregates = json.loads((run_dir / "aggregates.json").read_text())
    for name in ("samples.csv", "aggregates.json", "report.md", "MANIFEST.json"):
        (run_dir / name).unlink()
    assert not tools.verify_output(tmp_path)["ok"]

    result = tools.rebuild_run(tmp_path, run_dir)
    assert result["n_records"] == 2
    assert result["n_samples"] == 2
    for name in ("samples.csv", "aggregates.json", "report.md", "MANIFEST.json"):
        assert (run_dir / name).is_file(), name
    rebuilt_aggregates = json.loads((run_dir / "aggregates.json").read_text())
    for key in ("n_records", "n_samples", "n_cells", "overall", "cells"):
        assert rebuilt_aggregates[key] == original_aggregates[key]
    from modelbench.adapters.storage.writer import SAMPLES_CSV_FIELDS

    header = (run_dir / "samples.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == ",".join(SAMPLES_CSV_FIELDS)
    assert (run_dir / "samples.csv").read_text(encoding="utf-8") == original_samples
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert built["m1"]["writer"].run_id in report
    assert tools.verify_output(tmp_path)["ok"]


def test_verify_detects_missing_blob_and_bad_manifest(tmp_path):
    built = _build_batch(tmp_path)
    assert tools.verify_output(tmp_path)["ok"]

    # 1) a referenced prompt blob disappears
    run_dir = built["m1"]["writer"].run_dir
    sha = catalog.prompt_sha256s(run_dir)[0]
    blob = catalog.PromptBlobStore(tmp_path).path_for(sha)
    assert blob.is_file()
    blob.unlink()
    report = tools.verify_output(tmp_path)
    assert not report["ok"]
    assert report["stats"]["blob_missing"] == 1
    assert any("blob" in issue for issue in report["issues"])

    # 2) MANIFEST digest no longer matches the bytes on disk
    manifest_path = run_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = tools.verify_output(tmp_path)
    assert not report["ok"]
    assert any("sha256 mismatch" in issue for issue in report["issues"])


def test_verify_detects_outcome_mismatch(tmp_path):
    built = _build_batch(tmp_path)
    run_dir = built["m1"]["writer"].run_dir
    path = run_dir / "run.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["outcome"]["n_measured"] = 99
    payload["outcome"]["n_selected"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = tools.verify_output(tmp_path)
    assert not report["ok"]
    assert any("n_measured" in issue for issue in report["issues"])
    assert any("n_selected" in issue for issue in report["issues"])


def test_verify_flags_run_not_claimed_by_any_batch(tmp_path):
    built = _build_batch(tmp_path, write_batch_json=False)
    report = tools.verify_output(tmp_path)
    assert not report["ok"]
    assert any("not referenced by any batches" in issue for issue in report["issues"])
    # Writing the batch file makes the tree consistent again.
    _write_batch_file(
        built["batch_file"],
        BATCH_ID,
        [
            {
                "model_id": "m1",
                "slug": "m1",
                "run_id": built["m1"]["writer"].run_id,
                "n_records": 2,
                "n_samples": 2,
            }
        ],
    )
    assert tools.verify_output(tmp_path)["ok"]


def test_verify_flags_preflatten_leftovers(tmp_path):
    # The earlier nested layout ran an extra migration step; verify must not
    # silently ignore such leftovers under runs/.
    nested = (
        tmp_path / layout.RUNS / "2026-11" / "old-batch" / "performance" / "m1"
    )
    nested.mkdir(parents=True)
    (nested / "run.json").write_text("{}", encoding="utf-8")
    report = tools.verify_output(tmp_path)
    assert not report["ok"]
    assert any(
        "unexpected directory under runs/" in issue for issue in report["issues"]
    )


def test_verify_empty_output_root_is_ok(tmp_path):
    report = tools.verify_output(tmp_path)
    assert report["ok"] is True
    assert report["issues"] == []
    assert report["stats"]["runs"] == 0


# --------------------------------------------------------------------------- #
# tools: gc
# --------------------------------------------------------------------------- #


def test_gc_keeps_blobs_referenced_from_archive(tmp_path):
    built = _build_batch(tmp_path)
    run_dir = built["m1"]["writer"].run_dir
    referenced = set(catalog.prompt_sha256s(run_dir))
    # Retire the run + its batch file into archive/ (still an evidence root).
    archived_run = tmp_path / "archive" / "runs" / run_dir.name
    archived_batch = tmp_path / "archive" / "batches" / built["batch_file"].name
    archived_run.parent.mkdir(parents=True)
    archived_batch.parent.mkdir(parents=True)
    shutil.move(str(run_dir), str(archived_run))
    shutil.move(str(built["batch_file"]), str(archived_batch))
    catalog.rebuild_index(tmp_path)
    assert catalog.iter_run_dirs(tmp_path) == []
    assert catalog.iter_batch_files(tmp_path) == []

    # Add one genuinely orphaned blob.
    orphan_sha = catalog.PromptBlobStore(tmp_path).put(
        {"messages": [{"role": "user", "content": "orphan"}], "chars": 6}
    )
    result = tools.gc_blobs(tmp_path)
    assert result["ok"] is True
    assert result["deleted_count"] == 1
    assert result["deleted"] == [orphan_sha]
    store = catalog.PromptBlobStore(tmp_path)
    assert not store.exists(orphan_sha)
    assert all(store.exists(sha) for sha in referenced)


def test_gc_dry_run_does_not_delete(tmp_path):
    _build_batch(tmp_path)
    orphan_sha = catalog.PromptBlobStore(tmp_path).put(
        {"messages": [{"role": "user", "content": "orphan two"}], "chars": 10}
    )
    result = tools.gc_blobs(tmp_path, dry_run=True)
    assert result["dry_run"] is True
    assert result["deleted_count"] == 1
    assert catalog.PromptBlobStore(tmp_path).exists(orphan_sha)
    # ... and a real run removes it.
    assert tools.gc_blobs(tmp_path)["deleted_count"] == 1
    assert not catalog.PromptBlobStore(tmp_path).exists(orphan_sha)


def test_gc_refuses_without_reference_roots(tmp_path):
    with pytest.raises(RuntimeError, match="refusing to GC"):
        tools.gc_blobs(tmp_path)
    (tmp_path / layout.RUNS).mkdir()
    assert tools.gc_blobs(tmp_path)["ok"] is True


def test_gc_refuses_when_blob_refs_are_missing(tmp_path):
    built = _build_batch(tmp_path)
    run_dir = built["m1"]["writer"].run_dir
    sha = catalog.prompt_sha256s(run_dir)[0]
    catalog.PromptBlobStore(tmp_path).path_for(sha).unlink()
    result = tools.gc_blobs(tmp_path)
    assert result["ok"] is False and result["refused"] is True
    assert result["deleted_count"] == 0


# --------------------------------------------------------------------------- #
# tools: pack
# --------------------------------------------------------------------------- #


def test_pack_batch_roundtrip_verifies_in_fresh_root(tmp_path):
    built = _build_batch(tmp_path, models=(("m1", "m1"), ("m2", "m2")))
    zip_path = tmp_path / "exports" / f"{BATCH_ID}.zip"
    result = tools.pack_batch(tmp_path, BATCH_ID, zip_path)
    assert result == zip_path and zip_path.is_file()

    import zipfile

    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
    assert f"{layout.BATCHES}/{BATCH_ID}.json" in names
    assert any(
        name.startswith(f"{layout.RUNS}/{built['m1']['writer'].run_id}/")
        for name in names
    )
    assert any(name.startswith("blobs/prompts/") for name in names)

    unpacked = tmp_path / "unpacked"
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(unpacked)
    report = tools.verify_output(unpacked)
    assert report["ok"], report["issues"]
    assert report["stats"]["runs"] == 2
    index = catalog.rebuild_index(unpacked)
    assert index["n_batches"] == 1 and index["n_runs"] == 2
    assert index["batches"][0]["path"] == f"{layout.BATCHES}/{BATCH_ID}.json"


def test_pack_archive_written_inside_run_dir_is_not_self_included(tmp_path):
    built = _build_batch(tmp_path)
    out = built["m1"]["writer"].run_dir / "self.zip"
    tools.pack_batch(tmp_path, BATCH_ID, out)
    import zipfile

    with zipfile.ZipFile(out) as archive:
        assert not any(name.endswith("self.zip") for name in archive.namelist())


def test_pack_refuses_unknown_batch_and_missing_blobs(tmp_path):
    built = _build_batch(tmp_path)
    with pytest.raises(KeyError):
        tools.pack_batch(tmp_path, "does-not-exist", tmp_path / "x.zip")
    sha = catalog.prompt_sha256s(built["m1"]["writer"].run_dir)[0]
    catalog.PromptBlobStore(tmp_path).path_for(sha).unlink()
    with pytest.raises(FileNotFoundError):
        tools.pack_batch(tmp_path, BATCH_ID, tmp_path / "y.zip")


# --------------------------------------------------------------------------- #
# CLI surface
# --------------------------------------------------------------------------- #


def _subparsers(parser):
    import argparse

    return next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )


def test_cli_registers_all_subcommands_and_flags():
    from modelbench.interfaces import cli

    parser = cli.build_parser()
    commands = _subparsers(parser).choices
    assert {
        "run",
        "validate",
        "hardware",
        "list",
        "download",
        "analyze",
        "runs",
        "index",
        "export-analysis",
        "web",
    } <= set(commands)
    assert {"gc", "verify", "rebuild", "pack"} <= set(commands)

    helps = {name: command.format_help() for name, command in commands.items()}
    assert "--dry-run" in helps["gc"]
    assert "--json" in helps["verify"]
    assert "--json" in helps["rebuild"]
    assert "--out" in helps["pack"]
    assert "--run-id" in helps["run"]
    assert "--paired" in helps["analyze"]
    # The new commands must be parseable end-to-end.
    assert parser.parse_args(["gc", "--dry-run"]).func is cli.cmd_gc
    assert parser.parse_args(["verify"]).func is cli.cmd_verify
    assert parser.parse_args(["rebuild"]).run_id is None
    assert parser.parse_args(["rebuild", "r-1"]).run_id == "r-1"
    assert parser.parse_args(["pack", "b-1", "--out", "x.zip"]).out == "x.zip"
    assert parser.parse_args(["analyze", "a", "b", "--paired"]).paired is True


def test_cli_runs_reads_index_schema_2(tmp_path, capsys):
    from modelbench.interfaces import cli

    built = _build_batch(tmp_path)
    assert cli.main(["runs", "--output-dir", str(tmp_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "2.0"
    assert payload["n_runs"] == 1
    assert payload["runs"][0]["run_id"] == built["m1"]["writer"].run_id
    assert (
        payload["runs"][0]["run_path"]
        == f"{layout.RUNS}/{built['m1']['writer'].run_id}"
    )

    assert cli.main(["runs", "--output-dir", str(tmp_path)]) == 0
    text = capsys.readouterr().out
    assert BATCH_ID in text
    assert built["m1"]["writer"].run_id in text


def test_cli_verify_gc_and_rebuild(tmp_path, capsys):
    from modelbench.interfaces import cli

    built = _build_batch(tmp_path)
    run_dir = built["m1"]["writer"].run_dir
    assert cli.main(["verify", "--output-dir", str(tmp_path)]) == 0
    assert "verify OK" in capsys.readouterr().out

    orphan = catalog.PromptBlobStore(tmp_path).put(
        {"messages": [{"role": "user", "content": "cli orphan"}], "chars": 10}
    )
    assert cli.main(["gc", "--dry-run", "--json", "--output-dir", str(tmp_path)]) == 0
    gc_report = json.loads(capsys.readouterr().out)
    assert gc_report["deleted_count"] == 1
    assert catalog.PromptBlobStore(tmp_path).exists(orphan)

    for name in ("samples.csv", "aggregates.json", "report.md", "MANIFEST.json"):
        (run_dir / name).unlink()
    assert cli.main(["verify", "--output-dir", str(tmp_path)]) == 1
    assert (
        cli.main(
            ["rebuild", built["m1"]["writer"].run_id, "--output-dir", str(tmp_path)]
        )
        == 0
    )
    assert "rebuilt" in capsys.readouterr().out
    assert cli.main(["verify", "--output-dir", str(tmp_path)]) == 0


def test_cli_rebuild_all_and_pack(tmp_path, capsys):
    from modelbench.interfaces import cli

    _build_batch(tmp_path)
    assert cli.main(["rebuild", "--output-dir", str(tmp_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["index"]["n_runs"] == 1
    assert len(payload["rebuilt"]) == 1

    out = tmp_path / "exports" / f"{BATCH_ID}.zip"
    assert (
        cli.main(["pack", BATCH_ID, "--out", str(out), "--output-dir", str(tmp_path)])
        == 0
    )
    assert out.is_file()
    assert (
        cli.main(
            ["pack", "missing-batch", "--out", str(out), "--output-dir", str(tmp_path)]
        )
        == 1
    )


# --------------------------------------------------------------------------- #
# web-console helpers that do not depend on the run bundle layout
# --------------------------------------------------------------------------- #


def test_scenario_samples_endpoint_uses_cache(monkeypatch, tmp_path):
    from modelbench.adapters.config.settings import load_config
    from modelbench.interfaces.web import server as web

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
    from modelbench.interfaces.web import server as web

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)

    monkeypatch.setattr(web.subprocess, "run", fake_run)
    monkeypatch.setattr(web.sys, "platform", "darwin")
    web._open_in_file_manager(Path("/tmp/run"))
    assert calls == [["open", "/tmp/run"]]


def test_sample_worker_preview_and_large_isolation(monkeypatch, tmp_path):
    from modelbench.interfaces.web import sample_worker
    from modelbench.interfaces.web import server as web

    preview = sample_worker.build_preview(
        str(tmp_path / "raw"), str(tmp_path / "processed"), "synthetic", {"count": 2}
    )
    assert preview["total"] == 2
    assert [row["sample_id"] for row in preview["samples"]] == [
        "synthetic-0000",
        "synthetic-0001",
    ]

    calls = []
    monkeypatch.setattr(
        web, "_dataset_raw_bytes", lambda raw_dir, dataset: 999 * 1024 * 1024
    )
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


def test_paired_comparison_uses_common_sample_ids(tmp_path):
    from modelbench.interfaces.analysis.compare import (
        paired_comparison,
        render_paired,
    )

    header = "scenario,dataset,context_length,sample_id,repeat,status,ttft_ms,gen_tps,tpot_ms,e2e_ms,peak_memory_gb,input_context_tokens\n"
    run_a = tmp_path / "runs" / "a"
    run_b = tmp_path / "runs" / "b"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)
    (run_a / "samples.csv").write_text(
        header
        + "coding,d,1024,s1,0,ok,100,50,10,200,1.0,1024\n"
        + "coding,d,1024,s2,0,ok,120,40,12,240,1.2,1024\n"
        + "coding,d,1024,s3,0,ok,110,45,11,220,1.1,1024\n",
        encoding="utf-8",
    )
    (run_b / "samples.csv").write_text(
        header
        + "coding,d,1024,s1,0,ok,90,60,9,180,0.9,1024\n"
        + "coding,d,1024,s2,0,ok,100,50,10,200,1.0,1024\n"
        + "coding,d,1024,s4,0,ok,130,30,13,260,1.3,1024\n",  # s3/s4 not paired
        encoding="utf-8",
    )
    result = paired_comparison(run_a, run_b, metrics=["ttft_ms", "gen_tps"])
    assert result["n_common"] == 2
    ttft = result["metrics"]["ttft_ms"]
    assert ttft["n"] == 2
    assert ttft["mean_a"] == pytest.approx(110.0)
    assert ttft["mean_b"] == pytest.approx(95.0)
    assert ttft["delta"] == pytest.approx(-15.0)
    assert ttft["ci95_low"] <= ttft["delta"] <= ttft["ci95_high"]
    text = render_paired(result)
    assert "Paired comparison" in text and "ttft_ms" in text


def test_verify_detects_stale_aggregates(tmp_path):
    built = _build_batch(tmp_path)
    run_dir = built["m1"]["writer"].run_dir
    path = run_dir / "aggregates.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["overall"][0]["perf"]["ttft_ms"]["mean"] = 999999.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = tools.verify_output(tmp_path)
    assert not report["ok"]
    assert any("ttft_ms" in issue and "recomputed" in issue for issue in report["issues"])
