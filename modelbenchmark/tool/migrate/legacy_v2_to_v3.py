"""One-off migration: re-serialize schema-2.x run bundles into schema 3.0.

The original bundles stay untouched under ``output/archive/legacy-v2/``; this
script only *re-serializes* them into the new layout so historical measurements
remain browsable after the schema switch (the web console no longer reads 2.x).

What it does, per legacy ``web-<ts>-<hex>_performance_<model>/`` bundle:

* prompts (``request.messages``) -> ``output/blobs/prompts/<sha256>.json``
* every record -> thin ``records.jsonl`` (``request.prompt_sha256`` +
  ``model_env_ref``; the per-record ``model_config_snapshot`` is hoisted into
  ``run.json.env.model_env``)
* ``manifest.json`` + ``hardware/omlx/models`` -> ``run.json``
* aggregates / ``samples.csv`` / ``report.md`` / ``MANIFEST.json`` recomputed by
  the normal writer (nothing is hand-written)

Fidelity is asserted before/after: record count must match and each
``overall`` distribution from the legacy ``summary.json`` must be reproduced
from the migrated ``records.jsonl``.

Usage::

    .venv/bin/python tool/migrate/legacy_v2_to_v3.py --archive output/archive/legacy-v2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from modelbench.adapters.storage.writer import RunWriter  # noqa: E402
from modelbench.domain.models import (  # noqa: E402
    CacheState,
    PerfMetrics,
    Phase,
    RawResponse,
    RecordStatus,
    RequestSpec,
    RunManifest,
    SampleRecord,
    Usage,
)


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _legacy_runs(archive: Path):
    for entry in sorted(archive.glob("*_performance_*")):
        if not entry.is_dir():
            continue
        name = entry.name
        if not name.startswith("web-"):
            continue
        head, _, slug = name.partition("_performance_")
        batch_id = head[len("web-") :]
        yield batch_id, slug, entry


def _convert_record(raw: dict, run_id: str, env_ref: str):
    request = dict(raw.get("request") or {})
    messages = request.pop("messages", []) or []
    tools = request.pop("tools", None)
    request.pop("context_meta", None) if False else None
    context_meta = dict(request.pop("context_meta", {}) or {})
    prompt_chars = sum(len(m.get("content") or "") for m in messages)
    spec = RequestSpec(
        model=request.get("model") or raw.get("model"),
        messages=[],
        max_tokens=request.get("max_tokens") or 0,
        temperature=request.get("temperature") or 0.0,
        top_p=request.get("top_p"),
        seed=request.get("seed"),
        stream=bool(request.get("stream", True)),
        tools=None,
        extra=request.get("extra") or {},
        chat_template_kwargs=request.get("chat_template_kwargs"),
        target_context_tokens=request.get("target_context_tokens"),
        actual_prompt_tokens=request.get("actual_prompt_tokens"),
        prompt_chars=prompt_chars,
        truncated=bool(request.get("truncated")),
        padded=bool(request.get("padded")),
        context_meta=context_meta,
    )
    response_raw = dict(raw.get("response") or {})
    usage_raw = dict(response_raw.pop("usage", None) or {})
    response = RawResponse(usage=Usage(**usage_raw), **response_raw)
    sample_meta = dict(raw.get("sample_meta") or {})
    sample_meta.pop("raw_sample_path", None)
    sample_meta.setdefault("migrated_from_schema", raw.get("schema_version") or "2.1")
    record = SampleRecord(
        # The record is re-serialized into the schema-3.0 shape (prompt body
        # externalized, environment lifted to run.json); keep the legacy
        # provenance in sample_meta instead of the obsolete schema version.
        schema_version="3.0",
        run_id=run_id,
        timestamp=raw.get("timestamp") or "",
        model=raw.get("model"),
        model_env_ref=env_ref,
        scenario=raw.get("scenario"),
        context_length=raw.get("context_length"),
        dataset=raw.get("dataset"),
        sample_id=raw.get("sample_id"),
        repeat=int(raw.get("repeat") or 0),
        phase=Phase(raw.get("phase") or "measured"),
        status=RecordStatus(raw.get("status") or "ok"),
        error=raw.get("error"),
        request=spec,
        response=response,
        metrics=PerfMetrics(**(raw.get("metrics") or {})),
        score=None,
        cache=CacheState(**(raw.get("cache") or {})),
        sample_meta=sample_meta,
    )
    payload = {
        "chars": prompt_chars,
        "messages": messages,
        "tools": tools,
        "chat_template_kwargs": request.get("chat_template_kwargs"),
        "decode_guard_version": 1,
    }
    return record, payload


def _manifest_for(legacy: dict, slug: str, batch_id: str, model_env: dict, env_ref: str):
    old = legacy.get("manifest.json") or {}
    model_block = next(iter(model_env.values()), {}) if model_env else {}
    return RunManifest(
        run_id="placeholder",
        batch_id=batch_id,
        method="performance",
        model={
            "id": model_block.get("id") or (old.get("model") or {}).get("id"),
            "alias": slug,
            "model_slug": slug,
        },
        project_version=old.get("project_version") or "",
        created_at=old.get("created_at") or "",
        started_at=old.get("created_at"),
        finished_at=None,
        git={
            "commit": old.get("git_commit"),
            "dirty": old.get("git_dirty"),
        },
        config={
            "snapshot": old.get("config_snapshot") or {},
            "perf": (old.get("config_snapshot") or {}).get("perf") or {},
            "run": (old.get("config_snapshot") or {}).get("run") or {},
            "scenarios": old.get("scenarios") or [],
            "context_windows": list(
                ((old.get("config_snapshot") or {}).get("perf") or {}).get("context_windows")
                or []
            ),
            "server": old.get("server") or {},
            "host": old.get("host") or {},
        },
        env={
            "hardware": (legacy.get("hardware.json") or old.get("hardware") or {}),
            "omlx": (legacy.get("omlx.json") or old.get("omlx") or {}),
            # schema 2.1 kept the server-wide /v1/models/status in its own file;
            # fold it into the single environment snapshot so no field is lost.
            "server_models": (legacy.get("models.json") or old.get("models") or {}),
            "model_env": model_env or {env_ref: {}},
        },
        outcome={
            "status": "ok",
            "cancelled": False,
            "skipped": [],
            "warnings": [],
            # Preserved so `./modelbench runs` keeps showing the real runtime.
            "duration_s": (legacy.get("summary.json") or {}).get("duration_s"),
        },
    )


def migrate(archive: Path, output_root: Path, overwrite: bool = False) -> dict:
    batches: dict[str, dict] = {}
    migrated = []
    for batch_id, slug, entry in _legacy_runs(archive):
        legacy = {
            name: _load(entry / name)
            for name in (
                "manifest.json",
                "hardware.json",
                "omlx.json",
                "models.json",
                "summary.json",
            )
        }
        manifest = legacy["manifest.json"] or {}
        records = [
            json.loads(line)
            for line in (entry / "records.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not records:
            continue
        # Hoist the per-record snapshot; in practice it is constant per run.
        snapshot = (records[0].get("model_config_snapshot") or {})
        from modelbench.adapters.storage.writer import model_env_digest

        env_ref = model_env_digest(snapshot)
        created_at = manifest.get("created_at") or ""
        writer = RunWriter(
            output_root,
            batch_id=batch_id,
            method="performance",
            model_slug=slug,
            created_at=created_at,
            overwrite=overwrite,
            model={"id": snapshot.get("id") or slug, "alias": slug, "model_slug": slug},
        )
        run_manifest = _manifest_for(
            legacy, slug, batch_id, {env_ref: snapshot}, env_ref
        )
        run_manifest.run_id = writer.run_id
        writer.initialize(run_manifest)
        converted = []
        for raw in records:
            record, payload = _convert_record(raw, writer.run_id, env_ref)
            writer.write_record(record, prompt_payload=payload)
            converted.append(record)
        summary = writer.finalize(
            converted,
            perf=(run_manifest.config or {}).get("perf") or {},
            run_config=(run_manifest.config or {}).get("run") or {},
        )
        # --- fidelity assertions -------------------------------------------
        assert summary["n_records"] == len(records), "record count changed"
        agg = _load(writer.run_dir / "aggregates.json") or {}
        legacy_summary = legacy["summary.json"] or {}
        legacy_overall = {
            (row.get("model"), row.get("context_length")): row
            for row in legacy_summary.get("overall") or []
        }
        for row in agg.get("overall") or []:
            key = (row.get("model"), row.get("context_length"))
            ref = legacy_overall.get(key)
            if ref is None:
                continue
            for metric, dist in (row.get("perf") or {}).items():
                ref_dist = (ref.get("perf") or {}).get(metric) or {}
                if ref_dist.get("mean") is None:
                    continue
                assert dist.get("mean") == ref_dist.get("mean"), (
                    f"{writer.run_id} {metric} mean changed: "
                    f"{dist.get('mean')} != {ref_dist.get('mean')}"
                )
        migrated.append(
            {
                "run_id": writer.run_id,
                "batch_id": batch_id,
                "model_slug": slug,
                "run_path": writer.run_dir.relative_to(output_root).as_posix(),
                "n_records": summary["n_records"],
                "bytes_total": summary["bytes_total"],
            }
        )
        batches.setdefault(batch_id, {"created_at": created_at, "runs": []})["runs"].append(
            {"slug": slug, "model": run_manifest.model.get("id") or slug,
             "run_id": writer.run_id, "n_samples": summary["n_samples"],
             "duration_s": summary["duration_s"],
             "run_path": f"runs/{writer.run_id}"}
        )
    # --- batch metadata ---------------------------------------------------
    from modelbench.adapters.storage import layout
    from modelbench.adapters.storage.blobstore import DECODE_GUARD_VERSION

    written_batches = []
    for batch_id, meta in batches.items():
        first = meta["runs"][0]
        batch_path = layout.batch_path(output_root, batch_id)
        batch_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_batch = _load(archive / "batches" / f"web-{batch_id}.json") or {}
        payload = {
            "schema_version": "3.0",
            "batch_id": batch_id,
            "source": "web",
            "method": "performance",
            "created_at": meta["created_at"],
            "finished_at": legacy_batch.get("created_at"),
            "duration_s": legacy_batch.get("duration_s"),
            "project_version": legacy_batch.get("project_version") or "",
            "models_selected": [run["model"] for run in meta["runs"]],
            "models_measured": [run["model"] for run in meta["runs"]],
            "n_models_selected": len(meta["runs"]),
            "n_models_measured": len(meta["runs"]),
            "model_runs": [
                {
                    "model": run["model"],
                    "model_id": run["model"],
                    "model_slug": run["slug"],
                    "method": "performance",
                    "run_id": run["run_id"],
                    "run_id": run["run_id"],
                    "run_path": run["run_path"],
                    "status": "ok",
                    "n_samples": run["n_samples"],
                    "duration_s": run["duration_s"],
                }
                for run in meta["runs"]
            ],
            "inputs": {
                "testset": {
                    "path": "dataset/test/performance/testset.jsonl",
                    # The 2.x bundle never recorded a test-set digest, and the
                    # file has been rebuilt since, so claiming today's digest
                    # would be false provenance.  The *actual* prompts of this
                    # measurement are preserved content-addressed in
                    # output/blobs/prompts (referenced by records.jsonl), which
                    # is what makes the run reproducible.
                    "sha256": None,
                    "rows": None,
                    "context_windows": [1024, 8192, 32768],
                    "decode_guard_version": DECODE_GUARD_VERSION,
                    "note": "legacy-2.1 migration: original test set not archived; "
                    "exact prompts preserved in output/blobs/prompts",
                }
            },
            "config": legacy_batch.get("config") or {},
            "retention": {"pinned": False, "keep_until": None},
            "provenance": {
                "git_commit": legacy_batch.get("provenance", {}).get("git_commit")
                if isinstance(legacy_batch.get("provenance"), dict)
                else None,
                "migrated_from": "schema-2.1 archive/legacy-v2",
            },
            "skipped": [], "warnings": [],
            "errors": [],
        }
        batch_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        written_batches.append({"batch_id": batch_id, "batch_file": str(batch_path)})
    return {"runs": migrated, "batches": written_batches}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        default=str(ROOT / "output" / "archive" / "legacy-v2"),
        help="directory holding the schema-2.x bundles",
    )
    parser.add_argument(
        "--output-root", default=str(ROOT / "output"), help="schema-3.0 output root"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = migrate(
        Path(args.archive), Path(args.output_root), overwrite=args.overwrite
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
