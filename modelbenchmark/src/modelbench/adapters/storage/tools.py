"""Derived-artifact tools for the schema-3.0 output tree.

Four operations back the ``modelbench rebuild|verify|gc|pack`` subcommands:

``rebuild_run``
    Recompute a run's **derived** files (``samples.csv``, ``aggregates.json``,
    ``report.md``, ``MANIFEST.json``) from the authoritative ``records.jsonl``.
    Deleting the derived files must never lose information: this is the
    recovery path promised by the layout design (acceptance §11.7).

``verify_output``
    Reconcile the bundle against itself and the prompt pool (design §8.3):

    1. every ``batch.json.models_selected`` entry is in ``model_runs`` and every
       ``model_runs`` entry exists on disk (and vice versa);
    2. ``run.json.outcome.n_measured`` equals the number of ``phase=measured``
       records (error/timeout included) with ``n_ok`` as the ok subset — runs
       without ``n_ok`` are legacy and reconcile against the ok-only count —
       and ``n_selected == n_measured + len(skipped)``;
    3. every ``request.prompt_sha256`` resolves in ``blobs/prompts`` and every
       ``MANIFEST.json`` file entry matches the bytes on disk.

``gc_blobs``
    Mark-and-sweep of the prompt pool.  Reference roots are ``runs/`` **and**
    ``archive/``: batches retired into ``archive/`` are still evidence, so
    dropping them from the root set would silently delete the prompt text their
    records point at (design §8.2).  Refuses to run when the roots are absent or
    when ``verify`` reports missing blobs.

``pack_batch``
    Zip one batch directory plus exactly the prompt blobs it references, laid
    out relative to the output root (``runs/...`` + ``blobs/prompts/...``), so
    unzipping into an ``output/`` root yields a valid batch and ``verify``
    passes (acceptance §11.11).
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Set

from modelbench.adapters.storage import catalog, layout
from modelbench.adapters.storage.blobstore import PromptBlobStore
from modelbench.domain.context_tolerance import TOLERANCE_RATIO
from modelbench.domain.metrics import (
    MAX_PLAUSIBLE_TPS,
    aggregate_overall,
    aggregate_records,
    sample_results,
)
from modelbench.domain.models import SampleRecord

#: MANIFEST entry order is by path; these are the files a run is expected to
#: contain (``run.json`` + the derived set + the log).
_MANIFEST_NAMES = (
    layout.RUN_JSON,
    layout.RECORDS,
    layout.SAMPLES_CSV,
    layout.AGGREGATES_JSON,
    layout.REPORT_MD,
    f"{layout.LOGS_DIR}/{layout.LOG_NAME}",
)


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# rebuild
# --------------------------------------------------------------------------- #


def rebuild_run(output_root: Path, run_dir: Path) -> Dict[str, Any]:
    """Recompute one run's derived files from ``records.jsonl``.

    Rewrites ``aggregates.json``, ``samples.csv``, ``report.md`` and
    ``MANIFEST.json`` using exactly the same projections the writer uses, so a
    rebuilt bundle is semantically identical to the original (float formatting
    and timestamps may differ).  ``run.json`` and ``records.jsonl`` are read
    only.  Returns the same keys ``RunWriter.finalize()`` returns.
    """
    output_root = Path(output_root)
    run_dir = Path(run_dir)
    run_json = catalog.read_json(run_dir / layout.RUN_JSON)
    if not run_json:
        raise FileNotFoundError(f"missing or unreadable run.json: {run_dir}")
    records: List[SampleRecord] = catalog.load_records(run_dir)
    run_id = str(
        run_json.get("run_id")
        or layout.make_run_id(
            str(run_json.get("batch_id") or ""),
            str(run_json.get("method") or "performance"),
            str((run_json.get("model") or {}).get("model_slug") or run_dir.name),
        )
    )

    cells = aggregate_records(records)
    overall = aggregate_overall(records)
    samples = sample_results(records)
    aggregates = {
        "schema_version": _schema_version(),
        "run_id": run_id,
        "generated_at": _utc_now(),
        "n_records": len(records),
        "n_samples": len(samples),
        "n_cells": len(cells),
        "overall": [entry.model_dump(mode="json") for entry in overall],
        "cells": [cell.model_dump(mode="json") for cell in cells],
    }
    _write_json(run_dir / layout.AGGREGATES_JSON, aggregates)
    _write_samples_csv(run_dir / layout.SAMPLES_CSV, samples)

    config = run_json.get("config") or {}
    outcome = run_json.get("outcome") or {}
    env = run_json.get("env") or {}
    report_summary = {
        "perf": config.get("perf") or {},
        "run": config.get("run") or {},
        "n_records": len(records),
        "n_samples": len(samples),
        "n_cells": len(cells),
        "duration_s": outcome.get("duration_s", run_json.get("duration_s")),
        "n_isl_in_band": sum(cell.n_isl_in_band for cell in cells),
        "n_isl_out_of_band": sum(cell.n_isl_out_of_band for cell in cells),
    }
    (run_dir / layout.REPORT_MD).write_text(
        _render_report(
            run_id=run_id,
            hardware=env.get("hardware") or {},
            omlx=env.get("omlx") or {},
            overall=overall,
            cells=cells,
            samples=samples,
            summary=report_summary,
        ),
        encoding="utf-8",
    )
    manifest = _write_manifest(run_dir, output_root, run_id)
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "run_json": str(run_dir / layout.RUN_JSON),
        "manifest_json": str(run_dir / layout.MANIFEST_JSON),
        "records_jsonl": str(run_dir / layout.RECORDS),
        "samples_csv": str(run_dir / layout.SAMPLES_CSV),
        "aggregates_json": str(run_dir / layout.AGGREGATES_JSON),
        "report_md": str(run_dir / layout.REPORT_MD),
        "n_records": len(records),
        "n_samples": len(samples),
        "n_cells": len(cells),
        "n_blobs": manifest["blob_refs"]["prompts"],
        "duration_s": report_summary["duration_s"],
        "bytes_total": manifest["totals"]["bytes"],
    }


def _schema_version() -> str:
    from modelbench.domain.models import SCHEMA_VERSION

    return SCHEMA_VERSION


def _write_samples_csv(path: Path, samples: Sequence[Any]) -> None:
    """Write ``samples.csv`` with the frozen column order from the writer."""
    import csv

    from modelbench.adapters.storage.writer import SAMPLES_CSV_FIELDS

    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLES_CSV_FIELDS)
        writer.writeheader()
        for row in samples:
            payload = row.model_dump(mode="json")
            writer.writerow({key: payload.get(key) for key in SAMPLES_CSV_FIELDS})


def _render_report(
    *,
    run_id: str,
    hardware: Dict[str, Any],
    omlx: Dict[str, Any],
    overall: Sequence[Any],
    cells: Sequence[Any],
    samples: Sequence[Any],
    summary: Dict[str, Any],
) -> str:
    """Render ``report.md`` with the writer's own renderer.

    The renderer is currently a ``RunWriter`` method that only reads three
    attributes (``run_id``/``_hardware``/``_omlx``), so a duck-typed shell is
    enough to reuse it without touching the bundle directory.  If the writer
    exposes a module-level ``render_run_report`` (preferred), that is used.
    """
    from modelbench.adapters.storage import writer as writer_module

    renderer = getattr(writer_module, "render_run_report", None)
    if callable(renderer):
        return renderer(
            run_id=run_id,
            hardware=hardware,
            omlx=omlx,
            overall=overall,
            cells=cells,
            samples=samples,
            summary=summary,
        )
    shell = SimpleNamespace(run_id=run_id, _hardware=hardware, _omlx=omlx)
    return writer_module.RunWriter._render_report(  # type: ignore[attr-defined]
        shell, overall, cells, samples, summary
    )


def _write_manifest(run_dir: Path, output_root: Path, run_id: str) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []
    for name in _MANIFEST_NAMES:
        path = run_dir / name
        if not path.is_file():
            continue
        entries.append(
            {"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    entries.sort(key=lambda item: item["path"])
    shas = catalog.prompt_sha256s(run_dir)
    store = PromptBlobStore(output_root)
    missing = store.missing(shas)
    pool = store.stats()
    # Same semantics as RunWriter.finalize(): ``bytes`` = what THIS run
    # references (shared blobs counted per run), ``pool_bytes`` = whole pool.
    referenced_bytes = 0
    for sha in shas:
        try:
            referenced_bytes += store.path_for(sha).stat().st_size
        except OSError:
            pass
    manifest = {
        "schema_version": _schema_version(),
        "run_id": run_id,
        "generated_at": _utc_now(),
        "algorithm": "sha256",
        "files": entries,
        "totals": {"files": len(entries), "bytes": sum(item["bytes"] for item in entries)},
        "blob_refs": {
            "prompts": len(shas),
            "bytes": referenced_bytes,
            "pool_bytes": pool["bytes"],
            "missing": len(missing),
        },
    }
    _write_json(run_dir / layout.MANIFEST_JSON, manifest)
    return manifest


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #


def verify_output(output_root: Path) -> Dict[str, Any]:
    """Reconcile batches, runs, records and the prompt pool (design §8.3).

    Returns ``{"ok": bool, "issues": [str, ...], "stats": {...}}``.  ``issues``
    are human-readable strings prefixed with the offending path; ``stats``
    carries the counters (including ``blob_missing``) so callers such as
    :func:`gc_blobs` can make decisions without parsing messages.

    Flat layout: ``batches/*.json`` declare their members by ``run_id``; every
    declared member must exist as ``runs/<run_id>/`` (and vice versa — a run on
    disk that no batch file claims is reported too).
    """
    output_root = Path(output_root)
    issues: List[str] = []
    stats: Dict[str, int] = {
        "batches": 0,
        "runs": 0,
        "records": 0,
        "measured_ok": 0,
        "blobs": 0,
        "blob_missing": 0,
        "gen_tps_invalid": 0,
        "isl_out_of_tolerance": 0,
    }
    store = PromptBlobStore(output_root)
    stats["blobs"] = store.stats()["prompts"]

    claimed: Set[str] = set()
    for batch_file in catalog.iter_batch_files(output_root):
        batch = catalog.load_batch(batch_file)
        batch_rel = catalog.rel_path(output_root, batch_file) or str(batch_file)
        if batch is None:
            issues.append(f"{batch_rel}: batch metadata missing or unreadable")
            continue
        stats["batches"] += 1
        batch_id = str(batch.get("batch_id") or batch_file.stem)
        if batch_id != batch_file.stem:
            issues.append(
                f"{batch_rel}: batch_id {batch_id!r} != file name {batch_file.stem!r}"
            )
        run_meta = [item for item in batch.get("model_runs") or [] if isinstance(item, dict)]

        # --- 1) batch self-consistency + run directory reconciliation ------ #
        selected = [str(item) for item in batch.get("models_selected") or []]
        metas_by_model = {str(item.get("model")): item for item in run_meta}
        for model in selected:
            if model not in metas_by_model:
                issues.append(
                    f"{batch_rel}: models_selected entry not in model_runs: {model}"
                )
        for item in run_meta:
            run_id = str(item.get("run_id") or "")
            if not run_id:
                issues.append(
                    f"{batch_rel}: model_runs[{item.get('model')}] has no run_id"
                )
                continue
            expected_path = f"{layout.RUNS}/{run_id}"
            declared_path = item.get("run_path")
            if declared_path is not None and str(declared_path) != expected_path:
                issues.append(
                    f"{batch_rel}: model_runs[{run_id}] run_path {declared_path!r} "
                    f"!= {expected_path!r}"
                )
            if "/" in run_id or "\\" in run_id:
                issues.append(f"{batch_rel}: model_runs run_id is not a single segment: {run_id!r}")
                continue
            claimed.add(run_id)
            run_dir = layout.run_dir(output_root, run_id)
            if not (run_dir / layout.RUN_JSON).is_file():
                issues.append(f"{batch_rel}: model_runs[{run_id}] run directory missing")
                continue
            run_json = catalog.read_json(run_dir / layout.RUN_JSON) or {}
            if str(run_json.get("run_id") or "") != run_id:
                issues.append(
                    f"{batch_rel}: runs/{run_id}/run.json declares run_id "
                    f"{run_json.get('run_id')!r}"
                )
            if str(run_json.get("batch_id") or "") != batch_id:
                issues.append(
                    f"{batch_rel}: runs/{run_id}/run.json declares batch_id "
                    f"{run_json.get('batch_id')!r}"
                )

    # --- 2) per-run record/outcome reconciliation ------------------------- #
    for run_dir in catalog.iter_run_dirs(output_root):
        stats["runs"] += 1
        if run_dir.name not in claimed:
            run_rel = catalog.rel_path(output_root, run_dir) or str(run_dir)
            issues.append(f"{run_rel}: run not referenced by any batches/*.json")
        issues.extend(_verify_run(run_dir, output_root, store, stats))

    # --- 3) no un-migrated leftovers: every runs/ child is a run dir ------ #
    runs_root = layout.runs_dir(output_root)
    if runs_root.is_dir():
        for child in sorted(runs_root.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            if not (child / layout.RUN_JSON).is_file():
                issues.append(
                    f"{layout.RUNS}/{child.name}: unexpected directory under runs/ "
                    "(not a run, e.g. a pre-flatten leftover)"
                )

    return {"ok": not issues, "issues": issues, "stats": stats}


_AGG_NUMBER_FIELDS = (
    "n_samples",
    "n_valid",
    "n_error",
    "n_length_capped",
    "n_gen_tps_invalid",
    "n_isl_in_band",
    "n_isl_out_of_band",
    "insufficient_samples",
)


def _compare_aggregates(
    run_rel: str,
    declared: Dict[str, Any],
    expected_cells: Sequence[Any],
    expected_overall: Sequence[Any],
) -> List[str]:
    """Check that derived aggregates can be recomputed from records.jsonl."""
    issues: List[str] = []

    def key_cell(item: Dict[str, Any]) -> tuple:
        return (
            item.get("model"),
            item.get("scenario"),
            item.get("dataset"),
            item.get("context_length"),
        )

    def key_overall(item: Dict[str, Any]) -> tuple:
        return (item.get("model"), item.get("context_length"))

    def compare_dist(where: str, declared_dist: Any, expected_dist: Any) -> None:
        if not isinstance(declared_dist, dict) or not isinstance(expected_dist, dict):
            issues.append(f"{run_rel}: {where}: missing/invalid distribution")
            return
        for field in ("n", "mean", "median", "std", "min", "max", "p95", "ci95_low", "ci95_high"):
            a = declared_dist.get(field)
            b = expected_dist.get(field)
            if a is None and b is None:
                continue
            if a is None or b is None:
                issues.append(
                    f"{run_rel}: {where}: {field} {a!r} != recomputed {b!r}"
                )
                continue
            try:
                if abs(float(a) - float(b)) > 1e-6:
                    issues.append(
                        f"{run_rel}: {where}: {field} {a!r} != recomputed {b!r}"
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"{run_rel}: {where}: non-numeric {field} {a!r} "
                    f"!= recomputed {b!r}"
                )

    declared_cells = {
        key_cell(item): item
        for item in declared.get("cells") or []
        if isinstance(item, dict)
    }
    exp_cells = {key_cell(item.model_dump(mode="json")): item for item in expected_cells}
    for key, expected in exp_cells.items():
        where = f"cells[{key}]"
        actual = declared_cells.get(key)
        if actual is None:
            issues.append(f"{run_rel}: {where}: missing from aggregates.json")
            continue
        expected_json = expected.model_dump(mode="json")
        for field in _AGG_NUMBER_FIELDS:
            if actual.get(field) != expected_json.get(field):
                issues.append(
                    f"{run_rel}: {where}: {field} {actual.get(field)!r} != "
                    f"recomputed {expected_json.get(field)!r}"
                )
        for metric, expected_dist in (expected_json.get("perf") or {}).items():
            compare_dist(where + f".perf[{metric}]", (actual.get("perf") or {}).get(metric), expected_dist)
        # T-067: the all-sample calibre must round-trip too; a legacy
        # aggregates.json without perf_all is only tolerated when the
        # recomputation also produced none (empty means "no valid records").
        for metric, expected_dist in (expected_json.get("perf_all") or {}).items():
            compare_dist(
                where + f".perf_all[{metric}]",
                (actual.get("perf_all") or {}).get(metric),
                expected_dist,
            )
    for key in declared_cells.keys() - exp_cells.keys():
        issues.append(f"{run_rel}: cells[{key}]: not recomputable from records")

    declared_overall = {
        key_overall(item): item
        for item in declared.get("overall") or []
        if isinstance(item, dict)
    }
    exp_overall = {key_overall(item.model_dump(mode="json")): item for item in expected_overall}
    for key, expected in exp_overall.items():
        where = f"overall[{key}]"
        actual = declared_overall.get(key)
        if actual is None:
            issues.append(f"{run_rel}: {where}: missing from aggregates.json")
            continue
        expected_json = expected.model_dump(mode="json")
        for field in _AGG_NUMBER_FIELDS:
            if actual.get(field) != expected_json.get(field):
                issues.append(
                    f"{run_rel}: {where}: {field} {actual.get(field)!r} != "
                    f"recomputed {expected_json.get(field)!r}"
                )
        for metric, expected_dist in (expected_json.get("perf") or {}).items():
            compare_dist(where + f".perf[{metric}]", (actual.get("perf") or {}).get(metric), expected_dist)
        # T-067: same dual-calibre check as cells (see above).
        for metric, expected_dist in (expected_json.get("perf_all") or {}).items():
            compare_dist(
                where + f".perf_all[{metric}]",
                (actual.get("perf_all") or {}).get(metric),
                expected_dist,
            )
    for key in declared_overall.keys() - exp_overall.keys():
        issues.append(f"{run_rel}: overall[{key}]: not recomputable from records")
    return issues


def _verify_run(
    run_dir: Path,
    output_root: Path,
    store: PromptBlobStore,
    stats: Dict[str, int],
) -> List[str]:
    issues: List[str] = []
    run_rel = catalog.rel_path(output_root, run_dir) or str(run_dir)
    run_json = catalog.read_json(run_dir / layout.RUN_JSON)
    if not run_json:
        # Reported by the batch pass; nothing to reconcile here.
        return issues
    # Flat-layout identity: the directory name *is* the run id.
    if str(run_json.get("run_id") or "") != run_dir.name:
        issues.append(
            f"{run_rel}: run.json run_id {run_json.get('run_id')!r} != directory name"
        )
    records = list(catalog.iter_records(run_dir))
    stats["records"] += len(records)

    measured_ok = sum(
        1
        for record in records
        if record.get("phase") == "measured" and record.get("status") == "ok"
    )
    stats["measured_ok"] += measured_ok
    outcome = run_json.get("outcome") or {}
    if not outcome:
        issues.append(f"{run_rel}: run.json has no outcome block")
        return issues
    skipped = list(outcome.get("skipped") or [])
    # New semantics (n_ok present): n_measured counts ALL measured-phase
    # records (error/timeout included) and n_ok the ok subset.  Legacy runs
    # without n_ok wrote n_measured as the ok-only count; keep accepting them.
    measured_all = sum(1 for record in records if record.get("phase") == "measured")
    n_measured = outcome.get("n_measured")
    n_ok = outcome.get("n_ok")
    if n_ok is not None:
        if n_measured != measured_all:
            issues.append(
                f"{run_rel}: outcome.n_measured={n_measured} != "
                f"{measured_all} measured records"
            )
        if int(n_ok) != measured_ok:
            issues.append(
                f"{run_rel}: outcome.n_ok={n_ok} != {measured_ok} measured+ok records"
            )
    elif n_measured != measured_ok:
        issues.append(
            f"{run_rel}: outcome.n_measured={n_measured} != "
            f"{measured_ok} measured+ok records (legacy, no n_ok)"
        )
    n_selected = outcome.get("n_selected")
    expected_selected = (measured_all if n_ok is not None else measured_ok) + len(
        skipped
    )
    if n_selected is not None and int(n_selected) != expected_selected:
        issues.append(
            f"{run_rel}: outcome.n_selected={n_selected} != "
            f"n_measured({expected_selected - len(skipped)}) + skipped({len(skipped)})"
        )
    n_records = outcome.get("n_records")
    if n_records is not None and int(n_records) != len(records):
        issues.append(
            f"{run_rel}: outcome.n_records={n_records} != {len(records)} records.jsonl lines"
        )

    # --- 2a) derived aggregates must be recomputable from records ------- #
    aggregates = catalog.read_json(run_dir / layout.AGGREGATES_JSON)
    if aggregates is None:
        issues.append(f"{run_rel}: aggregates.json missing or unreadable")
    else:
        try:
            loaded = catalog.load_records(run_dir)
            issues.extend(
                _compare_aggregates(
                    run_rel,
                    aggregates,
                    aggregate_records(loaded),
                    aggregate_overall(loaded),
                )
            )
        except Exception as exc:  # noqa: BLE001 - report, do not crash verify
            issues.append(f"{run_rel}: cannot recompute aggregates: {type(exc).__name__}: {exc}")

    # --- 2b) measurement validity --------------------------------------- #
    invalid_tps = 0
    isl_out = 0
    isl_in = 0
    for record in records:
        if record.get("phase") != "measured" or record.get("status") != "ok":
            continue
        metrics = record.get("metrics") or {}
        actual_tps = metrics.get("gen_tps")
        record_invalid = False
        if actual_tps is not None:
            try:
                value = float(actual_tps)
            except (TypeError, ValueError):
                value = None
            if value is None or value < 0 or value > MAX_PLAUSIBLE_TPS:
                # Legacy/migrated runs can contain values measured before the
                # validity guard; they are excluded from aggregates by
                # ``metrics._effective_gen_tps``.  Report as a counter, not as
                # a structural verify failure, so the bundle stays usable.
                record_invalid = True
                stats.setdefault("legacy_gen_tps_invalid", 0)
                stats["legacy_gen_tps_invalid"] += 1
            if metrics.get("gen_tps_invalid"):
                record_invalid = True
                issues.append(
                    f"{run_rel}: gen_tps_invalid={metrics.get('gen_tps_invalid')!r} "
                    f"but gen_tps is present for {record.get('sample_id')}"
                )
        if record_invalid:
            invalid_tps += 1
        request_payload = record.get("request") or {}
        tolerance = (request_payload.get("context_meta") or {}).get(
            "isl_tolerance"
        ) or {}
        metrics_payload = record.get("metrics") or {}
        actual = (
            request_payload.get("tokenized_prompt_tokens")
            if request_payload.get("tokenized_prompt_tokens") is not None
            else metrics_payload.get("input_context_tokens")
        )
        target = record.get("context_length") or tolerance.get("target")
        if actual is not None and target:
            in_band = abs(int(actual) - int(target)) <= TOLERANCE_RATIO * int(target)
        else:
            # Legacy records without any actual/target evidence are kept.
            in_band = True
        if in_band:
            isl_in += 1
        else:
            isl_out += 1
    stats["gen_tps_invalid"] += invalid_tps
    stats["isl_out_of_tolerance"] += isl_out
    stats.setdefault("isl_in_band", 0)
    stats["isl_in_band"] += isl_in
    declared_in_band = outcome.get("n_isl_in_band")
    if declared_in_band is not None and int(declared_in_band) != isl_in:
        issues.append(
            f"{run_rel}: outcome.n_isl_in_band={declared_in_band} != "
            f"{isl_in} detected"
        )
    declared_out_band = outcome.get("n_isl_out_of_band")
    if declared_out_band is not None and int(declared_out_band) != isl_out:
        issues.append(
            f"{run_rel}: outcome.n_isl_out_of_band={declared_out_band} != "
            f"{isl_out} detected"
        )
    declared_invalid = outcome.get("n_gen_tps_invalid")
    if declared_invalid is not None and int(declared_invalid) != invalid_tps:
        issues.append(
            f"{run_rel}: outcome.n_gen_tps_invalid={declared_invalid} != "
            f"{invalid_tps} detected"
        )

    # --- 3a) prompt pool resolution ------------------------------------- #
    shas = catalog.prompt_sha256s(run_dir)
    missing = store.missing(shas)
    if missing:
        stats["blob_missing"] += len(missing)
        issues.append(
            f"{run_rel}: {len(missing)} prompt blob(s) missing from blobs/prompts "
            f"(first: {', '.join(missing[:3])})"
        )

    # --- 3b) MANIFEST file digests --------------------------------------- #
    manifest = catalog.read_json(run_dir / layout.MANIFEST_JSON)
    if manifest is None:
        issues.append(f"{run_rel}: MANIFEST.json missing or unreadable")
        return issues
    declared = 0
    for item in manifest.get("files") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("path") or "")
        declared += 1
        if not layout.is_safe_rel_path(name):
            issues.append(f"{run_rel}: MANIFEST path escapes run dir: {name!r}")
            continue
        path = run_dir / name
        if not path.is_file():
            issues.append(f"{run_rel}: MANIFEST file missing: {name}")
            continue
        size = path.stat().st_size
        if item.get("bytes") is not None and int(item["bytes"]) != size:
            issues.append(
                f"{run_rel}: MANIFEST bytes mismatch for {name}: "
                f"{item['bytes']} != {size}"
            )
        expected = item.get("sha256")
        if expected and sha256_file(path) != expected:
            issues.append(f"{run_rel}: MANIFEST sha256 mismatch for {name}")
    if not declared:
        issues.append(f"{run_rel}: MANIFEST.json declares no files")
    return issues


# --------------------------------------------------------------------------- #
# gc
# --------------------------------------------------------------------------- #


def gc_blobs(output_root: Path, dry_run: bool = False) -> Dict[str, Any]:
    """Delete prompt blobs no surviving run references (mark and sweep).

    Reference roots are ``runs/`` **and** ``archive/`` (archived batches are
    still evidence).  Refuses when neither root exists (that would look like
    "everything is garbage") and when ``verify`` finds missing blobs — deleting
    around a hole would make the evidence unrecoverable.
    """
    output_root = Path(output_root)
    runs_root = layout.runs_dir(output_root)
    archive_root = layout.archive_dir(output_root)
    roots = [path for path in (runs_root, archive_root) if path.is_dir()]
    if not roots:
        raise RuntimeError(
            "refusing to GC prompt blobs: neither runs/ nor archive/ exists under "
            f"{output_root} (nothing would anchor the reference roots)"
        )
    report = verify_output(output_root)
    missing = int(report["stats"].get("blob_missing") or 0)
    if missing:
        return {
            "ok": False,
            "refused": True,
            "reason": f"verify found {missing} missing prompt blob reference(s)",
            "dry_run": bool(dry_run),
            "deleted_count": 0,
            "freed_bytes": 0,
            "kept": 0,
            "referenced": 0,
        }
    referenced: Set[str] = set()
    for root in roots:
        for records_path in sorted(root.rglob(layout.RECORDS)):
            referenced.update(catalog.prompt_sha256s(records_path.parent))
    store = PromptBlobStore(output_root)
    existing = set(store.iter_shas())
    orphans = sorted(existing - referenced)
    freed = 0
    for sha in orphans:
        try:
            freed += store.path_for(sha).stat().st_size
        except OSError:
            continue
    if not dry_run:
        for sha in orphans:
            try:
                store.path_for(sha).unlink()
            except OSError:
                continue
        _prune_empty_blob_shards(output_root)
    return {
        "ok": True,
        "refused": False,
        "dry_run": bool(dry_run),
        "deleted_count": len(orphans),
        "deleted": orphans,
        "freed_bytes": freed,
        "kept": len(existing) - len(orphans),
        "referenced": len(referenced),
    }


def _prune_empty_blob_shards(output_root: Path) -> None:
    root = layout.blobs_root(output_root)
    if not root.is_dir():
        return
    for shard in sorted(path for path in root.iterdir() if path.is_dir()):
        try:
            next(shard.iterdir())
        except StopIteration:
            try:
                shard.rmdir()
            except OSError:
                continue
        except OSError:
            continue


# --------------------------------------------------------------------------- #
# pack
# --------------------------------------------------------------------------- #


def pack_batch(output_root: Path, batch_id: str, out_path: Path) -> Path:
    """Zip one batch (metadata + its runs) plus the prompt blobs it references.

    Flat layout: ``batches/<batch_id>.json`` declares its members; each member
    is ``runs/<run_id>/``.  Paths inside the archive are relative to the output
    root (``batches/<id>.json`` + ``runs/<run_id>/...`` + ``blobs/prompts/...``),
    so extracting it at an ``output/`` root recreates a valid, movable batch
    that passes :func:`verify_output`.
    """
    output_root = Path(output_root)
    batch_file = catalog.find_batch_file(output_root, batch_id)
    if batch_file is None:
        raise KeyError(f"unknown batch id: {batch_id}")
    batch = catalog.load_batch(batch_file) or {}
    listed = [
        str(item.get("run_id"))
        for item in batch.get("model_runs") or []
        if isinstance(item, dict) and item.get("run_id")
    ]
    run_dirs = catalog.run_dirs_in_batch(output_root, batch)
    missing_runs = [run_id for run_id in listed if not layout.run_dir(output_root, run_id).is_dir()]
    if missing_runs:
        raise FileNotFoundError(
            f"cannot pack batch {batch_id}: {len(missing_runs)} listed run(s) missing "
            f"on disk (first: {', '.join(missing_runs[:3])})"
        )
    shas: Set[str] = set()
    for run_dir in run_dirs:
        shas.update(catalog.prompt_sha256s(run_dir))
    store = PromptBlobStore(output_root)
    missing = store.missing(sorted(shas))
    if missing:
        raise FileNotFoundError(
            f"cannot pack batch {batch_id}: {len(missing)} prompt blob(s) missing "
            f"(first: {', '.join(missing[:3])})"
        )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f".{out_path.name}.tmp-{os.getpid()}")
    skip = {tmp.resolve(), out_path.resolve()}
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(batch_file, batch_file.relative_to(output_root).as_posix())
            for run_dir in run_dirs:
                for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
                    if path.resolve() in skip:  # never zip the archive into itself
                        continue
                    archive.write(path, path.relative_to(output_root).as_posix())
            for sha in sorted(shas):
                blob = store.path_for(sha)
                archive.write(blob, blob.relative_to(output_root).as_posix())
        os.replace(tmp, out_path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:  # pragma: no cover - best effort
                pass
    return out_path
