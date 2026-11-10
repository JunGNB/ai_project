"""Run catalog (``output/index.json``) and read helpers — index schema 2.0.

The catalog is the single stable entry point for tooling: a small derived JSON
document describing every run so consumers (web console, CLI, Excel, analysis)
do not have to stat the tree or parse large aggregates.

Layout (schema 3.0, flat — see ``reports/impl/FLATTEN.md``)::

    output/
    ├── index.json                              # this module (derived)
    ├── blobs/prompts/<sha[:2]>/<sha>.json      # content-addressed prompt pool
    ├── runs/<run_id>/                          # one flat directory per run
    │   ├── run.json  MANIFEST.json  records.jsonl
    │   ├── samples.csv  aggregates.json  report.md
    │   └── logs/run.log
    ├── batches/<batch_id>.json                 # one test action -> one file
    ├── exports/<batch_id>.xlsx
    ├── archive/  .state/

``run_id`` is a single path segment (``<batch_id>__<method>__<model_slug>``) and
is also the run directory name.  Batches no longer own a directory: their
metadata is one JSON file under ``batches/`` and their members are listed by
``batch.json.model_runs[].run_id``.

Index 2.0 rules:

* **relative paths only** — ``runs[].run_path`` is ``runs/<run_id>`` and
  ``batches[].path`` is ``batches/<batch_id>.json``; both are relative to the
  output root, so moving ``output/`` keeps the index valid.
* one flat ``runs[]`` row per run plus a grouped ``batches[]`` view derived from
  ``batches/*.json``.
* ``files`` is copied from the run's ``MANIFEST.json`` so rebuilding the index
  never needs a recursive ``stat`` of large bundles.
* only schema-3.0 bundles are indexed; retired 2.x bundles live under
  ``output/archive/`` and are deliberately invisible here (they are still GC
  reference roots, see :mod:`modelbench.adapters.storage.tools`).
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from modelbench.adapters.host import filelock
from modelbench.adapters.storage import layout
from modelbench.adapters.storage.blobstore import PromptBlobStore

#: Name of the catalog file at the output root (kept for existing callers).
INDEX_NAME = layout.INDEX_JSON
#: Version of ``index.json`` itself; independent from ``SCHEMA_VERSION``.
INDEX_SCHEMA_VERSION = "2.0"

#: Files exposed by the web console's run detail ("原始文件" panel) and the
#: ``/api/runs/<id>/file`` endpoint.  ``records.jsonl`` is download-only.
VIEWABLE_FILES = (
    layout.RUN_JSON,
    layout.MANIFEST_JSON,
    layout.RECORDS,
    layout.SAMPLES_CSV,
    layout.AGGREGATES_JSON,
    layout.REPORT_MD,
    f"{layout.LOGS_DIR}/{layout.LOG_NAME}",
)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def human_size(num_bytes: Optional[int]) -> str:
    if not num_bytes:
        return "0 B"
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    """Read a JSON object; ``None`` for missing/unreadable/non-object files."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def iter_records(run_dir: Path) -> Iterator[Dict[str, Any]]:
    """Yield the raw JSON objects of ``records.jsonl`` (skipping broken lines)."""
    path = Path(run_dir) / layout.RECORDS
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            if isinstance(payload, dict):
                yield payload


def load_records(run_dir: Path) -> List[Any]:
    """Parse ``records.jsonl`` back into :class:`SampleRecord` objects.

    Persisted records drop ``request.messages`` (the prompt body lives in the
    blob pool), so the required field is restored as an empty list before
    validation; nothing downstream re-reads prompt text from a record.
    """
    from modelbench.domain.models import SampleRecord

    records = []
    for payload in iter_records(run_dir):
        request = payload.get("request")
        if isinstance(request, dict):
            request.setdefault("messages", [])
        records.append(SampleRecord.model_validate(payload))
    return records


def prompt_sha256s(run_dir: Path) -> List[str]:
    """Every distinct prompt digest referenced by a run's records."""
    seen: Dict[str, None] = {}
    for payload in iter_records(run_dir):
        sha = ((payload.get("request") or {}).get("prompt_sha256")) or ""
        if sha:
            seen.setdefault(str(sha), None)
    return list(seen)


# --------------------------------------------------------------------------- #
# layout discovery (flat schema)
# --------------------------------------------------------------------------- #


def _output_root_for(run_dir: Path) -> Optional[Path]:
    """Infer the output root from ``runs/<run_id>/`` (one flat level)."""
    path = Path(run_dir)
    parents = path.parents
    # parents[0]=runs dir, parents[1]=output root
    if len(parents) > 1 and parents[0].name == layout.RUNS:
        return parents[1]
    return None


def _rel_path(output_root: Optional[Path], path: Optional[Path]) -> Optional[str]:
    """POSIX path of ``path`` relative to ``output_root``; never absolute."""
    if output_root is None or path is None:
        return None
    try:
        root = Path(output_root).resolve()
        candidate = Path(path).resolve()
        return candidate.relative_to(root).as_posix()
    except (OSError, ValueError):
        return None


def rel_path(output_root: Path, path: Path) -> Optional[str]:
    """Public form of :func:`_rel_path` (relative-to-output-root POSIX path)."""
    return _rel_path(output_root, path)


def _is_safe_segment(value: Any) -> bool:
    text = str(value or "")
    return bool(text) and "/" not in text and "\\" not in text and text not in (".", "..")


def iter_run_dirs(output_root: Path) -> List[Path]:
    """Every run directory (``runs/<run_id>/`` containing ``run.json``)."""
    root = layout.runs_dir(Path(output_root))
    if not root.is_dir():
        return []
    return [
        path
        for path in sorted(root.iterdir())
        if path.is_dir() and (path / layout.RUN_JSON).is_file()
    ]


def iter_batch_files(output_root: Path) -> List[Path]:
    """Every batch metadata file (``batches/<batch_id>.json``)."""
    root = layout.batches_dir(Path(output_root))
    if not root.is_dir():
        return []
    return [path for path in sorted(root.glob("*.json")) if path.is_file()]


# --------------------------------------------------------------------------- #
# batch helpers
# --------------------------------------------------------------------------- #


def load_batch(batch_path: Path) -> Optional[Dict[str, Any]]:
    """Read a batch metadata file (``batches/<batch_id>.json``).

    Accepts the file path; returns ``None`` when absent/unreadable.
    """
    return read_json(Path(batch_path))


def find_batch_file(output_root: Path, batch_id: str) -> Optional[Path]:
    """Resolve ``batch_id`` to ``batches/<batch_id>.json`` (``None`` if absent)."""
    text = str(batch_id or "").strip()
    if not _is_safe_segment(text):
        return None
    candidate = layout.batch_path(Path(output_root), text)
    return candidate if candidate.is_file() else None


def batch_file_for_run(output_root: Path, run_dir: Path) -> Optional[Path]:
    """``runs/<run_id>/`` -> ``batches/<batch_id>.json`` (via ``run.json``)."""
    run_json = read_json(Path(run_dir) / layout.RUN_JSON) or {}
    batch_id = run_json.get("batch_id")
    if not _is_safe_segment(batch_id):
        try:
            batch_id = layout.parse_run_id(Path(run_dir).name)[0]
        except ValueError:
            return None
    return layout.batch_path(Path(output_root), str(batch_id))


def _batch_meta_map(output_root: Path) -> Dict[str, Dict[str, Any]]:
    """``batch_id -> {"batch_file": Path, "batch": dict}`` for every batch."""
    meta: Dict[str, Dict[str, Any]] = {}
    for batch_file in iter_batch_files(output_root):
        batch = load_batch(batch_file) or {}
        batch_id = str(batch.get("batch_id") or batch_file.stem)
        meta[batch_id] = {"batch_file": batch_file, "batch": batch}
    return meta


def run_dirs_in_batch(output_root: Path, batch: Dict[str, Any]) -> List[Path]:
    """Run directories listed by ``batch["model_runs"][].run_id``.

    Only directories that actually exist with a ``run.json`` are returned, in
    the batch's declared order, without duplicates.
    """
    root = Path(output_root)
    seen: Dict[str, None] = {}
    run_dirs: List[Path] = []
    for item in (batch or {}).get("model_runs") or []:
        if not isinstance(item, dict):
            continue
        run_id = str(item.get("run_id") or "")
        if not _is_safe_segment(run_id) or run_id in seen:
            continue
        seen[run_id] = None
        candidate = layout.run_dir(root, run_id)
        if (candidate / layout.RUN_JSON).is_file():
            run_dirs.append(candidate)
    return run_dirs


# --------------------------------------------------------------------------- #
# run entry (index row)
# --------------------------------------------------------------------------- #


def model_size_gb_from_run(
    run_json: Optional[Dict[str, Any]], model_id: Any = None
) -> Optional[float]:
    """Weights size in GiB from ``run.json.env.model_env[*].estimated_size``.

    ``model_env`` blocks carry the per-model oMLX snapshot; the block whose
    ``id`` matches ``model_id`` wins, otherwise the first block with a size is
    used (a run normally has exactly one block).  Returns ``None`` for partial
    bundles so the list page shows a blank instead of a wrong number.
    """
    if not run_json:
        return None
    blocks = ((run_json.get("env") or {}).get("model_env") or {}).values()
    fallback: Optional[float] = None
    for block in blocks:
        if not isinstance(block, dict):
            continue
        size = block.get("estimated_size")
        try:
            size_f = float(size) if size is not None else None
        except (TypeError, ValueError):
            size_f = None
        if not size_f or size_f <= 0:
            continue
        value = round(size_f / (1024 ** 3), 1)
        if model_id and str(block.get("id")) == str(model_id):
            return value
        if fallback is None:
            fallback = value
    return fallback


def _manifest_files(manifest: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    files: Dict[str, Dict[str, Any]] = {}
    for item in (manifest or {}).get("files") or []:
        if not isinstance(item, dict):
            continue
        name = item.get("path")
        if not name:
            continue
        files[str(name)] = {
            "bytes": item.get("bytes"),
            "sha256": item.get("sha256"),
        }
    return files


def _measured_models(aggregates: Optional[Dict[str, Any]]) -> List[str]:
    return sorted(
        {
            str(entry.get("model"))
            for entry in (aggregates or {}).get("overall") or []
            if isinstance(entry, dict) and entry.get("model")
        }
    )


def _datasets(aggregates: Optional[Dict[str, Any]]) -> List[str]:
    return sorted(
        {
            str(cell.get("dataset"))
            for cell in (aggregates or {}).get("cells") or []
            if isinstance(cell, dict) and cell.get("dataset")
        }
    )


def run_entry(
    run_dir: Path,
    *,
    output_root: Optional[Path] = None,
    run_json: Optional[Dict[str, Any]] = None,
    aggregates: Optional[Dict[str, Any]] = None,
    manifest_json: Optional[Dict[str, Any]] = None,
    batch: Optional[Dict[str, Any]] = None,
    batch_file: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Build one flat ``runs[]`` row for ``index.json``.

    ``run.json`` is the existence criterion.  ``aggregates.json`` and
    ``MANIFEST.json`` default to ``{}`` when absent so an interrupted run still
    shows up instead of disappearing from the catalog.  All paths written here
    are relative to ``output_root``; ``None`` is returned rather than emitting
    an absolute path.
    """
    run_dir = Path(run_dir)
    root = Path(output_root) if output_root is not None else _output_root_for(run_dir)
    if run_json is None:
        run_json = read_json(run_dir / layout.RUN_JSON)
    if not run_json:
        return None
    run_path = _rel_path(root, run_dir)
    if not run_path:
        return None
    if aggregates is None:
        aggregates = read_json(run_dir / layout.AGGREGATES_JSON) or {}
    if manifest_json is None:
        manifest_json = read_json(run_dir / layout.MANIFEST_JSON) or {}
    model = run_json.get("model") or {}
    model_name = str(
        model.get("id") or model.get("alias") or model.get("model_slug") or ""
    )
    run_id = str(run_json.get("run_id") or run_dir.name)
    model_slug = str(model.get("model_slug") or "")
    if not model_slug:
        try:
            model_slug = layout.parse_run_id(run_id)[2]
        except ValueError:
            model_slug = ""
    batch_id = str(run_json.get("batch_id") or "")
    if not batch_id:
        try:
            batch_id = layout.parse_run_id(run_id)[0]
        except ValueError:
            batch_id = ""
    if batch_file is None and batch_id:
        batch_file = layout.batch_path(root, batch_id) if root is not None else None
    if batch is None and batch_file is not None:
        batch = load_batch(batch_file) or {}
    batch = batch or {}

    outcome = run_json.get("outcome") or {}
    measured = _measured_models(aggregates)
    if not measured and model_name:
        measured = [model_name]
    selected = [str(item) for item in batch.get("models_selected") or []]
    if not selected:
        selected = measured[:1] or ([model_name] if model_name else [])

    model_runs = batch.get("model_runs") or []
    batch_index = None
    batch_status = None
    for position, item in enumerate(model_runs):
        if isinstance(item, dict) and str(item.get("run_id")) == run_id:
            batch_index = position
            batch_status = item.get("status")
            break
    batch_models_total = (
        len(model_runs)
        or batch.get("n_models_selected")
        or (len(selected) if selected else None)
    )

    skipped = list(outcome.get("skipped") or [])
    files = _manifest_files(manifest_json)
    totals = (manifest_json or {}).get("totals") or {}
    bytes_total = int(totals.get("bytes") or 0)
    if not bytes_total:
        bytes_total = sum(int(item.get("bytes") or 0) for item in files.values())
    duration_s = run_json.get("duration_s")
    if duration_s is None:
        duration_s = outcome.get("duration_s")

    return {
        "run_id": run_id,
        "run_path": run_path,
        "batch_id": batch_id or None,
        "batch_index": batch_index,
        "batch_models_total": batch_models_total,
        "method": str(run_json.get("method") or batch.get("method") or "performance"),
        "mode": "performance",
        "model": model_name,
        "model_id": str(model.get("id") or model_name),
        "model_slug": model_slug,
        "model_size_gb": _model_size_gb_for_entry(batch, run_json, model, model_name),
        "created_at": run_json.get("created_at"),
        "duration_s": duration_s,
        "status": str(outcome.get("status") or batch_status or "unknown"),
        "cancelled": bool(outcome.get("cancelled")),
        "models_selected": selected,
        "models_measured": measured,
        "n_models_selected": len(selected),
        "n_models_measured": len(measured),
        "n_skipped": len(skipped),
        "skipped": skipped,
        "memory_errors": int(outcome.get("memory_errors") or 0),
        "n_samples": _first_int(outcome.get("n_samples"), aggregates.get("n_samples")),
        "n_records": _first_int(outcome.get("n_records"), aggregates.get("n_records")),
        "n_cells": _first_int(outcome.get("n_cells"), aggregates.get("n_cells")),
        "datasets": _datasets(aggregates),
        "bytes_total": bytes_total,
        "bytes_total_human": human_size(bytes_total),
        "files": files,
        "legacy": False,
    }


def _first_int(*values: Any) -> Optional[int]:
    for value in values:
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


def _model_size_gb_for_entry(
    batch: Dict[str, Any],
    run_json: Dict[str, Any],
    model: Dict[str, Any],
    model_name: str,
) -> Optional[float]:
    """Prefer the batch's cached size, fall back to this run's env snapshot."""
    run_id = run_json.get("run_id")
    for item in batch.get("model_runs") or []:
        if not isinstance(item, dict):
            continue
        if run_id is not None and str(item.get("run_id")) != str(run_id):
            continue
        value = item.get("model_size_gb")
        if isinstance(value, (int, float)) and value > 0:
            return round(float(value), 1)
    return model_size_gb_from_run(
        run_json, model.get("id") or model_name or model.get("alias")
    )


# --------------------------------------------------------------------------- #
# scanning / index building
# --------------------------------------------------------------------------- #


def scan_runs(output_root: Path) -> List[Dict[str, Any]]:
    """Every schema-3.0 run under ``output/runs`` as a flat catalog entry."""
    output_root = Path(output_root)
    batch_meta = _batch_meta_map(output_root)
    entries: List[Dict[str, Any]] = []
    for run_dir in iter_run_dirs(output_root):
        run_json = read_json(run_dir / layout.RUN_JSON) or {}
        batch_id = str(run_json.get("batch_id") or "")
        meta = batch_meta.get(batch_id) or {}
        entry = run_entry(
            run_dir,
            output_root=output_root,
            run_json=run_json,
            batch=meta.get("batch"),
            batch_file=meta.get("batch_file"),
        )
        if entry:
            entries.append(entry)
    entries.sort(
        key=lambda entry: (entry.get("created_at") or "", entry.get("run_id") or ""),
        reverse=True,
    )
    return entries


def _batch_groups(
    output_root: Path,
    entries: List[Dict[str, Any]],
    batch_meta: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Group flat run entries into the ``batches[]`` view (batch files)."""
    groups: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        batch_id = entry.get("batch_id")
        if not batch_id:
            continue
        key = str(batch_id)
        group = groups.get(key)
        if group is None:
            meta = batch_meta.get(key) or {}
            batch = meta.get("batch") or {}
            group = {
                "batch_id": key,
                "created_at": batch.get("created_at") or entry.get("created_at"),
                "path": _rel_path(output_root, meta.get("batch_file")),
                "methods": [],
                "models": [],
                "run_ids": [],
                "n_models": 0,
                "n_samples": 0,
                "duration_s": 0.0,
            }
            groups[key] = group
        method = entry.get("method") or "performance"
        if method not in group["methods"]:
            group["methods"].append(method)
        group["models"].append(
            {
                "model": entry.get("model"),
                "model_id": entry.get("model_id"),
                "model_slug": entry.get("model_slug"),
                "run_id": entry.get("run_id"),
                "run_path": entry.get("run_path"),
                "status": entry.get("status"),
                "n_samples": entry.get("n_samples"),
                "duration_s": entry.get("duration_s"),
            }
        )
        group["run_ids"].append(entry.get("run_id"))
        group["n_models"] += 1
        group["n_samples"] += int(entry.get("n_samples") or 0)
        group["duration_s"] = round(
            group["duration_s"] + float(entry.get("duration_s") or 0.0), 1
        )
    # A batch that failed before writing any run still exists on disk; show it
    # instead of silently hiding the failure.
    for batch_id, meta in batch_meta.items():
        if batch_id in groups:
            continue
        batch = meta.get("batch") or {}
        groups[batch_id] = {
            "batch_id": batch_id,
            "created_at": batch.get("created_at"),
            "path": _rel_path(output_root, meta.get("batch_file")),
            "methods": [str(batch.get("method"))] if batch.get("method") else [],
            "models": [],
            "run_ids": [],
            "n_models": 0,
            "n_samples": 0,
            "duration_s": 0.0,
        }
    return sorted(
        groups.values(),
        key=lambda item: (item.get("created_at") or "", item.get("batch_id") or ""),
        reverse=True,
    )


def _totals(
    entries: List[Dict[str, Any]], blobs: Dict[str, int], batch_meta: Dict[str, Any]
) -> Dict[str, int]:
    """Cheap output-size total derived from MANIFESTs (no recursive stat)."""
    total_bytes = sum(int(entry.get("bytes_total") or 0) for entry in entries)
    total_bytes += int(blobs.get("bytes") or 0)
    total_files = sum(len(entry.get("files") or {}) for entry in entries)
    total_files += int(blobs.get("prompts") or 0)
    for meta in batch_meta.values():
        batch_file = Path(meta["batch_file"])
        try:
            total_bytes += batch_file.stat().st_size
            total_files += 1
        except OSError:
            continue
    return {"bytes": total_bytes, "files": total_files}


def _build_index_unlocked(output_root: Path) -> Dict[str, Any]:
    output_root = Path(output_root)
    batch_meta = _batch_meta_map(output_root)
    entries = scan_runs(output_root)
    blobs = PromptBlobStore(output_root).stats()
    batches = _batch_groups(output_root, entries, batch_meta)
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_batches": len(batches),
        "n_runs": len(entries),
        "n_blobs": {"prompts": blobs.get("prompts", 0), "bytes": blobs.get("bytes", 0)},
        "totals": _totals(entries, blobs, batch_meta),
        "batches": batches,
        "runs": entries,
    }


def build_index(output_root: Path) -> Dict[str, Any]:
    """Build the catalog document in memory (no writes, no lock)."""
    return _build_index_unlocked(Path(output_root))


def _atomic_write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


@contextmanager
def _index_lock(output_root: Path):
    """Serialize read-modify-write of the catalog across concurrent jobs.

    The lock lives under ``output/.state/`` so it is not part of the evidence
    tree.  A platform without advisory locks is an explicit error instead of a
    silent no-op that could corrupt the catalog.
    """
    lock_path = layout.index_lock_path(output_root)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = lock_path.open("a+")
    status = filelock.lock(fh)  # blocking: catalog updates are short
    try:
        if status != filelock.LOCKED:
            raise RuntimeError(
                "cannot serialize the results catalog: OS file locking is "
                "unsupported on this platform/filesystem "
                f"(lock file: {lock_path}; see "
                "modelbench.adapters.host.filelock)."
            )
        yield
    finally:
        try:
            if status == filelock.LOCKED:
                filelock.unlock(fh)
        finally:
            fh.close()


def write_index(output_root: Path, index: Dict[str, Any]) -> Path:
    path = layout.index_path(output_root)
    _atomic_write(path, index)
    return path


def rebuild_index(output_root: Path) -> Dict[str, Any]:
    """Rebuild ``output/index.json`` from the runs on disk (atomic, locked)."""
    output_root = Path(output_root)
    with _index_lock(output_root):
        index = _build_index_unlocked(output_root)
        write_index(output_root, index)
    return index


def load_index(output_root: Path) -> Optional[Dict[str, Any]]:
    return read_json(layout.index_path(output_root))


def _index_is_fresh(output_root: Path, index: Dict[str, Any]) -> bool:
    if index.get("schema_version") != INDEX_SCHEMA_VERSION:
        return False
    try:
        index_mtime = layout.index_path(output_root).stat().st_mtime
    except OSError:
        return False
    batch_files = iter_batch_files(output_root)
    if int(index.get("n_batches") or 0) != len(batch_files):
        return False
    run_dirs = iter_run_dirs(output_root)
    if int(index.get("n_runs") or 0) != len(run_dirs):
        return False
    newest = 0.0
    for run_dir in run_dirs:
        for name in (layout.RUN_JSON, layout.MANIFEST_JSON):
            try:
                newest = max(newest, (run_dir / name).stat().st_mtime)
            except OSError:
                continue
    for batch_file in batch_files:
        try:
            newest = max(newest, batch_file.stat().st_mtime)
        except OSError:
            continue
    return newest <= index_mtime


def ensure_index(output_root: Path, force: bool = False) -> Dict[str, Any]:
    """Return a fresh catalog, rebuilding when missing/stale/older schema."""
    output_root = Path(output_root)
    if not force:
        index = load_index(output_root)
        if index is not None and _index_is_fresh(output_root, index):
            return index
    return rebuild_index(output_root)


def _index_payload(
    output_root: Path,
    entries: List[Dict[str, Any]],
    batch_meta: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Assemble the index document from already-scanned rows."""
    entries = sorted(
        entries,
        key=lambda item: (item.get("created_at") or "", item.get("run_id") or ""),
        reverse=True,
    )
    batches = _batch_groups(output_root, entries, batch_meta)
    blobs = PromptBlobStore(output_root).stats()
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_batches": len(batches),
        "n_runs": len(entries),
        "n_blobs": {"prompts": blobs.get("prompts", 0), "bytes": blobs.get("bytes", 0)},
        "totals": _totals(entries, blobs, batch_meta),
        "batches": batches,
        "runs": entries,
    }


def upsert_run(output_root: Path, run_dir: Path) -> Path:
    """Insert/update one run in the catalog; returns the index path.

    The on-disk ``run.json``/``aggregates.json``/``MANIFEST.json`` are the
    authority (the writer calls this after finalizing a run), so nothing has to
    be injected by the caller.  A run whose batch file has not been written yet
    (BatchRunner writes it only at the very end) is still catalogued.
    """
    output_root = Path(output_root)
    run_dir = Path(run_dir)
    with _index_lock(output_root):
        index = load_index(output_root)
        if not index or index.get("schema_version") != INDEX_SCHEMA_VERSION:
            index = _build_index_unlocked(output_root)
        entry = run_entry(run_dir, output_root=output_root)
        run_key = (entry or {}).get("run_id")
        entries = [
            item
            for item in index.get("runs") or []
            if item.get("run_id") != run_key and _rel_exists(output_root, item)
        ]
        if entry is not None:
            entries.append(entry)
        payload = _index_payload(output_root, entries, _batch_meta_map(output_root))
        write_index(output_root, payload)
    return layout.index_path(output_root)


def upsert_batch(output_root: Path, batch_path: Path) -> Path:
    """Refresh one batch's rows in the catalog; returns the index path.

    ``batch_path`` is ``batches/<batch_id>.json``.  Only the runs belonging to
    this batch are re-read; the other entries are reused.
    """
    output_root = Path(output_root)
    batch_file = Path(batch_path)
    batch = load_batch(batch_file) or {}
    batch_id = str(batch.get("batch_id") or batch_file.stem)
    with _index_lock(output_root):
        index = load_index(output_root)
        if not index or index.get("schema_version") != INDEX_SCHEMA_VERSION:
            index = _build_index_unlocked(output_root)
        entries = [
            item
            for item in index.get("runs") or []
            if item.get("batch_id") != batch_id and _rel_exists(output_root, item)
        ]
        # Listed members first (declared order), then any on-disk run whose
        # run.json claims this batch but is missing from model_runs (a crashed
        # batch must not silently lose a completed run).
        candidates = run_dirs_in_batch(output_root, batch)
        seen = {run_dir.name for run_dir in candidates}
        for run_dir in iter_run_dirs(output_root):
            if run_dir.name in seen:
                continue
            run_json = read_json(run_dir / layout.RUN_JSON) or {}
            if str(run_json.get("batch_id") or "") == batch_id:
                candidates.append(run_dir)
                seen.add(run_dir.name)
        for run_dir in candidates:
            entry = run_entry(
                run_dir,
                output_root=output_root,
                batch=batch,
                batch_file=batch_file,
            )
            if entry is not None:
                entries.append(entry)
        payload = _index_payload(output_root, entries, _batch_meta_map(output_root))
        write_index(output_root, payload)
    return layout.index_path(output_root)


def _rel_exists(output_root: Path, entry: Dict[str, Any]) -> bool:
    rel = entry.get("run_path")
    if not rel:
        return False
    try:
        return (Path(output_root) / str(rel)).is_dir()
    except OSError:
        return False


# --------------------------------------------------------------------------- #
# run resolution (single-segment run_id -> directory)
# --------------------------------------------------------------------------- #


def resolve_run(output_root: Path, run_id: str) -> Path:
    """Resolve a single-segment ``run_id`` to its run directory via the index.

    Raises :class:`KeyError` when the id is not in the catalog (or the indexed
    path no longer exists) and :class:`ValueError` for ids containing a path
    separator (they must never be joined onto the output root).
    """
    output_root = Path(output_root)
    text = str(run_id or "").strip()
    if not _is_safe_segment(text):
        raise ValueError(f"invalid run id (single segment expected): {run_id!r}")
    index = ensure_index(output_root)
    for entry in index.get("runs") or []:
        if entry.get("run_id") != text:
            continue
        run_path = entry.get("run_path")
        try:
            candidate = layout.resolve_under(output_root, run_path)
        except ValueError as exc:
            raise KeyError(text) from exc
        if candidate.is_dir():
            return candidate
    raise KeyError(text)


# --------------------------------------------------------------------------- #
# safe file access (web console)
# --------------------------------------------------------------------------- #


def resolve_run_file(run_dir: Path, rel_path: str) -> Path:
    """Resolve a viewable run file; reject traversal and non-whitelisted paths."""
    run_dir = Path(run_dir).resolve()
    rel = str(rel_path or "").replace("\\", "/").lstrip("/")
    if rel not in VIEWABLE_FILES:
        raise ValueError(f"not a viewable run file: {rel_path!r}")
    candidate = (run_dir / rel).resolve()
    if candidate == run_dir or run_dir not in candidate.parents:
        raise ValueError("path escapes the run directory")
    if not candidate.is_file():
        raise FileNotFoundError(f"not a file: {rel_path}")
    return candidate


def run_file_listing(run_dir: Path) -> List[Dict[str, Any]]:
    """Inventory of files exposed by the web console for one run."""
    run_dir = Path(run_dir)
    items: List[Dict[str, Any]] = []
    for name in VIEWABLE_FILES:
        path = run_dir / name
        if path.is_file():
            size = path.stat().st_size
            items.append(
                {
                    "name": name,
                    "hidden": name == layout.RECORDS,
                    "size": size,
                    "size_human": human_size(size),
                }
            )
    return items
