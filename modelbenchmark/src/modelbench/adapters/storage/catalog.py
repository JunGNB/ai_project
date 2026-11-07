"""Run-bundle catalog (``output/index.json``) and file helpers.

The catalog is the stable entry point for tooling/analysis: a small JSON
document listing every run bundle with its key metadata, so consumers do not
have to parse every large ``summary.json``.

Layout::

    output/
      index.json                 # catalog (this module)
      <run_id>/
        manifest.json hardware.json omlx.json models.json
        records.jsonl summary.json summary.csv cells.csv samples.csv report.md
        analysis/                # small, analysis-friendly projections
          run.json samples_long.csv environment.json README.md
        raw/ logs/
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

INDEX_NAME = "index.json"
INDEX_SCHEMA_VERSION = "1.0"

#: Files exposed through the web console "raw output" entry points.
VIEWABLE_FILES = (
    "summary.json",
    "summary.csv",
    "cells.csv",
    "samples.csv",
    "report.md",
    "manifest.json",
    "hardware.json",
    "omlx.json",
    "models.json",
    "records.jsonl",
)


def human_size(num_bytes: Optional[int]) -> str:
    if not num_bytes:
        return "0 B"
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _models_from_manifest(manifest: Dict[str, Any]) -> List[str]:
    selected: List[str] = []
    for model in (manifest.get("models") or {}).get("selected") or []:
        selected.append(str(model.get("alias") or model.get("id")))
    return selected


def run_entry(run_dir: Path, summary: Optional[Dict[str, Any]] = None,
              manifest: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Build one catalog entry from an existing run bundle."""
    run_dir = Path(run_dir)
    summary = summary if summary is not None else _load_json(run_dir / "summary.json")
    if not summary:
        return None
    manifest = manifest if manifest is not None else (_load_json(run_dir / "manifest.json") or {})
    measured = sorted(
        {entry.get("model") for entry in summary.get("overall", []) if entry.get("model")}
    )
    # Legacy means the bundle predates the schema-2.x performance layout, not
    # "has zero valid rows": a 2.x run where every model failed still uses the
    # current layout.  Schema 2.1 added per-context ``overall`` rows; 2.0
    # bundles remain readable (context label degrades to "全部").
    version = str(summary.get("schema_version") or "")
    legacy = not version.startswith("2.")
    if not measured:
        # Schema 1.x bundles only have cells; derive measured models from them.
        measured = sorted(
            {cell.get("model") for cell in summary.get("cells", []) if cell.get("model")}
        )
    n_samples = summary.get("n_samples")
    if n_samples is None:
        cells = summary.get("cells") or []
        n_samples = sum(int(cell.get("n_samples") or 0) for cell in cells) or None
    selected = _models_from_manifest(manifest)
    if not selected:
        # Old bundles before schema 2.0 stored a flat "selected" list.
        selected = [
            str(model.get("alias") or model.get("id"))
            for model in (manifest.get("selected") or [])
        ] or list(measured)
    skipped = list(summary.get("skipped") or [])
    files: Dict[str, Any] = {}
    for name in VIEWABLE_FILES:
        path = run_dir / name
        if path.is_file():
            files[name] = {"size": path.stat().st_size}
    analysis_dir = run_dir / "analysis"
    analysis_files = []
    if analysis_dir.is_dir():
        for path in sorted(analysis_dir.iterdir()):
            if path.is_file():
                analysis_files.append(
                    {"name": path.name, "size": path.stat().st_size}
                )
    total_bytes = 0
    for path in run_dir.rglob("*"):
        if path.is_file():
            try:
                total_bytes += path.stat().st_size
            except OSError:
                pass
    batch = summary.get("batch") or {}
    created_at = (
        summary.get("generated_at")
        or manifest.get("created_at")
        or datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc).isoformat()
    )
    return {
        "run_id": run_dir.name,
        "path": str(run_dir),
        "schema_version": summary.get("schema_version"),
        "mode": summary.get("mode") or manifest.get("mode") or "performance",
        "method": batch.get("method") or summary.get("mode") or "performance",
        "batch_id": batch.get("batch_id"),
        "batch_index": batch.get("model_index"),
        "batch_models_total": batch.get("models_total"),
        "model": (
            measured[0] if len(measured) == 1 else batch.get("model")
        ),
        "created_at": created_at,
        "duration_s": summary.get("duration_s"),
        "cancelled": bool(summary.get("cancelled")),
        "models_selected": selected,
        "models_measured": measured,
        "n_models_selected": len(selected),
        "n_models_measured": len(measured),
        "n_skipped": len(skipped),
        "skipped": skipped,
        "datasets": sorted({cell.get("dataset") for cell in summary.get("cells", []) if cell.get("dataset")}),
        "n_samples": n_samples,
        "n_records": summary.get("n_records"),
        "n_cells": summary.get("n_cells"),
        "memory_errors": summary.get("memory_errors") or 0,
        "perf": summary.get("perf") or {},
        "run": summary.get("run") or {},
        "git_commit": manifest.get("git_commit"),
        "files": files,
        "analysis": analysis_files,
        "bytes_total": total_bytes,
        "bytes_total_human": human_size(total_bytes),
        "legacy": bool(legacy),
    }


def scan_runs(output_root: Path) -> List[Dict[str, Any]]:
    """Scan every bundle under ``output_root`` and return catalog entries."""
    output_root = Path(output_root)
    entries: List[Dict[str, Any]] = []
    if not output_root.is_dir():
        return entries
    for path in sorted(output_root.iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        entry = run_entry(path)
        if entry:
            entries.append(entry)
    entries.sort(key=lambda entry: entry.get("created_at") or "", reverse=True)
    return entries


def _batch_groups(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        batch_id = entry.get("batch_id")
        if not batch_id:
            continue
        group = groups.setdefault(
            str(batch_id),
            {
                "batch_id": str(batch_id),
                "method": entry.get("method") or "performance",
                "created_at": entry.get("created_at"),
                "models": [],
                "n_models": 0,
                "n_samples": 0,
                "duration_s": 0.0,
                "run_ids": [],
            },
        )
        group["models"].append(
            {
                "model": entry.get("model") or (entry.get("models_measured") or [None])[0],
                "run_id": entry.get("run_id"),
                "n_samples": entry.get("n_samples"),
                "duration_s": entry.get("duration_s"),
                "skipped": entry.get("skipped") or [],
                "memory_errors": entry.get("memory_errors") or 0,
            }
        )
        group["n_models"] += 1
        group["n_samples"] += int(entry.get("n_samples") or 0)
        group["duration_s"] = round(
            group["duration_s"] + float(entry.get("duration_s") or 0.0), 1
        )
        group["run_ids"].append(entry.get("run_id"))
    return sorted(groups.values(), key=lambda item: item.get("created_at") or "", reverse=True)


def build_index(output_root: Path) -> Dict[str, Any]:
    entries = scan_runs(Path(output_root))
    batches = _batch_groups(entries)
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(Path(output_root)),
        "n_runs": len(entries),
        "n_batches": len(batches),
        "runs": entries,
        "batches": batches,
    }


def _atomic_write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


@contextmanager
def _index_lock(output_root: Path):
    """Serialize read-modify-write of the catalog across concurrent jobs."""
    lock_path = Path(output_root) / ".index.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = lock_path.open("a+")
    try:
        try:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):  # pragma: no cover - non-POSIX fallback
            pass
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):  # pragma: no cover
            pass
        fh.close()


def write_index(output_root: Path, index: Dict[str, Any]) -> Path:
    path = Path(output_root) / INDEX_NAME
    _atomic_write(path, index)
    return path


def rebuild_index(output_root: Path) -> Dict[str, Any]:
    output_root = Path(output_root)
    with _index_lock(output_root):
        index = build_index(output_root)
        write_index(output_root, index)
    return index


def ensure_index(output_root: Path, force: bool = False) -> Dict[str, Any]:
    """Return a fresh catalog, rebuilding when missing/stale."""
    output_root = Path(output_root)
    index = None if force else load_index(output_root)
    if index is not None:
        try:
            index_mtime = (output_root / INDEX_NAME).stat().st_mtime
            run_dirs = [
                path
                for path in output_root.iterdir()
                if path.is_dir() and (path / "summary.json").is_file()
            ]
            newest = max((path.stat().st_mtime for path in run_dirs), default=0.0)
            if newest <= index_mtime and len(run_dirs) == int(index.get("n_runs") or 0):
                return index
        except OSError:
            pass
    return rebuild_index(output_root)


def upsert_run(
    output_root: Path,
    run_dir: Path,
    summary: Dict[str, Any],
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    """Insert/update one run in the catalog, preserving other entries."""
    output_root = Path(output_root)
    with _index_lock(output_root):
        index_path = output_root / INDEX_NAME
        index = _load_json(index_path) or {}
        entries = [
            entry
            for entry in (index.get("runs") or [])
            if entry.get("run_id") != Path(run_dir).name
            and Path(str(entry.get("path") or "")).is_dir()
        ]
        entry = run_entry(run_dir, summary=summary, manifest=manifest)
        if entry:
            entries.append(entry)
        entries.sort(key=lambda item: item.get("created_at") or "", reverse=True)
        batches = _batch_groups(entries)
        index = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(output_root),
            "n_runs": len(entries),
            "n_batches": len(batches),
            "runs": entries,
            "batches": batches,
        }
        write_index(output_root, index)
    return index


def load_index(output_root: Path) -> Optional[Dict[str, Any]]:
    return _load_json(Path(output_root) / INDEX_NAME)


# --------------------------------------------------------------------------- #
# safe file access (web console)
# --------------------------------------------------------------------------- #


def resolve_run_file(run_dir: Path, rel_path: str) -> Path:
    """Resolve ``rel_path`` inside ``run_dir``; reject traversal/absolute paths."""
    run_dir = Path(run_dir).resolve()
    candidate = (run_dir / rel_path).resolve()
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
            items.append(
                {
                    "name": name,
                    "hidden": name == "records.jsonl",
                    "size": path.stat().st_size,
                    "size_human": human_size(path.stat().st_size),
                }
            )
    analysis_dir = run_dir / "analysis"
    if analysis_dir.is_dir():
        for path in sorted(analysis_dir.iterdir()):
            if path.is_file():
                rel = f"analysis/{path.name}"
                items.append(
                    {
                        "name": rel,
                        "hidden": False,
                        "size": path.stat().st_size,
                        "size_human": human_size(path.stat().st_size),
                    }
                )
    return items
