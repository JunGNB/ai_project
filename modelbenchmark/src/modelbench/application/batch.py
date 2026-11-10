"""Batch runner: one benchmark action -> one batch of model runs.

Layout (schema 3.0, flat)::

    output/runs/<run_id>/              # run_id = <batch_id>__<method>__<model_slug>
      run.json MANIFEST.json records.jsonl samples.csv aggregates.json report.md
      logs/run.log
    output/batches/<batch_id>.json     # the single authoritative batch metadata
    output/exports/<batch_id>.xlsx     # derived, on demand (accuracy runs may
                                       # reuse the same batch dir per model)

Excel is *not* produced here any more: it is a derived deliverable, generated on
demand into ``output/exports/<batch_id>.xlsx`` (see ``modelbench export-analysis``).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from modelbench import __version__
from modelbench.adapters.config.settings import ProjectConfig
from modelbench.adapters.storage import layout
from modelbench.adapters.storage.blobstore import DECODE_GUARD_VERSION
from modelbench.adapters.storage.writer import RunWriter, sanitize_name
from modelbench.application.benchmark import Orchestrator
from modelbench.application.locking import RunLock

#: Test method label used in directory names and the catalog.
DEFAULT_METHOD = "performance"

#: Repository root, used to keep persisted paths relative.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _normalize_batch_id(raw: str) -> tuple[str, str]:
    """Split a caller-supplied id into ``(batch_id, source)``.

    The old layout encoded the origin in the directory name (``web-<ts>-<hex>``).
    The origin is metadata now, so it moves into ``batch.json.source`` and the
    directory keeps only the identifier.
    """
    text = sanitize_name(str(raw))
    if text.startswith("web-"):
        return text[len("web-") :], "web"
    return text, "cli"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _testset_fingerprint(cfg: ProjectConfig) -> Dict[str, Any]:
    """Locate and fingerprint the performance test set the run consumes.

    The bundle must be able to prove *which* rows it measured; previously the
    recorded path did not even exist and there was no digest at all, which is why
    the prompt bodies had to be duplicated into every bundle.
    """
    testset_dir = Path(str(getattr(cfg.datasets, "testset_dir", "dataset/test")))
    candidate = testset_dir / DEFAULT_METHOD / "testset.jsonl"
    # Store the path relative to the project root when possible: batch.json must
    # stay valid when the whole output/ tree (or the repo) is moved.  A test set
    # outside the repo cannot be expressed portably, and an absolute path would
    # make the batch machine-specific, so only the digest is kept in that case.
    try:
        display_path = candidate.resolve().relative_to(_PROJECT_ROOT).as_posix()
    except (ValueError, OSError):
        display_path = None
    info: Dict[str, Any] = {
        "path": display_path,
        "sha256": None,
        "rows": None,
        "context_windows": list(getattr(cfg.perf, "context_windows", None) or []),
        "decode_guard_version": DECODE_GUARD_VERSION,
    }
    try:
        if candidate.is_file():
            info["sha256"] = _sha256_file(candidate)
            with candidate.open("rb") as handle:
                info["rows"] = sum(1 for _ in handle)
    except OSError:
        pass
    return info


def _model_size_gb_from_run(run_dir: Path) -> Optional[float]:
    """Weights size in GiB straight from ``run.json`` (no catalog dependency)."""
    path = Path(run_dir) / layout.RUN_JSON
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    blocks = ((payload.get("env") or {}).get("model_env") or {}).values()
    for block in blocks:
        size = (block or {}).get("estimated_size")
        if isinstance(size, (int, float)) and size > 0:
            return round(float(size) / (1024 ** 3), 1)
    return None


class BatchRunner:
    """Run ``cfg.selected_models()`` sequentially, one run per model."""

    def __init__(
        self,
        cfg: ProjectConfig,
        client: Any,
        output_root: Path,
        *,
        batch_id: Optional[str] = None,
        method: str = DEFAULT_METHOD,
        progress_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        cancel_event: Any = None,
        overwrite: bool = False,
        orchestrator_cls: Any = Orchestrator,
    ):
        self.cfg = cfg
        self.client = client
        self.output_root = Path(output_root)
        raw_id = (
            sanitize_name(str(batch_id))
            if batch_id
            else datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
        self.batch_prefix, self.source = _normalize_batch_id(raw_id)
        self.method = sanitize_name(method) or DEFAULT_METHOD
        self.progress_cb = progress_cb
        self.cancel_event = cancel_event
        self.overwrite = overwrite
        self.orchestrator_cls = orchestrator_cls

    # ------------------------------------------------------------------ #

    def _model_slug(self, model: Any) -> str:
        return sanitize_name(str(model.alias or model.id))

    def _run_id_for(self, model: Any) -> str:
        return layout.make_run_id(self.batch_prefix, self.method, self._model_slug(model))

    def _progress(self, event: str, payload: Dict[str, Any]) -> None:
        if self.progress_cb is None:
            return
        try:
            self.progress_cb(event, payload)
        except Exception:  # noqa: BLE001 - progress must not break a batch
            pass

    def _cancelled(self) -> bool:
        return self.cancel_event is not None and self.cancel_event.is_set()

    # ------------------------------------------------------------------ #

    def run(self) -> Dict[str, Any]:
        # Cross-process single-instance guard: the Web JobManager lock only
        # serialises jobs inside one process.  Two CLI/Web processes on the same
        # oMLX instance would evict each other's models (AGENTS invariant 10).
        # The endpoint lock is keyed on the normalised server base_url, so it
        # also covers processes using different --output-dir roots; the output
        # root lock keeps one output tree free of interleaved writers.
        with RunLock(self.output_root, self.cfg.server.base_url):
            return self._run_batch()

    def _run_batch(self) -> Dict[str, Any]:
        models = self.cfg.selected_models()
        started_at = _iso_now()
        started = time.time()
        batch_path = layout.batch_path(self.output_root, self.batch_prefix)
        if batch_path.exists() and not self.overwrite:
            raise FileExistsError(
                "batch already exists (choose another --run-id or use "
                f"--overwrite): {batch_path}"
            )
        batch_path.parent.mkdir(parents=True, exist_ok=True)

        batch_meta = {
            "batch_id": self.batch_prefix,
            "source": self.source,
            "method": self.method,
            "created_at": started_at,
            "project_version": __version__,
            "models_selected": [model.label for model in models],
            "models_total": len(models),
        }
        model_runs: List[Dict[str, Any]] = []
        skipped: List[str] = []
        warnings: List[str] = []
        cancelled = False
        remaining = self.cfg.run.max_requests

        for index, model in enumerate(models, start=1):
            if self._cancelled() or (remaining is not None and remaining <= 0):
                cancelled = cancelled or self._cancelled()
                reason = (
                    "cancelled by user"
                    if self._cancelled()
                    else f"max_requests cap reached ({self.cfg.run.max_requests})"
                )
                # Remaining models were never started; make that explicit in the
                # batch summary instead of silently dropping them (AGENTS §9/§13).
                for pending in models[index - 1 :]:
                    skipped.append(f"model {pending.label}: not measured ({reason})")
                break
            run_id = self._run_id_for(model)
            batch_info = {
                **batch_meta,
                "model_index": index,
                "model": model.label,
                "model_id": model.id,
                "run_id": run_id,
            }
            writer = RunWriter(
                self.output_root,
                batch_id=self.batch_prefix,
                method=self.method,
                model_slug=self._model_slug(model),
                created_at=started_at,
                overwrite=self.overwrite,
                batch=batch_info,
                model={
                    "id": model.id,
                    "alias": sanitize_name(str(model.alias or model.id)),
                    "model_slug": self._model_slug(model),
                },
            )
            model_cfg = self.cfg.model_copy(deep=True)
            model_cfg.selection.models = [model.id]
            if remaining is not None:
                model_cfg.run.max_requests = remaining
            self._progress(
                "batch_model_start",
                {
                    "batch_id": self.batch_prefix,
                    "method": self.method,
                    "model": model.label,
                    "model_id": model.id,
                    "index": index,
                    "total": len(models),
                    "run_id": run_id,
                    "run_dir": str(writer.run_dir),
                },
            )
            try:
                orchestrator = self.orchestrator_cls(
                    model_cfg,
                    self.client,
                    writer,
                    progress_cb=self.progress_cb,
                    cancel_event=self.cancel_event,
                )
                summary = orchestrator.run()
            except Exception as exc:  # noqa: BLE001 - keep other models unless fail_fast
                if self.cfg.run.fail_fast:
                    raise
                message = f"model {model.id}: batch run failed: {type(exc).__name__}: {exc}"
                skipped.append(message)
                warnings.append(message)
                summary = {
                    "run_dir": str(writer.run_dir),
                    "n_records": 0,
                    "n_samples": 0,
                    "duration_s": 0,
                    "skipped": [message],
                    "warnings": [message],
                    "cancelled": False,
                }
            model_runs.append(
                {
                    "model": model.label,
                    "model_id": model.id,
                    "model_slug": self._model_slug(model),
                    "method": self.method,
                    "run_id": run_id,
                    # Flat layout: one directory per run, name == run_id.
                    "run_path": f"{layout.RUNS}/{run_id}",
                    "model_size_gb": _model_size_gb_from_run(writer.run_dir),
                    "status": "ok" if summary.get("n_records") else "empty",
                    "n_records": summary.get("n_records"),
                    "n_samples": summary.get("n_samples"),
                    "duration_s": summary.get("duration_s"),
                    "memory_errors": summary.get("memory_errors", 0),
                    "skipped": summary.get("skipped") or [],
                    "cancelled": bool(summary.get("cancelled")),
                }
            )
            skipped.extend(summary.get("skipped") or [])
            warnings.extend(summary.get("warnings") or [])
            if remaining is not None:
                remaining = max(0, remaining - int(summary.get("n_samples") or 0))
            self._progress(
                "batch_model_done",
                {
                    "batch_id": self.batch_prefix,
                    "method": self.method,
                    "model": model.label,
                    "run_id": run_id,
                    "run_dir": str(writer.run_dir),
                    "index": index,
                    "total": len(models),
                    "summary": summary,
                },
            )
            if summary.get("cancelled"):
                cancelled = True
                break

        finished_at = _iso_now()
        batch_summary: Dict[str, Any] = {
            "schema_version": "3.0",
            "batch_id": self.batch_prefix,
            "source": self.source,
            "method": self.method,
            "created_at": started_at,
            "finished_at": finished_at,
            "project_version": __version__,
            "duration_s": round(time.time() - started, 1),
            "cancelled": cancelled,
            "models_selected": [model.label for model in models],
            "models_measured": [run["model"] for run in model_runs],
            "n_models_selected": len(models),
            "n_models_measured": len(model_runs),
            "model_runs": model_runs,
            "inputs": {"testset": _testset_fingerprint(self.cfg)},
            "config": {
                "perf": self.cfg.perf.model_dump(mode="json"),
                "run": {
                    "repeats": self.cfg.run.repeats,
                    "warmup_requests": self.cfg.run.warmup_requests,
                    "warmup_per_model": self.cfg.run.warmup_per_model,
                    "cache_mode": self.cfg.run.cache_mode.value,
                    "thinking_mode": self.cfg.run.thinking_mode,
                },
                "scenarios": list(self.cfg.selected_scenarios()),
            },
            "retention": {"pinned": False, "keep_until": None},
            "provenance": {
                "git_commit": None,
                "host": {"python": os.sys.version.split()[0]},
            },
            "skipped": skipped,
            "warnings": warnings,
            "errors": [],
        }
        # Absolute paths stay in the return value (callers need them) but are
        # deliberately not persisted: batch.json must remain movable.
        _atomic_write(batch_path, batch_summary)
        batch_summary["batch_file"] = str(batch_path)
        batch_summary["batch_dir"] = str(batch_path.parent)

        # Refresh the derived index once the batch metadata exists.
        try:
            from modelbench.adapters.storage import catalog

            rebuild = getattr(catalog, "upsert_batch", None)
            if rebuild is not None:
                rebuild(self.output_root, batch_path)
            else:
                indexer = getattr(catalog, "rebuild_index", None)
                if indexer is not None:
                    indexer(self.output_root)
        except Exception as exc:  # noqa: BLE001 - index is rebuildable
            warnings.append(f"index refresh failed: {type(exc).__name__}: {exc}")
            batch_summary["warnings"] = warnings

        self._progress("batch_done", batch_summary)
        return batch_summary
