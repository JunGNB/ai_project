"""Batch runner: split one multi-model benchmark action into per-model bundles.

Naming (new layout)::

    output/
      <batch_id>_performance_<model>/       # one bundle per model
        manifest.json summary.json samples.csv ... analysis/
      batches/<batch_id>.json               # batch grouping metadata
      index.json                            # per-run catalog + batches[]

The model bundle is a normal schema-2.x bundle, so analysis tools and the web
run detail keep working unchanged; the batch id links model results that came
from the same user action without mixing their raw records together.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from modelbench import __version__
from modelbench.adapters.config.settings import ProjectConfig
from modelbench.adapters.storage.excel_report import write_batch_excel
from modelbench.adapters.storage.writer import RunWriter, sanitize_name
from modelbench.application.benchmark import Orchestrator
from modelbench.application.locking import RunLock

#: Test method label used in directory names and the catalog.
DEFAULT_METHOD = "performance"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


class BatchRunner:
    """Run ``cfg.selected_models()`` sequentially, one bundle per model."""

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
        self.batch_prefix = (
            sanitize_name(str(batch_id))
            if batch_id
            else datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
        self.method = sanitize_name(method) or DEFAULT_METHOD
        self.progress_cb = progress_cb
        self.cancel_event = cancel_event
        self.overwrite = overwrite
        self.orchestrator_cls = orchestrator_cls

    # ------------------------------------------------------------------ #

    def _model_slug(self, model: Any) -> str:
        return sanitize_name(str(model.alias or model.id))

    def _run_id_for(self, model: Any) -> str:
        return f"{self.batch_prefix}_{self.method}_{self._model_slug(model)}"

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
        with RunLock(self.output_root):
            return self._run_batch()

    def _run_batch(self) -> Dict[str, Any]:
        models = self.cfg.selected_models()
        run_ids = [self._run_id_for(model) for model in models]
        existing = [
            str(self.output_root / run_id)
            for run_id in run_ids
            if (self.output_root / run_id).exists()
        ]
        if existing and not self.overwrite:
            raise FileExistsError(
                "batch output directories already exist (choose another --run-id "
                f"or use --overwrite): {existing}"
            )

        started_at = _iso_now()
        started = time.time()
        batch_meta = {
            "batch_id": self.batch_prefix,
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
            run_id = run_ids[index - 1]
            batch_info = {
                **batch_meta,
                "model_index": index,
                "model": model.label,
                "model_id": model.id,
                "run_id": run_id,
            }
            writer = RunWriter(
                self.output_root,
                run_id=run_id,
                overwrite=self.overwrite,
                batch=batch_info,
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
                    "run_id": run_id,
                    "run_dir": str(writer.run_dir),
                    "summary_path": str(writer.run_dir / "summary.json"),
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

        batch_summary = {
            "batch_id": self.batch_prefix,
            "method": self.method,
            "created_at": started_at,
            "project_version": __version__,
            "duration_s": round(time.time() - started, 1),
            "cancelled": cancelled,
            "models_selected": [model.label for model in models],
            "models_tested": [run["model"] for run in model_runs],
            "n_models_selected": len(models),
            "n_models_tested": len(model_runs),
            "model_runs": model_runs,
            "skipped": skipped,
            "warnings": warnings,
        }
        try:
            batch_summary["result_analysis"] = write_batch_excel(
                self.output_root, batch_summary
            )
        except Exception as exc:  # noqa: BLE001 - analysis must not fail the run
            message = (
                "result analysis excel failed: "
                f"{type(exc).__name__}: {exc}"
            )
            warnings.append(message)
            batch_summary["warnings"] = warnings
            batch_summary["result_analysis_error"] = message

        batch_path = self.output_root / "batches" / f"{self.batch_prefix}.json"
        _atomic_write(batch_path, batch_summary)
        batch_summary["batch_file"] = str(batch_path)
        self._progress("batch_done", batch_summary)
        return batch_summary
