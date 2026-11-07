"""Background job manager for the local web console."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from modelbench.adapters.omlx.backend import OMLXBackend
from modelbench.application.batch import BatchRunner
from modelbench.application.benchmark import RunCancelled
from modelbench.adapters.config.settings import load_config, resolve_project_path
from modelbench.domain.context_windows import normalize_windows
from modelbench.application.plan import (
    apply_testset,
    dataset_plan,
    scenario_plan_extras,
)
from modelbench.adapters.config.testsets import load_testsets


@dataclass
class Job:
    id: str
    request: Dict[str, Any]
    run_id: str
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    run_dir: Optional[str] = None
    error: Optional[str] = None
    summary: Optional[Dict[str, Any]] = None
    current: Dict[str, Any] = field(default_factory=dict)
    records: int = 0
    plan: Dict[str, Any] = field(default_factory=dict)
    models_done: int = 0
    cells_done: int = 0
    cell: Dict[str, Any] = field(default_factory=dict)
    batch_id: Optional[str] = None
    method: Optional[str] = None
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def _percent(self) -> Optional[float]:
        samples_total = self.plan.get("samples_total")
        if samples_total:
            return round(min(100.0, self.records / samples_total * 100.0), 1)
        cells_total = int(self.plan.get("cells_total") or 0)
        if not cells_total:
            return None
        fraction = float(self.cells_done)
        cell_total = int(self.cell.get("total") or 0)
        if cell_total:
            fraction += min(1.0, float(self.cell.get("done") or 0) / cell_total)
        return round(min(100.0, fraction / cells_total * 100.0), 1)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "status": self.status,
            "request": self.request,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": round((self.finished_at or time.time()) - (self.started_at or self.created_at), 1),
            "run_dir": self.run_dir,
            "error": self.error,
            "records": self.records,
            "current": self.current,
            "summary": self.summary,
            "cancelling": self.cancel_event.is_set(),
            "plan": self.plan,
            "batch_id": self.batch_id,
            "method": self.method,
            "progress": {
                "models_done": self.models_done,
                "models_total": self.plan.get("models_total"),
                "cells_done": self.cells_done,
                "cells_total": self.plan.get("cells_total"),
                "samples_done": self.records,
                "samples_total": self.plan.get("samples_total"),
                "cell": dict(self.cell),
                "percent": self._percent(),
            },
        }


def plan_counts(cfg: Any) -> Dict[str, Any]:
    """Cheap plan summary for the progress/estimate UI.

    Uses ``DatasetAdapter.plan()`` (cached index, no sample loading) for
    scenarios whose size is defined by the dataset (the eight ``perf_jsonl``
    slices).  Cells are counted per (scenario, dataset, context_length) group so
    the progress bar matches the aggregates written by the run.
    """
    models = cfg.selected_models()
    scenarios = cfg.selected_scenarios()
    labels = [model.label for model in models]
    samples_per_scenario: List[int] = []
    groups_total = 0
    samples_known = True
    cells_known = True
    for name, spec in scenarios.items():
        groups = dataset_plan(cfg, spec, scenario_plan_extras(cfg, name))
        if groups is not None:
            samples_per_scenario.append(sum(int(item["n"]) for item in groups))
            groups_total += len(groups)
            continue
        # Legacy adapter without a plan: sample count may still be known, but
        # the number of (dataset, bucket) cells is not.
        cells_known = False
        explicit = cfg.perf.sample_ids_for(name)
        count = cfg.perf.samples_for(name)
        if explicit:
            samples_per_scenario.append(len(explicit))
        elif count is not None:
            samples_per_scenario.append(int(count))
        else:
            samples_known = False
            break
    samples_total = (
        sum(samples_per_scenario) * len(models) * int(cfg.run.repeats)
        if samples_known and samples_per_scenario
        else None
    )
    cells_total = (
        groups_total * len(models)
        if cells_known and groups_total
        else len(models) * len(scenarios)
    )
    return {
        "models": labels,
        "models_total": len(models),
        "scenarios": list(scenarios),
        "scenarios_total": len(scenarios),
        "cells_total": cells_total,
        "cells_known": cells_known and bool(groups_total),
        "samples_total": samples_total,
        "repeats": int(cfg.run.repeats),
    }


def apply_progress(job: Job, event: str, payload: Dict[str, Any]) -> None:
    """Update a job's progress state from one orchestrator event (pure-ish)."""
    if event == "record":
        if payload.get("status") != "warmup":
            job.records += 1
            if job.cell:
                job.cell["done"] = int(job.cell.get("done") or 0) + 1
        job.current = {"event": event, **payload}
    elif event == "cell_start":
        total = int(payload.get("samples") or 0) * int(payload.get("repeats") or 1)
        job.cell = {
            "model": payload.get("model"),
            "scenario": payload.get("scenario"),
            "dataset": payload.get("dataset"),
            "samples": int(payload.get("samples") or 0),
            "repeats": int(payload.get("repeats") or 1),
            "done": 0,
            "total": total,
            "cell_index": payload.get("cell_index"),
            "cell_total": payload.get("cell_total"),
        }
        job.current = {"event": event, **payload}
    elif event == "cell_done":
        job.cells_done += 1
        job.current = {"event": event, **payload}
    elif event == "model_start":
        job.current = {"event": event, **payload}
    elif event == "model_done":
        job.models_done += 1
        job.current = {"event": event, **payload}
    elif event in ("batch_model_start", "batch_model_done"):
        if payload.get("batch_id"):
            job.batch_id = str(payload["batch_id"])
        if payload.get("method"):
            job.method = str(payload["method"])
        if payload.get("run_dir"):
            job.run_dir = str(payload["run_dir"])
        job.current = {"event": event, **payload}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _optional_int(value: Any, field_name: str) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if parsed < 1:
        raise ValueError(f"{field_name} must be >= 1")
    return parsed


def build_config(payload: Dict[str, Any]):
    """Resolve UI request -> ProjectConfig (same semantics as CLI)."""
    cfg = load_config()
    testset_name = str(payload.get("testset") or "performance")

    testsets = load_testsets(Path(cfg.datasets.testset_dir))
    if testset_name not in testsets:
        raise ValueError(f"unknown testset: {testset_name}")
    apply_testset(cfg, testsets[testset_name])

    models = payload.get("models") or []
    if not isinstance(models, list) or not models:
        raise ValueError("select at least one model")
    cfg.selection.models = [str(m) for m in models]

    scenarios = payload.get("scenarios")
    if scenarios is not None:
        if not isinstance(scenarios, list) or not scenarios:
            raise ValueError("select at least one dataset/scenario")
        cfg.selection.scenarios = [str(name) for name in scenarios]

    sample_ids = payload.get("sample_ids") or {}
    if sample_ids:
        if not isinstance(sample_ids, dict):
            raise ValueError("sample_ids must be an object of scenario -> [sample_id]")
        cleaned: Dict[str, List[str]] = {}
        for name, ids in sample_ids.items():
            if not isinstance(ids, list):
                raise ValueError(f"sample_ids.{name} must be a list of sample ids")
            values = [str(item) for item in ids if str(item).strip()]
            if values:
                cleaned[str(name)] = values
        cfg.perf.sample_ids = cleaned

    scenario_samples = payload.get("scenario_samples") or {}
    if scenario_samples:
        if not isinstance(scenario_samples, dict):
            raise ValueError("scenario_samples must be an object of scenario -> count")
        cfg.perf.samples_by_scenario = {
            str(name): _optional_int(value, f"scenario_samples.{name}")
            for name, value in scenario_samples.items()
        }
        cfg.perf.samples_by_scenario = {
            name: value
            for name, value in cfg.perf.samples_by_scenario.items()
            if value is not None
        }

    context_windows = payload.get("context_windows")
    if context_windows not in (None, []):
        if not isinstance(context_windows, list):
            raise ValueError("context_windows must be a list of token counts")
        try:
            cfg.perf.context_windows = normalize_windows(
                [int(value) for value in context_windows]
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("context_windows entries must be integers (tokens)") from exc

    samples = _optional_int(payload.get("samples"), "samples")
    if samples is not None:
        cfg.perf.samples_per_dataset = samples
    max_tokens = _optional_int(payload.get("max_tokens"), "max_tokens")
    if max_tokens is not None:
        cfg.perf.max_tokens = max_tokens

    repeats = _optional_int(payload.get("repeats"), "repeats")
    if repeats is not None:
        cfg.run.repeats = repeats
    if payload.get("cache_mode"):
        from modelbench.domain.models import CacheMode

        cfg.run.cache_mode = CacheMode(str(payload["cache_mode"]))
    if payload.get("thinking_mode"):
        cfg.run.thinking_mode = str(payload["thinking_mode"])
    if payload.get("max_requests"):
        cfg.run.max_requests = int(payload["max_requests"])
    if payload.get("warmup_requests") not in (None, ""):
        try:
            warmup = int(payload["warmup_requests"])
        except (TypeError, ValueError) as exc:
            raise ValueError("warmup_requests must be an integer") from exc
        if warmup < 0:
            raise ValueError("warmup_requests must be >= 0")
        cfg.run.warmup_requests = warmup
    if payload.get("warmup_per_model") not in (None, ""):
        try:
            model_warmup = int(payload["warmup_per_model"])
        except (TypeError, ValueError) as exc:
            raise ValueError("warmup_per_model must be 0 or 1") from exc
        if model_warmup < 0:
            raise ValueError("warmup_per_model must be >= 0")
        cfg.run.warmup_per_model = model_warmup
    if payload.get("output_dir"):
        # Keep the already-resolved absolute output root unless the caller
        # explicitly overrides it; a relative override must resolve against the
        # project root (not the server process CWD), otherwise runs are written
        # outside the catalog that /api/runs reads.
        cfg.run.output_dir = str(resolve_project_path(str(payload["output_dir"])))
    cfg.run.isolate_models = True
    cfg.run.restore_state = True
    return cfg


class JobManager:
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        # oMLX model isolation is global, so concurrent benchmarks would evict
        # each other's models and corrupt results. Serialize runs per process.
        self._run_lock = threading.Lock()

    def create(self, payload: Dict[str, Any]) -> Job:
        cfg = build_config(payload)
        job_id = uuid.uuid4().hex[:10]
        run_id = str(payload.get("run_id") or f"web-{_now_iso()}-{job_id[:4]}")
        job = Job(id=job_id, request=payload, run_id=run_id, plan=plan_counts(cfg))
        with self._lock:
            self.jobs[job_id] = job
        thread = threading.Thread(target=self._run, args=(job, cfg), daemon=True)
        thread.start()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def list(self, limit: int = 20) -> List[Job]:
        jobs = sorted(self.jobs.values(), key=lambda job: job.created_at, reverse=True)
        return jobs[:limit]

    def has_running(self) -> bool:
        """True while any background benchmark job is queued/running."""
        return any(
            job.status in ("queued", "running", "cancelling")
            for job in self.jobs.values()
        )

    def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job is None or job.status in ("done", "failed", "cancelled"):
            return False
        job.cancel_event.set()
        job.status = "cancelling"
        return True

    def _run(self, job: Job, cfg) -> None:
        """Queue the job behind any active run, then execute it."""
        if self._run_lock.locked():
            job.status = "queued"
            job.current = {
                "event": "queued",
                "message": "等待当前测试完成（同一 oMLX 实例串行执行）",
            }
        with self._run_lock:
            if job.cancel_event.is_set():
                job.status = "cancelled"
                return
            job.status = "running"
            job.started_at = time.time()
            self._run_job(job, cfg)

    def _run_job(self, job: Job, cfg) -> None:
        backend = OMLXBackend(
            base_url=cfg.server.base_url,
            api_key=cfg.resolve_api_key(),
            admin_api_key=cfg.resolve_admin_api_key(),
            request_timeout_s=cfg.server.request_timeout_s,
            connect_timeout_s=cfg.server.connect_timeout_s,
            verify_ssl=cfg.server.verify_ssl,
            pid_file=cfg.server.pid_file,
        )

        def progress(event: str, payload: Dict[str, Any]) -> None:
            apply_progress(job, event, payload)

        try:
            summary = BatchRunner(
                cfg,
                backend,
                Path(cfg.run.output_dir),
                batch_id=job.run_id,
                progress_cb=progress,
                cancel_event=job.cancel_event,
            ).run()
            job.summary = summary
            job.status = "cancelled" if summary.get("cancelled") else "done"
        except RunCancelled:
            job.status = "cancelled"
        except Exception as exc:  # noqa: BLE001 - surface to UI
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                backend.close()
            except Exception:  # noqa: BLE001
                pass
            job.finished_at = time.time()


def read_log(run_dir: Optional[str], limit: int = 120) -> str:
    if not run_dir:
        return ""
    path = Path(run_dir) / "logs" / "run.log"
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-limit:])
