"""Local benchmark web console (stdlib only).

Run:
    python -m modelbench.interfaces.web.server [--host 127.0.0.1] [--port 8765]

The console exposes:
  * model selection and test-set directory tree (scenario + sample selection);
  * background run start, live progress and cancel;
  * run catalog (``output/index.json``), overall/cells/per-sample results;
  * raw output file view/download and "open in Finder" entry points.

It is loopback-only and unauthenticated; do not expose it.
"""

from __future__ import annotations

import argparse
import errno
import json
import mimetypes
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

from modelbench.adapters.config.settings import (
    PROJECT_ROOT,
    load_config,
    models_config_path,
    sync_models_file,
)
from modelbench.adapters.config.testsets import load_testsets
from modelbench.adapters.datasets.registry import get_adapter
from modelbench.adapters.storage import catalog as run_catalog
from modelbench.application.plan import apply_testset, dataset_plan
from modelbench.domain.context_windows import describe_windows
from .jobs import JobManager, build_config, plan_counts, read_log

STATIC_DIR = Path(__file__).resolve().parent / "static"
OUTPUT_DIR = PROJECT_ROOT / "output"
_manager = JobManager(output_dir=str(OUTPUT_DIR))

#: Sample metadata cache: key -> {"total", "dataset", "samples": [...]}
_SCENARIO_SAMPLE_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}
_CATALOG_CACHE: Dict[str, Any] = {"at": 0.0, "data": None}
_CACHE_TTL_S = 3.0
_MAX_SAMPLE_PAGE = 1000
#: Datasets whose raw files exceed this size are previewed in a short-lived
#: worker process so the web server does not retain large parse buffers.
_LARGE_DATASET_BYTES = 200 * 1024 * 1024
#: Last time any HTTP request was served (used by the optional idle watchdog).
_LAST_REQUEST_AT = 0.0


# --------------------------------------------------------------------------- #
# catalog / test-set tree helpers
# --------------------------------------------------------------------------- #


def _catalog(force: bool = False) -> Dict[str, Any]:
    """Return the run catalog, rebuilding it when missing or stale."""
    now = time.time()
    if (
        not force
        and _CATALOG_CACHE.get("data") is not None
        and now - float(_CATALOG_CACHE.get("at") or 0.0) < _CACHE_TTL_S
    ):
        return _CATALOG_CACHE["data"]
    index = run_catalog.ensure_index(OUTPUT_DIR, force=force)
    _CATALOG_CACHE.update({"at": now, "data": index})
    return index


def _display_dir(path: str) -> str:
    """Repo-relative display path for a configured directory."""
    resolved = Path(path)
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _testset_tree() -> List[Dict[str, Any]]:
    """Present test sets as their on-disk directories with scenario children.

    Scenario rows include the dataset plan (``samples_total``/``cells_total``)
    so the console can estimate ``measured requests`` and ``test items`` without
    loading the 25 MB performance JSONL (the adapter caches its index).
    """
    base = load_config()
    testsets: List[Dict[str, Any]] = []
    discovered = list(load_testsets(Path(base.datasets.testset_dir)).values())
    # Present the default test set first; other directories keep alphabetical order.
    discovered.sort(key=lambda ts: (ts.name != "performance", ts.name))
    for ts in discovered:
        cfg = base.model_copy(deep=True)
        apply_testset(cfg, ts)
        scenarios = []
        buckets: set = set()
        for name in ts.scenarios:
            spec = cfg.scenarios.get(name)
            if spec is None:
                continue
            groups = dataset_plan(cfg, spec)
            if groups:
                for group in groups:
                    value = group.get("context_length")
                    if value:
                        buckets.add(int(value))
            scenarios.append(
                {
                    "name": name,
                    "dataset": spec.dataset,
                    "params": spec.params,
                    "datasets": (
                        sorted({str(item["dataset"]) for item in groups})
                        if groups
                        else []
                    ),
                    "samples_total": (
                        sum(int(item["n"]) for item in groups) if groups else None
                    ),
                    "cells_total": len(groups) if groups else None,
                }
            )
        testsets.append(
            {
                "name": ts.name,
                "title": ts.title or ts.name,
                "description": ts.description or "",
                "directory": f"{_display_dir(base.datasets.testset_dir)}/{ts.name}",
                # Declared context buckets present in the dataset (1K/8K/32K);
                # the console only offers windows that actually exist.
                "buckets": sorted(buckets),
                "scenarios": scenarios,
            }
        )
    return testsets


def _dataset_raw_bytes(raw_dir: str, dataset: str) -> int:
    root = Path(raw_dir) / dataset
    if not root.is_dir():
        return 0
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total


def _preview_via_worker(
    raw_dir: str,
    processed_dir: str,
    dataset: str,
    params: Dict[str, Any],
    testset_dir: Optional[str] = None,
) -> Dict[str, Any]:
    payload = json.dumps(
        {
            "raw_dir": raw_dir,
            "processed_dir": processed_dir,
            "testset_dir": testset_dir,
            "dataset": dataset,
            "params": params,
        },
        ensure_ascii=False,
    )
    proc = subprocess.run(
        [sys.executable, "-m", "modelbench.interfaces.web.sample_worker"],
        input=payload,
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(PROJECT_ROOT),
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise RuntimeError(detail[-1] if detail else "sample worker failed")
    return json.loads(proc.stdout)


def _preview_in_process(
    raw_dir: str,
    processed_dir: str,
    dataset: str,
    params: Dict[str, Any],
    testset_dir: Optional[str] = None,
) -> Dict[str, Any]:
    adapter = get_adapter(
        dataset,
        raw_dir=Path(raw_dir),
        processed_dir=Path(processed_dir),
        params=params,
        testset_dir=Path(testset_dir) if testset_dir else None,
    )
    samples = adapter.load()
    return {
        "dataset": dataset,
        "total": len(samples),
        "samples": [
            {
                "sample_id": sample.sample_id,
                "subtask": sample.subtask,
                "question": " ".join((sample.question or "").split())[:180],
                "context_chars": sample.context_chars,
                "est_context_tokens": sample.est_context_tokens,
            }
            for sample in samples
        ],
    }


def _build_preview(
    raw_dir: str,
    processed_dir: str,
    dataset: str,
    params: Dict[str, Any],
    testset_dir: Optional[str] = None,
) -> Dict[str, Any]:
    if _dataset_raw_bytes(raw_dir, dataset) >= _LARGE_DATASET_BYTES:
        return _preview_via_worker(
            raw_dir, processed_dir, dataset, params, testset_dir
        )
    return _preview_in_process(raw_dir, processed_dir, dataset, params, testset_dir)


def _scenario_samples(
    testset_name: str,
    scenario_name: str,
    offset: int = 0,
    limit: int = 200,
) -> Dict[str, Any]:
    """Load (and cache) sample metadata for one scenario of one test set."""
    testsets = load_testsets(Path(load_config().datasets.testset_dir))
    if testset_name not in testsets:
        raise ValueError(f"unknown testset: {testset_name}")
    testset = testsets[testset_name]
    if scenario_name not in testset.scenarios:
        raise ValueError(
            f"scenario {scenario_name} is not part of testset {testset_name}"
        )
    cfg = load_config()
    apply_testset(cfg, testset)
    spec = cfg.scenarios[scenario_name]
    params = dict(spec.params.get("dataset_params") or {})
    key = (spec.dataset, json.dumps(params, sort_keys=True))
    if key not in _SCENARIO_SAMPLE_CACHE:
        preview = _build_preview(
            str(cfg.datasets.raw_dir),
            str(cfg.datasets.processed_dir),
            spec.dataset,
            params,
            str(cfg.datasets.testset_dir),
        )
        _SCENARIO_SAMPLE_CACHE[key] = {
            "dataset": preview["dataset"],
            "total": preview["total"],
            "samples": preview["samples"],
        }
    cached = _SCENARIO_SAMPLE_CACHE[key]
    offset = max(0, int(offset))
    limit = max(1, min(int(limit), _MAX_SAMPLE_PAGE))
    page = cached["samples"][offset : offset + limit]
    return {
        "testset": testset_name,
        "scenario": scenario_name,
        "dataset": cached["dataset"],
        "total": cached["total"],
        "offset": offset,
        "limit": limit,
        "returned": len(page),
        "samples": page,
    }


def _run_dir(run_id: str) -> Path:
    candidate = (OUTPUT_DIR / run_id).resolve()
    if candidate == OUTPUT_DIR.resolve() or OUTPUT_DIR.resolve() not in candidate.parents:
        raise ValueError(f"invalid run id: {run_id}")
    if not candidate.is_dir():
        raise FileNotFoundError(run_id)
    return candidate


def _run_detail(run_id: str) -> Dict[str, Any]:
    run_dir = _run_dir(run_id)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    omlx_path = run_dir / "omlx.json"
    omlx = json.loads(omlx_path.read_text(encoding="utf-8")) if omlx_path.is_file() else {}
    report_path = run_dir / "report.md"
    report = report_path.read_text(encoding="utf-8") if report_path.is_file() else ""
    entry = run_catalog.run_entry(run_dir, summary=summary, manifest=manifest) or {}
    return {
        "summary": summary,
        "manifest": manifest,
        "omlx": omlx,
        "report": report,
        "files": run_catalog.run_file_listing(run_dir),
        "run_dir": str(run_dir),
        "models_selected": entry.get("models_selected") or [],
        "models_measured": entry.get("models_measured") or [],
        "skipped": summary.get("skipped") or [],
        "warnings": summary.get("warnings") or [],
        "legacy": bool(entry.get("legacy")),
    }


def _live_models() -> Tuple[Dict[str, Dict[str, Any]], Optional[str]]:
    """Read the live oMLX model catalog (``GET /v1/models/status``).

    Returns ``({"id": snapshot}, error)``.  Helper models are excluded because
    they are not benchmark targets.  A non-null ``error`` means the catalog
    could not be read; callers must not treat that as "model missing" (a
    network blip would otherwise wipe the selectable list).
    """
    cfg = load_config()
    api_key = cfg.resolve_api_key()
    if not api_key:
        return {}, "no oMLX API key configured"
    from modelbench.adapters.omlx.backend import OMLXBackend

    client = OMLXBackend(
        base_url=cfg.server.base_url,
        api_key=api_key,
        admin_api_key=cfg.resolve_admin_api_key(),
        request_timeout_s=10,
        connect_timeout_s=5,
    )
    try:
        status = client.models_status()
        entries = status.get("models")
        if not isinstance(entries, list):
            # Fail closed: a changed/partial payload must not look like
            # "every model was deleted".
            return {}, "unexpected /v1/models/status payload (no models list)"
        if not entries:
            return {}, "oMLX reports 0 models"
        models = {
            str(entry.get("id")): entry
            for entry in entries
            if entry.get("id") and not entry.get("is_helper")
        }
        if not models:
            return {}, "oMLX reports 0 benchmark models"
        return models, None
    except Exception as exc:  # noqa: BLE001 - surfaced to the console as status
        return {}, f"{type(exc).__name__}: {exc}"
    finally:
        client.close()


def _model_catalog_view(
    cfg: Any,
    live: Dict[str, Dict[str, Any]],
    error: Optional[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Configured models (with live flags) plus a catalog-sync summary."""
    configured = [spec for spec in cfg.models if spec.enabled]
    # The sync diff covers *all* catalog entries (including disabled ones), so
    # the "已一致" badge matches what "同步清单" would actually write.
    configured_ids = {spec.id for spec in cfg.models}
    server_ids = set(live)
    known = error is None
    models: List[Dict[str, Any]] = []
    for spec in configured:
        snapshot = live.get(spec.id) or {}
        models.append(
            {
                "id": spec.id,
                "alias": spec.alias or spec.id,
                "notes": spec.notes,
                "loaded": bool(snapshot.get("loaded")) if snapshot else None,
                "on_server": (spec.id in server_ids) if known else None,
                "engine_type": snapshot.get("engine_type"),
                "load_settings": spec.load.settings,
            }
        )
    missing = sorted(configured_ids - server_ids) if known else []
    server_only = sorted(server_ids - configured_ids) if known else []
    model_sync = {
        "error": error,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "server_model_count": len(server_ids) if known else None,
        "configured_model_count": len(configured_ids),
        "configured_missing": missing,
        "server_only": server_only,
        "in_sync": bool(known and not missing and not server_only),
        "models_path": _display_dir(str(models_config_path())),
    }
    return models, model_sync


def _server_overview() -> Dict[str, Any]:
    cfg = load_config()
    live, server_error = _live_models()
    models, model_sync = _model_catalog_view(cfg, live, server_error)

    index = _catalog()
    runs = [
        {
            key: entry.get(key)
            for key in (
                "run_id",
                "mode",
                "created_at",
                "n_samples",
                "n_records",
                "n_cells",
                "duration_s",
                "cancelled",
                "models_selected",
                "models_measured",
                "n_models_selected",
                "n_models_measured",
                "n_skipped",
                "skipped",
                "bytes_total_human",
                "legacy",
                "batch_id",
                "method",
                "model",
                "batch_index",
                "batch_models_total",
            )
        }
        for entry in index.get("runs", [])
    ]
    index_path = OUTPUT_DIR / run_catalog.INDEX_NAME
    return {
        "server": {"base_url": cfg.server.base_url, "error": server_error},
        "models": models,
        "model_sync": model_sync,
        "testsets": _testset_tree(),
        "perf": {
            "samples_per_dataset": cfg.perf.samples_per_dataset,
            "max_tokens": cfg.perf.max_tokens,
            "context_windows": list(cfg.perf.context_windows or []),
            "context_filter": describe_windows(cfg.perf.context_windows or []),
        },
        "run_defaults": {
            "repeats": cfg.run.repeats,
            "warmup_requests": cfg.run.warmup_requests,
            "warmup_per_model": cfg.run.warmup_per_model,
            "cache_mode": cfg.run.cache_mode.value,
            "thinking_mode": cfg.run.thinking_mode,
        },
        "runs": runs[:100],
        "batches": index.get("batches", []),
        "catalog": {
            "path": str(index_path),
            "n_runs": index.get("n_runs"),
            "n_batches": index.get("n_batches"),
            "size": index_path.stat().st_size if index_path.is_file() else None,
            "generated_at": index.get("generated_at"),
        },
        "defaults": {"testset": "performance", "cache_mode": "cold", "thinking_mode": "disabled"},
    }


# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #


def _touch() -> None:
    global _LAST_REQUEST_AT
    _LAST_REQUEST_AT = time.time()


def _open_browser(url: str) -> None:
    """Open the console in the user's default browser (best effort)."""
    try:
        if webbrowser.open(url, new=2):
            print(f"[web] opened browser: {url}")
        else:
            print(f"[web] could not open a browser automatically; open {url} manually")
    except Exception as exc:  # noqa: BLE001 - browser integration is optional
        print(f"[web] browser open failed ({exc}); open {url} manually")


def _start_idle_watchdog(server: ThreadingHTTPServer, idle_timeout_s: float) -> threading.Thread:
    """Shut the console down after ``idle_timeout_s`` without requests (unless a job runs)."""

    def run() -> None:
        while True:
            time.sleep(min(5.0, max(1.0, idle_timeout_s / 4)))
            idle_for = time.time() - _LAST_REQUEST_AT
            if idle_for < idle_timeout_s:
                continue
            if _manager.has_running():
                continue
            print(
                f"[web] idle for {idle_for:.0f}s (timeout {idle_timeout_s:.0f}s) "
                "and no benchmark job is running; shutting down"
            )
            server.shutdown()
            return

    thread = threading.Thread(target=run, name="web-idle-watchdog", daemon=True)
    thread.start()
    return thread




def _open_in_file_manager(path: Path) -> None:
    """Open a local directory in Finder/Explorer (local-only console)."""
    if sys.platform == "darwin":
        command = ["open", str(path)]
    elif sys.platform.startswith("linux"):
        command = ["xdg-open", str(path)]
    elif sys.platform.startswith("win"):
        command = ["explorer", str(path)]
    else:  # pragma: no cover - unsupported platform
        raise RuntimeError(f"unsupported platform: {sys.platform}")
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=20)


class Handler(BaseHTTPRequestHandler):
    server_version = "modelbench-web/0.4"

    # -- helpers ------------------------------------------------------- #
    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, message: str, status: int = 400) -> None:
        self._json({"error": message}, status=status)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            data = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON body: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        return data

    def _query(self) -> Dict[str, str]:
        parsed = parse_qs(urlparse(self.path).query)
        return {key: values[0] for key, values in parsed.items() if values}

    def _static(self, path: str) -> None:
        rel = path.lstrip("/") or "index.html"
        candidate = (STATIC_DIR / rel).resolve()
        if STATIC_DIR.resolve() not in candidate.parents and candidate != STATIC_DIR.resolve():
            self._error("forbidden", 403)
            return
        if not candidate.is_file():
            self._error("not found", 404)
            return
        body = candidate.read_bytes()
        ctype = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_run_file(self, run_id: str) -> None:
        query = self._query()
        rel = (query.get("path") or "").strip()
        if not rel:
            self._error("path is required", 400)
            return
        run_dir = _run_dir(run_id)
        path = run_catalog.resolve_run_file(run_dir, rel)
        download = query.get("download") == "1"
        size = path.stat().st_size
        if download or size > 2 * 1024 * 1024:
            ctype = "application/octet-stream"
            disposition = f'attachment; filename="{path.name}"'
        else:
            ctype = mimetypes.guess_type(path.name)[0] or "text/plain"
            if ctype.startswith("text/") or path.suffix in (".json", ".jsonl", ".md"):
                ctype += "; charset=utf-8"
            disposition = "inline"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        self.send_header("Content-Disposition", disposition)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                self.wfile.write(chunk)

    # -- routing ------------------------------------------------------- #
    def do_GET(self) -> None:  # noqa: N802
        _touch()
        path = unquote(urlparse(self.path).path)
        try:
            if path in ("/", "/index.html"):
                self._static("index.html")
            elif path.startswith("/static/"):
                self._static(path[len("/static/") :])
            elif path == "/api/overview":
                self._json(_server_overview())
            elif path == "/api/models":
                cfg = load_config()
                live, error = _live_models()
                models, model_sync = _model_catalog_view(cfg, live, error)
                self._json(
                    {
                        "server": {"base_url": cfg.server.base_url, "error": error},
                        "models": models,
                        "model_sync": model_sync,
                    }
                )
            elif path == "/api/index":
                self._json(_catalog())
            elif path == "/api/jobs":
                self._json({"jobs": [job.snapshot() for job in _manager.list()]})
            elif path.startswith("/api/jobs/"):
                job_id = path.rsplit("/", 1)[-1]
                job = _manager.get(job_id)
                if job is None:
                    self._error("job not found", 404)
                    return
                payload = job.snapshot()
                payload["log"] = read_log(job.run_dir)
                self._json(payload)
            elif path.startswith("/api/testsets/"):
                # /api/testsets/<testset>/scenarios/<scenario>/samples
                parts = [part for part in path.split("/") if part]
                if (
                    len(parts) == 6
                    and parts[3] == "scenarios"
                    and parts[5] == "samples"
                ):
                    query = self._query()
                    self._json(
                        _scenario_samples(
                            unquote(parts[2]),
                            unquote(parts[4]),
                            offset=int(query.get("offset") or 0),
                            limit=int(query.get("limit") or 200),
                        )
                    )
                else:
                    self._error("not found", 404)
            elif path == "/api/runs":
                self._json({"runs": _catalog().get("runs", [])})
            elif path.startswith("/api/runs/") and path.endswith("/file"):
                run_id = path[len("/api/runs/") : -len("/file")].strip("/")
                self._serve_run_file(unquote(run_id))
            elif path.startswith("/api/runs/"):
                run_id = unquote(path[len("/api/runs/") :].strip("/"))
                self._json(_run_detail(run_id))
            elif path == "/api/health":
                self._json({"ok": True})
            else:
                self._error("not found", 404)
        except FileNotFoundError as exc:
            self._error(f"not found: {exc}", 404)
        except ValueError as exc:
            self._error(str(exc), 400)
        except Exception as exc:  # noqa: BLE001
            self._error(f"{type(exc).__name__}: {exc}", 500)

    def do_POST(self) -> None:  # noqa: N802
        _touch()
        path = unquote(urlparse(self.path).path)
        try:
            if path == "/api/estimate":
                payload = self._read_json()
                cfg = build_config(payload)
                counts = plan_counts(cfg)
                self._json(
                    {
                        "estimate": counts,
                        "models_total": counts["models_total"],
                        "scenarios_total": counts["scenarios_total"],
                        "cells_total": counts["cells_total"],
                        "cells_known": counts.get("cells_known", False),
                        "samples_total": counts["samples_total"],
                        "repeats": counts["repeats"],
                    }
                )
            elif path == "/api/run":
                payload = self._read_json()
                cfg = build_config(payload)
                live, error = _live_models()
                if not error:
                    # Never plan cells for a model the server does not have: the
                    # load would 404 and the run would silently produce 0 records.
                    missing = [
                        spec.id for spec in cfg.selected_models() if spec.id not in live
                    ]
                    if missing:
                        self._error(
                            "模型不在 oMLX 服务端（请点「刷新模型」并「同步清单」后重试）："
                            + ", ".join(missing),
                            400,
                        )
                        return
                job = _manager.create(payload)
                self._json({"job_id": job.id, "run_id": job.run_id}, status=202)
            elif path == "/api/models/sync":
                payload = self._read_json()
                if _manager.has_running():
                    self._error("有测试任务正在运行，结束后再同步模型清单", 409)
                    return
                live, error = _live_models()
                if error:
                    self._error(f"无法读取 oMLX 模型列表：{error}", 502)
                    return
                self._json(
                    sync_models_file(
                        models_config_path(),
                        live.keys(),
                        dry_run=payload.get("apply") is not True,
                    )
                )
            elif path.startswith("/api/jobs/") and path.endswith("/cancel"):
                job_id = path.split("/")[-2]
                if not _manager.cancel(job_id):
                    self._error("job not running", 409)
                    return
                self._json({"ok": True})
            elif path.startswith("/api/runs/") and path.endswith("/reveal"):
                run_id = unquote(path[len("/api/runs/") : -len("/reveal")].strip("/"))
                run_dir = _run_dir(run_id)
                _open_in_file_manager(run_dir)
                self._json({"ok": True, "path": str(run_dir)})
            elif path == "/api/output/reveal":
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                _open_in_file_manager(OUTPUT_DIR)
                self._json({"ok": True, "path": str(OUTPUT_DIR)})
            elif path == "/api/testsets/reveal":
                cfg = load_config()
                testset_dir = Path(cfg.datasets.testset_dir)
                testset_dir.mkdir(parents=True, exist_ok=True)
                _open_in_file_manager(testset_dir)
                self._json({"ok": True, "path": str(testset_dir)})
            elif path == "/api/index/rebuild":
                index = _catalog(force=True)
                self._json(
                    {
                        "ok": True,
                        "n_runs": index.get("n_runs"),
                        "path": str(OUTPUT_DIR / run_catalog.INDEX_NAME),
                    }
                )
            else:
                self._error("not found", 404)
        except FileNotFoundError as exc:
            self._error(f"not found: {exc}", 404)
        except ValueError as exc:
            self._error(str(exc), 400)
        except subprocess.SubprocessError as exc:
            self._error(f"cannot open file manager: {exc}", 500)
        except Exception as exc:  # noqa: BLE001
            self._error(f"{type(exc).__name__}: {exc}", 500)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[web] {self.address_string()} {fmt % args}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m modelbench.interfaces.web.server", description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--allow-external",
        action="store_true",
        help="allow binding a non-loopback host (the console is unauthenticated)",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="do not open the console in the default browser automatically",
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="shut down after SECONDS without web requests when no benchmark job is running (0 = off)",
    )
    args = parser.parse_args(argv)
    loopback = {"127.0.0.1", "localhost", "::1"}
    if args.host not in loopback and not args.allow_external:
        print(
            f"ERROR: refusing to bind non-loopback host {args.host!r}; the console has "
            "no authentication. Use --allow-external only on a trusted network.",
            file=sys.stderr,
        )
        return 2
    if args.allow_external and args.host not in loopback:
        print(
            f"WARNING: binding {args.host} exposes the unauthenticated console "
            "beyond this machine",
            file=sys.stderr,
        )
    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            print(
                f"ERROR: port {args.port} is already in use; retry with "
                f"`./modelbench web --port <free-port>`",
                file=sys.stderr,
            )
            return 2
        print(
            f"ERROR: cannot start web console on {args.host}:{args.port}: {exc}",
            file=sys.stderr,
        )
        return 1
    actual_port = server.server_address[1]
    url = f"http://{args.host}:{actual_port}"
    _touch()
    print(f"model benchmark console: {url}")
    print("press Ctrl+C to stop the server (closing the browser tab does not stop it)")
    if not args.no_open:
        threading.Timer(0.4, _open_browser, args=(url,)).start()
    if args.idle_timeout and args.idle_timeout > 0:
        _start_idle_watchdog(server, float(args.idle_timeout))
        print(f"idle auto-shutdown enabled: {args.idle_timeout:g}s without requests")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()
        print("web console stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
