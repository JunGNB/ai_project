"""Web console startup and API smoke tests (no oMLX dependency for /api/health)."""

from __future__ import annotations

import contextlib
import csv
import json
import socket
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from modelbench.adapters.storage import catalog as run_catalog
from modelbench.interfaces.web import server as web

PERF_JSONL = Path(__file__).resolve().parents[1] / "dataset" / "test" / "performance" / "testset.jsonl"
requires_perf_data = __import__("pytest").mark.skipif(
    not PERF_JSONL.is_file(), reason="dataset/test/performance/testset.jsonl is not on disk"
)


def test_port_in_use_reports_clean_error(capsys):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    try:
        rc = web.main(["--host", "127.0.0.1", "--port", str(port)])
    finally:
        sock.close()
    assert rc == 2
    err = capsys.readouterr().err
    assert "already in use" in err
    assert "--port" in err


def test_non_loopback_bind_is_refused(capsys):
    rc = web.main(["--host", "0.0.0.0", "--port", "0", "--no-open"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "refusing to bind non-loopback host" in err
    assert "--allow-external" in err


def test_health_endpoint_on_ephemeral_port():
    httpd = web.ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        port = httpd.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=5) as resp:
            body = resp.read().decode()
        assert resp.status == 200
        assert '"ok": true' in body
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


@requires_perf_data
def test_overview_exposes_perf_defaults_without_strategies(monkeypatch):
    from modelbench.adapters.config.settings import load_config

    cfg = load_config()
    cfg.server.api_key = None
    cfg.server.api_key_from_omlx_settings = False
    cfg.server.api_key_env = "MB_NO_SUCH_API_KEY"
    monkeypatch.setattr(web, "load_config", lambda: cfg)
    overview = web._server_overview()
    assert "strategies" not in overview
    assert overview["perf"]["max_tokens"] == 50
    assert overview["perf"]["samples_per_dataset"] is None
    assert overview["perf"]["context_windows"] == []
    assert overview["perf"]["context_filter"] == "全部（不筛选）"
    assert overview["run_defaults"]["warmup_requests"] == 0
    assert overview["run_defaults"]["warmup_per_model"] == 1
    assert overview["models"]
    assert [ts["name"] for ts in overview["testsets"]] == ["performance"]
    performance = overview["testsets"][0]
    assert performance["directory"].startswith("dataset/test/")
    assert len(performance["scenarios"]) == 8
    # Only declared context buckets are offered as filters (no 4K/16K traps).
    assert performance["buckets"] == [1024, 8192, 32768]
    assert overview["defaults"]["testset"] == "performance"
    # Each scenario exposes its dataset plan (sample/cell totals) so the
    # console can show "240 measured / 24 test items" before starting.
    for scenario in performance["scenarios"]:
        assert "samples_total" in scenario and "cells_total" in scenario


def test_web_cli_auto_open_flags():
    from modelbench.interfaces.cli import build_parser

    args = build_parser().parse_args(["web", "--port", "8899"])
    assert args.command == "web" and args.no_open is False and args.idle_timeout == 0.0
    args = build_parser().parse_args(["web", "--no-open", "--idle-timeout", "30"])
    assert args.no_open is True and args.idle_timeout == 30.0


def test_open_browser_uses_webbrowser(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(web.webbrowser, "open", lambda url, new=0: calls.append(url) or True)
    web._open_browser("http://127.0.0.1:8765")
    assert calls == ["http://127.0.0.1:8765"]
    assert "opened browser" in capsys.readouterr().out


def test_job_manager_has_running():
    manager = web.JobManager(output_dir="/tmp/unused")
    assert manager.has_running() is False
    job = type("Job", (), {"status": "running"})()
    manager.jobs["x"] = job
    assert manager.has_running() is True
    job.status = "done"
    assert manager.has_running() is False


def _relative_luminance(hex_color: str) -> float:
    value = hex_color.lstrip("#")
    channels = [int(value[i : i + 2], 16) / 255 for i in (0, 2, 4)]

    def adjust(channel: float) -> float:
        return channel / 12.92 if channel <= 0.03928 else ((channel + 0.055) / 1.055) ** 2.4

    red, green, blue = (adjust(channel) for channel in channels)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast(foreground: str, background: str) -> float:
    high, low = sorted(
        (_relative_luminance(foreground), _relative_luminance(background)), reverse=True
    )
    return (high + 0.05) / (low + 0.05)


def test_web_css_contrast_meets_wcag_aa():
    from pathlib import Path

    css = (Path(__file__).resolve().parents[1] / "src/modelbench/interfaces/web/static/style.css").read_text(encoding="utf-8")
    variables = {
        match.group(1): match.group(2)
        for match in __import__("re").finditer(r"--([a-z0-9-]+):\s*(#[0-9a-fA-F]{6})", css)
    }
    # Normal text on page/card and key pill/badge pairs must be >= 4.5:1.
    pairs = [
        ("text", "panel"),
        ("muted", "panel"),
        ("muted", "panel-2"),
        ("accent", "panel"),
        ("accent", "accent-soft"),
        ("ok", "ok-soft"),
        ("warn", "warn-soft"),
        ("err", "err-soft"),
    ]
    for fg, bg in pairs:
        ratio = _contrast(variables[fg], variables[bg])
        assert ratio >= 4.5, f"{fg} on {bg} contrast {ratio:.2f} < 4.5"


def test_web_page_has_compact_tree_and_progress_hooks():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src/modelbench/interfaces/web/static"
    html = (root / "index.html").read_text(encoding="utf-8")
    js = (root / "app.js").read_text(encoding="utf-8")
    for element_id in ("job-stats", "sample-modal", "sample-modal-list", "job-progress"):
        assert f'id="{element_id}"' in html
    # Compact test-set list: one checkbox per dataset/test/<name> directory,
    # no per-scenario tree rows / per-row sample counts in the UI.
    assert "data-testset-check" in js and "testset-item" in js
    assert 'id="open-testset"' in html
    assert "data-ts-all" not in js and "data-ts-none" not in js
    assert "sample-panel" not in js  # no inline sample detail panel in the tree


def test_job_manager_serializes_concurrent_runs(monkeypatch):
    import threading
    import time

    from modelbench.interfaces.web.jobs import Job

    manager = web.JobManager(output_dir="/tmp/unused-serial")
    order = []

    def fake_run_job(job, cfg):
        order.append(("start", job.id))
        time.sleep(0.15)
        order.append(("end", job.id))

    monkeypatch.setattr(manager, "_run_job", fake_run_job)
    jobs = [Job(id="job1", request={}, run_id="r1"), Job(id="job2", request={}, run_id="r2")]
    threads = [threading.Thread(target=manager._run, args=(job, None)) for job in jobs]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert order == [
        ("start", "job1"), ("end", "job1"),
        ("start", "job2"), ("end", "job2"),
    ]
    assert all(job.status == "running" for job in jobs)


def test_hardware_profile_has_live_memory_fields():
    from modelbench.adapters.host.hardware import collect_hardware_profile

    profile = collect_hardware_profile()
    memory = profile["memory"]
    for key in ("total_gb", "available_gb", "used_gb", "percent", "swap_used_gb"):
        assert key in memory



def _post_estimate(port: int, payload: dict) -> dict:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/estimate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as resp:
        return json.loads(resp.read().decode())


@requires_perf_data
def test_estimate_endpoint_is_exact_for_ui_controls():
    httpd = web.ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        port = httpd.server_address[1]
        default = _post_estimate(port, {"models": ["minicpm5-1b"]})
        assert default["samples_total"] == 240
        assert default["cells_total"] == 24
        five = _post_estimate(port, {"models": ["minicpm5-1b"], "samples": 5})
        assert five["samples_total"] == 40
        assert five["cells_total"] == 24
        windows = _post_estimate(
            port,
            {"models": ["minicpm5-1b"], "context_windows": [1024, 8192]},
        )
        assert windows["samples_total"] == 160
        assert windows["cells_total"] == 16
        # The directory tree exposes source datasets per scenario (badge text).
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/overview", timeout=10
        ) as resp:
            overview = json.loads(resp.read().decode())
        knowledge = overview["testsets"][0]["scenarios"][0]
        assert any("longbench_v2" in d for d in knowledge["datasets"])
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_web_assets_use_exact_estimate_and_compact_testset_list():
    static = Path(__file__).resolve().parents[1] / "src/modelbench/interfaces/web/static"
    js = (static / "app.js").read_text(encoding="utf-8")
    html = (static / "index.html").read_text(encoding="utf-8")
    assert '/api/estimate' in js
    # Compact test-set list: directory/title + scenario/sample/cell totals.
    assert "data-testset-check" in js and "samples_total" in js and "cells_total" in js
    # Per-scenario tree badges were removed with the compact list.
    assert "renderScenario" not in js and "data-sc-count" not in js
    assert 'low === 0 ? "0"' in js
    # Context-window chips must refresh the estimate, like samples/models.
    assert "#context-windows input" in js and "updateEstimate" in js
    # Context-window chips are rendered from the dataset's declared buckets.
    assert "renderContextWindows" in js
    assert 'value="4096"' not in html and "16384" not in html
    # Cells table must disambiguate the same dataset across 1K/8K/32K.
    assert "<th>档位</th>" in html
    assert "bucketLabel(cell.context_length)" in js
    assert 'job.status === "cancelled"' in js and '"已取消"' in js


def test_web_warmup_is_a_switch_not_a_per_cell_number():
    static = Path(__file__).resolve().parents[1] / "src/modelbench/interfaces/web/static"
    html = (static / "index.html").read_text(encoding="utf-8")
    js = (static / "app.js").read_text(encoding="utf-8")

    # The console exposes one boolean switch for the default (model-level)
    # warmup; the diagnostic per-cell count is not a UI parameter.
    assert 'id="warmup-enabled"' in html
    assert 'type="checkbox"' in html.split('id="warmup-enabled"')[1][:120]
    assert "模型加载后预热 1 次" in html
    assert 'id="warmup"' not in html
    assert '$("#warmup")' not in js
    assert "payload.warmup_per_model" in js
    assert "payload.warmup_requests" not in js


def test_build_config_warmup_switch_controls_model_warmup():
    import pytest

    from modelbench.interfaces.web.jobs import build_config

    on = build_config({"models": ["minicpm5-1b"], "warmup_per_model": 1})
    off = build_config({"models": ["minicpm5-1b"], "warmup_per_model": 0})
    assert on.run.warmup_per_model == 1
    assert off.run.warmup_per_model == 0
    # The Web form no longer sends the diagnostic per-cell count.
    assert off.run.warmup_requests == 0
    with pytest.raises(ValueError, match="warmup_per_model"):
        build_config({"models": ["minicpm5-1b"], "warmup_per_model": -1})


# --------------------------------------------------------------------------- #
# schema 3.0: synthesized run detail + paginated per-sample API
# --------------------------------------------------------------------------- #

RUN_ID = "20261107T032029Z-4184__performance__gemma-4-e4b-oq4"
BATCH_ID = "20261107T032029Z-4184"
MODEL_ID = "gemma-4-E4B-it-oQ4"
CODING_DATASET = "longbench_v2:Code Repository Understanding"
LONGBENCH_DATASET = "longbench_v1:dureader"

#: ``samples.csv`` column order (CONTRACT §6 — frozen).
SAMPLES_CSV_FIELDS = (
    "model",
    "model_id",
    "scenario",
    "dataset",
    "subtask",
    "context_length",
    "sample_id",
    "repeat",
    "phase",
    "status",
    "finish_reason",
    "length_capped",
    "input_context_tokens",
    "input_context_chars",
    "actual_prompt_tokens",
    "ttft_ms",
    "ttft_visible_ms",
    "tpot_ms",
    "gen_tps",
    "e2e_ms",
    "prefill_tps",
    "total_tps",
    "completion_tokens",
    "cached_tokens",
    "cache_mode",
    "cache_cleared",
    "cache_polluted",
    "peak_memory_gb",
    "peak_rss_mb",
    "peak_cpu_pct",
    "mean_cpu_pct",
    "prompt_sha256",
    "truncated",
    "error",
    "timestamp",
)

#: Keys the schema-2.x console read from the run summary; CONTRACT §11 freezes
#: them even though the data now comes from run.json/aggregates.json/batch.json.
SUMMARY_KEYS = (
    "run_id",
    "schema_version",
    "mode",
    "batch",
    "perf",
    "run",
    "n_records",
    "n_samples",
    "n_cells",
    "overall",
    "cells",
    "environment",
    "skipped",
    "warnings",
    "cancelled",
    "memory_errors",
    "duration_s",
    "generated_at",
    "legacy",
    "context_windows",
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sample_row(**overrides) -> dict:
    row = {
        "model": MODEL_ID,
        "model_id": MODEL_ID,
        "scenario": "coding",
        "dataset": CODING_DATASET,
        "subtask": "py",
        "context_length": 1024,
        "sample_id": "perf-coding-1k-000",
        "repeat": 0,
        "phase": "measured",
        "status": "ok",
        "finish_reason": "length",
        "length_capped": True,
        "input_context_tokens": 1013,
        "input_context_chars": 3873,
        "actual_prompt_tokens": 1013,
        "ttft_ms": 1258.8,
        "ttft_visible_ms": 1258.8,
        "tpot_ms": 15.27,
        "gen_tps": 65.5,
        "e2e_ms": 2007.3,
        "prefill_tps": 932.7,
        "total_tps": 609.77,
        "completion_tokens": 50,
        "cached_tokens": 0,
        "cache_mode": "cold",
        "cache_cleared": True,
        "cache_polluted": False,
        "peak_memory_gb": 6.72,
        "peak_rss_mb": 6490.0,
        "peak_cpu_pct": 127.7,
        "mean_cpu_pct": 43.9,
        "prompt_sha256": "ab12" * 16,
        "truncated": False,
        "error": None,
        "timestamp": "2026-11-07T03:20:29+00:00",
    }
    row.update(overrides)
    return row


SAMPLE_ROWS = [
    _sample_row(),
    _sample_row(
        sample_id="perf-coding-1k-001",
        finish_reason="stop",
        length_capped=False,
        gen_tps=64.0,
    ),
    _sample_row(
        scenario="chinese_long_context",
        dataset=LONGBENCH_DATASET,
        subtask=None,
        context_length=8192,
        sample_id="perf-chinese_long_context-8k-000",
        input_context_tokens=8000,
        ttft_ms=1300.0,
    ),
    _sample_row(
        scenario="chinese_long_context",
        dataset=LONGBENCH_DATASET,
        subtask=None,
        context_length=8192,
        sample_id="perf-chinese_long_context-8k-001",
        status="error",
        error="timeout",
        finish_reason="",
        length_capped=False,
        gen_tps=None,
        ttft_ms=None,
    ),
    _sample_row(
        context_length=8192,
        sample_id="perf-coding-8k-000",
        finish_reason="stop",
        length_capped=False,
    ),
]


def _run_json_payload() -> dict:
    return {
        "schema_version": "3.0",
        "run_id": RUN_ID,
        "batch_id": BATCH_ID,
        "method": "performance",
        "model": {
            "id": MODEL_ID,
            "alias": "gemma-4-e4b-oq4",
            "model_slug": "gemma-4-e4b-oq4",
        },
        "project_version": "0.3.0",
        "created_at": "2026-11-07T03:20:29+00:00",
        "started_at": "2026-11-07T03:20:29+00:00",
        "finished_at": "2026-11-07T03:45:00+00:00",
        "duration_s": 3012.4,
        "git": {"commit": "deadbeef", "dirty": False},
        "config": {
            "perf": {
                "samples_per_dataset": None,
                "max_tokens": 50,
                "decode_guard": True,
            },
            "run": {
                "repeats": 1,
                "warmup_requests": 0,
                "warmup_per_model": 1,
                "cache_mode": "cold",
                "thinking_mode": "disabled",
            },
            "scenarios": ["coding", "chinese_long_context"],
            "context_windows": [1024, 8192, 32768],
        },
        "env": {
            "hardware": {
                "chip": "Apple M4 Pro",
                "hostname": "bench-host",
                "arch": "arm64",
                "cpu": {"logical_cores": 14},
                "memory": {"total_gb": 48},
                "os": {"version": "15.0"},
                "gpu": [],
                "python": {"version": "3.12"},
                "upload_fields": {
                    "chip_name": "Apple M4 Pro",
                    "memory_gb": 48,
                    "gpu_cores": 20,
                    "os_version": "15.0",
                    "omlx_version": "1.2.3",
                },
            },
            "omlx": {
                "base_url": "http://127.0.0.1:12434",
                "omlx_version": "1.2.3",
                "mode": "performance",
                "default_config": True,
                "global_settings": {},
                "device": {},
                "os": {"version": "15.0"},
                "models": {
                    MODEL_ID: {
                        "model_name": MODEL_ID,
                        "engine_type": "vlm",
                        "quantization": "oQ4",
                        "context_length": 131072,
                        "estimated_size": 5612040826,
                        "settings": {"moe_expert_offload_enabled": False},
                        "feature_flags": [],
                    }
                },
                "feature_flags": [],
                "notes": [],
            },
            "model_env": {
                "3f9c0d1e2a4b5c6d": {
                    "id": MODEL_ID,
                    "engine_type": "vlm",
                    "model_type": "vlm",
                    "quantization": "oQ4",
                    "estimated_size": 5612040826,
                    "resident_estimated_size": 5612040826,
                    "context_length": 131072,
                    "settings": {},
                    "feature_flags": {},
                    "loaded": True,
                }
            },
        },
        "outcome": {
            "status": "ok",
            "cancelled": False,
            "memory_errors": 1,
            "n_selected": 6,
            "n_measured": 5,
            "n_records": 5,
            "n_samples": 5,
            "n_cells": 3,
            "skipped": [
                {
                    "scenario": "coding",
                    "dataset": CODING_DATASET,
                    "sample_id": "perf-coding-1k-002",
                    "repeat": 0,
                    "kind": "cell_aborted",
                    "reason": "memory guard",
                }
            ],
            "warnings": ["decode guard active"],
        },
        "notes": None,
    }


def _aggregates_payload() -> dict:
    overall = []
    for context_length in (1024, 8192):
        overall.append(
            {
                "model": MODEL_ID,
                "context_length": context_length,
                "n_datasets": 2,
                "n_samples": 2,
                "n_valid": 2,
                "n_warmup": 0,
                "n_contaminated": 0,
                "n_error": 0,
                "n_length_capped": 1,
                "stability_cv": 0.11,
                "unstable": False,
                "perf": {
                    "ttft_ms": {
                        "n": 2,
                        "mean": 1250.0,
                        "median": 1250.0,
                        "std": 1.0,
                        "min": 1249.0,
                        "max": 1251.0,
                        "p95": 1251.0,
                    },
                    "gen_tps": {"n": 2, "mean": 64.75},
                },
                # T-067 dual calibre: all-sample means ride along to the Web
                # console so both columns render from the same source.
                "perf_all": {
                    "ttft_ms": {"n": 2, "mean": 1260.0},
                    "gen_tps": {"n": 2, "mean": 63.9},
                },
            }
        )
    cells = [
        {
            "model": MODEL_ID,
            "scenario": "coding",
            "dataset": CODING_DATASET,
            "context_length": 1024,
            "n_samples": 2,
            "n_repeats": 1,
            "n_valid": 2,
            "n_warmup": 0,
            "n_contaminated": 0,
            "n_error": 0,
            "n_length_capped": 1,
            "perf": {"ttft_ms": {"mean": 1250.0}},
        },
        {
            "model": MODEL_ID,
            "scenario": "chinese_long_context",
            "dataset": LONGBENCH_DATASET,
            "context_length": 8192,
            "n_samples": 2,
            "n_repeats": 1,
            "n_valid": 1,
            "n_warmup": 0,
            "n_contaminated": 0,
            "n_error": 1,
            "n_length_capped": 1,
            "perf": {"ttft_ms": {"mean": 1300.0}},
        },
        {
            "model": MODEL_ID,
            "scenario": "coding",
            "dataset": CODING_DATASET,
            "context_length": 8192,
            "n_samples": 1,
            "n_repeats": 1,
            "n_valid": 1,
            "n_warmup": 0,
            "n_contaminated": 0,
            "n_error": 0,
            "n_length_capped": 0,
            "perf": {"ttft_ms": {"mean": 1210.0}},
        },
    ]
    return {
        "schema_version": "3.0",
        "run_id": RUN_ID,
        "generated_at": "2026-11-07T03:45:00+00:00",
        "n_records": 5,
        "n_samples": 5,
        "n_cells": 3,
        "overall": overall,
        "cells": cells,
    }


def _batch_payload() -> dict:
    return {
        "schema_version": "3.0",
        "batch_id": BATCH_ID,
        "source": "web",
        "method": "performance",
        "created_at": "2026-11-07T03:20:29+00:00",
        "finished_at": "2026-11-07T04:00:00+00:00",
        "project_version": "0.3.0",
        "models_selected": [MODEL_ID, "MiniCPM-V-4.7-1B"],
        "models_measured": [MODEL_ID, "MiniCPM-V-4.7-1B"],
        "n_models_selected": 2,
        "n_models_measured": 2,
        "model_runs": [
            {
                "model": "gemma-4-e4b-oq4",
                "model_id": MODEL_ID,
                "model_slug": "gemma-4-e4b-oq4",
                "method": "performance",
                "run_id": RUN_ID,
                "run_path": f"runs/{RUN_ID}",
                "model_size_gb": 5.23,
                "status": "ok",
                "n_records": 5,
                "n_samples": 5,
                "duration_s": 3012.4,
                "memory_errors": 1,
                "skipped": [],
                "cancelled": False,
            },
            {
                "model": "minicpm-v-4.7-1b",
                "model_id": "MiniCPM-V-4.7-1B",
                "model_slug": "minicpm-v-4.7-1b",
                "method": "performance",
                "run_id": "20261107T032029Z-4184__performance__minicpm-v-4.7-1b",
                "run_path": "runs/20261107T032029Z-4184__performance__minicpm-v-4.7-1b",
                "model_size_gb": 1.02,
                "status": "ok",
                "n_records": 5,
                "n_samples": 5,
                "duration_s": 900.0,
                "memory_errors": 0,
                "skipped": [],
                "cancelled": False,
            },
        ],
        "inputs": {
            "testset": {
                "path": "dataset/test/performance/testset.jsonl",
                "sha256": "1099e8b8" + "0" * 56,
                "rows": 240,
                "context_windows": [1024, 8192, 32768],
                "decode_guard_version": 1,
            }
        },
        "config": {},
        "retention": {"pinned": False, "keep_until": None},
        "provenance": {"git_commit": "deadbeef", "host": {}},
        "warnings": [],
        "errors": [],
    }


def _write_output_tree(root: Path) -> Path:
    """Create a flat schema-3.0 output tree with one measured run."""
    run_dir = root / "runs" / RUN_ID
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "run.json", _run_json_payload())
    _write_json(run_dir / "aggregates.json", _aggregates_payload())
    _write_json(
        run_dir / "MANIFEST.json",
        {
            "schema_version": "3.0",
            "run_id": RUN_ID,
            "generated_at": "2026-11-07T03:45:00+00:00",
            "algorithm": "sha256",
            "files": [],
            "totals": {"files": 0, "bytes": 0},
            "blob_refs": {"prompts": 5, "bytes": 100, "missing": 0},
        },
    )
    (run_dir / "records.jsonl").write_text(
        json.dumps({"schema_version": "3.0", "run_id": RUN_ID}) + "\n", encoding="utf-8"
    )
    (run_dir / "report.md").write_text("# report\n", encoding="utf-8")
    (run_dir / "logs" / "run.log").write_text("run log line\n", encoding="utf-8")
    with (run_dir / "samples.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLES_CSV_FIELDS)
        writer.writeheader()
        for row in SAMPLE_ROWS:
            writer.writerow({key: row.get(key) for key in SAMPLES_CSV_FIELDS})
    # Batch metadata is one file per batch, outside the run directory.
    _write_json(root / "batches" / f"{BATCH_ID}.json", _batch_payload())
    return run_dir


@pytest.fixture()
def output_tree(tmp_path, monkeypatch):
    root = tmp_path / "output"
    run_dir = _write_output_tree(root)
    run_catalog.rebuild_index(root)
    monkeypatch.setattr(web, "OUTPUT_DIR", root)
    web._CATALOG_CACHE.update({"at": 0.0, "data": None})
    try:
        yield root
    finally:
        web._CATALOG_CACHE.update({"at": 0.0, "data": None})


@contextlib.contextmanager
def _serving():
    httpd = web.ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _get(port: int, path: str):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def test_run_detail_synthesizes_old_summary_view(output_tree):
    detail = web._run_detail(RUN_ID)

    summary = detail["summary"]
    for key in SUMMARY_KEYS:
        assert key in summary, f"missing synthesized summary key: {key}"
    # Per-sample rows are no longer embedded in the summary.
    assert "samples" not in summary
    assert summary["run_id"] == RUN_ID
    assert summary["schema_version"] == "3.0"
    assert summary["mode"] == "performance"
    assert summary["legacy"] is False
    assert summary["n_samples"] == 5
    assert summary["n_records"] == 5
    assert summary["n_cells"] == 3
    assert summary["cancelled"] is False
    assert summary["memory_errors"] == 1
    assert summary["duration_s"] == 3012.4
    assert summary["generated_at"] == "2026-11-07T03:45:00+00:00"
    assert summary["perf"]["max_tokens"] == 50
    assert summary["perf"]["context_windows"] == [1024, 8192, 32768]
    assert summary["run"]["cache_mode"] == "cold"
    assert summary["run"]["warmup_per_model"] == 1
    assert summary["context_windows"] == [1024, 8192, 32768]
    # The environment pointers resolve to the single run.json snapshot.
    assert summary["environment"] == {
        "run": "run.json",
        "hardware": "run.json",
        "omlx": "run.json",
    }
    # Aggregates are passed through unchanged — including the T-067 dual
    # calibre: both perf (±10% filtered) and perf_all (all-sample) reach the
    # renderer so the two column groups come from one source of truth.
    assert len(summary["overall"]) == 2
    assert len(summary["cells"]) == 3
    assert summary["overall"][0]["model"] == MODEL_ID
    assert summary["cells"][0]["dataset"] == CODING_DATASET
    assert summary["overall"][0]["perf_all"]["ttft_ms"]["mean"] == 1260.0
    assert summary["overall"][0]["perf"]["ttft_ms"]["mean"] == 1250.0
    # Batch context drives the list grouping / subtitle.
    batch = summary["batch"]
    assert batch["batch_id"] == BATCH_ID
    assert batch["model_index"] == 1
    assert batch["models_total"] == 2
    assert batch["model_id"] == MODEL_ID
    assert batch["models_selected"] == [MODEL_ID, "MiniCPM-V-4.7-1B"]
    # Structured skip reason survives verbatim for the banner.
    assert summary["skipped"][0]["kind"] == "cell_aborted"
    assert summary["skipped"][0]["reason"] == "memory guard"
    assert summary["warnings"] == ["decode guard active"]

    # Environment evidence now lives only in run.json.
    run_json = detail["run"]
    assert run_json["schema_version"] == "3.0"
    assert run_json["env"]["hardware"]["upload_fields"]["chip_name"] == "Apple M4 Pro"
    assert run_json["env"]["model_env"]["3f9c0d1e2a4b5c6d"]["loaded"] is True
    assert detail["model_size_gb"] == pytest.approx(5.23, abs=0.01)
    # Flat layout: the run directory name is the run_id itself.
    assert Path(detail["run_dir"]).name == RUN_ID
    assert Path(detail["run_dir"]).parent == output_tree / "runs"
    assert detail["report"] == "# report\n"

    # File bar exposes the schema-3.0 names; records.jsonl stays download-only.
    names = {item["name"] for item in detail["files"]}
    assert names == {
        "run.json",
        "MANIFEST.json",
        "records.jsonl",
        "samples.csv",
        "aggregates.json",
        "report.md",
        "logs/run.log",
    }
    assert {item["name"] for item in detail["files"] if item["hidden"]} == {"records.jsonl"}

    # Facet values feed the per-sample filter selects.
    facets = detail["sample_facets"]
    assert facets["models"] == [MODEL_ID]
    assert facets["scenarios"] == ["chinese_long_context", "coding"]
    assert facets["datasets"] == [LONGBENCH_DATASET, CODING_DATASET]
    assert facets["statuses"] == ["error", "ok"]


def test_samples_api_paginates_and_filters(output_tree):
    page = web._run_samples(RUN_ID, {})
    assert set(page) == {"total", "offset", "limit", "rows"}
    assert page["total"] == 5
    assert page["offset"] == 0
    assert page["limit"] == 200
    assert [row["sample_id"] for row in page["rows"]] == [
        row["sample_id"] for row in SAMPLE_ROWS
    ]

    # CSV strings are converted to real JSON types for the console.
    first = page["rows"][0]
    assert first["context_length"] == 1024
    assert first["gen_tps"] == pytest.approx(65.5)
    assert first["length_capped"] is True
    assert first["cache_cleared"] is True
    assert first["cache_polluted"] is False
    assert first["subtask"] == "py"
    assert first["actual_prompt_tokens"] == 1013
    assert first["error"] is None
    assert first["timestamp"] == "2026-11-07T03:20:29+00:00"

    # Paging slices the filtered set but reports the pre-slice total.
    sliced = web._run_samples(RUN_ID, {"offset": "1", "limit": "2"})
    assert sliced["total"] == 5
    assert sliced["offset"] == 1
    assert sliced["limit"] == 2
    assert [row["sample_id"] for row in sliced["rows"]] == [
        "perf-coding-1k-001",
        "perf-chinese_long_context-8k-000",
    ]

    assert web._run_samples(RUN_ID, {"dataset": LONGBENCH_DATASET})["total"] == 2
    assert web._run_samples(RUN_ID, {"context_length": "8192"})["total"] == 3
    assert web._run_samples(RUN_ID, {"scenario": "coding"})["total"] == 3
    assert web._run_samples(RUN_ID, {"model": MODEL_ID})["total"] == 5
    errors = web._run_samples(RUN_ID, {"status": "error"})
    assert errors["total"] == 1
    assert errors["rows"][0]["error"] == "timeout"
    search = web._run_samples(RUN_ID, {"q": "8k-001"})
    assert search["total"] == 1
    assert search["rows"][0]["sample_id"] == "perf-chinese_long_context-8k-001"
    combined = web._run_samples(
        RUN_ID, {"scenario": "coding", "context_length": "1024", "limit": "1"}
    )
    assert combined["total"] == 2
    assert len(combined["rows"]) == 1

    # Clamp: offset never negative, limit never above the per-page ceiling.
    clamped = web._run_samples(RUN_ID, {"offset": "-5", "limit": "999999"})
    assert clamped["offset"] == 0
    assert clamped["limit"] == web._MAX_SAMPLE_PAGE


def test_web_samples_endpoint_serves_pages(output_tree):
    with _serving() as port:
        status, body = _get(port, f"/api/runs/{RUN_ID}/samples?offset=2&limit=1")
        assert status == 200
        payload = json.loads(body)
        assert payload["total"] == 5
        assert payload["offset"] == 2
        assert payload["limit"] == 1
        assert payload["rows"][0]["sample_id"] == "perf-chinese_long_context-8k-000"

        status, body = _get(
            port, f"/api/runs/{RUN_ID}/samples?context_length=8192&status=ok"
        )
        assert status == 200
        assert json.loads(body)["total"] == 2

        # Run detail over HTTP: synthesized summary + raw run.json + report.
        status, body = _get(port, f"/api/runs/{RUN_ID}")
        assert status == 200
        detail = json.loads(body)
        assert detail["summary"]["run_id"] == RUN_ID
        assert "samples" not in detail["summary"]
        assert detail["run"]["schema_version"] == "3.0"
        assert detail["report"] == "# report\n"
        assert detail["sample_facets"]["statuses"] == ["error", "ok"]

        # The catalog must list the run with its flat relative run_path.
        status, body = _get(port, "/api/runs")
        assert status == 200
        runs = json.loads(body)["runs"]
        assert [run["run_id"] for run in runs] == [RUN_ID]
        assert runs[0]["run_path"] == f"runs/{RUN_ID}"


def test_run_file_endpoint_has_explicit_whitelist(output_tree):
    from urllib.parse import quote

    with _serving() as port:
        status, body = _get(port, f"/api/runs/{RUN_ID}/file?path=run.json")
        assert status == 200
        assert json.loads(body)["run_id"] == RUN_ID

        status, body = _get(
            port, f"/api/runs/{RUN_ID}/file?path=samples.csv&download=1"
        )
        assert status == 200
        assert body.startswith("model,model_id,scenario")

        # Hidden in the file bar but explicitly downloadable.
        status, _ = _get(port, f"/api/runs/{RUN_ID}/file?path=records.jsonl")
        assert status == 200

        # Everything outside the frozen viewable whitelist is refused.
        for rel in ("secret.txt", "../run.json", "logs", "summary.json"):
            status, payload = _get(
                port, f"/api/runs/{RUN_ID}/file?path={quote(rel)}"
            )
            assert status == 403, (rel, status, payload)


def test_run_resolution_rejects_escapes(output_tree):
    with pytest.raises(ValueError):
        web._run_dir("../outside")
    with pytest.raises(ValueError):
        web._run_dir("batch/method/model")
    with pytest.raises(ValueError):
        web._run_dir("")
    with pytest.raises(FileNotFoundError):
        web._run_dir(f"{BATCH_ID}__performance__missing-model")
    with pytest.raises(ValueError):
        web._run_samples(RUN_ID, {"context_length": "1k"})
    with pytest.raises(ValueError):
        web._run_samples(RUN_ID, {"offset": "abc"})

    with _serving() as port:
        status, _ = _get(port, "/api/runs/%2E%2E/file?path=run.json")
        assert status in (400, 404)
        status, _ = _get(port, "/api/runs/no-such-run/samples")
        assert status == 404
        status, _ = _get(port, "/api/runs/no-such-run")
        assert status == 404


def test_web_app_js_paginates_samples_and_reads_run_json():
    static = Path(__file__).resolve().parents[1] / "src/modelbench/interfaces/web/static"
    js = (static / "app.js").read_text(encoding="utf-8")

    # Per-sample tab talks to the paginated API, not to an embedded row list.
    assert "/samples?" in js
    assert "summary.samples" not in js
    assert "offset: String(" in js and "limit: String(PAGE_SIZE)" in js
    assert "samplesLoaded" in js  # lazy first fetch on tab activation
    # Existing filters stay server-side; typed queries are debounced.
    assert 'params.set("dataset"' in js and 'params.set("status"' in js
    assert "sampleSearchTimer" in js
    # File bar is server-driven and no longer references retired 2.x artifacts.
    assert "file.hidden === true" in js
    assert "analysis/" not in js
    for legacy_name in (
        "summary.json",
        "manifest.json",
        "hardware.json",
        "omlx.json",
        "summary.csv",
        "cells.csv",
    ):
        assert legacy_name not in js, f"app.js still references {legacy_name}"
    # Environment tab reads the single run.json snapshot.
    assert "data.run" in js and "env.model_env" in js
    assert "data.manifest" not in js


@requires_perf_data
def test_cross_origin_write_is_rejected():
    httpd = web.ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        port = httpd.server_address[1]
        payload = json.dumps({"models": ["minicpm5-1b"]}).encode()
        cross = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/estimate",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Origin": "http://evil.example",
            },
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(cross, timeout=10)
        assert excinfo.value.code == 403
        same = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/estimate",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Origin": f"http://127.0.0.1:{port}",
            },
            method="POST",
        )
        with urllib.request.urlopen(same, timeout=10) as resp:
            assert resp.status == 200
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
