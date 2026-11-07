"""Web console startup and API smoke tests (no oMLX dependency for /api/health)."""

from __future__ import annotations

import json
import socket
import threading
import urllib.request
from pathlib import Path

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
    assert [ts["name"] for ts in overview["testsets"]] == ["performance", "performance_10"]
    performance = overview["testsets"][0]
    assert performance["directory"].startswith("dataset/test/")
    assert len(performance["scenarios"]) == 8
    # Only declared context buckets are offered as filters (no 4K/16K traps).
    assert performance["buckets"] == [1024, 8192, 32768]
    assert overview["defaults"]["testset"] == "performance"
    # Each scenario exposes its dataset plan (sample/cell totals) so the
    # console can show "480 measured / 24 test items" before starting.
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
        assert default["samples_total"] == 480
        assert default["cells_total"] == 24
        five = _post_estimate(port, {"models": ["minicpm5-1b"], "samples": 5})
        assert five["samples_total"] == 40
        assert five["cells_total"] == 24
        windows = _post_estimate(
            port,
            {"models": ["minicpm5-1b"], "context_windows": [1024, 8192]},
        )
        assert windows["samples_total"] == 320
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
