"""Model-catalog refresh/sync and stale-model guards (T-032).

Covers the failure seen on 2026-11-06: ``models.yaml`` still listed model ids
whose directories had already been deleted from oMLX, so the web run loaded
them, got HTTP 404 and skipped both models.  The console now re-reads the live
``/v1/models`` catalog, marks stale entries, can text-level sync the catalog
back into ``models.yaml``, and refuses to start a run for models the server
does not have.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from modelbench.adapters.config import settings as cfg_settings
from modelbench.adapters.config.settings import ModelSpec, load_config
from modelbench.interfaces.web import server as web

MODELS_YAML = """\
# Models available in the local oMLX instance.
# `id` must match GET /v1/models exactly.
# Last synced 2026-11-01 (2 models).
models:
  - id: Keep-Model
    alias: keep
    notes: "kept verbatim"
  - id: Stale-Model
    alias: stale
"""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _get_json(port: int, path: str):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as resp:
        return resp.status, json.loads(resp.read().decode())


def _post_json(port: int, path: str, payload: dict):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


class _Console:
    """Ephemeral web console on a loopback port."""

    def __init__(self) -> None:
        self.httpd = web.ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.port = self.httpd.server_address[1]

    def __enter__(self) -> int:
        self.thread.start()
        return self.port

    def __exit__(self, *exc_info) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


def _live(*model_ids: str, loaded: tuple[str, ...] = ()) -> dict:
    return {
        model_id: {
            "id": model_id,
            "loaded": model_id in loaded,
            "engine_type": "batched",
            "is_helper": False,
        }
        for model_id in model_ids
    }


# --------------------------------------------------------------------------- #
# models.yaml text-level sync
# --------------------------------------------------------------------------- #


def test_sync_models_file_adds_removes_and_preserves_comments(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(MODELS_YAML, encoding="utf-8")

    result = cfg_settings.sync_models_file(path, {"Keep-Model", "New-Model"})

    assert result["changed"] is True
    assert result["applied"] is True
    assert result["added"] == ["New-Model"]
    assert result["removed"] == ["Stale-Model"]
    assert result["path"] == str(path)

    text = path.read_text(encoding="utf-8")
    assert "# Models available in the local oMLX instance." in text
    assert "# `id` must match GET /v1/models exactly." in text
    assert 'notes: "kept verbatim"' in text
    assert "Stale-Model" not in text
    assert "(2 models)" in text

    cfg = load_config(models_path=path)
    assert [m.id for m in cfg.models] == ["Keep-Model", "New-Model"]
    assert cfg.models[0].alias == "keep"
    assert cfg.models[1].alias == "new-model"

    # A second sync is a no-op and leaves the bytes untouched.
    again = cfg_settings.sync_models_file(path, {"Keep-Model", "New-Model"})
    assert again["changed"] is False
    assert again["applied"] is False
    assert path.read_text(encoding="utf-8") == text


def test_sync_models_file_dry_run_does_not_touch_the_file(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(MODELS_YAML, encoding="utf-8")
    before = path.read_text(encoding="utf-8")

    result = cfg_settings.sync_models_file(path, {"Keep-Model"}, dry_run=True)

    assert result["changed"] is True
    assert result["applied"] is False
    assert result["removed"] == ["Stale-Model"]
    assert path.read_text(encoding="utf-8") == before


def test_sync_models_file_quotes_ids_that_need_it(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text("models:\n  - id: Plain\n    alias: plain\n", encoding="utf-8")

    cfg_settings.sync_models_file(path, {"Plain", "Org/Model:7B", "with space"})

    cfg = load_config(models_path=path)
    assert {m.id for m in cfg.models} == {"Plain", "Org/Model:7B", "with space"}


def test_sync_models_file_keeps_alias_unique(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(
        "models:\n  - id: Foo-Bar\n    alias: foo-bar\n", encoding="utf-8"
    )

    cfg_settings.sync_models_file(path, {"Foo-Bar", "Foo-Bar-2"})

    aliases = [m.alias for m in load_config(models_path=path).models]
    assert len(aliases) == len(set(aliases))


def test_models_config_path_points_at_the_loaded_catalog():
    from modelbench.adapters.config.settings import DEFAULT_MODELS

    assert cfg_settings.models_config_path() == DEFAULT_MODELS.resolve()


# --------------------------------------------------------------------------- #
# /api/overview model flags
# --------------------------------------------------------------------------- #


def test_overview_marks_configured_models_missing_from_server(monkeypatch):
    cfg = load_config()
    cfg.models.append(ModelSpec(id="Gone-Model", alias="gone"))
    monkeypatch.setattr(web, "load_config", lambda: cfg)
    monkeypatch.setattr(
        web,
        "_live_models",
        lambda: (_live("MiniCPM5-2B", "Server-Only", loaded=("MiniCPM5-2B",)), None),
    )

    overview = web._server_overview()

    by_id = {model["id"]: model for model in overview["models"]}
    assert by_id["MiniCPM5-2B"]["on_server"] is True
    assert by_id["MiniCPM5-2B"]["loaded"] is True
    assert by_id["Gone-Model"]["on_server"] is False
    assert by_id["Gone-Model"]["loaded"] is None

    sync = overview["model_sync"]
    assert sync["error"] is None
    assert "Gone-Model" in sync["configured_missing"]
    assert sync["server_only"] == ["Server-Only"]
    assert sync["in_sync"] is False
    assert str(sync["models_path"]).endswith("models.yaml")


def test_overview_never_claims_missing_models_when_server_is_unreachable(monkeypatch):
    cfg = load_config()
    monkeypatch.setattr(web, "load_config", lambda: cfg)
    monkeypatch.setattr(web, "_live_models", lambda: ({}, "connection refused"))

    overview = web._server_overview()

    assert overview["server"]["error"] == "connection refused"
    assert all(model["on_server"] is None for model in overview["models"])
    assert overview["model_sync"]["configured_missing"] == []
    assert overview["model_sync"]["server_only"] == []
    assert overview["model_sync"]["in_sync"] is False


# --------------------------------------------------------------------------- #
# API endpoints
# --------------------------------------------------------------------------- #


def test_models_endpoint_reports_drift(monkeypatch):
    monkeypatch.setattr(web, "_live_models", lambda: (_live("MiniCPM5-2B"), None))

    with _Console() as port:
        status, data = _get_json(port, "/api/models")

    assert status == 200
    assert {model["id"] for model in data["models"]} >= {"MiniCPM5-2B"}
    assert "MiniCPM-V-4.6" in data["model_sync"]["configured_missing"]
    assert data["model_sync"]["error"] is None


def test_run_rejects_models_the_server_does_not_have(monkeypatch):
    monkeypatch.setattr(web, "_live_models", lambda: (_live("MiniCPM5-2B"), None))
    created = []

    def fake_create(payload):
        created.append(payload)
        return SimpleNamespace(id="job-x", run_id="run-x")

    monkeypatch.setattr(web._manager, "create", fake_create)

    with _Console() as port:
        status, data = _post_json(port, "/api/run", {"models": ["minicpm-v-4.6"]})
        assert status == 400
        assert "MiniCPM-V-4.6" in data["error"]
        assert "刷新" in data["error"]
        assert created == []

        # A model that exists on the server is still accepted.
        status, data = _post_json(port, "/api/run", {"models": ["minicpm5-2b"]})
        assert status == 202
        assert data == {"job_id": "job-x", "run_id": "run-x"}
        assert created == [{"models": ["minicpm5-2b"]}]


def test_run_guard_is_skipped_when_the_server_is_unreachable(monkeypatch):
    monkeypatch.setattr(web, "_live_models", lambda: ({}, "connection refused"))
    created = []

    def fake_create(payload):
        created.append(payload)
        return SimpleNamespace(id="job-y", run_id="run-y")

    monkeypatch.setattr(web._manager, "create", fake_create)

    with _Console() as port:
        status, _ = _post_json(port, "/api/run", {"models": ["minicpm5-2b"]})

    assert status == 202
    assert created == [{"models": ["minicpm5-2b"]}]


def test_models_sync_endpoint_previews_then_writes(monkeypatch, tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(MODELS_YAML, encoding="utf-8")
    before = path.read_text(encoding="utf-8")
    monkeypatch.setattr(web, "models_config_path", lambda: path)
    monkeypatch.setattr(
        web, "_live_models", lambda: (_live("Keep-Model", "New-Model"), None)
    )
    monkeypatch.setattr(web._manager, "has_running", lambda: False)

    with _Console() as port:
        status, preview = _post_json(port, "/api/models/sync", {"apply": False})
        assert status == 200
        assert preview["changed"] is True
        assert preview["applied"] is False
        assert preview["added"] == ["New-Model"]
        assert preview["removed"] == ["Stale-Model"]
        assert path.read_text(encoding="utf-8") == before

        status, result = _post_json(port, "/api/models/sync", {"apply": True})
        assert status == 200
        assert result["changed"] is True
        assert result["applied"] is True

    assert [m.id for m in load_config(models_path=path).models] == [
        "Keep-Model",
        "New-Model",
    ]


def test_models_sync_refuses_while_a_run_is_active(monkeypatch, tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(MODELS_YAML, encoding="utf-8")
    before = path.read_text(encoding="utf-8")
    monkeypatch.setattr(web, "models_config_path", lambda: path)
    monkeypatch.setattr(web, "_live_models", lambda: (_live("Keep-Model"), None))
    monkeypatch.setattr(web._manager, "has_running", lambda: True)

    with _Console() as port:
        status, data = _post_json(port, "/api/models/sync", {"apply": True})

    assert status == 409
    assert "运行" in data["error"]
    assert path.read_text(encoding="utf-8") == before


def test_models_sync_refuses_when_catalog_cannot_be_read(monkeypatch, tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(MODELS_YAML, encoding="utf-8")
    before = path.read_text(encoding="utf-8")
    monkeypatch.setattr(web, "models_config_path", lambda: path)
    monkeypatch.setattr(web, "_live_models", lambda: ({}, "connection refused"))
    monkeypatch.setattr(web._manager, "has_running", lambda: False)

    with _Console() as port:
        status, data = _post_json(port, "/api/models/sync", {"apply": True})

    assert status == 502
    assert "connection refused" in data["error"]
    assert path.read_text(encoding="utf-8") == before


# --------------------------------------------------------------------------- #
# frontend wiring
# --------------------------------------------------------------------------- #


def test_web_assets_expose_model_refresh_and_sync():
    static = Path(__file__).resolve().parents[1] / "src/modelbench/interfaces/web/static"
    html = (static / "index.html").read_text(encoding="utf-8")
    js = (static / "app.js").read_text(encoding="utf-8")

    assert 'id="refresh-models"' in html and "刷新模型" in html
    assert 'id="sync-models"' in html and "同步清单" in html
    assert 'id="model-sync"' in html
    assert '"/api/models"' in js
    assert "/api/models/sync" in js
    assert "服务端不存在" in js
    assert "model_sync" in js


# --------------------------------------------------------------------------- #
# fail-closed edges (verifier B1): a zero/!list catalog must never be written
# --------------------------------------------------------------------------- #


def test_sync_models_file_refuses_to_empty_the_catalog(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(MODELS_YAML, encoding="utf-8")
    before = path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="empty model catalog"):
        cfg_settings.sync_models_file(path, set())

    assert path.read_text(encoding="utf-8") == before


def test_sync_models_file_keeps_comments_between_entries(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(
        "models:\n"
        "  - id: Drop-Me\n"
        "    alias: drop-me\n"
        "  # about Keep-Me (retained)\n"
        "  - id: Keep-Me\n"
        "    alias: keep-me\n",
        encoding="utf-8",
    )

    cfg_settings.sync_models_file(path, {"Keep-Me"})

    text = path.read_text(encoding="utf-8")
    assert "Drop-Me" not in text
    assert "# about Keep-Me (retained)" in text


def test_sync_models_file_updates_inline_sync_stamp(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(
        "# any configured/server-only difference.  Last synced 2026-11-01 (2 models).\n"
        "models:\n  - id: A\n    alias: a\n",
        encoding="utf-8",
    )

    cfg_settings.sync_models_file(path, {"A", "B"}, today="2026-11-07")

    assert "Last synced 2026-11-07 (2 models)" in path.read_text(encoding="utf-8")


def _fake_live_backend(monkeypatch, status_payload):
    from modelbench.adapters.omlx import backend as backend_mod

    class FakeBackend:
        def __init__(self, *args, **kwargs):
            pass

        def models_status(self):
            return status_payload

        def close(self):
            pass

    monkeypatch.setattr(backend_mod, "OMLXBackend", FakeBackend)
    cfg = load_config()
    cfg.server.api_key = "test-key"
    cfg.server.api_key_from_omlx_settings = False
    cfg.server.api_key_env = "MB_NO_SUCH_API_KEY"
    monkeypatch.setattr(web, "load_config", lambda: cfg)


def test_live_models_treats_empty_catalog_as_unknown(monkeypatch):
    _fake_live_backend(monkeypatch, {"model_count": 0, "models": []})

    live, error = web._live_models()

    assert live == {}
    assert error is not None and "0 models" in error


def test_live_models_rejects_malformed_status_payload(monkeypatch):
    _fake_live_backend(monkeypatch, {"models": None})

    live, error = web._live_models()

    assert live == {}
    assert error is not None and "payload" in error


def test_live_models_ignores_helper_models(monkeypatch):
    _fake_live_backend(
        monkeypatch,
        {"models": [{"id": "Real", "loaded": False}, {"id": "Helper", "is_helper": True}]},
    )

    live, error = web._live_models()

    assert error is None
    assert set(live) == {"Real"}


def test_models_endpoint_does_not_disable_models_when_catalog_is_empty(monkeypatch):
    monkeypatch.setattr(web, "_live_models", lambda: ({}, "oMLX reports 0 models"))

    with _Console() as port:
        status, data = _get_json(port, "/api/models")

    assert status == 200
    assert all(model["on_server"] is None for model in data["models"])
    assert data["model_sync"]["configured_missing"] == []
    assert data["model_sync"]["server_only"] == []


def test_models_sync_endpoint_refuses_empty_catalog(monkeypatch, tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text(MODELS_YAML, encoding="utf-8")
    before = path.read_text(encoding="utf-8")
    monkeypatch.setattr(web, "models_config_path", lambda: path)
    monkeypatch.setattr(web, "_live_models", lambda: ({}, "oMLX reports 0 models"))
    monkeypatch.setattr(web._manager, "has_running", lambda: False)

    with _Console() as port:
        status, data = _post_json(port, "/api/models/sync", {"apply": True})

    assert status == 502
    assert "0 models" in data["error"]
    assert path.read_text(encoding="utf-8") == before
