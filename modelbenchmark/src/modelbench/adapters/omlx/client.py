"""oMLX HTTP client: OpenAI-compatible inference + oMLX admin API.

The harness talks to oMLX exclusively through this module.  Two APIs are used:

* Inference API (``/v1/...``): chat completions (streaming), tokenization,
  model listing/status, model load/unload.  Authenticated by API key.
* Admin API (``/admin/api/...``): cache clearing and environment snapshots.
  Authenticated by an admin session cookie created from the API key.

All client-side timing is measured here with ``time.perf_counter`` so that
latency numbers do not depend on server-side logging.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

import httpx

from modelbench.domain.models import (
    ChatMessage,
    RequestSpec,
    ToolCall,
    ToolSpec,
    Usage,
)


class OMLXError(RuntimeError):
    """Raised for transport or protocol failures talking to oMLX."""


@dataclass
class StreamResult:
    """Raw outcome of one streaming chat completion."""

    text: str = ""
    reasoning_text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: Optional[str] = None
    usage: Usage = field(default_factory=Usage)
    num_chunks: int = 0
    first_token_ts: Optional[float] = None      # seconds since request start
    first_visible_ts: Optional[float] = None
    last_token_ts: Optional[float] = None
    http_status: int = 0
    raw_usage: Dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0


def _usage_from_dict(payload: Dict[str, Any]) -> Usage:
    details = payload.get("prompt_tokens_details") or {}
    completion_details = payload.get("completion_tokens_details") or {}
    return Usage(
        prompt_tokens=payload.get("prompt_tokens"),
        completion_tokens=payload.get("completion_tokens"),
        total_tokens=payload.get("total_tokens"),
        cached_tokens=details.get("cached_tokens"),
        reasoning_tokens=completion_details.get("reasoning_tokens"),
        time_to_first_token=payload.get("time_to_first_token"),
        time_to_first_visible_token=payload.get("time_to_first_visible_token"),
        generation_duration=payload.get("generation_duration"),
        prompt_eval_duration=payload.get("prompt_eval_duration"),
        generation_tokens_per_second=payload.get("generation_tokens_per_second"),
        prompt_tokens_per_second=payload.get("prompt_tokens_per_second"),
        total_time=payload.get("total_time"),
    )


def _as_dict(item: Any) -> Dict[str, Any]:
    """Accept pydantic models or plain dicts (ease of use in scripts/tests)."""
    if hasattr(item, "model_dump"):
        return item.model_dump(exclude_none=True)
    return dict(item)


class OMLXClient:
    """Synchronous client (the harness runs requests sequentially by design)."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        api_key: Optional[str] = None,
        admin_api_key: Optional[str] = None,
        request_timeout_s: float = 900.0,
        connect_timeout_s: float = 10.0,
        verify_ssl: bool = True,
        pid_file: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.admin_api_key = admin_api_key or api_key
        self.pid_file = pid_file
        timeout = httpx.Timeout(request_timeout_s, connect=connect_timeout_s)
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            verify=verify_ssl,
            headers=self._auth_headers(api_key),
        )
        self._admin = httpx.Client(base_url=self.base_url, timeout=timeout, verify=verify_ssl)
        self._admin_logged_in = False
        self._token_cache: Dict[str, int] = {}
        self._server_pid: Optional[int] = None

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #

    def close(self) -> None:
        self._client.close()
        self._admin.close()

    def __enter__(self) -> "OMLXClient":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    @staticmethod
    def _auth_headers(api_key: Optional[str]) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            headers["X-API-Key"] = api_key
        return headers

    # ------------------------------------------------------------------ #
    # inference API
    # ------------------------------------------------------------------ #

    def health(self) -> Dict[str, Any]:
        resp = self._client.get("/health")
        if resp.status_code != 200:
            raise OMLXError(f"health check failed: HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    def list_models(self) -> List[Dict[str, Any]]:
        resp = self._client.get("/v1/models")
        resp.raise_for_status()
        return resp.json().get("data", [])

    def models_status(self) -> Dict[str, Any]:
        resp = self._client.get("/v1/models/status")
        resp.raise_for_status()
        return resp.json()

    def model_config(self, model_id: str) -> Dict[str, Any]:
        status = self.models_status()
        for entry in status.get("models", []):
            if entry.get("id") == model_id:
                return entry
        raise OMLXError(f"model not found on server: {model_id}")

    def effective_context(self, model_id: str) -> Optional[int]:
        try:
            cfg = self.model_config(model_id)
        except OMLXError:
            return None
        return cfg.get("max_context_window") or cfg.get("model_context_length")

    def tokenize_text(self, text: str, model: Optional[str] = None) -> int:
        payload: Dict[str, Any] = {"prompt": text}
        if model:
            payload["model"] = model
        resp = self._client.post("/v1/tokenize", json=payload)
        if resp.status_code != 200:
            raise OMLXError(f"tokenize failed: HTTP {resp.status_code}: {resp.text[:300]}")
        return int(resp.json()["count"])

    def tokenize_messages(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        tools: Optional[List[ToolSpec]] = None,
    ) -> int:
        """Exact prompt token count for a rendered chat request (cached)."""
        msg_dicts = [_as_dict(m) for m in messages]
        tool_dicts = [_as_dict(t) for t in (tools or [])]
        key = json.dumps(
            {"model": model, "messages": msg_dicts, "tools": tool_dicts or None},
            ensure_ascii=False,
            sort_keys=True,
        )
        if key in self._token_cache:
            return self._token_cache[key]
        payload: Dict[str, Any] = {"messages": msg_dicts}
        if model:
            payload["model"] = model
        if tool_dicts:
            payload["tools"] = tool_dicts
        resp = self._client.post("/v1/tokenize", json=payload)
        if resp.status_code != 200:
            raise OMLXError(f"tokenize failed: HTTP {resp.status_code}: {resp.text[:300]}")
        count = int(resp.json()["count"])
        if len(self._token_cache) > 4096:
            self._token_cache.clear()
        self._token_cache[key] = count
        return count

    def chat_stream(
        self,
        request: RequestSpec,
        *,
        cache_store: bool = False,
        timeout_s: Optional[float] = None,
    ) -> StreamResult:
        """Execute one streaming chat completion and timestamp every chunk.

        ``cache_store=False`` sends ``X-oMLX-Cache-Store: off`` so benchmark
        prompts never pollute oMLX's prefix cache.
        """
        payload: Dict[str, Any] = {
            "model": request.model,
            "messages": [m.model_dump(exclude_none=True) for m in request.messages],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.tools:
            payload["tools"] = [t.model_dump(exclude_none=True) for t in request.tools]
        if request.chat_template_kwargs:
            payload["chat_template_kwargs"] = dict(request.chat_template_kwargs)
        payload.update(request.extra or {})

        headers = {}
        if not cache_store:
            headers["X-oMLX-Cache-Store"] = "off"

        result = StreamResult()
        t0 = time.perf_counter()
        text_parts: List[str] = []
        reasoning_parts: List[str] = []
        tool_calls: Dict[int, Dict[str, Any]] = {}
        last_token_ts: Optional[float] = None

        # ``httpx`` treats an explicit ``timeout=None`` as "disable all request
        # timeouts" (not "use the client default").  Passing it unconditionally
        # therefore silently disabled ``server.request_timeout_s`` for every
        # inference request, so a stalled oMLX could hang a run forever and
        # ``status=timeout`` was unreachable.  Only override the client-level
        # timeout when the caller explicitly asks for one.
        request_kwargs: Dict[str, Any] = {}
        if timeout_s is not None:
            request_kwargs["timeout"] = httpx.Timeout(timeout_s)
        try:
            with self._client.stream(
                "POST",
                "/v1/chat/completions",
                json=payload,
                headers=headers,
                **request_kwargs,
            ) as response:
                result.http_status = response.status_code
                if response.status_code != 200:
                    body = response.read().decode("utf-8", "replace")
                    raise OMLXError(
                        f"chat completion failed: HTTP {response.status_code}: {body[:800]}"
                    )
                for line in response.iter_lines():
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    result.num_chunks += 1
                    now = time.perf_counter()

                    usage_raw = chunk.get("usage")
                    if usage_raw:
                        result.raw_usage = usage_raw
                        result.usage = _usage_from_dict(usage_raw)

                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        content = delta.get("content")
                        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                        if reasoning:
                            if result.first_token_ts is None:
                                result.first_token_ts = now - t0
                            reasoning_parts.append(reasoning)
                            last_token_ts = now
                        if content:
                            if result.first_token_ts is None:
                                result.first_token_ts = now - t0
                            if result.first_visible_ts is None:
                                result.first_visible_ts = now - t0
                            text_parts.append(content)
                            last_token_ts = now
                        for call_delta in delta.get("tool_calls") or []:
                            if result.first_token_ts is None:
                                result.first_token_ts = now - t0
                            idx = int(call_delta.get("index") or 0)
                            slot = tool_calls.setdefault(
                                idx, {"id": None, "name": "", "arguments": ""}
                            )
                            if call_delta.get("id"):
                                slot["id"] = call_delta["id"]
                            fn = call_delta.get("function") or {}
                            if fn.get("name"):
                                slot["name"] += fn["name"]
                            if fn.get("arguments"):
                                slot["arguments"] += fn["arguments"]
                            last_token_ts = now
                        if choice.get("finish_reason"):
                            result.finish_reason = choice["finish_reason"]

                    # Usage-only / finish chunks are not token emissions and
                    # must not extend the decode window.
        except httpx.HTTPError as exc:  # transport level
            raise OMLXError(f"transport error talking to oMLX: {exc}") from exc

        result.text = "".join(text_parts)
        result.reasoning_text = "".join(reasoning_parts)
        result.last_token_ts = (last_token_ts - t0) if last_token_ts else None
        result.tool_calls = [
            ToolCall(id=slot.get("id"), name=slot.get("name"), arguments=slot.get("arguments"))
            for _, slot in sorted(tool_calls.items())
            if slot.get("name")
        ]
        result.elapsed_s = time.perf_counter() - t0
        return result

    # ------------------------------------------------------------------ #
    # model lifecycle
    # ------------------------------------------------------------------ #

    def unload_model(self, model_id: str) -> Dict[str, Any]:
        resp = self._client.post(f"/v1/models/{model_id}/unload")
        if resp.status_code not in (200, 202):
            raise OMLXError(f"unload {model_id} failed: HTTP {resp.status_code}: {resp.text[:300]}")
        return self._safe_json(resp)

    def load_model(self, model_id: str) -> Dict[str, Any]:
        resp = self._client.post(f"/v1/models/{model_id}/load")
        if resp.status_code not in (200, 202):
            raise OMLXError(f"load {model_id} failed: HTTP {resp.status_code}: {resp.text[:300]}")
        return self._safe_json(resp)

    def loaded_models(self) -> List[str]:
        status = self.models_status()
        return [m["id"] for m in status.get("models", []) if m.get("loaded")]

    def wait_until_loaded(self, model_id: str, timeout_s: float = 300.0) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            for entry in self.models_status().get("models", []):
                if entry.get("id") == model_id and entry.get("loaded"):
                    return True
            time.sleep(1.0)
        return False

    def wait_until_unloaded(self, model_id: str, timeout_s: float = 120.0) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            for entry in self.models_status().get("models", []):
                if entry.get("id") == model_id and not entry.get("loaded"):
                    return True
            time.sleep(0.5)
        return False

    # ------------------------------------------------------------------ #
    # admin API (cache management + environment snapshot)
    # ------------------------------------------------------------------ #

    def _admin_login(self) -> None:
        if self._admin_logged_in:
            return
        if not self.admin_api_key:
            raise OMLXError("admin API key unavailable; cannot manage caches")
        resp = self._admin.post("/admin/api/login", json={"api_key": self.admin_api_key})
        if resp.status_code != 200:
            raise OMLXError(f"admin login failed: HTTP {resp.status_code}: {resp.text[:300]}")
        self._admin_logged_in = True

    def _admin_post(self, path: str) -> Dict[str, Any]:
        self._admin_login()
        resp = self._admin.post(path)
        if resp.status_code != 200:
            raise OMLXError(f"admin POST {path} failed: HTTP {resp.status_code}: {resp.text[:400]}")
        return self._safe_json(resp)

    def _admin_get(self, path: str) -> Dict[str, Any]:
        self._admin_login()
        resp = self._admin.get(path)
        if resp.status_code != 200:
            raise OMLXError(f"admin GET {path} failed: HTTP {resp.status_code}: {resp.text[:400]}")
        return self._safe_json(resp)

    def _admin_put(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._admin_login()
        resp = self._admin.put(path, json=payload)
        if resp.status_code not in (200, 201):
            raise OMLXError(f"admin PUT {path} failed: HTTP {resp.status_code}: {resp.text[:400]}")
        return self._safe_json(resp)

    @staticmethod
    def _safe_json(resp: httpx.Response) -> Dict[str, Any]:
        try:
            data = resp.json()
        except ValueError:
            return {"raw": resp.text[:2000]}
        return data if isinstance(data, dict) else {"data": data}

    def clear_hot_cache(self) -> Dict[str, Any]:
        return self._admin_post("/admin/api/hot-cache/clear")

    def clear_ssd_cache(self) -> Dict[str, Any]:
        return self._admin_post("/admin/api/ssd-cache/clear")

    def clear_caches(self) -> Dict[str, Any]:
        """Clear both cache tiers; returns timing/report for the record."""
        started = time.perf_counter()
        hot = self.clear_hot_cache()
        ssd = self.clear_ssd_cache()
        return {
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 1),
            "hot": hot,
            "ssd": ssd,
        }

    def update_model_settings(self, model_id: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Apply per-model settings (oMLX admin API).

        The endpoint accepts a partial dict; only the provided keys are
        changed.  Settings that affect engine construction require a reload.
        """
        from urllib.parse import quote

        if not settings:
            return {"status": "skipped", "reason": "empty settings"}
        return self._admin_put(
            f"/admin/api/models/{quote(model_id, safe='')}/settings", dict(settings)
        )

    def admin_models(self) -> Dict[str, Any]:
        return self._admin_get("/admin/api/models")

    def global_settings(self) -> Dict[str, Any]:
        return self._admin_get("/admin/api/global-settings")

    def device_info(self) -> Dict[str, Any]:
        return self._admin_get("/admin/api/device-info")

    # ------------------------------------------------------------------ #
    # server process discovery (for CPU/RSS sampling)
    # ------------------------------------------------------------------ #

    def discover_server_pid(self) -> Optional[int]:
        if self._server_pid is not None:
            return self._server_pid
        if self.pid_file:
            try:
                pid = int(Path(os.path.expanduser(self.pid_file)).read_text().strip())
                self._server_pid = pid
                return pid
            except (OSError, ValueError):
                pass
        port = urlparse(self.base_url).port or 80
        try:
            out = subprocess.check_output(
                ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
                text=True,
                timeout=10,
            )
            pids = [int(x) for x in out.split() if x.strip().isdigit()]
            if pids:
                self._server_pid = pids[0]
                return self._server_pid
        except (subprocess.SubprocessError, OSError, ValueError):
            pass
        return None

    # ------------------------------------------------------------------ #
    # diagnostics
    # ------------------------------------------------------------------ #

    def probe(self) -> Dict[str, Any]:
        """Connectivity self-check used by ``modelbenchmark validate``."""
        report: Dict[str, Any] = {"base_url": self.base_url}
        try:
            health = self.health()
            report["health"] = health
            report["ok_health"] = True
        except Exception as exc:
            report["health_error"] = str(exc)
            report["ok_health"] = False
            return report
        try:
            models = self.list_models()
            report["models"] = [m.get("id") for m in models]
            report["ok_auth"] = True
        except Exception as exc:
            report["models_error"] = str(exc)
            report["ok_auth"] = False
        try:
            status = self.models_status()
            report["loaded"] = [m["id"] for m in status.get("models", []) if m.get("loaded")]
            report["ok_status"] = True
        except Exception as exc:
            report["status_error"] = str(exc)
            report["ok_status"] = False
        try:
            report["tokenize_count"] = self.tokenize_text("hello world")
            report["ok_tokenize"] = True
        except Exception as exc:
            report["tokenize_error"] = str(exc)
            report["ok_tokenize"] = False
        try:
            self._admin_login()
            report["ok_admin"] = True
        except Exception as exc:
            report["admin_error"] = str(exc)
            report["ok_admin"] = False
        report["server_pid"] = self.discover_server_pid()
        return report
