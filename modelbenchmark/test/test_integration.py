"""End-to-end integration test against a fake oMLX server (no model, no GPU).

Exercises the real orchestrator, scenarios, dataset adapter, writer, client and
environment collector against a deterministic local HTTP server that speaks the
same protocol as oMLX (SSE chat completions, tokenize, model status, admin
cache clearing, device/global/model configuration).

The benchmark is performance-only: records must not be scored, every measured
record must carry its input-context size, and the bundle must include the
hardware/oMLX configuration snapshot.
"""

from __future__ import annotations

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

import pytest

from modelbench.adapters.omlx.client import OMLXClient
from modelbench.adapters.storage import layout
from modelbench.adapters.storage.writer import RunWriter


def _aggregates(run_dir):
    import json as _json

    return _json.loads((Path(run_dir) / layout.AGGREGATES_JSON).read_text())


def _run_json(run_dir):
    import json as _json

    return _json.loads((Path(run_dir) / layout.RUN_JSON).read_text())


def _find_run_dir(output_root, batch_id, model_slug="model", method="performance"):
    """Locate a run directory: the flat name *is* the run id."""
    return layout.run_dir_for(Path(output_root), batch_id, method, model_slug)


def _samples_csv(run_dir):
    import csv as _csv

    with (Path(run_dir) / layout.SAMPLES_CSV).open(encoding="utf-8", newline="") as fh:
        return list(_csv.DictReader(fh))
from modelbench.application.batch import BatchRunner
from modelbench.application.benchmark import Orchestrator
from modelbench.adapters.config.settings import ModelLoadConfig, ModelSpec, load_config
from modelbench.domain.models import CacheMode, SCHEMA_VERSION


class FakeOMLXState:
    def __init__(self, mode: str = "mc") -> None:
        self.mode = mode
        self.max_window = 4096
        self.agent_steps = 0
        self.clears = 0
        self.chat_requests: list[dict] = []
        self.cache_store_headers: list[str] = []
        self.loads: list[str] = []
        self.unloads: list[str] = []
        self.settings_puts: list[dict] = []
        self.settings_status: int = 200
        self.clear_status: int = 200
        self.model_settings: dict = {}
        self.memory_attempts: int = 0
        self.events: list[str] = []
        # Chat stream knobs: a capped run sets completion_tokens=max_tokens and
        # finish_reason="length"; those samples must still be aggregated.
        self.completion_tokens: int = 20
        self.finish_reason: str = "stop"
        #: Fake tokenize override (tests actual-ISL/window logic deterministically).
        self.tokenize_override: Optional[int] = None
        #: Delay between SSE chunks; tests that need a plausible decode window
        #: set this (the default writes the whole stream in one burst).
        self.stream_chunk_delay_s: float = 0.0
        # Transport/contamination retry knobs: fail the first N chat attempts,
        # then optionally report cached_tokens from this per-attempt sequence.
        self.chat_attempts: int = 0
        self.fail_chat_attempts: int = 0
        self.cached_tokens_sequence: list[int] = []


def _agent_sse(step: int) -> bytes:
    """Step 0: ask for a search tool call.  Step 1: answer in plain text."""
    if step == 0:
        delta = {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_0",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": json.dumps({"query": "Albert Einstein"}),
                                },
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    else:
        delta = {
            "choices": [
                {"delta": {"content": "Albert Einstein"}, "finish_reason": "stop"}
            ]
        }
    usage = {
        "prompt_tokens": 200,
        "completion_tokens": 20,
        "total_tokens": 220,
        "prompt_tokens_details": {"cached_tokens": 0},
    }
    lines = [
        "data: " + json.dumps(delta, ensure_ascii=False),
        "data: " + json.dumps({"choices": [], "usage": usage}),
        "data: [DONE]",
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _sse_chunks(
    completion_tokens: int = 20, finish_reason: str = "stop", cached_tokens: int = 0
) -> bytes:
    pieces = ["答", "案：", "C"]
    lines = []
    for index, piece in enumerate(pieces):
        delta: dict = {"delta": {"content": piece}}
        if index == len(pieces) - 1:
            delta["finish_reason"] = finish_reason
        lines.append("data: " + json.dumps({"choices": [delta]}, ensure_ascii=False))
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": completion_tokens,
        "total_tokens": 100 + completion_tokens,
        "prompt_tokens_details": {"cached_tokens": cached_tokens},
        "time_to_first_token": 0.01,
        "generation_tokens_per_second": 30.0,
    }
    lines.append("data: " + json.dumps({"choices": [], "usage": usage}))
    lines.append("data: [DONE]")
    return ("\n".join(lines) + "\n").encode("utf-8")


class FakeOMLXHandler(BaseHTTPRequestHandler):
    state: FakeOMLXState  # set by test

    def log_message(self, *args):  # silence
        pass

    # -- helpers ----------------------------------------------------- #

    def _json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}")

    # -- routing ----------------------------------------------------- #

    def do_GET(self):  # noqa: N802
        path = self.path.split("?")[0]
        if path == "/health":
            return self._json({"status": "healthy"})
        if path == "/v1/models":
            return self._json(
                {
                    "data": [
                        {
                            "id": "fake-model",
                            "object": "model",
                            "max_model_len": self.state.max_window,
                        }
                    ]
                }
            )
        if path == "/v1/models/status":
            return self._json(
                {
                    "model_count": 1,
                    "loaded_count": 1,
                    "models": [
                        {
                            "id": "fake-model",
                            "model_path": "/tmp/fake-model",
                            "loaded": (
                                self.state.mode != "cold-model"
                                or (bool(self.state.loads) and not self.state.unloads)
                            ),
                            "max_context_window": self.state.max_window,
                            "model_context_length": self.state.max_window,
                            "max_tokens": 1024,
                            "engine_type": "batched",
                            "model_type": "llm",
                            "estimated_size": 1,
                        }
                    ],
                }
            )
        if path == "/admin/api/device-info":
            return self._json(
                {
                    "chip_name": "M4",
                    "chip_variant": "Pro",
                    "memory_gb": 24,
                    "gpu_cores": 16,
                }
            )
        if path == "/admin/api/global-settings":
            return self._json(
                {
                    "sampling": {"max_tokens": 1024, "temperature": 1.0},
                    "auth": {"api_key": "super-secret", "secret_key": "nope"},
                    "cache": {"enabled": True},
                }
            )
        if path == "/admin/api/models":
            return self._json(
                {
                    "models": [
                        {
                            "id": "fake-model",
                            "model_path": "/tmp/fake-model-4bit",
                            "engine_type": "batched",
                            "model_type": "llm",
                            "model_context_length": 4096,
                            "max_tokens": 1024,
                            "estimated_size": 1,
                            "moe_expert_offload_supported": False,
                            "settings": {
                                "max_tokens": 1024,
                                "temperature": 1.0,
                                "turboquant_kv_enabled": False,
                                "dflash_enabled": False,
                                "mtp_enabled": False,
                                "moe_expert_offload_enabled": False,
                                **self.state.model_settings,
                            },
                        }
                    ]
                }
            )
        return self._json({"error": f"unknown GET {path}"}, 404)

    def do_PUT(self):  # noqa: N802
        path = self.path.split("?")[0]
        if path.endswith("/settings") and path.startswith("/admin/api/models/"):
            body = self._read_json()
            self.state.settings_puts.append(body)
            self.state.model_settings.update(body)
            self.state.events.append(f"settings:{body}")
            if self.state.settings_status != 200:
                return self._json({"detail": "unsupported setting"}, self.state.settings_status)
            return self._json({"status": "ok", "requires_reload": False})
        return self._json({"error": f"unknown PUT {path}"}, 404)

    def do_POST(self):  # noqa: N802
        path = self.path.split("?")[0]
        if path == "/v1/tokenize":
            body = self._read_json()
            text = ""
            for message in body.get("messages") or []:
                text += message.get("content") or ""
            if body.get("prompt"):
                text += body["prompt"]
            if body.get("tools"):
                text += json.dumps(body["tools"])
            target_marker = re.search(r"target=(\d+)", text)
            token_marker = re.search(r"tokens=(\d+)", text)
            count = (
                self.state.tokenize_override
                if self.state.tokenize_override is not None
                else int(target_marker.group(1))
                if target_marker
                else int(token_marker.group(1))
                if token_marker
                else max(1, len(text) // 4 + 8)
            )
            return self._json(
                {"count": int(count), "max_model_len": self.state.max_window}
            )
        if path.startswith("/v1/models/") and path.endswith("/load"):
            self.state.loads.append(path.split("/")[3])
            self.state.events.append(f"load:{path.split('/')[3]}")
            return self._json({"status": "ok"})
        if path.startswith("/v1/models/") and path.endswith("/unload"):
            self.state.unloads.append(path.split("/")[3])
            self.state.events.append(f"unload:{path.split('/')[3]}")
            return self._json({"status": "ok"})
        if path == "/admin/api/login":
            self._read_json()
            return self._json({"status": "ok"})
        if path in ("/admin/api/hot-cache/clear", "/admin/api/ssd-cache/clear"):
            self.state.clears += 1
            if self.state.clear_status != 200:
                return self._json({"detail": "clear failed"}, self.state.clear_status)
            return self._json({"status": "ok", "total_cleared": 0, "bytes_reclaimed": 0})
        if path == "/v1/chat/completions":
            self.state.chat_attempts += 1
            if self.state.chat_attempts <= self.state.fail_chat_attempts:
                self._json({"error": {"message": "synthetic transient failure"}}, 500)
                return
            if self.state.mode == "error":
                self._json({"error": {"message": "synthetic failure"}}, 500)
                return
            if self.state.mode == "bad-tools":
                self.state.chat_requests.append({})
                self._json(
                    {
                        "error": {
                            "message": (
                                "invalid tools: function.parameters.type must be "
                                "object (got dict)"
                            ),
                            "code": "invalid_request_error",
                        }
                    },
                    400,
                )
                return
            if self.state.mode == "memory":
                self.state.memory_attempts += 1
                self._json(
                    {
                        "error": {
                            "message": (
                                "oMLX prefill memory guard rejected this prompt: "
                                "prefill_memory_exceeded (dynamic memory ceiling 3.70 GB)"
                            ),
                            "code": "prefill_memory_exceeded",
                        }
                    },
                    400,
                )
                return
            body = self._read_json()
            self.state.chat_requests.append(body)
            self.state.cache_store_headers.append(self.headers.get("X-oMLX-Cache-Store", ""))
            assert body.get("stream") is True
            if self.state.mode == "agent":
                step = self.state.agent_steps
                self.state.agent_steps += 1
                payload = _agent_sse(step)
            else:
                cached_tokens = 0
                if self.state.cached_tokens_sequence:
                    cached_tokens = self.state.cached_tokens_sequence.pop(0)
                payload = _sse_chunks(
                    self.state.completion_tokens,
                    self.state.finish_reason,
                    cached_tokens=cached_tokens,
                )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if self.state.stream_chunk_delay_s:
                for line in payload.splitlines(keepends=True):
                    self.wfile.write(line)
                    self.wfile.flush()
                    if line.startswith(b"data: {") and b"usage" not in line:
                        time.sleep(self.state.stream_chunk_delay_s)
            else:
                self.wfile.write(payload)
                self.wfile.flush()
            return
        return self._json({"error": f"unknown POST {path}"}, 404)


def _server(mode: str):
    state = FakeOMLXState(mode=mode)
    handler = type("Handler", (FakeOMLXHandler,), {"state": state})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, state


@pytest.fixture()
def _server_mode():
    created = []

    def factory(mode: str):
        server, state = _server(mode)
        created.append(server)
        return server, state

    yield factory
    for server in created:
        server.shutdown()
        server.server_close()


@pytest.fixture()
def fake_server():
    server, state = _server("mc")
    try:
        yield server, state
    finally:
        server.shutdown()
        server.server_close()


#: Fixture dataset per scenario (matches the real performance test set naming).
_FIXTURE_DATASETS = {
    "knowledge_qa": "mmlu",
    "instruction_following": "ifeval",
    "long_context": "longbench_v1:gov_report",
    "math_logic": "gsm8k",
    "coding": "longbench_v1:repobench-p",
    "general_agent": "bfcl",
    "search_agent": "hotpotqa",
    "multilingual": "mmmlu",
}


def _perf_rows(
    scenarios,
    *,
    per_scenario: int = 2,
    tokens: int = 100,
    bucket: str = "1k",
    prebuilt: bool = True,
    question: str | None = None,
    dataset: str | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for scenario in scenarios:
        for index in range(per_scenario):
            row = {
                "id": f"fixture-{scenario}-{bucket}-{index:03d}",
                "scenario": scenario,
                "bucket": bucket,
                "dataset": dataset or _FIXTURE_DATASETS.get(scenario, "mmlu"),
                "tokens": int(tokens),
                "tools": [],
                "constructed": bool(prebuilt),
                "meta": {"lang": "en", "gold": "A"},
            }
            if prebuilt:
                row["messages"] = [
                    {
                        "role": "user",
                        "content": (
                            f"Fixture question {scenario} {index} "
                            f"target={ {'1k': 1024, '8k': 8192, '32k': 32768}[bucket] } "
                            f"tokens={int(tokens)}"
                        ),
                    }
                ]
            else:
                row["question"] = question or f"Fixture question {scenario} {index}"
            rows.append(row)
    return rows


def _write_perf_file(tmp_path, rows) -> Path:
    root = tmp_path / "perfdata"
    path = root / "performance" / "testset.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return root


def _apply_perf_fixture(
    cfg,
    tmp_path,
    *,
    scenarios=("knowledge_qa",),
    per_scenario: int = 2,
    tokens: int = 100,
    bucket: str = "1k",
    prebuilt: bool = True,
    question: str | None = None,
    extra_rows=(),
):
    rows = list(extra_rows) + _perf_rows(
        scenarios,
        per_scenario=per_scenario,
        tokens=tokens,
        bucket=bucket,
        prebuilt=prebuilt,
        question=question,
    )
    cfg.datasets.testset_dir = str(_write_perf_file(tmp_path, rows))
    cfg.selection.scenarios = list(scenarios)
    cfg.perf.samples_per_dataset = per_scenario
    return cfg


def _fake_config(
    fake_server,
    tmp_path,
    *,
    repeats: int = 1,
    samples: int = 2,
    scenarios=("knowledge_qa",),
    tokens: int = 100,
    bucket: str = "1k",
    prebuilt: bool = True,
    question: str | None = None,
):
    server, state = fake_server
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    cfg = load_config()
    cfg.run.warmup_per_model = 0
    cfg.server.base_url = base_url
    cfg.server.api_key = "fake-key"
    cfg.run.output_dir = str(tmp_path)
    cfg.run.warmup_requests = 1
    cfg.run.warmup_per_model = 0  # legacy tests count per-cell warmups only
    cfg.run.cache_settle_ms = 0
    cfg.run.cpu_sample_interval_ms = 20
    cfg.run.repeats = repeats
    cfg.run.max_requests = None
    cfg.selection.models = ["fake-model"]
    cfg.models = [ModelSpec(id="fake-model", alias="fake-model")]
    _apply_perf_fixture(
        cfg,
        tmp_path,
        scenarios=scenarios,
        per_scenario=samples,
        tokens=tokens,
        bucket=bucket,
        prebuilt=prebuilt,
        question=question,
    )
    # The fake tokenizer otherwise counts the tiny fixture text, which would
    # (correctly) look far outside the declared bucket and be skipped by the
    # actual-ISL tolerance.  Tests that exercise tolerance/window logic override
    # this explicitly afterwards.
    return cfg


def test_end_to_end_performance_run(fake_server, tmp_path):
    server, state = fake_server
    cfg = _fake_config(fake_server, tmp_path, repeats=2, samples=2)

    client = OMLXClient(base_url=cfg.server.base_url, api_key="fake-key")
    writer = RunWriter(tmp_path, run_id="integration")
    try:
        summary = Orchestrator(cfg, client, writer).run()
    finally:
        client.close()

    run_dir = Path(summary["run_dir"])
    for name in (
        "run.json",
        "MANIFEST.json",
        "records.jsonl",
        "samples.csv",
        "aggregates.json",
        "report.md",
        "logs/run.log",
    ):
        assert (run_dir / name).is_file(), f"missing {name}"
    # NOTE: APFS/HFS+ is case-insensitive, so "manifest.json" and "MANIFEST.json"
    # collide; only assert on names that cannot alias the new layout.
    for gone in ("summary.json", "summary.csv", "cells.csv", "analysis", "raw",
                 "hardware.json", "omlx.json", "models.json"):
        assert not (run_dir / gone).exists(), f"{gone} must no longer be written"
    # TG TPS coverage count lives in the aggregate distributions (F08).
    agg = _aggregates(run_dir)
    assert agg["overall"][0]["perf"]["gen_tps"]["n"] >= 0
    assert agg["cells"][0]["perf"]["gen_tps"]["n"] >= 0
    # The prompt bodies live once in the global blob pool.
    assert list((tmp_path / layout.BLOBS / layout.PROMPTS).glob("*/*.json"))
    index = json.loads((tmp_path / "index.json").read_text())
    entry = [
        item
        for item in index["runs"]
        if item["run_id"] == "integration__performance__model"
    ][0]
    assert entry["n_samples"] == 4
    assert entry["models_measured"] == ["fake-model"]
    assert entry["files"]["records.jsonl"]["sha256"]

    records = [json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines()]
    measured = [r for r in records if r["phase"] == "measured"]
    warmups = [r for r in records if r["phase"] == "warmup"]
    assert len(measured) == 4  # 2 samples x 2 repeats
    assert len(warmups) == 1

    # --- cache hygiene -------------------------------------------------
    assert state.clears >= 4
    assert state.cache_store_headers and all(h == "off" for h in state.cache_store_headers)
    assert state.chat_requests and all(
        req.get("chat_template_kwargs") == {"enable_thinking": False}
        for req in state.chat_requests
    )
    for record in measured:
        assert record["status"] == "ok", record.get("error")
        assert record["context_length"] == 1024  # declared 1K bucket
        assert record["request"]["context_meta"]["prebuilt"] is True
        assert record["request"]["context_meta"]["bucket"] == "1k"
        assert record["request"]["actual_prompt_tokens"] == 100
        # Decode guard: appended to every performance request so TG TPS has a
        # decode window even when the dataset prompt ends with "Answer:".
        assert record["request"]["context_meta"]["decode_guard"] is True
        # The prompt body lives in the content-addressed blob pool now, so the
        # guard is verified through the digest the record carries.
        prompt_sha = record["request"]["prompt_sha256"]
        assert prompt_sha, "record must reference its prompt blob"
        blob = json.loads(
            (tmp_path / layout.BLOBS / layout.PROMPTS / prompt_sha[:2] / f"{prompt_sha}.json").read_text()
        )
        assert blob["sha256"] == prompt_sha
        assert blob["chars"] == record["request"]["prompt_chars"]
        guard = blob["messages"][-1]
        assert guard["role"] == "user"
        assert guard["content"].startswith("This request is a performance measurement")
        assert blob["decode_guard_version"] == 1
        assert record["cache"]["cleared_before_request"] is True
        assert record["cache"]["cached_tokens"] == 0
        assert record["request"]["target_context_tokens"] is None
        assert record["metrics"]["input_context_tokens"] > 0
        assert record["metrics"]["input_context_chars"] > 0
        assert record["metrics"]["ttft_ms"] is not None
        assert record["metrics"]["e2e_ms"] is not None
        assert record["metrics"]["peak_memory_gb"] is not None
        assert record["metrics"]["peak_memory_gb"] > 0
        assert record["score"] is None  # performance-only: never scored

    # --- aggregates + run.json shape ------------------------------------
    summary_json = _aggregates(run_dir)
    run_meta = _run_json(run_dir)
    assert summary_json["schema_version"] == SCHEMA_VERSION
    assert run_meta["method"] == "performance"
    assert run_meta["config"]["perf"]["max_tokens"] == 50
    assert len(summary_json["overall"]) == 1
    overall = summary_json["overall"][0]
    assert overall["model"] == "fake-model"
    assert overall["context_length"] == 1024
    assert overall["n_valid"] == 4
    assert overall["perf"]["ttft_ms"]["mean"] is not None
    assert overall["perf"]["input_context_tokens"]["mean"] > 0
    assert "accuracy" not in overall
    assert len(summary_json["cells"]) == 1
    assert "accuracy" not in summary_json["cells"][0]
    sample_rows = _samples_csv(run_dir)
    assert len(sample_rows) == 4
    assert int(sample_rows[0]["input_context_tokens"]) > 0
    assert int(sample_rows[0]["context_length"]) == 1024
    assert sample_rows[0]["cache_cleared"] == "True"
    # The headline averages must be recomputable from records.jsonl alone.
    valid = [r for r in measured if r["status"] == "ok"]
    recomputed_ttft = round(sum(r["metrics"]["ttft_ms"] for r in valid) / len(valid), 3)
    recomputed_ctx = round(
        sum(r["metrics"]["input_context_tokens"] for r in valid) / len(valid), 3
    )
    assert overall["perf"]["ttft_ms"]["mean"] == recomputed_ttft
    assert overall["perf"]["input_context_tokens"]["mean"] == recomputed_ctx
    cell = summary_json["cells"][0]
    assert (cell["scenario"], cell["dataset"], cell["context_length"]) == (
        "knowledge_qa",
        "mmlu",
        1024,
    )

    # --- environment snapshot (single run.json, no side files) ----------
    env = run_meta["env"]
    # The server-wide /v1/models/status snapshot (memory ceiling, load timings,
    # full model inventory) used to be models.json; it must still be recorded.
    assert env.get("server_models"), "server model snapshot missing from run.json"
    assert "server" in env["server_models"]
    assert env["server_models"]["server"]
    omlx = env["omlx"]
    assert omlx["device"]["chip_name"] == "M4"
    assert omlx["feature_flags"] == []  # default config: no acceleration enabled
    assert omlx["default_config"] is True
    assert omlx["upload_projection"]["chip_name"] == "M4"
    assert omlx["upload_projection"]["models"]
    assert "auth" not in omlx["global_settings"]
    assert omlx["models"]["fake-model"]["quantization"] == "4bit"
    memory = omlx["models"]["fake-model"].get("memory") or {}
    assert "baseline_gb" in memory and "idle_gb" in memory
    assert "baseline_rss_mb" in memory and "idle_rss_mb" in memory
    report = (run_dir / "report.md").read_text()
    assert "进程内存" in report
    hardware = env["hardware"]
    assert hardware["upload_fields"]["chip_name"] == "M4"
    assert hardware["upload_fields"]["memory_gb"] == 24
    assert hardware["upload_fields"]["gpu_cores"] == 16

    # --- raw evidence: prompt bodies are content-addressed ---------------
    records = [
        json.loads(line)
        for line in (run_dir / "records.jsonl").read_text().splitlines()
    ]
    shas = {r["request"]["prompt_sha256"] for r in records}
    assert shas and all(shas)
    blob_root = tmp_path / layout.BLOBS / layout.PROMPTS
    stored = {path.stem for path in blob_root.glob("*/*.json")}
    assert shas <= stored, "some prompts are not resolvable from the blob pool"


def test_length_capped_run_keeps_samples_and_metrics(fake_server, tmp_path):
    """max_tokens cap: every sample stays in the aggregates with TG TPS measured."""
    server, state = fake_server
    cfg = _fake_config(fake_server, tmp_path, repeats=1, samples=2)
    cfg.perf.max_tokens = 50
    state.completion_tokens = 50
    state.finish_reason = "length"
    # A realistic token-by-token stream is required for TG TPS to be meaningful
    # (a single-burst response is suppressed by design, see test below).
    state.stream_chunk_delay_s = 0.04

    client = OMLXClient(base_url=cfg.server.base_url, api_key="fake-key")
    writer = RunWriter(tmp_path, run_id="capped")
    try:
        summary = Orchestrator(cfg, client, writer).run()
    finally:
        client.close()

    run_dir = Path(summary["run_dir"])
    records = [
        json.loads(line)
        for line in (run_dir / "records.jsonl").read_text().splitlines()
    ]
    measured = [r for r in records if r["phase"] == "measured"]
    assert len(measured) == 2
    for record in measured:
        assert record["status"] == "ok"
        assert record["response"]["finish_reason"] == "length"
        assert record["metrics"]["completion_tokens"] == 50
        # TG TPS is computed from the emitted window, not null / not dropped.
        assert record["metrics"]["gen_tps"] is not None
        assert record["metrics"]["tpot_ms"] is not None

    summary_json = _aggregates(run_dir)
    overall = summary_json["overall"][0]
    assert overall["n_valid"] == 2
    assert overall["n_length_capped"] == 2
    assert overall["perf"]["gen_tps"]["n"] == 2
    assert overall["perf"]["gen_tps"]["mean"] is not None
    samples = _samples_csv(run_dir)
    assert len(samples) == 2
    assert all(
        row["length_capped"] == "True" and row["finish_reason"] == "length"
        for row in samples
    )
    header = (run_dir / "samples.csv").read_text().splitlines()[0]
    assert "length_capped" in header and "finish_reason" in header
    report = (run_dir / "report.md").read_text()
    assert "触顶" in report
    assert "不是丢弃" in report


def test_agentic_search_loop_against_fake_server(_server_mode, tmp_path):
    """Legacy (non-prebuilt) samples still run the scenario tool loop."""
    server, state = _server_mode("agent")
    base_url = f"http://127.0.0.1:{server.server_address[1]}"

    cfg = load_config()
    cfg.run.warmup_per_model = 0
    cfg.server.base_url = base_url
    cfg.server.api_key = "fake-key"
    cfg.run.output_dir = str(tmp_path)
    cfg.run.warmup_requests = 0
    cfg.run.cache_settle_ms = 0
    cfg.run.repeats = 1
    cfg.selection.models = ["fake-model"]
    cfg.models = [ModelSpec(id="fake-model", alias="fake-model")]
    _apply_perf_fixture(
        cfg, tmp_path, scenarios=("search_agent",), per_scenario=1, prebuilt=False
    )
    cfg.scenarios["search_agent"].params["max_steps"] = 2

    client = OMLXClient(base_url=base_url, api_key="fake-key")
    writer = RunWriter(tmp_path, run_id="agent-integration")
    try:
        summary = Orchestrator(cfg, client, writer).run()
    finally:
        client.close()

    records = [
        json.loads(line)
        for line in (Path(summary["run_dir"]) / "records.jsonl").read_text().splitlines()
    ]
    measured = [r for r in records if r["phase"] == "measured"]
    assert len(measured) == 1
    record = measured[0]
    assert record["status"] == "ok", record.get("error")
    assert record["response"]["text"] == "Albert Einstein"
    trace = record["sample_meta"]["agent_trace"]
    assert len(trace) >= 2
    assert trace[0]["tool_calls"][0]["name"] == "search"
    assert record["score"] is None


def test_prebuilt_search_agent_skips_tool_loop(fake_server, tmp_path):
    """Pre-built agentic samples are sent exactly once (no tool loop)."""
    server, state = fake_server
    cfg = _fake_config(fake_server, tmp_path, samples=1, scenarios=("search_agent",))
    cfg.run.warmup_requests = 0
    client = OMLXClient(base_url=cfg.server.base_url, api_key="fake-key")
    writer = RunWriter(tmp_path, run_id="prebuilt-agent")
    try:
        summary = Orchestrator(cfg, client, writer).run()
    finally:
        client.close()
    records = [
        json.loads(line)
        for line in (Path(summary["run_dir"]) / "records.jsonl").read_text().splitlines()
    ]
    measured = [r for r in records if r["phase"] == "measured"]
    assert len(measured) == 1
    record = measured[0]
    assert record["status"] == "ok", record.get("error")
    assert record["scenario"] == "search_agent"
    assert record["request"]["context_meta"]["prebuilt"] is True
    assert "agent_trace" not in record["sample_meta"]
    # Exactly one chat request: the tool loop must not run.
    assert len(state.chat_requests) == 1


def test_fail_fast_stops_the_run(_server_mode, tmp_path):
    server, state = _server_mode("error")
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    cfg = load_config()
    cfg.run.warmup_per_model = 0
    cfg.server.base_url = base_url
    cfg.server.api_key = "fake-key"
    cfg.run.output_dir = str(tmp_path)
    cfg.run.warmup_requests = 0
    cfg.run.repeats = 1
    cfg.run.max_retries = 0
    cfg.run.cache_settle_ms = 0
    cfg.run.fail_fast = True
    cfg.selection.models = ["fake-model"]
    cfg.models = [ModelSpec(id="fake-model", alias="fake-model")]
    _apply_perf_fixture(cfg, tmp_path, scenarios=("knowledge_qa",), per_scenario=1)

    client = OMLXClient(base_url=base_url, api_key="fake-key")
    writer = RunWriter(cfg.run.output_dir, run_id="fail-fast")
    with pytest.raises(RuntimeError, match="fail_fast"):
        Orchestrator(cfg, client, writer).run()
    client.close()


def test_restore_state_unloads_model_loaded_by_run(_server_mode, tmp_path):
    server, state = _server_mode("cold-model")
    cfg = load_config()
    cfg.run.warmup_per_model = 0
    cfg.server.base_url = f"http://127.0.0.1:{server.server_address[1]}"
    cfg.server.api_key = "fake-key"
    cfg.run.output_dir = str(tmp_path)
    cfg.run.warmup_requests = 0
    cfg.run.repeats = 1
    cfg.run.cache_settle_ms = 0
    cfg.run.isolate_models = False
    cfg.run.restore_state = True
    cfg.selection.models = ["fake-model"]
    cfg.models = [ModelSpec(id="fake-model", alias="fake-model")]
    _apply_perf_fixture(cfg, tmp_path, scenarios=("knowledge_qa",), per_scenario=1)

    client = OMLXClient(base_url=cfg.server.base_url, api_key="fake-key")
    writer = RunWriter(cfg.run.output_dir, run_id="restore-state")
    try:
        Orchestrator(cfg, client, writer).run()
    finally:
        client.close()
    assert state.loads == ["fake-model"]
    assert state.unloads == ["fake-model"]


def _tiny_cfg(cfg, tmp_path, load=None):
    cfg.run.output_dir = str(tmp_path)
    cfg.run.warmup_requests = 0
    cfg.run.repeats = 1
    cfg.run.max_retries = 0
    cfg.run.cache_settle_ms = 0
    cfg.run.isolate_models = False
    cfg.selection.models = ["fake-model"]
    cfg.models = [ModelSpec(id="fake-model", alias="fake-model", load=load)]
    _apply_perf_fixture(cfg, tmp_path, scenarios=("knowledge_qa",), per_scenario=1)
    return cfg


def test_moe_load_setting_applied_before_load(_server_mode, tmp_path):
    server, state = _server_mode("cold-model")
    cfg = load_config()
    cfg.run.warmup_per_model = 0
    cfg.server.base_url = f"http://127.0.0.1:{server.server_address[1]}"
    cfg.server.api_key = "fake-key"
    load = ModelLoadConfig(settings={"moe_expert_offload_enabled": True}, optional=False)
    cfg = _tiny_cfg(cfg, tmp_path, load=load)

    client = OMLXClient(base_url=cfg.server.base_url, api_key="fake-key")
    writer = RunWriter(cfg.run.output_dir, run_id="settings")
    try:
        summary = Orchestrator(cfg, client, writer).run()
    finally:
        client.close()

    assert state.settings_puts == [{"moe_expert_offload_enabled": True}]
    assert state.events.index("settings:{'moe_expert_offload_enabled': True}") < state.events.index(
        "load:fake-model"
    )
    assert summary["n_cells"] == 1


def test_optional_model_load_setting_failure_is_warning(_server_mode, tmp_path):
    server, state = _server_mode("cold-model")
    state.settings_status = 400
    cfg = load_config()
    cfg.run.warmup_per_model = 0
    cfg.server.base_url = f"http://127.0.0.1:{server.server_address[1]}"
    cfg.server.api_key = "fake-key"
    load = ModelLoadConfig(settings={"moe_expert_offload_enabled": True}, optional=True)
    cfg = _tiny_cfg(cfg, tmp_path, load=load)

    client = OMLXClient(base_url=cfg.server.base_url, api_key="fake-key")
    writer = RunWriter(cfg.run.output_dir, run_id="settings-optional")
    try:
        summary = Orchestrator(cfg, client, writer).run()
    finally:
        client.close()

    assert state.settings_puts == [{"moe_expert_offload_enabled": True}]
    assert summary["warnings"] and "load settings failed" in summary["warnings"][0]
    assert summary["n_cells"] == 1


def test_native_prompt_is_truncated_to_fit_model_window(fake_server, tmp_path):
    """Legacy (non-prebuilt) samples still truncate head+tail to fit the window."""
    cfg = _fake_config(
        fake_server,
        tmp_path,
        repeats=1,
        samples=1,
        scenarios=("long_context",),
        prebuilt=False,
        question="long document " * 4000,
    )
    cfg.run.warmup_requests = 0

    client = OMLXClient(base_url=cfg.server.base_url, api_key="fake-key")
    writer = RunWriter(tmp_path, run_id="truncation")
    try:
        summary = Orchestrator(cfg, client, writer).run()
    finally:
        client.close()

    records = [
        json.loads(line)
        for line in (Path(summary["run_dir"]) / "records.jsonl").read_text().splitlines()
    ]
    measured = [r for r in records if r["phase"] == "measured"]
    assert len(measured) == 1
    record = measured[0]
    assert record["status"] == "ok", record.get("error")
    assert record["request"]["truncated"] is True
    assert record["request"]["context_meta"]["native_truncated"] is True
    # Legacy (non-prebuilt) fixture rows still carry the declared bucket.
    assert record["context_length"] == 1024
    assert record["metrics"]["input_context_tokens"] <= 4096
    assert record["metrics"]["input_context_chars"] > 0


def test_prebuilt_prompt_too_long_is_skipped_not_truncated(fake_server, tmp_path):
    """Pre-built samples must be skipped instead of silently truncated."""
    server, state = fake_server
    cfg = _fake_config(
        fake_server,
        tmp_path,
        repeats=1,
        samples=2,
        scenarios=("long_context",),
        tokens=9000,
        bucket="8k",
    )
    cfg.run.warmup_requests = 0
    # The fake tokenizer would count the short fixture as tiny; override so the
    # *actual* tokenized count exceeds the model window (the declared 9000 is
    # only the reference-tokenizer value and must not be the sole basis).
    state.tokenize_override = 5000
    client = OMLXClient(base_url=cfg.server.base_url, api_key="fake-key")
    writer = RunWriter(tmp_path, run_id="prebuilt-overflow")
    try:
        summary = Orchestrator(cfg, client, writer).run()
    finally:
        client.close()
    assert summary["n_samples"] == 0
    assert summary["n_records"] == 0
    assert any("prebuilt prompt too long" in item for item in summary["skipped"])
    # No chat request was sent for the overflowing samples.
    assert state.chat_requests == []


def test_cells_split_by_dataset_and_context_length(fake_server, tmp_path):
    """One cell per (scenario, dataset, context_length), matching the aggregates."""
    server, state = fake_server
    state.max_window = 65536
    rows = (
        _perf_rows(["knowledge_qa"], per_scenario=2, tokens=100, bucket="1k")
        + _perf_rows(["coding"], per_scenario=2, tokens=8200, bucket="8k")
        + _perf_rows(["coding"], per_scenario=2, tokens=32000, bucket="32k")
    )
    cfg = load_config()
    cfg.run.warmup_per_model = 0
    cfg.server.base_url = f"http://127.0.0.1:{server.server_address[1]}"
    cfg.server.api_key = "fake-key"
    cfg.run.output_dir = str(tmp_path)
    cfg.run.warmup_requests = 1
    cfg.run.warmup_per_model = 0  # legacy tests count per-cell warmups only
    cfg.run.cache_settle_ms = 0
    cfg.run.cpu_sample_interval_ms = 20
    cfg.run.repeats = 1
    cfg.selection.models = ["fake-model"]
    cfg.models = [ModelSpec(id="fake-model", alias="fake-model")]
    cfg.datasets.testset_dir = str(_write_perf_file(tmp_path, rows))
    cfg.selection.scenarios = ["knowledge_qa", "coding"]
    cfg.perf.samples_per_dataset = None
    events = []

    client = OMLXClient(base_url=cfg.server.base_url, api_key="fake-key")
    writer = RunWriter(tmp_path, run_id="cells")
    try:
        summary = Orchestrator(
            cfg,
            client,
            writer,
            progress_cb=lambda event, payload: events.append((event, payload)),
        ).run()
    finally:
        client.close()

    summary_json = _aggregates(summary["run_dir"])
    assert summary_json["n_samples"] == 6
    keys = {
        (cell["scenario"], cell["dataset"], cell["context_length"])
        for cell in summary_json["cells"]
    }
    assert keys == {
        ("knowledge_qa", "mmlu", 1024),
        ("coding", "longbench_v1:repobench-p", 8192),
        ("coding", "longbench_v1:repobench-p", 32768),
    }
    assert len(summary_json["cells"]) == 3
    # Progress cell_total matches the aggregates and never exceeds 100%.
    cell_starts = [payload for event, payload in events if event == "cell_start"]
    assert len(cell_starts) == 3
    assert [payload["cell_index"] for payload in cell_starts] == [1, 2, 3]
    assert all(payload["cell_total"] == 3 for payload in cell_starts)
    # Warmup runs once per cell (3 cells), all discarded from aggregation.
    records = [
        json.loads(line)
        for line in (Path(summary["run_dir"]) / "records.jsonl").read_text().splitlines()
    ]
    warmups = [r for r in records if r["phase"] == "warmup"]
    assert len(warmups) == 3
    measured = [r for r in records if r["phase"] == "measured"]
    assert {r["context_length"] for r in measured} == {1024, 8192, 32768}


def test_persisted_acceleration_settings_are_reset_to_default(_server_mode, tmp_path):
    server, state = _server_mode("cold-model")
    state.model_settings = {"turboquant_kv_enabled": True, "dflash_enabled": True}
    cfg = load_config()
    cfg.run.warmup_per_model = 0
    cfg.server.base_url = f"http://127.0.0.1:{server.server_address[1]}"
    cfg.server.api_key = "fake-key"
    cfg = _tiny_cfg(cfg, tmp_path, load=ModelLoadConfig(settings={}, optional=True))

    client = OMLXClient(base_url=cfg.server.base_url, api_key="fake-key")
    writer = RunWriter(cfg.run.output_dir, run_id="reset-settings")
    try:
        Orchestrator(cfg, client, writer).run()
    finally:
        client.close()

    assert state.settings_puts == [
        {"dflash_enabled": False, "turboquant_kv_enabled": False}
    ]
    assert state.events.index(
        "settings:{'dflash_enabled': False, 'turboquant_kv_enabled': False}"
    ) < state.events.index("load:fake-model")
    run_dir = _find_run_dir(Path(cfg.run.output_dir), "reset-settings")
    omlx = _run_json(run_dir)["env"]["omlx"]
    assert omlx["feature_flags"] == []
    assert omlx["default_config"] is True


def test_warm_mode_runs_without_cache_clear_errors(fake_server, tmp_path):
    server, state = fake_server
    cfg = _fake_config(fake_server, tmp_path, repeats=1, samples=2)
    cfg.run.cache_mode = CacheMode.WARM
    cfg.run.warmup_requests = 0
    client = OMLXClient(base_url=cfg.server.base_url, api_key="fake-key")
    writer = RunWriter(tmp_path, run_id="warm-mode")
    try:
        summary = Orchestrator(cfg, client, writer).run()
    finally:
        client.close()
    assert summary["n_samples"] == 2
    assert summary["warnings"] == []
    records = [
        json.loads(line)
        for line in (Path(summary["run_dir"]) / "records.jsonl").read_text().splitlines()
    ]
    measured = [r for r in records if r["phase"] == "measured"]
    assert len(measured) == 2
    assert all(r["status"] == "ok" for r in measured), [r.get("error") for r in measured]
    # Warm mode must actually allow the prefix cache to be populated (warmup 0):
    # no X-oMLX-Cache-Store: off, and only the first request of the cell clears.
    assert state.cache_store_headers and all(h == "" for h in state.cache_store_headers)
    # One clear at the start of the cell (hot+SSD = 2 calls) + one final cleanup.
    assert state.clears == 4
    assert measured[0]["cache"]["cleared_before_request"] is True
    assert measured[1]["cache"]["cleared_before_request"] is False


def test_cold_mode_keeps_cache_store_off(fake_server, tmp_path):
    cfg = _fake_config(fake_server, tmp_path, repeats=1, samples=1)
    cfg.run.warmup_requests = 0
    client = OMLXClient(base_url=cfg.server.base_url, api_key="fake-key")
    writer = RunWriter(tmp_path, run_id="cold-store")
    try:
        Orchestrator(cfg, client, writer).run()
    finally:
        client.close()
    state = fake_server[1]
    assert state.cache_store_headers == ["off"]


def test_max_requests_cap_is_reported_in_warnings(fake_server, tmp_path):
    cfg = _fake_config(fake_server, tmp_path, repeats=1, samples=3)
    cfg.run.max_requests = 1
    client = OMLXClient(base_url=cfg.server.base_url, api_key="fake-key")
    writer = RunWriter(tmp_path, run_id="cap")
    try:
        summary = Orchestrator(cfg, client, writer).run()
    finally:
        client.close()
    assert summary["n_samples"] == 1
    assert any("max_requests cap" in warning for warning in summary["warnings"])
    cap_skips = [item for item in summary["skipped"] if "max_requests cap" in item]
    assert len(cap_skips) == 2  # the two selected samples that were never measured
    assert all("not measured" in item for item in cap_skips)


def test_cleanup_failure_is_reported(fake_server, tmp_path):
    cfg = _fake_config(fake_server, tmp_path, repeats=1, samples=1)
    cfg.run.warmup_requests = 0
    client = OMLXClient(base_url=cfg.server.base_url, api_key="fake-key")
    writer = RunWriter(tmp_path, run_id="cleanup-fail")
    orchestrator = Orchestrator(cfg, client, writer)
    orchestrator.controller.cleanup = lambda: {"error": "admin endpoint down"}
    try:
        summary = orchestrator.run()
    finally:
        client.close()
    assert any("cache cleanup failed" in warning for warning in summary["warnings"])
    log = (Path(summary["run_dir"]) / "logs" / "run.log").read_text()
    assert "cache cleanup failed" in log


def test_cache_clear_failure_aborts_cell(fake_server, tmp_path):
    server, state = fake_server
    state.clear_status = 500
    cfg = _fake_config(fake_server, tmp_path, repeats=1, samples=2)
    cfg.run.warmup_requests = 0
    client = OMLXClient(base_url=cfg.server.base_url, api_key="fake-key")
    writer = RunWriter(tmp_path, run_id="clear-fail")
    try:
        summary = Orchestrator(cfg, client, writer).run()
    finally:
        client.close()
    records = [
        json.loads(line)
        for line in (Path(summary["run_dir"]) / "records.jsonl").read_text().splitlines()
    ]
    measured = [r for r in records if r["phase"] == "measured"]
    assert len(measured) == 1  # first request errors, then the cell is aborted
    assert measured[0]["status"] == "error"
    assert "cache clear failed" in (measured[0]["error"] or "")
    assert any("cell aborted" in warning for warning in summary["warnings"])
    # Aborted cells must not silently drop selected samples (AGENTS invariant 13).
    abort_skips = [item for item in summary["skipped"] if "cell aborted" in item]
    assert len(abort_skips) == 1
    assert "not measured" in abort_skips[0]
    assert "r0" in abort_skips[0]


def test_aborted_cell_skips_pending_repeats(fake_server, tmp_path):
    server, state = fake_server
    state.clear_status = 500
    cfg = _fake_config(fake_server, tmp_path, repeats=2, samples=2)
    cfg.run.warmup_requests = 0
    client = OMLXClient(base_url=cfg.server.base_url, api_key="fake-key")
    writer = RunWriter(tmp_path, run_id="clear-fail-repeats")
    try:
        summary = Orchestrator(cfg, client, writer).run()
    finally:
        client.close()
    # sample#0/r0 is the error record; sample#0/r1 and sample#1/r0..r1 are skipped.
    abort_skips = [item for item in summary["skipped"] if "cell aborted" in item]
    assert len(abort_skips) == 3, abort_skips
    assert any("/r1:" in item for item in abort_skips)
    joined = "\n".join(abort_skips)
    assert "fixture-knowledge_qa-1k-000" in joined
    assert "fixture-knowledge_qa-1k-001" in joined



def test_per_scenario_sample_count_selection(fake_server, tmp_path):
    cfg = _fake_config(fake_server, tmp_path, repeats=1, samples=4)
    cfg.perf.samples_per_dataset = None
    cfg.perf.samples_by_scenario = {"knowledge_qa": 1}
    client = OMLXClient(base_url=cfg.server.base_url, api_key="fake-key")
    writer = RunWriter(tmp_path, run_id="scenario-count")
    try:
        summary = Orchestrator(cfg, client, writer).run()
    finally:
        client.close()
    records = [
        json.loads(line)
        for line in (Path(summary["run_dir"]) / "records.jsonl").read_text().splitlines()
    ]
    measured = [r for r in records if r["phase"] == "measured"]
    assert len(measured) == 1
    assert summary["n_samples"] == 1


def test_explicit_sample_id_selection(fake_server, tmp_path):
    cfg = _fake_config(fake_server, tmp_path, repeats=1, samples=4)
    # Fixture sample ids are deterministic (sorted before slicing).
    ids = sorted(f"fixture-knowledge_qa-1k-{index:03d}" for index in range(4))
    chosen = [ids[3], ids[1]]
    cfg.perf.sample_ids = {"knowledge_qa": chosen}
    client = OMLXClient(base_url=cfg.server.base_url, api_key="fake-key")
    writer = RunWriter(tmp_path, run_id="explicit-ids")
    try:
        summary = Orchestrator(cfg, client, writer).run()
    finally:
        client.close()
    records = [
        json.loads(line)
        for line in (Path(summary["run_dir"]) / "records.jsonl").read_text().splitlines()
    ]
    measured = [r for r in records if r["phase"] == "measured"]
    assert [r["sample_id"] for r in measured] == chosen
    assert summary["n_samples"] == 2


def test_missing_explicit_sample_ids_are_recorded_as_skipped(fake_server, tmp_path):
    cfg = _fake_config(fake_server, tmp_path, repeats=1, samples=2)
    cfg.perf.sample_ids = {"knowledge_qa": ["does-not-exist"]}
    client = OMLXClient(base_url=cfg.server.base_url, api_key="fake-key")
    writer = RunWriter(tmp_path, run_id="missing-ids")
    try:
        summary = Orchestrator(cfg, client, writer).run()
    finally:
        client.close()
    assert summary["n_samples"] == 0
    assert any("none of the 1 explicitly selected" in item for item in summary["skipped"])


def test_progress_events_report_models_and_cells(fake_server, tmp_path):
    cfg = _fake_config(
        fake_server,
        tmp_path,
        repeats=1,
        samples=2,
        scenarios=("knowledge_qa", "long_context"),
    )
    events = []
    client = OMLXClient(base_url=cfg.server.base_url, api_key="fake-key")
    writer = RunWriter(tmp_path, run_id="progress-events")
    try:
        Orchestrator(
            cfg, client, writer, progress_cb=lambda event, payload: events.append((event, payload))
        ).run()
    finally:
        client.close()
    names = [event for event, _ in events]
    assert names.count("model_start") == 1
    assert names.count("model_done") == 1
    assert names.count("cell_start") == 2
    assert names.count("cell_done") == 2
    cell_starts = [payload for event, payload in events if event == "cell_start"]
    assert [payload["cell_index"] for payload in cell_starts] == [1, 2]
    assert all(payload["cell_total"] == 2 for payload in cell_starts)
    assert {payload["scenario"] for payload in cell_starts} == {
        "knowledge_qa",
        "long_context",
    }
    assert cell_starts[0]["samples"] == 2


def test_tool_schema_rejection_is_reported_without_retry(_server_mode, tmp_path):
    server, state = _server_mode("bad-tools")
    cfg = load_config()
    cfg.run.warmup_per_model = 0
    cfg.server.base_url = f"http://127.0.0.1:{server.server_address[1]}"
    cfg.server.api_key = "fake-key"
    cfg.run.output_dir = str(tmp_path)
    cfg.run.warmup_requests = 0
    cfg.run.max_retries = 2
    cfg.run.cache_settle_ms = 0
    cfg.selection.models = ["fake-model"]
    cfg.models = [ModelSpec(id="fake-model", alias="fake-model")]
    # A row with a tool schema so the runner sends `tools`.
    rows = _perf_rows(["general_agent"], per_scenario=1)
    rows[0]["tools"] = [
        {"type": "function", "function": {"name": "f", "parameters": {"type": "dict"}}}
    ]
    cfg.datasets.testset_dir = str(_write_perf_file(tmp_path, rows))
    cfg.selection.scenarios = ["general_agent"]
    cfg.perf.samples_per_dataset = 1
    client = OMLXClient(base_url=cfg.server.base_url, api_key="fake-key")
    writer = RunWriter(cfg.run.output_dir, run_id="bad-tools")
    try:
        summary = Orchestrator(cfg, client, writer).run()
    finally:
        client.close()
    assert len(state.chat_requests) == 1  # deterministic rejection: no retries
    assert any("tools schema" in warning for warning in summary["warnings"])
    records = [
        json.loads(line)
        for line in (Path(summary["run_dir"]) / "records.jsonl").read_text().splitlines()
    ]
    measured = [r for r in records if r["phase"] == "measured"]
    assert measured and measured[0]["status"] == "error"
    assert "invalid tools" in (measured[0]["error"] or "")


def test_memory_guard_errors_are_not_retried_and_warn(_server_mode, tmp_path):
    server, state = _server_mode("memory")
    cfg = load_config()
    cfg.run.warmup_per_model = 0
    cfg.server.base_url = f"http://127.0.0.1:{server.server_address[1]}"
    cfg.server.api_key = "fake-key"
    cfg.run.output_dir = str(tmp_path)
    cfg.run.warmup_requests = 0
    cfg.run.max_retries = 2
    cfg.run.cache_settle_ms = 0
    cfg.selection.models = ["fake-model"]
    cfg.models = [ModelSpec(id="fake-model", alias="fake-model")]
    _apply_perf_fixture(cfg, tmp_path, scenarios=("knowledge_qa",), per_scenario=2)

    client = OMLXClient(base_url=cfg.server.base_url, api_key="fake-key")
    writer = RunWriter(cfg.run.output_dir, run_id="memory-guard")
    try:
        summary = Orchestrator(cfg, client, writer).run()
    finally:
        client.close()
    assert state.memory_attempts == 2  # one attempt per sample, no retries
    assert summary["memory_errors"] == 2
    assert any("memory guard" in warning for warning in summary["warnings"])
    records = [
        json.loads(line)
        for line in (Path(summary["run_dir"]) / "records.jsonl").read_text().splitlines()
    ]
    measured = [r for r in records if r["phase"] == "measured"]
    assert len(measured) == 2
    assert all(r["status"] == "error" for r in measured)


def test_batch_runner_writes_one_bundle_per_model(fake_server, tmp_path):
    cfg = _fake_config(fake_server, tmp_path, repeats=1, samples=2)
    cfg.selection.models = ["fake-model"]
    client = OMLXClient(base_url=cfg.server.base_url, api_key="fake-key")
    try:
        batch = BatchRunner(
            cfg, client, tmp_path, batch_id="itest-batch"
        ).run()
    finally:
        client.close()
    assert batch["batch_id"] == "itest-batch"
    assert batch["method"] == "performance"
    assert batch["n_models_measured"] == 1
    run = batch["model_runs"][0]
    assert run["run_id"] == "itest-batch__performance__fake-model"
    # Flat layout: run_path is relative to the output root.
    assert run["run_path"] == "runs/itest-batch__performance__fake-model"
    run_dir = tmp_path / run["run_path"]
    assert run_dir.is_dir()
    assert run_dir == layout.run_dir(tmp_path, run["run_id"])
    assert (_run_json(run_dir))["batch_id"] == "itest-batch"
    assert (_aggregates(run_dir))["n_samples"] == 2
    batch_path = layout.batch_path(tmp_path, "itest-batch")
    assert batch_path.is_file()
    batch_payload = json.loads(batch_path.read_text())
    # A batch directory must stay movable: no absolute paths in its metadata.
    assert str(tmp_path) not in json.dumps(batch_payload)
    assert batch_payload["model_runs"][0]["run_path"].startswith("runs/")
    assert "run_dir" not in batch_payload["model_runs"][0]
    # batch metadata is a sidecar file in batches/, not a per-batch directory tree
    assert sorted(p.name for p in (tmp_path / "batches").iterdir()) == ["itest-batch.json"]
    assert not (tmp_path / "result_analysis").exists()
    index = json.loads((tmp_path / "index.json").read_text())
    assert index["n_batches"] == 1
    assert index["batches"][0]["batch_id"] == "itest-batch"
    assert [item["run_id"] for item in index["batches"][0]["models"]] == [
        "itest-batch__performance__fake-model"
    ]


def test_batch_excel_analysis_has_three_sheets(fake_server, tmp_path):
    from openpyxl import load_workbook

    cfg = _fake_config(fake_server, tmp_path, repeats=1, samples=2)
    cfg.selection.models = ["fake-model"]
    client = OMLXClient(base_url=cfg.server.base_url, api_key="fake-key")
    try:
        batch = BatchRunner(cfg, client, tmp_path, batch_id="excel-batch").run()
    finally:
        client.close()
    # T-068: the Excel deliverable is generated automatically when the batch
    # ends; the returned summary points at the already-written workbook.
    assert "result_analysis" not in batch
    workbook_path = Path(batch["export_file"])
    assert workbook_path.is_file()
    assert workbook_path.parent == layout.exports_dir(tmp_path)
    assert workbook_path.name == "excel-batch.xlsx"
    assert not (tmp_path / "result_analysis").exists()
    # The movable batch.json records the export with a relative name only.
    batch_json = json.loads(
        layout.batch_path(tmp_path, "excel-batch").read_text(encoding="utf-8")
    )
    assert batch_json["exports"] == {"xlsx": "exports/excel-batch.xlsx"}
    wb = load_workbook(workbook_path)
    assert wb.sheetnames == ["硬件配置", "模型性能平均", "详细数据"]
    detail = wb["详细数据"]
    headers = [cell.value for cell in detail[1]]
    assert headers[:8] == [
        "测试模型", "模型ID", "样本来源", "样本所属场景", "样本ID", "重复",
        "API 输入内容", "输入 prompt token 大小",
    ]
    assert detail.max_row == 3  # header + two measured samples
    row = [cell.value for cell in detail[2]]
    assert row[0] == "fake-model"
    assert row[2] == "mmlu"          # sample source (dataset)
    assert row[3] == "knowledge_qa"  # sample scenario
    assert row[6]                    # API input content
    assert row[7] > 0                # prompt tokens
    assert row[9] is not None        # output token size
    averages = wb["模型性能平均"]
    # header + one data row + the one-line ISL calibre note (T-064)
    assert averages.max_row == 3
    assert [cell.value for cell in averages[1]][:4] == [
        "测试模型", "模型ID", "模型大小(GB)", "上下文档位",
    ]
    assert averages.cell(row=2, column=1).value == "fake-model"
    assert averages.cell(row=2, column=4).value == "1K"      # per-context row
    assert averages.cell(row=2, column=10).value is not None  # n_length_capped
    assert averages.cell(row=2, column=11).value is not None  # ttft mean
    assert averages.cell(row=2, column=15).value is not None  # gen_tps n


def test_stratified_sampling_covers_buckets_end_to_end(fake_server, tmp_path):
    server, state = fake_server
    state.max_window = 65536
    rows = _perf_rows(
        ["long_context"], per_scenario=4, tokens=8200, bucket="8k", dataset="longbench_v1:gov_report"
    ) + _perf_rows(
        ["long_context"], per_scenario=4, tokens=32000, bucket="32k", dataset="longbench_v1:gov_report"
    )
    cfg = load_config()
    cfg.run.warmup_per_model = 0
    cfg.server.base_url = f"http://127.0.0.1:{server.server_address[1]}"
    cfg.server.api_key = "fake-key"
    cfg.run.output_dir = str(tmp_path)
    cfg.run.warmup_requests = 0
    cfg.run.cache_settle_ms = 0
    cfg.run.cpu_sample_interval_ms = 20
    cfg.selection.models = ["fake-model"]
    cfg.models = [ModelSpec(id="fake-model", alias="fake-model")]
    cfg.datasets.testset_dir = str(_write_perf_file(tmp_path, rows))
    cfg.selection.scenarios = ["long_context"]
    cfg.perf.samples_per_dataset = 2

    client = OMLXClient(base_url=cfg.server.base_url, api_key="fake-key")
    writer = RunWriter(tmp_path, run_id="stratified")
    try:
        summary = Orchestrator(cfg, client, writer).run()
    finally:
        client.close()
    records = [
        json.loads(line)
        for line in (Path(summary["run_dir"]) / "records.jsonl").read_text().splitlines()
    ]
    measured = [r for r in records if r["phase"] == "measured"]
    assert {r["context_length"] for r in measured} == {8192, 32768}
    assert summary["n_cells"] == 2


def test_context_window_filter_end_to_end(fake_server, tmp_path):
    """Selected windows only filter samples; aggregation stays one overall average."""
    server, state = fake_server
    cfg = _fake_config(fake_server, tmp_path, repeats=1, samples=2)
    cfg.perf.context_windows = [1024]  # 1K is special: 0..1.5K tokens

    client = OMLXClient(base_url=cfg.server.base_url, api_key="fake-key")
    writer = RunWriter(tmp_path, run_id="ctx-1k")
    try:
        summary = Orchestrator(cfg, client, writer).run()
    finally:
        client.close()

    run_dir = Path(summary["run_dir"])
    summary_json = _aggregates(run_dir)
    assert _run_json(run_dir)["config"]["context_windows"] == [1024]
    assert summary_json["n_samples"] == 2
    assert summary_json["cells"][0]["context_length"] == 1024
    records = [json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines()]
    measured = [r for r in records if r["phase"] == "measured"]
    assert len(measured) == 2
    for record in measured:
        tokens = record["metrics"]["input_context_tokens"]
        assert 0 <= tokens <= 1536
        assert record["context_length"] == 1024  # declared 1K bucket cell
    assert len(summary_json["cells"]) == 1
    report = (run_dir / "report.md").read_text()
    assert "上下文筛选" in report and "1K:0–1.5K" in report


def test_context_window_filter_without_matches_skips_scenario(fake_server, tmp_path):
    server, state = fake_server
    cfg = _fake_config(fake_server, tmp_path, repeats=1, samples=2)
    cfg.perf.context_windows = [8192]  # fake MMLU prompts are ~100 tokens

    client = OMLXClient(base_url=cfg.server.base_url, api_key="fake-key")
    writer = RunWriter(tmp_path, run_id="ctx-empty")
    try:
        summary = Orchestrator(cfg, client, writer).run()
    finally:
        client.close()

    assert summary["n_samples"] == 0
    assert summary["n_records"] == 0
    assert any("no samples inside selected context windows" in item for item in summary["skipped"])
    records_path = Path(summary["run_dir"]) / "records.jsonl"
    assert not records_path.exists() or records_path.read_text().strip() == ""


def test_model_warmup_is_short_and_not_recorded(fake_server, tmp_path):
    """Default oMLX-style warmup: one short request after model load, discarded."""
    server, state = fake_server
    cfg = _fake_config(fake_server, tmp_path, repeats=1, samples=2)
    cfg.run.warmup_requests = 0
    cfg.run.warmup_per_model = 1

    client = OMLXClient(base_url=cfg.server.base_url, api_key="fake-key")
    writer = RunWriter(tmp_path, run_id="model-warmup")
    try:
        summary = Orchestrator(cfg, client, writer).run()
    finally:
        client.close()

    records = [json.loads(line) for line in (Path(summary["run_dir"]) / "records.jsonl").read_text().splitlines()]
    assert [r["phase"] for r in records] == ["measured", "measured"]
    assert not [r for r in records if r["phase"] == "warmup"]

    # One warmup request (max_tokens=8) precedes the two measured requests.
    assert len(state.chat_requests) == 3
    warmup = state.chat_requests[0]
    assert warmup["max_tokens"] == 8
    assert "warmup" in warmup["messages"][-1]["content"].lower()
    assert all(req["max_tokens"] == cfg.perf.max_tokens for req in state.chat_requests[1:])
    run_meta = _run_json(summary["run_dir"])
    assert run_meta["config"]["run"]["warmup_per_model"] == 1
    # Cold-cache hygiene still applies to the warmup request.
    assert state.cache_store_headers and all(h == "off" for h in state.cache_store_headers)


def test_warmup_requests_are_configurable_and_discarded(fake_server, tmp_path):
    server, state = fake_server
    cfg = _fake_config(fake_server, tmp_path, repeats=1, samples=1)
    cfg.run.warmup_requests = 2

    client = OMLXClient(base_url=cfg.server.base_url, api_key="fake-key")
    writer = RunWriter(tmp_path, run_id="warmup-2")
    try:
        summary = Orchestrator(cfg, client, writer).run()
    finally:
        client.close()

    records = [json.loads(line) for line in (Path(summary["run_dir"]) / "records.jsonl").read_text().splitlines()]
    warmups = [r for r in records if r["phase"] == "warmup"]
    measured = [r for r in records if r["phase"] == "measured"]
    assert len(warmups) == 2
    assert len(measured) == 1
    run_meta = _run_json(summary["run_dir"])
    assert run_meta["config"]["run"]["warmup_requests"] == 2


def test_contamination_retry_has_independent_budget(fake_server, tmp_path):
    """F06: contamination retry must still run after transport retries are spent."""
    server, state = fake_server
    cfg = _fake_config(fake_server, tmp_path, samples=1)
    cfg.run.warmup_requests = 0
    cfg.run.warmup_per_model = 0
    cfg.run.max_retries = 2
    cfg.run.retry_backoff_s = 0
    state.fail_chat_attempts = 2  # two transient HTTP 500s
    state.cached_tokens_sequence = [50, 0]  # 3rd attempt polluted, 4th clean
    client = OMLXClient(base_url=cfg.server.base_url, api_key="fake-key")
    writer = RunWriter(tmp_path, run_id="contam-retry")
    try:
        summary = Orchestrator(cfg, client, writer).run()
    finally:
        client.close()

    run_dir = Path(summary["run_dir"])
    records = [json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines()]
    measured = [r for r in records if r["phase"] == "measured"]
    assert len(measured) == 1
    assert measured[0]["status"] == "ok"
    # 2 transport failures + 2 successful attempts (one contaminated, one clean).
    assert state.chat_attempts == 4
    log = (run_dir / "logs" / "run.log").read_text(encoding="utf-8")
    assert "retrying with fresh cache" in log
    assert "1/1" in log


def test_persistent_contamination_adds_summary_warning(fake_server, tmp_path):
    """F36: fail-closed contamination must surface an actionable warning."""
    server, state = fake_server
    cfg = _fake_config(fake_server, tmp_path, samples=1)
    cfg.run.warmup_requests = 0
    cfg.run.warmup_per_model = 0
    cfg.run.max_contamination_retries = 1
    cfg.run.retry_backoff_s = 0
    state.cached_tokens_sequence = [50, 50]  # always polluted
    client = OMLXClient(base_url=cfg.server.base_url, api_key="fake-key")
    writer = RunWriter(tmp_path, run_id="contam-warning")
    try:
        summary = Orchestrator(cfg, client, writer).run()
    finally:
        client.close()

    run_dir = Path(summary["run_dir"])
    records = [json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines()]
    measured = [r for r in records if r["phase"] == "measured"]
    assert len(measured) == 1
    assert measured[0]["status"] == "contaminated"
    assert any("cold-cache contamination" in item for item in summary["warnings"])
    assert state.chat_attempts == 2  # one attempt + one contamination retry


def test_cancel_records_unmeasured_samples_as_skipped(fake_server, tmp_path):
    """F07: cancel mid-cell must leave one skipped entry per unmeasured sample."""
    server, state = fake_server
    cfg = _fake_config(fake_server, tmp_path, samples=4, repeats=1)
    cfg.run.warmup_requests = 0
    cfg.run.warmup_per_model = 0
    client = OMLXClient(base_url=cfg.server.base_url, api_key="fake-key")
    writer = RunWriter(tmp_path, run_id="cancel-skip")
    cancel = threading.Event()

    def _cancel_after_first_measured(event, payload):
        if (
            event == "record"
            and payload.get("status") == "ok"
            and int(payload.get("repeat", 0)) >= 0
        ):
            cancel.set()

    try:
        summary = Orchestrator(
            cfg,
            client,
            writer,
            progress_cb=_cancel_after_first_measured,
            cancel_event=cancel,
        ).run()
    finally:
        client.close()

    assert summary["cancelled"] is True
    assert summary["n_samples"] == 1
    skipped = summary["skipped"]
    assert sum(
        1
        for item in skipped
        if "not measured (cell aborted: cancelled by user)" in item
    ) == 3


def test_cancel_between_cells_records_remaining_cell_samples(fake_server, tmp_path):
    """F07-EDGE: a cancel after one cell must record later cells' samples."""
    server, state = fake_server
    cfg = _fake_config(
        fake_server,
        tmp_path,
        samples=1,
        repeats=1,
        scenarios=("knowledge_qa", "instruction_following"),
    )
    cfg.run.warmup_requests = 0
    cfg.run.warmup_per_model = 0
    client = OMLXClient(base_url=cfg.server.base_url, api_key="fake-key")
    writer = RunWriter(tmp_path, run_id="cancel-between-cells")
    cancel = threading.Event()

    def _cancel_after_first_cell(event, payload):
        if event == "cell_done" and not cancel.is_set():
            cancel.set()

    try:
        summary = Orchestrator(
            cfg,
            client,
            writer,
            progress_cb=_cancel_after_first_cell,
            cancel_event=cancel,
        ).run()
    finally:
        client.close()

    assert summary["cancelled"] is True
    skipped = summary["skipped"]
    assert any(
        "instruction_following" in item and "fixture-instruction_following" in item
        for item in skipped
    ), skipped


def test_cancel_before_preparation_records_explicit_model_skip(fake_server, tmp_path):
    """F07-EDGE: cancel before any sample is loaded must still be auditable."""
    server, state = fake_server
    cfg = _fake_config(fake_server, tmp_path, samples=1)
    cfg.run.warmup_requests = 0
    cfg.run.warmup_per_model = 0
    client = OMLXClient(base_url=cfg.server.base_url, api_key="fake-key")
    writer = RunWriter(tmp_path, run_id="cancel-before-prep")
    cancel = threading.Event()
    cancel.set()
    try:
        summary = Orchestrator(cfg, client, writer, cancel_event=cancel).run()
    finally:
        client.close()

    assert summary["cancelled"] is True
    assert any(
        "fake-model" in item and "not measured (cancelled by user)" in item
        for item in summary["skipped"]
    ), summary["skipped"]


def test_tool_call_burst_suppresses_gen_tps(fake_server, tmp_path):
    """A >=16-token tool-call delivered in one burst must not yield 100k+ TPS."""
    server, state = fake_server
    cfg = _fake_config(fake_server, tmp_path, repeats=1, samples=2)
    state.completion_tokens = 50
    state.finish_reason = "tool_calls"
    state.stream_chunk_delay_s = 0.0  # single burst -> sub-ms decode window
    state.tokenize_override = 1000  # inside 1K tolerance [0, 1126]

    client = OMLXClient(base_url=cfg.server.base_url, api_key="fake-key")
    writer = RunWriter(tmp_path, run_id="tool-burst")
    try:
        summary = Orchestrator(cfg, client, writer).run()
    finally:
        client.close()

    run_dir = Path(summary["run_dir"])
    records = [
        json.loads(line)
        for line in (run_dir / "records.jsonl").read_text().splitlines()
    ]
    measured = [r for r in records if r["phase"] == "measured"]
    assert len(measured) == 2
    for record in measured:
        metrics = record["metrics"]
        assert metrics["completion_tokens"] == 50
        assert metrics["gen_tps"] is None
        assert metrics["tpot_ms"] is None
        assert metrics["gen_tps_invalid"] == "tool_calls"
        assert metrics["decode_window_ms"] is not None

    outcome = _run_json(run_dir)["outcome"]
    assert outcome["n_gen_tps_invalid"] == 2
    agg = _aggregates(run_dir)
    assert agg["overall"][0]["n_gen_tps_invalid"] == 2
    assert agg["cells"][0]["n_gen_tps_invalid"] == 2


def test_actual_isl_outside_band_is_measured_but_excluded(fake_server, tmp_path):
    """Out-of-band samples are still measured; aggregates only use in-band rows."""
    server, state = fake_server
    cfg = _fake_config(
        fake_server, tmp_path, repeats=1, samples=2, tokens=8192, bucket="8k"
    )
    state.max_window = 16384
    state.tokenize_override = 5000  # outside 8K tolerance [7372, 9011]
    state.stream_chunk_delay_s = 0.04

    client = OMLXClient(base_url=cfg.server.base_url, api_key="fake-key")
    writer = RunWriter(tmp_path, run_id="isl-tolerance")
    try:
        summary = Orchestrator(cfg, client, writer).run()
    finally:
        client.close()

    assert summary["n_samples"] == 2  # measured, never dropped
    run_dir = Path(summary["run_dir"])
    records = [
        json.loads(line)
        for line in (run_dir / "records.jsonl").read_text().splitlines()
        if json.loads(line)["phase"] == "measured"
    ]
    assert len(records) == 2
    for record in records:
        assert record["metrics"]["input_context_tokens"] is not None
        assert record["request"]["context_meta"]["isl_tolerance"]["ok"] is False
    outcome = _run_json(run_dir)["outcome"]
    assert outcome["n_measured"] == 2
    assert outcome["n_isl_in_band"] == 0
    assert outcome["n_isl_out_of_band"] == 2
    agg = _aggregates(run_dir)
    cell = agg["cells"][0]
    assert cell["n_isl_in_band"] == 0
    assert cell["n_isl_out_of_band"] == 2
    assert cell["perf"]["ttft_ms"]["n"] == 0
    samples = _samples_csv(run_dir)
    assert all(row["isl_band"] == "False" for row in samples)


def test_actual_isl_inside_tolerance_not_flagged(fake_server, tmp_path):
    server, state = fake_server
    cfg = _fake_config(
        fake_server, tmp_path, repeats=1, samples=2, tokens=8192, bucket="8k"
    )
    state.max_window = 16384
    state.tokenize_override = 8000  # inside [7372, 9011]
    state.stream_chunk_delay_s = 0.04

    client = OMLXClient(base_url=cfg.server.base_url, api_key="fake-key")
    writer = RunWriter(tmp_path, run_id="isl-ok")
    try:
        summary = Orchestrator(cfg, client, writer).run()
    finally:
        client.close()

    assert summary["n_samples"] == 2
    run_dir = Path(summary["run_dir"])
    records = [
        json.loads(line)
        for line in (run_dir / "records.jsonl").read_text().splitlines()
        if json.loads(line)["phase"] == "measured"
    ]
    assert all(
        r["request"]["context_meta"]["isl_tolerance"]["ok"] is True for r in records
    )
    outcome = _run_json(run_dir)["outcome"]
    assert outcome["n_isl_in_band"] == 2
    assert outcome["n_isl_out_of_band"] == 0


def test_actual_tokenized_window_overflow_is_skipped(fake_server, tmp_path):
    """The model-window check uses the target-tokenizer actual count."""
    server, state = fake_server
    cfg = _fake_config(fake_server, tmp_path, repeats=1, samples=2)
    state.max_window = 4096
    state.tokenize_override = 5000  # actual prompt alone exceeds the window
    state.stream_chunk_delay_s = 0.04

    client = OMLXClient(base_url=cfg.server.base_url, api_key="fake-key")
    writer = RunWriter(tmp_path, run_id="actual-overflow")
    try:
        summary = Orchestrator(cfg, client, writer).run()
    finally:
        client.close()

    assert summary["n_samples"] == 0
    assert any("prebuilt prompt too long" in item for item in summary["skipped"])
