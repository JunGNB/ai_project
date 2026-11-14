"""End-to-end tests for ``method=accuracy`` (WP-2).

A deterministic local HTTP server speaks the same protocol as oMLX (SSE chat
completions, tokenize, model status, admin cache clearing), so the real
``AccuracyOrchestrator`` + ``AccuracyRunWriter`` + judges run without a model or
GPU.  The fixture test set is a miniature accuracy set (2 scenarios x 3 rows,
``mc_letter`` + ``normalized_match``/``numeric_exact`` + ``tool_call``, one
over-window row) written into ``tmp_path`` so no offline asset is required.

Covered contract (docs/design/32-method-architecture.md §7,
docs/spec/22-accuracy-test-plan.md §2/§7):

* ``run_id``/``run.json``/``aggregates.json`` are ``method=accuracy``;
* every measured record is judged (``score`` non-null, family recorded), the
  generation cap comes from ``judge.max_new_tokens``, no decode guard is sent,
  the request is cold-cache cleaned before every measured sample and
  ``cached_tokens == 0``;
* over-window rows are ``skipped(over_window)`` (never truncated);
* ``judge_error`` is explicit, counted as 0 and kept in the denominator;
* the plan reports one cell per scenario for accuracy.
"""

from __future__ import annotations

import csv
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from modelbench.adapters.config.settings import ModelSpec, load_config
from modelbench.adapters.config.testsets import load_testsets
from modelbench.adapters.omlx.client import OMLXClient
from modelbench.adapters.storage import layout
from modelbench.application.batch import BatchRunner
from modelbench.application.plan import apply_testset, plan_overview
from modelbench.domain.method import RunMethod
from modelbench.domain.models import CacheMode


# --------------------------------------------------------------------------- #
# Availability gate: WP-3 owns the accuracy writer (do not implement it here).
# --------------------------------------------------------------------------- #


def _accuracy_writer_available() -> bool:
    try:
        from modelbench.adapters.storage.writer import AccuracyRunWriter  # noqa: F401
    except Exception:  # pragma: no cover - depends on the parallel work package
        return False
    return True


requires_accuracy_writer = pytest.mark.skipif(
    not _accuracy_writer_available(),
    reason="AccuracyRunWriter (WP-3) is not available in this checkout",
)


# --------------------------------------------------------------------------- #
# Self-contained fake oMLX server
# --------------------------------------------------------------------------- #


class FakeOMLXState:
    def __init__(self) -> None:
        self.max_window = 4096
        self.clears = 0
        self.chat_requests: List[Dict[str, Any]] = []
        #: ``(max_tokens, cache_clear_calls_so_far)`` for every chat request.
        self.clear_snapshots: List[tuple] = []
        self.cache_store_headers: List[str] = []
        self.loads: List[str] = []
        self.unloads: List[str] = []
        #: Optional hard override for /v1/tokenize (over-window tests).
        self.tokenize_override: Optional[int] = None


#: Deterministic response per judge family: the expected answer is embedded in
#: the prompt marker (``judge=mc`` / ``judge=numeric`` / ``judge=tool_call``).
_TOOL_CALL_RESPONSE = json.dumps(
    {"name": "get_weather", "arguments": {"city": "Paris"}}
)


def _reply_text(messages: List[Dict[str, Any]]) -> str:
    text = "".join(str(message.get("content") or "") for message in messages)
    if "judge=tool_call" in text:
        return _TOOL_CALL_RESPONSE
    if "judge=numeric" in text:
        return "The answer is 42"
    return "答案是 C"


def _sse_payload(reply: str, completion_tokens: int = 10) -> bytes:
    lines = [
        "data: "
        + json.dumps(
            {"choices": [{"delta": {"content": reply}, "finish_reason": "stop"}]},
            ensure_ascii=False,
        ),
        "data: "
        + json.dumps(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": completion_tokens,
                    "total_tokens": 100 + completion_tokens,
                    "prompt_tokens_details": {"cached_tokens": 0},
                    "time_to_first_token": 0.01,
                    "generation_tokens_per_second": 30.0,
                },
            }
        ),
        "data: [DONE]",
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


class FakeOMLXHandler(BaseHTTPRequestHandler):
    state: FakeOMLXState  # set by the fixture

    def log_message(self, *args):  # silence
        pass

    # -- helpers ----------------------------------------------------- #

    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict[str, Any]:
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
                            "loaded": True,
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
                    "auth": {"api_key": "super-secret"},
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
                            "model_context_length": self.state.max_window,
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
                            },
                        }
                    ]
                }
            )
        return self._json({"error": f"unknown GET {path}"}, 404)

    def do_PUT(self):  # noqa: N802
        path = self.path.split("?")[0]
        if path.endswith("/settings") and path.startswith("/admin/api/models/"):
            self._read_json()
            return self._json({"status": "ok", "requires_reload": False})
        return self._json({"error": f"unknown PUT {path}"}, 404)

    def do_POST(self):  # noqa: N802
        path = self.path.split("?")[0]
        if path == "/v1/tokenize":
            body = self._read_json()
            text = "".join(
                str(message.get("content") or "")
                for message in body.get("messages") or []
            )
            if body.get("prompt"):
                text += str(body["prompt"])
            if self.state.tokenize_override is not None:
                count = self.state.tokenize_override
            else:
                marker = re.search(r"tokens=(\d+)", text)
                count = (
                    int(marker.group(1))
                    if marker
                    else max(1, len(text) // 4 + 8)
                )
            return self._json(
                {"count": int(count), "max_model_len": self.state.max_window}
            )
        if path.startswith("/v1/models/") and path.endswith("/load"):
            self.state.loads.append(path.split("/")[3])
            return self._json({"status": "ok"})
        if path.startswith("/v1/models/") and path.endswith("/unload"):
            self.state.unloads.append(path.split("/")[3])
            return self._json({"status": "ok"})
        if path == "/admin/api/login":
            self._read_json()
            return self._json({"status": "ok"})
        if path in ("/admin/api/hot-cache/clear", "/admin/api/ssd-cache/clear"):
            self.state.clears += 1
            return self._json(
                {"status": "ok", "total_cleared": 0, "bytes_reclaimed": 0}
            )
        if path == "/v1/chat/completions":
            body = self._read_json()
            assert body.get("stream") is True
            self.state.chat_requests.append(body)
            self.state.clear_snapshots.append(
                (int(body.get("max_tokens") or 0), int(self.state.clears))
            )
            self.state.cache_store_headers.append(
                self.headers.get("X-oMLX-Cache-Store", "")
            )
            payload = _sse_payload(_reply_text(body.get("messages") or []))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()
            return
        return self._json({"error": f"unknown POST {path}"}, 404)


def _server() -> tuple[ThreadingHTTPServer, FakeOMLXState]:
    state = FakeOMLXState()
    handler = type("Handler", (FakeOMLXHandler,), {"state": state})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, state


@pytest.fixture()
def fake_server():
    server, state = _server()
    try:
        yield server, state
    finally:
        server.shutdown()
        server.server_close()


# --------------------------------------------------------------------------- #
# Miniature accuracy fixture (2 scenarios x 3 rows)
# --------------------------------------------------------------------------- #

#: Per-sample judge contract of the fixture rows (mirrors testset.jsonl lines).
_FIXTURE_ROWS: List[Dict[str, Any]] = [
    {
        "id": f"knowledge_qa:mc:{index}",
        "scenario": "knowledge_qa",
        "dataset": "global_mmlu",
        "language": "en",
        "upstream_id": f"abstract_algebra/test/{index}|en",
        "input_tokens": 120 + index,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Question {index}: choose A-D. "
                    f"tokens={120 + index} judge=mc"
                ),
            }
        ],
        "judge": {"family": "mc_letter", "answer": "C", "max_new_tokens": 64},
        "meta": {
            "sample_id": f"q{index}",
            "subject": "abstract_algebra",
            "stratum": "en",
        },
    }
    for index in range(3)
]

_FIXTURE_ROWS += [
    {
        "id": "math_logic:num:0",
        "scenario": "math_logic",
        "dataset": "gsm8k",
        "language": "en",
        "upstream_id": "gsm8k/test/0|en",
        "input_tokens": 140,
        "messages": [
            {
                "role": "user",
                "content": "Compute the answer. tokens=140 judge=numeric",
            }
        ],
        "judge": {
            "family": "normalized_match",
            "answers": ["42"],
            "metric": "numeric_exact",
            "max_new_tokens": 32,
        },
        "meta": {"sample_id": "m0", "subject": "algebra", "stratum": "en"},
    },
    {
        # Deliberately over the 4096-token fake window: must be skipped, not
        # truncated or judged (spec 22 §2 超窗).
        "id": "math_logic:num:1",
        "scenario": "math_logic",
        "dataset": "gsm8k",
        "language": "en",
        "upstream_id": "gsm8k/test/1|en",
        "input_tokens": 4000,
        "messages": [
            {
                "role": "user",
                "content": "Oversized prompt. tokens=99999 judge=numeric",
            }
        ],
        "judge": {
            "family": "normalized_match",
            "answers": ["42"],
            "metric": "numeric_exact",
            "max_new_tokens": 32,
        },
        "meta": {"sample_id": "m1", "subject": "algebra", "stratum": "en"},
    },
    {
        "id": "math_logic:tool:0",
        "scenario": "math_logic",
        "dataset": "gsm8k",
        "language": "en",
        "upstream_id": "bfcl/0|en",
        "input_tokens": 130,
        "messages": [
            {
                "role": "user",
                "content": "Call the weather tool. tokens=130 judge=tool_call",
            }
        ],
        "judge": {
            "family": "tool_call",
            "ground_truth": [{"get_weather": {"city": ["Paris"]}}],
            "functions": [{"name": "get_weather", "parameters": {}}],
            "max_new_tokens": 48,
        },
        "meta": {"sample_id": "t0", "subject": "tools", "stratum": "en"},
    },
]

#: ``sample_id -> judge.max_new_tokens`` (contract for ``request.max_tokens``).
_EXPECTED_MAX_TOKENS = {
    row["id"]: int(row["judge"]["max_new_tokens"]) for row in _FIXTURE_ROWS
}
_EXPECTED_FAMILIES = {
    row["id"]: str(row["judge"]["family"]) for row in _FIXTURE_ROWS
}
#: The over-window row never produces a record.
_OVER_WINDOW_SAMPLE_ID = "math_logic:num:1"


def _write_accuracy_fixture(tmp_path: Path) -> Path:
    """Write a miniature ``dataset/test/accuracy`` tree; return the root."""
    root = tmp_path / "testsets"
    accuracy_dir = root / "accuracy"
    accuracy_dir.mkdir(parents=True, exist_ok=True)
    data_path = accuracy_dir / "testset.jsonl"
    data_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in _FIXTURE_ROWS)
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "accuracy/v1",
        "name": "accuracy",
        "status": "wired",
        "rows": len(_FIXTURE_ROWS),
        "judges": {
            "mc_letter": "compare the option letter",
            "normalized_match": "numeric exact match",
            "tool_call": "structural function-call match",
        },
    }
    (accuracy_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    quoted_path = json.dumps(str(data_path))
    (accuracy_dir / "testset.yaml").write_text(
        f"""name: accuracy
title: 精度测试（fixture）
method: accuracy
scenarios:
  knowledge_qa:
    dataset: accuracy_jsonl
    dataset_params:
      scenario: knowledge_qa
      path: {quoted_path}
  math_logic:
    dataset: accuracy_jsonl
    dataset_params:
      scenario: math_logic
      path: {quoted_path}
""",
        encoding="utf-8",
    )
    return root


def _accuracy_config(
    fake_server,
    tmp_path: Path,
    *,
    samples: Optional[int] = None,
):
    """Resolved config wired to the fake server and the miniature test set."""
    server, state = fake_server
    root = _write_accuracy_fixture(tmp_path)
    cfg = load_config()
    cfg.server.base_url = f"http://127.0.0.1:{server.server_address[1]}"
    cfg.server.api_key = "fake-key"
    cfg.run.output_dir = str(tmp_path / "output")
    cfg.run.warmup_per_model = 1
    cfg.run.cache_settle_ms = 0
    cfg.run.cpu_sample_interval_ms = 20
    cfg.selection.models = ["fake-model"]
    cfg.models = [ModelSpec(id="fake-model", alias="fake-model")]
    cfg.datasets.testset_dir = str(root)
    if samples is not None:
        cfg.accuracy.samples_per_scenario = int(samples)
    apply_testset(
        cfg,
        load_testsets(root)["accuracy"],
        explicit_method=RunMethod.ACCURACY,
    )
    return cfg, state


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_records(run_dir: Path) -> List[Dict[str, Any]]:
    path = Path(run_dir) / layout.RECORDS
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _run_accuracy_batch(fake_server, tmp_path: Path, *, batch_id: str = "acc-batch"):
    cfg, state = _accuracy_config(fake_server, tmp_path)
    out_root = tmp_path / "out"
    client = OMLXClient(base_url=cfg.server.base_url, api_key="fake-key")
    try:
        batch = BatchRunner(cfg, client, out_root, batch_id=batch_id).run()
    finally:
        client.close()
    run = batch["model_runs"][0]
    return batch, state, out_root, layout.run_dir(out_root, run["run_id"])


# --------------------------------------------------------------------------- #
# Plan / protocol unit tests (no writer required)
# --------------------------------------------------------------------------- #


def test_accuracy_plan_overview_reports_one_cell_per_scenario(fake_server, tmp_path):
    cfg, _state = _accuracy_config(fake_server, tmp_path)
    plan = plan_overview(cfg)
    assert plan["mode"] == "accuracy"
    assert plan["models"] == ["fake-model"]
    assert plan["cells_estimate"] == 2  # 2 scenarios x 1 model, no buckets
    assert len(plan["groups"]) == 2
    assert {group["scenario"] for group in plan["groups"]} == {
        "knowledge_qa",
        "math_logic",
    }
    assert all("context_length" not in group for group in plan["groups"])
    assert plan["samples_total"] == len(_FIXTURE_ROWS)
    assert plan["request_estimate"] == len(_FIXTURE_ROWS)
    assert plan["samples_label"] == "全部可用样本"
    assert plan["judge_families"] == [
        "mc_letter",
        "normalized_match",
        "tool_call",
    ]
    assert plan["needs_judges"] is True
    assert plan["context_windows"] == []
    assert plan["context_filter"] == "全部（不筛选）"


def test_accuracy_plan_overview_honours_samples_per_scenario(fake_server, tmp_path):
    cfg, _state = _accuracy_config(fake_server, tmp_path, samples=2)
    plan = plan_overview(cfg)
    assert plan["samples_per_scenario"] == 2
    assert plan["samples_label"] == "2"
    # 2 per scenario, stratified by meta.stratum.
    assert plan["samples_total"] == 4
    assert plan["request_estimate"] == 4
    assert plan["cells_estimate"] == 2


def test_accuracy_method_forces_run_protocol():
    from modelbench.application.methods import get_method

    cfg = load_config()
    get_method(RunMethod.ACCURACY).configure(cfg)
    assert cfg.method is RunMethod.ACCURACY
    assert cfg.run.cache_mode is CacheMode.COLD
    assert cfg.run.repeats == 1
    assert cfg.run.max_retries == 1
    assert cfg.run.warmup_requests == 0

    warm = load_config()
    warm.run.cache_mode = CacheMode.WARM
    with pytest.raises(ValueError, match="cold"):
        get_method("accuracy").configure(warm)

    repeated = load_config()
    repeated.run.repeats = 3
    with pytest.raises(ValueError, match="repeats"):
        get_method("accuracy").configure(repeated)


def test_apply_testset_rejects_explicit_method_conflict(fake_server, tmp_path):
    _server_obj, _state = fake_server
    root = _write_accuracy_fixture(tmp_path)
    cfg = load_config()
    cfg.datasets.testset_dir = str(root)
    with pytest.raises(ValueError, match="method"):
        apply_testset(
            cfg,
            load_testsets(root)["accuracy"],
            explicit_method=RunMethod.PERFORMANCE,
        )
    # The conflict is detected before the config is mutated.
    assert cfg.method is RunMethod.PERFORMANCE


def test_method_registry_metadata_contract():
    from modelbench.application.methods import get_method, method_metadata

    cards = {card["name"]: card for card in method_metadata()}
    assert list(cards) == ["performance", "accuracy"]
    accuracy = cards["accuracy"]
    assert accuracy["default_testset"] == "accuracy"
    assert accuracy["needs_judges"] is True
    assert "scores" not in accuracy["description"].lower()
    assert "errors" in accuracy["result_views"]
    performance = cards["performance"]
    assert performance["needs_judges"] is False
    assert "errors" not in performance["result_views"]
    assert get_method("performance").config_block == "perf"
    assert get_method(RunMethod.ACCURACY).config_block == "accuracy"


# --------------------------------------------------------------------------- #
# End-to-end batch run (needs the WP-3 accuracy writer)
# --------------------------------------------------------------------------- #


@requires_accuracy_writer
def test_accuracy_batch_end_to_end(fake_server, tmp_path):
    batch, state, out_root, run_dir = _run_accuracy_batch(fake_server, tmp_path)

    # ---- identity / layout ------------------------------------------- #
    assert "__accuracy__" in batch["model_runs"][0]["run_id"]
    assert batch["method"] == "accuracy"
    run = _read_json(run_dir / layout.RUN_JSON)
    assert run["method"] == "accuracy"
    assert run["run_id"] == batch["model_runs"][0]["run_id"]
    assert run["config"]["accuracy"]["samples_per_scenario"] is None
    assert run["config"]["run"]["cache_mode"] == "cold"
    assert run["config"]["run"]["repeats"] == 1
    assert run["config"]["run"]["warmup_requests"] == 0

    records = _read_records(run_dir)
    measured = [record for record in records if record["phase"] == "measured"]
    assert len(measured) == len(_FIXTURE_ROWS) - 1  # one over-window row skipped
    assert all(record["status"] == "ok" for record in measured)

    # ---- judging: every measured record has a score + family ---------- #
    for record in measured:
        sample_id = record["sample_id"]
        score = record["score"]
        assert score is not None, sample_id
        assert score["detail"]["family"] == _EXPECTED_FAMILIES[sample_id]
        assert score["score"] == pytest.approx(1.0), sample_id
        assert score["correct"] is True, sample_id
        assert score["detail"]["judge_error"] is False
        assert score["detail"].get("format_failure") is False
        # Generation cap comes from the row's judge contract.
        assert record["request"]["max_tokens"] == _EXPECTED_MAX_TOKENS[sample_id]
        assert record["request"]["temperature"] == 0.0
        assert record["request"]["top_p"] == 1.0
        assert record["sample_meta"]["judge"]["family"] == _EXPECTED_FAMILIES[sample_id]
        assert record["sample_meta"]["language"] == "en"
        assert record["sample_meta"]["upstream_id"]

    # ---- protocol: no decode guard, cold cache, cached_tokens == 0 ---- #
    for record in measured:
        meta = record["request"]["context_meta"]
        assert "decode_guard" not in meta
        assert "decode_guard_instruction" not in meta
        assert meta["prebuilt"] is True
        assert meta["judge_family"] == _EXPECTED_FAMILIES[record["sample_id"]]
        assert meta["max_new_tokens"] == _EXPECTED_MAX_TOKENS[record["sample_id"]]
        assert meta["tokenized_prompt_tokens"] >= 1
        assert meta["tokenize_latency_ms"] is not None
        cache = record["cache"]
        assert cache["mode"] == "cold"
        assert cache["cleared_before_request"] is True
        assert cache["cached_tokens"] == 0
        assert cache["polluted"] is False

    # Every request was preceded by its own hot+SSD clear (2 admin calls each),
    # so every measured sample is a cold-cache measurement.
    assert state.clears >= 2 * len(measured)
    assert state.clears % 2 == 0
    previous_clears = 0
    for max_tokens, clears_at_request in state.clear_snapshots:
        if max_tokens == 8:
            previous_clears = clears_at_request
            continue  # model warmup request
        assert clears_at_request - previous_clears >= 2, (
            "measured request was not preceded by a fresh hot+SSD cache clear"
        )
        previous_clears = clears_at_request
    # No chat request carried the performance decode guard or a bad sampler cfg.
    assert state.chat_requests
    for body in state.chat_requests:
        if body["max_tokens"] == 8:
            # Model-level warmup request (max_tokens=8): not a measured sample.
            continue
        text = "".join(
            str(message.get("content") or "") for message in body["messages"]
        )
        assert "performance measurement" not in text
        assert body["max_tokens"] in set(_EXPECTED_MAX_TOKENS.values())
        assert body["temperature"] == 0.0
        assert body["top_p"] == 1.0

    # ---- outcome counts + over-window skip ---------------------------- #
    outcome = run["outcome"]
    assert outcome["n_measured"] == len(measured)
    assert outcome["n_ok"] == len(measured)
    assert outcome["n_judge_error"] == 0
    assert outcome["n_scored"] == len(measured)
    assert outcome["n_correct"] == len(measured)
    assert outcome["n_scenarios_scored"] == 2
    skipped = "\n".join(outcome.get("skipped") or [])
    assert "over_window" in skipped
    assert _OVER_WINDOW_SAMPLE_ID in skipped

    # ---- aggregates: accuracy shape ----------------------------------- #
    aggregates = _read_json(run_dir / layout.AGGREGATES_JSON)
    assert aggregates["method"] == "accuracy"
    assert aggregates["n_cells"] == 2
    assert aggregates["n_samples"] == len(measured)
    assert len(aggregates["overall"]) == 1
    overall = aggregates["overall"][0]
    assert overall["method"] == "accuracy"
    assert overall["macro_score"] == pytest.approx(1.0)
    assert overall["n_scenarios_scored"] == 2
    assert set(overall["scenario_scores"]) == {"knowledge_qa", "math_logic"}
    assert {cell["scenario"] for cell in aggregates["cells"]} == {
        "knowledge_qa",
        "math_logic",
    }
    assert all(cell["judge_family"] for cell in aggregates["cells"])

    # ---- samples.csv: one judged row per measured sample -------------- #
    with (run_dir / layout.SAMPLES_CSV).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        rows = list(reader)
    expected_fields = {
        "model",
        "scenario",
        "dataset",
        "language",
        "sample_id",
        "upstream_id",
        "repeat",
        "status",
        "judge_family",
        "judge_metric",
        "score",
        "correct",
        "format_failure",
        "judge_error",
        "max_new_tokens",
        "cache_mode",
        "cache_cleared",
        "cache_polluted",
    }
    assert expected_fields <= fieldnames
    assert len(rows) == len(measured)
    for row in rows:
        assert float(row["score"]) == pytest.approx(1.0)
        assert row["correct"] in ("True", "true", "1")
        assert row["judge_family"] == _EXPECTED_FAMILIES[row["sample_id"]]
        assert row["cache_mode"] == "cold"
        assert row["cache_cleared"] in ("True", "true", "1")
        assert row["cache_polluted"] in ("False", "false", "0", "")

    # ---- batch.json: method + accuracy config + testset evidence ------ #
    batch_json = _read_json(layout.batch_path(out_root, "acc-batch"))
    assert batch_json["method"] == "accuracy"
    assert "accuracy" in batch_json["config"]
    assert "perf" not in batch_json["config"]
    assert batch_json["config"]["run"]["cache_mode"] == "cold"
    testset = batch_json["inputs"]["testset"]
    assert testset["method"] == "accuracy"
    assert testset["manifest_sha256"]
    assert testset["judge_contract_version"].startswith("judges-")
    assert testset["rows"] == len(_FIXTURE_ROWS)


@requires_accuracy_writer
def test_judge_error_is_counted_as_zero_and_kept(fake_server, tmp_path, monkeypatch):
    import modelbench.application.accuracy as accuracy_module

    def boom(row, response):  # noqa: ANN001 - mimics a crashing judge
        raise RuntimeError("judge exploded")

    # The orchestrator must survive a judge that raises instead of returning a
    # JudgeResult (evaluate itself never raises, but a broken patch/plugin must
    # not abort the run).
    monkeypatch.setattr(accuracy_module, "evaluate", boom)

    cfg, _state = _accuracy_config(fake_server, tmp_path)
    out_root = tmp_path / "out"
    client = OMLXClient(base_url=cfg.server.base_url, api_key="fake-key")
    try:
        batch = BatchRunner(cfg, client, out_root, batch_id="acc-judge-error").run()
    finally:
        client.close()

    run_dir = layout.run_dir(out_root, batch["model_runs"][0]["run_id"])
    records = _read_records(run_dir)
    measured = [record for record in records if record["phase"] == "measured"]
    assert measured
    for record in measured:
        assert record["status"] == "judge_error"
        assert record["score"] is not None
        assert record["score"]["score"] == 0.0
        assert record["score"]["correct"] is False
        assert record["score"]["detail"]["judge_error"] is True

    run = _read_json(run_dir / layout.RUN_JSON)
    outcome = run["outcome"]
    assert outcome["n_judge_error"] == len(measured)
    assert outcome["n_scored"] == len(measured)  # zero but still in the denominator
    assert outcome["n_correct"] == 0

    aggregates = _read_json(run_dir / layout.AGGREGATES_JSON)
    overall = aggregates["overall"][0]
    assert overall["n_judge_error"] == len(measured)
    assert overall["macro_score"] == pytest.approx(0.0)
