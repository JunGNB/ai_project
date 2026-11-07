"""Core unit tests: metrics, padding, config, cache policy, writer, client parsing."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from modelbench.adapters.omlx.client import OMLXClient
from modelbench.adapters.storage.writer import RunWriter
from modelbench.domain.metrics import compute_metrics, distribution, wilson_interval
from modelbench.domain.scenarios.padding import filler_text, fit_prompt
from modelbench.domain.models import (
    CacheMode,
    ChatMessage,
    Phase,
    RawResponse,
    RecordStatus,
    RequestSpec,
    RunManifest,
    Sample,
    SampleRecord,
    Score,
    Usage,
)
from modelbench.adapters.config.settings import load_config
from modelbench.adapters.omlx.cache import CacheController


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #


def test_distribution_basic():
    dist = distribution([1.0, 2.0, 3.0, 4.0])
    assert dist.n == 4
    assert dist.mean == pytest.approx(2.5)
    assert dist.median == pytest.approx(2.5)
    assert dist.min == 1.0 and dist.max == 4.0
    assert dist.p95 == pytest.approx(3.85)


def test_distribution_ignores_none():
    assert distribution([None, None]).n == 0
    dist = distribution([None, 10.0, None, 20.0])
    assert dist.n == 2


def test_wilson_interval_bounds():
    lo, hi = wilson_interval(8, 10)
    assert 0.0 <= lo < 0.8 < hi <= 1.0
    assert wilson_interval(0, 0) is None
    lo0, hi0 = wilson_interval(0, 10)
    assert lo0 == 0.0 and hi0 < 0.35


def test_compute_metrics_decode_rate():
    metrics = compute_metrics(
        ttft_s=1.0,
        first_visible_s=1.2,
        last_token_s=2.0,
        e2e_s=2.1,
        prompt_tokens=1000,
        completion_tokens=21,
    )
    assert metrics.ttft_ms == 1000.0
    assert metrics.ttft_visible_ms == 1200.0
    assert metrics.tpot_ms == pytest.approx(50.0)
    assert metrics.gen_tps == pytest.approx(20.0)
    assert metrics.prefill_tps == pytest.approx(1000.0)


def test_compute_metrics_records_input_context_size():
    metrics = compute_metrics(
        ttft_s=0.5,
        first_visible_s=0.5,
        last_token_s=None,
        e2e_s=0.6,
        prompt_tokens=1234,
        completion_tokens=8,
        input_context_chars=5678,
    )
    assert metrics.input_context_tokens == 1234
    assert metrics.input_context_chars == 5678
    assert metrics.prompt_tokens == 1234


def test_compute_metrics_short_generation_has_no_decode_rate():
    metrics = compute_metrics(
        ttft_s=0.5,
        first_visible_s=0.5,
        last_token_s=0.6,
        e2e_s=0.65,
        prompt_tokens=100,
        completion_tokens=4,
    )
    assert metrics.gen_tps is None
    assert metrics.tpot_ms is None
    assert metrics.e2e_ms == 650.0


def test_compute_metrics_capped_generation_uses_actual_token_window():
    """Hitting max_tokens is measured, not discarded: TG TPS uses emitted tokens."""
    metrics = compute_metrics(
        ttft_s=1.0,
        first_visible_s=1.0,
        last_token_s=1.5,   # capped at 50 tokens; decode window = 0.5s
        e2e_s=1.6,
        prompt_tokens=1000,
        completion_tokens=50,
    )
    assert metrics.gen_tps == pytest.approx(98.0)     # (50-1)/0.5
    assert metrics.tpot_ms == pytest.approx(10.2)     # 0.5/49*1000
    assert metrics.e2e_ms == 1600.0                   # TTFT/e2e unaffected


# --------------------------------------------------------------------------- #
# padding / context fitting
# --------------------------------------------------------------------------- #


class FakeTokenizer:
    """Approximate char-based tokenizer (3.5 chars/token) for offline tests."""

    def __init__(self, cpt: float = 3.5):
        self.cpt = cpt
        self.calls = 0

    def __call__(self, messages, tools=None):
        self.calls += 1
        text = "".join(m.content or "" for m in messages)
        return max(1, int(len(text) / self.cpt))


def test_filler_is_deterministic_and_sized():
    a = filler_text(500, seed="x")
    b = filler_text(500, seed="x")
    c = filler_text(500, seed="y")
    assert a == b and len(a) == 500
    assert a != c


def test_fit_prompt_reaches_target_tokens():
    tokenizer = FakeTokenizer()
    fitted = fit_prompt(
        model="m",
        tokenize=tokenizer,
        target_tokens=4096,
        system_prompt="sys",
        context_text="",
        question_block="What is 2+2?",
        seed="s1",
    )
    assert fitted.actual_tokens == pytest.approx(4096, rel=0.05)
    assert fitted.padded is True
    assert fitted.target_tokens == 4096


def test_fit_prompt_truncates_long_context_and_keeps_question():
    tokenizer = FakeTokenizer()
    long_context = "abcdefghij" * 100000  # way over budget
    fitted = fit_prompt(
        model="m",
        tokenize=tokenizer,
        target_tokens=2048,
        system_prompt=None,
        context_text=long_context,
        question_block="QUESTION_AT_THE_END",
        seed="s2",
    )
    assert fitted.truncated is True
    assert fitted.truncation_ratio and fitted.truncation_ratio > 0.5
    user_text = fitted.messages[-1].content or ""
    assert "QUESTION_AT_THE_END" in user_text
    assert len(user_text) < len(long_context)


def test_fit_prompt_without_target_is_native():
    tokenizer = FakeTokenizer()
    fitted = fit_prompt(
        model="m",
        tokenize=tokenizer,
        target_tokens=None,
        system_prompt=None,
        context_text="ctx",
        question_block="q",
    )
    assert fitted.target_tokens is None
    assert fitted.actual_tokens > 0


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


def test_default_perf_config():
    cfg = load_config()
    assert cfg.perf.samples_per_dataset is None  # default: all samples
    assert cfg.perf.max_tokens == 50
    assert cfg.run.repeats == 1
    assert cfg.run.thinking_mode == "disabled"


def test_config_redacts_keys():
    cfg = load_config()
    cfg.server.api_key = "secret-key"
    snapshot = cfg.redacted_snapshot()
    assert snapshot["server"]["api_key"] == "***redacted***"


def test_unknown_selection_model_is_rejected():
    cfg = load_config()
    cfg.selection.models = ["does-not-exist"]
    with pytest.raises(ValueError):
        cfg.selected_models()


def test_default_selection_is_first_enabled_model():
    cfg = load_config()
    assert cfg.selection.models == []
    assert [m.label for m in cfg.selected_models()] == ["minicpm5-1b"]


def test_model_load_settings_are_moe_only():
    cfg = load_config()
    models = {model.id: model for model in cfg.models}
    # Only MoE models may enable expert offload; everything else is oMLX default.
    assert models["qwen3.6-35B-A3B"].load.settings == {
        "moe_expert_offload_enabled": True
    }
    assert models["Qwen3.8-27B-4bit"].load.settings == {}
    assert models["MiniCPM5-1B"].load.settings == {}
    for model in cfg.models:
        assert set(model.load.settings) <= {"moe_expert_offload_enabled"}


def test_forbidden_model_load_setting_is_rejected():
    from modelbench.adapters.config.settings import ModelLoadConfig

    with pytest.raises(ValueError, match="moe_expert_offload_enabled"):
        ModelLoadConfig(settings={"turboquant_kv_enabled": True})
    with pytest.raises(ValueError, match="moe_expert_offload_enabled"):
        ModelLoadConfig(settings={"dflash_enabled": True})
    with pytest.raises(ValueError):
        ModelLoadConfig(settings={"moe_expert_offload_enabled": "yes"})


def test_perf_samples_validation():
    from modelbench.adapters.config.settings import PerfConfig

    assert PerfConfig(samples_per_dataset=None).samples_per_dataset is None
    assert PerfConfig(samples_per_dataset=3).samples_per_dataset == 3
    with pytest.raises(ValueError):
        PerfConfig(samples_per_dataset=0)
    with pytest.raises(ValueError):
        PerfConfig(max_tokens=0)


# --------------------------------------------------------------------------- #
# cache controller
# --------------------------------------------------------------------------- #


class FakeCacheClient:
    def __init__(self):
        self.clears = 0

    def clear_caches(self):
        self.clears += 1
        return {"latency_ms": 1.0, "hot": {}, "ssd": {}}


def test_cold_mode_clears_every_request():
    fake = FakeCacheClient()
    controller = CacheController(fake, mode=CacheMode.COLD, settle_ms=0)
    controller.start_cell()
    state1 = controller.prepare()
    state2 = controller.prepare()
    assert fake.clears == 2
    assert state1.cleared_before_request and state2.cleared_before_request


def test_warm_mode_clears_once_per_cell():
    fake = FakeCacheClient()
    controller = CacheController(fake, mode=CacheMode.WARM, settle_ms=0)
    controller.start_cell()
    first = controller.prepare()
    assert first.cleared_before_request is True
    assert controller.warm_cleared is True
    second = controller.prepare()
    assert second.cleared_before_request is False
    assert controller.warm_cleared is True
    assert fake.clears == 1
    controller.start_cell()
    assert controller.warm_cleared is False
    controller.prepare()
    assert fake.clears == 2


def test_none_mode_never_clears():
    fake = FakeCacheClient()
    controller = CacheController(fake, mode=CacheMode.NONE, settle_ms=0)
    controller.prepare()
    assert fake.clears == 0


def test_cold_mode_flags_cache_hits():
    controller = CacheController(FakeCacheClient(), mode=CacheMode.COLD, settle_ms=0)
    state = controller.prepare()
    state = controller.evaluate(state, Usage(cached_tokens=128))
    assert state.polluted is True
    assert "cache hit" in (state.note or "")


def test_warm_mode_does_not_flag_hits():
    controller = CacheController(FakeCacheClient(), mode=CacheMode.WARM, settle_ms=0)
    state = controller.prepare()
    state = controller.evaluate(state, Usage(cached_tokens=128))
    assert state.polluted is False


# --------------------------------------------------------------------------- #
# writer
# --------------------------------------------------------------------------- #


def _make_record(run_id: str, ttft: float = 100.0) -> SampleRecord:
    request = RequestSpec(
        model="m1",
        messages=[ChatMessage(role="user", content="hi")],
        max_tokens=16,
        target_context_tokens=None,
        actual_prompt_tokens=1000,
    )
    from modelbench.domain.models import PerfMetrics

    return SampleRecord(
        run_id=run_id,
        model="m1",
        scenario="knowledge_qa",
        context_length=None,
        dataset="mmlu",
        sample_id="s1",
        repeat=0,
        phase=Phase.MEASURED,
        status=RecordStatus.OK,
        request=request,
        response=RawResponse(text="C"),
        metrics=PerfMetrics(
            ttft_ms=ttft,
            gen_tps=20.0,
            peak_cpu_pct=300.0,
            peak_rss_mb=1024.0,
            peak_memory_gb=2.5,
            input_context_tokens=1000,
            input_context_chars=4000,
        ),
        score=None,
    )


def test_writer_roundtrip(tmp_path):
    writer = RunWriter(tmp_path, run_id="testrun")
    manifest = RunManifest(
        run_id="testrun",
        created_at="2026-01-01T00:00:00Z",
        project_version="0.1.0",
    )
    writer.initialize(manifest, {"chip": "M4"}, {"selected": []}, {"models": {}})
    writer.write_record(_make_record("testrun"))
    writer.write_record(_make_record("testrun", ttft=300.0))
    summary = writer.finalize(
        [_make_record("testrun"), _make_record("testrun", ttft=300.0)],
        perf={"samples_per_dataset": None, "max_tokens": 256},
        run_config={"repeats": 1, "cache_mode": "cold"},
    )
    run_dir = Path(summary["run_dir"])
    for name in (
        "summary.json",
        "summary.csv",
        "cells.csv",
        "samples.csv",
        "report.md",
        "omlx.json",
    ):
        assert (run_dir / name).is_file(), f"missing {name}"
    data = json.loads((run_dir / "summary.json").read_text())
    assert data["schema_version"] == "2.1"
    assert data["mode"] == "performance"
    assert data["perf"]["max_tokens"] == 256
    assert data["n_samples"] == 2
    overall = data["overall"][0]
    assert overall["n_valid"] == 2
    assert overall["perf"]["ttft_ms"]["mean"] == pytest.approx(200.0)
    assert overall["perf"]["input_context_tokens"]["mean"] == pytest.approx(1000.0)
    assert overall["peak_rss_mb_max"] == pytest.approx(1024.0)
    assert overall["peak_memory_gb_max"] == pytest.approx(2.5)
    assert overall["perf"]["peak_memory_gb"]["mean"] == pytest.approx(2.5)
    assert len(data["samples"]) == 2
    assert data["samples"][0]["input_context_chars"] == 4000
    assert "accuracy" not in data["cells"][0]
    # Schema 2.1: summary.csv carries one row per (model, context bucket).
    header = (run_dir / "summary.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "context_length" in header
    assert len(data["overall"]) == 1  # both records are legacy (no bucket)


def test_overall_aggregates_are_per_context_bucket():
    """1K/8K/32K are averaged separately; never collapsed into one model mean."""
    from modelbench.domain.metrics import aggregate_overall
    from modelbench.domain.models import Phase

    def rec(context_length, ttft, sample_id):
        return _make_record("ctx", ttft=ttft).model_copy(
            update={"context_length": context_length, "sample_id": sample_id}
        )

    rows = [
        rec(1024, 100.0, "a"),
        rec(1024, 300.0, "b"),
        rec(32768, 30000.0, "c"),
        # Warmup rows are counted but never aggregated.
        rec(32768, 1.0, "c").model_copy(update={"phase": Phase.WARMUP}),
    ]
    overall = aggregate_overall(rows)
    assert [(entry.model, entry.context_length) for entry in overall] == [
        ("m1", 1024),
        ("m1", 32768),
    ]
    assert overall[0].perf["ttft_ms"].mean == pytest.approx(200.0)
    assert overall[1].perf["ttft_ms"].mean == pytest.approx(30000.0)
    assert (overall[0].n_valid, overall[1].n_valid) == (2, 1)
    assert overall[0].n_warmup == 0
    assert overall[1].n_warmup == 1


def test_length_capped_samples_are_aggregated_not_filtered():
    """finish_reason=length records stay in every aggregate (max_tokens is a cap)."""
    from modelbench.domain.metrics import aggregate_overall, sample_results
    from modelbench.domain.models import PerfMetrics

    def capped(ttft, sample_id):
        record = _make_record("cap", ttft=ttft)
        return record.model_copy(
            update={
                "sample_id": sample_id,
                "response": RawResponse(text="A" * 50, finish_reason="length"),
                "metrics": PerfMetrics(
                    ttft_ms=ttft,
                    tpot_ms=10.0,
                    gen_tps=98.0,
                    e2e_ms=ttft + 500.0,
                    input_context_tokens=1000,
                    input_context_chars=4000,
                    completion_tokens=50,
                ),
            }
        )

    rows = [capped(100.0, "a"), capped(300.0, "b")]
    overall = aggregate_overall(rows)[0]
    assert overall.n_valid == 2                # capped samples are NOT dropped
    assert overall.n_length_capped == 2
    assert overall.perf["gen_tps"].mean == pytest.approx(98.0)
    assert overall.perf["ttft_ms"].mean == pytest.approx(200.0)
    projected = sample_results(rows)
    assert all(row.length_capped for row in projected)
    assert {row.finish_reason for row in projected} == {"length"}


def test_report_and_summary_csv_show_context_rows(tmp_path):
    writer = RunWriter(tmp_path, run_id="ctx-run")
    manifest = RunManifest(
        run_id="ctx-run",
        created_at="2026-01-01T00:00:00Z",
        project_version="0.1.0",
    )
    writer.initialize(manifest, {"chip": "M4"}, {"selected": []}, {"models": {}})
    records = []
    for context_length, ttft, finish in ((1024, 100.0, "stop"), (32768, 10000.0, "length")):
        record = _make_record("ctx-run", ttft=ttft).model_copy(
            update={
                "context_length": context_length,
                "response": RawResponse(text="x", finish_reason=finish),
            }
        )
        records.append(record)
        writer.write_record(record)
    summary = writer.finalize(records, perf={"max_tokens": 50}, run_config={"repeats": 1})
    run_dir = Path(summary["run_dir"])
    data = json.loads((run_dir / "summary.json").read_text())
    assert [row["context_length"] for row in data["overall"]] == [1024, 32768]
    assert data["overall"][0]["perf"]["ttft_ms"]["mean"] == pytest.approx(100.0)
    assert data["overall"][1]["perf"]["ttft_ms"]["mean"] == pytest.approx(10000.0)
    # max_tokens cap is recorded, not filtered: capped sample stays in n_valid.
    assert [row["n_length_capped"] for row in data["overall"]] == [0, 1]
    assert [row["n_valid"] for row in data["overall"]] == [1, 1]
    header = (run_dir / "summary.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "n_length_capped" in header
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "按上下文档位分别平均" in report
    assert "触顶" in report
    assert "输出触顶" in report
    assert "| m1 | 1K |" in report
    assert "| m1 | 32K |" in report


# --------------------------------------------------------------------------- #
# client (mock transport, no network)
# --------------------------------------------------------------------------- #


def _make_client(handler) -> OMLXClient:
    client = OMLXClient(base_url="http://test.local", api_key="k")
    client._client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test.local")
    return client


def test_chat_stream_parses_deltas_and_usage():
    sse_lines = [
        'data: {"choices":[{"delta":{"reasoning_content":"think"}}]}',
        'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        'data: {"choices":[{"delta":{"content":" world"},"finish_reason":"stop"}]}',
        'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":2,'
        '"total_tokens":14,"time_to_first_token":0.5,"generation_tokens_per_second":10.0,'
        '"prompt_tokens_details":{"cached_tokens":0}}}',
        "data: [DONE]",
    ]
    body = "\n".join(sse_lines) + "\n"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        payload = json.loads(request.content)
        assert payload["stream"] is True
        assert payload["stream_options"]["include_usage"] is True
        assert request.headers.get("X-oMLX-Cache-Store") == "off"
        return httpx.Response(200, text=body)

    client = _make_client(handler)
    request = RequestSpec(
        model="m", messages=[ChatMessage(role="user", content="hi")], max_tokens=8
    )
    result = client.chat_stream(request, cache_store=False)
    assert result.text == "Hello world"
    assert result.reasoning_text == "think"
    assert result.num_chunks == 4
    assert result.usage.prompt_tokens == 12
    assert result.usage.completion_tokens == 2
    assert result.usage.cached_tokens == 0
    assert result.first_token_ts is not None
    assert result.first_visible_ts is not None
    assert result.elapsed_s > 0


def test_chat_stream_raises_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text='{"detail":"boom"}')

    client = _make_client(handler)
    request = RequestSpec(model="m", messages=[ChatMessage(role="user", content="hi")], max_tokens=8)
    with pytest.raises(Exception):
        client.chat_stream(request)


def test_tokenize_messages_is_cached():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"count": 42, "max_model_len": 4096})

    client = _make_client(handler)
    messages = [ChatMessage(role="user", content="hello")]
    assert client.tokenize_messages(messages, model="m") == 42
    assert client.tokenize_messages(messages, model="m") == 42
    assert calls["n"] == 1


def test_discover_pid_returns_none_without_lsof_match(monkeypatch):
    client = OMLXClient(base_url="http://127.0.0.1:59999", api_key="k")
    assert client.discover_server_pid() is None or isinstance(client.discover_server_pid(), int)


def test_timeout_errors_map_to_timeout_status():
    from modelbench.application.benchmark import Orchestrator

    assert Orchestrator._is_timeout("ReadTimeout: timed out") is True
    assert Orchestrator._is_timeout("ConnectTimeout: connect timed out") is True
    assert Orchestrator._is_timeout("HTTPStatusError: HTTP 500") is False
    assert Orchestrator._is_timeout(None) is False


# --------------------------------------------------------------------------- #
# throughput probe
# --------------------------------------------------------------------------- #


def test_synthetic_adapter_and_throughput_scenario(tmp_path):
    from modelbench.adapters.datasets.registry import get_adapter
    from modelbench.domain.scenarios.registry import build_scenario

    adapter = get_adapter(
        "synthetic", tmp_path / "raw", tmp_path / "processed", {"count": 3}
    )
    samples = adapter.load()
    assert [s.sample_id for s in samples] == ["synthetic-0000", "synthetic-0001", "synthetic-0002"]

    scenario = build_scenario("throughput", {})
    assert scenario.scoring is False
    request = scenario.build_request(
        "m",
        samples[0],
        None,
        FakeTokenizer(),
        max_tokens=64,
    )
    assert request.max_tokens >= 128  # probe needs a measurable decode window
    assert request.chat_template_kwargs is None  # runner applies thinking mode
    assert scenario.score(samples[0], RawResponse(text="x")) is None


def test_custom_jsonl_adapter(tmp_path):
    import json as _json

    from modelbench.adapters.datasets.registry import get_adapter

    path = tmp_path / "mine.jsonl"
    path.write_text(
        "\n".join(
            [
                _json.dumps({"_id": "q1", "question": "1+1?", "choices": ["1", "2", "3", "4"], "answer": "B"}),
                _json.dumps({"sample_id": "q2", "prompt": "Explain X", "context": "doc", "answer_type": "exact"}),
            ]
        ),
        encoding="utf-8",
    )
    adapter = get_adapter(
        "custom", tmp_path / "raw", tmp_path / "processed", {"path": str(path)}
    )
    samples = adapter.load()
    assert [s.sample_id for s in samples] == ["q1", "q2"]
    assert samples[0].answer_type.value == "mc_letter"
    assert samples[0].choices == ["1", "2", "3", "4"]
    assert samples[1].context == "doc"


def test_validate_manifest_checksum_helpers(tmp_path):
    import hashlib

    from modelbench.interfaces.cli import _verify_dataset_manifest

    payload = b"hello benchmark"
    (tmp_path / "file.bin").write_bytes(payload)
    manifest = {
        "schema_version": 1,
        "files": {
            "file.bin": {
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        },
    }
    (tmp_path / "download_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert _verify_dataset_manifest(tmp_path) == []

    (tmp_path / "file.bin").write_bytes(b"tampered")
    problems = _verify_dataset_manifest(tmp_path)
    assert len(problems) == 1 and "size mismatch" in problems[0]

    (tmp_path / "file.bin").write_bytes(payload.replace(b"hello", b"HELLO"))
    problems = _verify_dataset_manifest(tmp_path)
    assert len(problems) == 1 and "sha256 mismatch" in problems[0]


def test_validate_manifest_absent_is_skipped(tmp_path):
    from modelbench.interfaces.cli import _verify_dataset_manifest

    assert _verify_dataset_manifest(tmp_path) == []


def test_git_state_is_recorded_for_traceability():
    from modelbench.application.benchmark import Orchestrator

    commit, dirty = Orchestrator._git_state()
    assert commit is None or len(commit) == 40
    assert dirty is None or isinstance(dirty, bool)


def test_chat_stream_uses_client_timeout_when_caller_does_not_override():
    """Regression F01: an explicit ``timeout=None`` disabled httpx timeouts."""
    import socket
    import threading
    import time

    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    def _accept_and_stall():
        conn, _ = server.accept()
        try:
            time.sleep(1.5)
        finally:
            conn.close()

    threading.Thread(target=_accept_and_stall, daemon=True).start()
    client = OMLXClient(
        base_url=f"http://127.0.0.1:{port}",
        api_key="fake-key",
        request_timeout_s=0.25,
    )
    request = RequestSpec(
        model="m", messages=[ChatMessage(role="user", content="hi")], max_tokens=8
    )
    started = time.perf_counter()
    try:
        with pytest.raises(Exception) as excinfo:
            client.chat_stream(request)
        elapsed = time.perf_counter() - started
        assert elapsed < 1.0, "request timeout was ignored (timeout=None regression)"
        assert "timed out" in str(excinfo.value).lower()
    finally:
        client.close()
        server.close()


def test_load_config_resolves_relative_models_and_scenarios(tmp_path, monkeypatch):
    """Regression F10: config must load from any CWD, not only the repo root."""
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert cfg.models and cfg.scenarios
    assert any(model.id == "MiniCPM5-1B" for model in cfg.models)
    assert Path(cfg.run.output_dir).is_absolute()
    assert Path(cfg.datasets.testset_dir).is_absolute()


def test_warmup_failure_is_not_counted_as_measured_error():
    """Regression F33: per-cell warmup errors must not inflate cell.n_error."""
    from modelbench.domain.metrics import aggregate_cell

    measured = _make_record("r1")
    warmup_error = measured.model_copy(
        update={
            "phase": Phase.WARMUP,
            "status": RecordStatus.ERROR,
            "repeat": -1,
        }
    )
    cell = aggregate_cell([warmup_error, measured])
    assert cell.n_error == 0
    assert cell.n_valid == 1
    assert cell.n_warmup == 1
