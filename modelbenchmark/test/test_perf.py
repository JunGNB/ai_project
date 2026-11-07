"""Tests for the performance-only plan, sample selection and oMLX environment."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modelbench.adapters.config.settings import load_config
from modelbench.adapters.config.testsets import available_testsets, load_testsets
from modelbench.paths import PROJECT_ROOT
from modelbench.adapters.omlx.environment import (
    collect_omlx_environment,
    default_settings_repair,
    derive_feature_flags,
    detect_quantization,
    filter_model_settings,
)
from modelbench.application.plan import apply_testset, plan_overview
from modelbench.domain.models import AnswerType, Sample
from modelbench.domain.scenarios.base import deterministic_sample


PERF_JSONL = PROJECT_ROOT / "dataset" / "test" / "performance" / "testset.jsonl"
PERF_DATA = PERF_JSONL.is_file()
requires_perf_data = pytest.mark.skipif(
    not PERF_DATA, reason="dataset/test/performance/testset.jsonl is not on disk"
)

#: Default test-set scenario -> expected sample count (480 total, 8 scenarios).
DEFAULT_SCENARIO_COUNTS = {
    "knowledge_qa": 60,
    "instruction_following": 60,
    "long_context": 60,
    "math_logic": 60,
    "coding": 60,
    "general_agent": 60,
    "search_agent": 60,
    "chinese_long_context": 60,
}
DEFAULT_SAMPLES = 480
DEFAULT_CELLS = 24
FIVE_SAMPLE_CELLS = 24  # 8 scenarios x 3 buckets; 5 samples still touch all cells


def _base():
    cfg = load_config()
    apply_testset(cfg, load_testsets()["performance"])
    return cfg


def _perf_adapter(tmp_path, rows, params=None):
    import json as _json

    from modelbench.adapters.datasets.registry import get_adapter

    path = tmp_path / "testset.jsonl"
    path.write_text(
        "\n".join(_json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return get_adapter(
        "perf_jsonl",
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        params={"path": str(path), **(params or {})},
    )


def _perf_row(index: int, scenario: str = "knowledge_qa", bucket: str = "1k", tokens: int = 100,
              dataset: str = "mmlu", messages=None, tools=None, meta=None):
    return {
        "id": f"test-{scenario}-{index:03d}",
        "scenario": scenario,
        "bucket": bucket,
        "dataset": dataset,
        "tokens": tokens,
        "messages": messages if messages is not None
        else [{"role": "user", "content": f"Question {index}"}],
        "tools": tools or [],
        "constructed": True,
        "meta": {"lang": "en", "gold": "A", **(meta or {})},
    }


def test_no_strategy_layer():
    from modelbench.adapters.config import settings

    assert not hasattr(settings, "MatrixSpec")
    with pytest.raises(ImportError):
        __import__("modelbench.domain.strategies")


def test_perf_plan_defaults():
    cfg = _base()
    plan = plan_overview(cfg)
    assert plan["mode"] == "performance"
    assert plan["models"] == ["minicpm5-1b"]
    assert plan["samples_per_dataset"] is None
    assert plan["samples_label"] == "全部可用样本"
    assert plan["max_tokens"] == 50
    assert plan["repeats"] == 1
    assert plan["warmup_per_model"] == 1
    assert len(plan["scenarios"]) == 8
    assert all("=" in item["label"] for item in plan["scenarios"])


def test_perf_plan_with_fixed_sample_count():
    cfg = _base()
    cfg.perf.samples_per_dataset = 3
    plan = plan_overview(cfg)
    assert plan["samples_label"] == "3"
    assert plan["request_estimate"] == 1 * 8 * 3 * 1


def test_deterministic_sample_takes_first_n_sorted():
    def sample(sample_id: str) -> Sample:
        return Sample(
            sample_id=sample_id,
            dataset="d",
            scenario="s",
            question="q",
            answer_type="exact",
        )

    samples = [sample("c"), sample("a"), sample("b")]
    assert [s.sample_id for s in deterministic_sample(samples, None)] == ["a", "b", "c"]
    assert [s.sample_id for s in deterministic_sample(samples, 2)] == ["a", "b"]
    assert [s.sample_id for s in deterministic_sample(samples, 99)] == ["a", "b", "c"]


def test_custom_testset_points_at_custom_samples(tmp_path):
    """User test sets live under dataset/test/<name>/ and may point at custom JSONL."""
    import json as _json

    rows = [
        {"sample_id": f"q{i}", "question": f"{i}+1?", "choices": ["1", "2"], "answer": "B"}
        for i in range(5)
    ]
    custom = tmp_path / "currency.jsonl"
    custom.write_text(
        "\n".join(_json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    testset_dir = tmp_path / "testsets" / "currency_test"
    testset_dir.mkdir(parents=True)
    (testset_dir / "testset.yaml").write_text(
        "name: currency_test\nscenarios:\n"
        "  math_logic:\n"
        "    dataset: custom\n"
        "    dataset_params:\n"
        f"      path: {custom}\n",
        encoding="utf-8",
    )
    testsets = load_testsets(tmp_path / "testsets")
    assert set(available_testsets(tmp_path / "testsets")) == {"currency_test"}
    cfg = load_config()
    apply_testset(cfg, testsets["currency_test"])
    spec = cfg.scenarios["math_logic"]
    assert spec.dataset == "custom"
    assert spec.params["dataset_params"]["path"] == str(custom)
    from modelbench.adapters.datasets.registry import get_adapter

    adapter = get_adapter(
        "custom",
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        params={"path": str(custom)},
    )
    samples = adapter.load()
    assert len(samples) == 5
    assert samples[0].answer == "B"


def test_web_build_config_matches_cli_semantics():
    from modelbench.interfaces.web.jobs import build_config

    cfg = build_config(
        {
            "models": ["minicpm5-1b"],
            "testset": "performance",
            "samples": 1,
            "max_tokens": 256,
            "repeats": 2,
        }
    )
    assert cfg.selection.models == ["minicpm5-1b"]
    assert cfg.perf.samples_per_dataset == 1
    assert cfg.perf.max_tokens == 256
    assert cfg.run.repeats == 2
    assert cfg.run.output_dir.endswith("output")


def test_web_build_config_rejects_bad_samples():
    from modelbench.interfaces.web.jobs import build_config

    with pytest.raises(ValueError):
        build_config({"models": ["minicpm5-1b"], "samples": 0})
    with pytest.raises(ValueError):
        build_config({"models": [], "samples": 1})


def test_environment_setting_projection_and_flags():
    settings = {
        "moe_expert_offload_enabled": True,
        "turboquant_kv_enabled": True,
        "turboquant_kv_scheme": "affine4",
        "dflash_enabled": False,
        "dflash_draft_model": "/secret/path/draft-model",
        "api_key": "must-not-leak",
        "display_name": "free text",
    }
    filtered = filter_model_settings(settings)
    assert filtered["moe_expert_offload_enabled"] is True
    assert filtered["turboquant_kv_enabled"] is True
    assert filtered["dflash_draft_model"] == "draft-model"
    assert "api_key" not in filtered
    assert "display_name" not in filtered
    flags = derive_feature_flags(settings)
    assert [flag["key"] for flag in flags] == ["affine4_kv"]
    assert derive_feature_flags({"dflash_enabled": False}) == []


def test_detect_quantization(tmp_path):
    model_dir = tmp_path / "MyModel-4bit"
    model_dir.mkdir()
    assert detect_quantization(str(model_dir)) == "4bit"
    (model_dir / "config.json").write_text(
        json.dumps({"quantization_config": {"bits": 8}}), encoding="utf-8"
    )
    assert detect_quantization(str(model_dir)) == "8bit"
    assert detect_quantization(None) == "unknown"


class _FakeEnvClient:
    base_url = "http://fake"

    def device_info(self):
        return {"chip_name": "M4", "chip_variant": "Pro", "memory_gb": 24, "gpu_cores": 16}

    def global_settings(self):
        return {"sampling": {"max_tokens": 1024}, "auth": {"api_key": "secret"}}

    def admin_models(self):
        return {
            "models": [
                {
                    "id": "m1",
                    "model_path": "/models/m1-4bit",
                    "engine_type": "batched",
                    "model_type": "llm",
                    "model_context_length": 4096,
                    "max_tokens": 1024,
                    "estimated_size": 1,
                    "moe_expert_offload_supported": True,
                    "settings": {
                        "moe_expert_offload_enabled": True,
                        "turboquant_kv_enabled": False,
                        "max_tokens": 1024,
                    },
                }
            ]
        }

    def models_status(self):
        return {"models": [{"id": "m1", "max_context_window": 4096}]}


def test_collect_omlx_environment_snapshot():
    env = collect_omlx_environment(_FakeEnvClient(), ["m1"])
    assert env["device"]["chip_name"] == "M4"
    assert env["feature_flags"] == []
    assert env["default_config"] is True
    assert env["upload_projection"]["models"][0]["model_name"] == "m1-4bit"
    assert env["upload_projection"]["os_version"]
    assert "auth" not in env["global_settings"]
    model = env["models"]["m1"]
    assert model["quantization"] == "4bit"
    assert model["settings"]["turboquant_kv_enabled"] is False
    assert model["settings"]["moe_expert_offload_enabled"] is True
    assert model["feature_flags"] == []


def test_default_settings_repair():
    assert default_settings_repair({}) == {}
    assert default_settings_repair({"turboquant_kv_enabled": False}) == {}
    repair = default_settings_repair(
        {"turboquant_kv_enabled": True, "dflash_enabled": True, "moe_expert_offload_enabled": False},
        {"moe_expert_offload_enabled": True},
    )
    assert repair == {
        "dflash_enabled": False,
        "turboquant_kv_enabled": False,
        "moe_expert_offload_enabled": True,
    }


class _TokenClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def tokenize_messages(self, messages, model=None, tools=None):
        text = "".join(m.content or "" for m in messages)
        self.calls.append(model)
        return len(text) // 4 + (100 if model == "model-b" else 8)


def test_request_cache_is_scoped_per_model(tmp_path):
    """A multi-model run must not reuse another model's prompt/token counts."""
    from modelbench.adapters.storage.writer import RunWriter
    from modelbench.application.benchmark import Orchestrator
    from modelbench.domain.scenarios.registry import build_scenario

    cfg = load_config()
    spec = cfg.scenarios["knowledge_qa"]
    scenario = build_scenario("knowledge_qa", spec.params)
    sample = Sample(
        sample_id="s1",
        dataset="mmlu",
        scenario="knowledge_qa",
        question="2+2?",
        choices=["3", "4"],
        answer="B",
        answer_type=AnswerType.MC_LETTER,
    )
    client = _TokenClient()
    writer = RunWriter(tmp_path, run_id="cache")
    orchestrator = Orchestrator(cfg, client, writer)

    first = orchestrator._build_request(scenario, spec, "model-a", sample, None)
    second = orchestrator._build_request(scenario, spec, "model-b", sample, None)
    assert first is not second
    assert first.actual_prompt_tokens != second.actual_prompt_tokens
    assert len(orchestrator._request_cache) == 2
    # Same model still hits its cache (no extra tokenizer call).
    third = orchestrator._build_request(scenario, spec, "model-a", sample, None)
    assert third is first
    assert client.calls == ["model-a", "model-b"]


def test_perf_config_scenario_overrides():
    from modelbench.adapters.config.settings import PerfConfig

    perf = PerfConfig(
        samples_per_dataset=5,
        samples_by_scenario={"long_context": 2},
        sample_ids={"math_logic": ["a", "a", "b"]},
    )
    assert perf.samples_for("long_context") == 2
    assert perf.samples_for("knowledge_qa") == 5
    assert perf.sample_ids_for("math_logic") == ["a", "b"]
    assert perf.sample_ids_for("coding") == []
    with pytest.raises(ValueError):
        PerfConfig(samples_by_scenario={"long_context": 0})


def test_web_build_config_sample_selection():
    from modelbench.interfaces.web.jobs import build_config

    cfg = build_config(
        {
            "models": ["minicpm5-1b"],
            "testset": "performance",
            "scenarios": ["knowledge_qa", "long_context"],
            "samples": 5,
            "scenario_samples": {"long_context": 2},
            "sample_ids": {"knowledge_qa": ["id-1", "id-2"]},
        }
    )
    assert cfg.selection.scenarios == ["knowledge_qa", "long_context"]
    assert cfg.perf.samples_for("long_context") == 2
    assert cfg.perf.sample_ids_for("knowledge_qa") == ["id-1", "id-2"]
    assert cfg.perf.samples_for("math_logic") == 5
    with pytest.raises(ValueError):
        build_config({"models": ["minicpm5-1b"], "scenarios": []})


def test_web_rejects_non_list_sample_ids():
    from modelbench.interfaces.web.jobs import build_config

    with pytest.raises(ValueError, match="sample_ids"):
        build_config(
            {
                "models": ["minicpm5-1b"],
                "testset": "performance",
                "sample_ids": {"knowledge_qa": "abcdef"},
            }
        )


@requires_perf_data
def test_plan_counts_and_progress_tracking():
    from modelbench.interfaces.web.jobs import Job, apply_progress, plan_counts

    cfg = _base()
    cfg.selection.models = ["minicpm5-1b", "qwen3.5-9b"]
    cfg.selection.scenarios = ["knowledge_qa", "long_context"]
    cfg.perf.samples_by_scenario = {"long_context": 2}
    cfg.perf.sample_ids = {
        "knowledge_qa": [
            "perf-knowledge_qa-1k-000",
            "perf-knowledge_qa-1k-001",
            "perf-knowledge_qa-1k-002",
        ]
    }
    plan = plan_counts(cfg)
    assert plan["models_total"] == 2
    assert plan["scenarios_total"] == 2
    # knowledge_qa (3 ids -> one 1k cell) + long_context (N=2 -> one/two cells).
    assert plan["cells_total"] == (1 + 2) * 2
    assert plan["samples_total"] == (3 + 2) * 2 * 1

    job = Job(id="j", request={}, run_id="r", plan=plan)
    apply_progress(job, "model_start", {"model": "m1", "index": 1, "total": 2})
    apply_progress(job, "cell_start", {
        "model": "m1", "scenario": "knowledge_qa", "dataset": "mmlu", "context_length": 1024,
        "samples": 3, "repeats": 1, "cell_index": 1, "cell_total": 4,
    })
    apply_progress(job, "record", {"status": "warmup", "sample_id": "w"})
    assert job.records == 0
    apply_progress(job, "record", {"status": "ok", "sample_id": "a"})
    assert job.records == 1
    assert job.cell["done"] == 1
    snapshot = job.snapshot()
    assert snapshot["progress"]["models_done"] == 0
    assert snapshot["progress"]["models_total"] == 2
    assert snapshot["progress"]["cells_done"] == 0
    assert snapshot["progress"]["samples_done"] == 1
    assert 0 < snapshot["progress"]["percent"] < 100
    apply_progress(job, "cell_done", {"model": "m1", "scenario": "knowledge_qa"})
    apply_progress(job, "model_done", {"model": "m1"})
    apply_progress(
        job,
        "batch_model_start",
        {
            "batch_id": "b1",
            "method": "performance",
            "model": "m1",
            "index": 1,
            "total": 2,
            "run_id": "b1_performance_m1",
            "run_dir": "/tmp/b1_performance_m1",
        },
    )
    snapshot = job.snapshot()
    assert snapshot["progress"]["cells_done"] == 1
    assert snapshot["progress"]["models_done"] == 1
    assert snapshot["batch_id"] == "b1"
    assert snapshot["method"] == "performance"
    assert snapshot["run_dir"] == "/tmp/b1_performance_m1"


def test_context_window_bounds_and_membership():
    from modelbench.domain.context_windows import (
        describe_windows,
        in_context_windows,
        normalize_windows,
        window_bounds,
    )

    # 1K is special: 0 .. 1.5K tokens.
    assert window_bounds(1024) == (0, 1536)
    # every other window is N +/- 0.5K.
    assert window_bounds(8192) == (7680, 8704)
    assert window_bounds(32768) == (32256, 33280)
    # empty selection keeps everything (historical behaviour).
    assert in_context_windows(123, []) is True
    assert in_context_windows(0, [1024]) is True
    assert in_context_windows(1536, [1024]) is True
    assert in_context_windows(1537, [1024]) is False
    assert in_context_windows(7679, [8192]) is False
    assert in_context_windows(7680, [8192]) is True
    assert in_context_windows(8704, [8192]) is True
    assert in_context_windows(8705, [8192]) is False
    # multiple windows combine as a union.
    assert in_context_windows(500, [8192, 1024]) is True
    assert in_context_windows(8000, [8192, 1024]) is True
    assert in_context_windows(3000, [8192, 1024]) is False
    assert normalize_windows([8192, 1024, 8192]) == [1024, 8192]
    assert describe_windows([]) == "全部（不筛选）"
    assert describe_windows([1024, 8192]) == "1K:0–1.5K, 8K:7.5K–8.5K"
    with pytest.raises(ValueError):
        window_bounds(0)


def test_perf_config_context_windows_and_warmup():
    from modelbench.adapters.config.settings import PerfConfig, RunConfig

    perf = PerfConfig(context_windows=[8192, 1024, 8192])
    assert perf.context_windows == [1024, 8192]
    assert PerfConfig().context_windows == []
    with pytest.raises(ValueError):
        PerfConfig(context_windows=[0])
    run = RunConfig(warmup_requests=0)
    assert run.warmup_requests == 0
    assert RunConfig().warmup_per_model == 1
    with pytest.raises(ValueError):
        RunConfig(warmup_requests=-1)
    with pytest.raises(ValueError):
        RunConfig(warmup_per_model=-1)


def test_plan_overview_includes_context_filter_and_warmup():
    cfg = _base()
    cfg.perf.context_windows = [1024, 8192]
    cfg.run.warmup_requests = 2
    plan = plan_overview(cfg)
    assert plan["context_windows"] == [1024, 8192]
    assert plan["context_filter"] == "1K:0–1.5K, 8K:7.5K–8.5K"
    assert plan["warmup_requests"] == 2
    # Filtering never turns the plan into per-context cells.
    assert plan["scenarios"] and all("context" not in item for item in plan["scenarios"])


def test_web_build_config_context_windows_and_warmup():
    from modelbench.interfaces.web.jobs import build_config

    cfg = build_config(
        {
            "models": ["minicpm5-1b"],
            "testset": "performance",
            "scenarios": ["knowledge_qa"],
            "context_windows": [8192, 1024, 8192],
            "warmup_requests": 3,
        }
    )
    assert cfg.perf.context_windows == [1024, 8192]
    assert cfg.run.warmup_requests == 3
    cfg = build_config({"models": ["minicpm5-1b"], "scenarios": ["knowledge_qa"]})
    assert cfg.perf.context_windows == []
    assert cfg.run.warmup_requests == 0
    with pytest.raises(ValueError, match="context_windows"):
        build_config(
            {
                "models": ["minicpm5-1b"],
                "scenarios": ["knowledge_qa"],
                "context_windows": "1024,8192",
            }
        )
    with pytest.raises(ValueError, match="warmup_requests"):
        build_config(
            {
                "models": ["minicpm5-1b"],
                "scenarios": ["knowledge_qa"],
                "warmup_requests": -1,
            }
        )


# --------------------------------------------------------------------------- #
# perf_jsonl adapter (dataset/test/performance)
# --------------------------------------------------------------------------- #


@requires_perf_data
def test_perf_jsonl_file_has_480_unique_samples():
    rows = [
        json.loads(line)
        for line in PERF_JSONL.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == DEFAULT_SAMPLES
    assert len({row["id"] for row in rows}) == DEFAULT_SAMPLES
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["scenario"]] = counts.get(row["scenario"], 0) + 1
    assert {key: counts[key] for key in DEFAULT_SCENARIO_COUNTS} == DEFAULT_SCENARIO_COUNTS
    assert sum(DEFAULT_SCENARIO_COUNTS.values()) == DEFAULT_SAMPLES


@requires_perf_data
def test_performance_manifest_declares_token_semantics():
    from modelbench.paths import PROJECT_ROOT

    manifest = json.loads(
        (PROJECT_ROOT / "dataset" / "test" / "performance" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert "messages + tools" in manifest["tokens_semantics"]
    assert "total prompt" in manifest["bucket_basis"]
    assert manifest["tokens_mean"] > 0
    assert manifest["default_run"]["measured_requests"] == 480
    assert manifest["default_run"]["cells"] == 24
    for stat in manifest["stats"].values():
        assert "tokens_mean" in stat


@requires_perf_data
def test_perf_quick_subset_file_is_10_per_cell():
    import json as _json

    from modelbench.paths import PROJECT_ROOT

    path = PROJECT_ROOT / "dataset" / "test" / "performance" / "testset_10.jsonl"
    rows = [_json.loads(line) for line in path.open() if line.strip()]
    assert len(rows) == 240
    cells = {}
    targets = {}
    for r in rows:
        cells[(r["scenario"], r["bucket"])] = cells.get((r["scenario"], r["bucket"]), 0) + 1
        targets.setdefault(r["scenario"], set()).add(r["target_id"])
    assert len(cells) == 24 and set(cells.values()) == {10}
    assert all(len(v) == 10 for v in targets.values())
    by_target = {}
    for r in rows:
        by_target.setdefault(r["target_id"], []).append(r["bucket"])
    assert all(sorted(v) == ["1k", "32k", "8k"] for v in by_target.values())


@requires_perf_data
def test_perf_jsonl_load_counts_match_default_scenarios():
    from modelbench.adapters.datasets.registry import get_adapter

    cfg = load_config()
    total = 0
    for scenario, expected in DEFAULT_SCENARIO_COUNTS.items():
        spec = cfg.scenarios[scenario]
        adapter = get_adapter(
            "perf_jsonl",
            raw_dir=Path(cfg.datasets.raw_dir),
            processed_dir=Path(cfg.datasets.processed_dir),
            params=dict(spec.params["dataset_params"]),
            testset_dir=Path(cfg.datasets.testset_dir),
        )
        samples = adapter.load()
        assert len(samples) == expected, scenario
        assert all(sample.scenario == scenario for sample in samples)
        assert all(sample.dataset for sample in samples)
        assert all(sample.metadata["prebuilt"] for sample in samples)
        assert all(sample.metadata["messages"] for sample in samples)
        total += len(samples)
    assert total == DEFAULT_SAMPLES


@requires_perf_data
def test_perf_jsonl_plan_matches_load_and_bucket_mapping():
    from modelbench.adapters.datasets.registry import get_adapter

    cfg = load_config()
    spec = cfg.scenarios["long_context"]
    adapter = get_adapter(
        "perf_jsonl",
        raw_dir=Path(cfg.datasets.raw_dir),
        processed_dir=Path(cfg.datasets.processed_dir),
        params=dict(spec.params["dataset_params"]),
        testset_dir=Path(cfg.datasets.testset_dir),
    )
    plan = adapter.plan()
    assert plan is not None
    assert sum(item["n"] for item in plan) == len(adapter.load())
    # long_context now has all three buckets, 20 samples each.
    assert {item["context_length"] for item in plan} == {1024, 8192, 32768}
    assert sum(item["n"] for item in plan if item["context_length"] == 32768) == 20
    # bucket filter narrows the plan and the load identically.
    filtered = get_adapter(
        "perf_jsonl",
        raw_dir=Path(cfg.datasets.raw_dir),
        processed_dir=Path(cfg.datasets.processed_dir),
        params={**spec.params["dataset_params"], "buckets": ["32k"]},
        testset_dir=Path(cfg.datasets.testset_dir),
    )
    assert sum(item["n"] for item in filtered.plan()) == 20
    assert len(filtered.load()) == 20
    assert {sample.metadata["context_length"] for sample in filtered.load()} == {32768}


def test_perf_jsonl_filters_limit_and_streams(tmp_path):
    rows = [
        _perf_row(0, dataset="mmlu"),
        _perf_row(1, scenario="math_logic", dataset="gsm8k"),
        _perf_row(2, bucket="8k", tokens=8200, dataset="longbench_v1:gov_report"),
        _perf_row(3),
    ]
    adapter = _perf_adapter(tmp_path, rows, {"scenario": "knowledge_qa", "limit": 2})
    samples = adapter.load()
    assert [sample.sample_id for sample in samples] == ["test-knowledge_qa-000", "test-knowledge_qa-002"]
    assert samples[0].metadata["context_length"] == 1024
    assert samples[1].metadata["context_length"] == 8192
    assert samples[0].metadata["bucket"] == "1k"
    assert samples[0].metadata["tokens"] == 100
    # The plan uses the same filter/limit semantics.
    assert sum(item["n"] for item in adapter.plan()) == 2
    bucket_filtered = _perf_adapter(
        tmp_path, rows, {"scenario": "knowledge_qa", "buckets": ["8k"]}
    )
    assert len(bucket_filtered.load()) == 1
    assert {item["context_length"] for item in bucket_filtered.plan()} == {8192}
    with pytest.raises(ValueError, match="no samples"):
        _perf_adapter(tmp_path, rows, {"scenario": "coding"}).load()
    with pytest.raises(ValueError, match="requires params.dataset_params.scenario"):
        _perf_adapter(tmp_path, rows, {}).load()
    with pytest.raises(ValueError, match="unknown bucket"):
        _perf_adapter(tmp_path, rows, {"scenario": "knowledge_qa", "buckets": ["64k"]}).load()


def test_perf_jsonl_missing_file_raises(tmp_path):
    from modelbench.adapters.datasets.registry import get_adapter

    adapter = get_adapter(
        "perf_jsonl",
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        params={"scenario": "knowledge_qa", "path": str(tmp_path / "missing.jsonl")},
        testset_dir=tmp_path / "test",
    )
    with pytest.raises(FileNotFoundError, match="performance test set not found"):
        adapter.load()
    assert adapter.plan() is None


# --------------------------------------------------------------------------- #
# pre-built request passthrough
# --------------------------------------------------------------------------- #


def _prebuilt_sample(**overrides):
    from modelbench.domain.models import AnswerType, Sample, ToolSpec

    messages = overrides.pop(
        "messages",
        [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}],
    )
    tools = overrides.pop(
        "tools",
        [{"type": "function", "function": {"name": "f", "parameters": {}}}],
    )
    metadata = {
        "prebuilt": True,
        "messages": messages,
        "tools": tools,
        "tokens": 1234,
        "bucket": "1k",
        "context_length": 1024,
        "constructed": True,
        "lang": "en",
        "gold": "A",
    }
    metadata.update(overrides.pop("metadata", {}))
    return Sample(
        sample_id="prebuilt-1",
        dataset="mmlu",
        scenario="knowledge_qa",
        question="hi",
        answer="A",
        answer_type=AnswerType.MC_LETTER,
        tools=[
            ToolSpec(**tool) for tool in tools
        ] or None,
        metadata=metadata,
        **overrides,
    )


def test_prebuilt_request_passthrough_skips_rendering_and_tokenizer():
    from modelbench.domain.scenarios.registry import build_scenario

    sample = _prebuilt_sample()
    scenario = build_scenario("knowledge_qa", {})

    def forbidden(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("prebuilt requests must not be tokenized/rendered")

    request = scenario.build_request(
        "model-a", sample, None, forbidden, max_tokens=64
    )
    assert [message.role for message in request.messages] == ["system", "user"]
    assert request.messages[1].content == "hi"
    assert request.tools and request.tools[0].function["name"] == "f"
    assert request.actual_prompt_tokens == 1234
    assert request.target_context_tokens is None
    assert request.truncated is False
    assert request.context_meta["prebuilt"] is True
    assert request.context_meta["bucket"] == "1k"
    assert request.context_meta["context_length"] == 1024


def test_prebuilt_request_too_long_for_window_raises_valueerror(tmp_path):
    """Window overflow is a skip, never a silent truncation."""
    from modelbench.adapters.storage.writer import RunWriter
    from modelbench.application.benchmark import Orchestrator
    from modelbench.domain.scenarios.registry import build_scenario

    cfg = load_config()
    spec = cfg.scenarios["long_context"]
    scenario = build_scenario("long_context", spec.params)
    sample = _prebuilt_sample(metadata={"tokens": 9000, "context_length": 8192, "bucket": "8k"})
    writer = RunWriter(tmp_path, run_id="prebuilt-overflow")
    orchestrator = Orchestrator(cfg, object(), writer)
    with pytest.raises(ValueError, match="prebuilt prompt too long"):
        orchestrator._build_request(scenario, spec, "model-a", sample, 8192)
    # It fits once the window is large enough.
    request = orchestrator._build_request(scenario, spec, "model-a", sample, 32768)
    assert request.actual_prompt_tokens == 9000
    assert request.truncated is False


def test_window_check_matches_documented_242_token_reserve(tmp_path):
    """F04: docs promise ``declared + 242``; runtime must not add extra slack."""
    from modelbench.adapters.storage.writer import RunWriter
    from modelbench.application.benchmark import Orchestrator
    from modelbench.domain.scenarios.registry import build_scenario

    cfg = load_config()
    spec = cfg.scenarios["long_context"]
    scenario = build_scenario("long_context", spec.params)
    window = 32768
    fits = window - (
        cfg.perf.max_tokens + cfg.run.context_safety_margin_tokens + 64
    )  # 32526, exactly the documented boundary
    ok_sample = _prebuilt_sample(
        metadata={"tokens": fits, "context_length": 32768, "bucket": "32k"},
    )
    over_sample = _prebuilt_sample(
        metadata={"tokens": fits + 1, "context_length": 32768, "bucket": "32k"},
    )
    writer = RunWriter(tmp_path, run_id="window-reserve")
    orchestrator = Orchestrator(cfg, object(), writer)
    orchestrator._build_request(scenario, spec, "model-ok", ok_sample, window)
    with pytest.raises(ValueError, match="prebuilt prompt too long"):
        orchestrator._build_request(scenario, spec, "model-over", over_sample, window)


def test_decode_guard_appends_long_output_suffix(tmp_path):
    """Performance mode appends a fixed long-output prompt so TG TPS is measurable."""
    from modelbench.adapters.storage.writer import RunWriter
    from modelbench.application.benchmark import DECODE_GUARD_INSTRUCTION, Orchestrator
    from modelbench.domain.scenarios.registry import build_scenario

    cfg = load_config()
    spec = cfg.scenarios["knowledge_qa"]
    scenario = build_scenario("knowledge_qa", spec.params)
    sample = _prebuilt_sample()
    writer = RunWriter(tmp_path, run_id="guard-on")
    orchestrator = Orchestrator(cfg, object(), writer)
    request = orchestrator._build_request(scenario, spec, "model-a", sample, 32768)
    assert request.messages[-1].role == "user"
    assert request.messages[-1].content == DECODE_GUARD_INSTRUCTION
    assert request.context_meta["decode_guard"] is True
    assert request.context_meta["decode_guard_instruction"] == DECODE_GUARD_INSTRUCTION
    # Declared tokens remain the dataset/bucket definition (usage is authoritative).
    assert request.actual_prompt_tokens == 1234

    cfg.perf.decode_guard = False
    writer_off = RunWriter(tmp_path, run_id="guard-off")
    orchestrator_off = Orchestrator(cfg, object(), writer_off)
    request_off = orchestrator_off._build_request(scenario, spec, "model-a", sample, 32768)
    assert [message.role for message in request_off.messages] == ["system", "user"]
    assert "decode_guard" not in request_off.context_meta


def test_decode_guard_reserves_window_headroom(tmp_path):
    """The guard must never push a prebuilt prompt past the model window."""
    from modelbench.adapters.storage.writer import RunWriter
    from modelbench.application.benchmark import Orchestrator
    from modelbench.domain.scenarios.registry import build_scenario

    cfg = load_config()
    cfg.perf.max_tokens = 64
    spec = cfg.scenarios["knowledge_qa"]
    scenario = build_scenario("knowledge_qa", spec.params)
    writer = RunWriter(tmp_path, run_id="guard-window")
    orchestrator = Orchestrator(cfg, object(), writer)
    # Fits without the 64-token guard reserve, overflows with it.
    tokens = 32768 - 64 - cfg.run.context_safety_margin_tokens - 30
    sample = _prebuilt_sample(metadata={"tokens": tokens})
    with pytest.raises(ValueError, match="prebuilt prompt too long"):
        orchestrator._build_request(scenario, spec, "model-a", sample, 32768)
    cfg.perf.decode_guard = False
    request = orchestrator._build_request(scenario, spec, "model-a", sample, 32768)
    assert request.context_meta["prebuilt"] is True


def test_perf_config_decode_guard_default_and_override():
    from modelbench.adapters.config.settings import PerfConfig

    assert PerfConfig().decode_guard is True
    assert PerfConfig(decode_guard=False).decode_guard is False


def test_context_window_filter_uses_declared_tokens_for_prebuilt(tmp_path):
    """Prebuilt samples are filtered by their declared tokens (not model tokenizer)."""
    from modelbench.adapters.storage.writer import RunWriter
    from modelbench.application.benchmark import Orchestrator
    from modelbench.domain.scenarios.registry import build_scenario

    cfg = load_config()
    spec = cfg.scenarios["long_context"]
    scenario = build_scenario("long_context", spec.params)
    sample = _prebuilt_sample(metadata={"tokens": 8197, "context_length": 8192, "bucket": "8k"})
    writer = RunWriter(tmp_path, run_id="ctx-declared")
    orchestrator = Orchestrator(cfg, object(), writer)

    def forbidden(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("prebuilt context filter must use declared tokens")

    orchestrator._build_request = forbidden  # type: ignore[method-assign]
    cfg.perf.context_windows = [8192]
    kept = orchestrator._filter_by_context(
        "long_context", scenario, spec, "model-a", [sample], 65536
    )
    assert [item.sample_id for item in kept] == ["prebuilt-1"]
    cfg.perf.context_windows = [1024]
    assert orchestrator._filter_by_context(
        "long_context", scenario, spec, "model-a", [sample], 65536
    ) == []


@requires_perf_data
def test_plan_overview_reports_default_requests_and_cell_groups():
    cfg = _base()
    plan = plan_overview(cfg)
    assert plan["request_estimate"] == DEFAULT_SAMPLES
    assert plan["samples_total"] == DEFAULT_SAMPLES
    assert plan["cells_estimate"] == DEFAULT_CELLS
    assert len(plan["groups"]) == DEFAULT_CELLS
    assert sum(group["n"] for group in plan["groups"]) == DEFAULT_SAMPLES
    # Every group is a distinct (scenario, dataset, context_length) cell key.
    keys = {
        (group["scenario"], group["dataset"], group["context_length"])
        for group in plan["groups"]
    }
    assert len(keys) == DEFAULT_CELLS
    assert {group["context_length"] for group in plan["groups"]} == {1024, 8192, 32768}
    # Context-window filtering is exact for perf_jsonl (declared tokens).
    cfg.perf.context_windows = [1024]
    filtered = plan_overview(cfg)
    assert filtered["request_estimate"] == 160
    assert filtered["cells_estimate"] == 8
    assert filtered["samples_total"] == 160


@requires_perf_data
def test_web_plan_counts_matches_default_samples_and_cells():
    from modelbench.interfaces.web.jobs import build_config, plan_counts

    cfg = build_config({"models": ["minicpm5-1b"]})
    counts = plan_counts(cfg)
    assert counts["scenarios_total"] == 8
    assert counts["samples_total"] == DEFAULT_SAMPLES
    assert counts["cells_total"] == DEFAULT_CELLS
    assert counts["cells_known"] is True



# --------------------------------------------------------------------------- #
# stratified sampling / declared-token window rule
# --------------------------------------------------------------------------- #


def _bucket_sample(sample_id, context_length, lang="en"):
    from modelbench.domain.models import AnswerType, Sample

    return Sample(
        sample_id=sample_id,
        dataset="d",
        scenario="s",
        question="q",
        answer_type=AnswerType.EXACT,
        metadata={
            "prebuilt": True,
            "context_length": context_length,
            "bucket": {1024: "1k", 8192: "8k", 32768: "32k"}[context_length],
            "lang": lang,
        },
    )


def test_stratified_sample_covers_buckets_proportionally():
    from modelbench.domain.scenarios.base import stratified_sample

    samples = [_bucket_sample(f"a-8k-{i:03d}", 8192) for i in range(6)]
    samples += [_bucket_sample(f"b-32k-{i:03d}", 32768) for i in range(4)]
    chosen = stratified_sample(samples, 5)
    counts = {}
    for sample in chosen:
        counts[sample.metadata["context_length"]] = (
            counts.get(sample.metadata["context_length"], 0) + 1
        )
    assert counts == {8192: 3, 32768: 2}  # 50/50 split by largest remainder
    assert chosen == sorted(chosen, key=lambda sample: sample.sample_id)
    # Global 32k-first ordering must not starve the 8K bucket.
    assert any(sample.metadata["bucket"] == "8k" for sample in chosen)
    # n >= len keeps everything sorted; legacy samples fall back to first-N.
    assert len(stratified_sample(samples, None)) == 10
    assert [s.sample_id for s in stratified_sample(samples, 99)] == sorted(
        sample.sample_id for sample in samples
    )
    legacy = [
        Sample(sample_id=f"l{i}", dataset="d", scenario="s", question="q", answer_type="exact")
        for i in range(3)
    ]
    assert [s.sample_id for s in stratified_sample(legacy, 2)] == ["l0", "l1"]


@requires_perf_data
def test_samples_5_on_mixed_bucket_scenario_keeps_both_buckets():
    from modelbench.adapters.datasets.registry import get_adapter

    cfg = load_config()
    spec = cfg.scenarios["long_context"]
    adapter = get_adapter(
        "perf_jsonl",
        raw_dir=Path(cfg.datasets.raw_dir),
        processed_dir=Path(cfg.datasets.processed_dir),
        params=dict(spec.params["dataset_params"]),
        testset_dir=Path(cfg.datasets.testset_dir),
    )
    from modelbench.domain.scenarios.base import stratified_sample

    chosen = stratified_sample(adapter.load(), 5)
    assert len(chosen) == 5
    assert {sample.metadata["context_length"] for sample in chosen} == {1024, 8192, 32768}


def test_declared_32k_bucket_over_model_window_is_not_testable():
    """Nominal 32K samples measure up to ~32,785 tokens; a 32,768 window must skip."""
    from modelbench.adapters.storage.writer import RunWriter
    from modelbench.application.benchmark import Orchestrator
    from modelbench.domain.scenarios.registry import build_scenario

    cfg = load_config()
    spec = cfg.scenarios["long_context"]
    scenario = build_scenario("long_context", spec.params)
    # Real 32K declarations are 32,155-32,785 tokens (manifest/rows).
    sample = _prebuilt_sample(metadata={"tokens": 32785, "context_length": 32768, "bucket": "32k"})
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        writer = RunWriter(Path(tmp), run_id="32k-window")
        orchestrator = Orchestrator(cfg, object(), writer)
        with pytest.raises(ValueError, match="prebuilt prompt too long"):
            orchestrator._build_request(scenario, spec, "model-a", sample, 32768)
        # A 40,960-token model window (smallest in models.yaml) fits it.
        request = orchestrator._build_request(scenario, spec, "model-b", sample, 40960)
        assert request.actual_prompt_tokens == 32785
        assert request.truncated is False


@requires_perf_data
def test_prebuilt_tools_are_preserved_verbatim():
    """Agentic scenarios keep their dataset tool schemas; the run never calls them."""
    from modelbench.adapters.datasets.registry import get_adapter
    from modelbench.domain.scenarios.registry import build_scenario

    cfg = load_config()
    standard = {"object", "number", "array", "integer", "string", "boolean", "null", "function"}

    def walk_types(value, seen):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "type" and isinstance(item, str):
                    seen.add(item)
                walk_types(item, seen)
        elif isinstance(value, list):
            for item in value:
                walk_types(item, seen)

    for scenario_name in ("general_agent", "search_agent"):
        spec = cfg.scenarios[scenario_name]
        adapter = get_adapter(
            "perf_jsonl",
            raw_dir=Path(cfg.datasets.raw_dir),
            processed_dir=Path(cfg.datasets.processed_dir),
            params=dict(spec.params["dataset_params"]),
            testset_dir=Path(cfg.datasets.testset_dir),
        )
        sample = adapter.load()[0]
        scenario = build_scenario(scenario_name, spec.params)
        request = scenario.build_request(
            "model-a", sample, None, lambda *args, **kwargs: 0, max_tokens=64
        )
        assert request.context_meta["prebuilt"] is True
        assert sample.dataset
        assert request.tools, scenario_name
        seen = set()
        walk_types(request.tools, seen)
        assert seen <= standard, (scenario_name, seen - standard)
        assert sample.metadata["tokens"] > sample.metadata["messages_tokens"]
        assert sample.metadata["tools_tokens"] > 0
        # Declared total equals messages+tools tokens (agent rows carry both).
        assert (
            sample.metadata["tokens"]
            == sample.metadata["messages_tokens"] + sample.metadata["tools_tokens"]
        )



@requires_perf_data
def test_plan_overview_exact_for_sample_count_and_windows():
    cfg = _base()
    cfg.perf.samples_per_dataset = 5
    five = plan_overview(cfg)
    assert five["samples_total"] == 40
    assert five["request_estimate"] == 40
    # Stratified N drops buckets/datasets that receive no sample.
    assert five["cells_estimate"] == FIVE_SAMPLE_CELLS
    cfg.perf.samples_per_dataset = None
    cfg.perf.context_windows = [1024, 8192]
    windows = plan_overview(cfg)
    assert windows["samples_total"] == 320
    assert windows["request_estimate"] == 320
    assert windows["cells_estimate"] == 16


@requires_perf_data
def test_web_plan_counts_exact_for_sample_count_and_windows():
    from modelbench.interfaces.web.jobs import build_config, plan_counts

    five = plan_counts(build_config({"models": ["minicpm5-1b"], "samples": 5}))
    assert five["samples_total"] == 40
    assert five["cells_total"] == FIVE_SAMPLE_CELLS
    assert five["cells_known"] is True
    windows = plan_counts(
        build_config(
            {
                "models": ["minicpm5-1b"],
                "context_windows": [1024, 8192],
            }
        )
    )
    assert windows["samples_total"] == 320
    assert windows["cells_total"] == 16


def test_perf_jsonl_plan_honours_samples_windows_and_ids(tmp_path):
    rows = (
        [_perf_row(0, tokens=100, dataset="mmlu")]
        + [_perf_row(1, tokens=100, dataset="ceval", meta={"lang": "zh"})]
        + [_perf_row(2, bucket="8k", tokens=8200, dataset="longbench_v1:gov_report")]
        + [_perf_row(3, bucket="8k", tokens=8200, dataset="longbench_v1:gov_report")]
        + [_perf_row(4, bucket="32k", tokens=32000, dataset="niah_style")]
    )
    base = {"scenario": "knowledge_qa"}
    adapter = _perf_adapter(tmp_path, rows, base)
    default_plan = adapter.plan()
    assert sum(item["n"] for item in default_plan) == 5
    assert len(default_plan) == 4  # mmlu / ceval / gov_report(2) / niah_style
    strata = adapter.strata()
    assert sum(item["n"] for item in strata) == 5
    # The plan exposes (context_length, lang) strata, not just aggregate cells.
    assert {(item["context_length"], item["lang"]) for item in strata} >= {
        (1024, "en"),
        (1024, "zh"),
        (8192, "en"),
        (32768, "en"),
    }
    # N=2 stratifies across buckets instead of taking the first 2 file rows (both 1k).
    sampled = adapter.plan({**base, "samples": 2})
    assert sum(item["n"] for item in sampled) == 2
    assert {item["context_length"] for item in sampled} == {1024, 8192}
    # Window filter uses declared tokens.
    windowed = adapter.plan({**base, "context_windows": [8192]})
    assert sum(item["n"] for item in windowed) == 2
    assert {item["context_length"] for item in windowed} == {8192}
    # Explicit ids are honoured exactly.
    by_id = adapter.plan({**base, "sample_ids": ["test-knowledge_qa-002"]})
    assert sum(item["n"] for item in by_id) == 1
    assert by_id[0]["dataset"] == "longbench_v1:gov_report"
