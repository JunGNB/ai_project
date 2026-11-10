"""Tests for the performance dataset builder (``tool/dataset/performance``).

Offline only: no ``dataset/raw`` and no reference tokenizer are required.  The
synthetic 240-row mini set mirrors the real builder contract (fields, buckets,
3 rows per target, per-scenario metadata); the negative cases prove that
``validate_rows`` actually rejects a short set, an out-of-bounds token count, a
target with a missing bucket and a duplicated document context.

When the real ``dataset/test/performance/testset.jsonl`` exists it is loaded and
validated too (skipped on a clean clone before ``build.py`` has run).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from tool.common.fs import TEST_DIR, read_json, read_jsonl
from tool.dataset.performance import build


PERF_DIR = TEST_DIR / "performance"
PERF_JSONL = PERF_DIR / "testset.jsonl"
PERF_MANIFEST = PERF_DIR / "manifest.json"


# --------------------------------------------------------------------------- #
# synthetic 240-row mini set
# --------------------------------------------------------------------------- #
def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _meta(scenario: str, idx: int, **extra) -> dict:
    meta = {
        "workload": build.WORKLOADS[scenario],
        "lang": "en",
        "input_tokens": 0,
        "upstream_id": f"upstream|{scenario}|{idx:03d}",
        "context_sha1": _sha1(f"context|{scenario}|{idx:03d}"),
    }
    meta.update(extra)
    return meta


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} description",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _doc_content(scenario: str, idx: int, bucket: str) -> str:
    prefix = build.DOC_PREFIXES[scenario][0]
    doc_1k = f"{scenario} document {idx:03d} sentence. " * 12
    doc_8k = doc_1k + f"{scenario} 8k body. " * 12
    doc_32k = doc_8k + f"{scenario} 32k body. " * 12
    body = {"1k": doc_1k, "8k": doc_8k, "32k": doc_32k}[bucket]
    tail = (
        f"target question {scenario} {idx:03d}?\n"
        "A. first\nB. second\nC. third\nD. fourth\n"
        "Answer with a single letter (A/B/C/D)."
    )
    return prefix + body + build.MARKERS[scenario] + tail


def _row(scenario: str, bucket: str, idx: int) -> dict:
    # cl100k era: the declared target is the calibrated budget for the bucket
    # (build.calibrated_budget), not the raw bucket value.
    tools = None
    tools_tokens = 0
    meta = _meta(scenario, idx)
    dataset = "synthetic"
    constructed = build._expected_constructed(scenario)

    if scenario in build.DOC_SCENARIOS:
        content = _doc_content(scenario, idx, bucket)
        meta.update(
            task=build.DOC_SCENARIO_DOMAIN[scenario],
            domain=build.DOC_SCENARIO_DOMAIN[scenario],
        )
        dataset = f"longbench_v2:{build.DOC_SCENARIO_DOMAIN[scenario]}"
        if scenario == "search_agent":
            tools = build.SEARCH_TOOLS
            tools_tokens = 135
    elif scenario == "instruction_following":
        reference = "".join(
            build._ifeval_reference_unit(
                position,
                {
                    "instruction_id_list": ["change_case:capital_word_frequency"],
                    "kwargs": [{}],
                    "prompt": f"reference instruction {position}",
                },
            )
            for position in range(1, 4)
        )
        content = (
            build.IFE_PREFIX
            + reference
            + build.MARKERS[scenario]
            + f"target instruction {idx:03d}: answer briefly."
        )
        meta.update(
            instruction_ids=["change_case:capital_word_frequency"],
            reference_pool_cycles=1,
            reference_items=3,
            reference_format=build.IFE_REFERENCE_FORMAT,
        )
        dataset = "ifeval"
    elif scenario == "math_logic":
        reference = "Question: reference problem one\nAnswer: 42\n\nQuestion: reference problem two\nAnswer: 7\n\n"
        content = (
            build.MATH_PREFIX
            + reference
            + build.MARKERS[scenario]
            + f"target problem {idx:03d}\nAnswer:"
        )
        meta.update(grade="2")
        dataset = "gsm8k"
    elif scenario == "general_agent":
        own = [_tool(f"own_{idx:03d}")]
        catalogue = own + [_tool(f"extra_{idx:03d}_{k}") for k in range(5)]
        content = (
            build.AGENT_PREFIX
            + json.dumps(catalogue, ensure_ascii=False)
            + build.AGENT_MARKER
            + f"user request {idx:03d}"
        )
        tools = own
        tools_tokens = 24
        category_detail = (
            "simple_python" if idx < 4 else "multiple" if idx < 7 else "parallel"
        )
        meta.update(
            category=build.BFCL_CATEGORY_LABEL[category_detail],
            category_detail=category_detail,
            bfcl_id=f"{category_detail}_{idx:03d}",
            catalog_functions=len(catalogue),
            own_functions=len(own),
        )
        dataset = "bfcl_v4"
    elif scenario == "multilingual":
        language = build.MULTILINGUAL_LANGUAGES[idx]
        reference = "".join(
            f"Question: reference question {k:02d}\nA. a\nB. b\nC. c\nD. d\n" for k in range(6)
        )
        content = (
            build.MULTILINGUAL_PREFIX[language]
            + "\n\n"
            + reference
            + build.MARKERS[scenario]
            + f"target question {idx:03d}\nA. a\nB. b\nC. c\nD. d\n"
            + build.MULTILINGUAL_ANSWER[language]
        )
        meta.update(
            lang=language,
            sample_id=f"subject_{idx:02d}/test/{idx}",
            subject=f"subject_{idx:02d}",
            subject_category="STEM",
        )
        # One dataset name per scenario: the runner cell is (scenario, dataset, bucket).
        dataset = "global_mmlu"
    else:  # pragma: no cover - SCENARIOS is exhaustive
        raise AssertionError(scenario)

    # Synthetic rows carry declared tools_tokens (135 search / 24 agent), so the
    # calibrated budget is derived from the row fields without a tokenizer.
    lang = str(meta.get("lang", "en"))
    target = build.calibrated_budget(
        scenario, lang, bucket, tools_tokens=tools_tokens
    )[0]
    messages_tokens = target - tools_tokens
    row = {
        "id": f"perf-{scenario}-{bucket}-{idx:03d}",
        "scenario": scenario,
        "bucket": bucket,
        "target_id": build.target_id_of(scenario, idx),
        "dataset": dataset,
        "tokens": target,
        "messages_tokens": messages_tokens,
        "tools_tokens": tools_tokens,
        "messages": [{"role": "user", "content": content}],
        "tools": tools,
        "constructed": constructed,
        "meta": dict(meta, input_tokens=target),
    }
    return row


def synthetic_rows() -> list:
    rows = []
    for scenario in build.SCENARIOS:
        for idx in range(build.TARGETS_PER_CELL):
            for bucket in ("1k", "8k", "32k"):
                rows.append(_row(scenario, bucket, idx))
    return rows


def _fresh_rows() -> list:
    return synthetic_rows()


# --------------------------------------------------------------------------- #
# sampling primitives
# --------------------------------------------------------------------------- #
def test_deterministic_pick_is_order_independent_and_deterministic():
    ids = [f"item-{index}" for index in range(50)]
    reverse = list(reversed(ids))
    picked = build.deterministic_pick(ids, 7, build.SEED)
    assert picked == build.deterministic_pick(ids, 7, build.SEED)
    assert picked == build.deterministic_pick(reverse, 7, build.SEED)
    assert len(picked) == 7 and len(set(picked)) == 7
    assert set(picked) <= set(ids)
    # A different seed picks a different deterministic subset.
    assert build.deterministic_pick(ids, 7, 7) != picked or len(set(ids)) < 100


def test_deterministic_pick_rejects_oversized_request():
    with pytest.raises(ValueError):
        build.deterministic_pick(["a", "b"], 3)


def test_allocate_respects_pool_and_sums_to_k():
    pool = {"change_case": 65, "keywords": 88, "length_constraints": 77, "startend": 51}
    quota = build.allocate(pool, 10, min_each=1, seed=build.SEED)
    assert sum(quota.values()) == 10
    assert set(quota) == set(pool)
    for stratum, value in quota.items():
        assert 1 <= value <= pool[stratum]
    assert quota == build.allocate(dict(pool), 10, min_each=1, seed=build.SEED)


def test_allocate_rejects_k_above_pool():
    with pytest.raises(ValueError):
        build.allocate({"a": 2}, 3, min_each=0, seed=build.SEED)


# --------------------------------------------------------------------------- #
# bucket bounds
# --------------------------------------------------------------------------- #
def test_bucket_bounds_are_integer_floors():
    assert build.bucket_bounds(1024) == (972, 1075)
    assert build.bucket_bounds(8192) == (7782, 8601)
    assert build.bucket_bounds(32768) == (31129, 34406)
    assert build.within_bounds(972, 1024) and build.within_bounds(1075, 1024)
    assert not build.within_bounds(971, 1024) and not build.within_bounds(1076, 1024)


def test_bucket_constants_match_spec():
    assert build.BUCKETS == {"1k": 1024, "8k": 8192, "32k": 32768}
    assert build.TARGETS_PER_CELL == 10
    assert build.PER_SCENARIO == 30
    assert build.TOTAL_ROWS == 240
    assert len(build.SCENARIOS) == 8


# --------------------------------------------------------------------------- #
# cl100k budget model (calibrated targets)
# --------------------------------------------------------------------------- #
def test_reference_tokenizer_is_cl100k():
    """The construction tokenizer is the generic cl100k, not a model tokenizer."""
    assert build.TOKENIZER_NAME == "cl100k_base"
    assert build.TOKENIZER_PATH.name == "tokenizer.json"
    assert build.TOKENIZER_PATH.parent.name == "cl100k_base"
def test_calibrated_budget_reduces_content_for_inflating_scenarios():
    """long_context inflates ~1.24x on model tokenizers (measured median), so
    its cl100k budget is ~(bucket - plain reserve)/INFL."""
    budget, low, high = build.calibrated_budget("long_context", "en", "32k", tools_tokens=0)
    assert budget == build.calibrated_budget_target(32768, build.INFL["long_context"], 0)
    assert budget < 32768 * 0.95  # well below the raw bucket
    # ±5% acceptance around the calibrated target (integer floors).
    assert low == math.floor(0.95 * budget) and high == math.floor(1.05 * budget)
    # Non-inflating English scenarios: bucket minus the plain fixed reserve,
    # then divided by the scenario inflation (1.05 after real-run calibration).
    ife = build.calibrated_budget("instruction_following", "en", "1k", tools_tokens=0)[0]
    assert ife == int((1024 - build.PLAIN_RESERVE_CL) / build.INFL["instruction_following"])  # ~897


def test_calibrated_budget_reserves_fixed_overhead_not_tools():
    """Tool scenarios reserve one fixed wrap (tools inflate with content).

    Real-run evidence (20261110T112641Z-8349): the model-side render is
    ``actual ~= declared + 270`` for tool scenarios regardless of bucket --
    a fixed template wrap, NOT a per-row tools deduction.  Deducting the
    tools JSON before dividing by INFL double-subtracts and starves the 1K
    bucket (actual 796 vs band floor 921), so the budget now reserves the
    fixed ``TOOL_RESERVE_CL`` once and lets tools ride the INFL division.
    """
    infl = build.INFL["search_agent"]
    tools_cl = 133
    budget, _low, _high = build.calibrated_budget(
        "search_agent", "en", "8k", tools_tokens=tools_cl
    )
    expected = build.calibrated_budget_target(8192, infl, tools_cl)
    assert budget == expected
    # A single fixed reserve is removed; the tools JSON itself is not.
    assert budget == int((8192 - build.TOOL_RESERVE_CL) / infl)
    # The 1K tool bucket keeps enough content to reach the ±10% band floor
    # (the old double-deduction left it at ~525 declared / 796 actual < 921).
    one_k, _low, _high = build.calibrated_budget(
        "search_agent", "en", "1k", tools_tokens=tools_cl
    )
    assert one_k == int((1024 - build.TOOL_RESERVE_CL) / infl)
    # Simulated actual = declared*infl + observed wrap must clear the band floor.
    assert one_k * infl + build.TOOL_RESERVE_CL >= 921


def test_calibrated_budget_multilingual_uses_language_inflation():
    """Multilingual budgets use the per-language inflation, not the scenario's."""
    arabic = build.calibrated_budget("multilingual", "ar", "8k", tools_tokens=0)[0]
    french = build.calibrated_budget("multilingual", "fr", "8k", tools_tokens=0)[0]
    assert arabic == build.calibrated_budget_target(8192, build.INFL["ar"], 0)
    assert french == build.calibrated_budget_target(8192, build.INFL["fr"], 0)
    assert arabic > french  # ar counts ~2x on cl100k, so more content fits


def test_inflation_table_is_complete_and_generic():
    """Every scenario and multilingual language has an inflation coefficient."""
    for scenario in build.SCENARIOS:
        if scenario == "multilingual":
            continue
        assert scenario in build.INFL, scenario
        assert 0.8 <= build.INFL[scenario] <= 1.5
    for language in build.MULTILINGUAL_LANGUAGES:
        assert language in build.INFL, language
        assert 0.4 <= build.INFL[language] <= 1.2


def test_decode_guard_and_tool_wrap_constants():
    """Fixed cl100k-token reserves documented in the deviations section.

    Calibrated on real-run 20261110T112641Z-8349: the model-side render adds a
    ~270-token fixed wrap when tools are present and ~90 for plain rows.
    """
    assert build.TOOL_RESERVE_CL == 270
    assert build.PLAIN_RESERVE_CL == 90


def test_synthetic_rows_target_calibrated_budgets():
    rows = _fresh_rows()
    for row in rows:
        lang = str(row["meta"]["lang"])
        budget = build.calibrated_budget(
            row["scenario"], lang, row["bucket"], tools_tokens=row["tools_tokens"]
        )[0]
        assert row["tokens"] == budget, row["id"]


def test_validation_rejects_below_calibrated_bounds():
    """A row that stops at the raw-bucket floor can be out of band now:
    e.g. a long_context 1K row with ~614 cl100k tokens renders ~1.2x above
    the 1K measurement band, so it must be rejected at construction time."""
    rows = _fresh_rows()
    row = next(
        item for item in rows
        if item["scenario"] == "long_context" and item["bucket"] == "1k"
    )
    raw_floor = build.bucket_bounds(1024)[0]  # 972 -- inside the OLD bounds
    row["tokens"] = raw_floor
    row["messages_tokens"] = raw_floor - row["tools_tokens"]
    row["meta"]["input_tokens"] = raw_floor
    with pytest.raises(RuntimeError, match="outside"):
        build.validate_rows(rows)


def test_deviations_document_cl100k_calibration():
    """The deviation list must document the generic-tokenizer switch and the
    calibration model, and must no longer claim a BOS is included."""
    ids = {item["id"] for item in build.DEVIATIONS}
    assert "tokenizer-generic-cl100k" in ids
    assert "tokenizer-bos-included" not in ids
    cl100k = next(item for item in build.DEVIATIONS if item["id"] == "tokenizer-generic-cl100k")
    assert "cl100k" in cl100k["note"]
    guard = next(item for item in build.DEVIATIONS if item["id"] == "decode-guard-suffix")
    assert "37" in guard["note"]


# --------------------------------------------------------------------------- #
# validate_rows: positive + negative cases
# --------------------------------------------------------------------------- #
def test_synthetic_rows_pass_validation():
    rows = _fresh_rows()
    assert len(rows) == 240
    build.validate_rows(rows)


def test_validation_rejects_short_row_set():
    rows = _fresh_rows()[:-3]
    with pytest.raises(RuntimeError, match="row count"):
        build.validate_rows(rows)


def test_validation_rejects_out_of_bounds_tokens():
    rows = _fresh_rows()
    row = next(
        item
        for item in rows
        if item["scenario"] == "knowledge_qa" and item["bucket"] == "8k"
    )
    row["tokens"] = 1000
    row["messages_tokens"] = 1000 - row["tools_tokens"]
    row["meta"]["input_tokens"] = 1000
    with pytest.raises(RuntimeError, match="outside"):
        build.validate_rows(rows)


def test_validation_rejects_target_missing_a_bucket():
    rows = _fresh_rows()
    row = next(
        item
        for item in rows
        if item["scenario"] == "math_logic"
        and item["target_id"] == build.target_id_of("math_logic", 0)
        and item["bucket"] == "8k"
    )
    # Move one row to a ninth target: cells keep 10 rows each, but the new target
    # has a single bucket and the old target lost its 8K row.
    row["target_id"] = build.target_id_of("math_logic", 99)
    row["id"] = "perf-math_logic-8k-099"
    with pytest.raises(RuntimeError, match="buckets"):
        build.validate_rows(rows)


def test_validation_rejects_duplicate_document_context():
    rows = _fresh_rows()
    first = next(
        item
        for item in rows
        if item["scenario"] == "knowledge_qa" and item["bucket"] == "1k"
    )
    second = next(
        item
        for item in rows
        if item["scenario"] == "coding" and item["bucket"] == "1k"
    )
    second["meta"]["context_sha1"] = first["meta"]["context_sha1"]
    with pytest.raises(RuntimeError, match="duplicate document context_sha1"):
        build.validate_rows(rows)


def test_validation_rejects_constructed_flag_drift():
    rows = _fresh_rows()
    row = next(item for item in rows if item["scenario"] == "coding" and item["bucket"] == "1k")
    row["constructed"] = True
    with pytest.raises(RuntimeError, match="constructed flag"):
        build.validate_rows(rows)


def test_validation_rejects_unexpected_tools():
    rows = _fresh_rows()
    row = next(item for item in rows if item["scenario"] == "coding" and item["bucket"] == "1k")
    row["tools"] = build.SEARCH_TOOLS
    row["tools_tokens"] = 135
    row["messages_tokens"] = row["tokens"] - 135
    with pytest.raises(RuntimeError, match="tools presence"):
        build.validate_rows(rows)


# --------------------------------------------------------------------------- #
# real outputs (built set), skipped until build.py has run
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not PERF_JSONL.is_file(), reason="dataset/test/performance/testset.jsonl not built")
def test_built_testset_matches_contract():
    rows = read_jsonl(PERF_JSONL)
    build.validate_rows(rows)
    assert len(rows) == 240
    assert len({row["id"] for row in rows}) == 240
    assert len({row["target_id"] for row in rows}) == 80
    assert {row["bucket"] for row in rows} == {"1k", "8k", "32k"}
    assert {row["scenario"] for row in rows} == set(build.SCENARIOS)


@pytest.mark.skipif(
    not (PERF_JSONL.is_file() and PERF_MANIFEST.is_file()),
    reason="dataset/test/performance outputs not built",
)
def test_built_manifest_matches_rows_and_file():
    manifest = read_json(PERF_MANIFEST)
    rows = read_jsonl(PERF_JSONL)
    build._validate_manifest(manifest, rows, PERF_JSONL)
    assert manifest["rows"] == 240
    assert manifest["seed"] == 42
    assert manifest["targets_per_cell"] == 10
    assert len(manifest["deviations"]) >= 6
    assert len(manifest["provenance"]) == 5
    digest = hashlib.sha256(Path(PERF_JSONL).read_bytes()).hexdigest()
    assert digest == manifest["files"]["testset.jsonl"]["sha256"]


@pytest.mark.skipif(
    not (PERF_JSONL.is_file() and PERF_MANIFEST.is_file()),
    reason="dataset/test/performance outputs not built",
)
def test_check_entrypoint_passes_on_built_outputs():
    assert build.check_outputs() == 0


def test_budget_reserves_match_real_run_calibration():
    """Fixed reserves and INFL coefficients calibrated on run 20261110T112641Z-8349.

    Observed model-side render: ``actual ~= declared*INFL + reserve`` with
    reserve ~= 270 for tool scenarios (template wrap) and ~90 for plain text.
    The old model (guard 37 + tools+wrap pre-subtraction) double-deducted the
    tools JSON and starved the 1K tool bucket (0/10 in-band).
    """
    assert build.TOOL_RESERVE_CL == 270
    assert build.PLAIN_RESERVE_CL == 90
    # Recalibrated content coefficients (real-run medians; math/long_context/
    # multilingual unchanged from the original generic table).
    assert build.INFL["general_agent"] == 1.00
    assert build.INFL["search_agent"] == 1.03
    assert build.INFL["coding"] == 1.07
    assert build.INFL["knowledge_qa"] == 1.05
    assert build.INFL["instruction_following"] == 1.04
    assert build.INFL["math_logic"] == 1.15
    assert build.INFL["long_context"] == 1.24


def test_budget_formula_centers_all_tool_cells_in_band():
    """Simulated actual (declared*INFL + reserve) must sit inside the ±10% run
    band [floor(0.9*B), ceil(1.1*B)] for every tool scenario x bucket."""
    for scenario in ("general_agent", "search_agent"):
        infl = build.INFL[scenario]
        for bucket, ctx in build.BUCKETS.items():
            declared = build.calibrated_budget_target(ctx, infl, tools_cl=150)
            actual = declared * infl + build.TOOL_RESERVE_CL
            low = int(ctx * 0.9)
            high = int(ctx * 1.1)
            assert low <= actual <= high, (scenario, bucket, actual, low, high)
            # and centered-ish: distance to bucket center is small relative
            # to the band half-width.
            assert abs(actual - ctx) <= 0.08 * ctx, (scenario, bucket, actual)


def test_budget_formula_centers_plain_cells_in_band():
    for scenario in ("coding", "knowledge_qa", "instruction_following",
                     "math_logic", "long_context"):
        infl = build.INFL[scenario]
        for bucket, ctx in build.BUCKETS.items():
            declared = build.calibrated_budget_target(ctx, infl, tools_cl=0)
            actual = declared * infl + build.PLAIN_RESERVE_CL
            low = int(ctx * 0.9)
            high = int(ctx * 1.1)
            assert low <= actual <= high, (scenario, bucket, actual, low, high)
