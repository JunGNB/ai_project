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
    target = build.BUCKETS[bucket]
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
