#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the accuracy builder's pure helpers.

Scope: only in-memory functions / tiny fixtures — no full ``build.py`` run, no
tokenizer encoding and no ``testset.jsonl`` reads.  The module-level guard skips
everything on a clean clone where ``dataset/raw`` (gitignored) is absent.

Covered:
* ``_pair_aligned_sample_ids`` — Global-MMLU en/zh answer alignment and the
  per-subject mismatch-rate exclusion (upstream ``high_school_world_history``
  rows are misordered by ``sample_id``).
* ``_union_answers`` — LongBench duplicate-prompt answers are unioned in order.
* MBPP prompt rendering — entry point extraction + task + first official tests.
* Metric routing constants and ``judge.max_new_tokens`` budgets (design §6.1).
"""

from __future__ import annotations

import pytest

from tool.common.fs import RAW_DIR
from tool.dataset.accuracy import build, prompts

pytestmark = pytest.mark.skipif(
    not RAW_DIR.exists(),
    reason="dataset/raw is gitignored; run tool/dataset/accuracy/fetch.py to enable accuracy builder tests",
)


# ---------------------------------------------------------------- pairing


def test_pair_aligned_drops_mismatching_subject_and_unaligned_ids():
    en = {"s/a": "A", "s/b": "B", "s/c": "C", "s/d": "D", "s/e": "A"}
    zh = {
        "s/a": "A",
        "s/b": "B",
        "s/c": "C",
        "s/d": "C",  # answer letter differs -> subject "bad" is dropped
        "s/e": "A",
    }
    subject = {"s/a": "good", "s/b": "good", "s/c": "good", "s/d": "bad", "s/e": "bad"}

    aligned, stats, excluded = build._pair_aligned_sample_ids(en, zh, subject)

    assert aligned == ["s/a", "s/b", "s/c"]  # input order kept; "bad" dropped entirely
    assert stats["good"] == {"pairs": 3, "mismatches": 0, "mismatch_rate": 0.0}
    assert stats["bad"]["pairs"] == 2
    assert stats["bad"]["mismatches"] == 1
    assert stats["bad"]["mismatch_rate"] == 0.5
    assert excluded == [{
        "subject": "bad",
        "count": 2,
        "reason": "en/zh answer mismatch rate 50.0% > 0%; upstream rows are misaligned by sample_id",
    }]


def test_pair_aligned_ignores_ids_missing_from_one_language():
    en = {"only_en": "A", "shared": "B"}
    zh = {"only_zh": "C", "shared": "B"}

    aligned, stats, excluded = build._pair_aligned_sample_ids(
        en, zh, {"only_en": "x", "only_zh": "x", "shared": "y"}
    )

    assert aligned == ["shared"]
    assert set(stats) == {"y"}
    assert excluded == []


def test_pair_aligned_threshold_can_keep_mismatching_subjects():
    en = {"s/a": "A", "s/b": "B"}
    zh = {"s/a": "A", "s/b": "C"}

    aligned, _stats, excluded = build._pair_aligned_sample_ids(
        en, zh, {"s/a": "sub", "s/b": "sub"}, max_mismatch_rate=0.6
    )

    assert excluded == []
    assert aligned == ["s/a"]  # mismatching pairs are never aligned, only the subject survives


# ---------------------------------------------------------------- union


def test_union_answers_preserves_order_and_deduplicates():
    assert build._union_answers(["A monk."], ["a priest"]) == ["A monk.", "a priest"]
    assert build._union_answers(["x", "y"], ["y", "z", "x"]) == ["x", "y", "z"]
    assert build._union_answers([], []) == []
    assert build._union_answers(["same"], ["same"]) == ["same"]


# ---------------------------------------------------------------- MBPP prompt


def test_mbpp_entry_point_extracts_first_top_level_def():
    code = "def kth_element(arr, k):\n    return sorted(arr)[k - 1]\n"
    assert prompts.mbpp_entry_point(code) == "kth_element"
    assert prompts.mbpp_entry_point("  def spaced(x):\n      return x\n") == "spaced"


def test_mbpp_entry_point_raises_without_definition():
    with pytest.raises(ValueError):
        prompts.mbpp_entry_point("print('no function here')")


def test_mbpp_prompt_names_entry_point_and_quotes_first_three_tests():
    tests = [
        "assert kth_element([1, 2, 3], 2) == 2",
        "assert kth_element([3, 2, 1], 1) == 1",
        "assert kth_element([5], 1) == 5",
        "assert kth_element([9, 8], 2) == 9",  # fourth test must not be quoted
    ]
    rendered = prompts.mbpp_prompt("Find the kth element.", "kth_element", tests)

    assert "Find the kth element." in rendered
    assert "`kth_element`" in rendered
    assert "test_imports" not in rendered
    for test in tests[:3]:
        assert test in rendered
    assert tests[3] not in rendered
    assert "Output only the Python code" in rendered
    # The judge contract must be able to find the name in the prompt.
    assert "kth_element" in rendered


def test_mbpp_prompt_rejects_missing_tests():
    with pytest.raises(ValueError):
        prompts.mbpp_prompt("Do something.", "fun", [])


# ---------------------------------------------------------------- metric routing


def test_long_context_metric_routing_matches_judges_contract():
    assert build.LONG_CONTEXT_TASK_METRIC == {
        "narrativeqa": "f1",
        "qasper": "f1",
        "hotpotqa": "f1",
        "2wikimqa": "f1",
        "multifieldqa_en": "f1",
        "dureader": "rouge_zh",
        "multifieldqa_zh": "f1_zh",
    }
    assert set(build.LONG_CONTEXT_TASK_METRIC.values()) <= build.ALLOWED_METRICS
    assert build.LONG_CONTEXT_TASK_METRIC["dureader"] == "rouge_zh"
    assert build.LONG_CONTEXT_TASK_METRIC["multifieldqa_zh"] == "f1_zh"


def test_expected_max_new_tokens_budgets():
    assert build.EXPECTED_MAX_NEW_TOKENS == {
        "knowledge_qa": 1536,
        "multilingual": 1536,
        "instruction_following": 1024,
        "math_logic": 512,
        "coding": 768,
        "general_agent": 256,
        "search_agent": 32,
    }


# ---------------------------------------------------------------- small parsers


def test_multilingual_options_requires_four_distinct_non_empty():
    assert build._multilingual_options_ok(["a", "b", "c", "d"]) is True
    assert build._multilingual_options_ok(["a", "a", "c", "d"]) is False
    assert build._multilingual_options_ok(["a", "b", "c", " "]) is False
    assert build._multilingual_options_ok(["a", "b", "c"]) is False


def test_mc_options_round_trip_from_rendered_prompt():
    for template in (prompts.MC_EN, prompts.MC_ZH):
        rendered = template.format(q="Question?", a="alpha", b="beta", c="gamma", d="delta")
        assert build._mc_options_from_prompt(rendered) == ["alpha", "beta", "gamma", "delta"]


# ---------------------------------------------------------------- tokenizer


def test_accuracy_tokenizer_is_cl100k_generic():
    """The accuracy side counts input_tokens with the generic cl100k tokenizer.

    User decision (2026-11-10): no side may bind to a model-specific tokenizer;
    both performance and accuracy reference OpenAI's cl100k_base so declared
    counts stay model-agnostic and cross-side comparable.
    """
    from tool.common import tokenizer as tok

    assert tok.ACCURACY_TOKENIZER_NAME == tok.TOKENIZER_NAME == "cl100k_base"
    assert "MiniCPM" not in tok.ACCURACY_TOKENIZER_NAME
    assert tok.ACCURACY_TOKENIZER_PATH == tok.TOKENIZER_PATH
    assert tok.ACCURACY_TOKENIZER_PATH.name == "tokenizer.json"
    assert tok.ACCURACY_TOKENIZER_PATH.parent.name == "cl100k_base"


def test_accuracy_build_module_uses_generic_tokenizer():
    assert build.TOKENIZER_NAME == "cl100k_base"
    assert build.ACCURACY_TOKENIZER_PATH.name == "tokenizer.json"
    assert "MiniCPM" not in str(build.ACCURACY_TOKENIZER_PATH)
    assert build.TOKENIZER.path == build.ACCURACY_TOKENIZER_PATH


def test_accuracy_raw_spec_has_no_model_bound_tokenizer():
    """raw_spec.json lists only generic tokenizer assets (blob + derived)."""
    import json

    spec = json.loads((RAW_DIR.parent.parent / "tool/dataset/accuracy/raw_spec.json").read_text())

    file_paths = [f["path"] for f in spec["files"]]
    assert all("MiniCPM" not in p for p in file_paths), file_paths

    blob = [f for f in spec["files"] if f["path"].endswith("cl100k_base.tiktoken")]
    assert len(blob) == 1
    assert blob[0]["dataset"] == "tokenizer"
    assert blob[0]["kind"] == "tooling"
    assert blob[0]["required_for"] == ["accuracy"]

    generated = [g for g in spec["generated"] if g["path"].endswith("tokenizer.json")]
    assert len(generated) == 1
    entry = generated[0]
    assert entry["generator"] == "tool/dataset/performance/derive_cl100k.py"
    assert entry["bytes"] == 2729984
    assert entry["sha256"] == "745dc2f8e2eb601a99627c0bef0c11fcb5bf8b311b74472c9b87184aa7c1a154"
    # --only tokenizer covers the generated entry too (dataset key, not the
    # ifeval default).
    assert entry["dataset"] == "tokenizer"


def test_accuracy_sources_tokenizer_entry_is_generated_cl100k():
    from tool.dataset.accuracy import sources

    entry = sources.tokenizer_entry()
    assert entry["path"].endswith("cl100k_base/tokenizer.json")
    assert entry["sha256"] == "745dc2f8e2eb601a99627c0bef0c11fcb5bf8b311b74472c9b87184aa7c1a154"

    tooling = sources.dataset_provenance("tooling")
    tokenizer_files = tooling["tokenizer"]["files"]
    assert len(tokenizer_files) == 2
    assert any(f["path"].endswith("cl100k_base.tiktoken") for f in tokenizer_files)
    derived = [f for f in tokenizer_files if f["path"].endswith("tokenizer.json")]
    assert len(derived) == 1
    assert derived[0].get("derived_by") == "tool/dataset/performance/derive_cl100k.py"
    assert "openai/tiktoken" in tooling["tokenizer"]["source_url"]


def test_accuracy_fetch_generated_rule_supports_generator():
    """The accuracy fetch honours generator entries (same contract as perf fetch)."""
    import inspect

    from tool.dataset.accuracy import fetch

    source = inspect.getsource(fetch.main)
    assert "generator" in source, "accuracy fetch.py must run generator scripts"
    assert "derive_cl100k" not in source, "must stay generic: driven by raw_spec entries"
