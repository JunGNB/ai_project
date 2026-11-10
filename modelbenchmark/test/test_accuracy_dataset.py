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
