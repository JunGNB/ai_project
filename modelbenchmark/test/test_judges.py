#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for the deterministic accuracy judges.

These cover the P0 defects found in the 2026-11 dataset audit: Chinese
``mc_letter`` parsing, BFCL irrelevance / nested-array JSON, the HumanEval
``sys.exit(0)`` false pass, LongBench Chinese metrics and numeric extraction.
"""
from __future__ import annotations

from fractions import Fraction

import pytest

from tool.system import judges


# ---------------------------------------------------------------- mc_letter

@pytest.mark.parametrize(
    "response,expected",
    [
        ("答案是 C", "C"),
        ("答案是 C。", "C"),
        ("答案：c", "C"),
        ("答案是 Ｃ", "C"),                       # full-width letter
        ("The answer is C.", "C"),
        ("The answer is (C).", "C"),
        ("Option A seems plausible, but option C is correct.", "C"),
        ("A. 8\nB. 2\nC. 24\nD. 120\n答案是 C", "C"),
        ("c", "C"),
    ],
)
def test_mc_letter_parses_answer_formats(response, expected):
    assert judges.extract_letter(response) == expected


@pytest.mark.parametrize(
    "response",
    [
        "Cannot determine from the given options.",
        "无法确定这道题的答案。",
        "There is not enough information to answer.",
        "",
    ],
)
def test_mc_letter_refuses_to_guess(response):
    assert judges.extract_letter(response) is None


def test_mc_letter_explicit_answer_wins_over_refusal_words():
    """Regression from the 2026-11 real accuracy runs.

    The refusal regex used to be checked before the explicit answer, so a
    response like "... we don't have enough information about P(B) ... The answer
    is B." became ``format_failure`` even though the model committed to B.
    """
    assert judges.extract_letter("We don't have enough information about P(B). The answer is B.") == "B"
    assert judges.extract_letter("Cannot determine A and C. The answer is D.") == "D"
    assert judges.extract_letter("无法确定其他选项。答案是 C。") == "C"
    # Refusal-only responses still return None (no false positives).
    assert judges.extract_letter("Cannot determine; insufficient information.") is None
    # Markdown/option-prefixed explicit answers are also recognised.
    assert judges.extract_letter("The answer is **C**.") == "C"
    assert judges.extract_letter("Answer: Option B") == "B"


def test_mc_letter_gold_wrong_negative_control():
    row = {"judge": {"family": "mc_letter", "answer": "C"}}
    assert judges.evaluate(row, "答案是 C")["pass"] is True
    assert judges.evaluate(row, "答案是 B")["pass"] is False
    # "Cannot" contains a capital C but must not become a false positive.
    assert judges.evaluate(row, "Cannot determine")["pass"] is False


# ---------------------------------------------------------------- F1 / zh metrics

def test_token_f1_english():
    assert judges.token_f1("Buenos Aires", ["Buenos Aires"]) == pytest.approx(1.0)
    assert judges.token_f1("Buenos", ["Buenos Aires"]) == pytest.approx(2 / 3)


def test_token_f1_zh_uses_jieba_and_fullwidth_punctuation():
    assert judges.token_f1_zh("北京，中国", ["北京中国"]) == pytest.approx(1.0)
    assert judges.token_f1_zh("2.9%", ["2.9%。"]) == pytest.approx(1.0)


def test_rouge_l_zh():
    assert judges.rouge_l_zh("滚动捺印、平面捺印和局部捺印", ["滚动捺印、平面捺印和局部捺印"]) == pytest.approx(1.0)
    assert judges.rouge_l_zh("滚动捺印", ["滚动捺印、平面捺印和局部捺印"]) < 1.0


def test_normalized_match_routes_zh_metrics():
    row = {
        "judge": {
            "family": "normalized_match",
            "metric": "f1_zh",
            "answers": ["北京，中国"],
        }
    }
    assert judges.evaluate(row, "北京中国")["pass"] is True
    row["judge"]["metric"] = "rouge_zh"
    assert judges.evaluate(row, "北京，中国")["pass"] is True


# ---------------------------------------------------------------- numeric

def test_parse_number_accepts_fractions_and_commas():
    assert judges.parse_number("#### 2/5") == Fraction(2, 5)
    assert judges.parse_number("#### 114,200") == Fraction(114200)
    assert judges.parse_number("答案是 -3.5 元") == Fraction(-7, 2)


def test_numeric_takes_last_two_candidates():
    row = {"judge": {"family": "normalized_match", "metric": "numeric_exact", "answers": ["30"]}}
    # The final sentence mentions a second number; CMATH-style last-two matching
    # must still find the answer.
    assert judges.evaluate(row, "总价30元，其中3元是税")["pass"] is True
    assert judges.evaluate(row, "#### 30\n再检查两步")["pass"] is True


def test_numeric_fraction_equivalence_and_tolerance():
    frac = {"judge": {"family": "normalized_match", "metric": "numeric_exact", "answers": ["10/27"]}}
    assert judges.evaluate(frac, "#### 0.37037037037037035")["pass"] is True
    rounded = {
        "judge": {
            "family": "normalized_match",
            "metric": "numeric_exact",
            "answers": ["50.868"],
            "tolerance": "relative_1pct",
        }
    }
    assert judges.evaluate(rounded, "#### 50.87")["pass"] is True


def test_numeric_wrong_answer_fails():
    row = {"judge": {"family": "normalized_match", "metric": "numeric_exact", "answers": ["1884"]}}
    assert judges.evaluate(row, "#### 1885")["pass"] is False


# ---------------------------------------------------------------- code_exec

def _code_judge(test_code: str, prefix: str | None = None) -> dict:
    return {
        "family": "code_exec",
        "prompt_prefix": prefix,
        "test_code": test_code,
        "entry_point": "add",
        "timeout_s": 5,
    }


def test_code_exec_correct_solution_passes():
    row = {"judge": _code_judge("assert add(1, 2) == 3")}
    result = judges.evaluate(row, "```python\ndef add(a, b):\n    return a + b\n```")
    assert result["pass"] is True, result["error"]


def test_code_exec_wrong_solution_fails():
    row = {"judge": _code_judge("assert add(1, 2) == 3")}
    assert judges.evaluate(row, "def add(a, b):\n    return 0")["pass"] is False


@pytest.mark.parametrize("cheat", ["import sys\nsys.exit(0)", "import os\nos._exit(0)"])
def test_code_exec_early_exit_cannot_bypass_tests(cheat):
    row = {"judge": _code_judge("assert add(1, 2) == 3")}
    result = judges.evaluate(row, "```python\n" + cheat + "\n```")
    assert result["pass"] is False
    assert "sentinel" in result["error"] or "exit" in result["error"]


# ---------------------------------------------------------------- tool_call

def _tool_judge(ground_truth):
    return {"family": "tool_call", "ground_truth": ground_truth, "functions": []}


def test_irrelevance_accepts_natural_language_refusal():
    judge = _tool_judge([])
    assert judges.evaluate({"judge": judge}, "I cannot answer that with the given functions.")["pass"] is True
    assert judges.evaluate({"judge": judge}, "[]")["pass"] is True
    assert judges.evaluate({"judge": judge}, '{"name": "f", "arguments": {}}')["pass"] is False


def test_tool_call_nested_array_argument_parses():
    response = '{"name": "f", "arguments": {"xs": [1, 2]}}'
    assert judges.parse_calls(response) == [{"name": "f", "arguments": {"xs": [1, 2]}}]
    judge = _tool_judge([{"f": {"xs": [[1, 2]]}}])
    assert judges.evaluate({"judge": judge}, response)["pass"] is True


def test_tool_call_strings_are_standardised_like_bfcl():
    judge = _tool_judge([{"f": {"city": ["Beijing, China"]}}])
    assert judges.evaluate({"judge": judge}, '{"name": "f", "arguments": {"city": "beijing china"}}')["pass"] is True


def test_tool_call_extra_argument_fails():
    judge = _tool_judge([{"f": {"x": ["1"]}}])
    assert judges.evaluate({"judge": judge}, '{"name": "f", "arguments": {"x": "1", "y": 2}}')["pass"] is False


def test_tool_call_parallel_is_unordered():
    judge = _tool_judge([{"a": {"x": ["1"]}}, {"b": {"y": ["2"]}}])
    response = '[{"name": "b", "arguments": {"y": "2"}}, {"name": "a", "arguments": {"x": "1"}}]'
    assert judges.evaluate({"judge": judge}, response)["pass"] is True


# ---------------------------------------------------------------- ifeval

def test_ifeval_checker_loads_and_rejects_empty_response():
    from modelbench.domain.judges.instruction import checker_root

    try:
        root, _label = checker_root()
    except RuntimeError:  # pragma: no cover - vendored copy missing from wheel
        pytest.skip("official IFEval checker not present (dataset/raw nor vendored)")
    assert root.is_dir()
    if not judges.has_nltk_punkt():
        pytest.skip("NLTK punkt data missing; run tool/dataset/accuracy/fetch.py")
    ev = judges._load_ifeval()
    assert ev.instructions_registry.INSTRUCTION_DICT
    row = {
        "judge": {
            "family": "ifeval",
            "instruction_id_list": ["length_constraints:number_words"],
            "kwargs": [{"relation": "at least", "num_words": 3}],
        }
    }
    assert judges.evaluate(row, "")["pass"] is False
    assert judges.evaluate(row, "one two three")["pass"] is True


def test_vendored_ifeval_checker_is_shipped_with_the_package():
    """A plain install must never lose the instruction_following scenario."""
    assert judges.VENDORED_IFE_PKG.is_dir()
    for name in (
        "evaluation_lib.py",
        "instructions.py",
        "instructions_registry.py",
        "instructions_util.py",
    ):
        assert (judges.VENDORED_IFE_PKG / name).is_file()


def test_missing_assets_reports_nltk_data_only_when_ifeval_is_used():
    from modelbench.domain.judges import missing_assets

    assert missing_assets([]) == []
    if judges.has_nltk_punkt():
        assert missing_assets(["ifeval"]) == []
    else:
        assert any("NLTK punkt" in item for item in missing_assets(["ifeval"]))
    # Non-IFEval families never require the punkt data.
    assert missing_assets(["mc_letter"]) == []


def test_missing_assets_flags_absent_checker_and_punkt(monkeypatch, tmp_path):
    """The advisory preflight must see missing *files*, not just missing modules."""
    from modelbench.domain.judges import base as judge_base

    monkeypatch.setattr(judge_base, "IFE_PKG", tmp_path / "ifeval")
    monkeypatch.setattr(judge_base, "VENDORED_IFE_PKG", tmp_path / "no-vendored")
    monkeypatch.setattr(judge_base, "_NLTK_DATA_DIRS", (tmp_path / "nltk",))

    missing = judge_base.missing_assets(["ifeval"])
    assert any("IFEval checker" in item for item in missing)
    assert any("NLTK punkt" in item for item in missing)
    assert judge_base.missing_assets(["mc_letter"]) == []


def test_missing_assets_accepts_judge_contracts():
    from modelbench.domain.judges import missing_assets

    contract = {"family": "ifeval", "instruction_id_list": ["length_constraints:number_words"]}
    assert missing_assets([contract]) == missing_assets(["ifeval"])


def test_unknown_family_raises():
    with pytest.raises(ValueError, match="unknown judge family"):
        judges.evaluate({"judge": {"family": "nope"}}, "x")


# ---------------------------------------------------------------- WP-1 regression

def test_domain_evaluate_never_raises_on_broken_contracts():
    """The domain entry point (used by the accuracy runner) must never throw."""
    from modelbench.domain.judges import JUDGE_FAMILIES, evaluate as domain_evaluate

    assert JUDGE_FAMILIES == ("mc_letter", "ifeval", "normalized_match", "code_exec", "tool_call")
    broken = [
        {},                                            # missing judge contract
        {"judge": {}},                                 # missing family
        {"judge": {"family": "nope"}},                 # unknown family
        {"judge": {"family": "normalized_match"}},     # missing answers
        {"judge": {"family": "code_exec", "test_code": "raise SystemExit"}},
        {"judge": {"family": "tool_call"}},            # missing ground_truth
        {"judge": {"family": "ifeval"}},               # missing instruction list
    ]
    for row in broken:
        result = domain_evaluate(row, "some response")
        assert result.score == 0.0 and result.passed is False
    assert domain_evaluate({}, "x").detail["judge_error"] is True
    empty = domain_evaluate({"judge": {"family": "mc_letter", "answer": "A"}}, "")
    assert empty.detail["format_failure"] is True and empty.passed is False
    ok = domain_evaluate({"judge": {"family": "mc_letter", "answer": "A"}}, "The answer is A.")
    assert ok.to_dict()["pass"] is True and ok.detail["family"] == "mc_letter"


def test_code_exec_infrastructure_error_is_judge_error(monkeypatch):
    """A broken execution environment is a judge error, not a wrong answer.

    A model solution that hangs is a deterministic 0 (timeout); a missing/failed
    interpreter is an infrastructure failure and must be surfaced as
    ``judge_error`` so it stays visible (0 points, kept in the denominator).
    """
    from modelbench.domain.judges import code as code_judge
    from modelbench.domain.judges import evaluate as domain_evaluate

    def boom(*args, **kwargs):
        raise OSError("interpreter unavailable")

    monkeypatch.setattr(code_judge.subprocess, "run", boom)
    row = {
        "judge": {
            "family": "code_exec",
            "test_code": "assert True",
            "prompt_prefix": "def f():\n    pass\n",
        }
    }
    result = domain_evaluate(row, "def f():\n    pass\n")
    assert result.score == 0.0 and result.passed is False
    assert result.detail["judge_error"] is True
    assert "infrastructure" in result.detail["reason"]
