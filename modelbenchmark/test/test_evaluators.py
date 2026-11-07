"""Unit tests for the evaluator layer (``src/core/evaluators``).

Coverage:

* ``mc``: extraction from many phrasings, option-body false positives,
  ``num_choices`` limits, scoring + defensive behaviour.
* ``math``: ``####``/``\\boxed``/answer-phrase/last-number extraction, commas,
  currency, percent, negatives, tolerance matching.
* ``code``: real subprocess execution of correct/wrong/fenced/de-indented
  completions, syntax errors, timeouts, missing metadata, batch mode.
* ``instruction``: every implemented IFEval instruction family (18 ids), the
  unknown-id ``None`` contract and strict/partial scoring.
* ``agent``: argument normalisation, type-aware comparison, possible-answer
  lists, stringified JSON, tool-call structure scoring.
* ``search``: answer normalisation, EM/token-F1, empty trace degradation.

The last test module assert that *every* score entry point returns a
``modelbench.domain.models.Score`` (never raises) for an empty response.
"""

from __future__ import annotations

import json

import pytest

from modelbench.domain.evaluators import agent, code, instruction, math, mc, search
from modelbench.domain.models import AnswerType, RawResponse, Sample, Score, ToolCall


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def make_sample(
    *,
    answer=None,
    choices=None,
    metadata=None,
    answer_type: AnswerType = AnswerType.MC_LETTER,
    question: str = "q",
) -> Sample:
    return Sample(
        sample_id="s1",
        dataset="unit-test",
        scenario="unit-test",
        question=question,
        choices=choices,
        answer=answer,
        answer_type=answer_type,
        metadata=metadata or {},
    )


HUMANEVAL_PROMPT = 'def add(a, b):\n    """Return the sum of a and b."""\n'
HUMANEVAL_TEST = (
    "def check(candidate):\n"
    "    assert candidate(2, 3) == 5\n"
    "    assert candidate(0, 0) == 0\n"
)


def code_sample(**metadata) -> Sample:
    payload = {
        "prompt": HUMANEVAL_PROMPT,
        "test": HUMANEVAL_TEST,
        "entry_point": "add",
    }
    payload.update(metadata)
    return make_sample(answer_type=AnswerType.CODE, metadata=payload)


def constraint_sample(instruction_ids, kwargs_list) -> Sample:
    return make_sample(
        answer_type=AnswerType.CONSTRAINTS,
        metadata={
            "instruction_id_list": list(instruction_ids),
            "kwargs": list(kwargs_list),
        },
    )


# --------------------------------------------------------------------------- #
# mc: extraction
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text,expected",
    [
        ("The answer is C", "C"),
        ("Answer: C", "C"),
        ("answer: (C)", "C"),
        ("answer is (C)", "C"),
        ("The correct answer is option D.", "D"),
        ("The correct option is [B].", "B"),
        ("C is the correct answer", "C"),
        ("答案是C", "C"),
        ("答案：C", "C"),
        ("答案为（B）", "B"),
        ("选项C正确", "C"),
        ("应选B。", "B"),
        ("I choose A.", "A"),
        ("Let me pick D", "D"),
        ("**C**", "C"),
        ("'B'", "B"),
        ("Final answer:\nC", "C"),
        ("C", "C"),
        ("After some analysis.\n\nD.", "D"),
        ("The answer is not A, but C", "C"),
        ("Option A is incorrect, so the answer is B", "B"),
        ("A. lorem ipsum\nB. dolor sit amet\n\nThe answer is D", "D"),
        ("I do not know.", None),
        ("A good answer requires careful thought.", None),
        ("", None),
        ("...", None),
    ],
)
def test_extract_choice_letter(text, expected):
    assert mc.extract_choice_letter(text) == expected


def test_extract_choice_letter_num_choices_limits_letters():
    assert mc.extract_choice_letter("The answer is E", num_choices=5) == "E"
    assert mc.extract_choice_letter("The answer is E", num_choices=4) is None
    assert mc.extract_choice_letter("The answer is B", num_choices=2) == "B"
    assert mc.extract_choice_letter("The answer is C", num_choices=2) is None


def test_extract_choice_letter_does_not_raise_on_weird_input():
    assert mc.extract_choice_letter(None) is None  # type: ignore[arg-type]
    assert mc.extract_choice_letter("\x00\x01 混沌 " * 500) is None
    assert mc.extract_choice_letter("The answer is C", num_choices=0) == "C"


# --------------------------------------------------------------------------- #
# mc: scoring
# --------------------------------------------------------------------------- #


def test_score_mc_correct():
    sample = make_sample(answer="C", choices=["a", "b", "c", "d"])
    result = mc.score_mc(sample, RawResponse(text="So the answer is (C)."))
    assert result.correct is True
    assert result.score == 1.0
    assert result.detail["extracted"] == "C"
    assert result.detail["gold"] == "C"
    assert "first_200_chars" in result.detail


def test_score_mc_wrong():
    sample = make_sample(answer="C", choices=["a", "b", "c", "d"])
    result = mc.score_mc(sample, RawResponse(text="Answer: A"))
    assert result.correct is False
    assert result.score == 0.0
    assert result.detail["extracted"] == "A"


def test_score_mc_gold_lowercase_and_prefixed_answer():
    assert mc.score_mc(make_sample(answer="b"), RawResponse(text="Answer: B")).correct
    assert mc.score_mc(
        make_sample(answer="B. dolor sit amet"), RawResponse(text="Answer: B")
    ).correct


def test_score_mc_missing_gold():
    result = mc.score_mc(make_sample(answer=None), RawResponse(text="Answer: C"))
    assert result.correct is False
    assert "error" in result.detail


def test_score_mc_empty_response_does_not_raise():
    result = mc.score_mc(make_sample(answer="A"), RawResponse(text=""))
    assert isinstance(result, Score)
    assert result.correct is False


def test_score_mc_garbage_response_does_not_raise():
    result = mc.score_mc(
        make_sample(answer="A"), RawResponse(text="\x00" * 100_000)
    )
    assert isinstance(result, Score)
    assert result.correct is False


# --------------------------------------------------------------------------- #
# math: extraction
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text,expected",
    [
        ("#### 18", 18.0),
        ("#### -5", -5.0),
        ("#### 1,234.50", 1234.5),
        ("Steps... then 2+3=5.\n#### 18", 18.0),
        ("The answer is 18", 18.0),
        ("答案是：-5", -5.0),
        ("So the final answer is $3.50.", 3.5),
        ("结果是 1,234", 1234.0),
        ("\\boxed{42}", 42.0),
        ("The discount is 50%", 50.0),
        ("18", 18.0),
        ("The answer is -12.5.", -12.5),
        ("no number here", None),
        ("", None),
        (None, None),
    ],
)
def test_extract_final_number(text, expected):
    assert math.extract_final_number(text) == expected


def test_extract_final_number_uses_last_fallback_number():
    assert math.extract_final_number("I ate 3 apples then 7 pears, total 10") == 10.0


def test_extract_final_number_does_not_split_hyphen_ranges():
    # "18-20" must not be read as "-20".
    assert math.extract_final_number("between 18-20") == 20.0


def test_extract_final_number_huge_input_is_safe():
    assert math.extract_final_number("9" * 400) is None  # overflows to inf -> None
    assert math.extract_final_number("奶" * 10_000) is None


# --------------------------------------------------------------------------- #
# math: scoring
# --------------------------------------------------------------------------- #


def test_score_numeric_from_gsm8k_marker():
    sample = make_sample(answer="#### 18", answer_type=AnswerType.NUMERIC)
    result = math.score_numeric(
        sample, RawResponse(text="The answer is 18. #### 18")
    )
    assert result.correct is True
    assert result.score == 1.0
    assert result.detail["extracted"] == 18.0
    assert result.detail["gold"] == 18.0


def test_score_numeric_from_metadata_gold_number():
    sample = make_sample(
        answer="irrelevant",
        answer_type=AnswerType.NUMERIC,
        metadata={"gold_number": 42},
    )
    assert math.score_numeric(sample, RawResponse(text="#### 42")).correct
    assert math.score_numeric(sample, RawResponse(text="#### 42.0000001")).correct


def test_score_numeric_wrong_and_tolerance():
    sample = make_sample(answer="#### 42", answer_type=AnswerType.NUMERIC)
    assert math.score_numeric(sample, RawResponse(text="#### 43")).correct is False
    assert math.score_numeric(sample, RawResponse(text="#### 42.000001")).correct


def test_score_numeric_missing_gold_and_no_number():
    missing_gold = make_sample(answer=None, answer_type=AnswerType.NUMERIC)
    result = math.score_numeric(missing_gold, RawResponse(text="#### 1"))
    assert result.correct is False and "error" in result.detail

    sample = make_sample(answer="#### 42", answer_type=AnswerType.NUMERIC)
    no_number = math.score_numeric(sample, RawResponse(text="I have no idea"))
    assert no_number.correct is False
    assert no_number.detail["extracted"] is None


def test_score_numeric_empty_response_does_not_raise():
    sample = make_sample(answer="#### 42", answer_type=AnswerType.NUMERIC)
    assert isinstance(math.score_numeric(sample, RawResponse(text="")), Score)


# --------------------------------------------------------------------------- #
# code: execution
# --------------------------------------------------------------------------- #


def test_score_code_correct_completion_passes():
    result = code.score_code(code_sample(), RawResponse(text="    return a + b"))
    assert result.correct is True
    assert result.score == 1.0
    assert result.detail["exit_code"] == 0
    assert result.detail["timed_out"] is False


def test_score_code_wrong_completion_fails_with_stderr():
    result = code.score_code(code_sample(), RawResponse(text="    return a - b"))
    assert result.correct is False
    assert result.score == 0.0
    assert result.detail["exit_code"] != 0
    assert "AssertionError" in result.detail["stderr"]


def test_score_code_markdown_fence_is_unwrapped():
    completion = "```python\ndef add(a, b):\n    return a + b\n```"
    result = code.score_code(code_sample(), RawResponse(text=completion))
    assert result.correct is True
    assert result.detail["mode"] == "replaced_signature"


def test_score_code_full_function_completion_passes():
    result = code.score_code(
        code_sample(),
        RawResponse(text="def add(a, b):\n    return a + b\n"),
    )
    assert result.correct is True


def test_score_code_deindented_body_is_indented():
    result = code.score_code(code_sample(), RawResponse(text="return a + b"))
    assert result.correct is True
    assert result.detail["used_indent_fallback"] is True


def test_score_code_syntax_error_fails():
    result = code.score_code(code_sample(), RawResponse(text="    return (a + b"))
    assert result.correct is False
    assert result.detail["error"] == "syntax_error"
    assert result.detail["syntax_error"]


def test_score_code_timeout_fails(monkeypatch):
    monkeypatch.setattr(code, "EXEC_TIMEOUT_S", 1.0)
    result = code.score_code(
        code_sample(), RawResponse(text="    while True: pass")
    )
    assert result.correct is False
    assert result.detail["timed_out"] is True
    assert "timeout" in result.detail["error"]


def test_score_code_missing_metadata_and_empty_response():
    empty_meta = make_sample(
        answer_type=AnswerType.CODE, metadata={"prompt": HUMANEVAL_PROMPT}
    )
    result = code.score_code(empty_meta, RawResponse(text="    return a + b"))
    assert result.correct is False
    assert "entry_point" in result.detail["error"]

    empty_completion = code.score_code(code_sample(), RawResponse(text=""))
    assert empty_completion.correct is False
    assert empty_completion.detail["error"] == "empty completion"


def test_score_code_captures_stdout_but_caps_it():
    result = code.score_code(
        code_sample(),
        RawResponse(text="    print('x' * 5000)\n    return a + b"),
    )
    assert result.correct is True
    assert len(result.detail["stdout"]) <= 500


def test_score_code_batch():
    scores = code.score_code_batch(
        [code_sample(), code_sample()],
        [RawResponse(text="    return a + b"), RawResponse(text="    return a - b")],
    )
    assert [score.correct for score in scores] == [True, False]


# --------------------------------------------------------------------------- #
# instruction: individual checks
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "instruction_id,text,kwargs,expected",
    [
        ("keywords:existence", "The cat and the dog.", {"keywords": ["cat", "dog"]}, True),
        ("keywords:existence", "The cat and the dog.", {"keywords": ["cat", "bird"]}, False),
        (
            "keywords:frequency",
            "aa bb aa aa",
            {"keyword": "aa", "frequency": 3, "relation": "at least"},
            True,
        ),
        (
            "keywords:frequency",
            "aa bb",
            {"keyword": "aa", "frequency": 3, "relation": "at least"},
            False,
        ),
        ("keywords:forbidden_words", "clean text", {"forbidden_words": ["dirty"]}, True),
        ("keywords:forbidden_words", "dirty text", {"forbidden_words": ["dirty"]}, False),
        (
            "keywords:letter_frequency",
            "banana",
            {"letter": "a", "let_frequency": 3, "let_relation": "at least"},
            True,
        ),
        (
            "language:response_language",
            "This is a simple answer in English with the words and the text.",
            {"language": "en"},
            True,
        ),
        (
            "language:response_language",
            "Bonjour, je suis très heureux de vous aider avec le texte et la réponse.",
            {"language": "fr"},
            True,
        ),
        ("language:response_language", "这是一个中文回答。", {"language": "zh"}, True),
        (
            "length_constraints:number_words",
            "one two three four",
            {"num_words": 3, "relation": "at least"},
            True,
        ),
        (
            "length_constraints:number_words",
            "one two three four",
            {"num_words": 3, "relation": "less than"},
            False,
        ),
        (
            "length_constraints:number_sentences",
            "One. Two! Three?",
            {"num_sentences": 3, "relation": "at least"},
            True,
        ),
        (
            "length_constraints:number_sentences",
            "One. Two.",
            {"num_sentences": 3, "relation": "at least"},
            False,
        ),
        (
            "length_constraints:number_paragraphs",
            "para one\n\npara two\n\npara three",
            {"num_paragraphs": 3},
            True,
        ),
        (
            "length_constraints:number_paragraphs",
            "single paragraph",
            {"num_paragraphs": 3},
            False,
        ),
        (
            "length_constraints:nth_paragraph_first_word",
            "Intro here\n\nHowever this is second\n\nLast one",
            {"num_paragraphs": 3, "nth_paragraph": 2, "first_word": "However"},
            True,
        ),
        (
            "length_constraints:nth_paragraph_first_word",
            "Intro here\n\nSecond is this\n\nLast one",
            {"num_paragraphs": 3, "nth_paragraph": 2, "first_word": "However"},
            False,
        ),
        (
            "detectable_format:number_bullet_lists",
            "* one\n* two\n* three",
            {"num_bullets": 3},
            True,
        ),
        (
            "detectable_format:number_bullet_lists",
            "one\ntwo\nthree",
            {"num_bullets": 3},
            False,
        ),
        (
            "detectable_format:number_highlighted_sections",
            "Here is *one* and *two* highlights.",
            {"num_highlights": 2},
            True,
        ),
        (
            "detectable_format:multiple_sections",
            "SECTION 1 aaa SECTION 2 bbb SECTION 3 ccc",
            {"section_spliter": "SECTION", "num_sections": 3},
            True,
        ),
        (
            "detectable_format:multiple_sections",
            "no markers here",
            {"section_spliter": "SECTION", "num_sections": 3},
            False,
        ),
        ("change_case:english_capital", "THIS IS ALL CAPS", {}, True),
        ("change_case:english_capital", "This is not", {}, False),
        ("change_case:english_lowercase", "this is lower", {}, True),
        ("change_case:english_lowercase", "This Is Not", {}, False),
        ("startend:end_checker", "The end is here. Done", {"end_phrase": "Done"}, True),
        ("startend:end_checker", "Done is here.", {"end_phrase": "Done"}, False),
        ("punctuation:no_comma", "no commas here", {}, True),
        ("punctuation:no_comma", "a, b", {}, False),
        (
            "combination:two_responses",
            "First answer\n******\nSecond answer",
            {},
            True,
        ),
        ("combination:two_responses", "only one answer", {}, False),
        (
            "combination:repeat_prompt",
            "Explain gravity. Gravity is a force.",
            {"prompt_to_repeat": "Explain gravity."},
            True,
        ),
        (
            "combination:repeat_prompt",
            "Gravity is a force.",
            {"prompt_to_repeat": "Explain gravity."},
            False,
        ),
    ],
)
def test_check_instruction(instruction_id, text, kwargs, expected):
    assert instruction.check_instructions(text, [instruction_id], [kwargs]) == [expected]


def test_check_instructions_unknown_id_is_none():
    results = instruction.check_instructions(
        "whatever", ["definitely:not-an-instruction"], [{}]
    )
    assert results == [None]


def test_check_instructions_malformed_kwargs_are_not_judgeable():
    # Missing required kwargs -> None (excluded from the denominator), no raise.
    results = instruction.check_instructions(
        "some text",
        ["length_constraints:number_words", "startend:end_checker"],
        [{}, {}],
    )
    assert results == [None, None]


def test_check_instructions_result_length_matches_ids():
    results = instruction.check_instructions(
        "hello", ["punctuation:no_comma", "keywords:existence"], [{}]
    )
    assert len(results) == 2
    assert results[1] is None  # kwargs missing -> not judgeable


def test_supported_instructions_covers_required_families():
    supported = set(instruction.supported_instructions())
    assert len(supported) >= 14
    for instruction_id in (
        "keywords:existence",
        "keywords:frequency",
        "keywords:forbidden_words",
        "keywords:letter_frequency",
        "language:response_language",
        "length_constraints:number_words",
        "length_constraints:number_sentences",
        "length_constraints:number_paragraphs",
        "length_constraints:nth_paragraph_first_word",
        "detectable_format:number_bullet_lists",
        "detectable_format:number_highlighted_sections",
        "detectable_format:multiple_sections",
        "change_case:english_capital",
        "change_case:english_lowercase",
        "startend:end_checker",
        "punctuation:no_comma",
        "combination:two_responses",
        "combination:repeat_prompt",
    ):
        assert instruction_id in supported


# --------------------------------------------------------------------------- #
# instruction: scoring
# --------------------------------------------------------------------------- #


def test_score_instructions_all_pass():
    sample = constraint_sample(
        ["keywords:existence", "punctuation:no_comma"],
        [{"keywords": ["cat"]}, {}],
    )
    result = instruction.score_instructions(
        sample, RawResponse(text="The cat sat down")
    )
    assert result.correct is True
    assert result.score == 1.0
    assert result.detail["n_passed"] == 2
    assert result.detail["n_judged"] == 2
    assert result.detail["n_unknown"] == 0


def test_score_instructions_partial_pass_is_not_correct():
    sample = constraint_sample(
        ["keywords:existence", "punctuation:no_comma"],
        [{"keywords": ["cat"]}, {}],
    )
    result = instruction.score_instructions(
        sample, RawResponse(text="The cat sat, then left")
    )
    assert result.correct is False
    assert result.score == 0.5
    assert result.detail["n_passed"] == 1


def test_score_instructions_unknown_ids_are_excluded():
    sample = constraint_sample(
        ["keywords:existence", "unknown:instruction"],
        [{"keywords": ["cat"]}, {}],
    )
    result = instruction.score_instructions(sample, RawResponse(text="The cat sat"))
    assert result.correct is True
    assert result.detail["n_judged"] == 1
    assert result.detail["n_unknown"] == 1


def test_score_instructions_no_judgeable_instruction():
    sample = constraint_sample(["unknown:instruction"], [{}])
    result = instruction.score_instructions(sample, RawResponse(text="anything"))
    assert result.correct is False
    assert result.score == 0.0
    assert "error" in result.detail


def test_score_instructions_list_of_dicts_form():
    sample = make_sample(
        answer_type=AnswerType.CONSTRAINTS,
        metadata={
            "instruction_id_list": [
                {"instruction_id": "keywords:existence", "kwargs": {"keywords": ["cat"]}}
            ]
        },
    )
    result = instruction.score_instructions(sample, RawResponse(text="the cat"))
    assert result.correct is True


def test_score_instructions_empty_response_does_not_raise():
    sample = constraint_sample(["keywords:existence"], [{"keywords": ["cat"]}])
    result = instruction.score_instructions(sample, RawResponse(text=""))
    assert isinstance(result, Score)
    assert result.correct is False


# --------------------------------------------------------------------------- #
# agent: normalize_args / args_match
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"a": 1}', {"a": 1.0}),
        ('[1, 2]', [1.0, 2.0]),
        ("  hello  ", "hello"),
        ("not json", "not json"),
        ({"b": " x ", "a": [1, True]}, {"b": "x", "a": [1.0, True]}),
        (None, None),
        (True, True),
        (3, 3.0),
    ],
)
def test_normalize_args(raw, expected):
    assert agent.normalize_args(raw) == expected


@pytest.mark.parametrize(
    "gold,pred,expected",
    [
        ({"count": 2}, {"count": 2.0}, True),
        ({"count": 2}, {"count": 3}, False),
        ({"flag": True}, {"flag": True}, True),
        ({"flag": True}, {"flag": 1}, False),
        ({"flag": 1}, {"flag": True}, False),
        ({"city": " Paris "}, {"city": "Paris"}, True),
        ({"city": "Paris"}, {"city": "paris"}, False),
        ({"items": [1, 2]}, {"items": [1, 2.0]}, True),
        ({"items": [1, 2]}, {"items": [2, 1]}, False),
        ({"city": ["Paris", "Lyon"]}, {"city": "Lyon"}, True),
        ({"city": ["Paris", "Lyon"]}, {"city": "Berlin"}, False),
        ({"nested": {"a": {"b": 1}}}, {"nested": {"a": {"b": 1.0}}}, True),
        ({"a": 1}, {"a": 1, "b": 2}, False),
        ({"a": 1}, {"b": 1}, False),
        ({"a": 1}, '{"a": 1}', True),
        ({"x": None}, {"x": None}, True),
        ({}, {}, True),
        ({"a": [1, 2]}, '{"a": [1, 2]}', True),
    ],
)
def test_args_match(gold, pred, expected):
    assert agent.args_match(gold, pred) is expected


def test_args_match_never_raises_on_cycles():
    gold = {"a": 1}
    pred: dict = {}
    pred["self"] = pred  # cyclic structure must not blow up recursion unhandled
    assert agent.args_match(gold, pred) is False


# --------------------------------------------------------------------------- #
# agent: score_tool_calls
# --------------------------------------------------------------------------- #


def tool_sample(gold_calls) -> Sample:
    return make_sample(
        answer_type=AnswerType.TOOL_CALL, metadata={"gold_calls": gold_calls}
    )


def test_score_tool_calls_match_single_call():
    sample = tool_sample([{"name": "get_weather", "arguments": {"city": "Paris"}}])
    response = RawResponse(
        tool_calls=[ToolCall(name="get_weather", arguments='{"city": "Paris"}')]
    )
    result = agent.score_tool_calls(sample, response)
    assert result.correct is True
    assert result.score == 1.0
    assert result.detail["n_gold"] == 1
    assert "gold" in result.detail and "pred" in result.detail


def test_score_tool_calls_empty_gold_no_call_passes():
    result = agent.score_tool_calls(tool_sample([]), RawResponse())
    assert result.correct is True
    assert "no call expected" in result.detail["reason"]


def test_score_tool_calls_empty_gold_unexpected_call_fails():
    response = RawResponse(tool_calls=[ToolCall(name="x", arguments="{}")])
    result = agent.score_tool_calls(tool_sample([]), response)
    assert result.correct is False


def test_score_tool_calls_name_mismatch_fails():
    sample = tool_sample([{"name": "get_weather", "arguments": {"city": "Paris"}}])
    response = RawResponse(
        tool_calls=[ToolCall(name="get_temperature", arguments='{"city": "Paris"}')]
    )
    assert agent.score_tool_calls(sample, response).correct is False


def test_score_tool_calls_argument_mismatch_fails():
    sample = tool_sample([{"name": "get_weather", "arguments": {"city": "Paris"}}])
    response = RawResponse(
        tool_calls=[ToolCall(name="get_weather", arguments='{"city": "Berlin"}')]
    )
    assert agent.score_tool_calls(sample, response).correct is False


def test_score_tool_calls_count_mismatch_fails():
    sample = tool_sample(
        [
            {"name": "a", "arguments": {}},
            {"name": "b", "arguments": {}},
        ]
    )
    response = RawResponse(tool_calls=[ToolCall(name="a", arguments="{}")])
    assert agent.score_tool_calls(sample, response).correct is False


def test_score_tool_calls_possible_answers():
    sample = tool_sample([{"name": "pick", "arguments": {"city": ["Paris", "Lyon"]}}])
    response = RawResponse(
        tool_calls=[ToolCall(name="pick", arguments='{"city": "Lyon"}')]
    )
    assert agent.score_tool_calls(sample, response).correct is True


def test_score_tool_calls_unparseable_arguments_fails():
    sample = tool_sample([{"name": "f", "arguments": {"a": 1}}])
    response = RawResponse(tool_calls=[ToolCall(name="f", arguments="{not json")])
    result = agent.score_tool_calls(sample, response)
    assert result.correct is False
    assert "unparseable" in result.detail["reason"]


def test_score_tool_calls_gold_metadata_as_json_string():
    sample = tool_sample(
        json.dumps([{"name": "f", "arguments": {"a": 1}}])
    )
    response = RawResponse(tool_calls=[ToolCall(name="f", arguments='{"a": 1}')])
    assert agent.score_tool_calls(sample, response).correct is True


def test_score_tool_calls_detail_is_truncated():
    huge = [{"name": "f", "arguments": {"a": "x" * 5000}}]
    result = agent.score_tool_calls(
        tool_sample(huge),
        RawResponse(tool_calls=[ToolCall(name="f", arguments='{"a": "x"}')]),
    )
    assert len(result.detail["gold"]) <= 1200


def test_score_tool_calls_empty_response_does_not_raise():
    sample = tool_sample([{"name": "f", "arguments": {}}])
    result = agent.score_tool_calls(sample, RawResponse(text=""))
    assert isinstance(result, Score)
    assert result.correct is False


# --------------------------------------------------------------------------- #
# search: normalisation / EM / F1
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("The  Beatles!", "beatles"),
        ("New York City", "new york city"),
        ("a, b. c!", "b c"),
        ("", ""),
        (None, ""),
        ("  Paris  ", "paris"),
        ("中文，回答。", "中文 回答"),
    ],
)
def test_normalize_answer(raw, expected):
    assert search.normalize_answer(raw) == expected


@pytest.mark.parametrize(
    "prediction,gold,expected",
    [
        ("new york", "new york city", pytest.approx(0.8)),
        ("paris", "paris", 1.0),
        ("", "", 1.0),
        ("x", "", 0.0),
        ("red apple", "green apple", pytest.approx(0.5)),
    ],
)
def test_token_f1(prediction, gold, expected):
    assert search.token_f1(prediction, gold) == expected


def test_score_search_exact_match():
    sample = make_sample(answer="Paris", answer_type=AnswerType.SEARCH_ANSWER)
    result = search.score_search(sample, RawResponse(text="The Paris"))
    assert result.correct is True
    assert result.score == 1.0
    assert result.detail["em"] == 1.0
    assert result.detail["f1"] == 1.0


def test_score_search_partial_credit_but_not_correct():
    sample = make_sample(
        answer="New York City", answer_type=AnswerType.SEARCH_ANSWER
    )
    result = search.score_search(sample, RawResponse(text="New York"))
    assert result.correct is False
    assert result.detail["em"] == 0.0
    assert result.detail["f1"] == pytest.approx(0.8)


def test_score_search_no_overlap():
    sample = make_sample(answer="Paris", answer_type=AnswerType.SEARCH_ANSWER)
    result = search.score_search(sample, RawResponse(text="Berlin"))
    assert result.correct is False
    assert result.detail["f1"] == 0.0


def test_score_search_missing_gold():
    sample = make_sample(answer=None, answer_type=AnswerType.SEARCH_ANSWER)
    result = search.score_search(sample, RawResponse(text="Paris"))
    assert result.correct is False
    assert "error" in result.detail


def test_score_search_accepts_pre_extracted_answer_text():
    # The search scenario builds RawResponse(text=extracted_answer).
    sample = make_sample(answer="The Beatles", answer_type=AnswerType.SEARCH_ANSWER)
    result = search.score_search(sample, RawResponse(text="beatles"))
    assert result.correct is True


def test_score_search_empty_response_does_not_raise():
    sample = make_sample(answer="Paris", answer_type=AnswerType.SEARCH_ANSWER)
    result = search.score_search(sample, RawResponse(text=""))
    assert isinstance(result, Score)
    assert result.correct is False


def test_score_search_with_trace_empty_trace_degrades():
    sample = make_sample(answer="Paris", answer_type=AnswerType.SEARCH_ANSWER)
    response = RawResponse(text="Paris")
    plain = search.score_search(sample, response)
    traced = search.score_search_with_trace(sample, response, [])
    assert traced.correct == plain.correct
    assert traced.score == plain.score
    assert traced.detail == plain.detail


def test_score_search_with_trace_records_trace():
    sample = make_sample(answer="Paris", answer_type=AnswerType.SEARCH_ANSWER)
    trace = [
        {"tool": "search", "query": "capital of France"},
        {"tool": "lookup", "query": "Paris"},
    ]
    result = search.score_search_with_trace(
        sample, RawResponse(text="Paris"), trace
    )
    assert result.correct is True
    assert result.detail["trace_len"] == 2
    assert len(result.detail["trace"]) == 2


# --------------------------------------------------------------------------- #
# Cross-cutting: every score entry point tolerates an empty response
# --------------------------------------------------------------------------- #


def _empty_response_cases():
    return [
        pytest.param(
            mc.score_mc,
            make_sample(answer="A", choices=["a", "b", "c", "d"]),
            id="score_mc",
        ),
        pytest.param(
            math.score_numeric,
            make_sample(answer="#### 1", answer_type=AnswerType.NUMERIC),
            id="score_numeric",
        ),
        pytest.param(code.score_code, code_sample(), id="score_code"),
        pytest.param(
            instruction.score_instructions,
            constraint_sample(["punctuation:no_comma"], [{}]),
            id="score_instructions",
        ),
        pytest.param(
            agent.score_tool_calls,
            tool_sample([]),
            id="score_tool_calls",
        ),
        pytest.param(
            search.score_search,
            make_sample(answer="x", answer_type=AnswerType.SEARCH_ANSWER),
            id="score_search",
        ),
    ]


@pytest.mark.parametrize("score_fn,sample", _empty_response_cases())
def test_empty_response_never_raises(score_fn, sample):
    result = score_fn(sample, RawResponse(text=""))
    assert isinstance(result, Score)
    assert 0.0 <= result.score <= 1.0
    assert isinstance(result.detail, dict)


@pytest.mark.parametrize("score_fn,sample", _empty_response_cases())
def test_garbage_response_never_raises(score_fn, sample):
    garbage = ("\x00\x01\x02 混 沌 � " * 2_000)
    result = score_fn(sample, RawResponse(text=garbage))
    assert isinstance(result, Score)
    assert 0.0 <= result.score <= 1.0
