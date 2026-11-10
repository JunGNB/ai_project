#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prompt templates and gold-answer canonicalisation for the accuracy builder.

LongBench v1 prompts are read from the upstream ``dataset2prompt.json`` (fetched
into ``dataset/raw/longbench_v1/prompts/``) so the official wording is preserved.
Everything else is defined here to keep the built rows byte-stable.
"""
from __future__ import annotations

import re
from typing import List

MC_EN = (
    "Answer the following multiple-choice question. Reply with only the option letter "
    "(A, B, C, or D); if you reason, end with 'The answer is <letter>.'.\n\n"
    "Question: {q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\n\nAnswer:"
)
MC_ZH = (
    "请回答下面的单项选择题，只输出选项字母（A、B、C 或 D）；如需推理，请在最后写“答案是 <字母>”。\n\n"
    "题目：{q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\n\n答案："
)

GSM8K_PROMPT = (
    "Solve the following math problem step by step. On the last line, write the "
    "final answer as:\n#### <number>\n\nProblem: {q}\nSolution:"
)
CMATH_PROMPT = "请逐步解答下面的数学题。最后一行写：\n#### <数值>\n\n题目：{q}\n解答："

HUMANEVAL_PROMPT = (
    "Complete the following Python function. Output only the complete function code.\n\n{prompt}"
)
#: MBPP sanitized prompts only contain the task description; the official
#: few-shot wording names the entry point and shows tests.  Without both, the
#: model cannot know which function name to define (124/125 rows were
#: unscoreable in the 2026-11 audit), so the rendered prompt carries the name
#: extracted from ``code`` and the first official tests.
MBPP_PROMPT = (
    "Write a Python function for the following task. Output only the Python code, "
    "without explanations and without test code.\n\n"
    "Task: {prompt}\n\n"
    "Your code must be named `{entry_point}`. The code must satisfy these tests:\n"
    "{tests}"
)
#: Number of official MBPP tests quoted in the prompt (enough to pin the calling
#: convention without turning the prompt into a full test transcript).
MBPP_PROMPT_TESTS = 3

BFCL_PROMPT = (
    "You are a function-calling assistant. Reply with JSON only.\n\n"
    "Available functions (JSON Schema):\n{functions}\n\n"
    "User request: {question}\n\n"
    "Rules: for one call output {{\"name\": \"<function>\", \"arguments\": {{...}}}}; "
    "for several calls output a JSON array of such objects; "
    "if no function is appropriate output [].\n"
)

SEARCH_PROMPT = (
    "Read the paragraphs and answer the question with a short answer "
    "(a word or phrase; yes/no when applicable).\n\n"
    "{paragraphs}\n\nQuestion: {question}\nAnswer:"
)

_GSM_RE = re.compile(r"####\s*([-+]?[\d,]+(?:\.\d+)?)")
_NUM_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")
_NUM_OK = re.compile(r"[-+]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?")


def gsm8k_final(answer: str) -> str:
    match = _GSM_RE.search(answer)
    if match:
        return match.group(1)
    numbers = _NUM_RE.findall(answer)
    if not numbers:
        raise ValueError("GSM8K answer has no numeric value")
    return numbers[-1]


def cmath_final(golden: str) -> str:
    value = golden.strip().replace(",", "")
    if not _NUM_OK.fullmatch(value):
        numbers = _NUM_RE.findall(value)
        if not numbers:
            raise ValueError(f"CMATH golden is not numeric: {golden!r}")
        value = numbers[-1]
    return value


_MBPP_DEF_RE = re.compile(r"(?m)^\s*def\s+([A-Za-z_]\w*)\s*\(")


def mbpp_entry_point(code: str) -> str:
    """Extract the entry point from an MBPP sanitized reference solution.

    Every sanitized MBPP ``code`` field starts with the top-level ``def`` that
    the unit tests call; using it keeps ``judge.entry_point`` and the prompt in
    sync without parsing the tests.
    """
    match = _MBPP_DEF_RE.search(code or "")
    if match is None:
        raise ValueError("MBPP reference code has no top-level function definition")
    return match.group(1)


def mbpp_prompt(task: str, entry_point: str, tests: List[str], num_tests: int = MBPP_PROMPT_TESTS) -> str:
    """Render the MBPP prompt: task + required function name + first official tests."""
    sample_tests = "\n".join(str(t).strip() for t in list(tests)[:num_tests] if str(t).strip())
    if not sample_tests:
        raise ValueError(f"MBPP task for {entry_point!r} has no tests to quote")
    return MBPP_PROMPT.format(prompt=task, entry_point=entry_point, tests=sample_tests)
