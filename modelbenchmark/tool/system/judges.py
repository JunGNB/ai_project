#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility forwarder for the deterministic accuracy judges.

The single source of truth is :mod:`modelbench.domain.judges` (five families:
``mc_letter`` / ``normalized_match`` / ``ifeval`` / ``code_exec`` / ``tool_call``).
This module keeps the historical ``tool.system.judges`` surface working for
``tool/system/validate_judges.py`` and ``tool/system/pilot.py``:

* :func:`evaluate` returns the flat legacy dict (``score`` / ``pass`` / detail
  keys) produced by :meth:`JudgeResult.to_dict`;
* every public/private helper name of the old module is re-exported;
* an **unknown judge family** still raises ``ValueError`` here (the legacy
  contract asserted by ``test/test_judges.py``).  Judge *execution* failures are
  never raised: the domain layer converts them into ``judge_error`` results.
"""

from __future__ import annotations

from typing import Any, Dict

from modelbench.domain import judges as _impl
from modelbench.domain.judges import (  # noqa: F401  (re-exported API surface)
    BINARY_FAMILIES,
    BINARY_METRICS,
    IFE_PKG,
    JUDGE_FAMILIES,
    NLTK_DATA,
    RAW_DIR,
    ROOT,
    VENDORED_IFE_DIR,
    VENDORED_IFE_PKG,
    VENDORED_NLTK_DATA,
    JudgeResult,
    _call_args,
    _canon,
    _decode_json_any,
    _fraction,
    _load_ifeval,
    _nfkc,
    _normalize_squad,
    _normalize_zh_answer,
    _numbers_match,
    _rouge,
    _standardize_string,
    _value_ok,
    clamp01,
    evaluate as _domain_evaluate,
    has_nltk_punkt,
    extract_code,
    extract_letter,
    is_binary,
    judge_code_exec,
    judge_ifeval,
    judge_mc_letter,
    judge_normalized_match,
    judge_tool_call,
    metric_for,
    missing_assets,
    missing_dependencies,
    nfkc,
    numeric_candidates,
    parse_calls,
    parse_number,
    rouge_l_zh,
    token_f1,
    token_f1_zh,
)


def evaluate(row: Dict[str, Any], response: str) -> Dict[str, Any]:
    """Legacy dict-shaped entry point (compatibility only).

    Unknown/missing families keep the historical ``ValueError``; every other
    failure is a ``judge_error`` dict and never raises.
    """
    family = str(((row or {}).get("judge") or {}).get("family"))
    if family not in JUDGE_FAMILIES:
        raise ValueError(f"unknown judge family {family}")
    return _impl.evaluate(row, response).to_dict()


__all__ = [
    "JudgeResult",
    "JUDGE_FAMILIES",
    "BINARY_FAMILIES",
    "BINARY_METRICS",
    "evaluate",
    "metric_for",
    "missing_dependencies",
    "missing_assets",
    "has_nltk_punkt",
    "is_binary",
    "clamp01",
    "nfkc",
    "ROOT",
    "RAW_DIR",
    "IFE_PKG",
    "NLTK_DATA",
    "VENDORED_IFE_DIR",
    "VENDORED_IFE_PKG",
    "VENDORED_NLTK_DATA",
    "_nfkc",
    "_load_ifeval",
    "extract_letter",
    "judge_mc_letter",
    "token_f1",
    "token_f1_zh",
    "rouge_l_zh",
    "numeric_candidates",
    "parse_number",
    "judge_normalized_match",
    "judge_ifeval",
    "extract_code",
    "judge_code_exec",
    "parse_calls",
    "judge_tool_call",
    "_fraction",
    "_numbers_match",
    "_normalize_squad",
    "_normalize_zh_answer",
    "_rouge",
    "_call_args",
    "_canon",
    "_decode_json_any",
    "_standardize_string",
    "_value_ok",
]
