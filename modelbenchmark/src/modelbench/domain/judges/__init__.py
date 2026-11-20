"""Deterministic accuracy judges — the single source of truth (5 families).

    mc_letter        -> 0/1  (A-D option letter extraction)
    normalized_match -> F1 / zh F1 / zh ROUGE-L / numeric exact
    ifeval           -> 0/1  (official Google IFEval prompt-level strict)
    code_exec        -> 0/1  (pass@1 in an isolated subprocess)
    tool_call        -> 0/1  (BFCL possible-answer structural match)

``evaluate(row, response)`` is the only entry point used by the accuracy
orchestrator.  It **never raises**: any exception (unknown family, missing
optional dependency, malformed contract, checker crash) becomes
``JudgeResult(score=0.0, passed=False, detail={... "judge_error": True ...})``
so a broken judge can never abort a run.  Empty/unparseable responses are scored
0 and flagged with ``detail["format_failure"] = True``.

``tool/system/judges.py`` forwards to this package for backwards compatibility.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from .base import (
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
    clamp01,
    has_nltk_punkt,
    is_binary,
    missing_assets,
    missing_dependencies,
    nfkc,
    _nfkc,
)
from .code import extract_code, judge_code_exec
from .instruction import _load_ifeval, judge_ifeval
from .match import (
    _fraction,
    _normalize_squad,
    _normalize_zh_answer,
    _numbers_match,
    _rouge,
    judge_normalized_match,
    numeric_candidates,
    parse_number,
    rouge_l_zh,
    token_f1,
    token_f1_zh,
)
from .mc import extract_letter, judge_mc_letter
from .tool import (
    _call_args,
    _canon,
    _decode_json_any,
    _standardize_string,
    _value_ok,
    judge_tool_call,
    parse_calls,
)

_IMPL = {
    "mc_letter": judge_mc_letter,
    "normalized_match": judge_normalized_match,
    "ifeval": judge_ifeval,
    "code_exec": judge_code_exec,
    "tool_call": judge_tool_call,
}


def metric_for(family: str, judge: Optional[dict] = None) -> str:
    """Normalized metric name recorded under ``score.detail.metric``."""
    if family == "normalized_match":
        return str((judge or {}).get("metric") or "f1")
    return "accuracy"


def _format_failure(family: str, response: str, raw: dict) -> bool:
    """True when the response was empty or could not be parsed into a verdict.

    A zero score is *not* a format failure: a parseable but wrong answer is a
    normal incorrect answer.  A refusal counts as unparseable for ``mc_letter``.
    """
    if not str(response or "").strip():
        return True
    if family == "mc_letter":
        return raw.get("pred") is None
    if family == "normalized_match" and raw.get("metric") == "numeric_exact":
        return not raw.get("pred")
    if family == "tool_call":
        return raw.get("reason") == "unparseable_output"
    return False


def _judge_error(family: str, metric: str, started: float, exc: BaseException) -> JudgeResult:
    return JudgeResult(
        score=0.0,
        passed=False,
        detail={
            "family": family,
            "metric": metric,
            "pass": False,
            "format_failure": False,
            "judge_error": True,
            "reason": f"{type(exc).__name__}: {exc}",
            "judge_latency_ms": round((time.perf_counter() - started) * 1000, 3),
        },
    )


def evaluate(row: dict, response: str) -> JudgeResult:
    """Judge one response against ``row["judge"]``; never raises.

    The judge contract is read from ``row["judge"]`` (``family`` + family-specific
    fields).  All exceptions are converted into a ``judge_error`` result; the
    response text itself is preserved by the caller for evidence.
    """
    started = time.perf_counter()
    family = "unknown"
    metric = "accuracy"
    try:
        judge = (row or {})["judge"]
        family = str(judge["family"])
        metric = metric_for(family, judge)
        if family not in _IMPL:
            raise ValueError(f"unknown judge family {family}")
        raw = dict(_IMPL[family](response if response is not None else "", judge) or {})
        score = clamp01(raw.pop("score", 0.0))
        passed = bool(raw.pop("pass", score >= 1.0))
        detail: Dict[str, Any] = dict(raw)
        detail.update(
            {
                "family": family,
                "metric": metric,
                "pass": passed,
                "format_failure": _format_failure(family, response, detail),
                "judge_error": False,
            }
        )
        if "max_new_tokens" in judge:
            detail["max_new_tokens"] = judge["max_new_tokens"]
        detail["judge_latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return JudgeResult(score=score, passed=passed, detail=detail)
    except Exception as exc:  # noqa: BLE001 - a judge must never abort the runner
        return _judge_error(family, metric, started, exc)


__all__ = [
    # contract
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
    # path constants / legacy module surface
    "ROOT",
    "RAW_DIR",
    "IFE_PKG",
    "NLTK_DATA",
    "VENDORED_IFE_DIR",
    "VENDORED_IFE_PKG",
    "VENDORED_NLTK_DATA",
    "_nfkc",
    "_load_ifeval",
    # family judges + parsers
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
    # private helpers re-exported for legacy ``tool/system/judges.py`` users
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
