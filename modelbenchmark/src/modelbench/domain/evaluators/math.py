"""Numeric-answer evaluator (GSM8K).

GSM8K gold answers end with ``#### <number>``; model outputs are free-form
chain-of-thought answers that may contain many intermediate numbers.  Extraction
is therefore priority-ordered:

1. the number after the ``####`` marker,
2. a ``\\boxed{...}`` value,
3. the number directly after an explicit answer phrase
   (``final answer is`` / ``答案是`` / ``结果为`` ...),
4. the last number mentioned anywhere in the text (fallback).

Commas (``1,234``), currency symbols (``$3.50``), percent signs (``50%``) and a
leading minus sign (``-5``) are handled.  Percentages keep their numeric value
(``50%`` -> ``50.0``) to match how GSM8K gold numbers are stored.

Pure function: no IO, no randomness, never raises out of the public entry points.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any, Optional

from ..models import RawResponse, Sample, Score
from .base import make_score

#: A single number token; the tier-3 fallback adds a lookbehind so that
#: "18-20" does not become "-20".
_NUMBER_CORE = r"(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"

_NUMBER_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_.,])[-+]?" + _NUMBER_CORE
)

_CURRENCY = r"[$€£¥￥]"
_PERCENT = r"%?"

#: Priority-ordered explicit extraction patterns; group(1) is the number.
#: Sign may precede or follow the currency symbol ("-$1,234.50" / "$-1,234.50").
_SIGNED_NUMBER = r"([-+]?\s*" + _CURRENCY + r"?\s*[-+]?\s*" + _NUMBER_CORE + r")"

_EXPLICIT_PATTERNS = (
    # 1. GSM8K "#### 18" (also "#### -$1,234.50")
    re.compile(r"####\s*" + _SIGNED_NUMBER + r"\s*" + _PERCENT),
    # 2. LaTeX \boxed{18}
    re.compile(r"\\boxed\{\s*" + _SIGNED_NUMBER + r"\s*" + _PERCENT + r"\s*\}"),
    # 3. "final answer is 18", "答案是：-5", "结果为 1,234", "answer: $3.50"
    re.compile(
        r"(?i:final\s+answer|correct\s+answer|answer|答案|最终答案|正确答案|"
        r"结果为|结果是|答案为|得数|所以)"
        r"\s*(?:(?i:is|are|was|were|equals?)\s*|(?i:为|是|应是|应该是)\s*|[:：=]\s*)*"
        + _SIGNED_NUMBER + r"\s*" + _PERCENT
    ),
)

_GOLD_METADATA_KEYS = ("gold_number", "answer_number", "numeric_answer", "gold")


def _to_float(raw: Any) -> Optional[float]:
    """Convert a raw numeric token to float, tolerating formatting noise."""
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
        return value if math.isfinite(value) else None
    try:
        text = unicodedata.normalize("NFKC", str(raw)).strip()
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None
    if not text:
        return None
    for source, target in (("−", "-"), ("–", "-"), ("—", "-")):
        text = text.replace(source, target)
    text = text.replace(",", "").replace(" ", "").replace("\u00a0", "")
    for symbol in ("$", "€", "£", "¥", "￥"):
        text = text.replace(symbol, "")
    text = text.strip("%")
    if text in {"", "-", "+", ".", "-.", "+."}:
        return None
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def extract_final_number(text: str) -> Optional[float]:
    """Extract the final numeric answer from free-form model output.

    Returns ``None`` when no number is present or the input is empty/garbled.
    Never raises.
    """
    if not text or not isinstance(text, str):
        return None
    normalized = unicodedata.normalize("NFKC", text).replace("\u00a0", " ")

    # 1-3) explicit markers: take the last occurrence of the highest tier found.
    for pattern in _EXPLICIT_PATTERNS:
        matches = list(pattern.finditer(normalized))
        for match in reversed(matches):
            value = _to_float(match.group(1))
            if value is not None:
                return value

    # 4) fallback: last standalone number in the text.
    matches = list(_NUMBER_TOKEN_RE.finditer(normalized))
    for match in reversed(matches):
        value = _to_float(match.group(0))
        if value is not None:
            return value
    return None


def _gold_number(sample: Sample) -> Optional[float]:
    """Resolve the gold number from metadata (preferred) or ``Sample.answer``."""
    metadata = sample.metadata or {}
    for key in _GOLD_METADATA_KEYS:
        if key not in metadata:
            continue
        value = metadata[key]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = _to_float(value)
            if number is not None:
                return number
        elif isinstance(value, str) and value.strip():
            number = extract_final_number(value)
            if number is None:
                number = _to_float(value)
            if number is not None:
                return number

    if sample.answer is not None:
        answer = str(sample.answer)
        number = extract_final_number(answer)
        if number is None:
            number = _to_float(answer)
        if number is not None:
            return number
    return None


def numbers_equal(predicted: float, gold: float, rel_tol: float = 1e-6) -> bool:
    """Relative-tolerance numeric comparison (absolute floor at 1e-6)."""
    try:
        if predicted is None or gold is None:
            return False
        if not (math.isfinite(predicted) and math.isfinite(gold)):
            return False
        tolerance = rel_tol * max(1.0, abs(float(gold)))
        return abs(float(predicted) - float(gold)) <= tolerance
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return False


def score_numeric(sample: Sample, response: RawResponse) -> Score:
    """Score a GSM8K sample with numeric tolerance matching (1e-6 relative)."""
    try:
        text = (response.text or "") if response is not None else ""
        gold = _gold_number(sample)
        predicted = extract_final_number(text)
        if gold is None:
            return Score(
                correct=False,
                score=0.0,
                detail={"error": "missing gold number", "extracted": predicted},
            )
        if predicted is None:
            return Score(
                correct=False,
                score=0.0,
                detail={"extracted": None, "gold": gold, "first_200_chars": text[:200]},
            )
        correct = numbers_equal(predicted, gold)
        return make_score(
            correct,
            {
                "extracted": predicted,
                "gold": gold,
                "abs_error": abs(float(predicted) - float(gold)),
                "rel_tolerance": 1e-6,
                "first_200_chars": text[:200],
            },
        )
    except Exception as exc:  # pragma: no cover - defensive: never raise upstream
        return Score(
            correct=False,
            score=0.0,
            detail={"error": f"{type(exc).__name__}: {exc}"},
        )
