"""``mc_letter`` judge: multiple-choice A-D letter extraction (0/1).

Algorithm moved verbatim from ``tool/system/judges.py`` (semantics unchanged):
prefer the *last* explicit answer phrase, then the last standalone A-D letter,
then the last A-D character; refusals are not guesses.
"""

from __future__ import annotations

import re
from typing import Optional

from .base import nfkc

_REFUSAL_RE = re.compile(
    r"(cannot\s+(?:be\s+)?determin|can'?t\s+determin|unable\s+to\s+(?:determine|answer)|"
    r"no\s+(?:correct|valid)\s+(?:answer|option)|not\s+enough\s+information|"
    r"insufficient\s+information|无法(?:确定|判断|回答)|不能确定|没有(?:正确|合适)的?(?:答案|选项)|"
    r"信息不足|无法作答)",
    re.I,
)
_ANSWER_PATTERNS = (
    r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|:|=|：)\s*[*_`'\"\(\[\{<]*\s*(?:option|choice|选项)?\s*([A-D])\b",
    r"答案\s*(?:是|为)?\s*[:：=]?\s*[*_`'\"\(\[\{<]*\s*(?:选项)?\s*([A-D])",
)


def extract_letter(text: str) -> Optional[str]:
    """Return the option letter a response commits to (or ``None``).

    Order of preference: the *last* explicit answer phrase ("answer is X" /
    "答案是 X" / "Answer: **X**") -> the last standalone A-D letter -> the last
    A-D character.  Normalises full-width characters and accepts lower case.

    An explicit answer phrase always wins over a refusal word: models often
    reason "we don't have enough information about P(B) ... The answer is B.",
    and the earlier refusal sentence must not discard the committed answer
    (found by the 2026-11 accuracy run audit).  The refusal check is therefore
    applied only when the response does not commit to a letter.
    """
    if not text:
        return None
    t = nfkc(text).strip()
    explicit = []
    for pattern in _ANSWER_PATTERNS:
        explicit.extend((m.start(), m.group(1).upper()) for m in re.finditer(pattern, t, re.I))
    if explicit:
        return max(explicit)[1]
    if _REFUSAL_RE.search(t):
        return None
    standalone = re.findall(r"(?<![A-Za-z0-9])([A-Da-d])(?![A-Za-z0-9])", t)
    if standalone:
        return standalone[-1].upper()
    loose = re.findall(r"([A-Da-d])", t)
    return loose[-1].upper() if loose else None


def judge_mc_letter(response: str, judge: dict) -> dict:
    pred = extract_letter(response)
    ok = pred == judge["answer"]
    return {"score": 1.0 if ok else 0.0, "pass": ok, "pred": pred, "gold": judge["answer"]}
