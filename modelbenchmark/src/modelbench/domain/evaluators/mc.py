"""Multiple-choice evaluator (MMLU / MMMLU / LongBench-v2).

The datasets in this project ask for a single option letter (``A``-``D``).
Models, however, express the answer in many ways::

    Answer: C                    答案是C                  answer is (C)
    The correct option is [B]    **D**                    C
    I choose A.                  选项C正确。

``extract_choice_letter`` implements a deterministic, priority-ordered
extraction:

1. explicit answer phrases (``answer``/``答案``/``correct answer`` ...),
2. ``<letter> is the correct answer`` / ``选项C``-style statements,
3. a line that contains nothing but a single letter,
4. the last standalone uppercase letter in the text.

Deliberately *not* counted: letters that appear inside option text (``A. ...``)
unless they are the last standalone candidate, and an English article ``A`` in a
sentence-initial position.  Everything is pure (no IO) and never raises.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Tuple

from ..models import RawResponse, Sample, Score
from .base import make_score

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

#: (priority, take_first, pattern).  The first priority level with at least one
#: non-negated candidate wins; ``take_first`` decides whether the first or the
#: last candidate of that level is returned.
_EXPLICIT_PATTERNS: Tuple[Tuple[int, bool, "re.Pattern[str]"], ...] = (
    # "final answer is C", "答案：C", "correct answer is (C)", "answer: option D"
    (
        0,
        True,
        re.compile(
            r"(?i:final\s+answer|correct\s+(?:answer|option|choice)|answer|"
            r"答案|最终答案|正确答案|正确选项|所选项)"
            r"\s*(?:(?i:is|are|was|were|should\s+be|would\s+be|equals|为|是|"
            r"应该是|应为)\s*)?[:\：=]?\s*"
            r"(?:(?i:option|choice|选项)\s*)?[\(\{\[<“\"']?\s*([A-Z])"
        ),
    ),
    # "C is the correct answer", "B 是正确答案"
    (
        1,
        False,
        re.compile(
            r"(?<![A-Za-z])([A-Z])(?![A-Za-z])\s*"
            r"(?i:is|are|was|为|是)\s*(?i:the\s+)?"
            r"(?i:correct|right|best|chosen|answer|答案|正确)"
        ),
    ),
    # "I choose C", "选择 C", "选项 C"
    (
        2,
        False,
        re.compile(
            r"(?i:choose|select|pick|go\s+with|选择|选|选项)\s*"
            r"(?:(?i:option|choice|选项)\s*)?[\(\{\[<“\"']?\s*([A-Z])"
        ),
    ),
)

#: A negation immediately before the letter means the model is talking about a
#: *wrong* option ("the answer is not A, but C"), so that candidate is skipped.
_NEGATION_RE = re.compile(
    r"(?i)(?:\bnot\b|\bnever\b|\bno\b|isn'?t|aren'?t|wasn'?t|doesn'?t|don'?t|"
    r"不是|并非|不选|排除|错误|不对|不正确)\s*"
    r"(?:(?i:option|choice|选项)\s*)?[\(\{\[<]?\s*$"
)

_BRACKETED_RE = re.compile(r"[\(\{\[<]\s*([A-Z])\s*[\)\}\]>]")

_STANDALONE_UPPER_RE = re.compile(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])")

_ARTICLE_RE = re.compile(r"\s+[a-z]")

_SENTENCE_START_RE = re.compile(r"[.!?。！？\n]\s*$")

#: Characters stripped when a whole line is just an emphasised letter.
_LINE_STRIP_CHARS = "*_`#>[](){}<>.,;:!?\"'“”‘’。，、：；！？（）【】《》"


def _plain(text: str) -> str:
    """NFKC-normalise the text and neutralise markdown emphasis characters."""
    plain = unicodedata.normalize("NFKC", text or "").replace("\u00a0", " ")
    # ``**C**`` / ``*C*`` / ``_C_`` / ``## C`` -> letter stays a standalone token
    return re.sub(r"[`*_#]+", " ", plain)


def _negated(plain: str, letter_pos: int) -> bool:
    """True when the letter is introduced by a negation ("not A")."""
    window = plain[max(0, letter_pos - 28):letter_pos]
    return _NEGATION_RE.search(window) is not None


def _looks_like_article(plain: str, pos: int) -> bool:
    """True when ``A`` at ``pos`` is the English article, not an option label."""
    following = plain[pos + 1:pos + 18]
    preceding = plain[max(0, pos - 2):pos]
    followed_by_word = _ARTICLE_RE.match(following) is not None
    sentence_initial = preceding == "" or _SENTENCE_START_RE.search(preceding) is not None
    return followed_by_word and sentence_initial


def extract_choice_letter(text: str, num_choices: int = 4) -> Optional[str]:
    """Extract the chosen option letter from free-form model output.

    Args:
        text: visible answer text (reasoning already removed by the client).
        num_choices: number of answer options; only ``A``..(num_choices) are
            accepted.  Values outside ``1..26`` are clamped.

    Returns:
        A single uppercase letter, or ``None`` when no answer can be found.
        Never raises, even for empty/None/garbled input.
    """
    if not text or not isinstance(text, str) or not text.strip():
        return None
    try:
        num = int(num_choices or 4)
    except (TypeError, ValueError):
        num = 4
    num = max(1, min(num, 26))
    valid = set(_LETTERS[:num])

    plain = _plain(text)

    # 1) Explicit answer phrases, highest priority first.
    for _priority, take_first, pattern in _EXPLICIT_PATTERNS:
        candidates: List[str] = []
        for match in pattern.finditer(plain):
            letter = match.group(1).upper()
            if letter not in valid:
                continue
            if _negated(plain, match.start(1)):
                continue
            candidates.append(letter)
        if candidates:
            return candidates[0] if take_first else candidates[-1]

    # 2) A bracketed single letter, e.g. "(C)" / "[D]".
    bracketed = [m.group(1).upper() for m in _BRACKETED_RE.finditer(plain)]
    bracketed = [letter for letter in bracketed if letter in valid]
    if bracketed:
        return bracketed[-1]

    # 3) A line whose only content is a single letter ("C", "C.", "**C**").
    for line in reversed(plain.splitlines()):
        stripped = line.strip().strip(_LINE_STRIP_CHARS).strip()
        if len(stripped) == 1 and stripped.upper() in valid:
            return stripped.upper()

    # 4) A single standalone uppercase letter.  Multiple distinct candidates
    #    mean the text lists the options (e.g. "A. 2 B. 4 C. 6 D. 8") rather
    #    than answering, so refuse to guess instead of taking the last one.
    candidates = set()
    for match in _STANDALONE_UPPER_RE.finditer(plain):
        letter = match.group(1)
        if letter not in valid:
            continue
        if letter == "A" and _looks_like_article(plain, match.start(1)):
            continue
        if _negated(plain, match.start(1)):
            continue
        candidates.add(letter)
    if len(candidates) == 1:
        return candidates.pop()
    return None


def _first_letter_in_gold(raw: str) -> Optional[str]:
    """Extract the canonical option letter from a gold answer string."""
    raw = unicodedata.normalize("NFKC", raw or "").strip()
    if not raw:
        return None
    # "A", "A.", "(A)", "A) ..." -> the leading letter is the answer.
    match = re.match(r"^[\(\[]?\s*([A-Za-z])\s*[\)\]]?\s*(?:[.:：、,，)\]）]|$)", raw)
    if match:
        return match.group(1).upper()
    # "Option C" / "the answer is D"
    match = re.search(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])", raw)
    if match:
        return match.group(1).upper()
    match = re.search(r"([A-Za-z])", raw)
    return match.group(1).upper() if match else None


def _gold_letter(sample: Sample) -> Optional[str]:
    """Resolve the expected option letter from ``Sample.answer``/metadata."""
    metadata = sample.metadata or {}
    candidates: List[str] = []
    for key in ("answer_letter", "gold_letter", "label"):
        value = metadata.get(key)
        if value is not None:
            candidates.append(str(value))
    if sample.answer is not None:
        candidates.append(str(sample.answer))
    for raw in candidates:
        letter = _first_letter_in_gold(raw)
        if letter is not None:
            return letter
    return None


def score_mc(sample: Sample, response: RawResponse) -> Score:
    """Score a multiple-choice sample by exact option-letter match.

    ``detail`` records the extracted letter, the gold letter and the first 200
    characters of the response (for post-hoc inspection of scoring errors).
    """
    try:
        text = (response.text or "") if response is not None else ""
        expected = _gold_letter(sample)
        if expected is None:
            return Score(
                correct=False,
                score=0.0,
                detail={"error": "missing gold answer letter", "first_200_chars": text[:200]},
            )
        num_choices = len(sample.choices) if sample.choices else 4
        predicted = extract_choice_letter(text, num_choices)
        correct = predicted is not None and predicted == expected
        return make_score(
            correct,
            {
                "extracted": predicted,
                "gold": expected,
                "num_choices": num_choices,
                "first_200_chars": text[:200],
            },
        )
    except Exception as exc:  # pragma: no cover - defensive: never raise upstream
        return Score(
            correct=False,
            score=0.0,
            detail={"error": f"{type(exc).__name__}: {exc}"},
        )
