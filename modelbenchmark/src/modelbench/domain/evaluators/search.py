"""HotpotQA search-agent final-answer evaluator.

The search scenario runs a deterministic local-retrieval tool loop and then
scores the *final answer string* against the gold answer with the standard SQuAD
normalisation:

* **EM** (exact match after normalisation) is the primary metric and the
  correctness flag,
* **token F1** is reported as a secondary metric in ``Score.detail``.

``score_search`` accepts a ``RawResponse`` whose ``text`` is either the model's
full final message or an already-extracted answer string (the search scenario
builds ``RawResponse(text=answer)``).  Both work because the answer is the only
thing normalised.

Pure functions: no IO, no network, never raises out of the public entry points.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from ..models import RawResponse, Sample, Score

_GOLD_METADATA_KEYS = ("gold_answer", "answer", "gold", "answers")

_ARTICLE_RE = re.compile(r"\b(a|an|the)\b")
_WHITESPACE_RE = re.compile(r"\s+")

#: Number of trace steps summarised inside ``Score.detail``.
_TRACE_PREVIEW_STEPS = 20
_TRACE_PREVIEW_CHARS = 200


def normalize_answer(text: Any) -> str:
    """SQuAD-style answer normalisation.

    Lowercase -> strip punctuation -> drop articles (a/an/the) -> collapse
    whitespace.  Unicode punctuation (e.g. Chinese ``，。``) is stripped as well.
    """
    if text is None:
        return ""
    try:
        normalized = unicodedata.normalize("NFKC", str(text)).lower()
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return ""
    # Keep alphanumerics and whitespace; punctuation and separators are dropped.
    normalized = "".join(
        character if not unicodedata.category(character).startswith("P") else " "
        for character in normalized
    )
    normalized = _ARTICLE_RE.sub(" ", normalized)
    normalized = normalized.replace("\u00a0", " ")
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def _tokens(normalized: str) -> List[str]:
    return normalized.split() if normalized else []


def token_f1(prediction: str, gold: str) -> float:
    """Token-level F1 between two *already normalised* strings."""
    pred_tokens = _tokens(prediction)
    gold_tokens = _tokens(gold)
    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(pred_tokens)
    recall = same / len(gold_tokens)
    if precision + recall == 0:  # pragma: no cover - same > 0 implies > 0
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def exact_match(prediction: str, gold: str) -> bool:
    """EM on *already normalised* strings."""
    return prediction == gold


def _gold_answer(sample: Sample) -> Optional[str]:
    metadata = sample.metadata or {}
    for key in _GOLD_METADATA_KEYS:
        value = metadata.get(key)
        if isinstance(value, (list, tuple)):
            for item in value:
                if item is not None and str(item).strip():
                    return str(item)
        elif value is not None and str(value).strip():
            return str(value)
    if sample.answer is not None and str(sample.answer).strip():
        return str(sample.answer)
    return None


def score_search(sample: Sample, response: RawResponse) -> Score:
    """Score a HotpotQA answer: EM is the score, token F1 is auxiliary."""
    try:
        text = (response.text or "") if response is not None else ""
        gold_raw = _gold_answer(sample)
        if gold_raw is None:
            return Score(
                correct=False,
                score=0.0,
                detail={"error": "missing gold answer", "pred": text[:200]},
            )
        pred_normalized = normalize_answer(text)
        gold_normalized = normalize_answer(gold_raw)
        em = exact_match(pred_normalized, gold_normalized)
        f1 = token_f1(pred_normalized, gold_normalized)
        return Score(
            correct=bool(em),
            score=1.0 if em else 0.0,
            detail={
                "em": 1.0 if em else 0.0,
                "f1": f1,
                "pred": pred_normalized,
                "gold": gold_normalized,
                "first_200_chars": text[:200],
            },
        )
    except Exception as exc:  # pragma: no cover - defensive: never raise upstream
        return Score(
            correct=False,
            score=0.0,
            detail={"error": f"{type(exc).__name__}: {exc}"},
        )


def _trace_summary(trace: Sequence[Any]) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for step in list(trace)[:_TRACE_PREVIEW_STEPS]:
        if isinstance(step, dict):
            entry: Dict[str, Any] = {}
            for key in ("tool", "name", "action", "query", "arguments", "args", "result"):
                if key in step:
                    entry[key] = str(step[key])[:_TRACE_PREVIEW_CHARS]
            if not entry:
                entry = {"repr": repr(step)[:_TRACE_PREVIEW_CHARS]}
            summary.append(entry)
        else:
            summary.append({"repr": repr(step)[:_TRACE_PREVIEW_CHARS]})
    return summary


def score_search_with_trace(
    sample: Sample, response: RawResponse, trace: Sequence[Any]
) -> Score:
    """Score the final answer and, when a tool trace exists, record it.

    The correctness/score semantics are identical to :func:`score_search`; the
    trace is *observability only* (the local retriever is deterministic, so tool
    correctness is validated by the scenario itself).  An empty trace degrades
    to :func:`score_search`.
    """
    score = score_search(sample, response)
    try:
        trace_items = list(trace or [])
    except TypeError:  # pragma: no cover - defensive
        trace_items = []
    if not trace_items:
        return score
    detail: Dict[str, Any] = dict(score.detail)
    detail["trace_len"] = len(trace_items)
    detail["trace"] = _trace_summary(trace_items)
    return Score(correct=score.correct, score=score.score, detail=detail)
