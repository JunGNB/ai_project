"""``normalized_match`` judge: SQuAD F1, Chinese F1/ROUGE-L, numeric exact.

Algorithm moved verbatim from ``tool/system/judges.py`` (semantics unchanged):
LongBench English F1 / HotpotQA F1, LongBench zh QA F1 (jieba), dureader
ROUGE-L (jieba), or numeric equivalence (``####`` last-number, fractions,
percent, thousands separators, optional relative tolerance).  ``jieba`` and
``rouge`` are optional (``.[eval]``) dependencies imported lazily.
"""

from __future__ import annotations

import re
import string
from fractions import Fraction
from typing import Optional

from .base import nfkc

_ZH_PUNCT = set(
    "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏"
)


def _normalize_squad(s: str) -> str:
    s = nfkc(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation) and ch not in _ZH_PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(pred: str, golds) -> float:
    best = 0.0
    p = _normalize_squad(pred).split()
    for gold in golds:
        g = _normalize_squad(str(gold)).split()
        if not p or not g:
            best = max(best, 1.0 if not p and not g else 0.0)
            continue
        common = 0
        gc = list(g)
        for tok in p:
            if tok in gc:
                gc.remove(tok)
                common += 1
        if common == 0:
            continue
        precision = common / len(p)
        recall = common / len(g)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def _jieba_cut(text: str):
    import jieba  # noqa: PLC0415  (optional eval dependency)

    return list(jieba.cut(text, cut_all=False))


def _normalize_zh_answer(s: str) -> str:
    """Official LongBench ``normalize_zh_answer`` (ASCII+CJK punctuation removed)."""
    s = nfkc(s).lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation) and ch not in _ZH_PUNCT)
    return "".join(s.split())


def token_f1_zh(pred: str, golds) -> float:
    """Official LongBench ``qa_f1_zh_score``: jieba tokens + zh normalisation."""
    p_tokens = [t for t in (_normalize_zh_answer(x) for x in _jieba_cut(pred)) if t]
    best = 0.0
    for gold in golds:
        g_tokens = [t for t in (_normalize_zh_answer(x) for x in _jieba_cut(str(gold))) if t]
        if not p_tokens or not g_tokens:
            continue
        common = 0
        gc = list(g_tokens)
        for tok in p_tokens:
            if tok in gc:
                gc.remove(tok)
                common += 1
        if common == 0:
            continue
        precision = common / len(p_tokens)
        recall = common / len(g_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


_ROUGE = None


def _rouge():
    global _ROUGE
    if _ROUGE is None:
        from rouge import Rouge  # noqa: PLC0415  (official LongBench dependency)

        _ROUGE = Rouge()
    return _ROUGE


def rouge_l_zh(pred: str, golds) -> float:
    """Official LongBench ``rouge_zh_score``: jieba + ROUGE-L F."""
    tokenized_pred = " ".join(_jieba_cut(nfkc(pred)))
    best = 0.0
    for gold in golds:
        tokenized_gold = " ".join(_jieba_cut(nfkc(str(gold))))
        if not tokenized_pred.strip() or not tokenized_gold.strip():
            continue
        try:
            scores = _rouge().get_scores([tokenized_pred], [tokenized_gold], avg=True)
            best = max(best, float(scores["rouge-l"]["f"]))
        except Exception:  # pragma: no cover - defensive, no refs on empty input
            continue
    return best


_NUM_TOKEN = re.compile(
    r"[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:/\d+(?:\.\d+)?)?(?:[eE][-+]?\d+)?"
)


def _fraction(token: str):
    token = token.replace(",", "")
    try:
        return Fraction(token)
    except (ValueError, ZeroDivisionError):
        try:
            return Fraction(str(float(token)))
        except (ValueError, OverflowError):
            return None


def numeric_candidates(text: str):
    """All numbers in the answer segment, most-recent-last (CMATH uses last two)."""
    if not text:
        return []
    t = nfkc(str(text)).replace("\u2212", "-")
    if "####" in t:
        t = t.split("####")[-1]
    values = []
    for match in _NUM_TOKEN.finditer(t):
        token = match.group(0)
        # Match a trailing percent sign only when it directly follows the number.
        pct = match.end() < len(t) and t[match.end()] == "%"
        value = _fraction(token)
        if value is None:
            continue
        values.append(value / 100 if pct else value)
    return values


def parse_number(s: str):
    """Parse the final number of a response (kept for callers/tests)."""
    values = numeric_candidates(s)
    return values[-1] if values else None


def _numbers_match(pred: Fraction, gold: Fraction, tolerance: Optional[str]) -> bool:
    if abs(float(pred) - float(gold)) <= 1e-6:
        return True
    if tolerance == "relative_1pct" and gold.denominator != 1:
        return abs(float(pred) - float(gold)) <= 0.01 * max(abs(float(gold)), 1e-9)
    return False


def judge_normalized_match(response: str, judge: dict) -> dict:
    metric = judge.get("metric", "f1")
    if metric == "f1":
        score = token_f1(response, judge["answers"])
        return {"score": score, "pass": score >= 0.999, "metric": "f1", "gold": judge["answers"]}
    if metric == "f1_zh":
        score = token_f1_zh(response, judge["answers"])
        return {"score": score, "pass": score >= 0.999, "metric": "f1_zh", "gold": judge["answers"]}
    if metric == "rouge_zh":
        score = rouge_l_zh(response, judge["answers"])
        return {"score": score, "pass": score >= 0.999, "metric": "rouge_zh", "gold": judge["answers"]}
    golds = [g for g in (parse_number(a) for a in judge["answers"]) if g is not None]
    preds = numeric_candidates(response)
    tolerance = judge.get("tolerance")
    ok = any(_numbers_match(p, g, tolerance) for p in preds for g in golds)
    return {
        "score": 1.0 if ok else 0.0,
        "pass": ok,
        "metric": "numeric_exact",
        "pred": [str(p) for p in preds],
        "gold": [str(g) for g in golds],
    }
