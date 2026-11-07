"""Evaluator contract and shared helpers.

Evaluators are pure functions (no network, no filesystem) that turn a model
response into a :class:`modelbench.domain.models.Score`.  Keeping them pure makes them
trivial to unit-test and safe to reuse across scenarios.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Dict, List, Optional

from ..models import RawResponse, Sample, Score


def normalize_text(text: str) -> str:
    """Lowercase, unicode-normalize and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip().lower()


def strip_markdown(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"[*_`#>]", " ", text)
    return text


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first balanced JSON object from free-form text."""
    if not text:
        return None
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : idx + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def parse_number(text: str) -> Optional[float]:
    """Extract the last standalone number, tolerating thousands separators."""
    if not text:
        return None
    matches = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if not matches:
        return None
    raw = matches[-1].replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def make_score(correct: bool, detail: Optional[Dict[str, Any]] = None, weight: float = 1.0) -> Score:
    return Score(correct=bool(correct), score=(weight if correct else 0.0), detail=detail or {})


def response_text(response: RawResponse) -> str:
    """Text used for scoring (visible answer, never reasoning)."""
    return response.text or ""
