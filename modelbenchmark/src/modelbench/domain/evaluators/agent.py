"""BFCL-style tool-call evaluator.

BFCL scores the *structure* of a function call (name + arguments), not text
similarity.  The model response carries OpenAI-style ``tool_calls``; the gold
call list lives in ``sample.metadata["gold_calls"]``::

    gold_calls = [{"name": "get_weather", "arguments": {"city": "Paris"}}]

An empty ``gold_calls`` list means "the model must not call any tool" (BFCL
``irrelevance`` category).

Argument matching is type-aware:

* ``True``/``False`` are booleans, never integers (Python's ``bool`` subclass of
  ``int`` is explicitly rejected);
* ``int``/``float`` compare numerically with a tiny tolerance;
* strings compare after stripping surrounding whitespace;
* lists and nested dicts compare recursively;
* a JSON-encoded string (``"{\\"city\\": \\"Paris\\"}"``) is parsed before
  comparison;
* a gold value that is a list is interpreted in BFCL *possible-answer* style:
  the prediction matches when it equals the whole list or any one alternative.

Pure functions only: no IO, no network, never raises out of the public API.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional

from ..models import RawResponse, Sample, Score
from .base import extract_json_object, make_score

#: Characters kept from ``repr``/JSON previews inside ``Score.detail``.
_DETAIL_LIMIT = 1200


def normalize_args(obj: Any) -> Any:
    """Normalise a tool-call argument payload for structural comparison.

    JSON strings are parsed; dict keys become stripped strings; numeric values
    become floats; strings are stripped.  Unparseable strings are returned
    stripped (not dropped), so the caller can still report a mismatch.
    """
    if isinstance(obj, str):
        text = obj.strip()
        if not text:
            return ""
        if text[0] in "{[":
            try:
                parsed = json.loads(text)
            except (json.JSONDecodeError, ValueError, TypeError):
                return text
            return normalize_args(parsed)
        return text
    if isinstance(obj, dict):
        return {str(key).strip(): normalize_args(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [normalize_args(value) for value in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float)):
        value = float(obj)
        return value if math.isfinite(value) else obj
    return obj


def _normalize_string(value: str) -> str:
    return " ".join(str(value).strip().split())


def _numeric_equal(gold: float, pred: float) -> bool:
    return abs(float(gold) - float(pred)) <= 1e-9 * max(1.0, abs(float(gold)))


def _match_list_literal(gold: List[Any], pred: List[Any]) -> bool:
    if len(gold) != len(pred):
        return False
    return all(_value_match(g, p) for g, p in zip(gold, pred))


def _value_match(gold: Any, pred: Any) -> bool:
    """Type-aware recursive comparison of two normalized values."""
    # bool must be handled before int (bool is a subclass of int in Python).
    if isinstance(gold, bool):
        return isinstance(pred, bool) and gold == pred
    if isinstance(pred, bool) and not isinstance(gold, bool):
        return False
    if isinstance(gold, (int, float)):
        if isinstance(pred, bool) or not isinstance(pred, (int, float)):
            return False
        return _numeric_equal(gold, pred)
    if isinstance(gold, str):
        if not isinstance(pred, str):
            return False
        return _normalize_string(gold) == _normalize_string(pred)
    if isinstance(gold, dict):
        if not isinstance(pred, dict):
            return False
        if set(gold.keys()) != set(pred.keys()):
            return False
        return all(_value_match(gold[key], pred[key]) for key in gold)
    if isinstance(gold, (list, tuple)):
        if isinstance(pred, (list, tuple)) and _match_list_literal(gold, pred):
            return True
        # BFCL possible-answer semantics: any alternative may match.
        return any(_value_match(item, pred) for item in gold)
    return gold == pred


def args_match(gold: Any, pred: Any) -> bool:
    """Return True when predicted arguments match the gold arguments."""
    try:
        return _value_match(normalize_args(gold), normalize_args(pred))
    except Exception:  # pragma: no cover - defensive
        return False


def _parse_argument_payload(raw: Any) -> Optional[Any]:
    """Parse a tool call's ``arguments`` field into a normalized structure."""
    if isinstance(raw, dict):
        return normalize_args(raw)
    if raw is None:
        return {}
    if not isinstance(raw, str):
        return normalize_args(raw)
    text = raw.strip()
    if not text:
        return {}
    try:
        return normalize_args(json.loads(text))
    except (json.JSONDecodeError, ValueError):
        pass
    extracted = extract_json_object(text)
    if extracted is not None:
        return normalize_args(extracted)
    return None


def _normalize_gold_call(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    if "function" in raw and isinstance(raw["function"], dict):
        function = raw["function"]
        name = function.get("name")
        arguments = function.get("arguments")
    else:
        name = raw.get("name")
        arguments = raw.get("arguments", raw.get("args"))
    if name is None:
        return None
    parsed_arguments = _parse_argument_payload(arguments)
    if parsed_arguments is None and isinstance(arguments, str):
        parsed_arguments = str(arguments)
    if parsed_arguments is None:
        parsed_arguments = {}
    return {"name": str(name).strip(), "arguments": parsed_arguments}


def _normalize_gold_calls(raw: Any) -> Optional[List[Dict[str, Any]]]:
    """Normalise metadata gold calls; ``None`` when the metadata is unusable."""
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            raw = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return None
    calls: List[Dict[str, Any]] = []
    for item in raw:
        call = _normalize_gold_call(item)
        if call is None:
            return None
        calls.append(call)
    return calls


def _calls_from_response(response: Optional[RawResponse]) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    if response is None:
        return calls
    for tool_call in response.tool_calls or []:
        name = getattr(tool_call, "name", None)
        if callable(name):  # pragma: no cover - pydantic field, defensive
            name = name()
        arguments = getattr(tool_call, "arguments", None)
        if callable(arguments):  # pragma: no cover - pydantic field, defensive
            arguments = arguments()
        calls.append(
            {
                "name": "" if name is None else str(name).strip(),
                "arguments": _parse_argument_payload(arguments),
            }
        )
    return calls


def _preview(value: Any, limit: int = _DETAIL_LIMIT) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        text = repr(value)
    return text[:limit]


def score_tool_calls(sample: Sample, response: RawResponse) -> Score:
    """Score one BFCL sample by comparing tool calls (count, name, arguments).

    Empty gold calls pass only when the model made no call.  Otherwise the
    number of calls, their order, names and arguments must all match.
    ``detail`` records gold/pred previews truncated to 1200 characters plus the
    first failing reason.
    """
    try:
        metadata = sample.metadata or {}
        gold_raw = metadata.get("gold_calls")
        if gold_raw is None:
            gold_raw = metadata.get("expected_calls")
        gold = _normalize_gold_calls(gold_raw)
        pred = _calls_from_response(response)
        preview_gold = _preview(gold if gold is not None else gold_raw)
        preview_pred = _preview(
            [
                {"name": call["name"], "arguments": call["arguments"]}
                for call in pred
            ]
        )

        if gold is None:
            return Score(
                correct=False,
                score=0.0,
                detail={
                    "error": "unusable gold_calls metadata",
                    "gold": preview_gold,
                    "pred": preview_pred,
                },
            )

        def _result(
            correct: bool, reason: str, failing_index: Optional[int] = None
        ) -> Score:
            detail: Dict[str, Any] = {
                "match": correct,
                "reason": reason,
                "n_gold": len(gold or []),
                "n_pred": len(pred),
                "gold": preview_gold,
                "pred": preview_pred,
            }
            if failing_index is not None:
                detail["failing_index"] = failing_index
            return make_score(correct, detail)

        if not gold:
            if not pred:
                return _result(True, "no call expected and none made")
            return _result(False, "no call expected but a call was made")

        if len(pred) != len(gold):
            return _result(
                False, f"call count mismatch: gold={len(gold)} pred={len(pred)}"
            )

        for index, (gold_call, pred_call) in enumerate(zip(gold, pred)):
            if gold_call["name"] != pred_call["name"]:
                return _result(
                    False,
                    f"name mismatch at index {index}: "
                    f"gold={gold_call['name']!r} pred={pred_call['name']!r}",
                    failing_index=index,
                )
            pred_arguments = pred_call["arguments"]
            if pred_arguments is None:
                return _result(
                    False,
                    f"unparseable arguments at index {index}",
                    failing_index=index,
                )
            if not args_match(gold_call["arguments"], pred_arguments):
                return _result(
                    False,
                    f"argument mismatch at index {index}",
                    failing_index=index,
                )
        return _result(True, "all calls matched")
    except Exception as exc:  # pragma: no cover - defensive: never raise upstream
        return Score(
            correct=False,
            score=0.0,
            detail={"error": f"{type(exc).__name__}: {exc}"},
        )
