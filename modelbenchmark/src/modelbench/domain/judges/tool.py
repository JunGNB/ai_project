"""``tool_call`` judge: BFCL possible-answer structural match (0/1).

Parses one JSON/JSONL function-call response, then compares function names and
argument values against the sample's ``ground_truth`` (unordered multi-call;
``irrelevance`` requires an empty call list).  Algorithm moved verbatim from
``tool/system/judges.py``.
"""

from __future__ import annotations

import json
import re

from .base import nfkc

_STRIP_CHARS = set(" ,./-_*^")
_CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def _standardize_string(value: str) -> str:
    """BFCL ``standardize_string``: case/punctuation/space-insensitive strings."""
    text = nfkc(value).strip().lower().replace("'", '"')
    return "".join(ch for ch in text if ch not in _STRIP_CHARS)


def _canon(v):
    if isinstance(v, bool):
        return ("bool", v)
    if isinstance(v, (int, float)):
        return ("num", round(float(v), 6))
    if isinstance(v, str):
        return ("str", _standardize_string(v))
    if isinstance(v, list):
        return ("list", [_canon(x) for x in v])
    if isinstance(v, dict):
        return ("dict", {k: _canon(x) for k, x in sorted(v.items())})
    if v is None:
        return ("null", None)
    return ("other", str(v))


def _value_ok(pred_missing: bool, pred, acceptable) -> bool:
    for acc in acceptable:
        if acc == "" and pred_missing:
            return True
        if pred_missing:
            continue
        if _canon(pred) == _canon(acc):
            return True
        if isinstance(pred, (int, float)) and isinstance(acc, (int, float)) and not isinstance(acc, bool):
            if abs(float(pred) - float(acc)) <= 1e-6:
                return True
    return False


def _decode_json_any(text: str):
    t = text.strip()
    blocks = _CODE_FENCE.findall(t)
    if blocks:
        t = blocks[0].strip()
    # 1) the whole response is one JSON value (most reliable; handles nested arrays)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    # 2) scan from a top-level object first, then from arrays; never start inside a
    #    nested array (the old bug made {"arguments": {"xs": [1, 2]}} unparseable).
    for opener in ("{", "["):
        start = t.find(opener)
        while start >= 0:
            try:
                obj, _ = json.JSONDecoder().raw_decode(t[start:])
                if isinstance(obj, dict) or isinstance(obj, list):
                    return obj
            except json.JSONDecodeError:
                pass
            start = t.find(opener, start + 1)
    return None


def _call_args(obj: dict):
    args = obj.get("arguments", obj.get("parameters", obj.get("args", {})))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    return args if isinstance(args, dict) else {}


def parse_calls(text: str):
    # JSONL-style: two or more standalone JSON objects on separate lines
    lines = [ln.strip().rstrip(",") for ln in text.strip().splitlines() if ln.strip()]
    objs = []
    if len(lines) >= 2:
        for ln in lines:
            try:
                obj = json.loads(ln)
            except json.JSONDecodeError:
                objs = []
                break
            if isinstance(obj, dict) and "name" in obj:
                args = _call_args(obj)
                if args is None:
                    objs = []
                    break
                objs.append({"name": obj["name"], "arguments": args})
        if objs:
            return objs
    obj = _decode_json_any(text)
    if obj is None:
        return None
    if isinstance(obj, dict):
        if "name" in obj:
            args = _call_args(obj)
            return None if args is None else [{"name": obj["name"], "arguments": args}]
        return None
    if isinstance(obj, list):
        calls = []
        for item in obj:
            if not isinstance(item, dict) or "name" not in item:
                return None
            args = _call_args(item)
            if args is None:
                return None
            calls.append({"name": item["name"], "arguments": args})
        return calls
    return None


def judge_tool_call(response: str, judge: dict) -> dict:
    calls = parse_calls(response)
    gt = judge["ground_truth"]
    if not gt:
        # BFCL irrelevance: no function call means correct. A natural-language
        # refusal (AST decoding failure) is "no call", not a failure.
        ok = calls is None or len(calls) == 0
        return {"score": 1.0 if ok else 0.0, "pass": ok, "reason": "irrelevance", "pred_calls": calls or []}
    if calls is None:
        return {"score": 0.0, "pass": False, "reason": "unparseable_output", "n_gt": len(gt)}
    if len(calls) != len(gt):
        return {"score": 0.0, "pass": False, "reason": f"call_count {len(calls)} != {len(gt)}"}
    used = [False] * len(calls)
    for gt_call in gt:
        (gt_name, gt_args), = gt_call.items()
        matched = False
        for i, call in enumerate(calls):
            if used[i] or call["name"] != gt_name:
                continue
            pred_args = call["arguments"]
            if set(pred_args) - set(gt_args):
                continue
            good = True
            for arg, acceptable in gt_args.items():
                if not _value_ok(arg not in pred_args, pred_args.get(arg), acceptable):
                    good = False
                    break
            if good:
                used[i] = True
                matched = True
                break
        if not matched:
            return {"score": 0.0, "pass": False, "reason": f"no_match_for {gt_name}", "pred_calls": calls}
    return {"score": 1.0 if all(used) else 0.0, "pass": all(used)}
