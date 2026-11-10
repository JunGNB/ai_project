#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic judges for the 8-scenario accuracy set (5 judge families).

  mc_letter        -> 0/1
  normalized_match -> F1 (LongBench English / HotpotQA), Chinese F1 (LongBench
                      zh QA), ROUGE-L (LongBench dureader), or numeric match
  ifeval           -> official Google IFEval prompt-level strict accuracy (0/1)
  code_exec        -> pass@1 via unit tests in an isolated subprocess (0/1)
  tool_call        -> BFCL possible-answer structural match (0/1)

No LLM judge, no network, no rubric. Every sample's contract is in its jsonl
"judge" field; this module only executes the contract.

The raw-data root is ``dataset/raw`` (older revisions used ``data/raw``); an
``MB_RAW_DIR`` environment variable can override it for deployed layouts.
"""
from __future__ import annotations

import json
import os
import re
import string
import subprocess
import sys
import tempfile
import unicodedata
import uuid
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = Path(os.environ.get("MB_RAW_DIR", str(ROOT / "dataset" / "raw")))
IFE_PKG = RAW_DIR / "ifeval"
NLTK_DATA = RAW_DIR / "nltk_data"


# ---------------------------------------------------------------- mc_letter

_REFUSAL_RE = re.compile(
    r"(cannot\s+(?:be\s+)?determin|can'?t\s+determin|unable\s+to\s+(?:determine|answer)|"
    r"no\s+(?:correct|valid)\s+(?:answer|option)|not\s+enough\s+information|"
    r"insufficient\s+information|无法(?:确定|判断|回答)|不能确定|没有(?:正确|合适)的?(?:答案|选项)|"
    r"信息不足|无法作答)",
    re.I,
)
_ANSWER_PATTERNS = (
    r"(?:the\s+)?answer\s*(?:is|:|=|：)\s*\(?([A-D])\)?",
    r"答案\s*(?:是|为)?\s*[:：=]?\s*\(?([A-D])\)?",
)


def _nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "")


def extract_letter(text: str):
    """Return the option letter a response commits to (or ``None``).

    Order of preference: the *last* explicit answer phrase ("answer is X" /
    "答案是 X" / "Answer: X") -> the last standalone A-D letter -> the last
    A-D character.  Normalises full-width characters and accepts lower case.
    """
    if not text:
        return None
    t = _nfkc(text).strip()
    if _REFUSAL_RE.search(t):
        return None
    explicit = []
    for pattern in _ANSWER_PATTERNS:
        explicit.extend((m.start(), m.group(1).upper()) for m in re.finditer(pattern, t, re.I))
    if explicit:
        return max(explicit)[1]
    standalone = re.findall(r"(?<![A-Za-z0-9])([A-Da-d])(?![A-Za-z0-9])", t)
    if standalone:
        return standalone[-1].upper()
    loose = re.findall(r"([A-Da-d])", t)
    return loose[-1].upper() if loose else None


def judge_mc_letter(response: str, judge: dict) -> dict:
    pred = extract_letter(response)
    ok = pred == judge["answer"]
    return {"score": 1.0 if ok else 0.0, "pass": ok, "pred": pred, "gold": judge["answer"]}


# ---------------------------------------------------------------- normalized match

_ZH_PUNCT = set(
    "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏"
)


def _normalize_squad(s: str) -> str:
    s = _nfkc(s).lower()
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
    s = _nfkc(s).lower()
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
    tokenized_pred = " ".join(_jieba_cut(_nfkc(pred)))
    best = 0.0
    for gold in golds:
        tokenized_gold = " ".join(_jieba_cut(_nfkc(str(gold))))
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
    t = _nfkc(str(text)).replace("\u2212", "-")
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


def _numbers_match(pred: Fraction, gold: Fraction, tolerance: str | None) -> bool:
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


# ---------------------------------------------------------------- ifeval

_ifeval_mod = None


def _load_ifeval():
    global _ifeval_mod
    if _ifeval_mod is None:
        if not IFE_PKG.is_dir():
            raise RuntimeError(
                f"IFEval checker not found at {IFE_PKG}; expected dataset/raw/ifeval "
                "(set MB_RAW_DIR for a deployed layout)"
            )
        os.environ["NLTK_DATA"] = str(NLTK_DATA)
        sys.path.insert(0, str(IFE_PKG))
        try:
            from instruction_following_eval import evaluation_lib  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover - environment problem
            raise RuntimeError(
                f"IFEval checker failed to import from {IFE_PKG}: {type(exc).__name__}: {exc}"
            ) from exc
        _ifeval_mod = evaluation_lib
    return _ifeval_mod


def judge_ifeval(response: str, judge: dict) -> dict:
    ev = _load_ifeval()  # configuration errors must stay loud, never score 0
    inp = ev.InputExample(
        key=0,
        instruction_id_list=judge["instruction_id_list"],
        prompt="__prompt__",
        kwargs=judge["kwargs"],
    )
    try:
        out = ev.test_instruction_following_strict(inp, {"__prompt__": response})
        ok = bool(out.follow_all_instructions)
        detail = out.follow_instruction_list
    except Exception as exc:  # checker misuse -> treat as not followed, keep evidence
        ok = False
        detail = f"checker_error: {type(exc).__name__}: {exc}"
    return {"score": 1.0 if ok else 0.0, "pass": ok, "follow": detail}


# ---------------------------------------------------------------- code_exec

_CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(text: str) -> str:
    if not text:
        return ""
    blocks = _CODE_FENCE.findall(text)
    if blocks:
        return max(blocks, key=len).strip("\n")
    return text.strip()


def _limits(timeout_s: int):
    def _apply():
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (timeout_s + 1, timeout_s + 2))
        # RLIMIT_AS is not settable on macOS (current limit exceeds hard max); the
        # subprocess timeout + CPU limit are the safety bounds here.
        try:  # pragma: no cover - platform specific
            resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
        except (ValueError, OSError):
            pass
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        resource.setrlimit(resource.RLIMIT_FSIZE, (8 << 20, 8 << 20))

    return _apply


def judge_code_exec(response: str, judge: dict) -> dict:
    code = extract_code(response)
    prefix = judge.get("prompt_prefix")
    # Official HumanEval protocol: prompt prefix + generated completion. The model
    # often repeats the signature; a duplicate def is harmless and keeps the imports.
    if prefix:
        code = prefix.rstrip() + "\n" + code
    marker = "__TESTS_PASS__" + uuid.uuid4().hex
    script = (
        code
        + "\n\n"
        + judge["test_code"]
        + f"\n\nprint({marker!r})\n"
    )
    timeout = int(judge.get("timeout_s", 6))
    with tempfile.TemporaryDirectory(prefix="acc_code_") as td:
        path = Path(td) / "candidate.py"
        path.write_text(script, encoding="utf-8")
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": td,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
        }
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=td,
                env=env,
                preexec_fn=_limits(timeout) if os.name == "posix" else None,
            )
            # A test run only passes when the sentinel is reached: sys.exit(0) /
            # os._exit(0) in generated code no longer yields a false pass.
            ok = proc.returncode == 0 and marker in (proc.stdout or "")
            err = "" if ok else (proc.stderr or proc.stdout or "")[-1200:]
            if not ok and proc.returncode == 0 and not err:
                err = "tests did not reach completion sentinel (exit()/os._exit()?)"
        except subprocess.TimeoutExpired:
            ok, err = False, f"timeout>{timeout}s"
        except Exception as exc:  # noqa: BLE001
            ok, err = False, f"exec_error: {type(exc).__name__}: {exc}"
    return {"score": 1.0 if ok else 0.0, "pass": ok, "error": err}


# ---------------------------------------------------------------- tool_call

_STRIP_CHARS = set(" ,./-_*^")


def _standardize_string(value: str) -> str:
    """BFCL ``standardize_string``: case/punctuation/space-insensitive strings."""
    text = _nfkc(value).strip().lower().replace("'", '"')
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


# ---------------------------------------------------------------- dispatch

def evaluate(row: dict, response: str) -> dict:
    fam = row["judge"]["family"]
    if fam == "mc_letter":
        return judge_mc_letter(response, row["judge"])
    if fam == "normalized_match":
        return judge_normalized_match(response, row["judge"])
    if fam == "ifeval":
        return judge_ifeval(response, row["judge"])
    if fam == "code_exec":
        return judge_code_exec(response, row["judge"])
    if fam == "tool_call":
        return judge_tool_call(response, row["judge"])
    raise ValueError(f"unknown judge family {fam}")
