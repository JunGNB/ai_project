#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the performance test set: 8 scenarios x 1K/8K/32K x 10 = 240 rows in ONE jsonl.

Design: ``reports/perf_design/SPEC.md`` (authoritative).  Each *target* appears in
all three context buckets with a byte-identical question/instruction; only the
preceding context (document prefix, reference block, function catalogue) changes.
Rows are pre-built chat requests for the ``perf_jsonl`` adapter; scoring is never
wired into run mode, so rows carry no gold answer contract.

Inputs  (gitignored, fetch with ``tool/dataset/performance/fetch.py``):
    dataset/raw/longbench_v2/data.json           4 document scenarios
    dataset/raw/ifeval/ifeval_input_data.jsonl   instruction_following
    dataset/raw/gsm8k/...parquet                 math_logic
    dataset/raw/bfcl/BFCL_v4_{simple_python,multiple,parallel}.json   general_agent
    dataset/raw/global_mmlu/<lang>/test.parquet  multilingual (10 languages)
    dataset/raw/tokenizer/cl100k_base/cl100k_base.tiktoken + tokenizer.json
        reference tokenizer (generic OpenAI cl100k_base; tokenizer.json is derived)
Outputs:
    dataset/test/performance/testset.jsonl    240 rows (gitignored)
    dataset/test/performance/manifest.json    sampling / provenance / SHA-256 (no timestamps)
    dataset/test/performance/testset.yaml     human-readable test-set definition
    reports/dataset-build/performance/selection_report.md generated sampling report (intermediate)

    python tool/dataset/performance/build.py
    python tool/dataset/performance/build.py --check            # verify existing outputs
    python tool/dataset/performance/build.py --only coding,gsm8k  # debug: counts only, no writes
    python tool/dataset/performance/build.py --dry-run           # build + assert, no writes

Token semantics: declared ``tokens`` are *content* tokens of
``messages[0].content`` plus the JSON text of ``tools``, counted with the
generic **cl100k_base** tokenizer (no chat template, no special tokens), so
``tokens = messages_tokens + tools_tokens``.  Each row targets the
**calibrated budget** ``(bucket - tool_render) / INFL - 37`` for its
(scenario, language, bucket) -- content-type inflation coefficients bring the
actual model-token count into the run-side +/-10% ISL band -- and construction
acceptance is ``floor(0.95*budget) <= tokens <= floor(1.05*budget)``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tool.common.fs import (  # noqa: E402
    REPORTS_DIR,
    RAW_DIR,
    TEST_DIR,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)
from tool.common.tokenizer import TOKENIZER_NAME, TOKENIZER_PATH, default_tokenizer  # noqa: E402
from tool.dataset.performance import sources  # noqa: E402
from tool.dataset.performance.sampling import allocate, deterministic_pick  # noqa: E402

SEED = 42
TARGETS_PER_CELL = 10
BUCKETS: Dict[str, int] = {"1k": 1024, "8k": 8192, "32k": 32768}
BUCKET_ORDER = {"1k": 0, "8k": 1, "32k": 2}
MULTILINGUAL_LANGUAGES: Tuple[str, ...] = ("fr", "de", "es", "pt", "ru", "ja", "ko", "ar", "hi", "vi")

#: Scenario order of §0/§3 of the spec: also the row order of ``testset.jsonl``.
SCENARIOS: Tuple[str, ...] = (
    "knowledge_qa",
    "instruction_following",
    "long_context",
    "math_logic",
    "coding",
    "general_agent",
    "search_agent",
    "multilingual",
)
PER_SCENARIO = TARGETS_PER_CELL * len(BUCKETS)  # 30
TOTAL_ROWS = PER_SCENARIO * len(SCENARIOS)  # 240
TOTAL_TARGETS = TARGETS_PER_CELL * len(SCENARIOS)  # 80

ACCEPT_LOW, ACCEPT_HIGH = 0.95, 1.05
OSL = 50  # run-time default decode length (perf.max_tokens)
CONTEXT_SAFETY_MARGIN_TOKENS = 128
DECODE_GUARD_TOKEN_RESERVE = 64
#: declared tokens + max_tokens + safety margin + decode-guard reserve must fit the window.
WINDOW_RESERVE = OSL + CONTEXT_SAFETY_MARGIN_TOKENS + DECODE_GUARD_TOKEN_RESERVE  # 242

#: LongBench v2 document pool filter (the spec's sampling rule).
LBV2_MIN_CONTEXT_CHARS = 120_000
#: A target question/options block larger than this fraction of 1K cannot leave 1K of context.
PROMPT_GUARD_TOKEN_FRACTION = 0.6
#: Same-set reference rotations.
MATH_ROTATION_STEP = 41

BFCL_CATEGORIES: Tuple[str, ...] = ("simple_python", "multiple", "parallel")
BFCL_CATEGORY_LABEL = {"simple_python": "simple", "multiple": "multiple", "parallel": "parallel"}
BFCL_QUOTAS = {"simple_python": 4, "multiple": 3, "parallel": 3}

DOC_SCENARIOS: Tuple[str, ...] = ("knowledge_qa", "long_context", "coding", "search_agent")
DOC_SCENARIO_DOMAIN: Dict[str, str] = {
    "knowledge_qa": "Single-Document QA",
    "long_context": "Long Structured Data Understanding",
    "coding": "Code Repository Understanding",
    "search_agent": "Multi-Document QA",
}
CONSTRUCTED_SCENARIOS: Tuple[str, ...] = (
    "instruction_following",
    "math_logic",
    "general_agent",
    "multilingual",
)
TOOL_SCENARIOS: Tuple[str, ...] = ("general_agent", "search_agent")

WORKLOADS: Dict[str, str] = {
    "knowledge_qa": "longbench_v2:Single-Document QA",
    "instruction_following": "ifeval:instruction_following",
    "long_context": "longbench_v2:Long Structured Data Understanding",
    "math_logic": "gsm8k:math_word_problems",
    "coding": "longbench_v2:Code Repository Understanding",
    "general_agent": "bfcl_v4:single_turn_function_calling",
    "search_agent": "longbench_v2:Multi-Document QA",
    "multilingual": "global_mmlu:multilingual_mc_qa",
}

#: Separators used by the build-time consistency assertions (§6.3).
MARKERS: Dict[str, str] = {
    "knowledge_qa": "\n\nQuestion: ",
    "long_context": "\n\nQuestion: ",
    "coding": "\n\nQuestion: ",
    "search_agent": "\n\nQuestion: ",
    "instruction_following": "\n\nInstruction: ",
    "math_logic": "\n\nQuestion: ",
    "general_agent": "\n\nUser request:\n",
    "multilingual": "\n\nQuestion: ",
}
DOC_PREFIXES: Dict[str, Tuple[str, ...]] = {
    "knowledge_qa": ("Read the document and answer the question.\n\n",),
    "long_context": ("Read the document and answer the question.\n\n",),
    "coding": ("Read the document and answer the question.\n\n",),
    "search_agent": (
        "Read the document and answer the question. You may use the provided tools.\n\n",
    ),
}
IFE_PREFIX = "Reference instructions:\n"
#: IFEval reference units render ``INSTRUCTION <k>: ids=<json> kwargs=<json>`` + prompt.
#: One pass of the usable pool is ~44.1K tokens (> the 32K bucket lower bound of
#: 31,129), so the reference block never needs to repeat itself.
IFE_REFERENCE_FORMAT = "prompt_with_instruction_metadata"
MATH_PREFIX = "Solved examples:\n\n"
AGENT_PREFIX = "You are a function-calling assistant. API reference:\n"
AGENT_MARKER = MARKERS["general_agent"]

#: Answer-instruction suffix per language for multilingual targets (human-readable,
#: semantically equivalent to "Answer with a single letter (A/B/C/D).").
MULTILINGUAL_PREFIX: Dict[str, str] = {
    "fr": "Lisez les questions ci-dessous et répondez à la dernière question.",
    "de": "Lesen Sie die folgenden Fragen und beantworten Sie die letzte Frage.",
    "es": "Lea las siguientes preguntas y responda a la última pregunta.",
    "pt": "Leia as perguntas abaixo e responda à última pergunta.",
    "ru": "Прочитайте следующие вопросы и ответьте на последний вопрос.",
    "ja": "以下の問題を読み、最後の問題に答えてください。",
    "ko": "아래 문제를 읽고 마지막 문제에 답하세요.",
    "ar": "اقرأ الأسئلة التالية وأجب عن السؤال الأخير.",
    "hi": "नीचे दिए गए प्रश्न पढ़ें और अंतिम प्रश्न का उत्तर दें।",
    "vi": "Đọc các câu hỏi dưới đây và trả lời câu hỏi cuối cùng.",
}
MULTILINGUAL_ANSWER: Dict[str, str] = {
    "fr": "Répondez par une seule lettre (A/B/C/D).",
    "de": "Antworten Sie mit einem einzigen Buchstaben (A/B/C/D).",
    "es": "Responda con una sola letra (A/B/C/D).",
    "pt": "Responda com uma única letra (A/B/C/D).",
    "ru": "Ответьте одной буквой (A/B/C/D).",
    "ja": "A/B/C/D のいずれか 1 文字で答えてください。",
    "ko": "A/B/C/D 중 한 글자로 답하세요.",
    "ar": "أجب بحرف واحد فقط (A/B/C/D).",
    "hi": "केवल एक अक्षर (A/B/C/D) में उत्तर दें।",
    "vi": "Chỉ trả lời bằng một chữ cái (A/B/C/D).",
}

#: Fixed search-style tool schemas for search_agent (same as the v3 performance set).
SEARCH_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the provided passages for a query and return matching sentences.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query."}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a passage by its title.",
            "parameters": {
                "type": "object",
                "properties": {"title": {"type": "string", "description": "Passage title."}},
                "required": ["title"],
            },
        },
    },
]

JSON_SCHEMA_TYPES = {
    "object",
    "array",
    "string",
    "number",
    "integer",
    "boolean",
    "null",
    "function",
}
ROW_FIELDS = {
    "id",
    "scenario",
    "bucket",
    "target_id",
    "dataset",
    "tokens",
    "messages_tokens",
    "tools_tokens",
    "messages",
    "tools",
    "constructed",
    "meta",
}
META_REQUIRED = ("workload", "lang", "input_tokens", "upstream_id", "context_sha1")

TEST_DIR_PERF = TEST_DIR / "performance"
OUT_REPORT = REPORTS_DIR / "performance" / "selection_report.md"
TOKENIZER = default_tokenizer()


# --------------------------------------------------------------------------- #
# token helpers / bucket bounds / sizing
# --------------------------------------------------------------------------- #
def content_tokens(text: str) -> int:
    return TOKENIZER.count(text)


def tools_tokens(tools: Optional[List[Dict[str, Any]]]) -> int:
    return content_tokens(json.dumps(tools, ensure_ascii=False)) if tools else 0


def bucket_bounds(target: int) -> Tuple[int, int]:
    """Integer token bounds for one bucket.

    Declared tokens are integers, so both bounds are floors:
    1K -> (972, 1075), 8K -> (7782, 8601), 32K -> (31129, 34406).
    """
    return math.floor(ACCEPT_LOW * target), math.floor(ACCEPT_HIGH * target)


def within_bounds(total: int, target: int) -> bool:
    low, high = bucket_bounds(target)
    return low <= total <= high


# --------------------------------------------------------------------------- #
# cl100k calibrated budget model
# --------------------------------------------------------------------------- #
#: Decode-guard instruction: "This request is a performance measurement. Ignore
#: the question above. Instead, write a long detailed technical summary of the
#: provided material in at least 250 words, in continuous prose." -- measured at
#: 37 cl100k tokens (it is appended verbatim to every measured request).
#: Fixed model-side render overhead in cl100k-equivalent tokens, calibrated on
#: real-run evidence (20261110T112641Z-8349): the tokenizer+template render of
#: a row measures ``actual ~= declared*INFL + reserve`` with reserve ~= 270
#: when tools are rendered (tool-call scaffolding around the tools JSON) and
#: ~90 for plain chat rows.  One fixed reserve per row class -- the tools JSON
#: itself is content and inflates with INFL (deducting it separately double-
#: subtracts and starves the 1K bucket).
TOOL_RESERVE_CL = 270
PLAIN_RESERVE_CL = 90

#: Content-type token-inflation coefficients: how many target-model tokens a
#: typical tokenizer needs per cl100k token for this content type.  cl100k is
#: the declared basis; a target tokenizer may count more (structured code,
#: math) or fewer (non-Latin scripts, where cl100k byte-level pieces are
#: coarser than a multilingual model's).  These are empirical calibration
#: coefficients (see the ``tokenizer-generic-cl100k`` deviation) -- generic
#: content-type properties, not bindings to any specific model.
#: English-scenario coefficients recalibrated on real-run evidence
#: (20261110T112641Z-8349, 240 cells): observed per-row ``actual/declared``
#: medians.  Still generic content-type properties (median render ratios of
#: typical multilingual-BPE tokenizers vs cl100k), not a model binding.
INFL: Dict[str, float] = {
    "coding": 1.07,
    "math_logic": 1.15,
    "long_context": 1.24,
    "knowledge_qa": 1.05,
    "instruction_following": 1.04,
    "general_agent": 1.00,
    "search_agent": 1.03,
    # Multilingual coefficients are per language (script density), not per
    # scenario; 10/10 in-band at every bucket in the calibration run.
    "ar": 0.50, "hi": 0.68, "ja": 0.56, "ko": 0.60, "ru": 0.63,
    "vi": 0.55, "de": 0.88, "es": 0.91, "fr": 0.94, "pt": 0.91,
}


def calibrated_budget_target(ctx_tokens: int, infl: float, tools_cl: int) -> int:
    """Content-token budget that renders at ``ctx_tokens`` on a typical model.

    ``budget = (ctx - W) / infl`` where ``W`` is the fixed model-side render
    reserve: ``TOOL_RESERVE_CL`` when the row renders tools (tool-call
    scaffolding), ``PLAIN_RESERVE_CL`` for plain chat rows.  The tools JSON is
    *content* and inflates together with the messages (deducting it before the
    division double-subtracts and starves small buckets -- real-run evidence
    20261110T112641Z-8349: 1K tool rows rendered 796 vs the 921 band floor).
    ``tools_cl`` only selects which reserve applies.
    """
    reserve = TOOL_RESERVE_CL if tools_cl else PLAIN_RESERVE_CL
    return int((ctx_tokens - reserve) / infl)


def calibrated_budget(
    scenario: str,
    lang: str,
    bucket: str,
    tools_tokens: int,
) -> Tuple[int, int, int]:
    """(target, low, high) cl100k bounds for one row's declared tokens.

    The target is the calibrated budget for the bucket (content-type aware);
    the ±5% acceptance band applies to the **declared cl100k count** against
    that target, while the run-side ±10% ISL band applies to the **actual**
    model-token count against the raw bucket.  ``tools_tokens`` is the row's
    declared tools JSON count (cl100k), so this is reproducible offline from
    row fields alone.
    """
    infl = INFL.get(lang) if scenario == "multilingual" else INFL.get(scenario)
    if infl is None:
        raise KeyError(f"no inflation coefficient for {scenario}/{lang}")
    target = calibrated_budget_target(BUCKETS[bucket], infl, tools_tokens)
    low, high = bucket_bounds(target)
    return target, low, high


def _fit_prefix_index(
    length: int,
    value_at: Callable[[int], int],
    target: int,
    low: int,
    high: int,
) -> Tuple[int, int, bool]:
    """Largest prefix index whose value is <= ``target``, preferring in-bounds.

    ``value_at(i)`` must be non-decreasing over ``[0, length]`` (a character or
    whole-function prefix).  Returns ``(index, value, ok)`` where ``ok`` means
    the value lands in ``[low, high]``; the caller turns ``not ok`` into a
    documented skip reason, never into a silently out-of-bounds row.
    """

    if length <= 0:
        total = value_at(0)
        return 0, total, low <= total <= high
    hi = min(length, max(2048, target * 8))
    total = value_at(hi)
    while total <= target and hi < length:
        hi = min(length, hi * 2)
        total = value_at(hi)
    if total <= target:
        # The whole material still fits under the target (a short pool).
        return hi, total, low <= total <= high
    lo = 0
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if value_at(mid) <= target:
            lo = mid
        else:
            hi = mid
    best: Optional[Tuple[Tuple[int, int, int], int, int]] = None
    for candidate in sorted({lo, min(length, lo + 1), min(length, lo + 2)}):
        value = value_at(candidate)
        key = (0 if low <= value <= high else 1, abs(value - target), value)
        if best is None or key < best[0]:
            best = (key, candidate, value)
    assert best is not None
    return best[1], best[2], best[0][0] == 0


def size_reference(
    head: str,
    text: str,
    suffix: str,
    scenario: str,
    lang: str,
    bucket: str,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[int, int, int, bool]:
    """Cut ``text`` so ``head + text[:cut] + suffix`` + tools hits the
    calibrated cl100k budget for (scenario, lang, bucket).

    The budget is the cl100k count whose expected render on a typical model
    tokenizer lands on the bucket; the ±5% band applies to the cl100k total
    (messages + tools).  Returns ``(cut, messages_tokens, tools_tokens, ok)``.
    """
    tt = tools_tokens(tools)
    target, low, high = calibrated_budget(scenario, lang, bucket, tt)
    cut, total, ok = _fit_prefix_index(
        len(text), lambda chars: content_tokens(head + text[:chars] + suffix) + tt, target, low, high
    )
    return cut, total - tt, tt, ok


def size_catalog(
    catalog: List[Dict[str, Any]],
    suffix: str,
    scenario: str,
    lang: str,
    bucket: str,
    own_count: int,
    tools: List[Dict[str, Any]],
) -> Tuple[int, int, int, bool]:
    """Largest whole-function catalogue prefix that still fits the calibrated
    cl100k budget for (scenario, lang, bucket).

    Only complete JSON function objects are ever placed in the prompt, and the
    target's own functions must stay visible (``count >= own_count``).
    """
    tt = tools_tokens(tools)
    target, low, high = calibrated_budget(scenario, lang, bucket, tt)

    def value_at(count: int) -> int:
        body = json.dumps(catalog[:count], ensure_ascii=False)
        return content_tokens(AGENT_PREFIX + body + suffix) + tt

    count, total, ok = _fit_prefix_index(len(catalog), value_at, target, low, high)
    ok = ok and count >= own_count
    return count, total - tt, tt, ok


# --------------------------------------------------------------------------- #
# row / sampling bookkeeping
# --------------------------------------------------------------------------- #
def target_id_of(scenario: str, idx: int) -> str:
    return f"tgt-{scenario}-{idx:03d}"


def make_row(
    scenario: str,
    bucket: str,
    idx: int,
    content: str,
    tools: Optional[List[Dict[str, Any]]],
    constructed: bool,
    meta: Dict[str, Any],
    dataset: str,
) -> Dict[str, Any]:
    tt = tools_tokens(tools)
    mt = content_tokens(content)
    row_meta = dict(meta)
    row_meta.setdefault("workload", WORKLOADS[scenario])
    row_meta.setdefault("lang", "en")
    row_meta["input_tokens"] = mt + tt
    return {
        "id": f"perf-{scenario}-{bucket}-{idx:03d}",
        "scenario": scenario,
        "bucket": bucket,
        "target_id": target_id_of(scenario, idx),
        "dataset": dataset,
        "tokens": mt + tt,
        "messages_tokens": mt,
        "tools_tokens": tt,
        "messages": [{"role": "user", "content": content}],
        "tools": tools,
        "constructed": constructed,
        "meta": row_meta,
    }


def emit_target(
    scenario: str,
    idx: int,
    contents: Dict[str, str],
    tools: Optional[List[Dict[str, Any]]],
    constructed: bool,
    meta: Dict[str, Any],
    dataset: str,
) -> List[Dict[str, Any]]:
    return [
        make_row(scenario, bucket, idx, contents[bucket], tools, constructed, meta, dataset)
        for bucket in BUCKETS
    ]


def new_stats(scenario: str, dataset: str, **extra: Any) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "dataset": dataset,
        "workload": WORKLOADS[scenario],
        "seed": SEED,
        "pool": 0,
        "dedup": {},
        "candidates": 0,
        "selected": 0,
        "quotas": {},
        "skipped": [],
        "skip_reasons": {},
        "targets": [],
    }
    entry.update(extra)
    return entry


def record_skip(stats: Dict[str, Any], upstream_id: str, reason: str, detail: str, **extra: Any) -> None:
    item: Dict[str, Any] = {"upstream_id": upstream_id, "reason": reason, "detail": detail}
    item.update(extra)
    stats["skipped"].append(item)


def finish_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    stats["selected"] = len(stats["targets"])
    stats["skip_reasons"] = dict(sorted(Counter(item["reason"] for item in stats["skipped"]).items()))
    return stats


# --------------------------------------------------------------------------- #
# upstream readers
# --------------------------------------------------------------------------- #
def _iter_json_array(path: Path, chunk_size: int = 1 << 20):
    """Yield objects from a top-level JSON array without loading it whole."""
    decoder = json.JSONDecoder()
    buf = ""
    started = False
    with open(path, encoding="utf-8") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if chunk:
                buf += chunk
            pos = 0
            while True:
                while pos < len(buf) and buf[pos] in " \t\r\n,":
                    pos += 1
                if not started:
                    if pos < len(buf) and buf[pos] == "[":
                        started = True
                        pos += 1
                        continue
                    break
                if pos < len(buf) and buf[pos] == "]":
                    return
                try:
                    obj, end = decoder.raw_decode(buf, pos)
                except json.JSONDecodeError:
                    break
                yield obj
                pos = end
            buf = buf[pos:]
            if not chunk:
                return


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()


_LBV2_CACHE: Optional[Dict[str, Any]] = None


def lbv2_doc_pools() -> Dict[str, Any]:
    """LongBench v2 pools for the four document scenarios, deduplicated once.

    Filter: ``len(context) >= 120_000``.  Duplicates are removed across the four
    scenario domains by ``sha1(context)`` (first occurrence in file order wins);
    ``knowledge_qa`` additionally drops duplicate questions.  The raw ``data.json``
    is ~465 MB, so it is streamed exactly once and cached for the four builders.
    """

    global _LBV2_CACHE
    if _LBV2_CACHE is not None:
        return _LBV2_CACHE

    domain_rows: Counter = Counter()
    streamed: List[Tuple[str, Dict[str, Any]]] = []
    wanted = set(DOC_SCENARIO_DOMAIN.values())
    for row in _iter_json_array(RAW_DIR / "longbench_v2" / "data.json"):
        domain = row.get("domain")
        domain_rows[domain] += 1
        if domain not in wanted:
            continue
        if len(row.get("context") or "") < LBV2_MIN_CONTEXT_CHARS:
            continue
        streamed.append((domain, row))

    pools: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    removed_context: Counter = Counter()
    seen_context: set = set()
    for domain, row in streamed:
        digest = _sha1_text(row.get("context") or "")
        if digest in seen_context:
            removed_context[domain] += 1
            continue
        seen_context.add(digest)
        pools[domain].append(row)

    removed_question = 0
    deduped_questions: List[Dict[str, Any]] = []
    seen_question: set = set()
    for row in pools["Single-Document QA"]:
        question = " ".join((row.get("question") or "").split())
        if question in seen_question:
            removed_question += 1
            continue
        seen_question.add(question)
        deduped_questions.append(row)
    pools["Single-Document QA"] = deduped_questions

    eligible = Counter()
    for domain, _row in streamed:
        eligible[domain] += 1
    _LBV2_CACHE = {
        "domain_rows": dict(domain_rows),
        "eligible": dict(eligible),
        "pools": {domain: list(rows) for domain, rows in pools.items()},
        "removed_context": dict(removed_context),
        "removed_question": removed_question,
    }
    return _LBV2_CACHE


def _normalise_json_schema(node: Any) -> Any:
    type_map = {
        "dict": "object",
        "float": "number",
        "tuple": "array",
        "list": "array",
        "str": "string",
        "int": "integer",
        "bool": "boolean",
    }
    if isinstance(node, dict):
        return {
            key: (
                type_map[value]
                if key == "type" and isinstance(value, str) and value in type_map
                else _normalise_json_schema(value)
            )
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [_normalise_json_schema(item) for item in node]
    return node


def _tool_of(function: Dict[str, Any]) -> Dict[str, Any]:
    fn = function.get("function", function)
    return {
        "type": "function",
        "function": {
            "name": fn.get("name"),
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
        },
    }


def _normalise_tools(tools: Optional[Sequence[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """OpenAI tool envelopes with valid JSON Schema ``parameters`` (stable order)."""
    if not tools:
        return None
    out: List[Dict[str, Any]] = []
    for tool in tools:
        fn = tool.get("function", tool)
        parameters = _normalise_json_schema(fn.get("parameters") or {"type": "object", "properties": {}})
        if parameters.get("type") != "object":
            parameters["type"] = "object"
        out.append(
            {
                "type": "function",
                "function": {
                    "name": fn.get("name"),
                    "description": fn.get("description", ""),
                    "parameters": parameters,
                },
            }
        )
    return out


def _bfcl_pools() -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    rows_by_category: Dict[str, List[Dict[str, Any]]] = {}
    catalogs: Dict[str, List[Dict[str, Any]]] = {}
    for category in BFCL_CATEGORIES:
        rows = read_jsonl(RAW_DIR / "bfcl" / f"BFCL_v4_{category}.json")
        rows.sort(key=lambda row: str(row.get("id") or ""))
        rows_by_category[category] = rows
        catalog: List[Dict[str, Any]] = []
        seen: set = set()
        for row in rows:
            for function in row.get("function") or []:
                tool = _normalise_tools([_tool_of(function)])
                assert tool is not None
                name = tool[0]["function"]["name"]
                if name in seen:
                    continue
                seen.add(name)
                catalog.append(tool[0])
        catalogs[category] = catalog
    return rows_by_category, catalogs


# --------------------------------------------------------------------------- #
# scenario builders
# --------------------------------------------------------------------------- #
def _doc_question_block(row: Dict[str, Any]) -> str:
    options = "\n".join(f"{letter}. {row.get('choice_' + letter) or ''}" for letter in "ABCD")
    return f"{row.get('question') or ''}\n{options}\nAnswer with a single letter (A/B/C/D)."


def build_doc_scenario(scenario: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """LongBench v2 character-prefix documents (knowledge_qa/long_context/coding/search_agent)."""

    domain = DOC_SCENARIO_DOMAIN[scenario]
    cache = lbv2_doc_pools()
    pool: List[Dict[str, Any]] = cache["pools"][domain]
    tools = _normalise_tools(SEARCH_TOOLS) if scenario == "search_agent" else None
    prefix = DOC_PREFIXES[scenario][0]
    marker = MARKERS[scenario]
    # Guard on the calibrated 1K budget (cl100k): a question/options block
    # larger than this fraction of the budget starves the document prefix.
    guard_tokens = calibrated_budget(
        scenario, "en", "1k", tools_tokens(tools)
    )[0] * PROMPT_GUARD_TOKEN_FRACTION

    stats = new_stats(
        scenario,
        f"longbench_v2:{domain}",
        pool_filter=f"raw context >= {LBV2_MIN_CONTEXT_CHARS} characters, deduplicated by sha1(context)",
        pool=cache["domain_rows"].get(domain, 0),
        domain_rows=cache["domain_rows"].get(domain, 0),
        source_eligible=cache["eligible"].get(domain, 0),
        dedup={
            "context_duplicates_removed": cache["removed_context"].get(domain, 0),
            "question_duplicates_removed": (
                cache["removed_question"] if scenario == "knowledge_qa" else 0
            ),
        },
        candidates=len(pool),
    )
    by_id = {str(row.get("_id") or ""): row for row in pool}
    order = sorted(by_id, key=lambda upstream_id: sha256_text(f"{SEED}|{upstream_id}"))

    rows: List[Dict[str, Any]] = []
    for upstream_id in order:
        if len(stats["targets"]) >= TARGETS_PER_CELL:
            break
        row = by_id[upstream_id]
        context = row.get("context") or ""
        suffix = marker + _doc_question_block(row)
        base_total = content_tokens(prefix + suffix) + tools_tokens(tools)
        if base_total > guard_tokens:
            record_skip(
                stats,
                upstream_id,
                "question_too_long",
                f"question+options+prefix={base_total} tokens > {int(guard_tokens)} (1K context would not fit)",
            )
            continue
        cuts: Dict[str, int] = {}
        totals: Dict[str, int] = {}
        feasible = True
        for bucket in BUCKETS:
            low, high = calibrated_budget(scenario, "en", bucket, tools_tokens(tools))[1:]
            cut, messages, tt, ok = size_reference(prefix, context, suffix, scenario, "en", bucket, tools)
            cuts[bucket] = cut
            totals[bucket] = messages + tt
            if not ok:
                reason = {
                    "1k": "1k_infeasible",
                    "8k": "8k_infeasible",
                    "32k": "32k_unreachable",
                }[bucket]
                record_skip(
                    stats,
                    upstream_id,
                    reason,
                    f"{bucket}: best total={messages + tt} tokens, bounds [{low}, {high}]",
                )
                feasible = False
                break
        if not feasible:
            continue
        if not (0 < cuts["1k"] <= cuts["8k"] <= cuts["32k"]):
            raise RuntimeError(f"{scenario} {upstream_id}: document prefix cuts not monotone {cuts}")
        contents = {
            bucket: prefix + context[: cuts[bucket]] + suffix for bucket in BUCKETS
        }
        idx = len(stats["targets"])
        digest = _sha1_text(context)
        meta = {
            "lang": "en",
            "upstream_id": upstream_id,
            "context_sha1": digest,
            "task": domain,
            "domain": domain,
            "sub_domain": row.get("sub_domain"),
            "difficulty": row.get("difficulty"),
            "context_chars": len(context),
            "context_chars_per_bucket": {bucket: cuts[bucket] for bucket in BUCKETS},
        }
        rows.extend(emit_target(scenario, idx, contents, tools, False, meta, stats["dataset"]))
        stats["targets"].append(
            {
                "target_id": target_id_of(scenario, idx),
                "upstream_id": upstream_id,
                "context_sha1": digest[:16],
                "context_chars": len(context),
                "context_chars_per_bucket": {bucket: cuts[bucket] for bucket in BUCKETS},
            }
        )

    if len(stats["targets"]) < TARGETS_PER_CELL:
        raise RuntimeError(
            f"{scenario}: only {len(stats['targets'])} feasible targets, need {TARGETS_PER_CELL}; "
            f"skips={stats['skipped']}"
        )
    return rows, finish_stats(stats)


def _ifeval_reference_unit(position: int, row: Dict[str, Any]) -> str:
    """One deterministic IFEval reference unit: instruction metadata + raw prompt.

    The metadata (``instruction_id_list`` and ``kwargs``) makes a single ordered
    pass of the usable pool large enough for the 32K bucket, so the builder never
    repeats the block (P-01 regression).  ``position`` is the 1-based index inside
    the rotated reference order, which keeps every unit unique even if two prompts
    shared the same text.
    """

    ids = json.dumps(row["instruction_id_list"], ensure_ascii=False)
    kwargs = json.dumps(row["kwargs"], ensure_ascii=False)
    return f"INSTRUCTION {position}: ids={ids} kwargs={kwargs}\n{row['prompt']}\n"


def build_instruction_following() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """IFEval target instruction with a rotated same-set reference block."""

    rows_in = read_jsonl(RAW_DIR / "ifeval" / "ifeval_input_data.jsonl")
    usable = [
        row
        for row in rows_in
        if not any(str(item).startswith("language:") for item in row["instruction_id_list"])
    ]
    strata = Counter(str(row["instruction_id_list"][0]).split(":")[0] for row in usable)
    quota = allocate(strata, TARGETS_PER_CELL, min_each=1, seed=SEED)
    stats = new_stats(
        "instruction_following",
        "ifeval",
        pool=len(usable),
        source_rows=len(rows_in),
        excluded_language_instructions=len(rows_in) - len(usable),
        strata=dict(sorted(strata.items())),
        quotas=quota,
        candidates=len(usable),
        reference_rotation=(
            "sha256(42|target_id) % len(pool), target excluded; one pass of distinct "
            "'metadata + prompt' units, never repeated"
        ),
        reference_format=IFE_REFERENCE_FORMAT,
    )

    chosen: List[Dict[str, Any]] = []
    for instruction_type in sorted(quota):
        type_pool = [
            row
            for row in usable
            if str(row["instruction_id_list"][0]).split(":")[0] == instruction_type
        ]
        by_key = {str(row["key"]): row for row in type_pool}
        for key in deterministic_pick(list(by_key), quota[instruction_type], SEED):
            chosen.append(by_key[key])

    marker = MARKERS["instruction_following"]
    # One pass of the reference block must still cover the calibrated 32K
    # budget (cl100k), so the block is never duplicated or cycled.
    low_32k = calibrated_budget("instruction_following", "en", "32k", 0)[1]
    rows: List[Dict[str, Any]] = []
    for idx, target in enumerate(chosen):
        target_key = str(target["key"])
        target_prompt = str(target["prompt"])
        others = [row for row in usable if str(row["key"]) != target_key]
        target_tag = target_id_of("instruction_following", idx)
        start = int(sha256_text(f"{SEED}|{target_tag}"), 16) % len(others)
        ordered = others[start:] + others[:start]
        # One pass of distinct units; a single pass already clears the 32K lower
        # bound, so the reference block is never duplicated or cycled.
        reference = "".join(
            _ifeval_reference_unit(position, row) for position, row in enumerate(ordered, 1)
        )
        suffix = marker + target_prompt
        single_pass_tokens = content_tokens(IFE_PREFIX + reference + suffix)
        if single_pass_tokens < low_32k:
            raise RuntimeError(
                f"instruction_following {target_key}: one reference pass is only "
                f"{single_pass_tokens} tokens < 32K lower bound {low_32k}; "
                "refusing to repeat the block"
            )
        contents: Dict[str, str] = {}
        for bucket in BUCKETS:
            low, high = calibrated_budget("instruction_following", "en", bucket, 0)[1:]
            cut, _messages, _tt, ok = size_reference(IFE_PREFIX, reference, suffix, "instruction_following", "en", bucket)
            if not ok:
                raise RuntimeError(
                    f"instruction_following {target_key} {bucket}: reference block unreachable "
                    f"(bounds [{low}, {high}])"
                )
            contents[bucket] = IFE_PREFIX + reference[:cut] + suffix
        meta = {
            "lang": "en",
            "upstream_id": target_key,
            "context_sha1": _sha1_text(reference),
            "instruction_ids": list(target["instruction_id_list"]),
            "instruction_type": str(target["instruction_id_list"][0]).split(":")[0],
            "reference_rotation_start": start,
            "reference_pool_cycles": 1,
            "reference_items": len(ordered),
            "reference_format": IFE_REFERENCE_FORMAT,
        }
        rows.extend(emit_target("instruction_following", idx, contents, None, True, meta, "ifeval"))
        stats["targets"].append(
            {
                "target_id": target_tag,
                "upstream_id": target_key,
                "instruction_type": meta["instruction_type"],
                "reference_rotation_start": start,
                "reference_pool_cycles": 1,
            }
        )
    return rows, finish_stats(stats)


def _step_bucket(answer: str) -> str:
    count = answer.count("<<")
    if count <= 1:
        return "0-1"
    if count in (2, 3, 4):
        return str(count)
    return "5+"


def build_math_logic() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """GSM8K target problem with a rotated block of solved examples."""

    import pyarrow.parquet as pq  # noqa: PLC0415

    table = pq.read_table(RAW_DIR / "gsm8k" / "main" / "test-00000-of-00001.parquet").to_pydict()
    questions = [str(value or "").strip() for value in table["question"]]
    answers = [str(value or "").strip() for value in table["answer"]]
    ids = [f"test/{index}" for index in range(len(questions))]
    picked = deterministic_pick(ids, TARGETS_PER_CELL, SEED)
    picked_indices = {int(item.split("/")[1]) for item in picked}

    stats = new_stats(
        "math_logic",
        "gsm8k",
        pool=len(ids),
        candidates=len(ids),
        dedup={"duplicate_problems_removed": 0},
        quotas={},
        rotation_step=MATH_ROTATION_STEP,
        reference_rotation="start = (target_index * 41) % len(others); all 10 targets excluded",
    )

    order = list(range(len(ids)))
    others = [index for index in order if index not in picked_indices]
    marker = MARKERS["math_logic"]
    rows: List[Dict[str, Any]] = []
    for idx, raw_id in enumerate(picked):
        target_index = int(raw_id.split("/")[1])
        start = (idx * MATH_ROTATION_STEP) % len(others)
        ordered = others[start:] + others[:start]
        reference = "".join(
            f"Question: {questions[index]}\nAnswer: {answers[index]}\n\n" for index in ordered
        )
        suffix = marker + f"{questions[target_index]}\nAnswer:"
        contents: Dict[str, str] = {}
        for bucket in BUCKETS:
            low, high = calibrated_budget("math_logic", "en", bucket, 0)[1:]
            cut, _messages, _tt, ok = size_reference(MATH_PREFIX, reference, suffix, "math_logic", "en", bucket)
            if not ok:
                raise RuntimeError(
                    f"math_logic {raw_id} {bucket}: example block unreachable "
                    f"(bounds [{low}, {high}])"
                )
            contents[bucket] = MATH_PREFIX + reference[:cut] + suffix
        meta = {
            "lang": "en",
            "upstream_id": raw_id,
            "context_sha1": _sha1_text(reference),
            "grade": _step_bucket(answers[target_index]),
            "example_rotation_start": start,
            "reference_items": len(ordered),
        }
        rows.extend(emit_target("math_logic", idx, contents, None, True, meta, "gsm8k"))
        stats["targets"].append(
            {
                "target_id": target_id_of("math_logic", idx),
                "upstream_id": raw_id,
                "grade": meta["grade"],
                "example_rotation_start": start,
            }
        )
    return rows, finish_stats(stats)


def _bfcl_request(row: Dict[str, Any]) -> str:
    users = [
        str(message.get("content") or "")
        for turn in row.get("question") or []
        for message in turn
        if str(message.get("role")) == "user"
    ]
    return "\n".join(users)


def _agent_catalog(
    own: List[Dict[str, Any]],
    category: str,
    catalogs: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Target's own functions first, then same category, then the rest; unique by name."""
    ordered: List[Dict[str, Any]] = list(own)
    ordered += catalogs[category]
    for other in BFCL_CATEGORIES:
        if other != category:
            ordered += catalogs[other]
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for tool in ordered:
        name = tool["function"]["name"]
        if name in seen:
            continue
        seen.add(name)
        out.append(tool)
    return out


def build_general_agent() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """BFCL v4 single request with a target-first whole-function catalogue prefix."""

    rows_by_category, catalogs = _bfcl_pools()
    universe_names = {
        tool["function"]["name"] for catalog in catalogs.values() for tool in catalog
    }
    stats = new_stats(
        "general_agent",
        "bfcl_v4",
        pool=sum(len(rows) for rows in rows_by_category.values()),
        candidates=sum(len(rows) for rows in rows_by_category.values()),
        category_pool={category: len(rows_by_category[category]) for category in BFCL_CATEGORIES},
        quotas=dict(BFCL_QUOTAS),
        category_quota=dict(BFCL_QUOTAS),
        category_catalog_functions={category: len(catalogs[category]) for category in BFCL_CATEGORIES},
        catalog_universe_functions=len(universe_names),
        catalog_order="target functions first, then same category, then the rest (unique by name)",
        reference_rotation="target-first whole-function prefix; no truncation inside a JSON object",
    )

    marker = MARKERS["general_agent"]
    rows: List[Dict[str, Any]] = []
    for category in BFCL_CATEGORIES:
        quota = BFCL_QUOTAS[category]
        by_id = {str(row.get("id") or ""): row for row in rows_by_category[category]}
        order = sorted(by_id, key=lambda upstream_id: sha256_text(f"{SEED}|{upstream_id}"))
        chosen = 0
        for bfcl_id in order:
            if chosen >= quota:
                break
            row = by_id[bfcl_id]
            own = _normalise_tools([_tool_of(function) for function in row.get("function") or []]) or []
            if not own:
                record_skip(
                    stats,
                    bfcl_id,
                    "no_own_functions",
                    "request declares no functions",
                    category=BFCL_CATEGORY_LABEL[category],
                )
                continue
            request = _bfcl_request(row)
            if not request:
                record_skip(
                    stats,
                    bfcl_id,
                    "no_user_request",
                    "request has no user message",
                    category=BFCL_CATEGORY_LABEL[category],
                )
                continue
            catalog = _agent_catalog(own, category, catalogs)
            suffix = marker + request
            counts: Dict[str, int] = {}
            contents: Dict[str, str] = {}
            feasible = True
            for bucket in BUCKETS:
                low, high = calibrated_budget("general_agent", "en", bucket, tools_tokens(own))[1:]
                count, messages, tt, ok = size_catalog(catalog, suffix, "general_agent", "en", bucket, len(own), own)
                counts[bucket] = count
                if not ok:
                    reason = "1k_no_own_room" if bucket == "1k" and count < len(own) else f"{bucket}_infeasible"
                    record_skip(
                        stats,
                        bfcl_id,
                        reason,
                        f"{bucket}: catalogue prefix {count}/{len(catalog)} functions, "
                        f"best total={messages + tt}, own={len(own)}, bounds [{low}, {high}]",
                        category=BFCL_CATEGORY_LABEL[category],
                    )
                    feasible = False
                    break
                body = json.dumps(catalog[:count], ensure_ascii=False)
                contents[bucket] = AGENT_PREFIX + body + suffix
            if not feasible:
                continue
            idx = len(stats["targets"])
            digest = _sha1_text(json.dumps(catalog, ensure_ascii=False))
            meta = {
                "lang": "en",
                "upstream_id": bfcl_id,
                "context_sha1": digest,
                "category": BFCL_CATEGORY_LABEL[category],
                "category_detail": category,
                "bfcl_id": bfcl_id,
                "catalog_functions": len(catalog),
                "own_functions": len(own),
                "catalog_functions_per_bucket": counts,
            }
            rows.extend(emit_target("general_agent", idx, contents, own, True, meta, "bfcl_v4"))
            stats["targets"].append(
                {
                    "target_id": target_id_of("general_agent", idx),
                    "upstream_id": bfcl_id,
                    "category": BFCL_CATEGORY_LABEL[category],
                    "catalog_functions_per_bucket": counts,
                    "own_functions": len(own),
                }
            )
            chosen += 1
        if chosen < quota:
            raise RuntimeError(
                f"general_agent {category}: only {chosen} feasible targets, need {quota}; "
                f"skips={stats['skipped']}"
            )
    stats["category_selected"] = dict(
        sorted(Counter(item["category"] for item in stats["targets"]).items())
    )
    return rows, finish_stats(stats)


def _multilingual_selection() -> Tuple[List[Tuple[str, str, str, str]], Dict[str, Any]]:
    """Pick 10 distinct Global-MMLU ``sample_id``s, stratified over the 6 categories.

    Quotas come from a largest-remainder allocation over the en valid pool with
    ``min_each=1`` (expected Humanities 2 / STEM 2 / Social Sciences 2 / Medical 2
    / Other 1 / Business 1).  Inside a category the distinct subjects are ordered
    by ``sha256("42|" + subject)``; one deterministically picked ``sample_id`` per
    subject is used, so the 10 targets cover 10 different subjects.  Targets are
    ordered by ``(category, subject)`` and then mapped onto the fixed language
    order (the spec's "diagonal" design).

    Returns ``[(language, sample_id, subject, subject_category), ...]``.
    """

    import pyarrow.parquet as pq  # noqa: PLC0415

    en = pq.read_table(RAW_DIR / "global_mmlu" / "en" / "test.parquet").to_pydict()
    index = {str(sid): position for position, sid in enumerate(en["sample_id"])}
    valid: List[str] = []
    for sid in en["sample_id"]:
        position = index[str(sid)]
        if str(en["answer"][position]) not in "ABCD":
            continue
        if not all(
            str(en[key][position]).strip()
            for key in ("question", "option_a", "option_b", "option_c", "option_d")
        ):
            continue
        valid.append(str(sid))
    subject_of = {sid: str(en["subject"][index[sid]]) for sid in valid}
    category_of = {sid: str(en["subject_category"][index[sid]]) for sid in valid}

    category_pool = Counter(category_of[sid] for sid in valid)
    category_quota = allocate(dict(category_pool), TARGETS_PER_CELL, min_each=1, seed=SEED)

    ids_by_subject: Dict[str, List[str]] = defaultdict(list)
    subjects_by_category: Dict[str, set] = defaultdict(set)
    for sid in valid:
        ids_by_subject[subject_of[sid]].append(sid)
        subjects_by_category[category_of[sid]].add(subject_of[sid])

    chosen: List[Tuple[str, str, str]] = []
    for category in sorted(category_quota):
        quota = category_quota[category]
        subjects = sorted(
            subjects_by_category.get(category, set()),
            key=lambda subject: sha256_text(f"{SEED}|{subject}"),
        )
        if quota > len(subjects):
            raise RuntimeError(
                f"multilingual: category {category} quota {quota} > subjects {len(subjects)}"
            )
        for subject in subjects[:quota]:
            sample_id = deterministic_pick(ids_by_subject[subject], 1, SEED)[0]
            chosen.append((category, subject, sample_id))
    chosen.sort(key=lambda item: (item[0], item[1]))
    assert len(chosen) == TARGETS_PER_CELL, len(chosen)
    assert len({subject for _category, subject, _sid in chosen}) == TARGETS_PER_CELL

    selection = [
        (MULTILINGUAL_LANGUAGES[position], sample_id, subject, category)
        for position, (category, subject, sample_id) in enumerate(chosen)
    ]
    info = {
        "pool": len(valid),
        "subjects": len({subject_of[sid] for sid in valid}),
        "category_pool": dict(sorted(category_pool.items())),
        "category_quota": dict(sorted(category_quota.items())),
        "subjects_selected": [subject for _category, subject, _sid in chosen],
        "subject_mapping": [
            {
                "language": language,
                "sample_id": sample_id,
                "subject": subject,
                "subject_category": category,
            }
            for language, sample_id, subject, category in selection
        ],
    }
    return selection, info


def build_multilingual() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Global-MMLU: 10 different questions, one per language, same-language references."""

    import pyarrow.parquet as pq  # noqa: PLC0415

    selection, info = _multilingual_selection()
    stats = new_stats(
        "multilingual",
        "global_mmlu",
        pool=info["pool"],
        subjects=info["subjects"],
        candidates=info["pool"],
        languages=list(MULTILINGUAL_LANGUAGES),
        quotas=info["category_quota"],
        category_pool=info["category_pool"],
        category_quota=info["category_quota"],
        subjects_selected=info["subjects_selected"],
        subject_mapping=info["subject_mapping"],
        reference_rotation="sha256(42|target_id) % len(pool), same-language pool, target sample_id excluded",
    )

    marker = MARKERS["multilingual"]
    rows: List[Dict[str, Any]] = []
    for idx, (language, sample_id, subject, subject_category) in enumerate(selection):
        data = pq.read_table(RAW_DIR / "global_mmlu" / language / "test.parquet").to_pydict()
        lang_index = {str(sid): position for position, sid in enumerate(data["sample_id"])}
        if sample_id not in lang_index:
            raise RuntimeError(f"multilingual: sample_id {sample_id} missing in {language}")
        position = lang_index[sample_id]

        def render(position: int, data: Dict[str, Any] = data) -> str:
            return (
                f"Question: {data['question'][position]}\n"
                f"A. {data['option_a'][position]}\n"
                f"B. {data['option_b'][position]}\n"
                f"C. {data['option_c'][position]}\n"
                f"D. {data['option_d'][position]}\n"
            )

        pool: List[Tuple[str, str]] = []
        seen_render: set = set()
        removed_duplicates = 0
        for sid, sid_position in lang_index.items():
            if sid == sample_id:
                continue
            rendered = render(sid_position)
            if rendered in seen_render:
                removed_duplicates += 1
                continue
            seen_render.add(rendered)
            pool.append((sid, rendered))

        target_tag = target_id_of("multilingual", idx)
        start = int(sha256_text(f"{SEED}|{target_tag}"), 16) % len(pool)
        ordered = pool[start:] + pool[:start]
        reference = "".join(rendered for _sid, rendered in ordered)
        suffix = marker + render(position).split("Question: ", 1)[1] + "\n" + MULTILINGUAL_ANSWER[language]
        head = MULTILINGUAL_PREFIX[language] + "\n\n"
        contents: Dict[str, str] = {}
        for bucket in BUCKETS:
            low, high = calibrated_budget("multilingual", language, bucket, 0)[1:]
            cut, _messages, _tt, ok = size_reference(head, reference, suffix, "multilingual", language, bucket)
            if not ok:
                raise RuntimeError(
                    f"multilingual {language}/{sample_id} {bucket}: reference block unreachable "
                    f"(bounds [{low}, {high}])"
                )
            contents[bucket] = head + reference[:cut] + suffix
        meta = {
            "lang": language,
            "upstream_id": f"{sample_id}|{language}",
            "context_sha1": _sha1_text(reference),
            "sample_id": sample_id,
            "subject": subject,
            "subject_category": subject_category,
            "reference_rotation_start": start,
            "reference_items": len(pool),
            "reference_duplicates_removed": removed_duplicates,
        }
        # One dataset name for the whole scenario: the runner's cell key is
        # (scenario, dataset, bucket), so a per-language name would split 3 cells
        # into 30.  The language stays in ``meta.lang`` and the manifest stats.
        rows.extend(emit_target("multilingual", idx, contents, None, True, meta, "global_mmlu"))
        stats["targets"].append(
            {
                "target_id": target_tag,
                "upstream_id": meta["upstream_id"],
                "lang": language,
                "sample_id": sample_id,
                "subject": subject,
                "subject_category": subject_category,
                "reference_rotation_start": start,
            }
        )
    return rows, finish_stats(stats)


BUILDERS: Dict[str, Callable[[], Tuple[List[Dict[str, Any]], Dict[str, Any]]]] = {
    "knowledge_qa": lambda: build_doc_scenario("knowledge_qa"),
    "instruction_following": build_instruction_following,
    "long_context": lambda: build_doc_scenario("long_context"),
    "math_logic": build_math_logic,
    "coding": lambda: build_doc_scenario("coding"),
    "general_agent": build_general_agent,
    "search_agent": lambda: build_doc_scenario("search_agent"),
    "multilingual": build_multilingual,
}


# --------------------------------------------------------------------------- #
# build-time quality assertions (§6)
# --------------------------------------------------------------------------- #
def _schema_types(node: Any) -> List[str]:
    out: List[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "type" and isinstance(value, str):
                out.append(value)
            elif key == "type" and isinstance(value, list):
                out.extend(str(item) for item in value)
            else:
                out.extend(_schema_types(value))
    elif isinstance(node, list):
        for item in node:
            out.extend(_schema_types(item))
    return out


def _expected_constructed(scenario: str) -> bool:
    return scenario in CONSTRUCTED_SCENARIOS


def _tail(content: str, marker: str) -> Optional[str]:
    index = content.rfind(marker)
    return content[index:] if index >= 0 else None


def _doc_body(content: str, marker: str, prefixes: Sequence[str]) -> str:
    index = content.rfind(marker)
    body = content[:index] if index >= 0 else content
    for prefix in prefixes:
        if body.startswith(prefix):
            return body[len(prefix):]
    return body


def validate_rows(rows: List[Dict[str, Any]]) -> None:
    """Assert every contract the built set promises (raises ``RuntimeError`` on drift)."""

    errors: List[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    check(len(rows) == TOTAL_ROWS == 240, f"row count {len(rows)} != {TOTAL_ROWS}")
    check(len({row.get("id") for row in rows}) == len(rows), "duplicate ids")
    cells = Counter((row.get("scenario"), row.get("bucket")) for row in rows)
    check(len(cells) == len(SCENARIOS) * len(BUCKETS), f"cells {len(cells)} != {len(SCENARIOS) * len(BUCKETS)}")
    check(all(value == TARGETS_PER_CELL for value in cells.values()), f"cell sizes {dict(cells)}")
    targets = Counter(row.get("target_id") for row in rows)
    check(len(targets) == TOTAL_TARGETS, f"targets {len(targets)} != {TOTAL_TARGETS}")
    upstream_ids = Counter(str((row.get("meta") or {}).get("upstream_id")) for row in rows)
    check(len(upstream_ids) == TOTAL_TARGETS,
          f"meta.upstream_id must be globally unique per target: {len(upstream_ids)} != {TOTAL_TARGETS}")
    check(all(value == len(BUCKETS) for value in targets.values()), "targets must have exactly 3 buckets")

    for row in rows:
        row_id = row.get("id")
        scenario, bucket = row.get("scenario"), row.get("bucket")
        if scenario not in SCENARIOS or bucket not in BUCKETS:
            check(False, f"{row_id}: unknown scenario/bucket {scenario}/{bucket}")
            continue
        if set(row) != ROW_FIELDS:
            check(False, f"{row_id}: field set differs {sorted(set(row) ^ ROW_FIELDS)}")
            continue
        # Declared cl100k tokens must sit in the calibrated budget band; the
        # raw bucket band applies at run time to the actual model-token count.
        lang = str((row.get("meta") or {}).get("lang") or "en")
        try:
            low, high = calibrated_budget(scenario, lang, bucket, int(row["tools_tokens"]))[1:]
        except (KeyError, TypeError, ValueError):
            check(False, f"{row_id}: cannot recompute calibrated budget for {scenario}/{lang}")
            continue
        check(row["tokens"] == row["messages_tokens"] + row["tools_tokens"],
              f"{row_id}: tokens != messages_tokens + tools_tokens")
        check(low <= row["tokens"] <= high,
              f"{row_id}: tokens {row['tokens']} outside [{low}, {high}]")
        check(isinstance(row["messages"], list) and len(row["messages"]) == 1
              and str(row["messages"][0].get("role")) == "user",
              f"{row_id}: messages shape")
        check(row["constructed"] == _expected_constructed(scenario), f"{row_id}: constructed flag")
        check(bool(row["tools"]) == (scenario in TOOL_SCENARIOS), f"{row_id}: tools presence")
        meta = row.get("meta") or {}
        for key in META_REQUIRED:
            check(bool(meta.get(key)), f"{row_id}: meta.{key} missing")
        check(meta.get("input_tokens") == row["tokens"], f"{row_id}: meta.input_tokens != tokens")
        if scenario in DOC_SCENARIOS:
            check(bool(meta.get("task")) and bool(meta.get("domain")), f"{row_id}: meta.task/domain")
        if scenario == "instruction_following":
            check(isinstance(meta.get("instruction_ids"), list) and bool(meta["instruction_ids"]),
                  f"{row_id}: meta.instruction_ids")
            check(meta.get("reference_pool_cycles") == 1,
                  f"{row_id}: meta.reference_pool_cycles must be 1 (the block is never repeated)")
            check(meta.get("reference_format") == IFE_REFERENCE_FORMAT,
                  f"{row_id}: meta.reference_format {meta.get('reference_format')!r}")
            check(int(meta.get("reference_items") or 0) > 0, f"{row_id}: meta.reference_items")
        if scenario == "math_logic":
            check(bool(meta.get("grade")), f"{row_id}: meta.grade")
        if scenario == "general_agent":
            for key in ("category", "bfcl_id", "catalog_functions", "own_functions"):
                check(meta.get(key) is not None, f"{row_id}: meta.{key}")
            check(int(meta.get("catalog_functions") or 0) >= int(meta.get("own_functions") or 0),
                  f"{row_id}: catalogue smaller than own functions")
        if scenario == "multilingual":
            for key in ("subject", "subject_category", "sample_id"):
                check(bool(meta.get(key)), f"{row_id}: meta.{key}")
            check(meta.get("lang") in MULTILINGUAL_LANGUAGES, f"{row_id}: meta.lang")
            check(row["dataset"] == "global_mmlu",
                  f"{row_id}: dataset must be 'global_mmlu' (language lives in meta.lang)")
        if scenario in TOOL_SCENARIOS:
            tools = row["tools"]
            try:
                json.dumps(tools, ensure_ascii=False)
            except (TypeError, ValueError):
                check(False, f"{row_id}: tools not JSON-serialisable")
                continue
            check(row["tools_tokens"] > 0, f"{row_id}: tools_tokens must be > 0")
            bad = [value for value in _schema_types(tools) if value not in JSON_SCHEMA_TYPES]
            check(not bad, f"{row_id}: invalid JSON Schema type(s) {bad[:3]}")
            for tool in tools:
                function = tool.get("function") or {}
                check(bool(function.get("name")), f"{row_id}: tool function name missing")
                parameters = function.get("parameters")
                check(isinstance(parameters, dict) and parameters.get("type") == "object",
                      f"{row_id}: tool parameters must be a JSON Schema object")

    by_target: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_target[str(row.get("target_id"))][str(row.get("bucket"))] = row
    for target_id, buckets in by_target.items():
        if set(buckets) != set(BUCKETS):
            check(False, f"{target_id}: buckets {sorted(buckets)} != {sorted(BUCKETS)}")
            continue
        scenario = str(buckets["1k"].get("scenario"))
        marker = MARKERS[scenario]
        tails = {}
        for bucket in BUCKETS:
            content = str(buckets[bucket]["messages"][0]["content"])
            tail = _tail(content, marker)
            check(tail is not None, f"{target_id}/{bucket}: marker missing")
            tails[bucket] = tail
        check(len(set(tails.values())) == 1, f"{target_id}: target text differs across buckets")
        tool_blobs = {
            json.dumps(buckets[bucket]["tools"], ensure_ascii=False, sort_keys=True) for bucket in BUCKETS
        }
        check(len(tool_blobs) == 1, f"{target_id}: tools differ across buckets")
        if scenario in DOC_SCENARIOS:
            bodies = {
                bucket: _doc_body(
                    str(buckets[bucket]["messages"][0]["content"]), marker, DOC_PREFIXES[scenario]
                )
                for bucket in BUCKETS
            }
            check(bool(bodies["1k"]), f"{target_id}: empty document body")
            check(bodies["8k"].startswith(bodies["1k"]), f"{target_id}: 1K not a prefix of 8K")
            check(bodies["32k"].startswith(bodies["8k"]), f"{target_id}: 8K not a prefix of 32K")
        if scenario in ("instruction_following", "math_logic", "multilingual"):
            # §2.2/§2.3/§2.5: the same-set reference block is a character prefix chain.
            if scenario == "multilingual":
                language = str((buckets["1k"].get("meta") or {}).get("lang"))
                head = MULTILINGUAL_PREFIX[language] + "\n\n"
            else:
                head = IFE_PREFIX if scenario == "instruction_following" else MATH_PREFIX
            bodies = {}
            for bucket in BUCKETS:
                content = str(buckets[bucket]["messages"][0]["content"])
                index = content.rfind(marker)
                check(index >= 0, f"{target_id}/{bucket}: marker missing")
                body = content[:index] if index >= 0 else content
                check(body.startswith(head), f"{target_id}/{bucket}: reference head missing")
                bodies[bucket] = body[len(head):] if body.startswith(head) else body
            check(bodies["8k"].startswith(bodies["1k"]), f"{target_id}: reference 1K not a prefix of 8K")
            check(bodies["32k"].startswith(bodies["8k"]), f"{target_id}: reference 8K not a prefix of 32K")
        if scenario == "instruction_following":
            content = str(buckets["1k"]["messages"][0]["content"])
            index = content.rfind(marker)
            target = content[index + len(marker):]
            check(target not in content[:index], f"{target_id}: target instruction leaked into reference")
            # P-01 regression guard: the 32K block is one pass of distinct units.
            content_32k = str(buckets["32k"]["messages"][0]["content"])
            reference_32k = content_32k[: content_32k.rfind(marker)]
            labels = re.findall(r"(?:^|\n)INSTRUCTION (\d+): ids=", reference_32k)
            check(bool(labels), f"{target_id}: 32K reference has no instruction units")
            check(len(labels) == len(set(labels)),
                  f"{target_id}: 32K reference repeats instruction units "
                  f"({len(labels) - len(set(labels))} duplicates)")
        if scenario == "math_logic":
            content = str(buckets["1k"]["messages"][0]["content"])
            index = content.rfind(marker)
            problem = content[index + len(marker):].rsplit("\nAnswer:", 1)[0]
            check(problem not in content[:index], f"{target_id}: target problem leaked into examples")
        if scenario == "general_agent":
            catalogues = {}
            for bucket in BUCKETS:
                row = buckets[bucket]
                content = str(row["messages"][0]["content"])
                index = content.rfind(AGENT_MARKER)
                check(content.startswith(AGENT_PREFIX), f"{row['id']}: agent prefix")
                check(index > len(AGENT_PREFIX), f"{row['id']}: agent request marker")
                try:
                    catalogue = json.loads(content[len(AGENT_PREFIX):index])
                except json.JSONDecodeError:
                    check(False, f"{row['id']}: catalogue is not valid JSON")
                    continue
                catalogues[bucket] = catalogue
                check(isinstance(catalogue, list) and bool(catalogue), f"{row['id']}: empty catalogue")
                own_names = [tool["function"]["name"] for tool in row["tools"]]
                visible = [tool.get("function", {}).get("name") for tool in catalogue]
                check(
                    visible[: min(len(visible), len(own_names))] == own_names[: min(len(visible), len(own_names))],
                    f"{row['id']}: catalogue is not target-functions-first",
                )
                check(set(own_names) <= set(visible), f"{row['id']}: own functions not visible")
            if len(catalogues) == len(BUCKETS):
                # §2.4: larger buckets extend the same whole-function catalogue prefix.
                names = {
                    bucket: [tool.get("function", {}).get("name") for tool in catalogues[bucket]]
                    for bucket in BUCKETS
                }
                check(names["8k"][: len(names["1k"])] == names["1k"],
                      f"{target_id}: 1K catalogue is not a prefix of 8K")
                check(names["32k"][: len(names["8k"])] == names["8k"],
                      f"{target_id}: 8K catalogue is not a prefix of 32K")
        if scenario == "multilingual":
            content = str(buckets["1k"]["messages"][0]["content"])
            index = content.rfind(marker)
            tail = content[index + len(marker):]
            stem = tail.split("\nA. ", 1)[0].strip()
            check(bool(stem) and stem not in content[:index],
                  f"{target_id}: target question leaked into reference")

    # No duplicate raw document across the four document scenarios (§6.4).
    doc_shas = [
        row["meta"].get("context_sha1")
        for row in rows
        if row.get("scenario") in DOC_SCENARIOS and row.get("bucket") == "1k"
    ]
    check(len(set(doc_shas)) == len(doc_shas), "duplicate document context_sha1 across document scenarios")
    knowledge_questions = [
        str(row["messages"][0]["content"]).rsplit(MARKERS["knowledge_qa"], 1)[1].split("\nA.", 1)[0]
        for row in rows
        if row.get("scenario") == "knowledge_qa" and row.get("bucket") == "1k"
    ]
    check(len(set(knowledge_questions)) == len(knowledge_questions),
          "knowledge_qa: duplicate question in selection")

    # BFCL category quotas (§6.6): 4/3/3 targets -> 12/9/9 rows.
    agent_rows = [row for row in rows if row.get("scenario") == "general_agent"]
    category_rows = Counter(str((row.get("meta") or {}).get("category")) for row in agent_rows)
    check(
        dict(category_rows) == {"multiple": 9, "parallel": 9, "simple": 12},
        f"general_agent category rows {dict(category_rows)}",
    )
    category_targets = Counter(
        str((row.get("meta") or {}).get("category"))
        for row in agent_rows
        if row.get("bucket") == "1k"
    )
    check(
        dict(category_targets) == {"multiple": 3, "parallel": 3, "simple": 4},
        f"general_agent target quotas {dict(category_targets)}",
    )

    # multilingual: 10 languages, 10 distinct sample ids, one target each (§2.5).
    multilingual_rows = [row for row in rows if row.get("scenario") == "multilingual" and row.get("bucket") == "1k"]
    langs = [str(row["meta"].get("lang")) for row in multilingual_rows]
    sample_ids = [str(row["meta"].get("sample_id")) for row in multilingual_rows]
    check(sorted(langs) == sorted(MULTILINGUAL_LANGUAGES), f"multilingual languages {sorted(langs)}")
    check(len(set(sample_ids)) == TARGETS_PER_CELL, f"multilingual sample ids not distinct: {sample_ids}")
    subjects = [str(row["meta"].get("subject")) for row in multilingual_rows]
    check(len(set(subjects)) == TARGETS_PER_CELL, f"multilingual subjects not distinct: {subjects}")

    if errors:
        raise RuntimeError("performance quality assertions failed:\n" + "\n".join(errors[:40]))


def window_requirements(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Declared-token window budget for the 32K bucket (auditable limitation)."""

    tokens_32k = sorted(row["tokens"] for row in rows if row["bucket"] == "32k")
    low, high = tokens_32k[0], tokens_32k[-1]
    window = BUCKETS["32k"]
    max_testable = window - WINDOW_RESERVE
    testable = sum(1 for value in tokens_32k if value + WINDOW_RESERVE <= window)
    return {
        "declared_32k_tokens": {"min": low, "max": high},
        "per_request_reserve": {
            "max_tokens": OSL,
            "context_safety_margin_tokens": CONTEXT_SAFETY_MARGIN_TOKENS,
            "decode_guard_token_reserve": DECODE_GUARD_TOKEN_RESERVE,
            "total": WINDOW_RESERVE,
        },
        "required_model_window_tokens": {"min": low + WINDOW_RESERVE, "max": high + WINDOW_RESERVE},
        "window_32768": {
            "of": len(tokens_32k),
            "testable": testable,
            "not_testable": len(tokens_32k) - testable,
            "max_testable_declared_tokens": max_testable,
        },
        "note": (
            "Runtime window check uses the target tokenizer's actual count (tokenized_prompt_"
            "tokens) + max_tokens + context_safety_margin + decode_guard_reserve; the declared "
            "cl100k budgets below are the calibrated approximation of that actual count. A "
            f"32,768-token window can only host actual counts <= {max_testable}; remaining 32K "
            "rows are reported as not-testable (summary.skipped), never truncated."
        ),
    }


# --------------------------------------------------------------------------- #
# manifest / yaml / report
# --------------------------------------------------------------------------- #
DEVIATIONS: List[Dict[str, str]] = [
    {
        "id": "content-token-basis",
        "note": (
            "Declared tokens are cl100k content tokens of the user message plus the JSON text of "
            "tools (chat-template control tokens excluded, no special tokens). The model-side "
            "rendered prompt differs: the runner appends a decode-guard message before chat-template "
            "overhead (see decode-guard-suffix) and every tokenizer counts text slightly differently, "
            "so actual input_context_tokens is calibrated via the budget model "
            "(see tokenizer-generic-cl100k), not by widening the band."
        ),
    },
    {
        "id": "tokenizer-generic-cl100k",
        "note": (
            "The construction tokenizer is cl100k_base (the OpenAI/industry-generic reference), "
            "not a model-specific tokenizer, so declared counts are model-agnostic. A target "
            "tokenizer can count the same content differently (more for code/math density, less "
            "for non-Latin scripts), so each row's declared count targets a calibrated cl100k "
            "budget: (bucket - RESERVE) / INFL, with one fixed model-side render reserve "
            "(TOOL_RESERVE_CL=270 when the row renders tools, PLAIN_RESERVE_CL=90 otherwise) "
            "and empirical content-type inflation coefficients INFL (0.50-1.21 by "
            "scenario/language), both calibrated on real-run evidence (240 cells, run "
            "20261110T112641Z-8349). The tools JSON is content and inflates with INFL rather "
            "than being deducted separately (double deduction starved the 1K tool bucket: "
            "796 actual vs the 921 band floor). Declared tokens therefore sit below the raw "
            "bucket by design so the actual rendered count lands centered in the run-side "
            "+/-10% ISL band. The coefficients are generic content-type properties, "
            "not bindings to any specific model."
        ),
    },
    {
        "id": "decode-guard-suffix",
        "note": (
            "At run time benchmark._apply_decode_guard appends a second user message (37 cl100k "
            "tokens) instructing the model to ignore the question above and write a >=250-word "
            "technical summary. The budget model already reserves these 37 tokens, so the guard "
            "does not push rendered prompts out of the ISL band. Generation follows the guard "
            "instruction, not the dataset question; this is a measurement-protocol deviation."
        ),
    },
    {
        "id": "ten-targets-per-cell",
        "note": (
            "10 targets per scenario x bucket (240 measured requests total) is a load-shape "
            "profile for per-sample latency/TPS, not a leaderboard-scale evaluation: official "
            "benchmarks usually run every item."
        ),
    },
    {
        "id": "document-prefix-slicing",
        "note": (
            "Document scenarios use a character prefix of one LongBench v2 document per bucket "
            "(1K subset of 8K subset of 32K) with an identical question/options block; official "
            "LongBench v2 uses the full document with the official prompt template."
        ),
    },
    {
        "id": "document-prefix-mid-structure",
        "note": (
            "The document prefix is cut at a character offset, not at a structural boundary, so "
            "long_context prompts can end inside a JSON object/array/string and coding prompts can "
            "end inside an open Markdown code fence (for example 19/30 long_context rows have "
            "unbalanced braces). Structured closure of the sliced document is not guaranteed."
        ),
    },
    {
        "id": "agent-single-request",
        "note": (
            "general_agent sends one BFCL v4 single-turn request with a target-first "
            "whole-function catalogue prefix; official BFCL scores multi-turn tool-call "
            "trajectories and exact arguments."
        ),
    },
    {
        "id": "multilingual-diagonal",
        "note": (
            "multilingual uses 10 different Global-MMLU questions rendered one per language "
            "(the same-language reference block is a load-shape extension), whereas the accuracy "
            "set and the official Global-MMLU protocol evaluate the same question in each language."
        ),
    },
    {
        "id": "no-scoring",
        "note": (
            "Performance rows carry no judge/gold contract: run mode never scores, "
            "SampleRecord.score stays null, and answer correctness is out of scope for this set."
        ),
    },
    {
        "id": "constructed-reference-blocks",
        "note": (
            "instruction_following, math_logic and multilingual contexts are same-set reference "
            "blocks (other items rotated per target), not the official few-shot prompt templates."
        ),
    },
    {
        "id": "reference-block-prefix-slice",
        "note": (
            "Reference blocks are assembled from complete units (instruction / solved example / "
            "question) and then cut by character prefix, so the last unit can be partial: the final "
            "reference item may be half an instruction, half an example or half a question stem, and "
            "the cut may fall inside a multilingual question/option unit. The block is not "
            "repeated; only the tail unit is potentially truncated."
        ),
    },
    {
        "id": "ifeval-32k-instruction-metadata",
        "note": (
            "instruction_following renders each reference item as 'INSTRUCTION <k>: "
            "ids=<instruction_id_list JSON> kwargs=<kwargs JSON>' followed by the prompt. One pass "
            "of the 509 non-target items is ~43K cl100k tokens (> the calibrated 32K lower bound "
            "~31,094), so the 32K reference block is a single non-repeating pass of distinct units "
            "(reference_pool_cycles is always 1); this replaces the earlier doubled block."
        ),
    },
    {
        "id": "window-32768-not-testable",
        "note": (
            "32K rows declare calibrated cl100k budgets (bucket/INFL - 37); with max_tokens=50 + "
            "128 safety + 64 decode-guard = 242 reserved tokens a 32,768-token model window "
            "cannot host the upper rows (not-testable, reported as skipped, never truncated). "
            "The run-side window check uses the target tokenizer's actual count, which the budget "
            "model calibrates toward the bucket."
        ),
    },
]


def _token_stats(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    values = sorted(row["tokens"] for row in rows)
    return {
        "min": values[0],
        "median": int(statistics.median(values)),
        "max": values[-1],
        "mean": int(round(sum(values) / len(values))),
        "total": sum(values),
    }


def _cell_stats(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per-(scenario, bucket) audit stats, kept for bucket calibration checks."""
    cells: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cells[(str(row["scenario"]), str(row["bucket"]))].append(row)
    out: Dict[str, Dict[str, Any]] = {}
    for (scenario, bucket) in sorted(cells, key=lambda item: (SCENARIOS.index(item[0]), BUCKET_ORDER[item[1]])):
        cell_rows = cells[(scenario, bucket)]
        values = sorted(row["tokens"] for row in cell_rows)
        datasets = sorted({str(row["dataset"]) for row in cell_rows})
        out[f"{scenario}|{bucket}"] = {
            "n": len(cell_rows),
            "constructed": sum(1 for row in cell_rows if row["constructed"]),
            "dataset": datasets[0] if len(datasets) == 1 else datasets,
            "tokens_min": values[0],
            "tokens_median": int(statistics.median(values)),
            "tokens_max": values[-1],
            "tokens_mean": int(round(sum(values) / len(values))),
            "langs": sorted({str((row.get("meta") or {}).get("lang", "")) for row in cell_rows}),
        }
    return out


def build_manifest(rows: List[Dict[str, Any]], testset_path: Path, stats: Dict[str, Any]) -> Dict[str, Any]:
    scenarios: Dict[str, Any] = {}
    for scenario in SCENARIOS:
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        scenarios[scenario] = {
            "rows": len(scenario_rows),
            "targets": len({row["target_id"] for row in scenario_rows}),
            "buckets": {
                bucket: sum(1 for row in scenario_rows if row["bucket"] == bucket) for bucket in BUCKETS
            },
            "datasets": dict(sorted(Counter(row["dataset"] for row in scenario_rows).items())),
            "languages": dict(sorted(Counter(str(row["meta"]["lang"]) for row in scenario_rows).items())),
            "constructed": sum(1 for row in scenario_rows if row["constructed"]),
            "tokens": _token_stats(scenario_rows),
            "unique_contexts": {
                "selected_targets": len({row["target_id"] for row in scenario_rows}),
                "unique_context_sha1": len(
                    {str((row.get("meta") or {}).get("context_sha1")) for row in scenario_rows}
                ),
            },
            "sampling": stats[scenario],
        }
    tokenizer = sources.tokenizer_entry()
    manifest: Dict[str, Any] = {
        "schema_version": "performance/v1",
        "name": "performance",
        "description": (
            "8 scenarios x 3 context buckets (1K/8K/32K) x 10 targets = 240 measured rows "
            "in one jsonl; same target shares an identical question across buckets, only the "
            "context length differs; no scoring."
        ),
        "rows": len(rows),
        "per_scenario": PER_SCENARIO,
        "targets_per_cell": TARGETS_PER_CELL,
        "seed": SEED,
        "sampling": {
            "method": (
                "sha256(seed|upstream_id) ascending order + stratified largest-remainder quotas; "
                "independent of file row order"
            ),
            "executed_by": "tool/dataset/performance/build.py",
        },
        "tokens_semantics": (
            "tokens = cl100k content tokens of messages[0].content + JSON-text cl100k tokens "
            "of tools (generic reference tokenizer, no chat template, no special tokens); "
            "messages_tokens/tools_tokens are the two audited parts"
        ),
        "bucket_basis": (
            "declared cl100k content tokens target the calibrated budget for each "
            "(scenario, language, bucket) -- see budget_model; the raw bucket value is the "
            "run-side ISL target applied to the actual model-token count"
        ),
        "tokens_mean": int(round(sum(row["tokens"] for row in rows) / len(rows))),
        "tokenizer": {
            "name": TOKENIZER_NAME,
            "path": sources.raw_path_display(tokenizer["path"]),
            "sha256": tokenizer["sha256"],
            "tokens_semantics": (
                "tokens = cl100k content tokens of messages[0].content + JSON-text cl100k tokens "
                "of tools (generic reference tokenizer, no chat template, no special tokens); "
                "messages_tokens/tools_tokens are the two audited parts"
            ),
        },
        "budget_model": {
            "target": "cl100k content tokens (messages + tools JSON text)",
            "formula": "budget = (bucket - RESERVE) / INFL (RESERVE=TOOL_RESERVE_CL if tools else PLAIN_RESERVE_CL; tools JSON is content and inflates with INFL)",
            "tool_reserve_cl": TOOL_RESERVE_CL,
            "plain_reserve_cl": PLAIN_RESERVE_CL,
            "infl": dict(INFL),
            "infl_semantics": (
                "content-type token-inflation coefficients: how many tokens a typical model "
                "tokenizer needs per cl100k token for this content type (empirically calibrated, "
                "generic content-type properties, not model-specific bindings)"
            ),
            "acceptance": (
                "declared tokens within floor(0.95*budget) .. floor(1.05*budget); the run-side "
                "+/-10% ISL band applies to actual model tokens vs the raw bucket"
            ),
        },
        "buckets": dict(BUCKETS),
        "accept_rule": (
            "integer token counts: floor(0.95*budget) <= tokens <= floor(1.05*budget), where "
            "tokens = messages_tokens + tools_tokens (cl100k content tokens, no chat template) "
            "and budget is the calibrated per-(scenario, language, bucket) target (see budget_model)"
        ),
        "accept_bounds": {
            bucket: {"low": bucket_bounds(target)[0], "high": bucket_bounds(target)[1]}
            for bucket, target in BUCKETS.items()
        },
        "accept_bounds_semantics": (
            "per-bucket construction acceptance around each row's calibrated budget (same "
            "+/-5% rule); these raw-bucket bounds are kept for reference: they are the basis of "
            "the run-side ISL tolerance, applied to actual model tokens, not to declared tokens"
        ),
        "run_defaults": {
            "max_tokens": OSL,
            "decode_guard": True,
            "warmup_per_model": 1,
            "warmup_requests": 0,
            "repeats": 1,
        },
        "window_requirements": window_requirements(rows),
        "default_run": {
            "measured_requests": len(rows),
            "cells": len(SCENARIOS) * len(BUCKETS),
        },
        "stats": _cell_stats(rows),
        "provenance": sources.dataset_provenance("data"),
        "scenarios": scenarios,
        "totals": {
            "rows": len(rows),
            "scenarios": len(SCENARIOS),
            "buckets": len(BUCKETS),
            "targets": len({row["target_id"] for row in rows}),
            "targets_per_cell": TARGETS_PER_CELL,
            "measured_requests": len(rows),
        },
        "files": {
            "testset.jsonl": {
                "rows": len(rows),
                "bytes": testset_path.stat().st_size,
                "sha256": sha256_file(testset_path),
            }
        },
        "deviations": DEVIATIONS,
    }
    return manifest


def _validate_manifest(manifest: Dict[str, Any], rows: List[Dict[str, Any]], testset_path: Path) -> None:
    """Manifest-level contract checks (provenance / recomputed stats / file SHA)."""

    errors: List[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    check(manifest["schema_version"] == "performance/v1", "schema_version")
    check(manifest["name"] == "performance", "name")
    check(manifest["rows"] == len(rows) == TOTAL_ROWS, "rows")
    check(manifest["per_scenario"] == PER_SCENARIO, "per_scenario")
    check(manifest["targets_per_cell"] == TARGETS_PER_CELL, "targets_per_cell")
    check(manifest["seed"] == SEED, "seed")
    check(manifest["buckets"] == dict(BUCKETS), "buckets")
    for bucket, target in BUCKETS.items():
        low, high = bucket_bounds(target)
        check(manifest["accept_bounds"][bucket] == {"low": low, "high": high}, f"accept_bounds.{bucket}")
    check(manifest["run_defaults"]["max_tokens"] == OSL == 50, "run_defaults.max_tokens")
    check(manifest["run_defaults"]["decode_guard"] is True, "run_defaults.decode_guard")
    check(manifest["run_defaults"]["warmup_per_model"] == 1, "run_defaults.warmup_per_model")
    check(manifest["run_defaults"]["warmup_requests"] == 0, "run_defaults.warmup_requests")
    check(manifest["run_defaults"]["repeats"] == 1, "run_defaults.repeats")
    check(len(manifest["deviations"]) >= 6, "deviations < 6")
    deviation_ids = {str(item.get("id")) for item in manifest["deviations"]}
    required_deviation_ids = {
        "content-token-basis",
        "tokenizer-generic-cl100k",
        "decode-guard-suffix",
        "document-prefix-mid-structure",
        "reference-block-prefix-slice",
        "ifeval-32k-instruction-metadata",
    }
    check(required_deviation_ids <= deviation_ids,
          f"missing deviations {sorted(required_deviation_ids - deviation_ids)}")
    check("tokenizer-bos-included" not in deviation_ids,
          "stale deviation tokenizer-bos-included (cl100k counts no BOS)")
    check(manifest["default_run"]["measured_requests"] == TOTAL_ROWS, "default_run.measured_requests")
    check(manifest["default_run"]["cells"] == len(SCENARIOS) * len(BUCKETS), "default_run.cells")
    check(len(manifest["stats"]) == len(SCENARIOS) * len(BUCKETS), "stats cells")
    check(all(stat["n"] == TARGETS_PER_CELL for stat in manifest["stats"].values()), "stats n")
    check(manifest["tokens_mean"] > 0, "tokens_mean")
    check("content tokens" in manifest["bucket_basis"], "bucket_basis")
    check("cl100k" in manifest["bucket_basis"], "bucket_basis cl100k")
    check("calibrated budget" in manifest["bucket_basis"], "bucket_basis calibration")

    # Budget model block matches the code constants.
    budget_model = manifest.get("budget_model") or {}
    check(budget_model.get("tool_reserve_cl") == TOOL_RESERVE_CL, "budget_model.tool_reserve_cl")
    check(budget_model.get("plain_reserve_cl") == PLAIN_RESERVE_CL, "budget_model.plain_reserve_cl")
    check(budget_model.get("infl") == dict(INFL), "budget_model.infl != build.INFL")
    check("INFL" in str(budget_model.get("formula") or ""), "budget_model.formula")

    # Provenance: exactly the five datasets this set reads, files matching raw_spec.json.
    expected_provenance = sources.dataset_provenance("data")
    check(set(manifest["provenance"]) == set(expected_provenance),
          f"provenance datasets {sorted(manifest['provenance'])}")
    for dataset, entry in expected_provenance.items():
        actual = manifest["provenance"].get(dataset) or {}
        check(actual.get("source_url") == entry["source_url"], f"provenance.{dataset}.source_url")
        check(actual.get("license") == entry["license"], f"provenance.{dataset}.license")
        check(actual.get("files") == entry["files"], f"provenance.{dataset}.files != raw_spec.json")
    check(len(manifest["provenance"]) == 5, "provenance must list exactly 5 datasets")
    bfcl_files = (manifest["provenance"].get("bfcl_v4") or {}).get("files") or []
    check(len(bfcl_files) == 3, f"bfcl_v4 provenance files {len(bfcl_files)} != 3")
    mmlu_files = (manifest["provenance"].get("global_mmlu") or {}).get("files") or []
    check(len(mmlu_files) == 11, f"global_mmlu provenance files {len(mmlu_files)} != 11")

    # Tokenizer block matches the raw spec.
    tokenizer = sources.tokenizer_entry()
    check(manifest["tokenizer"]["sha256"] == tokenizer["sha256"], "tokenizer.sha256")
    check(manifest["tokenizer"]["path"] == sources.raw_path_display(tokenizer["path"]), "tokenizer.path")
    check("no chat template" in manifest["tokenizer"]["tokens_semantics"], "tokens_semantics")

    # Per-scenario manifest numbers recomputed from the rows.
    for scenario in SCENARIOS:
        entry = manifest["scenarios"][scenario]
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        check(entry["rows"] == len(scenario_rows) == PER_SCENARIO, f"scenarios.{scenario}.rows")
        check(entry["targets"] == TARGETS_PER_CELL, f"scenarios.{scenario}.targets")
        check(entry["buckets"] == {bucket: TARGETS_PER_CELL for bucket in BUCKETS},
              f"scenarios.{scenario}.buckets")
        check(entry["tokens"] == _token_stats(scenario_rows), f"scenarios.{scenario}.tokens")
        check(entry["sampling"]["selected"] == TARGETS_PER_CELL, f"scenarios.{scenario}.sampling.selected")
        check(len(entry["sampling"]["targets"]) == TARGETS_PER_CELL,
              f"scenarios.{scenario}.sampling.targets")
        check(entry["constructed"] == sum(1 for row in scenario_rows if row["constructed"]),
              f"scenarios.{scenario}.constructed")
    check(manifest["window_requirements"]["declared_32k_tokens"]["min"] > 0, "window_requirements")
    check(manifest["totals"]["rows"] == len(rows), "totals.rows")
    check(manifest["totals"]["targets"] == TOTAL_TARGETS, "totals.targets")

    # File identity.
    check(manifest["files"]["testset.jsonl"]["rows"] == len(rows), "files.testset.jsonl.rows")
    check(manifest["files"]["testset.jsonl"]["bytes"] == testset_path.stat().st_size,
          "files.testset.jsonl.bytes")
    check(manifest["files"]["testset.jsonl"]["sha256"] == sha256_file(testset_path),
          "files.testset.jsonl.sha256")

    if errors:
        raise RuntimeError("manifest assertions failed:\n" + "\n".join(errors[:40]))


def write_testset_yaml() -> None:
    import yaml  # noqa: PLC0415

    doc = {
        "name": "performance",
        "title": "性能测试（8 场景 × 1K/8K/32K × 10，240 measured）",
        "description": (
            "8 场景 × 3 档 × 10 个 target；同 target 三档仅上下文长度不同；不判分。"
            "样本由 tool/dataset/performance/build.py 从 dataset/raw 的上游数据确定性抽样构建。"
        ),
        "scenarios": {
            scenario: {
                "dataset": "perf_jsonl",
                "dataset_params": {
                    "scenario": scenario,
                    "path": (TEST_DIR_PERF / "testset.jsonl").relative_to(ROOT).as_posix(),
                },
            }
            for scenario in SCENARIOS
        },
        "sampling": {
            "seed": SEED,
            "targets_per_cell": TARGETS_PER_CELL,
            "method": "sha256(seed|upstream_id) + stratified largest remainder",
        },
        "buckets": dict(BUCKETS),
        "osl": OSL,
    }
    path = TEST_DIR_PERF / "testset.yaml"
    path.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")


def write_report(manifest: Dict[str, Any], rows: Sequence[Dict[str, Any]]) -> None:
    lines = ["# 性能测试集：构造与抽样报告（自动生成）", ""]
    lines.append(
        f"- seed={manifest['seed']}；{len(SCENARIOS)} 场景 × 3 档 × {TARGETS_PER_CELL} target "
        f"= {manifest['rows']} 行（单文件 testset.jsonl）"
    )
    lines.append(
        f"- 参考 tokenizer：{manifest['tokenizer']['name']}（通用口径，不绑定模型）；tokens 为内容 "
        f"token（不含 chat template/特殊 token）= messages_tokens + tools_tokens；声明值按"
        f"场景/语种校准预算定标（budget_model: {manifest['budget_model']['formula']}）"
    )
    lines.append(f"- 抽样：{manifest['sampling']['method']}")
    lines.append("")
    lines.append("| 场景 | 数据集 | 语言 | 行数 | constructed | tokens min/median/max |")
    lines.append("|---|---|---|---:|---:|---|")
    for scenario in SCENARIOS:
        entry = manifest["scenarios"][scenario]
        datasets = ", ".join(f"{key}:{value}" for key, value in entry["datasets"].items())
        languages = ", ".join(f"{key}:{value}" for key, value in entry["languages"].items())
        tokens = entry["tokens"]
        lines.append(
            f"| {scenario} | {datasets} | {languages} | {entry['rows']} | {entry['constructed']} | "
            f"{tokens['min']}/{tokens['median']}/{tokens['max']} |"
        )
    lines.append("")
    lines.append("## 每场景每档 tokens")
    lines.append("")
    lines.append("| 场景 | 1k min/median/max | 8k min/median/max | 32k min/median/max |")
    lines.append("|---|---|---|---|")
    for scenario in SCENARIOS:
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        cells = []
        for bucket in BUCKETS:
            values = sorted(row["tokens"] for row in scenario_rows if row["bucket"] == bucket)
            cells.append(f"{values[0]}/{int(statistics.median(values))}/{values[-1]}")
        lines.append(f"| {scenario} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## 分场景抽样明细")
    lines.append("")
    for scenario in SCENARIOS:
        lines.append(f"### {scenario}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(manifest["scenarios"][scenario]["sampling"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
    lines.append("## 32K 窗口约束")
    lines.append("")
    lines.append(f"```json\n{json.dumps(manifest['window_requirements'], ensure_ascii=False, indent=2)}\n```")
    lines.append("")
    lines.append("## 数据源")
    lines.append("")
    lines.append("| 数据集 | 上游 | 许可 | 文件数 |")
    lines.append("|---|---|---|---:|")
    for name, info in manifest["provenance"].items():
        lines.append(f"| {name} | {info['source_url']} | {info['license']} | {len(info['files'])} |")
    lines.append("")
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# build / check entry points
# --------------------------------------------------------------------------- #
def build_rows(only: Optional[Sequence[str]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    selected = list(only) if only else list(SCENARIOS)
    rows: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {}
    for scenario in selected:
        scenario_rows, scenario_stats = BUILDERS[scenario]()
        stats[scenario] = scenario_stats
        rows.extend(scenario_rows)
        print(
            f"[perf] {scenario}: {len(scenario_stats['targets'])} targets / {len(scenario_rows)} rows; "
            f"skipped={len(scenario_stats['skipped'])}"
        )
    order = {scenario: position for position, scenario in enumerate(SCENARIOS)}
    rows.sort(key=lambda row: (order[row["scenario"]], BUCKET_ORDER[row["bucket"]], row["target_id"]))
    return rows, stats


def check_outputs() -> int:
    manifest_path = TEST_DIR_PERF / "manifest.json"
    testset_path = TEST_DIR_PERF / "testset.jsonl"
    problems: List[str] = []
    if not manifest_path.is_file() or not testset_path.is_file():
        print(f"FAIL: missing {manifest_path} or {testset_path} (run without --check first)")
        return 1
    manifest = read_json(manifest_path)
    rows = read_jsonl(testset_path)
    try:
        validate_rows(rows)
    except RuntimeError as exc:
        problems.append(str(exc))
    try:
        _validate_manifest(manifest, rows, testset_path)
    except RuntimeError as exc:
        problems.append(str(exc))
    # The yaml definition must exist and name every scenario (§5).
    yaml_path = TEST_DIR_PERF / "testset.yaml"
    if not yaml_path.is_file():
        problems.append("missing testset.yaml")
    else:
        import yaml  # noqa: PLC0415

        definition = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        if definition.get("name") != "performance":
            problems.append("testset.yaml name != performance")
        if sorted(definition.get("scenarios") or {}) != sorted(SCENARIOS):
            problems.append("testset.yaml scenarios mismatch")
        if definition.get("buckets") != dict(BUCKETS):
            problems.append("testset.yaml buckets mismatch")
        if definition.get("osl") != OSL:
            problems.append("testset.yaml osl mismatch")
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}")
        return 1
    digest = manifest["files"]["testset.jsonl"]["sha256"]
    print(
        f"[check] PASS rows={len(rows)} targets={len({row['target_id'] for row in rows})} "
        f"cells={len({(row['scenario'], row['bucket']) for row in rows})} sha={digest[:12]}"
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verify existing outputs only (no build)")
    parser.add_argument(
        "--only",
        default="",
        help="debug: build only these scenarios (comma-separated) and write nothing",
    )
    parser.add_argument("--dry-run", action="store_true", help="build + assert but write nothing")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.check:
        return check_outputs()

    only: List[str] = [item.strip() for item in args.only.split(",") if item.strip()]
    unknown = [name for name in only if name not in BUILDERS]
    if unknown:
        print(f"unknown --only scenario(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    missing = [path for path in sources.required_paths() if not path.is_file()]
    if missing:
        print(
            "missing raw asset(s):\n  "
            + "\n  ".join(str(path.relative_to(ROOT)) for path in missing)
            + "\nrun: python tool/dataset/performance/fetch.py",
            file=sys.stderr,
        )
        return 1

    started = time.monotonic()
    rows, stats = build_rows(only or None)
    if only:
        print("[perf] --only debug run: nothing written")
        return 0
    validate_rows(rows)
    if args.dry_run:
        print("[perf] --dry-run: assertions passed, nothing written")
        return 0

    TEST_DIR_PERF.mkdir(parents=True, exist_ok=True)
    testset_path = TEST_DIR_PERF / "testset.jsonl"
    write_jsonl(testset_path, rows)
    manifest = build_manifest(rows, testset_path, stats)
    _validate_manifest(manifest, rows, testset_path)
    write_json(TEST_DIR_PERF / "manifest.json", manifest)
    write_testset_yaml()
    write_report(manifest, rows)
    elapsed = time.monotonic() - started
    print(
        f"[perf] wrote {len(rows)} rows -> {testset_path.relative_to(ROOT)} "
        f"({testset_path.stat().st_size} bytes, sha={sha256_file(testset_path)[:12]}); "
        f"elapsed={elapsed:.1f}s"
    )
    return check_outputs()


if __name__ == "__main__":
    sys.exit(main())
