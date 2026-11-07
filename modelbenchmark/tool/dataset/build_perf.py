"""Build the performance test set: 8 scenarios x 1K/8K/32K x 20 = 480 rows.

Design: ``dataset/TEST_SAMPLE_SELECTION.md`` (v3) plus the audit fixes recorded
in ``reports/audit/PERF_CHANGES.md``.  Each target appears in all three buckets
with an identical question/instruction; only the context length changes.  The
quick subset (10 targets per bucket) is derived by ``build_perf_subset.py``.

Sampling is deterministic (fixed seed) and self-documented: every scenario
records its source pool, duplicate removals, selected targets and skip reasons
in ``manifest.json:sampling``.  A scenario that cannot reach 20 feasible
targets raises instead of relaxing the quota.

Outputs:
    dataset/test/performance/testset.jsonl     480 rows
    dataset/test/performance/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pyarrow.parquet as pq

import config as CFG
import ds_common as C

OVERHEAD = 20
GENERATOR_VERSION = "tool/dataset/build_perf.py v4"

BUCKETS: Tuple[str, ...] = ("1k", "8k", "32k")
BUCKET_ORDER = {bucket: index for index, bucket in enumerate(BUCKETS)}

#: Runtime window reserve (kept in sync with ``src/modelbench/application/benchmark.py``
#: and ``resources/config/default.yaml``): declared tokens + max_tokens + safety
#: margin + decode-guard reserve must fit the model window.  Recorded in the
#: manifest so the 32K-window limitation is auditable.
RUN_MAX_TOKENS = CFG.OSL
CONTEXT_SAFETY_MARGIN_TOKENS = 128
DECODE_GUARD_TOKEN_RESERVE = 64
WINDOW_RESERVE = RUN_MAX_TOKENS + CONTEXT_SAFETY_MARGIN_TOKENS + DECODE_GUARD_TOKEN_RESERVE

#: Document scenarios: body is a character prefix of one upstream document.
DOC_SCENARIOS = ("knowledge_qa", "long_context", "coding", "search_agent", "chinese_long_context")
#: Scenarios whose 8K/32K rows are built from same-set material.
CONSTRUCTED_8K_32K = ("chinese_long_context",)
#: Scenarios whose every bucket is constructed.
ALWAYS_CONSTRUCTED = ("instruction_following", "math_logic", "general_agent")
TOOL_SCENARIOS = ("general_agent", "search_agent")

#: Same-set material rotations (target-specific starting point, successives fix).
MATH_ROTATION_STEP = 41
IFEVAL_ROTATION_STEP = 37

#: LongBench v2 document pool filter (the audit-documented sampling rule).
LBV2_MIN_CONTEXT_CHARS = 120_000
#: A target question/options block larger than this cannot leave room for 1K context.
PROMPT_GUARD_TOKEN_FRACTION = 0.6

#: BFCL performance strata (20 targets = 10 simple / 5 multiple / 5 parallel).
BFCL_CATEGORIES: Tuple[str, ...] = ("simple_python", "multiple", "parallel")
BFCL_CATEGORY_LABEL = {"simple_python": "simple", "multiple": "multiple", "parallel": "parallel"}
BFCL_QUOTAS = {"simple_python": 10, "multiple": 5, "parallel": 5}

#: Suffix markers / prefixes used by the build-time quality assertions.
MARKERS = {
    "knowledge_qa": "\n\nQuestion: ",
    "long_context": "\n\nQuestion: ",
    "coding": "\n\nQuestion: ",
    "search_agent": "\n\nQuestion: ",
    "chinese_long_context": "\n\n问题：",
    "instruction_following": "\n\nInstruction: ",
    "math_logic": "\n\nQuestion: ",
    "general_agent": "\n\nUser request:\n",
}
DOC_PREFIXES = {
    "knowledge_qa": ("Read the document and answer the question.\n\n",),
    "long_context": ("Read the document and answer the question.\n\n",),
    "coding": ("Read the document and answer the question.\n\n",),
    "search_agent": ("Read the document and answer the question. You may use the provided tools.\n\n",),
    "chinese_long_context": ("阅读以下材料并回答问题。\n\n",),
}
AGENT_PREFIX = "You are a function-calling assistant. API reference:\n"
AGENT_MARKER = "\n\nUser request:\n"


# --------------------------------------------------------------------------- #
# token helpers
# --------------------------------------------------------------------------- #
def tools_tokens(tok: C.Tokenizer, tools: Optional[List[Dict[str, Any]]]) -> int:
    return tok.count_text(json.dumps(tools, ensure_ascii=False)) if tools else 0


def bucket_bounds(target: int) -> Tuple[int, int]:
    """Integer token bounds for one bucket.

    Declared tokens are integers, so the low bound is the floor of
    ``0.95 * target`` (1K: 972, not 972.8); the previous float comparison
    rejected ``972`` by 0.8 token.  The high bound is ``floor(1.05 * target)``.
    """

    return math.floor(CFG.ACCEPT_LOW * target), math.floor(CFG.ACCEPT_HIGH * target)


def within_bounds(total: int, target: int) -> bool:
    low, high = bucket_bounds(target)
    return low <= total <= high


def size_prompt(
    tok: C.Tokenizer,
    make_body: Callable[[int], str],
    suffix: str,
    target: int,
    tools: Optional[List[Dict[str, Any]]] = None,
    prefix: str = "",
) -> Tuple[List[Dict[str, str]], int, int, bool]:
    """Find a body so that messages + tools tokens land within the bucket bounds."""
    tt = tools_tokens(tok, tools)
    want = target - tt - OVERHEAD
    best: Optional[Tuple[int, List[Dict[str, str]], int, int]] = None
    for _ in range(6):
        if want < 64:
            content = prefix + suffix
        else:
            content = prefix + make_body(want) + suffix
        msgs = [{"role": "user", "content": content}]
        mt = tok.count_messages(msgs)
        total = mt + tt
        if best is None or abs(total - target) < abs(best[0] - target):
            best = (total, msgs, mt, tt)
        if within_bounds(total, target):
            return msgs, mt, tt, True
        want += target - total
    total, msgs, mt, tt = best  # type: ignore[misc]
    return msgs, mt, tt, within_bounds(total, target)


def slice_text(tok: C.Tokenizer, text: str, want: int, suffix: str) -> str:
    body, _ = C.fit_text_to_tokens(tok, text, max(64, want), suffix=suffix)
    return body


# --------------------------------------------------------------------------- #
# readers
# --------------------------------------------------------------------------- #
_LBV2_POOLS: Optional[Tuple[Dict[str, int], Dict[str, List[Dict[str, Any]]]]] = None


def lbv2_pools(min_chars: int = LBV2_MIN_CONTEXT_CHARS) -> Tuple[Dict[str, int], Dict[str, List[Dict[str, Any]]]]:
    """All LongBench v2 rows per domain + the eligible (>= min_chars) pool.

    Read once: the raw ``data.json`` is ~465 MB and four scenarios share it.
    """

    global _LBV2_POOLS
    if _LBV2_POOLS is None:
        counts: Dict[str, int] = {}
        pools: Dict[str, List[Dict[str, Any]]] = {}
        for row in C.iter_json_array(C.RAW / "longbench_v2" / "data.json"):
            domain = row.get("domain")
            counts[domain] = counts.get(domain, 0) + 1
            if len(row.get("context") or "") < min_chars:
                continue
            pools.setdefault(domain, []).append(row)
        _LBV2_POOLS = (counts, pools)
    return _LBV2_POOLS


def lb_v1_rows(task: str) -> List[Dict[str, Any]]:
    return C.load_zip_jsonl(C.RAW / "longbench_v1" / "data.zip", f"data/{task}.jsonl")


def ifeval_rows() -> List[Dict[str, Any]]:
    return C.load_jsonl(C.RAW / "ifeval" / "ifeval_input_data.jsonl")


def gsm_rows() -> List[Dict[str, Any]]:
    return pq.read_table(C.RAW / "gsm8k" / "main" / "test-00000-of-00001.parquet").to_pylist()


def tool_of(fn: Dict[str, Any]) -> Dict[str, Any]:
    f = fn.get("function", fn)
    return {
        "type": "function",
        "function": {
            "name": f.get("name"),
            "description": f.get("description", ""),
            "parameters": f.get("parameters") or {"type": "object", "properties": {}},
        },
    }


# --------------------------------------------------------------------------- #
# BFCL function catalogue: target functions first, then same category, then rest
# --------------------------------------------------------------------------- #
def bfcl_rows_by_category() -> Dict[str, List[Dict[str, Any]]]:
    return {
        name: C.load_jsonl(C.RAW / "bfcl" / f"BFCL_v4_{name}.json") for name in BFCL_CATEGORIES
    }


def bfcl_category_catalog(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Unique-by-name normalised functions of one category, first appearance order."""

    out: List[Dict[str, Any]] = []
    seen: set = set()
    for row in rows:
        for fn in row.get("function", []):
            tool = C.normalize_tools([tool_of(fn)])[0]
            name = tool["function"]["name"]
            if name in seen:
                continue
            seen.add(name)
            out.append(tool)
    return out


def target_catalog(
    own: List[Dict[str, Any]],
    category: str,
    category_catalogs: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Own functions first, then same-category, then the rest; unique by name."""

    ordered = list(own)
    ordered += category_catalogs[category]
    for other in BFCL_CATEGORIES:
        if other != category:
            ordered += category_catalogs[other]
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for tool in ordered:
        name = tool["function"]["name"]
        if name in seen:
            continue
        seen.add(name)
        out.append(tool)
    return out


# --------------------------------------------------------------------------- #
# workload / record
# --------------------------------------------------------------------------- #
def workload_of(scenario: str) -> str:
    """Workload description recorded in ``meta.workload`` and the manifest.

    ``chinese_long_context`` has no explicit ``workload`` in ``PERF_SOURCES``
    (the rename is owned by config.py); derive it from dataset:task so the field
    is always present.
    """

    spec = CFG.PERF_SOURCES[scenario]
    workload = spec.get("workload")
    if workload:
        return str(workload)
    return f"{spec.get('dataset')}:{spec.get('task') or spec.get('domain')}"


def rec(
    scenario: str,
    bucket: str,
    idx: int,
    msgs: List[Dict[str, str]],
    mt: int,
    tt: int,
    tools: Optional[List[Dict[str, Any]]],
    constructed: bool,
    meta: Dict[str, Any],
    dataset: str,
) -> Dict[str, Any]:
    row_meta = dict(meta)
    row_meta.setdefault("workload", workload_of(scenario))
    row_meta.setdefault("lang", "en")
    return {
        "id": f"perf-{scenario}-{bucket}-{idx:03d}",
        "scenario": scenario,
        "bucket": bucket,
        "target_id": f"tgt-{scenario}-{idx:03d}",
        "dataset": dataset,
        "tokens": mt + tt,
        "messages_tokens": mt,
        "tools_tokens": tt,
        "messages": msgs,
        "tools": tools,
        "constructed": constructed,
        "meta": row_meta,
    }


def emit(
    scenario: str,
    idx: int,
    build: Callable[[str], Tuple[List[Dict[str, str]], int, int, bool]],
    tools: Optional[List[Dict[str, Any]]],
    dataset: str,
    meta: Dict[str, Any],
    constructed_buckets: Optional[set] = None,
) -> List[Dict[str, Any]]:
    out = []
    for bucket in BUCKETS:
        msgs, mt, tt, ok = build(bucket)
        if not ok:
            low, high = bucket_bounds(CFG.BUCKETS[bucket])
            raise RuntimeError(
                f"{scenario} {bucket} idx={idx}: tokens {mt + tt} out of tolerance "
                f"[{low}, {high}]"
            )
        out.append(
            rec(
                scenario,
                bucket,
                idx,
                msgs,
                mt,
                tt,
                tools,
                bucket in (constructed_buckets or set()),
                meta,
                dataset,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# deterministic sampling helpers
# --------------------------------------------------------------------------- #
def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()


def dedupe_rows(
    rows: List[Dict[str, Any]], by_question: bool = False
) -> Tuple[
    List[Tuple[int, Dict[str, Any]]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """Drop duplicate raw documents (sha1 of context) and optionally duplicate questions.

    Returns the kept ``(pool_rank, row)`` pairs plus removal details, so the
    manifest can prove how many rows the dedupe step removed.
    """

    kept: List[Tuple[int, Dict[str, Any]]] = []
    seen_context: Dict[str, int] = {}
    seen_question: Dict[str, int] = {}
    removed_context: List[Dict[str, Any]] = []
    removed_question: List[Dict[str, Any]] = []
    for rank, row in enumerate(rows):
        context = row.get("context") or ""
        digest = _sha1_text(context)
        if digest in seen_context:
            removed_context.append(
                {"pool_rank": rank, "sha1": digest[:16], "kept_pool_rank": seen_context[digest]}
            )
            continue
        if by_question:
            question = " ".join((row.get("question") or "").split())
            if question in seen_question:
                removed_question.append(
                    {
                        "pool_rank": rank,
                        "question": question[:120],
                        "kept_pool_rank": seen_question[question],
                    }
                )
                continue
            seen_question[question] = rank
        seen_context[digest] = rank
        kept.append((rank, row))
    return kept, removed_context, removed_question


def _sampling(scenario: str, dataset: str, **extra: Any) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "dataset": dataset,
        "workload": workload_of(scenario),
        "seed": None,
        "pool_filter": None,
        "domain_rows": None,
        "source_rows": 0,
        "dedup": {
            "context_duplicates_removed": 0,
            "question_duplicates_removed": 0,
            "context_duplicates": [],
            "question_duplicates": [],
        },
        "candidates": 0,
        "selected": 0,
        "selected_targets": [],
        "skipped": [],
        "skip_reasons": {},
    }
    entry.update(extra)
    return entry


def _skip(stats: Dict[str, Any], pool_rank: int, reason: str, detail: str, **extra: Any) -> None:
    item: Dict[str, Any] = {"pool_rank": pool_rank, "reason": reason, "detail": detail}
    item.update(extra)
    stats["skipped"].append(item)


def _select(stats: Dict[str, Any], target_id: str, pool_rank: int, **extra: Any) -> None:
    item: Dict[str, Any] = {"target_id": target_id, "pool_rank": pool_rank}
    item.update(extra)
    stats["selected_targets"].append(item)


def _finish(stats: Dict[str, Any]) -> Dict[str, Any]:
    stats["selected"] = len(stats["selected_targets"])
    counts = Counter(item["reason"] for item in stats["skipped"])
    stats["skip_reasons"] = dict(sorted(counts.items()))
    return stats


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
def build_doc(
    tok: C.Tokenizer,
    scenario: str,
    domain: str,
    rows: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    by_question: bool = False,
    seed: int = CFG.SEED,
    domain_rows: Optional[int] = None,
    collision_reason: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """LongBench v2 prefix-slice samples (deduped by context sha1 / question)."""

    kept, removed_context, removed_question = dedupe_rows(rows, by_question=by_question)
    stats = _sampling(
        scenario,
        f"longbench_v2:{domain}",
        seed=seed,
        pool_filter=f"raw context >= {LBV2_MIN_CONTEXT_CHARS} characters",
        domain_rows=domain_rows,
        source_rows=len(rows),
        dedup={
            "context_duplicates_removed": len(removed_context),
            "question_duplicates_removed": len(removed_question),
            "context_duplicates": removed_context,
            "question_duplicates": removed_question,
        },
        candidates=len(kept),
    )
    rng = random.Random(seed)
    shuffled = list(kept)
    rng.shuffle(shuffled)

    prefix = "Read the document and answer the question."
    if tools:
        prefix += " You may use the provided tools."
    prefix += "\n\n"
    guard_tokens = CFG.BUCKETS["1k"] * PROMPT_GUARD_TOKEN_FRACTION
    out: List[Dict[str, Any]] = []
    for pool_rank, row in shuffled:
        if len(out) >= CFG.PERF_PER_BUCKET * len(BUCKETS):
            break
        question = row.get("question") or ""
        options = "\n".join(f"{c}. {row.get(f'choice_{c}') or ''}" for c in "ABCD")
        suffix = f"\n\nQuestion: {question}\n{options}\nAnswer:"
        doc = row.get("context") or ""
        prompt_tokens = tok.count_text(prefix + suffix)
        if prompt_tokens > guard_tokens:
            _skip(
                stats,
                pool_rank,
                "question_too_long",
                f"question+options+prefix={prompt_tokens} tokens > {int(guard_tokens)} "
                "(1K context would not fit)",
            )
            continue
        probe32 = size_prompt(
            tok, lambda w, _doc=doc: slice_text(tok, _doc, w, prefix + suffix), suffix,
            CFG.BUCKETS["32k"], tools, prefix,
        )
        if not probe32[3]:
            _skip(
                stats,
                pool_rank,
                "32k_unreachable",
                f"best total={probe32[1] + probe32[2]} tokens; document too short for 32K",
            )
            continue
        probe1 = size_prompt(
            tok, lambda w, _doc=doc: slice_text(tok, _doc, w, prefix + suffix), suffix,
            CFG.BUCKETS["1k"], tools, prefix,
        )
        if not probe1[3]:
            _skip(
                stats,
                pool_rank,
                "1k_low_token_density",
                f"best total={probe1[1] + probe1[2]} tokens; document slicing cannot reach 1K",
            )
            continue
        target_idx = len(out) // len(BUCKETS)

        def build(
            bucket: str,
            _doc: str = doc,
            _prefix: str = prefix,
            _suffix: str = suffix,
            _p32: Tuple = probe32,
            _p1: Tuple = probe1,
        ) -> Tuple[List[Dict[str, str]], int, int, bool]:
            if bucket == "32k":
                return _p32
            if bucket == "1k":
                return _p1
            return size_prompt(
                tok, lambda w, _d=_doc: slice_text(tok, _d, w, _prefix + _suffix), _suffix,
                CFG.BUCKETS["8k"], tools, _prefix,
            )

        digest = _sha1_text(doc)
        meta: Dict[str, Any] = {"domain": domain, "lang": "en", "context_sha1": digest}
        if by_question:
            meta["question_sha1"] = _sha1_text(" ".join(question.split()))
        out += emit(scenario, target_idx, build, tools, stats["dataset"], meta)
        _select(
            stats,
            f"tgt-{scenario}-{target_idx:03d}",
            pool_rank,
            context_sha1=digest[:16],
        )
    if len(out) < CFG.PERF_PER_BUCKET * len(BUCKETS):
        raise RuntimeError(
            f"{scenario}: only {len(out) // len(BUCKETS)} feasible targets, "
            f"need {CFG.PERF_PER_BUCKET}; skips={stats['skipped']}"
        )
    # Report (do not "fix") upstream prefix collisions: some eligible pools share
    # a boilerplate document prefix, so a few 1K bodies can be byte-identical.
    one_k_rows = [row for row in out if row["bucket"] == "1k"]
    unique_bodies = len(
        {
            _doc_body(row["messages"][0]["content"], MARKERS[scenario], DOC_PREFIXES[scenario])
            for row in one_k_rows
        }
    )
    stats["unique_1k_bodies"] = unique_bodies
    if unique_bodies < len(one_k_rows):
        stats["known_1k_prefix_collisions"] = {
            "unique_1k_bodies": unique_bodies,
            "selected_targets": len(one_k_rows),
            "reason": collision_reason
            or (
                "upstream shared document/boilerplate prefix; the eligible pool is too "
                "small to supply one distinct 1K body per target, so some 1K bodies are "
                "byte-identical. Target count is kept at 20 by design."
            ),
        }
    return out, _finish(stats)


def build_chinese_long_context(tok: C.Tokenizer) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """LongBench v1 dureader (Chinese long-form QA), target document first."""

    rows = lb_v1_rows("dureader")
    kept, removed_context, removed_question = dedupe_rows(rows)
    seed = CFG.SEED + 5
    stats = _sampling(
        "chinese_long_context",
        "longbench_v1:dureader",
        seed=seed,
        pool_filter="all dureader rows (200)",
        domain_rows=len(rows),
        source_rows=len(rows),
        dedup={
            "context_duplicates_removed": len(removed_context),
            "question_duplicates_removed": len(removed_question),
            "context_duplicates": removed_context,
            "question_duplicates": removed_question,
        },
        candidates=len(kept),
    )
    rng = random.Random(seed)
    shuffled = list(kept)
    rng.shuffle(shuffled)

    prefix = "阅读以下材料并回答问题。\n\n"
    guard_tokens = CFG.BUCKETS["1k"] * PROMPT_GUARD_TOKEN_FRACTION
    out: List[Dict[str, Any]] = []
    for pool_rank, row in shuffled:
        if len(out) >= CFG.PERF_PER_BUCKET * len(BUCKETS):
            break
        target_ctx = row.get("context") or ""
        suffix = f"\n\n问题：{row.get('input', '')}\n答案："
        prompt_tokens = tok.count_text(prefix + suffix)
        if prompt_tokens > guard_tokens:
            _skip(
                stats,
                pool_rank,
                "question_too_long",
                f"question+prefix={prompt_tokens} tokens > {int(guard_tokens)}",
            )
            continue
        probe1 = size_prompt(
            tok, lambda w, _c=target_ctx: slice_text(tok, _c, w, prefix + suffix), suffix,
            CFG.BUCKETS["1k"], None, prefix,
        )
        if not probe1[3]:
            _skip(
                stats,
                pool_rank,
                "1k_low_token_density",
                f"best total={probe1[1] + probe1[2]} tokens",
            )
            continue
        others = [
            (other.get("context") or "")
            for _rank, other in kept
            if (other.get("context") or "") != target_ctx
        ]
        docs = [target_ctx] + others

        def make_body(w: int, _docs=docs) -> str:
            picked: List[str] = []
            joined = ""
            for current in _docs:
                picked.append(current)
                joined = "\n\n".join(picked)
                if tok.count_text(prefix + joined + suffix) >= w:
                    break
            return slice_text(tok, joined, w, prefix + suffix)

        probe32 = size_prompt(tok, make_body, suffix, CFG.BUCKETS["32k"], None, prefix)
        if not probe32[3]:
            _skip(
                stats,
                pool_rank,
                "32k_unreachable",
                f"best total={probe32[1] + probe32[2]} tokens",
            )
            continue

        def build(
            bucket: str,
            _make=make_body,
            _suffix=suffix,
            _p1=probe1,
            _p32=probe32,
        ) -> Tuple[List[Dict[str, str]], int, int, bool]:
            if bucket == "1k":
                return _p1
            if bucket == "32k":
                return _p32
            return size_prompt(tok, _make, _suffix, CFG.BUCKETS["8k"], None, prefix)

        target_idx = len(out) // len(BUCKETS)
        digest = _sha1_text(target_ctx)
        out += emit(
            "chinese_long_context",
            target_idx,
            build,
            None,
            stats["dataset"],
            {"lang": "zh", "task": "dureader", "context_sha1": digest},
            constructed_buckets={"8k", "32k"},
        )
        _select(
            stats,
            f"tgt-chinese_long_context-{target_idx:03d}",
            pool_rank,
            context_sha1=digest[:16],
        )
    if len(out) < CFG.PERF_PER_BUCKET * len(BUCKETS):
        raise RuntimeError(
            f"chinese_long_context: only {len(out) // len(BUCKETS)} feasible targets, "
            f"need {CFG.PERF_PER_BUCKET}; skips={stats['skipped']}"
        )
    return out, _finish(stats)


def build_ifeval(tok: C.Tokenizer) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """IFEval target instruction + rotated same-set reference pool."""

    rows = ifeval_rows()
    pool = [row["prompt"] for row in rows]
    pool_index = {prompt: index for index, prompt in enumerate(pool)}
    seed = CFG.SEED + 2
    stats = _sampling(
        "instruction_following",
        "ifeval",
        seed=seed,
        pool_filter=None,
        domain_rows=len(rows),
        source_rows=len(rows),
        candidates=len(rows),
    )
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)

    prefix = "Reference instructions:\n"
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(shuffled[: CFG.PERF_PER_BUCKET]):
        target = row["prompt"]
        # Rotate the reference pool per target (target excluded) so the 32K
        # bucket is not 20 copies of one prefix order.
        others = [prompt for prompt in pool if prompt != target]
        start = (idx * IFEVAL_ROTATION_STEP) % len(others)
        ordered = others[start:] + others[:start]
        reference = "\n\n".join(ordered)
        cycles = 1
        if tok.count_text(reference) < CFG.BUCKETS["32k"]:
            reference = reference + "\n\n" + reference
            cycles = 2
        suffix = "\n\nInstruction: " + target

        def build(
            bucket: str,
            _ref: str = reference,
            _suffix: str = suffix,
        ) -> Tuple[List[Dict[str, str]], int, int, bool]:
            return size_prompt(
                tok, lambda w, _r=_ref: slice_text(tok, _r, w, prefix + _suffix), _suffix,
                CFG.BUCKETS[bucket], None, prefix,
            )

        out += emit(
            "instruction_following",
            idx,
            build,
            None,
            stats["dataset"],
            {
                "lang": "en",
                "instruction_ids": row.get("instruction_id_list", []),
                "reference_rotation_start": start,
                "reference_pool_cycles": cycles,
            },
            constructed_buckets=set(BUCKETS),
        )
        _select(
            stats,
            f"tgt-instruction_following-{idx:03d}",
            pool_index[target],
            reference_rotation_start=start,
            reference_pool_cycles=cycles,
        )
    if len(out) < CFG.PERF_PER_BUCKET * len(BUCKETS):
        raise RuntimeError(f"instruction_following: only {len(out) // len(BUCKETS)} samples")
    return out, _finish(stats)


def build_math(tok: C.Tokenizer) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """GSM8K target problem + solved examples with a per-target rotation."""

    rows = gsm_rows()
    seed = CFG.SEED + 3
    raw_index = {id(row): index for index, row in enumerate(rows)}
    rng = random.Random(seed)
    rng.shuffle(rows)
    targets = rows[: CFG.PERF_PER_BUCKET]
    others = rows[CFG.PERF_PER_BUCKET:]
    stats = _sampling(
        "math_logic",
        "gsm8k",
        seed=seed,
        pool_filter=None,
        domain_rows=len(rows),
        source_rows=len(rows),
        candidates=len(rows),
    )

    prefix = "Solved examples:\n"
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(targets):
        # Each target starts the solved-example block at a different example,
        # so the 20 contexts are not all the same prefix.
        start = (idx * MATH_ROTATION_STEP) % len(others)
        ordered = others[start:] + others[:start]
        ref = "\n\n".join(
            f"Question: {example['question'].strip()}\nAnswer: {example['answer'].strip()}"
            for example in ordered
        )
        suffix = f"\n\nQuestion: {row['question'].strip()}\nAnswer:"

        def build(
            bucket: str,
            _ref: str = ref,
            _suffix: str = suffix,
        ) -> Tuple[List[Dict[str, str]], int, int, bool]:
            return size_prompt(
                tok, lambda w, _r=_ref: slice_text(tok, _r, w, prefix + _suffix), _suffix,
                CFG.BUCKETS[bucket], None, prefix,
            )

        out += emit(
            "math_logic",
            idx,
            build,
            None,
            stats["dataset"],
            {
                "lang": "en",
                "gold": row["answer"].split("####")[-1].strip(),
                "example_rotation_start": start,
            },
            constructed_buckets=set(BUCKETS),
        )
        _select(
            stats,
            f"tgt-math_logic-{idx:03d}",
            raw_index[id(row)],
            example_rotation_start=start,
        )
    if len(out) < CFG.PERF_PER_BUCKET * len(BUCKETS):
        raise RuntimeError(f"math_logic: only {len(out) // len(BUCKETS)} samples")
    return out, _finish(stats)


def agent_body_functions(body: str) -> List[Dict[str, Any]]:
    """Parse the whole-function JSON catalogue out of an agent prompt body."""

    return json.loads(body) if body else []


def build_agent(tok: C.Tokenizer) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """BFCL single request + target-first whole-function catalogue prefix."""

    rows_by_category = bfcl_rows_by_category()
    category_catalogs = {
        category: bfcl_category_catalog(rows) for category, rows in rows_by_category.items()
    }
    universe_names = {
        tool["function"]["name"]
        for catalog in category_catalogs.values()
        for tool in catalog
    }
    stats = _sampling(
        "general_agent",
        "bfcl",
        seed=CFG.SEED + 4,
        pool_filter="BFCL v4 single-turn request files (simple_python/multiple/parallel)",
        domain_rows=sum(len(rows) for rows in rows_by_category.values()),
        source_rows=sum(len(rows) for rows in rows_by_category.values()),
        candidates=sum(len(rows) for rows in rows_by_category.values()),
        category_pool={category: len(rows) for category, rows in rows_by_category.items()},
        category_quota=dict(BFCL_QUOTAS),
        category_catalog_functions={
            category: len(catalog) for category, catalog in category_catalogs.items()
        },
        catalog_universe_functions=len(universe_names),
    )
    prefix = AGENT_PREFIX
    out: List[Dict[str, Any]] = []
    for category_index, category in enumerate(BFCL_CATEGORIES):
        quota = BFCL_QUOTAS[category]
        indexed = list(enumerate(rows_by_category[category]))
        rng = random.Random(CFG.SEED + 40 + category_index)
        rng.shuffle(indexed)
        chosen = 0
        for pool_rank, row in indexed:
            if chosen >= quota:
                break
            turns = [t.get("content", "") for turn in row.get("question", []) for t in turn]
            own = C.normalize_tools([tool_of(fn) for fn in row.get("function", [])]) or []
            catalog = target_catalog(own, category, category_catalogs)
            suffix = "\n\nUser request:\n" + "\n".join(turns)

            def make_body(w: int, _catalog=catalog) -> str:
                """Largest whole-function prefix whose tokens fit ``w``."""

                def render(k: int) -> str:
                    return json.dumps(_catalog[:k], ensure_ascii=False)

                lo, hi = 0, len(_catalog)
                best = render(0)
                while lo <= hi:
                    mid = (lo + hi) // 2
                    candidate = render(mid)
                    if tok.count_text(prefix + candidate + suffix) <= w:
                        best = candidate
                        lo = mid + 1
                    else:
                        hi = mid - 1
                return best

            def build(
                bucket: str,
                _make=make_body,
                _suffix=suffix,
                _own=own,
            ) -> Tuple[List[Dict[str, str]], int, int, bool]:
                return size_prompt(
                    tok, _make, _suffix, CFG.BUCKETS[bucket], _own, prefix
                )

            probe1 = build("1k")
            if not probe1[3]:
                _skip(
                    stats,
                    pool_rank,
                    "1k_infeasible",
                    f"tools+request best total={probe1[1] + probe1[2]} tokens "
                    f"(bucket bounds {bucket_bounds(CFG.BUCKETS['1k'])})",
                    category=BFCL_CATEGORY_LABEL[category],
                    category_detail=category,
                    bfcl_id=str(row.get("id") or ""),
                )
                continue
            body1 = probe1[0][0]["content"][len(prefix): -len(suffix)]
            if not body1:
                # No room for even an empty whole-function catalogue at 1K.
                _skip(
                    stats,
                    pool_rank,
                    "1k_no_catalog_room",
                    f"tools={probe1[2]} tokens leave no catalogue preamble",
                    category=BFCL_CATEGORY_LABEL[category],
                    category_detail=category,
                    bfcl_id=str(row.get("id") or ""),
                )
                continue
            target_idx = len(out) // len(BUCKETS)
            raw_id = str(row.get("id") or "")
            own_names = [tool["function"]["name"] for tool in own]
            catalog_names = [tool["function"]["name"] for tool in catalog]
            own_in_catalog = all(name in catalog_names for name in own_names)
            if not own_in_catalog:  # pragma: no cover - construction invariant
                raise RuntimeError(f"general_agent {raw_id}: own functions missing from catalog")
            meta = {
                "lang": "en",
                "category": BFCL_CATEGORY_LABEL[category],
                "category_detail": category,
                "bfcl_id": raw_id,
                "catalog_functions": len(catalog),
                "own_functions": len(own),
                "own_functions_in_catalog": own_in_catalog,
            }
            start_row = len(out)
            out += emit(
                "general_agent",
                target_idx,
                build,
                own,
                stats["dataset"],
                meta,
                constructed_buckets=set(BUCKETS),
            )
            body_counts: Dict[str, int] = {}
            own_visible: Dict[str, int] = {}
            for offset, bucket in enumerate(BUCKETS):
                content = out[start_row + offset]["messages"][0]["content"]
                index = content.rfind(AGENT_MARKER)
                body = content[len(prefix):index]
                functions = agent_body_functions(body)
                body_counts[bucket] = len(functions)
                visible = {
                    fn.get("function", {}).get("name") for fn in functions
                }
                own_visible[bucket] = sum(1 for name in own_names if name in visible)
            _select(
                stats,
                f"tgt-general_agent-{target_idx:03d}",
                pool_rank,
                bfcl_id=raw_id,
                category=BFCL_CATEGORY_LABEL[category],
                category_detail=category,
                body_catalog_functions=body_counts,
                own_functions_visible=own_visible,
            )
            chosen += 1
        if chosen < quota:
            raise RuntimeError(
                f"general_agent {category}: only {chosen} feasible targets, need {quota}; "
                f"skips={stats['skipped']}"
            )
    if len(out) < CFG.PERF_PER_BUCKET * len(BUCKETS):
        raise RuntimeError(f"general_agent: only {len(out) // len(BUCKETS)} samples")
    stats["category_skipped"] = dict(
        sorted(Counter(item["category"] for item in stats["skipped"] if "category" in item).items())
    )
    stats["category_selected"] = dict(
        sorted(Counter(item["category"] for item in stats["selected_targets"]).items())
    )
    return out, _finish(stats)


BUILDERS: Dict[str, Callable[[C.Tokenizer], Tuple[List[Dict[str, Any]], Dict[str, Any]]]] = {}


def _register() -> None:
    domains = {
        "knowledge_qa": "Single-Document QA",
        "long_context": "Long Structured Data Understanding",
        "coding": "Code Repository Understanding",
        "search_agent": "Multi-Document QA",
    }

    def make_doc(
        scenario: str,
        domain: str,
        with_tools: bool,
        by_question: bool = False,
        collision_reason: Optional[str] = None,
    ):
        def builder(tok: C.Tokenizer):
            counts, pools = lbv2_pools()
            rows = pools.get(domain) or []
            tools = C.normalize_tools(CFG.SEARCH_TOOLS) if with_tools else None
            return build_doc(
                tok,
                scenario,
                domain,
                rows,
                tools,
                by_question=by_question,
                domain_rows=counts.get(domain),
                collision_reason=collision_reason,
            )

        return builder

    BUILDERS.update(
        {
            "knowledge_qa": make_doc("knowledge_qa", "Single-Document QA", False, by_question=True),
            "long_context": make_doc(
                "long_context",
                "Long Structured Data Understanding",
                False,
                collision_reason=(
                    "upstream LongBench v2 Long Structured Data documents share a "
                    "2,651-character boilerplate prefix; the eligible pool (33 rows, "
                    "31 after dedupe) cannot supply 20 distinct 1K bodies, so 5 of the "
                    "20 1K bodies are byte-identical. This is an upstream property, not "
                    "a sampling bug; the target count stays 20 by design."
                ),
            ),
            "coding": make_doc("coding", "Code Repository Understanding", False),
            "search_agent": make_doc("search_agent", "Multi-Document QA", True),
            "chinese_long_context": build_chinese_long_context,
            "instruction_following": build_ifeval,
            "math_logic": build_math,
            "general_agent": build_agent,
        }
    )


_register()


# --------------------------------------------------------------------------- #
# build-time quality assertions
# --------------------------------------------------------------------------- #
_ID_FIELDS = {
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
_JSON_SCHEMA_TYPES = {
    "object",
    "array",
    "string",
    "number",
    "integer",
    "boolean",
    "null",
    "function",
}


def _schema_types(node: Any) -> List[str]:
    out: List[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "type" and isinstance(value, str):
                out.append(value)
            else:
                out.extend(_schema_types(value))
    elif isinstance(node, list):
        for item in node:
            out.extend(_schema_types(item))
    return out


def _expected_constructed(scenario: str, bucket: str) -> bool:
    if scenario in ALWAYS_CONSTRUCTED:
        return True
    if scenario in CONSTRUCTED_8K_32K:
        return bucket in ("8k", "32k")
    return False


def _suffix(content: str, marker: str) -> Optional[str]:
    index = content.rfind(marker)
    return content[index:] if index >= 0 else None


def _doc_body(content: str, marker: str, prefixes: Tuple[str, ...]) -> str:
    index = content.rfind(marker)
    body = content[:index] if index >= 0 else content
    for prefix in prefixes:
        if body.startswith(prefix):
            return body[len(prefix):]
    return body


def validate_rows(rows: List[Dict[str, Any]]) -> None:
    """Assert every contract the rebuilt set promises (raises on drift)."""

    errors: List[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    expected_rows = len(CFG.PERF_SOURCES) * len(BUCKETS) * CFG.PERF_PER_BUCKET
    check(len(rows) == expected_rows == 480, f"row count {len(rows)} != {expected_rows}")
    for row in rows:
        if set(row) != _ID_FIELDS:
            check(False, f"{row.get('id')}: field set {sorted(set(row) ^ _ID_FIELDS)}")
    check(len({row["id"] for row in rows}) == len(rows), "duplicate ids")
    cells = Counter((row["scenario"], row["bucket"]) for row in rows)
    check(len(cells) == 24, f"cells {len(cells)} != 24")
    check(all(value == CFG.PERF_PER_BUCKET for value in cells.values()), f"cell sizes {cells}")
    targets = Counter(row["target_id"] for row in rows)
    check(len(targets) == 160, f"targets {len(targets)} != 160")
    check(all(value == len(BUCKETS) for value in targets.values()), f"target triples {targets}")

    for row in rows:
        scenario, bucket = row["scenario"], row["bucket"]
        target = CFG.BUCKETS[bucket]
        low, high = bucket_bounds(target)
        check(
            row["tokens"] == row["messages_tokens"] + row["tools_tokens"],
            f"{row['id']}: tokens != messages + tools",
        )
        check(
            low <= row["tokens"] <= high,
            f"{row['id']}: tokens {row['tokens']} outside [{low}, {high}]",
        )
        check(
            len(row["messages"]) == 1 and row["messages"][0]["role"] == "user",
            f"{row['id']}: messages shape",
        )
        check(row["constructed"] == _expected_constructed(scenario, bucket),
              f"{row['id']}: constructed flag")
        check(bool(row["tools"]) == (scenario in TOOL_SCENARIOS), f"{row['id']}: tools presence")
        check(isinstance(row["meta"], dict) and bool(row["meta"].get("workload")),
              f"{row['id']}: meta.workload missing")
        check(bool(row["meta"].get("lang")), f"{row['id']}: meta.lang missing")
        if scenario in TOOL_SCENARIOS:
            bad = [value for value in _schema_types(row["tools"]) if value not in _JSON_SCHEMA_TYPES]
            check(not bad, f"{row['id']}: invalid tool schema types {bad[:3]}")
            check(row["tools_tokens"] > 0, f"{row['id']}: tools_tokens must be > 0")

    by_target: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        by_target.setdefault(row["target_id"], {})[row["bucket"]] = row
    for target_id, buckets in by_target.items():
        scenario = buckets["1k"]["scenario"]
        suffixes = set()
        for bucket in BUCKETS:
            content = buckets[bucket]["messages"][0]["content"]
            suffix = _suffix(content, MARKERS[scenario])
            check(suffix is not None, f"{target_id}/{bucket}: suffix marker missing")
            suffixes.add(suffix)
        check(len(suffixes) == 1, f"{target_id}: suffix differs across buckets")
        # Same target => identical tools in every bucket.
        tool_blobs = {json.dumps(buckets[bucket]["tools"], sort_keys=True) for bucket in BUCKETS}
        check(len(tool_blobs) == 1, f"{target_id}: tools differ across buckets")
        if scenario in DOC_SCENARIOS:
            bodies = {
                bucket: _doc_body(
                    buckets[bucket]["messages"][0]["content"],
                    MARKERS[scenario],
                    DOC_PREFIXES[scenario],
                )
                for bucket in BUCKETS
            }
            check(bodies["8k"].startswith(bodies["1k"]), f"{target_id}: 1K not prefix of 8K")
            check(bodies["32k"].startswith(bodies["8k"]), f"{target_id}: 8K not prefix of 32K")

    # No duplicate raw documents (and, for knowledge_qa, no duplicate question).
    for scenario in DOC_SCENARIOS:
        scenario_rows = [row for row in rows if row["scenario"] == scenario and row["bucket"] == "1k"]
        shas = [row["meta"].get("context_sha1") for row in scenario_rows]
        check(len(set(shas)) == len(shas), f"{scenario}: duplicate raw context in selection")
    knowledge_questions = [
        row["messages"][0]["content"].rsplit("\n\nQuestion: ", 1)[1].split("\nA.", 1)[0]
        for row in rows
        if row["scenario"] == "knowledge_qa" and row["bucket"] == "1k"
    ]
    check(
        len(set(knowledge_questions)) == len(knowledge_questions),
        "knowledge_qa: duplicate question in selection",
    )

    # BFCL: stratified category counts + target-first, whole-function catalogue.
    agent_rows = [row for row in rows if row["scenario"] == "general_agent"]
    category_counts = Counter(row["meta"]["category"] for row in agent_rows)
    check(
        dict(sorted(category_counts.items())) == {"multiple": 15, "parallel": 15, "simple": 30},
        f"general_agent category rows {dict(category_counts)}",
    )
    agent_targets = [
        row for row in agent_rows if row["bucket"] == "1k"
    ]
    target_category = Counter(row["meta"]["category"] for row in agent_targets)
    check(
        target_category == Counter({"simple": 10, "multiple": 5, "parallel": 5}),
        f"general_agent target strata {dict(target_category)}",
    )
    for row in agent_rows:
        content = row["messages"][0]["content"]
        check(content.startswith(AGENT_PREFIX), f"{row['id']}: agent prefix")
        index = content.rfind(AGENT_MARKER)
        check(index > 0, f"{row['id']}: agent request marker")
        functions = agent_body_functions(content[len(AGENT_PREFIX):index])
        own_names = [tool["function"]["name"] for tool in row["tools"]]
        visible = [fn.get("function", {}).get("name") for fn in functions]
        own_prefix = visible[: min(len(visible), len(own_names))]
        check(
            own_prefix == own_names[: len(own_prefix)],
            f"{row['id']}: catalogue is not target-functions-first",
        )
        check(row["meta"]["category"] == BFCL_CATEGORY_LABEL[row["meta"]["category_detail"]],
              f"{row['id']}: category/category_detail mismatch")
        check(row["meta"]["own_functions_in_catalog"] is True, f"{row['id']}: own_in_catalog")
        check(row["meta"]["catalog_functions"] >= row["meta"]["own_functions"],
              f"{row['id']}: catalog smaller than own tools")

    # ifeval: target instruction only in the suffix.
    for row in rows:
        if row["scenario"] != "instruction_following":
            continue
        content = row["messages"][0]["content"]
        check(content.startswith("Reference instructions:\n"), f"{row['id']}: ifeval prefix")
        index = content.rfind("\n\nInstruction: ")
        check(index > 0, f"{row['id']}: ifeval marker")
        target = content[index + len("\n\nInstruction: "):]
        check(target not in content[:index], f"{row['id']}: target instruction leaked into reference")

    # math: target problem must not appear in the solved-example block.
    for row in rows:
        if row["scenario"] != "math_logic":
            continue
        content = row["messages"][0]["content"]
        index = content.rfind("\n\nQuestion: ")
        check(index > 0, f"{row['id']}: math marker")
        problem = content[index + len("\n\nQuestion: "):].rsplit("\nAnswer:", 1)[0]
        check(problem not in content[:index], f"{row['id']}: target problem leaked into examples")

    # chinese: 1K body must be the target document's own prefix (already implied by
    # the doc prefix chain), and every target's document must be the body start.
    if errors:
        raise RuntimeError("performance quality assertions failed:\n" + "\n".join(errors[:30]))


def window_requirements(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Declared-token window budget for the 32K bucket (auditable limitation)."""

    tokens_32k = sorted(row["tokens"] for row in rows if row["bucket"] == "32k")
    low, high = tokens_32k[0], tokens_32k[-1]
    window = CFG.BUCKETS["32k"]
    max_testable = window - WINDOW_RESERVE
    testable = sum(1 for value in tokens_32k if value + WINDOW_RESERVE <= window)
    return {
        "declared_32k_tokens": {"min": low, "max": high},
        "per_request_reserve": {
            "max_tokens": RUN_MAX_TOKENS,
            "context_safety_margin_tokens": CONTEXT_SAFETY_MARGIN_TOKENS,
            "decode_guard_token_reserve": DECODE_GUARD_TOKEN_RESERVE,
            "total": WINDOW_RESERVE,
        },
        "required_model_window_tokens": {
            "min": low + WINDOW_RESERVE,
            "max": high + WINDOW_RESERVE,
        },
        "window_32768": {
            "of": len(tokens_32k),
            "testable": testable,
            "not_testable": len(tokens_32k) - testable,
            "max_testable_declared_tokens": max_testable,
        },
        "note": (
            "Runtime window check is declared_tokens + max_tokens + context_safety_margin "
            "+ decode_guard_reserve. A 32,768-token window can only host declared tokens "
            f"<= {max_testable}; the remaining 32K rows are reported as not-testable "
            "(summary.skipped), never truncated. Declared tokens exclude the decode guard."
        ),
    }


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #
def build_provenance() -> Dict[str, Any]:
    """Only the datasets this performance set actually reads (5 datasets / 7 files)."""

    out: Dict[str, Any] = {}
    for entry in CFG.provenance_for("performance"):
        files = [(C.RAW / relative, logical) for relative, logical in entry["files"]]
        out[entry["dataset"]] = C.provenance_entry(
            entry["source_url"], entry["license"], files
        )
    return out


def cell_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    for row in rows:
        key = f"{row['scenario']}|{row['bucket']}"
        entry = stats.setdefault(
            key,
            {"n": 0, "tokens": [], "constructed": 0, "dataset": row["dataset"], "langs": set()},
        )
        entry["n"] += 1
        entry["tokens"].append(row["tokens"])
        entry["constructed"] += int(row["constructed"])
        lang = (row.get("meta") or {}).get("lang")
        if lang:
            entry["langs"].add(lang)
    for entry in stats.values():
        values = sorted(entry.pop("tokens"))
        entry.update(
            tokens_min=values[0],
            tokens_median=values[len(values) // 2],
            tokens_max=values[-1],
            tokens_mean=round(sum(values) / len(values), 1),
            langs=sorted(entry.pop("langs")),
        )
    return stats


def validate_manifest(manifest: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    """Manifest-level contract checks (provenance / osl / new keys)."""

    errors: List[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    provenance = manifest["provenance"]
    check(len(provenance) == 5, f"provenance datasets {len(provenance)} != 5")
    check(
        set(provenance) == {"longbench_v1", "longbench_v2", "ifeval", "gsm8k", "bfcl"},
        f"provenance datasets {sorted(provenance)}",
    )
    total_files = sum(len(entry["files"]) for entry in provenance.values())
    check(total_files == 7, f"provenance files {total_files} != 7")
    check(len(provenance.get("bfcl", {}).get("files", [])) == 3, "bfcl provenance files != 3")
    check("livecodebench" not in provenance, "livecodebench must not be performance provenance")
    check(manifest["osl"] == CFG.OSL == 50, f"osl {manifest['osl']} != 50")
    check(manifest["run_defaults"]["max_tokens"] == CFG.OSL, "run_defaults.max_tokens")
    check(manifest["run_defaults"]["decode_guard"] is True, "run_defaults.decode_guard")
    check(manifest["run_defaults"]["warmup_per_model"] == 1, "run_defaults.warmup_per_model")
    check(manifest["run_defaults"]["warmup_requests"] == 0, "run_defaults.warmup_requests")
    check(manifest["run_defaults"]["repeats"] == 1, "run_defaults.repeats")
    check(len(manifest["sampling"]) == len(CFG.PERF_SOURCES), "sampling entries")
    check(len(manifest["workloads"]) == len(CFG.PERF_SOURCES), "workload entries")
    for scenario in CFG.PERF_SOURCES:
        check(bool(manifest["scenarios"][scenario].get("workload")),
              f"scenarios.{scenario}.workload missing")
        check(manifest["sampling"][scenario]["selected"] == CFG.PERF_PER_BUCKET,
              f"sampling.{scenario}.selected != {CFG.PERF_PER_BUCKET}")
    check(manifest["window_requirements"]["declared_32k_tokens"]["min"] > 0,
          "window_requirements missing")
    for scenario in DOC_SCENARIOS:
        entry = manifest["sampling"][scenario]
        if entry.get("unique_1k_bodies", entry["selected"]) < entry["selected"]:
            check(
                "known_1k_prefix_collisions" in entry,
                f"sampling.{scenario}: 1K prefix collisions not documented",
            )
    long_collision = manifest["sampling"]["long_context"].get("known_1k_prefix_collisions")
    check(
        bool(long_collision) and long_collision.get("selected_targets") == 20,
        "sampling.long_context known_1k_prefix_collisions missing",
    )
    agent_sampling = manifest["sampling"]["general_agent"]
    check(
        agent_sampling.get("category_selected") == {"multiple": 5, "parallel": 5, "simple": 10},
        f"general_agent category_selected {agent_sampling.get('category_selected')}",
    )
    check(
        sum(agent_sampling.get("category_skipped", {}).values())
        == sum(1 for item in agent_sampling["skipped"] if "category" in item),
        "general_agent category_skipped total",
    )
    check(manifest["totals"]["rows"] == len(rows), "totals.rows")
    if errors:
        raise RuntimeError("manifest assertions failed:\n" + "\n".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="*",
        choices=list(BUILDERS),
        help="debug: build only these scenarios and print counts (writes nothing)",
    )
    args = parser.parse_args()
    C.ensure_dirs()
    tok = C.Tokenizer()
    rows: List[Dict[str, Any]] = []
    sampling: Dict[str, Any] = {}
    for name, builder in BUILDERS.items():
        if args.only and name not in args.only:
            continue
        part, stats = builder(tok)
        rows += part
        sampling[name] = stats
        print(
            f"[perf] {name}: {len(part) // len(BUCKETS)} targets / {len(part)} rows; "
            f"skipped={len(stats['skipped'])}"
        )
    if args.only:
        print("[perf] --only dry run: nothing written")
        tok.close()
        return 0

    rows.sort(key=lambda row: (row["scenario"], BUCKET_ORDER[row["bucket"]], row["id"]))
    validate_rows(rows)
    C.write_jsonl(C.TEST_PERF / "testset.jsonl", rows)

    accept_bounds = {
        bucket: dict(zip(("low", "high"), bucket_bounds(target)))
        for bucket, target in CFG.BUCKETS.items()
    }
    manifest = {
        "schema_version": 3,
        "generator_version": GENERATOR_VERSION,
        "seed": CFG.SEED,
        "tokenizer": tok.info(),
        "tokenizer_cache": {"calls": tok.calls, "cache_hits": tok.cache_hits},
        "buckets": CFG.BUCKETS,
        "accept_range": [CFG.ACCEPT_LOW, CFG.ACCEPT_HIGH],
        "accept_bounds": accept_bounds,
        "accept_rule": (
            "integer token counts: floor(0.95*target) <= tokens <= floor(1.05*target) "
            "(1K lower bound is 972, not 972.8; fixes the previous 0.8-token false rejection)"
        ),
        "osl": CFG.OSL,
        "run_defaults": {
            "max_tokens": CFG.OSL,
            "decode_guard": True,
            "warmup_per_model": 1,
            "warmup_requests": 0,
            "repeats": 1,
        },
        "tokens_semantics": (
            "tokens = messages + tools (messages_tokens + tools_tokens); messages use the "
            "reference chat template, tools are counted as their JSON text"
        ),
        "bucket_basis": "total prompt tokens (messages + tools)",
        "files": ["testset.jsonl"],
        "construction_rules": CFG.CONSTRUCTION_RULES,
        "workloads": {scenario: workload_of(scenario) for scenario in CFG.PERF_SOURCES},
        "scenarios": {
            scenario: {
                "dataset": spec["dataset"],
                "workload": workload_of(scenario),
                "construction": spec["construction"],
                "domain": spec.get("domain"),
                "task": spec.get("task"),
            }
            for scenario, spec in CFG.PERF_SOURCES.items()
        },
        "window_requirements": window_requirements(rows),
        "provenance": build_provenance(),
        "sampling": sampling,
        "stats": cell_stats(rows),
        "tokens_mean": round(sum(row["tokens"] for row in rows) / len(rows), 1),
        "totals": {
            "rows": len(rows),
            "cells": len({(row["scenario"], row["bucket"]) for row in rows}),
            "per_cell": CFG.PERF_PER_BUCKET,
            "measured_requests": len(rows),
            "targets": len({row["target_id"] for row in rows}),
        },
        "default_run": {
            "measured_requests": len(rows),
            "cells": len({(row["scenario"], row["bucket"]) for row in rows}),
        },
        "sets": {
            "performance": {
                "file": "testset.jsonl",
                "per_bucket": CFG.PERF_PER_BUCKET,
                "rows": len(rows),
            },
            "performance_10": {
                "file": "testset_10.jsonl",
                "per_bucket": CFG.PERF_SUBSET_PER_BUCKET,
                "rows": CFG.EXPECTED["perf_subset_rows"],
                "selection": "per scenario: sorted target_id[::2] (10 of 20), all three buckets kept",
            },
        },
    }
    validate_manifest(manifest, rows)
    C.write_json(C.TEST_PERF / "manifest.json", manifest)
    tok.close()
    print(f"[perf] rows={len(rows)} cells={manifest['totals']['cells']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
