"""Build the accuracy test set: 6,589 rows across 8 scenarios.

Design: ``dataset/TEST_SAMPLE_SELECTION.md`` (v3).  Zero-shot, fixed per-scenario
prompt templates, no length bucketing, no few-shot.  Every row carries the judge
inputs its judge contract needs (choices, reference answers, instruction ids and
kwargs, hidden tests, possible answers, or retrieval corpus).

Scenario sizes: 6 x 250 (knowledge_qa / instruction_following / long_context /
math_logic / coding / search_agent) + general_agent 889 (BFCL v4, 11 single-turn
categories) + multilingual 4,200 (Global-MMLU, 42 languages x 100 paired ids).

The accuracy set is an **offline asset**: it is not wired into the run (the
current ``docs/TEST_PLAN.md`` is performance-only) and no evaluator/runner is
implemented here.  This builder only prepares data; ``manifest.json`` records the
judge contracts and their ``implementation_status`` as ``contract-only``.

Outputs:
    dataset/test/accuracy/testset.jsonl     6,589 rows
    dataset/test/accuracy/manifest.json
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pyarrow.parquet as pq

import config as CFG
import ds_common as C

GENERATOR_VERSION = "tool/dataset/build_acc.py v5"
SEED = CFG.SEED

ROW_FIELDS = {"id", "scenario", "dataset", "judge", "tokens", "messages", "choices", "answer", "tools", "meta"}

#: ``meta.upstream_id`` format per scenario.  The value always starts with the
#: upstream dataset name (plus ``task``/``lang`` where the raw key is not unique
#: inside the dataset), so the id is traceable and unique across all rows.
UPSTREAM_ID_FORMAT: Dict[str, str] = {
    "knowledge_qa": "mmlu_pro:<question_id>",
    "instruction_following": "ifbench:<key>",
    "long_context": "longbench_v1:<task>:<_id>",
    "math_logic": "math_500:<unique_id>",
    "coding": "livecodebench:<question_id>",
    "general_agent": "bfcl:<bfcl_id>",
    "search_agent": "hotpotqa:<id>",
    "multilingual": "global_mmlu:<lang>:<sample_id>",
}

JUDGE_IMPLEMENTATION_STATUS = (
    "contract-only: the judge contract is declared here, but accuracy is not wired "
    "into the run (current docs/TEST_PLAN.md is performance-only); no evaluator is invoked."
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def count_tools(tok: C.Tokenizer, tools: Optional[List[Dict[str, Any]]]) -> int:
    return tok.count_text(json.dumps(tools, ensure_ascii=False)) if tools else 0


def rec(
    scenario: str,
    idx: int,
    dataset: str,
    judge: str,
    messages: List[Dict[str, str]],
    tok: C.Tokenizer,
    answer: Any,
    choices: Optional[List[str]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    tools = C.normalize_tools(tools)
    mt = tok.count_messages(messages)
    tt = count_tools(tok, tools)
    return {
        "id": f"acc-{scenario}-{idx:03d}",
        "scenario": scenario,
        "dataset": dataset,
        "judge": judge,
        "tokens": mt + tt,
        "messages": messages,
        "choices": choices,
        "answer": answer,
        "tools": tools,
        "meta": meta or {},
    }


def round_robin(groups: Dict[Any, List[Any]], n: int, seed: int) -> List[Any]:
    """Interleave shuffled strata until ``n`` items are taken."""
    rng = random.Random(seed)
    for key in groups:
        rng.shuffle(groups[key])
    keys = sorted(groups, key=str)
    out: List[Any] = []
    i = 0
    while len(out) < n and any(groups[key] for key in keys):
        key = keys[i % len(keys)]
        if groups[key]:
            out.append(groups[key].pop())
        i += 1
    return out


def take_shuffled(rows: List[Any], k: int, seed: int) -> List[Any]:
    rng = random.Random(seed)
    pool = list(rows)
    rng.shuffle(pool)
    return pool[:k]


#: LongBench v1 accuracy length bands: ``(name, floor, ceiling_exclusive)``.
#: The top band is capped by ``ACC_MAX_TOKENS`` (inclusive), not by its ceiling.
LENGTH_BANDS: Tuple[Tuple[str, int, int], ...] = (
    ("4k-8k", CFG.ACC_MIN_TOKENS, 8192),
    ("8k-16k", 8192, 16384),
    ("16k-32k", 16384, CFG.ACC_MAX_TOKENS + 1),
)


def length_band(tokens: int) -> Optional[str]:
    """Length band for a reference token count, or ``None`` outside 4K-32K."""
    for name, floor, ceiling in LENGTH_BANDS:
        if floor <= tokens < ceiling:
            return name
    return None


def pick_round_robin(
    groups: Dict[str, List[Any]], target: int, seed: int
) -> List[Tuple[str, Any]]:
    """Shuffle each group, then interleave groups until ``target`` picks are made."""
    rng = random.Random(seed)
    for key in groups:
        rng.shuffle(groups[key])
    keys = list(groups)
    out: List[Tuple[str, Any]] = []
    i = 0
    while len(out) < target and any(groups[key] for key in keys):
        key = keys[i % len(keys)]
        if groups[key]:
            out.append((key, groups[key].pop()))
        i += 1
    if len(out) != target:
        raise RuntimeError(f"round-robin pools exhausted: {len(out)} < {target}")
    return out


def mcq_choices(options: List[str]) -> List[str]:
    letters = "ABCDEFGHIJ"
    return [f"{letters[i]}. {options[i]}" for i in range(len(options))]


def norm_key(text: Optional[str]) -> str:
    """Whitespace-insensitive key for cross-row / cross-scenario dedupe."""
    return " ".join((text or "").split())


def prompt_signature(row: Dict[str, Any]) -> str:
    return json.dumps(row["messages"], ensure_ascii=False, sort_keys=True)


# --------------------------------------------------------------------------- #
# 1. knowledge_qa: MMLU-Pro (0-shot CoT, choices kept)
# --------------------------------------------------------------------------- #
def build_mmlu_pro(tok: C.Tokenizer, n: int, state: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = pq.read_table(C.RAW / "mmlu_pro" / "test.parquet").to_pylist()
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["category"], []).append(row)
    picked = round_robin(groups, n, SEED)
    out: List[Dict[str, Any]] = []
    for row in picked:
        options = ast.literal_eval(row["options"]) if isinstance(row["options"], str) else list(row["options"])
        options = [str(value) for value in options]
        choices = mcq_choices(options)
        letters = "ABCDEFGHIJ"[: len(options)]
        prompt = (
            "Answer the following multiple choice question. Think step by step, then end with a "
            "single line of the form 'Answer: X', where X is the option letter "
            f"({letters[0]}-{letters[-1]}).\n\n"
            f"Question: {row['question']}\n" + "\n".join(choices) + "\nAnswer:"
        )
        out.append(
            rec(
                "knowledge_qa", len(out), "mmlu_pro", "mc_letter",
                [{"role": "user", "content": prompt}], tok, row["answer"], choices,
                meta={
                    "lang": "en",
                    "category": row["category"],
                    "question_id": row.get("question_id"),
                    "upstream_id": f"mmlu_pro:{row['question_id']}",
                    "num_options": len(options),
                },
            )
        )
    state.setdefault("sampling", {})["knowledge_qa"] = {
        "upstream_rows": len(rows),
        "strata": "category",
        "selected": len(out),
    }
    return out


# --------------------------------------------------------------------------- #
# 2. instruction_following: IFBench (verbatim prompt; judge inputs in meta)
# --------------------------------------------------------------------------- #
def build_ifbench(tok: C.Tokenizer, n: int, state: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = C.load_jsonl(C.RAW / "ifbench" / "IFBench_test.jsonl")
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        ids = row.get("instruction_id_list") or ["other"]
        groups.setdefault(str(ids[0]), []).append(row)
    picked = round_robin(groups, n, SEED + 1)
    out: List[Dict[str, Any]] = []
    for row in picked:
        upstream_id = f"ifbench:{row.get('key')}"
        out.append(
            rec(
                "instruction_following", len(out), "ifbench", "ifbench",
                [{"role": "user", "content": row["prompt"]}], tok, None,
                meta={
                    "lang": "en",
                    "ifbench_key": row.get("key"),
                    "instruction_id_list": row.get("instruction_id_list"),
                    "kwargs": row.get("kwargs"),
                    "upstream_id": upstream_id,
                },
            )
        )
    state.setdefault("sampling", {})["instruction_following"] = {
        "upstream_rows": len(rows),
        "strata": "primary instruction id",
        "selected": len(out),
    }
    return out


# --------------------------------------------------------------------------- #
# 3. long_context: LongBench v1 (official dataset2prompt templates, dedupe+union)
# --------------------------------------------------------------------------- #
def _dedupe_longbench(task: str, rows: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Merge upstream rows by (task, input, context); union their gold answers."""
    merged: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    order: List[Tuple[str, str, str]] = []
    duplicated = 0
    for row in rows:
        key = (task, row.get("input") or "", row.get("context") or "")
        answers = [str(item) for item in (row.get("answers") or []) if str(item).strip()]
        if key not in merged:
            merged[key] = {"row": row, "answers": answers, "merged_from": 1}
            order.append(key)
        else:
            duplicated += 1
            bucket = merged[key]
            bucket["merged_from"] += 1
            for answer in answers:
                if answer not in bucket["answers"]:
                    bucket["answers"].append(answer)
    return [merged[key] for key in order], duplicated


def build_long_context(tok: C.Tokenizer, n: int, state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """LongBench v1 stratified by length band x task (4K-32K prompts only).

    The whole 16K-32K pool is taken (51 rows across narrativeqa/qasper/hotpotqa);
    the 4K-8K and 8K-16K bands are filled round-robin across the four tasks so no
    single task dominates a band.  Per-task quotas are therefore no longer equal.
    """
    spec = CFG.ACC_SOURCES["long_context"]
    tasks = list(spec["tasks"])
    band_targets: Dict[str, int] = dict(spec["length_bands"])
    if sum(band_targets.values()) != n:
        raise RuntimeError(f"long_context: band targets {band_targets} do not sum to {n}")

    # Dedupe per task, render the official template, and bucket by token length.
    pools: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    task_stats: Dict[str, Any] = {}
    for task in tasks:
        rows = C.load_zip_jsonl(C.RAW / "longbench_v1" / "data.zip", f"data/{task}.jsonl")
        uniq, duplicated = _dedupe_longbench(task, rows)
        template = CFG.LONG_CONTEXT_TEMPLATES[task]
        banded: Dict[str, List[Dict[str, Any]]] = {band: [] for band in band_targets}
        for entry in uniq:
            row = entry["row"]
            prompt = template.format(context=row.get("context", ""), input=row.get("input", ""))
            tokens = tok.count_messages([{"role": "user", "content": prompt}])
            band = length_band(tokens)
            if band is not None:
                banded[band].append(
                    {"task": task, "entry": entry, "prompt": prompt, "tokens": tokens}
                )
        pools[task] = banded
        task_stats[task] = {
            "upstream_rows": len(rows),
            "unique_rows_after_dedupe": len(uniq),
            "merged_duplicate_rows": duplicated,
            "band_pools": {band: len(items) for band, items in banded.items()},
        }

    out: List[Dict[str, Any]] = []
    band_pools: Dict[str, Dict[str, int]] = {}
    band_selected: Dict[str, int] = {}
    band_selected_by_task: Dict[str, Dict[str, int]] = {}
    hotpot_questions: set = set()
    for band_index, (band, target) in enumerate(band_targets.items()):
        groups = {task: list(pools[task][band]) for task in tasks}
        pool_sizes = {task: len(items) for task, items in groups.items()}
        total_pool = sum(pool_sizes.values())
        band_pools[band] = pool_sizes
        if total_pool < target:
            raise RuntimeError(f"long_context band {band}: pool {total_pool} < target {target}")
        if band == "16k-32k" and total_pool != target:
            # The design takes the *whole* 16K-32K pool, not a sample of it.
            raise RuntimeError(
                f"long_context band {band}: pool {total_pool} != target {target} "
                "(the whole pool must be taken)"
            )
        picked = pick_round_robin(groups, target, SEED + 100 + band_index)
        by_task = Counter(task for task, _item in picked)
        band_selected[band] = len(picked)
        band_selected_by_task[band] = {task: by_task.get(task, 0) for task in tasks}
        for task, item in picked:
            entry = item["entry"]
            row = entry["row"]
            answers = list(entry["answers"])
            if not answers:  # pragma: no cover - asserted again at the end
                raise RuntimeError(f"long_context {task}: empty reference answers for _id={row.get('_id')}")
            out.append(
                rec(
                    "long_context", len(out), f"longbench_v1:{task}", "qa_f1",
                    [{"role": "user", "content": item["prompt"]}], tok, answers[0],
                    meta={
                        "lang": "zh" if task == "dureader" else "en",
                        "task": task,
                        "length_band": band,
                        "answers": answers,
                        "prompt_tokens": item["tokens"],
                        "upstream_id": f"longbench_v1:{task}:{row.get('_id')}",
                        "upstream_duplicates_merged": entry["merged_from"] - 1,
                    },
                )
            )
            if task == "hotpotqa":
                hotpot_questions.add(norm_key(row.get("input")))
    state.setdefault("sampling", {})["long_context"] = {
        "min_prompt_tokens": CFG.ACC_MIN_TOKENS,
        "max_prompt_tokens": CFG.ACC_MAX_TOKENS,
        "length_bands": {
            "targets": dict(band_targets),
            "ranges": {
                band: (
                    f"{floor} <= tokens < {ceiling}"
                    if band != "16k-32k"
                    else f"{floor} <= tokens <= {CFG.ACC_MAX_TOKENS}"
                )
                for band, floor, ceiling in LENGTH_BANDS
            },
            "pools": band_pools,
            "selected": band_selected,
            "selected_by_task": band_selected_by_task,
        },
        "tasks": task_stats,
        "dedupe_key": "(task, input, context); answers unioned",
    }
    state["long_context_hotpot_questions"] = hotpot_questions
    return out


# --------------------------------------------------------------------------- #
# 4. math_logic: MATH-500
# --------------------------------------------------------------------------- #
def build_math(tok: C.Tokenizer, n: int, state: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = [json.loads(line) for line in (C.RAW / "math_500" / "test.jsonl").open() if line.strip()]
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row.get("subject"), row.get("level")), []).append(row)
    picked = round_robin(groups, n, SEED + 3)
    out: List[Dict[str, Any]] = []
    for row in picked:
        prompt = (
            "Solve the problem. Show your reasoning, then put the final answer in \\boxed{}.\n\n"
            f"Problem: {row['problem']}\nAnswer:"
        )
        out.append(
            rec(
                "math_logic", len(out), "math_500", "math",
                [{"role": "user", "content": prompt}], tok, row["answer"],
                meta={
                    "lang": "en",
                    "level": row.get("level"),
                    "subject": row.get("subject"),
                    "upstream_id": f"math_500:{row.get('unique_id')}",
                },
            )
        )
    state.setdefault("sampling", {})["math_logic"] = {
        "upstream_rows": len(rows),
        "strata": "subject x level",
        "selected": len(out),
    }
    return out


# --------------------------------------------------------------------------- #
# 5. coding: LiveCodeBench (functional for LeetCode, stdin/stdout for AtCoder)
# --------------------------------------------------------------------------- #
LCB_SOURCES = ("test6", "test5")


def _lcb_judgeable(name: str) -> List[Any]:
    """Judgeable rows of one LCB release file: decode hidden tests, cap size."""
    path = C.RAW / "livecodebench" / f"{name}.jsonl"
    out: List[Any] = []
    with path.open(encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not row.get("public_test_cases"):
                continue
            try:
                hidden_all = C.decode_lcb_tests(row["private_test_cases"])
            except Exception:  # noqa: BLE001 - malformed upstream row
                continue
            kept = [test for test in hidden_all
                    if len(json.dumps(test, ensure_ascii=False)) <= CFG.ACC_MAX_TEST_BYTES]
            if not kept:
                continue  # every hidden test is oversized; cannot judge locally
            out.append((row, kept, len(hidden_all) - len(kept), name))
    return out


def _lcb_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    try:
        value = json.loads(row.get("metadata") or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def lcb_testtype(tests: List[Dict[str, Any]]) -> str:
    """Derive a row's test type from its hidden tests (functional / stdin / mixed)."""
    kinds = {str(test.get("testtype")) for test in tests}
    if kinds == {"functional"}:
        return "functional"
    if kinds == {"stdin"}:
        return "stdin"
    return "mixed"


def _lcb_prompt(row: Dict[str, Any], starter_code: str) -> str:
    """Official-style function prompt when starter code exists, else stdin/stdout."""
    if starter_code:
        return (
            "### Question:\n"
            f"{row['question_content']}\n\n"
            "### Format: You will use the following starter code to write the solution to the problem "
            "and enclose your code within delimiters.\n"
            f"```python\n{starter_code}\n```\n\n"
            "### Answer: (use the provided format with backticks)"
        )
    return (
        "### Question:\n"
        f"{row['question_content']}\n\n"
        "### Format: Write a complete Python program that reads from standard input and writes to "
        "standard output; enclose your code within delimiters.\n\n"
        "### Answer: (use the provided format with backticks)"
    )


def build_livecodebench(tok: C.Tokenizer, n: int, state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Newest release window first: all judgeable test6, then newest test5 rows."""
    seen: set = set()
    picked: List[Any] = []
    for item in _lcb_judgeable("test6"):
        qid = item[0].get("question_id")
        if qid in seen:
            continue
        seen.add(qid)
        picked.append(item)
    newest_first = sorted(
        _lcb_judgeable("test5"),
        key=lambda item: item[0].get("contest_date") or "",
        reverse=True,
    )
    for item in newest_first:
        if len(picked) >= n:
            break
        qid = item[0].get("question_id")
        if qid in seen:
            continue
        seen.add(qid)
        picked.append(item)
    if len(picked) < n:
        raise RuntimeError(f"livecodebench: only {len(picked)} judgeable rows in latest windows")
    out: List[Dict[str, Any]] = []
    for row, kept, dropped, source in picked[:n]:
        starter_code = row.get("starter_code") or ""
        prompt = _lcb_prompt(row, starter_code)
        public = json.loads(row["public_test_cases"])
        metadata = _lcb_metadata(row)
        testtype = lcb_testtype(kept)
        out.append(
            rec(
                "coding", len(out), "livecodebench", "code_pass1",
                [{"role": "user", "content": prompt}], tok, None,
                meta={
                    "lang": "en",
                    "question_id": row.get("question_id"),
                    "upstream_id": f"livecodebench:{row.get('question_id')}",
                    "difficulty": row.get("difficulty"),
                    "platform": row.get("platform"),
                    "contest_date": row.get("contest_date"),
                    "source_file": f"livecodebench/{source}.jsonl",
                    "starter_code": starter_code,
                    "testtype": testtype,
                    "fn_name": metadata.get("func_name") if starter_code else None,
                    "has_functional_tests": any(
                        str(test.get("testtype")) == "functional" for test in kept
                    ),
                    "public_tests": public,
                    "tests": kept,
                    "tests_total": len(kept) + dropped,
                    "tests_dropped_large": dropped,
                    "tests_max_bytes": CFG.ACC_MAX_TEST_BYTES,
                    "window": CFG.ACC_SOURCES["coding"]["window"],
                },
            )
        )
    state.setdefault("sampling", {})["coding"] = {
        "window": CFG.ACC_SOURCES["coding"]["window"],
        "selected": len(out),
        "testtypes": dict(Counter(row["meta"]["testtype"] for row in out)),
        "hidden_tests_kept": sum(len(row["meta"]["tests"]) for row in out),
        "hidden_tests_dropped_large": sum(row["meta"]["tests_dropped_large"] for row in out),
    }
    return out


# --------------------------------------------------------------------------- #
# 6. general_agent: BFCL v4 single-turn, 11 categories (user-question dedupe)
# --------------------------------------------------------------------------- #
def _bfcl(category: str) -> List[Dict[str, Any]]:
    return C.load_jsonl(C.RAW / "bfcl" / f"BFCL_v4_{category}.json")


def _possible_answers(category: str) -> Dict[str, Any]:
    """Official possible answers for one category (``{}`` when upstream has none)."""
    path = C.RAW / "bfcl" / "possible_answer" / f"BFCL_v4_{category}.json"
    if not path.is_file():
        return {}
    return {row["id"]: row.get("ground_truth") for row in C.load_jsonl(path)}


def _bfcl_user_question(row: Dict[str, Any]) -> str:
    return "\n".join(
        item.get("content", "") for turn in row.get("question", []) for item in turn
    )


def build_bfcl(tok: C.Tokenizer, n: int, state: Dict[str, Any]) -> List[Dict[str, Any]]:
    spec = CFG.ACC_SOURCES["general_agent"]
    plan = [(category, spec["category_counts"][category]) for category in spec["categories"]]
    expected_total = sum(quota for _category, quota in plan)
    if n != expected_total:
        raise RuntimeError(f"general_agent: n={n} != category targets {expected_total}")
    out: List[Dict[str, Any]] = []
    used_questions: set = set()
    stats: Dict[str, Any] = {}
    for offset, (category, quota) in enumerate(plan):
        gold = _possible_answers(category)
        rows = _bfcl(category)
        rng = random.Random(SEED + 10 + offset)
        shuffled = list(rows)
        rng.shuffle(shuffled)
        candidates: List[Dict[str, Any]] = []
        local_seen: set = set()
        skipped = 0
        for row in shuffled:
            qkey = norm_key(_bfcl_user_question(row))
            if qkey in used_questions or qkey in local_seen:
                skipped += 1
                continue
            local_seen.add(qkey)
            candidates.append(row)
        chosen = candidates[:quota]  # a pool smaller than the quota contributes all of it
        used_questions.update(norm_key(_bfcl_user_question(row)) for row in chosen)
        for row in chosen:
            turns = [item.get("content", "") for turn in row.get("question", []) for item in turn]
            tools = C.normalize_tools(
                [{"type": "function", "function": item.get("function", item)}
                 for item in row.get("function", [])]
            )
            messages = [
                {"role": "system", "content": "You are a function-calling assistant. Use the provided tools when needed."},
                {"role": "user", "content": "\n".join(turns)},
            ]
            # ``irrelevance`` ships no possible_answer file: the empty list is the
            # gold ("no tool call is correct"); every other category must have gold.
            possible = [] if category == "irrelevance" else gold.get(row["id"])
            out.append(
                rec(
                    "general_agent", len(out), "bfcl", "bfcl_ast", messages, tok,
                    possible, tools=tools,
                    meta={
                        "lang": "en",
                        "category": category,
                        "category_type": spec["category_type"],
                        "bfcl_id": row["id"],
                        "upstream_id": f"bfcl:{row['id']}",
                        "possible_answers": possible,
                    },
                )
            )
        stats[category] = {
            "upstream_rows": len(rows),
            "unique_questions_after_dedupe": len(candidates),
            "rows_skipped_duplicate_question": skipped,
            "target": quota,
            "selected": len(chosen),
        }
    if len(out) != expected_total:
        raise RuntimeError(
            f"general_agent: built {len(out)} rows, category targets sum to {expected_total}"
        )
    state.setdefault("sampling", {})["general_agent"] = {
        "dedupe": (
            "user question text (whitespace-normalised), global across all "
            f"{len(plan)} single-turn categories"
        ),
        "category_type": spec["category_type"],
        "category_order": list(spec["categories"]),
        "category_counts": dict(spec["category_counts"]),
        "categories": stats,
        "unique_questions_selected": len(used_questions),
        "selected": len(out),
    }
    return out


# --------------------------------------------------------------------------- #
# 7. search_agent: HotpotQA distractor, canonical subset + retrieval tools
# --------------------------------------------------------------------------- #
def build_hotpotqa(tok: C.Tokenizer, n: int, state: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = pq.read_table(
        C.RAW / "hotpotqa" / "distractor" / "validation-00000-of-00001.parquet"
    ).to_pylist()
    canonical = [
        row
        for row in rows
        if len(row["context"]["title"]) == 10 and len(row["supporting_facts"]["title"]) == 2
    ]
    excluded = set(state.get("long_context_hotpot_questions") or set())
    excluded_chosen = [row for row in canonical if norm_key(row["question"]) in excluded]
    pool = [row for row in canonical if norm_key(row["question"]) not in excluded]
    bridge = [row for row in pool if row.get("type") == "bridge"]
    comparison = [row for row in pool if row.get("type") != "bridge"]
    # proportional to the canonical pool (bridge 3827 / comparison 1113)
    n_bridge = round(n * sum(1 for row in canonical if row.get("type") == "bridge") / max(1, len(canonical)))
    n_comparison = n - n_bridge
    if len(bridge) < n_bridge or len(comparison) < n_comparison:
        raise RuntimeError(
            f"search_agent: pool too small after cross-scenario exclusion "
            f"(bridge={len(bridge)}/{n_bridge}, comparison={len(comparison)}/{n_comparison})"
        )
    picks = take_shuffled(bridge, n_bridge, SEED + 20) + take_shuffled(comparison, n_comparison, SEED + 21)
    picks.sort(key=lambda row: row["id"])
    out: List[Dict[str, Any]] = []
    for row in picks:
        context = row["context"]
        paragraphs = [
            {"title": title, "sentences": sentences}
            for title, sentences in zip(context["title"], context["sentences"])
        ]
        supporting = [
            {"title": title, "sent_id": int(sent_id)}
            for title, sent_id in zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"])
        ]
        corpus_text = json.dumps(paragraphs, ensure_ascii=False)
        prompt = (
            "Use the search and lookup tools to inspect the provided passages, then answer the "
            f"question.\n\nQuestion: {row['question']}"
        )
        out.append(
            rec(
                "search_agent", len(out), "hotpotqa", "qa_f1",
                [{"role": "user", "content": prompt}], tok, row["answer"],
                tools=C.normalize_tools(CFG.SEARCH_TOOLS),
                meta={
                    "lang": "en",
                    "type": row.get("type"),
                    "level": row.get("level"),
                    "upstream_id": f"hotpotqa:{row['id']}",
                    "paragraphs": paragraphs,
                    "supporting_facts": supporting,
                    "answers": [row["answer"]],
                    # Corpus is judge/retrieval input, NOT part of the prompt token count.
                    "corpus_chars": len(corpus_text),
                    "corpus_tokens": tok.count_text(corpus_text),
                },
            )
        )
    overlap = sum(
        1
        for row in out
        if norm_key(row["messages"][0]["content"].split("Question: ", 1)[1]) in excluded
    )
    state.setdefault("sampling", {})["search_agent"] = {
        "canonical_pool": len(canonical),
        "excluded_by_long_context_hotpotqa": len(excluded_chosen),
        "pool_after_exclusion": len(pool),
        "bridge_selected": n_bridge,
        "comparison_selected": n_comparison,
        "matching_excluded_questions_selected": overlap,
    }
    state["cross_scenario_duplicate_questions"] = overlap
    return out[:n]


# --------------------------------------------------------------------------- #
# 8. multilingual: Global-MMLU, 42 languages x the same 100 sample_ids
# --------------------------------------------------------------------------- #
def _gmmu_rows(lang: str) -> List[Dict[str, Any]]:
    return pq.read_table(C.RAW / "global_mmlu" / lang / "test.parquet").to_pylist()


def _gmmu_paired_ids(
    sample_ids_by_subject: Dict[str, List[str]], per_language: int
) -> List[str]:
    """Pick ``per_language`` sample_ids by round-robin over the English subjects."""
    rng = random.Random(SEED + 6)
    keys = sorted(sample_ids_by_subject, key=str)
    groups = {key: list(sample_ids_by_subject[key]) for key in keys}
    for key in keys:
        rng.shuffle(groups[key])
    picked: List[str] = []
    i = 0
    while len(picked) < per_language and any(groups[key] for key in keys):
        key = keys[i % len(keys)]
        if groups[key]:
            picked.append(groups[key].pop())
        i += 1
    if len(picked) != per_language:
        raise RuntimeError(
            f"global_mmlu: only {len(picked)} sample_ids selectable, need {per_language}"
        )
    return picked


def build_global_mmlu(tok: C.Tokenizer, n: int, state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """42 languages x 100 identical sample_ids (translated counterparts of one question set)."""
    spec = CFG.ACC_SOURCES["multilingual"]
    languages = list(spec["languages"])
    per_language = spec["per_language"]
    if n != per_language * len(languages):
        raise RuntimeError(f"global_mmlu: n={n} != {per_language} x {len(languages)}")

    reference = {row["sample_id"]: row for row in _gmmu_rows("en")}
    sample_ids_by_subject: Dict[str, List[str]] = defaultdict(list)
    for sample_id, row in reference.items():
        sample_ids_by_subject[row["subject"]].append(sample_id)
    paired_ids = _gmmu_paired_ids(sample_ids_by_subject, per_language)
    paired_id_set = set(paired_ids)
    if len(paired_id_set) != per_language:
        raise RuntimeError("global_mmlu: paired sample_ids are not unique")
    selected_subjects = {reference[sample_id]["subject"] for sample_id in paired_ids}
    selected_categories = {reference[sample_id]["subject_category"] for sample_id in paired_ids}

    upstream_rows: Dict[str, int] = {}
    out: List[Dict[str, Any]] = []
    for lang in languages:
        rows = _gmmu_rows(lang)
        upstream_rows[lang] = len(rows)
        by_id = {row["sample_id"]: row for row in rows}
        missing = [sample_id for sample_id in paired_ids if sample_id not in by_id]
        if missing:
            raise RuntimeError(f"global_mmlu {lang}: {len(missing)} paired sample_ids missing")
        for sample_id in paired_ids:
            row = by_id[sample_id]
            options = [str(row[f"option_{letter}"]) for letter in "abcd"]
            choices = mcq_choices(options)
            prompt = (
                "Answer the following multiple choice question. Reply with the option letter only.\n\n"
                f"Question: {row['question']}\n" + "\n".join(choices) + "\nAnswer:"
            )
            out.append(
                rec(
                    "multilingual", len(out), "global_mmlu", "mc_letter",
                    [{"role": "user", "content": prompt}], tok, row["answer"], choices,
                    meta={
                        "lang": lang,
                        "is_en_reference": lang == "en",
                        "subject": row.get("subject"),
                        "subject_category": row.get("subject_category"),
                        "sample_id": sample_id,
                        "upstream_id": f"global_mmlu:{lang}:{sample_id}",
                        "num_options": len(options),
                    },
                )
            )
    state.setdefault("sampling", {})["multilingual"] = {
        "upstream_rows_per_language": upstream_rows[languages[0]],
        "upstream_rows_identical_across_languages": len(set(upstream_rows.values())) == 1,
        "languages": languages,
        "language_count": len(languages),
        "per_language": per_language,
        "paired_ids": len(paired_ids),
        "paired_design": spec["pairing"],
        "paired_id_selection": (
            "subject round-robin over the common sample_ids (English reference rows); "
            "every language uses the same id list"
        ),
        "subjects_covered": len(selected_subjects),
        "subject_categories_covered": sorted(selected_categories),
        "selected": len(out),
    }
    return out


BUILDERS: Dict[str, Callable[[C.Tokenizer, int, Dict[str, Any]], List[Dict[str, Any]]]] = {
    "knowledge_qa": build_mmlu_pro,
    "instruction_following": build_ifbench,
    "long_context": build_long_context,
    "math_logic": build_math,
    "coding": build_livecodebench,
    "general_agent": build_bfcl,
    "search_agent": build_hotpotqa,
    "multilingual": build_global_mmlu,
}


# --------------------------------------------------------------------------- #
# build assertions (run before any artifact is written)
# --------------------------------------------------------------------------- #
def _judge_field_errors(rows: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    by_scenario: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scenario[row["scenario"]].append(row)

    for row in by_scenario["knowledge_qa"]:
        choices = row["choices"]
        if not isinstance(choices, list) or len(choices) < 3:
            errors.append(f"{row['id']}: mmlu choices")
        else:
            allowed = "ABCDEFGHIJ"[: len(choices)]
            if not (isinstance(row["answer"], str) and row["answer"] in allowed):
                errors.append(f"{row['id']}: mmlu answer out of range")
        if not row["meta"].get("category"):
            errors.append(f"{row['id']}: missing category")

    for row in by_scenario["instruction_following"]:
        meta = row["meta"]
        if meta.get("ifbench_key") in (None, ""):
            errors.append(f"{row['id']}: missing ifbench_key")
        if not isinstance(meta.get("instruction_id_list"), list) or not meta["instruction_id_list"]:
            errors.append(f"{row['id']}: missing instruction_id_list")
        if not isinstance(meta.get("kwargs"), list) or not meta["kwargs"]:
            errors.append(f"{row['id']}: missing kwargs")

    for row in by_scenario["long_context"]:
        meta = row["meta"]
        answers = meta.get("answers")
        if not isinstance(answers, list) or not answers or not all(
            isinstance(item, str) and item.strip() for item in answers
        ):
            errors.append(f"{row['id']}: empty long_context answers")
        elif row["answer"] != answers[0]:
            errors.append(f"{row['id']}: long_context answer != answers[0]")
        tokens = row["tokens"]
        if not (CFG.ACC_MIN_TOKENS <= tokens <= CFG.ACC_MAX_TOKENS):
            errors.append(
                f"{row['id']}: long_context tokens {tokens} outside "
                f"{CFG.ACC_MIN_TOKENS}-{CFG.ACC_MAX_TOKENS}"
            )
        band = meta.get("length_band")
        expected_band = length_band(tokens)
        if band is None or band != expected_band:
            errors.append(f"{row['id']}: length_band {band} != {expected_band}")
        if meta.get("prompt_tokens") != tokens:
            errors.append(f"{row['id']}: prompt_tokens != tokens")

    for row in by_scenario["math_logic"]:
        if not isinstance(row["answer"], str) or not row["answer"].strip():
            errors.append(f"{row['id']}: empty math answer")

    for row in by_scenario["coding"]:
        meta = row["meta"]
        tests = meta.get("tests")
        if not isinstance(tests, list) or not tests:
            errors.append(f"{row['id']}: no hidden tests")
            continue
        if any(len(json.dumps(test, ensure_ascii=False)) > CFG.ACC_MAX_TEST_BYTES for test in tests):
            errors.append(f"{row['id']}: oversized hidden test")
        derived = lcb_testtype(tests)
        if meta.get("testtype") != derived:
            errors.append(f"{row['id']}: testtype {meta.get('testtype')} != tests {derived}")
        if not isinstance(meta.get("public_tests"), list):
            errors.append(f"{row['id']}: missing public tests")
        if meta.get("tests_total") != len(tests) + int(meta.get("tests_dropped_large") or 0):
            errors.append(f"{row['id']}: test accounting")
        if meta.get("testtype") == "functional":
            if not meta.get("fn_name"):
                errors.append(f"{row['id']}: functional row missing fn_name")
            if not meta.get("starter_code"):
                errors.append(f"{row['id']}: functional row missing starter code")
        if meta.get("testtype") == "stdin" and meta.get("has_functional_tests"):
            errors.append(f"{row['id']}: stdin row marked has_functional_tests")
        if meta.get("testtype") == "mixed" and not meta.get("has_functional_tests"):
            errors.append(f"{row['id']}: mixed row without functional tests")

    bfcl_spec = CFG.ACC_SOURCES["general_agent"]
    for row in by_scenario["general_agent"]:
        meta = row["meta"]
        possible = meta.get("possible_answers")
        if meta.get("category") not in bfcl_spec["categories"]:
            errors.append(f"{row['id']}: bad bfcl category {meta.get('category')}")
        if meta.get("category_type") != bfcl_spec["category_type"]:
            errors.append(f"{row['id']}: bad bfcl category_type")
        if not isinstance(possible, list):
            errors.append(f"{row['id']}: possible_answers is not a list")
            continue
        if meta.get("category") == "irrelevance":
            if possible:
                errors.append(f"{row['id']}: irrelevance possible_answers must be empty")
            if row["answer"] != []:
                errors.append(f"{row['id']}: irrelevance answer must be []")
        elif not possible:
            errors.append(f"{row['id']}: empty possible_answers")
        if row["answer"] != possible:
            errors.append(f"{row['id']}: answer != possible_answers")

    for row in by_scenario["search_agent"]:
        meta = row["meta"]
        if not isinstance(meta.get("answers"), list) or not meta["answers"]:
            errors.append(f"{row['id']}: empty search answers")
        if len(meta.get("paragraphs") or []) != 10:
            errors.append(f"{row['id']}: expected 10 paragraphs")
        if len(meta.get("supporting_facts") or []) != 2:
            errors.append(f"{row['id']}: expected 2 supporting facts")
        if not row["tools"] or len(row["tools"]) != 2:
            errors.append(f"{row['id']}: search tools")
        if not (meta.get("corpus_chars") or 0) > 0 or not (meta.get("corpus_tokens") or 0) > 0:
            errors.append(f"{row['id']}: empty retrieval corpus accounting")

    for row in by_scenario["multilingual"]:
        meta = row["meta"]
        if len(row["choices"] or []) != 4:
            errors.append(f"{row['id']}: gmmu choices")
        if not (isinstance(row["answer"], str) and row["answer"] in "ABCD"):
            errors.append(f"{row['id']}: gmmu answer")
        if meta.get("is_en_reference") is not (meta.get("lang") == "en"):
            errors.append(f"{row['id']}: is_en_reference mismatch")
        if meta.get("lang") not in CFG.ACC_SOURCES["multilingual"]["languages"]:
            errors.append(f"{row['id']}: unexpected language {meta.get('lang')}")
    return errors


def assert_rows(
    rows: List[Dict[str, Any]],
    only: Optional[Iterable[str]],
    state: Dict[str, Any],
) -> None:
    """Fail the build (before writing artifacts) when any invariant is violated."""
    errors: List[str] = []
    only_set = set(only) if only else None
    expected_rows = CFG.EXPECTED["acc_rows"] if only_set is None else None
    if expected_rows is not None and len(rows) != expected_rows:
        errors.append(f"row count {len(rows)} != {expected_rows}")
    if len({row["id"] for row in rows}) != len(rows):
        errors.append("duplicate row ids")

    by_scenario: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scenario[row["scenario"]].append(row)

    for scenario, spec in CFG.ACC_SOURCES.items():
        if only_set is not None and scenario not in only_set:
            continue
        count = len(by_scenario.get(scenario, []))
        if count != spec["n"]:
            errors.append(f"{scenario}: {count} rows != {spec['n']}")

    seen_upstream: Dict[str, str] = {}
    for row in rows:
        if set(row) != ROW_FIELDS:
            errors.append(f"{row['id']}: unexpected field set")
        if not isinstance(row["tokens"], int) or row["tokens"] <= 0:
            errors.append(f"{row['id']}: invalid tokens {row['tokens']}")
        blob = json.dumps(row["messages"], ensure_ascii=False)
        if "$LETTER" in blob:
            errors.append(f"{row['id']}: literal $LETTER placeholder")
        if "{context}" in blob or "{input}" in blob:
            errors.append(f"{row['id']}: unrendered template placeholder")
        upstream_id = row["meta"].get("upstream_id")
        if not upstream_id:
            errors.append(f"{row['id']}: missing meta.upstream_id")
        elif upstream_id in seen_upstream:
            errors.append(f"{row['id']}: duplicate upstream_id {upstream_id}")
        else:
            seen_upstream[upstream_id] = row["id"]
        if row["tools"]:
            bad = C.invalid_schema_types(row["tools"])
            if bad:
                errors.append(f"{row['id']}: invalid tool schema types {bad[:3]}")
        if row["scenario"] not in CFG.ACC_SOURCES:
            errors.append(f"{row['id']}: unknown scenario")

    for scenario, scenario_rows in by_scenario.items():
        signatures = Counter(prompt_signature(row) for row in scenario_rows)
        duplicates = sum(value - 1 for value in signatures.values() if value > 1)
        if duplicates:
            errors.append(f"{scenario}: {duplicates} duplicate prompt(s)")

    # ---- scenario-specific multi-row invariants --------------------------- #
    if only_set is None or "long_context" in only_set:
        band_targets = CFG.ACC_SOURCES["long_context"]["length_bands"]
        band_counts = Counter()
        task_counts = Counter()
        for row in by_scenario.get("long_context", []):
            meta = row["meta"]
            band = meta.get("length_band")
            if band in band_targets:
                band_counts[band] += 1
            task_counts[meta.get("task")] += 1
        if dict(band_counts) != dict(band_targets):
            errors.append(f"long_context band counts {dict(band_counts)} != {band_targets}")
        if set(task_counts) != set(CFG.ACC_SOURCES["long_context"]["tasks"]):
            errors.append(f"long_context tasks {sorted(task_counts)}")
        elif any(count == 0 for count in task_counts.values()):
            errors.append(f"long_context task with no rows {dict(task_counts)}")

    if only_set is None or "multilingual" in only_set:
        multilingual_spec = CFG.ACC_SOURCES["multilingual"]
        languages = list(multilingual_spec["languages"])
        per_language = multilingual_spec["per_language"]
        per_lang = Counter(row["meta"].get("lang") for row in by_scenario.get("multilingual", []))
        if set(per_lang) != set(languages):
            errors.append(f"multilingual languages {sorted(per_lang)}")
        elif any(count != per_language for count in per_lang.values()):
            errors.append(f"multilingual per-language counts {dict(per_lang)} != {per_language}")
        paired_ids: Dict[str, set] = defaultdict(set)
        subject_by_id: Dict[str, set] = defaultdict(set)
        for row in by_scenario.get("multilingual", []):
            meta = row["meta"]
            paired_ids[meta.get("lang")].add(meta.get("sample_id"))
            subject_by_id[meta.get("sample_id")].add(
                (meta.get("subject"), meta.get("subject_category"))
            )
        id_sets = {frozenset(ids) for ids in paired_ids.values()}
        if len(id_sets) > 1:
            errors.append("multilingual paired sample_id sets differ across languages")
        if any(len(ids) != per_language for ids in paired_ids.values()):
            errors.append("multilingual paired sample_id count != per_language")
        inconsistent = [sid for sid, values in subject_by_id.items() if len(values) != 1]
        if inconsistent:
            errors.append(
                f"multilingual subject/subject_category differs across languages for "
                f"{len(inconsistent)} sample_ids"
            )

    if only_set is None or "general_agent" in only_set:
        bfcl_spec = CFG.ACC_SOURCES["general_agent"]
        categories = Counter(
            row["meta"].get("category") for row in by_scenario.get("general_agent", [])
        )
        if categories != Counter(bfcl_spec["category_counts"]):
            errors.append(f"general_agent categories {dict(categories)} != {bfcl_spec['category_counts']}")
        questions = Counter()
        for row in by_scenario.get("general_agent", []):
            questions[norm_key(row["messages"][-1]["content"])] += 1
        duplicate_questions = sum(value - 1 for value in questions.values() if value > 1)
        if duplicate_questions:
            errors.append(f"general_agent: {duplicate_questions} duplicate user question(s)")

    if state.get("cross_scenario_duplicate_questions"):
        errors.append(
            f"cross-scenario duplicates: {state['cross_scenario_duplicate_questions']} "
            "search_agent question(s) already used by long_context/hotpotqa"
        )

    errors.extend(_judge_field_errors(rows))
    if errors:
        raise SystemExit("build assertions failed (no artifact written):\n- " + "\n- ".join(errors[:40]))


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #
def build_provenance() -> Dict[str, Any]:
    """Accuracy provenance only (8 upstream datasets actually consumed here)."""
    out: Dict[str, Any] = {}
    for entry in CFG.provenance_for("accuracy"):
        files = [(C.RAW / relative, logical) for relative, logical in entry["files"]]
        out[entry["dataset"]] = C.provenance_entry(entry["source_url"], entry["license"], files)
    return out


def scenario_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    for row in rows:
        scenario = row["scenario"]
        entry = stats.setdefault(
            scenario,
            {"n": 0, "tokens": [], "chars": [], "datasets": set(), "judges": set(), "langs": set()},
        )
        entry["n"] += 1
        entry["tokens"].append(row["tokens"])
        entry["chars"].append(len(row["messages"][0]["content"]))
        entry["datasets"].add(row["dataset"])
        entry["judges"].add(row["judge"])
        lang = (row.get("meta") or {}).get("lang")
        if lang:
            entry["langs"].add(lang)
    for entry in stats.values():
        values = sorted(entry.pop("tokens"))
        chars = sorted(entry.pop("chars"))
        entry.update(
            tokens_min=values[0],
            tokens_median=values[len(values) // 2],
            tokens_max=values[-1],
            tokens_mean=round(sum(values) / len(values), 1),
            prompt_chars_min=chars[0],
            prompt_chars_median=chars[len(chars) // 2],
            prompt_chars_max=chars[-1],
            datasets=sorted(entry.pop("datasets")),
            judges=sorted(entry.pop("judges")),
            langs=sorted(entry.pop("langs")),
        )
    return stats


def _counts(rows: List[Dict[str, Any]], getter: Callable[[Dict[str, Any]], Any]) -> Dict[str, int]:
    counter = Counter()
    for row in rows:
        value = getter(row)
        if isinstance(value, list):
            value = "|".join(str(item) for item in value)
        counter[str(value)] += 1
    return dict(sorted(counter.items()))


def strata_counts(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-scenario counts for the sampling strata named by the design."""
    by_scenario: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scenario[row["scenario"]].append(row)
    meta = lambda row: row.get("meta") or {}  # noqa: E731
    out: Dict[str, Any] = {
        "knowledge_qa": {"category": _counts(by_scenario["knowledge_qa"], lambda r: meta(r).get("category"))},
        "instruction_following": {
            "instruction_id": _counts(
                by_scenario["instruction_following"],
                lambda r: (meta(r).get("instruction_id_list") or ["unknown"])[0],
            )
        },
        "long_context": {
            "task": _counts(by_scenario["long_context"], lambda r: meta(r).get("task")),
            "length_band": _counts(by_scenario["long_context"], lambda r: meta(r).get("length_band")),
            "lang": _counts(by_scenario["long_context"], lambda r: meta(r).get("lang")),
        },
        "math_logic": {
            "subject": _counts(by_scenario["math_logic"], lambda r: meta(r).get("subject")),
            "level": _counts(by_scenario["math_logic"], lambda r: meta(r).get("level")),
        },
        "coding": {
            "platform": _counts(by_scenario["coding"], lambda r: meta(r).get("platform")),
            "testtype": _counts(by_scenario["coding"], lambda r: meta(r).get("testtype")),
            "difficulty": _counts(by_scenario["coding"], lambda r: meta(r).get("difficulty")),
        },
        "general_agent": {"category": _counts(by_scenario["general_agent"], lambda r: meta(r).get("category"))},
        "search_agent": {
            "type": _counts(by_scenario["search_agent"], lambda r: meta(r).get("type")),
            "level": _counts(by_scenario["search_agent"], lambda r: meta(r).get("level")),
        },
        "multilingual": {
            "lang": _counts(by_scenario["multilingual"], lambda r: meta(r).get("lang")),
            "subject_category": _counts(by_scenario["multilingual"], lambda r: meta(r).get("subject_category")),
        },
    }
    return out


DEVIATIONS: List[str] = [
    "MMLU-Pro uses a 0-shot chain-of-thought prompt with the choices re-lettered A-J; "
    "the official 5-shot configuration is not used.",
    "LongBench v1 long-context rows are filtered to a 4K-32K reference-token window "
    f"({CFG.ACC_MIN_TOKENS} <= tokens <= {CFG.ACC_MAX_TOKENS}; no truncation) and are "
    "stratified by prompt length: 4k-8k 100 / 8k-16k 99 / 16k-32k 51. The whole 16K-32K "
    "pool is taken (narrativeqa 39 / qasper 3 / hotpotqa 9), so per-task quotas are no "
    "longer equal (the previous 63/63/62/62 split is retired).",
    "LongBench v1 rows are deduplicated by (task, input, context); upstream duplicate rows "
    "with different gold answers are merged and their answers unioned (narrativeqa: 18 rows merged).",
    "HotpotQA accuracy rows come from the distractor validation split, which is hard-only "
    "(level=hard for all 7,405 rows); they are converted into a single-request retrieval-agent "
    "task: the 10 distractor paragraphs live in meta and only the fixed search/lookup tool "
    "schemas are sent. The retrieval tools are not executed by the performance runner.",
    "Global-MMLU is scored generatively from option letters, not with lm-eval's "
    "likelihood-based scoring.",
    "Global-MMLU covers all 42 locally available languages x 100 rows, using a paired design: "
    "the same 100 sample_ids (chosen by subject round-robin from the 14,042 common ids) appear "
    "in every language, so cross-language differences compare translations of the same question.",
    "LiveCodeBench mixes two prompt types: function-level prompts for rows with starter code "
    "(LeetCode, functional hidden tests) and stdin/stdout prompts otherwise (AtCoder). "
    "code_pass1 must therefore support both testtype=functional and testtype=stdin rows; the "
    "window is release v5 + v6 (2024-09-22..2025-04-06).",
    "BFCL covers all 11 v4 single-turn categories (simple_python / simple_java / "
    "simple_javascript / multiple / parallel / parallel_multiple / irrelevance / live_simple / "
    "live_multiple / live_parallel / live_parallel_multiple); multi-turn categories are excluded. "
    "Category counts are 100/100/50/100/100/100/100/100/100/15/24 = 889 (pools smaller than the "
    "target contribute their whole pool; live_parallel 15, live_parallel_multiple 24). "
    "irrelevance has no upstream possible_answer file, so its gold is the empty answer list "
    "(meta.possible_answers = []) and rows are deduplicated by user question across all categories.",
    "search_agent excludes questions already selected by long_context/hotpotqa "
    "(cross-scenario duplication removed).",
    "Hidden LiveCodeBench tests are retained in full per row, except individual tests whose "
    "JSON exceeds the declared 256 KiB local-executor cap (meta.tests_dropped_large records them).",
    "Accuracy is data-only in this build: the set is not wired into the run "
    "(docs/TEST_PLAN.md is performance-only) and no evaluator/runner is implemented.",
]


def build_manifest(rows: List[Dict[str, Any]], state: Dict[str, Any], tok: C.Tokenizer) -> Dict[str, Any]:
    judges = {
        name: {**spec, "implementation_status": JUDGE_IMPLEMENTATION_STATUS}
        for name, spec in CFG.JUDGE_SPECS.items()
    }
    sampling = dict(state.get("sampling") or {})
    per_scenario = {name: spec["n"] for name, spec in CFG.ACC_SOURCES.items()}
    sampling.update(
        {
            "seed": CFG.SEED,
            "per_scenario": dict(per_scenario),
            "rules": {scenario: spec["strata"] for scenario, spec in CFG.ACC_SOURCES.items()},
            "upstream_id_format": dict(UPSTREAM_ID_FORMAT),
            "cross_scenario": {
                "long_context": "runs first; its selected hotpotqa questions are excluded from search_agent",
                "search_agent": "pool excludes long_context/hotpotqa questions",
                "duplicate_questions_after_build": state.get("cross_scenario_duplicate_questions", 0),
            },
        }
    )
    return {
        "schema_version": 3,
        "generator_version": GENERATOR_VERSION,
        "status": "offline-asset",
        "runner": "not wired（当前 docs/TEST_PLAN 性能-only）",
        "seed": CFG.SEED,
        "length_processing": "none",
        "construction": "none",
        "osl": CFG.OSL,
        "tokens_semantics": (
            "tokens = chat-template messages + tools JSON text; long-context rows use the rendered "
            "prompt only, and search_agent retrieval passages (meta.paragraphs) are excluded from tokens"
        ),
        "tokenizer": tok.info(),
        "tokenizer_cache": {"calls": tok.calls, "cache_hits": tok.cache_hits},
        "files": ["testset.jsonl"],
        "judges": judges,
        "scenarios": {
            scenario: {
                "dataset": spec["dataset"],
                "n": spec["n"],
                "judge": spec["judge"],
                "metric": spec["metric"],
                "strata": spec["strata"],
                "prompt": spec.get("prompt"),
                "tasks": spec.get("tasks"),
                "length_bands": spec.get("length_bands"),
                "min_tokens": spec.get("min_tokens"),
                "max_tokens": spec.get("max_tokens"),
                "categories": spec.get("categories"),
                "category_counts": spec.get("category_counts"),
                "category_type": spec.get("category_type"),
                "languages": spec.get("languages"),
                "per_language": spec.get("per_language"),
                "pairing": spec.get("pairing"),
                "corpus": spec.get("corpus"),
                "window": spec.get("window"),
                "upstream_id_format": UPSTREAM_ID_FORMAT[scenario],
            }
            for scenario, spec in CFG.ACC_SOURCES.items()
        },
        "sampling": sampling,
        "deviations": DEVIATIONS,
        "provenance": build_provenance(),
        "strata_counts": strata_counts(rows),
        "stats": scenario_stats(rows),
        "totals": {
            "rows": len(rows),
            "per_scenario": dict(per_scenario),
            "scenarios": len(CFG.ACC_SOURCES),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", choices=list(BUILDERS))
    args = parser.parse_args()
    C.ensure_dirs()
    tok = C.Tokenizer()
    try:
        rows: List[Dict[str, Any]] = []
        state: Dict[str, Any] = {"sampling": {}}
        for scenario, builder in BUILDERS.items():
            if args.only and scenario not in args.only:
                continue
            part = builder(tok, CFG.ACC_SOURCES[scenario]["n"], state)
            rows += part
            print(f"[acc] {scenario}: {len(part)}")

        rows.sort(key=lambda row: row["id"])
        # Assertions run before writing: a failed build leaves both artifacts untouched.
        assert_rows(rows, args.only, state)
        C.write_jsonl(C.TEST_ACC / "testset.jsonl", rows)
        manifest = build_manifest(rows, state, tok)
        C.write_json(C.TEST_ACC / "manifest.json", manifest)
    finally:
        tok.close()
    print(f"[acc] rows={len(rows)} scenarios={len(CFG.ACC_SOURCES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
