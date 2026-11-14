"""Build ``dataset/test/performance`` — 8 scenarios x 1K/8K/32K x 10 targets.

The set measures performance only (TTFT / TG TPS / peak memory), never accuracy.
Design contract (see ``docs/spec/11-performance-testset.md``):

* every scenario contributes **10 targets**; the same 10 targets are rendered in the
  three buckets, so a bucket difference isolates context length only;
* a target owns an **ordered unit stream** (documents / worked examples / tool schemas /
  multiple-choice demonstrations) plus a **fixed task block** appended at the end.  A
  bucket row is ``prefix(unit-stream, budget) + SEP + task``, so the three rows of one
  target are strictly nested (1K subset 8K subset 32K);
* ``tokens`` on a row is the **prompt token target** (1024 / 8192 / 32768); the build
  reaches it exactly with one reference tokenizer (default: the generic ``cl100k_base``,
  never a model under test).  ``--reserve`` accounts for the chat-template overhead of
  the tokenizer used for the build, so the *content* is ``bucket - reserve`` tokens;
* production runs do not use this reference build directly: ``align.py`` rebuilds the
  same 240 rows with the target model tokenizer and verifies ``prompt_tokens == bucket``;
* sampling is deterministic: ``sha256(seed|id)`` ordering + stratified largest remainder.

Usage::

    python tool/dataset/performance/build.py                  # reference build (240 rows)
    python tool/dataset/performance/build.py --scenarios knowledge_qa --buckets 1k
    python tool/dataset/performance/build.py --dry-run        # render, print stats, write nothing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from tool.dataset.performance.sampling import allocate, deterministic_pick  # noqa: E402
from tool.dataset.performance.sources import SCENARIO_SOURCE  # noqa: E402

RAW_DIR = PROJECT_ROOT / "dataset" / "raw"
OUT_DIR = PROJECT_ROOT / "dataset" / "test" / "performance"
#: generic reference tokenizer used for the committed build; never a model under test
REFERENCE_TOKENIZER_REL = "tokenizer/cl100k_base/tokenizer.json"
#: chat-template token overhead reserved by the build tokenizer (set from --reserve);
#: production alignment measures it for the real model instead
CONTENT_RESERVE = 0

BUCKETS: Dict[str, int] = {"1k": 1024, "8k": 8192, "32k": 32768}


def content_goal(bucket: str) -> int:
    """Content tokens a row must carry: prompt target minus template reserve."""
    return BUCKETS[bucket] - CONTENT_RESERVE
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
TARGETS_PER_SCENARIO = 10
SEED = 42
#: Output window for TG TPS: every sample decodes at most this many tokens.
#: Product default 50; a test set can declare another value and CLI/Web can
#: override it per run, but the measured TG TPS is always a fixed-window figure.
OSL = 50
#: Completion tokens a sample is expected to reach before the window closes.
MIN_COMPLETION_TOKENS = 50

#: each unit stream must exceed 32K + task block, so the 32K row is a real prefix
STREAM_MIN_TOKENS = 36_000
#: a target's task block must leave >=600 content tokens in the 1K bucket
TASK_MAX_TOKENS = 400
SEP = "\n\n"

#: join candidates tried in order; each costs a different number of tokens at the join, which
#: is what closes the 1-2 token gaps a tokenizer boundary effect can leave open
SEPARATOR_CANDIDATES = (SEP, "\n", " ", "", "\n\n\n", "\n ", "\n\n ", "\n \n", " \n\n")
#: last-resort joins: whitespace pads of a known token cost inserted between the body prefix
#: and the task block (the body stays a pure stream prefix so the nesting check still holds)
PAD_CANDIDATES = ("", " ", "\n", "\n ", "  ", "\n\n ", "\n\n\n", "* * *")

#: scenario -> language tag used by the perf_jsonl index / docs
SCENARIO_LANG: Dict[str, str] = {
    "knowledge_qa": "zh",
    "instruction_following": "zh",
    "long_context": "en",
    "math_logic": "en",
    "coding": "en",
    "general_agent": "en",
    "search_agent": "en",
    "multilingual": "multi",
}

#: scenario -> stable ``dataset`` label on every row (one label per scenario keeps the
#: plan at 24 cells; the precise upstream unit lives in ``meta.source`` / ``upstream_id``)
SCENARIO_DATASET: Dict[str, str] = {
    "knowledge_qa": "longbench_pro",
    "instruction_following": "longbench_v1+multi_if",
    "long_context": "longbench_pro",
    "math_logic": "aime_2026+math_500",
    "coding": "longbench_v2:code_repository",
    "general_agent": "bfcl_v4",
    "search_agent": "longbench_pro",
    "multilingual": "global_mmlu",
}

MULTILINGUAL_LANGS: Tuple[str, ...] = ("zh", "en", "fr", "de", "es", "pt", "ru", "ja", "ko", "ar")

#: per-language instruction tail (the payload itself is already in that language)
ML_TASK: Dict[str, str] = {
    "zh": "请选择正确选项，并用不少于 80 字说明理由，最后以「答案：X」结束。",
    "en": "Answer with the correct option, explain in at least 80 words, then finish with 'Answer: X'.",
    "fr": "Choisissez la bonne option, justifiez en au moins 80 mots, puis terminez par « Réponse : X ».",
    "de": "Wähle die richtige Option, begründe in mindestens 80 Wörtern und ende mit „Antwort: X“.",
    "es": "Elige la opción correcta, explica en al menos 80 palabras y termina con «Respuesta: X».",
    "pt": "Escolha a opção correta, explique em pelo menos 80 palavras e finalize com «Resposta: X».",
    "ru": "Выберите правильный вариант, объясните не менее чем в 80 словах и завершите «Ответ: X».",
    "ja": "正しい選択肢を選び、80 語以上で理由を説明し、最後に「答え：X」で終えてください。",
    "ko": "올바른 선택지를 고르고 80단어 이상으로 이유를 설명한 뒤 '답: X'로 끝내세요.",
    "ar": "اختر الخيار الصحيح، واشرح في 80 كلمة على الأقل، ثم اختم بـ«الإجابة: X».",
}


# --------------------------------------------------------------------------- #
# tokenizer wrapper
# --------------------------------------------------------------------------- #
class Counter:
    """Token counter for the *reference* tokenizer (content only, no specials)."""

    def __init__(self, path: Path) -> None:
        from tokenizers import Tokenizer  # noqa: PLC0415 - optional dependency

        self.path = path
        self._tok = Tokenizer.from_file(str(path))
        self._cache: Dict[int, int] = {}

    def ids(self, text: str) -> List[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: Sequence[int]) -> str:
        return self._tok.decode(list(ids))

    def n(self, text: str) -> int:
        key = hash(text)
        value = self._cache.get(key)
        if value is None:
            value = len(self.ids(text))
            self._cache[key] = value
        return value


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# raw loaders
# --------------------------------------------------------------------------- #
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_json_rows(path: Path) -> List[Dict[str, Any]]:
    """Read a JSON array, or JSONL when the upstream file is line-delimited."""
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        data = json.loads(text)
        return data if isinstance(data, list) else list(data.values())
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def iter_json_array(path: Path, chunk_size: int = 8 << 20):
    """Yield elements of a large top-level JSON array without loading it fully."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as fh:
        buffer = ""
        started = False
        while True:
            chunk = fh.read(chunk_size)
            if chunk:
                buffer += chunk
            if not started:
                buffer = buffer.lstrip()
                if buffer:
                    if buffer[0] != "[":
                        raise ValueError(f"{path}: not a JSON array")
                    buffer = buffer[1:]
                    started = True
            while True:
                probe = buffer.lstrip()
                if probe.startswith(","):
                    probe = probe[1:].lstrip()
                if probe.startswith("]"):
                    return
                try:
                    value, end = decoder.raw_decode(probe)
                except json.JSONDecodeError:
                    break
                buffer = probe[end:]
                yield value
            if not chunk:
                return


def load_parquet(path: Path) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq  # noqa: PLC0415

    table = pq.read_table(path)
    return table.to_pylist()


# --------------------------------------------------------------------------- #
# length alignment
# --------------------------------------------------------------------------- #
def fit_prefix(counter: Counter, units: Sequence[str], budget: int) -> Tuple[str, int]:
    """Token-exact prefix of ``SEP.join(units)`` with exactly ``budget`` tokens."""
    ids = counter.ids(SEP.join(units))
    if len(ids) < budget:
        raise ValueError(f"unit stream too short: {len(ids)} < {budget} tokens")
    n = budget
    for _ in range(12):
        text = counter.decode(ids[:n])
        got = counter.n(text)
        if got == budget:
            return text, got
        n = max(1, min(n + (budget - got), len(ids)))
    return text, counter.n(text)


def make_row(
    counter: Counter,
    scenario: str,
    target: Dict[str, Any],
    bucket: str,
    index: int,
) -> Dict[str, Any]:
    goal = content_goal(bucket)
    task = target["task"]
    task_tokens = counter.n(task)
    if goal - task_tokens <= 0:
        raise ValueError(f"{scenario}: task block longer than bucket {bucket}")
    stream_ids = counter.ids(SEP.join(target["units"]))
    last_n = max(1, goal - task_tokens)

    def attempt(sep: str) -> Tuple[str, int]:
        """Return (body, rendered token count) for the best cut under ``sep``."""
        nonlocal last_n
        n = goal - task_tokens
        body = counter.decode(stream_ids[:n]).rstrip("\ufffd")
        total = counter.n(body + sep + task)
        for _ in range(40):
            if total == goal:
                last_n = n
                return body, total
            n = max(1, min(n + (goal - total), len(stream_ids)))
            body = counter.decode(stream_ids[:n]).rstrip("\ufffd")
            total = counter.n(body + sep + task)
        last_n = n
        return body, total

    body, sep, total = "", SEP, -1
    # Different separators consume a different number of tokens at the join, which is what
    # makes an exact alignment possible when the tokenizer's boundary effect skips a value.
    for candidate in SEPARATOR_CANDIDATES:
        body, total = attempt(candidate)
        if total == goal:
            sep = candidate
            break
    if total != goal:
        # Last resort: a boundary effect can leave a 1-2 token gap that no single separator
        # closes.  Vary the body cut by +-2 stream tokens and append a whitespace pad, which
        # shifts the join by a measured number of tokens.  The pad is part of the join, not of
        # the body, so same-target rows keep the strict prefix relationship.
        body, sep, total = "", SEP, -1
        for n in (last_n - 2, last_n - 1, last_n, last_n + 1, last_n + 2):
            if n < 1 or n > len(stream_ids):
                continue
            prefix = counter.decode(stream_ids[:n]).rstrip("\ufffd")
            for pad in PAD_CANDIDATES:
                for candidate in SEPARATOR_CANDIDATES:
                    total = counter.n(prefix + candidate + pad + task)
                    if total == goal:
                        body, sep = prefix, candidate + pad
                        break
                if total == goal:
                    break
            if total == goal:
                break
    if total != goal:
        raise ValueError(f"{scenario}/{bucket}/{target['target_id']}: cannot align to {goal} ({total})")

    content = body + sep + task
    body_tokens = counter.n(body)
    meta = {
        "lang": SCENARIO_LANG[scenario],
        "source": target["source"],
        "upstream_id": target["upstream_id"],
        "target_tokens": BUCKETS[bucket],
        "content_tokens": total,
        "template_reserve_tokens": CONTENT_RESERVE,
        "task_tokens": counter.n(task),
        "stream_tokens": target["stream_tokens"],
        "stream_units": len(target["units"]),
        "alignment": "prefix+token-cut",
        "task_sha256": sha256_text(task)[:16],
        "body_sha256": sha256_text(body)[:16],
    }
    meta.update(target.get("meta") or {})
    row = {
        "id": f"perf-{scenario}-{bucket}-{index:03d}",
        "scenario": scenario,
        "bucket": bucket,
        "target_id": target["target_id"],
        "dataset": SCENARIO_DATASET[scenario],
        "tokens": BUCKETS[bucket],
        "messages_tokens": BUCKETS[bucket],
        "tools_tokens": 0,
        "messages": [{"role": "user", "content": content}],
        "tools": [],
        "constructed": True,
        "meta": meta,
    }
    return row, body


# --------------------------------------------------------------------------- #
# helpers shared by scenario builders
# --------------------------------------------------------------------------- #
def pick_stratified(strata: Dict[str, List[str]], k: int, min_each: int = 1) -> List[str]:
    pools = {name: sorted(ids) for name, ids in strata.items() if ids}
    quota = allocate({name: len(ids) for name, ids in pools.items()}, k, min_each=min_each, seed=SEED)
    picked: List[str] = []
    for name in sorted(pools):
        picked.extend(deterministic_pick(pools[name], quota[name], SEED))
    return sorted(picked, key=lambda i: sha256_text(f"{SEED}|{i}"))


def extend_stream(
    counter: Counter,
    units: List[str],
    pool: Iterable[Tuple[str, str]],
    min_tokens: int = STREAM_MIN_TOKENS,
) -> int:
    """Append pool units (id, text) in deterministic order until the stream is long enough."""
    total = counter.n(SEP.join(units))
    for _id, text in pool:
        if total >= min_tokens:
            break
        units.append(text)
        total += counter.n(text) + 2
    if total < min_tokens:
        raise ValueError(f"stream too short after exhausting pool: {total} < {min_tokens}")
    return total


def choice_block(row: Dict[str, Any]) -> str:
    return "\n".join(
        f"{letter}. {row[f'choice_{letter}']}" for letter in "ABCD" if row.get(f"choice_{letter}")
    )


# --------------------------------------------------------------------------- #
# scenario 1-4: native long document prefix (doc-first, task-last)
# --------------------------------------------------------------------------- #
DOC_TASKS: Dict[str, str] = {
    "knowledge_qa": (
        "请仅根据以上材料回答问题：{question}\n\n"
        "要求：先给出明确答案，再用不少于 100 字说明依据（引用材料中的原文片段）。"
    ),
    "long_context": (
        "Question: {question}\n\nChoices:\n{choices}\n\n"
        "Answer using only the long structured data above. Explain your reasoning in at "
        "least 80 words, then finish with 'Answer: <letter>'."
    ),
    "coding": (
        "The code repository above is the full context.\n\nQuestion: {question}\n\nChoices:\n{choices}\n\n"
        "Answer with the file/function evidence you used, in at least 80 words, then finish with "
        "'Answer: <letter>'."
    ),
    "search_agent": (
        "The blocks above are the search results retrieved for the question "
        "(one block per retrieved document).\n\nQuestion: {question}\n\nChoices:\n{choices}\n\n"
        "Answer citing the result blocks you used, in at least 80 words, then finish with "
        "'Answer: <letter>'."
    ),
}


def build_doc_scenario(
    counter: Counter,
    scenario: str,
    docs: List[Dict[str, Any]],
    doc_text,
    question,
    source: str,
    strata_key,
) -> List[Dict[str, Any]]:
    """Pick targets from a long-document corpus (question at the end of the prompt).

    Candidates whose question block would eat the 1K bucket are dropped first, so
    every kept target has at least ``1024 - TASK_MAX_TOKENS`` tokens of context even
    in the shortest bucket.
    """
    by_id = {str(doc["_id"]): doc for doc in docs}
    tasks: Dict[str, str] = {}
    strata: Dict[str, List[str]] = {}
    for doc in docs:
        task = DOC_TASKS[scenario].format(
            question=str(question(doc)).strip(), choices=choice_block(doc)
        )
        if counter.n(task) > TASK_MAX_TOKENS:
            continue
        doc_id = str(doc["_id"])
        tasks[doc_id] = task
        strata.setdefault(str(strata_key(doc)), []).append(doc_id)
    picked = pick_stratified(strata, TARGETS_PER_SCENARIO)

    targets: List[Dict[str, Any]] = []
    for index, doc_id in enumerate(picked):
        doc = by_id[doc_id]
        units = [doc_text(doc)]
        order = sorted(by_id, key=lambda i: sha256_text(f"{SEED}|ctx|{i}"))
        same = [
            (other, doc_text(by_id[other]))
            for other in order
            if other != doc_id and str(strata_key(by_id[other])) == str(strata_key(doc))
        ]
        rest = [(other, doc_text(by_id[other])) for other in order if other != doc_id]
        # same-stratum documents first, then every other document of the same corpus
        total = extend_stream(counter, units, same + rest)
        targets.append(
            {
                "target_id": f"tgt-{scenario}-{index:03d}",
                "units": units,
                "stream_tokens": total,
                "source": source,
                "upstream_id": doc_id,
                "task": tasks[doc_id],
                "meta": {"sub_domain": str(strata_key(doc))},
            }
        )
    return targets


# --------------------------------------------------------------------------- #
# scenario 5: math & logic — AIME 2026 problem, MATH-500 worked examples
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# LongBench-Pro (2026): native 8K/32K/64K bilingual long documents
# --------------------------------------------------------------------------- #
LBP_TASKS: Dict[str, str] = {
    "knowledge_qa": (
        "请仅根据以上材料回答问题：\n\n{q}\n\n"
        "要求：先给出明确答案，再用不少于 100 字说明依据（引用材料中的原文片段）。"
    ),
    "long_context": (
        "Question: {q}\n\nAnswer using only the material above. Explain your reasoning in at "
        "least 80 words, then state the final answer on its own line."
    ),
    "search_agent": (
        "You are given the retrieved documents above. Query: {q}\n\n"
        "Locate the supporting evidence, cite the document fragments you used, and answer in at "
        "least 80 words; finish with 'Answer: ...'."
    ),
}

LBP_PRIMARY_TASKS: Dict[str, Tuple[str, ...]] = {
    "knowledge_qa": ("T3. Evidence-Grounded QA",),
    "long_context": ("T4. Summarization & Synthesis",),
    "search_agent": ("T1. Retrieval & Ranking", "T5. Attribution & Citation Alignment"),
}

LBP_LANGUAGE: Dict[str, str] = {"knowledge_qa": "Chinese", "long_context": "English", "search_agent": "English"}

#: native length buckets kept as candidate targets (all three exceed the 32K bucket, so the
#: 32K row is a real prefix of a natively long document; the pool must stay >= 10 rows)
LBP_NATIVE_LENGTHS: Tuple[str, ...] = ("32k", "64k", "128k")


def build_lbp_scenario(counter: Counter, scenario: str) -> List[Dict[str, Any]]:
    """Targets from LongBench-Pro: one native long document per target, question at the end."""
    path = RAW_DIR / "longbench_pro" / "longbench_pro.json"
    language = LBP_LANGUAGE[scenario]
    tasks = LBP_PRIMARY_TASKS[scenario]
    rows: List[Dict[str, Any]] = []
    tasks_by_id: Dict[str, str] = {}
    strata: Dict[str, List[str]] = {}
    for row in iter_json_array(path):
        if row.get("language") != language or row.get("primary_task") not in tasks:
            continue
        if str(row.get("token_length")) not in LBP_NATIVE_LENGTHS:
            continue
        task = LBP_TASKS[scenario].format(q=str(row["question_nonthinking"]).strip())
        if counter.n(task) > TASK_MAX_TOKENS:
            continue
        doc_id = str(row["id"])
        tasks_by_id[doc_id] = task
        strata.setdefault(str(row.get("secondary_task")), []).append(doc_id)
        rows.append(row)
    by_id = {str(row["id"]): row for row in rows}
    picked = pick_stratified(strata, TARGETS_PER_SCENARIO)

    targets: List[Dict[str, Any]] = []
    for index, doc_id in enumerate(picked):
        row = by_id[doc_id]
        units = [str(row["context"]).strip()]
        order = sorted(by_id, key=lambda i: sha256_text(f"{SEED}|lbp|{i}"))
        same = [
            (other, str(by_id[other]["context"]).strip())
            for other in order
            if other != doc_id and str(by_id[other].get("secondary_task")) == str(row.get("secondary_task"))
        ]
        rest = [(other, str(by_id[other]["context"]).strip()) for other in order if other != doc_id]
        total = extend_stream(counter, units, same + rest)
        targets.append(
            {
                "target_id": f"tgt-{scenario}-{index:03d}",
                "dataset": f"longbench_pro:{row.get('primary_task')}",
                "units": units,
                "stream_tokens": total,
                "source": f"longbench_pro:{row.get('primary_task')}",
                "upstream_id": doc_id,
                "task": tasks_by_id[doc_id],
                "meta": {
                    "sub_domain": str(row.get("secondary_task")),
                    "native_token_length": str(row.get("token_length")),
                    "difficulty": str(row.get("difficulty")),
                },
            }
        )
    return targets


def build_math_logic(counter: Counter) -> List[Dict[str, Any]]:
    aime = load_parquet(RAW_DIR / "aime2026" / "data" / "train-00000-of-00001.parquet")
    by_idx = {str(row["problem_idx"]): row for row in aime}
    picked = pick_stratified({"aime": sorted(by_idx)}, TARGETS_PER_SCENARIO, min_each=1)

    math500 = load_jsonl(RAW_DIR / "math_500" / "test.jsonl")
    demos = [
        (str(row["unique_id"]), f"Problem: {row['problem'].strip()}\n\nSolution: {row['solution'].strip()}")
        for row in math500
    ]
    demos.sort(key=lambda item: sha256_text(f"{SEED}|demo|{item[0]}"))

    targets: List[Dict[str, Any]] = []
    for index, problem_idx in enumerate(picked):
        row = by_idx[problem_idx]
        units: List[str] = []
        total = extend_stream(counter, units, demos)
        targets.append(
            {
                "target_id": f"tgt-math_logic-{index:03d}",
                "units": units,
                "stream_tokens": total,
                "source": "aime_2026+math_500",
                "upstream_id": f"aime_2026#{problem_idx}",
                "task": (
                    f"Problem: {row['problem'].strip()}\n\n"
                    "The examples above are worked solutions in the same format. Solve the new problem "
                    "step by step, explain your reasoning in at least 80 words, and finish with "
                    "'Final answer: <value>'."
                ),
                "meta": {"answer_type": "numeric"},
            }
        )
    return targets


# --------------------------------------------------------------------------- #
# scenario 6: general agent — BFCL tool catalog + user request
# --------------------------------------------------------------------------- #
def _bfcl_question(row: Dict[str, Any]) -> str:
    question = row["question"]
    if isinstance(question, str):
        question = json.loads(question)
    turns = question if isinstance(question, list) else [question]
    flat: List[str] = []
    for turn in turns:
        if isinstance(turn, list):
            for item in turn:
                if isinstance(item, dict) and item.get("content"):
                    flat.append(str(item["content"]).strip())
        elif isinstance(turn, dict) and turn.get("content"):
            flat.append(str(turn["content"]).strip())
    return "\n".join(flat)


def _bfcl_tools(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    tools = row.get("function")
    if isinstance(tools, str):
        tools = json.loads(tools)
    return list(tools or [])


def _tool_text(tool: Dict[str, Any]) -> str:
    return "Tool: {name}\n```json\n{schema}\n```".format(
        name=tool.get("name", "unknown"),
        schema=json.dumps(tool, ensure_ascii=False, indent=2),
    )


def build_general_agent(counter: Counter) -> List[Dict[str, Any]]:
    simple = load_json_rows(RAW_DIR / "bfcl" / "BFCL_v4_simple_python.json")
    multiple = load_json_rows(RAW_DIR / "bfcl" / "BFCL_v4_multiple.json")
    pools: Dict[str, List[Dict[str, Any]]] = {"simple_python": simple, "multiple": multiple}
    strata = {name: sorted(str(row["id"]) for row in rows) for name, rows in pools.items()}
    picked = pick_stratified(strata, TARGETS_PER_SCENARIO)
    by_id = {str(row["id"]): row for rows in pools.values() for row in rows}

    # deterministic distractor order over every tool schema in the catalog
    catalog: List[Tuple[str, str]] = []
    seen: set = set()
    for rows in (simple, multiple):
        for row in rows:
            for tool in _bfcl_tools(row):
                name = tool.get("name")
                if name in seen:
                    continue
                seen.add(name)
                catalog.append((f"{row['id']}|{name}", _tool_text(tool)))
    catalog.sort(key=lambda item: sha256_text(f"{SEED}|tool|{item[0]}"))

    targets: List[Dict[str, Any]] = []
    for index, bfcl_id in enumerate(picked):
        row = by_id[bfcl_id]
        own = [_tool_text(tool) for tool in _bfcl_tools(row)]
        own_names = {tool.get("name") for tool in _bfcl_tools(row)}
        units = list(own)
        pool = [(name, text) for name, text in catalog if name.split("|")[-1] not in own_names]
        total = extend_stream(counter, units, pool)
        targets.append(
            {
                "target_id": f"tgt-general_agent-{index:03d}",
                "units": units,
                "stream_tokens": total,
                "source": "bfcl_v4:simple_python+multiple",
                "upstream_id": bfcl_id,
                "task": (
                    f"User request: {_bfcl_question(row)}\n\n"
                    "The tool library above is the MCP server's full catalog (the first tools are the "
                    "most relevant candidates). Reply with the tool name and the complete argument JSON "
                    "you would call, then justify in at least 80 words."
                ),
                "meta": {"tool_count": len(own_names)},
            }
        )
    return targets


# --------------------------------------------------------------------------- #
# scenario 7: instruction following — Chinese meeting transcript + constraint set
# --------------------------------------------------------------------------- #
CONSTRAINT_TEXT = {
    "punctuation:no_comma": lambda kw: "全文不得出现逗号",
    "length_constraints:number_words": lambda kw: (
        f"全文不少于 {kw.get('num_words', 300)} 字"
        if kw.get("relation", "at least") == "at least"
        else f"全文不超过 {kw.get('num_words', 300)} 字"
    ),
    "length_constraints:number_sentences": lambda kw: (
        f"全文不少于 {kw.get('num_sentences', 10)} 句"
        if kw.get("relation", "at least") == "at least"
        else f"全文不超过 {kw.get('num_sentences', 10)} 句"
    ),
    "length_constraints:number_paragraphs": lambda kw: f"全文分为 {kw.get('num_paragraphs', 3)} 段",
    "detectable_format:number_highlighted_sections": lambda kw: (
        f"用 ** 标出 {kw.get('num_highlights', 3)} 个小标题"
    ),
    "detectable_format:number_bullet_lists": lambda kw: (
        f"至少包含 {kw.get('num_bullets', 3)} 条项目符号列表"
    ),
    "detectable_format:title": lambda kw: "以《》包裹的标题开头",
    "detectable_format:json_format": lambda kw: "最终只输出一个合法 JSON 对象",
    "startend:end_checker": lambda kw: f"结尾必须是「{kw.get('end_phrase', '')}」",
    "startend:quotation": lambda kw: "全文用双引号包裹",
    "keywords:existence": lambda kw: "必须包含关键词：" + "、".join(kw.get("keywords") or []),
    "keywords:forbidden_words": lambda kw: "禁止出现词语：" + "、".join(kw.get("forbidden_words") or []),
    "keywords:frequency": lambda kw: (
        f"关键词「{kw.get('keyword', '')}」至少出现 {kw.get('frequency', 2)} 次"
    ),
}


def build_instruction_following(counter: Counter) -> List[Dict[str, Any]]:
    import csv  # noqa: PLC0415

    transcripts = load_jsonl(RAW_DIR / "longbench_v1" / "data" / "vcsum.jsonl")
    by_id = {str(row["_id"]): row for row in transcripts}
    strata: Dict[str, List[str]] = {}
    for row in transcripts:
        strata.setdefault(str(row.get("length") or "unknown"), []).append(str(row["_id"]))
    picked_docs = pick_stratified(strata, TARGETS_PER_SCENARIO)

    with (RAW_DIR / "multi_if" / "multiIF_20241018.csv").open(encoding="utf-8") as fh:
        rows = [row for row in csv.DictReader(fh) if row.get("language") == "Chinese"]
    constraints: Dict[str, Tuple[str, ...]] = {}
    for row in rows:
        ids = json.loads(row["turn_1_instruction_id_list"])
        if not ids or any(i not in CONSTRAINT_TEXT for i in ids):
            continue
        constraints.setdefault(row["key"], tuple(ids))
    picked_cons = pick_stratified({"zh": sorted(constraints)}, TARGETS_PER_SCENARIO, min_each=1)
    kw_by_key = {row["key"]: json.loads(row["turn_1_kwargs"]) for row in rows}

    targets: List[Dict[str, Any]] = []
    for index, (doc_id, con_key) in enumerate(zip(picked_docs, picked_cons)):
        units = [f"【会议记录片段 1】\n{by_id[doc_id]['context'].strip()}"]
        pool = [
            (other, f"【会议记录片段 {n}】\n{by_id[other]['context'].strip()}")
            for n, other in enumerate(
                sorted(by_id, key=lambda i: sha256_text(f"{SEED}|ctx|{i}")), start=2
            )
            if other != doc_id
        ]
        total = extend_stream(counter, units, pool)

        kws = kw_by_key[con_key]
        lines = []
        for position, (instruction_id, kw) in enumerate(zip(constraints[con_key], kws), start=1):
            if isinstance(kw, str):
                kw = json.loads(kw or "{}")
            lines.append(f"{position}. {CONSTRAINT_TEXT[instruction_id](kw)}")
        task = (
            "请阅读以上会议记录，并按下列要求写一段连贯的中文说明（对会议内容做归纳）：\n"
            + "\n".join(lines)
            + "\n\n必须同时满足上述全部要求，不要复述要求本身。"
        )
        targets.append(
            {
                "target_id": f"tgt-instruction_following-{index:03d}",
                "units": units,
                "stream_tokens": total,
                "source": "multi_if_zh+vcsum",
                "upstream_id": f"vcsum#{doc_id}|multi_if#{con_key}",
                "task": task,
                "meta": {"constraint_ids": list(constraints[con_key])},
            }
        )
    return targets


# --------------------------------------------------------------------------- #
# scenario 8: multilingual — Global-MMLU demonstrations, one language per target
# --------------------------------------------------------------------------- #
def build_multilingual(counter: Counter) -> List[Dict[str, Any]]:
    targets: List[Dict[str, Any]] = []
    for index, lang in enumerate(MULTILINGUAL_LANGS):
        rows = load_parquet(RAW_DIR / "global_mmlu" / lang / "test.parquet")
        by_id = {str(row["sample_id"]): row for row in rows}
        picked = deterministic_pick(sorted(by_id), 1, SEED)[0]

        def demo(row: Dict[str, Any]) -> str:
            return (
                f"Question: {row['question'].strip()}\n"
                f"A. {row['option_a']}\nB. {row['option_b']}\nC. {row['option_c']}\nD. {row['option_d']}\n"
                f"Answer: {row['answer']}"
            )

        pool = [
            (other, demo(by_id[other]))
            for other in sorted(by_id, key=lambda i: sha256_text(f"{SEED}|demo|{i}"))
            if other != picked
        ]
        units: List[str] = []
        total = extend_stream(counter, units, pool)
        row = by_id[picked]
        task = (
            f"Question: {row['question'].strip()}\n"
            f"A. {row['option_a']}\nB. {row['option_b']}\nC. {row['option_c']}\nD. {row['option_d']}\n\n"
            + ML_TASK[lang]
        )
        targets.append(
            {
                "target_id": f"tgt-multilingual-{index:03d}",
                "units": units,
                "stream_tokens": total,
                "source": f"global_mmlu:{lang}",
                "upstream_id": f"{lang}#{picked}",
                "task": task,
                "meta": {"lang": lang, "subject": str(row.get("subject"))},
                "lang": lang,
            }
        )
    return targets


# --------------------------------------------------------------------------- #
# build / validate / emit
# --------------------------------------------------------------------------- #
def build_targets(counter: Counter, scenarios: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    lb2 = load_json_rows(RAW_DIR / "longbench_v2" / "data.json")
    by_domain: Dict[str, List[Dict[str, Any]]] = {}
    for row in lb2:
        by_domain.setdefault(str(row["domain"]), []).append(row)

    for scenario in scenarios:
        t0 = time.time()
        if scenario in LBP_PRIMARY_TASKS:
            out[scenario] = build_lbp_scenario(counter, scenario)
        elif scenario == "coding":
            out[scenario] = build_doc_scenario(
                counter,
                scenario,
                by_domain["Code Repository Understanding"],
                doc_text=lambda row: row["context"].strip(),
                question=lambda row: row["question"],
                source="longbench_v2:Code Repository Understanding",
                strata_key=lambda row: str(row["sub_domain"]),
            )
        elif scenario == "math_logic":
            out[scenario] = build_math_logic(counter)
        elif scenario == "general_agent":
            out[scenario] = build_general_agent(counter)
        elif scenario == "instruction_following":
            out[scenario] = build_instruction_following(counter)
        elif scenario == "multilingual":
            out[scenario] = build_multilingual(counter)
        else:  # pragma: no cover - guarded by argparse choices
            raise ValueError(f"unknown scenario {scenario}")
        seen = {target["target_id"] for target in out[scenario]}
        if len(seen) != TARGETS_PER_SCENARIO or len(out[scenario]) != TARGETS_PER_SCENARIO:
            raise ValueError(f"{scenario}: expected {TARGETS_PER_SCENARIO} unique targets")
        print(f"  [targets] {scenario:<22} {len(out[scenario])} targets  ({time.time() - t0:.1f}s)")
    return out


def validate_rows(
    rows: Sequence[Dict[str, Any]],
    targets_by_scenario: Dict[str, List[Dict[str, Any]]],
    buckets: Sequence[str],
    bodies: Optional[Dict[Tuple[str, str], str]] = None,
) -> Dict[str, Any]:
    problems: List[str] = []
    cells: Dict[str, Dict[str, int]] = {}
    by_target: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        cell = (row["scenario"], row["bucket"])
        cells.setdefault(row["scenario"], {}).setdefault(row["bucket"], 0)
        cells[row["scenario"]][row["bucket"]] += 1
        if row["tokens"] != BUCKETS[row["bucket"]]:
            problems.append(f"{row['id']}: tokens {row['tokens']} != {BUCKETS[row['bucket']]}")
        if row["meta"].get("content_tokens") != content_goal(row["bucket"]):
            problems.append(
                f"{row['id']}: content {row['meta'].get('content_tokens')} != "
                f"{content_goal(row['bucket'])}"
            )
        if row["messages_tokens"] != row["tokens"] or row["tools_tokens"] != 0 or row["tools"]:
            problems.append(f"{row['id']}: messages/tools accounting mismatch")
        if not row["messages"] or row["messages"][0]["role"] != "user":
            problems.append(f"{row['id']}: expected one user message")
        by_target.setdefault(row["target_id"], {})[row["bucket"]] = row

    for scenario in targets_by_scenario:
        for bucket in buckets:
            count = cells.get(scenario, {}).get(bucket, 0)
            if count != TARGETS_PER_SCENARIO and bucket in BUCKETS:
                problems.append(f"{scenario}/{bucket}: {count} rows != {TARGETS_PER_SCENARIO}")

    # strict nesting: same target => same task block, and the shorter row's context is a
    # prefix of the longer one.  The task block sits at the end of the content, so the full
    # rendered strings are not prefixes of each other; nesting is checked on the contexts.
    order = [b for b in ("1k", "8k", "32k") if b in buckets]
    for target_id, per_bucket in by_target.items():
        task_hashes = {per_bucket[b]["meta"]["task_sha256"] for b in order if b in per_bucket}
        if len(task_hashes) > 1:
            problems.append(f"{target_id}: task block differs across buckets")
        if bodies is None:
            continue
        texts = [bodies.get((target_id, b), "") for b in order if b in per_bucket]
        for short, long in zip(texts, texts[1:]):
            if not short or not long.startswith(short):
                problems.append(f"{target_id}: context rows are not nested prefixes")
                break

    langs: Dict[str, int] = {}
    for row in rows:
        langs[row["meta"].get("lang", "?")] = langs.get(row["meta"].get("lang", "?"), 0) + 1
    if len(targets_by_scenario) == len(SCENARIOS) and langs.get("zh", 0) == 0:
        problems.append("no Chinese sample in the test set")
    return {"problems": problems, "cells": cells, "langs": langs}


def write_outputs(
    rows: List[Dict[str, Any]],
    counter: Counter,
    targets_by_scenario: Dict[str, List[Dict[str, Any]]],
    out_dir: Path,
    report: Dict[str, Any],
) -> None:
    import yaml  # noqa: PLC0415

    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "testset.jsonl"
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    jsonl.write_text(payload, encoding="utf-8")

    sources: Dict[str, Dict[str, Any]] = {}
    for scenario, targets in targets_by_scenario.items():
        ids = [target["upstream_id"] for target in targets]
        sources[scenario] = {"dataset": SCENARIO_SOURCE[scenario], "upstream_ids": ids}

    manifest = {
        "name": "performance",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "builder": "tool/dataset/performance/build.py",
        "seed": SEED,
        "targets_per_scenario": TARGETS_PER_SCENARIO,
        "buckets": BUCKETS,
        "max_tokens": OSL,
        "min_completion_tokens_required": MIN_COMPLETION_TOKENS,
        "tokenizer": {
            "path": str(counter.path),
            "sha256": sha256_file(counter.path),
            "vocab_size": counter._tok.get_vocab_size(),
            "role": "reference (never a model under test; production runs use align.py)",
        },
        "construction": {
            "prompt_tokens": BUCKETS,
            "template_reserve_tokens": CONTENT_RESERVE,
            "content_tokens": {b: content_goal(b) for b in BUCKETS},
            "tolerance_tokens": 0,
            "method": "ordered unit stream prefix + token-boundary cut + fixed task block",
            "stream_min_tokens": STREAM_MIN_TOKENS,
        },
        "alignment": {
            "unit": "prompt tokens of the rendered chat request (chat template included)",
            "tolerance_tokens": 0,
            "verified_by": "tool/dataset/performance/align.py against the target model",
        },
        "cells": report["cells"],
        "langs": report["langs"],
        "stats": {
            "task_tokens_min": min(row["meta"]["task_tokens"] for row in rows),
            "task_tokens_max": max(row["meta"]["task_tokens"] for row in rows),
            "content_chars": {
                bucket: {
                    "min": min(sizes),
                    "median": sizes[len(sizes) // 2],
                    "max": max(sizes),
                }
                for bucket in BUCKETS
                for sizes in [[
                    len(row["messages"][0]["content"]) for row in rows if row["bucket"] == bucket
                ]]
                if sizes
            },
        },
        "sources": sources,
        "rows": len(rows),
        "rows_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    scenarios = {}
    for scenario in SCENARIOS:
        if scenario not in targets_by_scenario:
            continue
        scenarios[scenario] = {
            "dataset": "perf_jsonl",
            "dataset_params": {
                "scenario": scenario,
                "path": f"dataset/test/{out_dir.name}/testset.jsonl",
            },
        }
    yaml_doc = {
        "name": "performance",
        "title": "性能测试（8 场景 × 1K/8K/32K × 10 target，240 measured）",
        "description": (
            "性能测试集：每个场景 10 个 target，同一 target 渲染 1K/8K/32K 三档；"
            "样本由 tool/dataset/performance/build.py 从 dataset/raw 的上游数据确定性构建，"
            "并由 tool/dataset/performance/align.py 在运行前用目标模型 tokenizer "
            "对齐到精确的 prompt token 数（1024/8192/32768，0 容差）。"
            "strict_input_tokens=true 时运行器要求逐样本 prompt token 数严格等于目标值，"
            "不等即跳过记录，不做误差带统计。含中文场景；生成长度 max_tokens=50。"
        ),
        "scenarios": scenarios,
        "sampling": {
            "seed": SEED,
            "targets_per_scenario": TARGETS_PER_SCENARIO,
            "method": "sha256(seed|id) + stratified largest remainder",
        },
        "buckets": BUCKETS,
        # Strict input alignment: the runner skips any sample whose measured prompt
        # token count differs from the bucket target (no tolerance band).
        "strict_input_tokens": True,
        # Declared OSL: apply_testset() copies this into perf.max_tokens so the set
        # cannot be run at the 50-token default (which makes TG TPS unmeasurable).
        "max_tokens": OSL,
    }
    (out_dir / "testset.yaml").write_text(
        yaml.safe_dump(yaml_doc, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    print(f"[out] {jsonl}  ({len(payload.encode('utf-8')) / 1e6:.1f} MB, {len(rows)} rows)")


def check_outputs(out_dir: Path) -> int:
    """Validate on-disk build artefacts without rebuilding (gate ``--check``).

    Accepts both artefacts:
    * the committed reference build (``testset.jsonl`` SHA == manifest ``rows_sha256``,
      verified with the manifest's reference tokenizer), and
    * a per-model aligned build written by ``align.py`` (SHA recorded in
      ``dataset/.cache/performance/alignment-*.json``, verified with the model tokenizer
      recorded there; every row must still hit the exact bucket target).
    """
    import yaml  # noqa: PLC0415

    problems: List[str] = []
    jsonl = out_dir / "testset.jsonl"
    manifest_path = out_dir / "manifest.json"
    yaml_path = out_dir / "testset.yaml"
    for path in (jsonl, manifest_path, yaml_path):
        if not path.is_file():
            print(f"[check] missing {path}")
            return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    yaml_doc = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    rows = load_jsonl(jsonl)
    file_sha = sha256_file(jsonl)

    if manifest.get("name") != "performance":
        problems.append(f"manifest name {manifest.get('name')!r} != 'performance'")
    if manifest.get("buckets") != BUCKETS:
        problems.append(f"manifest buckets {manifest.get('buckets')} != {BUCKETS}")
    construction = manifest.get("construction") or {}
    if construction.get("tolerance_tokens") != 0:
        problems.append("manifest construction.tolerance_tokens != 0")
    if manifest.get("max_tokens") != OSL:
        problems.append(f"manifest max_tokens {manifest.get('max_tokens')} != {OSL}")
    if len(rows) != len(SCENARIOS) * len(BUCKETS) * TARGETS_PER_SCENARIO:
        problems.append(f"{len(rows)} rows != {len(SCENARIOS) * len(BUCKETS) * TARGETS_PER_SCENARIO}")

    ids = set()
    cells: Dict[str, Dict[str, int]] = {}
    targets = set()
    langs: Dict[str, int] = {}
    for row in rows:
        rid = row.get("id")
        if rid in ids:
            problems.append(f"duplicate id {rid}")
        ids.add(rid)
        scenario, bucket = row.get("scenario"), row.get("bucket")
        if scenario not in SCENARIOS:
            problems.append(f"{rid}: unknown scenario {scenario!r}")
            continue
        if bucket not in BUCKETS:
            problems.append(f"{rid}: unknown bucket {bucket!r}")
            continue
        cells.setdefault(scenario, {}).setdefault(bucket, 0)
        cells[scenario][bucket] += 1
        targets.add((scenario, row.get("target_id")))
        langs[row.get("meta", {}).get("lang", "?")] = langs.get(row.get("meta", {}).get("lang", "?"), 0) + 1
        if row.get("tokens") != BUCKETS[bucket]:
            problems.append(f"{rid}: tokens {row.get('tokens')} != {BUCKETS[bucket]}")
        if row.get("messages_tokens") != row.get("tokens") or row.get("tools_tokens") != 0:
            problems.append(f"{rid}: messages/tools accounting mismatch")
        if row.get("tools"):
            problems.append(f"{rid}: tools must stay empty (text-only tool catalogue)")
        messages = row.get("messages") or []
        if len(messages) != 1 or messages[0].get("role") != "user":
            problems.append(f"{rid}: expected exactly one user message")
        meta = row.get("meta") or {}
        for key in ("lang", "source", "upstream_id", "task_tokens", "content_tokens", "task_sha256", "body_sha256"):
            if key not in meta:
                problems.append(f"{rid}: meta.{key} missing")
    for scenario in SCENARIOS:
        for bucket in BUCKETS:
            count = cells.get(scenario, {}).get(bucket, 0)
            if count != TARGETS_PER_SCENARIO:
                problems.append(f"{scenario}/{bucket}: {count} rows != {TARGETS_PER_SCENARIO}")
    if len(targets) != len(SCENARIOS) * TARGETS_PER_SCENARIO:
        problems.append(f"{len(targets)} distinct targets != {len(SCENARIOS) * TARGETS_PER_SCENARIO}")
    if langs.get("zh", 0) == 0:
        problems.append("no Chinese sample in the test set")

    if yaml_doc.get("name") != "performance":
        problems.append(f"testset.yaml name {yaml_doc.get('name')!r} != 'performance'")
    if yaml_doc.get("buckets") != BUCKETS:
        problems.append(f"testset.yaml buckets {yaml_doc.get('buckets')} != {BUCKETS}")
    if yaml_doc.get("strict_input_tokens") is not True:
        problems.append("testset.yaml strict_input_tokens must be true")
    if yaml_doc.get("max_tokens") != OSL:
        problems.append(f"testset.yaml max_tokens {yaml_doc.get('max_tokens')} != {OSL}")
    if "osl" in yaml_doc:
        problems.append("testset.yaml must not use the legacy 'osl' key")
    if set((yaml_doc.get("scenarios") or {}).keys()) != set(SCENARIOS):
        problems.append("testset.yaml scenarios differ from the 8 design scenarios")
    for name, spec in (yaml_doc.get("scenarios") or {}).items():
        params = (spec or {}).get("dataset_params") or {}
        if spec.get("dataset") != "perf_jsonl" or params.get("scenario") != name:
            problems.append(f"testset.yaml scenarios.{name} wiring is wrong")
        path = str(params.get("path", ""))
        if not path.endswith("dataset/test/performance/testset.jsonl"):
            problems.append(f"testset.yaml scenarios.{name} path {path!r} is wrong")

    # ---- tokenizer oracle: reference build, or an aligned build recorded by align.py
    tokenizer_path: Optional[Path] = None
    reserve = int(construction.get("template_reserve_tokens") or 0)
    if file_sha == manifest.get("rows_sha256"):
        raw = (manifest.get("tokenizer") or {}).get("path")
        if raw:
            tokenizer_path = Path(raw)
        source = "manifest (reference build)"
    else:
        source = "missing"
        report_dir = PROJECT_ROOT / "dataset" / ".cache" / "performance"
        for report_path in sorted(report_dir.glob("alignment-*.json")):
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if report.get("rows_sha256") != file_sha:
                continue
            raw = (report.get("tokenizer") or {}).get("path")
            if raw:
                tokenizer_path = Path(raw)
            reserve = int(report.get("reserve_tokens") or 0)
            source = f"alignment report {report_path.name}"
            break
        if tokenizer_path is None:
            problems.append(
                "testset.jsonl SHA does not match manifest rows_sha256 and no alignment report "
                "records it; rebuild with build.py or align.py"
            )
    if tokenizer_path is not None:
        if not tokenizer_path.is_file():
            problems.append(f"verification tokenizer missing: {tokenizer_path}")
        else:
            counter = Counter(tokenizer_path)
            for row in rows:
                content = (row.get("messages") or [{}])[0].get("content", "")
                got = counter.n(content)
                expected = row.get("tokens")
                if got + reserve != expected:
                    problems.append(
                        f"{row.get('id')}: content tokens {got} + reserve {reserve} != {expected}"
                    )
            print(f"[check] tokenizer oracle: {tokenizer_path} ({source})")

    for problem in problems:
        print(f"  [FAIL] {problem}")
    if problems:
        print(f"[check] {len(problems)} problem(s)")
        return 1
    print(f"[check] ok: {len(rows)} rows, {len(SCENARIOS) * len(BUCKETS)} cells, sha {file_sha[:16]}…")
    return 0


def main(argv=None) -> int:
    global RAW_DIR, OUT_DIR, CONTENT_RESERVE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument(
        "--tokenizer",
        default=str(RAW_DIR / REFERENCE_TOKENIZER_REL),
        help="tokenizer.json used for the build (generic reference by default)",
    )
    parser.add_argument(
        "--reserve",
        type=int,
        default=0,
        help="chat-template token overhead to reserve (content = bucket - reserve)",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--targets", type=int, default=TARGETS_PER_SCENARIO)
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--buckets", default=",".join(BUCKETS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the on-disk build (row/cell/token/SHA/yaml contract) without rebuilding",
    )
    args = parser.parse_args(argv)

    RAW_DIR = Path(args.raw_dir)
    OUT_DIR = Path(args.out_dir)
    CONTENT_RESERVE = int(args.reserve)
    if CONTENT_RESERVE < 0:
        parser.error("--reserve must be >= 0")
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    buckets = [b.strip() for b in args.buckets.split(",") if b.strip()]
    for name in scenarios:
        if name not in SCENARIOS:
            parser.error(f"unknown scenario {name}")
    for name in buckets:
        if name not in BUCKETS:
            parser.error(f"unknown bucket {name}")

    if args.check:
        return check_outputs(OUT_DIR)

    counter = Counter(Path(args.tokenizer))
    print(f"[tokenizer] {args.tokenizer} (reserve={CONTENT_RESERVE})")
    targets_by_scenario = build_targets(counter, scenarios)

    rows: List[Dict[str, Any]] = []
    bodies: Dict[Tuple[str, str], str] = {}
    for scenario in scenarios:
        for index, target in enumerate(targets_by_scenario[scenario]):
            for bucket in buckets:
                row, body = make_row(counter, scenario, target, bucket, index)
                rows.append(row)
                bodies[(row["target_id"], bucket)] = body

    report = validate_rows(rows, targets_by_scenario, buckets, bodies)
    for problem in report["problems"]:
        print(f"  [FAIL] {problem}")
    print(f"[stats] rows={len(rows)} langs={report['langs']}")
    print(f"[stats] cells={ {s: dict(v) for s, v in sorted(report['cells'].items())} }")
    if report["problems"]:
        return 1
    if args.dry_run:
        print("[dry-run] nothing written")
        return 0
    write_outputs(rows, counter, targets_by_scenario, OUT_DIR, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
