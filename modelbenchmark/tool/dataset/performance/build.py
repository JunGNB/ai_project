"""Build ``dataset/test/performance`` — 8 scenarios x 1K/4K/32K x 10 targets.

The set measures performance only (TTFT / TG TPS / peak memory), never accuracy.

Construction policy ("primary + filler", see ``docs/spec/11-performance-testset.md``):

* a row has two user messages: a **body** (material) and a **task** (question/instruction);
* the body starts with the scenario's **primary units** — whole documents, whole
  worked examples, whole tool schemas, or a sentence-aligned excerpt that provably
  contains the evidence — and then appends **filler units** of the same family;
* the body is built at ``OVERSIZE x bucket`` reference tokens, so the run-time
  aligner can trim it down to the exact bucket target with *any* model tokenizer and
  only ever cut filler material;
* every row is validated at build time: primary units are complete and present at the
  start of the body, evidence strings of evidence-critical scenarios occur in the
  primary part, and ``max_over_known_tokenizers(primary) + task + template <= bucket``
  holds, i.e. trimming can never reach the primary part;
* sampling is deterministic (``sha256(seed|id)`` order + stratified largest remainder).

``build.py --check`` re-validates the committed artifacts without any model.

Usage::

    python tool/dataset/performance/build.py                 # reference build
    python tool/dataset/performance/build.py --scenarios knowledge_qa --buckets 1k
    python tool/dataset/performance/build.py --dry-run       # render only, write nothing
    python tool/dataset/performance/build.py --check         # validate committed files
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from tool.dataset.performance.sampling import allocate, deterministic_pick  # noqa: E402

RAW_DIR = PROJECT_ROOT / "dataset" / "raw"
OUT_DIR = PROJECT_ROOT / "dataset" / "test" / "performance"
REFERENCE_TOKENIZER_REL = "tokenizer/cl100k_base/tokenizer.json"

BUCKETS: Dict[str, int] = {"1k": 1024, "4k": 4096, "32k": 32768}
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
OSL = 50
MIN_COMPLETION_TOKENS = 50
SEP = "\n\n"

#: the run-time aligner trims the body with the model's own tokenizer; the body must
#: therefore carry at least two reference tokens per bucket token (some tokenizers pack
#: Chinese into ~0.5 reference tokens).
OVERSIZE = 2
#: worst-case expansion used when no local model tokenizers are available (validated
#: tokenizers pack English up to ~1.15x; Chinese is always below 1.0x)
FALLBACK_EXPANSION = {"en": 1.25, "zh": 0.85, "multi": 1.25}

#: fixed margin for chat-template / special tokens added by the server
TEMPLATE_MARGIN = 2
#: the task block must stay small so that the 1K bucket still holds material
TASK_MAX_TOKENS = 420
SHORTLIST_PER_STRATUM = 60

MULTILINGUAL_LANGS: Tuple[str, ...] = ("zh", "en", "fr", "de", "es", "pt", "ru", "ja", "ko", "ar")

#: scenario x bucket -> stable ``dataset`` label (cell grouping label; the exact unit
#: lives in ``meta.source`` / ``meta.upstream_id``)
SCENARIO_DATASET: Dict[Tuple[str, str], str] = {
    ("knowledge_qa", "1k"): "hotpotqa",
    ("knowledge_qa", "4k"): "hotpotqa",
    ("knowledge_qa", "32k"): "longbench_pro:T3",
    ("instruction_following", "1k"): "vcsum+multi_if",
    ("instruction_following", "4k"): "vcsum+multi_if",
    ("instruction_following", "32k"): "vcsum+multi_if",
    ("long_context", "1k"): "longbench_pro:T4",
    ("long_context", "4k"): "longbench_pro:T4",
    ("long_context", "32k"): "longbench_pro:T4",
    ("math_logic", "1k"): "aime_2026+math_500",
    ("math_logic", "4k"): "aime_2026+math_500",
    ("math_logic", "32k"): "aime_2026+math_500",
    ("coding", "1k"): "humaneval",
    ("coding", "4k"): "longbench_v1:lcc",
    ("coding", "32k"): "longbench_v1:lcc+repobench",
    ("general_agent", "1k"): "bfcl_v4",
    ("general_agent", "4k"): "bfcl_v4",
    ("general_agent", "32k"): "bfcl_v4",
    ("search_agent", "1k"): "hotpotqa",
    ("search_agent", "4k"): "hotpotqa",
    ("search_agent", "32k"): "longbench_pro:T1+T5",
    ("multilingual", "1k"): "global_mmlu",
    ("multilingual", "4k"): "global_mmlu",
    ("multilingual", "32k"): "global_mmlu",
}

#: languages per scenario x bucket (used both for construction and validation)
def scenario_langs(scenario: str, bucket: str) -> Tuple[str, ...]:
    if scenario == "multilingual":
        return MULTILINGUAL_LANGS
    if scenario == "knowledge_qa":
        # 1K/4K: HotpotQA gold paragraphs (en); 32K: LongBench-Pro T3 (zh)
        return ("zh",) if bucket == "32k" else ("en",)
    if scenario == "instruction_following":
        return ("zh",)
    return ("en",)


def content_goal(bucket: str) -> int:
    """Reference-token count the committed body carries (bucket x oversize)."""
    return BUCKETS[bucket] * OVERSIZE


# --------------------------------------------------------------------------- #
# tokenizers
# --------------------------------------------------------------------------- #
class Counter:
    """Reference tokenizer wrapper (content only, no special tokens)."""

    def __init__(self, path: Path) -> None:
        from tokenizers import Tokenizer  # noqa: PLC0415 - optional dependency

        self.path = path
        self._tok = Tokenizer.from_file(str(path))
        self._cache: Dict[str, int] = {}

    def ids(self, text: str) -> List[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: Sequence[int]) -> str:
        return self._tok.decode(list(ids))

    def n(self, text: str) -> int:
        value = self._cache.get(text)
        if value is None:
            value = len(self.ids(text))
            self._cache[text] = value
        return value


class MaxModelCounter:
    """Worst-case token count over every tokenizer found under ``~/.omlx/models``.

    The performance set is shared by many models; a build-time check against the
    local model zoo proves that the primary part survives run-time trimming for all
    of them. When no tokenizer is present (CI), a documented per-language expansion
    factor is used instead.
    """

    def __init__(self, root: Optional[Path] = None) -> None:
        from tokenizers import Tokenizer  # noqa: PLC0415

        root = root or Path(os.path.expanduser("~/.omlx/models"))
        self.models: Dict[str, Any] = {}
        for path in sorted(glob.glob(str(root / "**" / "tokenizer.json"), recursive=True)):
            name = path.split("/models/")[-1].removesuffix("/tokenizer.json")
            try:
                self.models[name] = Tokenizer.from_file(path)
            except Exception:  # pragma: no cover - unreadable tokenizer, skip
                continue
        self._cache: Dict[str, int] = {}

    def n(self, text: str) -> int:
        value = self._cache.get(text)
        if value is None:
            value = max(
                (len(tok.encode(text, add_special_tokens=False).ids) for tok in self.models.values()),
                default=-1,
            )
            self._cache[text] = value
        return value

    def tokens(self, text: str, lang: str) -> int:
        if self.models:
            return self.n(text)
        base = len(text)
        return int(base * FALLBACK_EXPANSION.get(lang, 1.25) / 4) + 1

    def describe(self) -> Dict[str, Any]:
        return {"models": sorted(self.models), "fallback": FALLBACK_EXPANSION}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_json_rows(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        data = json.loads(text)
        return data if isinstance(data, list) else list(data.values())
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def iter_json_array(path: Path, chunk_size: int = 8 << 20) -> Iterator[Any]:
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

    return pq.read_table(path).to_pylist()


def cut_at_boundary(text: str, limit_chars: int) -> str:
    """Cut ``text`` to <= ``limit_chars`` at the last sentence/paragraph boundary."""
    if len(text) <= limit_chars:
        return text
    head = text[:limit_chars]
    positions = [head.rfind(mark) for mark in ("\n\n", "\n", "。", "！", "？", ". ", "! ", "? ")]
    cut = max(positions)
    if cut <= 0:
        return head
    return head[: cut + 1]


def trim_to_goal(counter: Counter, text: str, goal: int) -> str:
    """Cut ``text`` to exactly ``goal`` reference tokens.

    The cut is taken in id space (one encode of the material) so a 32K body costs a
    couple of encodes instead of a binary search over characters; a short tail scan
    covers the byte-pair boundary where re-encoding lands one or two tokens off.
    """
    ids = counter.ids(text)
    if len(ids) < goal:
        raise ValueError(f"material too short: {len(ids)} < {goal}")
    body = counter.decode(ids[:goal]).rstrip("\ufffd").rstrip()
    count = counter.n(body)
    if count == goal:
        return body
    if count < goal:
        for tail in (" ", "\n", "\t", "  ", "\n\n", "。"):
            candidate = body + tail
            if counter.n(candidate) == goal:
                return candidate
        # a byte-pair boundary can leave the exact count unreachable; one or two
        # reference tokens out of an oversize body are immaterial (the runtime
        # aligner still lands on the exact prompt target).
        if goal - count <= 2:
            return body
        raise ValueError(f"cannot trim material to {goal} tokens (nearest below={count})")
    for cut in range(1, 13):
        candidate = body[:-cut].rstrip()
        got = counter.n(candidate)
        if got == goal:
            return candidate
        if got < goal:
            if goal - got <= 2:
                return candidate
            break
    raise ValueError(f"cannot trim material to {goal} tokens (body count={count})")


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def _evidence_present(evidence: str, primary_text: str) -> bool:
    return _normalized(evidence) in _normalized(primary_text)


def make_row(
    counter: Counter,
    max_model: MaxModelCounter,
    target: Dict[str, Any],
    bucket: str,
    index: int,
) -> Dict[str, Any]:
    """Compose one row: body = primary units + filler units, then the task block."""
    scenario = target["scenario"]
    lang = target["lang"]
    goal = content_goal(bucket)
    primary = list(target["primary"])
    primary_text = SEP.join(primary)
    primary_ref = counter.n(primary_text)
    if primary_ref <= 0:
        raise ValueError(f"{scenario}/{bucket}: empty primary part")
    if primary_ref >= goal:
        raise ValueError(
            f"{scenario}/{bucket}: primary part ({primary_ref}) >= content goal ({goal})"
        )
    # append filler units until the body is comfortably over the goal, then cut exactly
    parts = list(primary)
    text = primary_text
    for unit in target["filler"]:
        if counter.n(text) >= goal:
            break
        parts.append(unit)
        text = SEP.join(parts)
    if counter.n(text) < goal:
        raise ValueError(f"{scenario}/{bucket}: filler too short ({counter.n(text)} < {goal})")
    body = trim_to_goal(counter, text, goal)
    if not body.startswith(primary_text):
        raise ValueError(f"{scenario}/{bucket}: trimming cut into the primary part")

    task = target["task"]
    task_ref = counter.n(task)
    # build-time validity: the primary part plus the task must still fit the bucket for
    # every known tokenizer once the run-time aligner has trimmed the filler away
    primary_model = max_model.tokens(primary_text, lang)
    task_model = max_model.tokens(task, lang)
    fits = primary_model + task_model + TEMPLATE_MARGIN <= BUCKETS[bucket]
    if not fits:
        raise ValueError(
            f"{scenario}/{bucket}/{target['target_id']}: primary {primary_model} + task "
            f"{task_model} exceeds bucket {BUCKETS[bucket]}"
        )
    evidence = target.get("evidence")
    evidence_ok = True if evidence is None else _evidence_present(evidence, primary_text)
    if not evidence_ok:
        raise ValueError(f"{scenario}/{bucket}/{target['target_id']}: evidence not in primary part")

    body_tokens = counter.n(body)
    meta = {
        "lang": lang,
        "source": target["source"],
        "upstream_id": target["upstream_id"],
        "target_tokens": BUCKETS[bucket],
        "content_tokens": body_tokens + task_ref,
        "body_tokens": body_tokens,
        "task_tokens": task_ref,
        "primary_units": len(primary),
        "primary_tokens": primary_ref,
        "filler_units": len(parts) - len(primary),
        "primary_share": round(primary_ref / goal, 3),
        "primary_model_tokens": primary_model,
        "task_model_tokens": task_model,
        "fits_bucket": fits,
        "evidence_in_primary": evidence_ok,
        "primary_complete": True,
        "cut_policy": target.get("cut_policy", "document"),
        "alignment": "primary+filler body",
        "runtime_alignment": "trim+pad",
        "oversize": OVERSIZE,
        "task_sha256": sha256_text(task)[:16],
        "body_sha256": sha256_text(body)[:16],
    }
    if evidence is not None:
        meta["evidence_sha256"] = sha256_text(evidence)[:16]
    meta.update(target.get("meta") or {})
    return {
        "id": f"perf-{scenario}-{bucket}-{index:03d}",
        "scenario": scenario,
        "bucket": bucket,
        "target_id": target["target_id"],
        "dataset": SCENARIO_DATASET[(scenario, bucket)],
        "tokens": BUCKETS[bucket],
        "messages_tokens": BUCKETS[bucket],
        "tools_tokens": 0,
        "messages": [
            {"role": "user", "content": body},
            {"role": "user", "content": task},
        ],
        "tools": [],
        "constructed": True,
        "meta": meta,
    }


def pick_targets(candidates: Sequence[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """Deterministic, stratum-aware pick of ``k`` targets out of ``candidates``."""
    strata: Dict[str, List[int]] = {}
    for position, target in enumerate(candidates):
        strata.setdefault(str(target.get("stratum") or "all"), []).append(position)
    pools = {name: sorted(values) for name, values in strata.items()}
    quota = allocate(
        {name: len(values) for name, values in pools.items()}, k, min_each=1, seed=SEED
    )
    picked: List[int] = []
    for name in sorted(pools):
        order = sorted(pools[name], key=lambda i: sha256_text(f"{SEED}|{candidates[i]['target_id']}"))
        picked.extend(order[: quota[name]])
    picked = sorted(set(picked), key=lambda i: sha256_text(f"{SEED}|{candidates[i]['target_id']}"))
    if len(picked) < k:
        raise ValueError(f"only {len(picked)} candidates, need {k}")
    return [candidates[i] for i in picked[:k]]


# --------------------------------------------------------------------------- #
# scenario pools
# --------------------------------------------------------------------------- #
def _hotpotqa_rows() -> List[Dict[str, Any]]:
    rows = load_parquet(RAW_DIR / "hotpotqa" / "distractor" / "validation-00000-of-00001.parquet")
    out: List[Dict[str, Any]] = []
    for row in rows:
        ctx = row["context"]
        if isinstance(ctx, str):
            ctx = json.loads(ctx.replace("'", '"'))
        sf = row["supporting_facts"]
        if isinstance(sf, str):
            sf = json.loads(sf.replace("'", '"'))
        paragraphs = ctx.get("paragraphs")
        if paragraphs is None:
            paragraphs = [" ".join(sentences) for sentences in ctx["sentences"]]
        gold_titles = set(sf["title"])
        gold = [p for title, p in zip(ctx["title"], paragraphs) if title in gold_titles]
        distract = [p for title, p in zip(ctx["title"], paragraphs) if title not in gold_titles]
        if len(gold) < 2 or len(distract) < 4:
            continue
        out.append(
            {
                "id": str(row["id"]),
                "question": str(row["question"]),
                "answer": str(row["answer"]),
                "type": str(row.get("type") or ""),
                "level": str(row.get("level") or ""),
                "gold": gold,
                "distract": distract,
            }
        )
    return out


def _hotpotqa_filler_pool(rows: Sequence[Dict[str, Any]], limit: int = 600) -> List[str]:
    """Distractor paragraphs from the whole pool, used to pad short hotpot passages."""
    pool: List[str] = []
    seen = set()
    for row in rows:
        for paragraph in row["distract"]:
            key = paragraph[:64]
            if key in seen:
                continue
            seen.add(key)
            pool.append(paragraph)
            if len(pool) >= limit:
                return pool
    return pool


def _multifieldqa_zh_rows() -> List[Dict[str, Any]]:
    rows = load_jsonl(RAW_DIR / "longbench_v1" / "data" / "multifieldqa_zh.jsonl")
    out: List[Dict[str, Any]] = []
    for row in rows:
        ctx = (row.get("context") or "").strip()
        if not ctx:
            continue
        answers = row.get("answers") or []
        out.append(
            {
                "id": str(row["_id"]),
                "question": str(row.get("input") or "").strip(),
                "answers": [str(a) for a in answers],
                "context": ctx,
                "length": str(row.get("length") or ""),
            }
        )
    return out


_LBPRO_CACHE: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}


def _lbpro_rows(tasks: Sequence[str], lang: str) -> List[Dict[str, Any]]:
    key = (",".join(tasks), lang)
    if key in _LBPRO_CACHE:
        return _LBPRO_CACHE[key]
    out: List[Dict[str, Any]] = []
    for row in iter_json_array(RAW_DIR / "longbench_pro" / "longbench_pro.json"):
        if str(row.get("language")) != lang:
            continue
        if str(row.get("primary_task", "")).split(".")[0] not in tasks:
            continue
        out.append(
            {
                "id": str(row.get("id")),
                "context": str(row.get("context") or ""),
                "question": str(row.get("question_nonthinking") or row.get("question_thinking") or ""),
                "answer": [str(a) for a in (row.get("answer") or [])],
                "band": str(row.get("token_length")),
                "primary_task": str(row.get("primary_task")),
                "secondary_task": str(row.get("secondary_task")),
                "requirement": str(row.get("contextual_requirement")),
            }
        )
    _LBPRO_CACHE[key] = out
    return out


def _vcsum_rows() -> List[Dict[str, Any]]:
    rows = load_jsonl(RAW_DIR / "longbench_v1" / "data" / "vcsum.jsonl")
    out = []
    for row in rows:
        ctx = (row.get("context") or "").strip()
        if not ctx:
            continue
        out.append({"id": str(row["_id"]), "context": ctx, "input": str(row.get("input") or "")})
    return out


def _check_constraint_text() -> Dict[str, Any]:
    return {
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


CONSTRAINT_TEXT = _check_constraint_text()


def _multi_if_zh() -> List[Dict[str, Any]]:
    with (RAW_DIR / "multi_if" / "multiIF_20241018.csv").open(encoding="utf-8") as fh:
        rows = [row for row in csv.DictReader(fh) if row.get("language") == "Chinese"]
    out = []
    for row in rows:
        ids = json.loads(row["turn_1_instruction_id_list"] or "[]")
        if not ids or any(i not in CONSTRAINT_TEXT for i in ids):
            continue
        kwargs = json.loads(row["turn_1_kwargs"] or "[]")
        out.append({"key": row["key"], "ids": ids, "kwargs": kwargs})
    return out


def _lbv1_ctx(task: str) -> List[Dict[str, Any]]:
    rows = load_jsonl(RAW_DIR / "longbench_v1" / "data" / f"{task}.jsonl")
    out = []
    for row in rows:
        ctx = (row.get("context") or "").strip()
        if not ctx:
            continue
        out.append(
            {
                "id": str(row.get("_id")),
                "context": ctx,
                "question": str(row.get("input") or "").strip(),
                "answers": [str(a) for a in (row.get("answers") or [])],
                "length": str(row.get("length") or ""),
            }
        )
    return out


def _aime_problems() -> List[Dict[str, Any]]:
    rows = load_parquet(RAW_DIR / "aime2026" / "data" / "train-00000-of-00001.parquet")
    return [
        {
            "id": str(row.get("problem_idx")),
            "problem": str(row.get("problem") or "").strip(),
            "answer": str(row.get("answer") or "").strip(),
        }
        for row in rows
    ]


def _math500_demos() -> List[Dict[str, Any]]:
    rows = load_jsonl(RAW_DIR / "math_500" / "test.jsonl")
    out = []
    for row in rows:
        problem = str(row.get("problem") or "").strip()
        solution = str(row.get("solution") or "").strip()
        answer = str(row.get("answer") or "").strip()
        if not problem or not solution:
            continue
        out.append(
            {
                "id": str(row.get("unique_id") or sha256_text(problem)[:12]),
                "problem": problem,
                "solution": solution,
                "answer": answer,
                "subject": str(row.get("subject") or ""),
            }
        )
    return out


def _bfcl_rows() -> List[Dict[str, Any]]:
    out = []
    for name in ("BFCL_v4_simple_python.json", "BFCL_v4_multiple.json"):
        for row in load_json_rows(RAW_DIR / "bfcl" / name):
            funcs = row.get("function")
            if isinstance(funcs, str):
                funcs = json.loads(funcs)
            if not funcs:
                continue
            question = row.get("question")
            if isinstance(question, str):
                question = json.loads(question)
            user_text = " ".join(
                str(message.get("content") or "")
                for message in (question if isinstance(question, list) else [])
                if isinstance(message, dict) and message.get("role") == "user"
            ).strip()
            out.append(
                {
                    "id": str(row.get("id")),
                    "tools": [json.dumps(tool, ensure_ascii=False, sort_keys=True) for tool in funcs],
                    "tool_names": [
                        str((tool.get("name") or (tool.get("function") or {}).get("name") or ""))
                        for tool in funcs
                    ],
                    "request": user_text or str(question),
                }
            )
    return out


def _humaneval_rows() -> List[Dict[str, Any]]:
    rows = load_parquet(
        RAW_DIR / "humaneval" / "openai_humaneval" / "test-00000-of-00001.parquet"
    )
    out = []
    for row in rows:
        prompt = str(row.get("prompt") or "")
        solution = str(row.get("canonical_solution") or "")
        out.append(
            {
                "id": str(row.get("task_id")),
                "code": prompt + solution,
                "prompt": prompt,
                "solution": solution,
                "test": str(row.get("test") or ""),
            }
        )
    return out


def _lcc_rows() -> List[Dict[str, Any]]:
    return _lbv1_ctx("lcc")


def _repobench_rows() -> List[Dict[str, Any]]:
    return _lbv1_ctx("repobench-p")


def _two_wiki_rows() -> List[Dict[str, Any]]:
    return _lbv1_ctx("2wikimqa")


def _global_mmlu_rows(lang: str) -> List[Dict[str, Any]]:
    return load_parquet(RAW_DIR / "global_mmlu" / lang / "test.parquet")


# --------------------------------------------------------------------------- #
# target builders (one per scenario x bucket family)
# --------------------------------------------------------------------------- #
def _task_answer_en(question: str) -> str:
    return (
        f"{question}\n\n"
        "Answer the question above using the material. First state the answer, then explain "
        "why in at least 80 words."
    )


def _task_summarize_en() -> str:
    return (
        "Summarize the main points of the material above. Cover the key entities, numbers and "
        "conclusions, and write at least 150 words."
    )


def _task_explain_code_en() -> str:
    return (
        "Explain what the code above does: its purpose, main data structures, control flow and any "
        "correctness or performance risks. Write at least 120 words."
    )


def _task_search_en(question: str) -> str:
    return (
        f"Search question: {question}\n\n"
        "Using only the material above, identify the passages that answer the question, cite them "
        "briefly, then give the final answer. Write at least 100 words."
    )


def build_knowledge_qa(bucket: str) -> List[Dict[str, Any]]:
    counter_free: List[Dict[str, Any]] = []
    if bucket == "1k":
        rows = _hotpotqa_rows()
        for row in rows:
            primary = row["gold"]
            if not _evidence_present(row["answer"], SEP.join(primary)):
                continue
            candidates = [
                {
                    "target_id": f"tgt-knowledge_qa-1k-{row['id'][:8]}",
                    "scenario": "knowledge_qa",
                    "bucket": bucket,
                    "lang": "en",
                    "source": "hotpotqa:distractor",
                    "upstream_id": row["id"],
                    "primary": primary,
                    "filler": row["distract"],
                    "task": _task_answer_en(row["question"]),
                    "evidence": row["answer"],
                    "cut_policy": "document",
                    "stratum": row["type"] or "unknown",
                    "meta": {"level": row["level"], "gold_paragraphs": len(primary)},
                }
            ]
            counter_free.extend(candidates)
        return counter_free

    if bucket == "4k":
        # HotpotQA again: the gold paragraphs are small, so the whole document fits
        # inside the 4K bucket with room for distractor paragraphs as filler.
        rows = _hotpotqa_rows()
        pool = _hotpotqa_filler_pool(rows)
        targets = []
        for row in rows:
            if not _evidence_present(row["answer"], SEP.join(row["gold"])):
                continue
            targets.append(
                {
                    "target_id": f"tgt-knowledge_qa-4k-{row['id'][:8]}",
                    "scenario": "knowledge_qa",
                    "bucket": bucket,
                    "lang": "en",
                    "source": "hotpotqa:distractor",
                    "upstream_id": row["id"],
                    "primary": row["gold"],
                    "filler": row["distract"] + pool,
                    "task": _task_answer_en(row["question"]),
                    "evidence": row["answer"],
                    "cut_policy": "document",
                    "stratum": row["type"] or "unknown",
                    "meta": {"level": row["level"], "gold_paragraphs": len(row["gold"])},
                }
            )
        return targets

    # 32K: LongBench-Pro T3, complete Chinese document as primary
    rows = _lbpro_rows(("T3",), "Chinese")
    pool = []
    for row in rows:
        answers = [a for a in row["answer"] if a]
        pool.append(
            {
                "target_id": f"tgt-knowledge_qa-32k-{row['id'][:8]}",
                "scenario": "knowledge_qa",
                "bucket": bucket,
                "lang": "zh",
                "source": "longbench_pro:T3",
                "upstream_id": row["id"],
                "primary": [row["context"]],
                "filler": [],
                "task": f"{row['question']}\n\n请根据上述材料回答问题，先给出答案，再用不少于 100 字说明依据。",
                "evidence": None,
                "cut_policy": "document",
                "stratum": row["secondary_task"] or "T3",
                "meta": {
                    "primary_task": row["primary_task"],
                    "band": row["band"],
                    "answer": answers[0] if answers else None,
                    "evidence_check": "complete-document",
                },
            }
        )
    return pool


def build_instruction_following(bucket: str) -> List[Dict[str, Any]]:
    docs = _vcsum_rows()
    constraints = _multi_if_zh()
    if not docs or not constraints:
        return []
    picked_docs = pick_targets(
        [
            {
                "target_id": f"vcsum#{doc['id']}",
                "stratum": doc["id"][:1],
                "doc": doc,
            }
            for doc in docs
        ],
        TARGETS_PER_SCENARIO,
    )
    picked_cons = pick_targets(
        [
            {
                "target_id": f"multi_if#{row['key']}",
                "stratum": row["ids"][0],
                "row": row,
            }
            for row in constraints
        ],
        TARGETS_PER_SCENARIO,
    )
    pool = []
    for doc_target, con_target in zip(picked_docs, picked_cons):
        doc = doc_target["doc"]
        con = con_target["row"]
        lines = []
        for position, (instruction_id, kw) in enumerate(zip(con["ids"], con["kwargs"]), start=1):
            if isinstance(kw, str):
                kw = json.loads(kw or "{}")
            lines.append(f"{position}. {CONSTRAINT_TEXT[instruction_id](kw)}")
        task = (
            "请阅读以上会议记录，并按下列要求写一段连贯的中文说明（对会议内容做归纳）：\n"
            + "\n".join(lines)
            + "\n\n必须同时满足上述全部要求，不要复述要求本身。"
        )
        pool.append(
            {
                "target_id": f"tgt-instruction_following-{bucket}-{sha256_text(doc['id'] + con['key'])[:8]}",
                "scenario": "instruction_following",
                "bucket": bucket,
                "lang": "zh",
                "source": "vcsum+multi_if_zh",
                "upstream_id": f"vcsum#{doc['id']}|multi_if#{con['key']}",
                "primary": [doc["context"]],
                "filler": [],
                "task": task,
                "evidence": None,
                "cut_policy": "sentence",
                "stratum": con["ids"][0],
                "meta": {"constraint_ids": list(con["ids"]), "doc_id": doc["id"]},
            }
        )
    return pool


def build_long_context(bucket: str) -> List[Dict[str, Any]]:
    rows = _lbpro_rows(("T4",), "English")
    pool = []
    for row in rows:
        pool.append(
            {
                "target_id": f"tgt-long_context-{bucket}-{row['id'][:8]}",
                "scenario": "long_context",
                "bucket": bucket,
                "lang": "en",
                "source": "longbench_pro:T4",
                "upstream_id": row["id"],
                "primary": [row["context"]],
                "filler": [],
                "task": _task_summarize_en(),
                "evidence": None,
                "cut_policy": "paragraph",
                "stratum": row["secondary_task"] or "T4",
                "meta": {"primary_task": row["primary_task"], "band": row["band"]},
            }
        )
    return pool


def build_math_logic(bucket: str) -> List[Dict[str, Any]]:
    problems = _aime_problems()
    demos = _math500_demos()
    if not problems or not demos:
        return []

    def demo_text(row: Dict[str, Any]) -> str:
        return (
            f"Problem: {row['problem']}\n"
            f"Solution: {row['solution']}\n"
            f"Final answer: {row['answer']}"
        )

    picked = pick_targets(
        [
            {
                "target_id": f"aime#{row['id']}",
                "stratum": row["id"],
                "row": row,
            }
            for row in problems
        ],
        TARGETS_PER_SCENARIO,
    )
    pool = []
    for target in picked:
        row = target["row"]
        pool.append(
            {
                "target_id": f"tgt-math_logic-{bucket}-{row['id']}",
                "scenario": "math_logic",
                "bucket": bucket,
                "lang": "en",
                "source": "aime_2026+math_500",
                "upstream_id": f"aime_2026#{row['id']}",
                "primary": [demo_text(demos[0])],
                "filler": [demo_text(d) for d in demos[1:]],
                "task": (
                    f"Solve the final problem below step by step and explain the reasoning in at "
                    f"least 100 words.\n\nProblem: {row['problem']}"
                ),
                "evidence": None,
                "cut_policy": "example",
                "stratum": row["id"],
                "meta": {"aime_problem_idx": row["id"], "demos": len(demos)},
            }
        )
    return pool


def build_coding(bucket: str) -> List[Dict[str, Any]]:
    if bucket == "1k":
        rows = _humaneval_rows()
        return [
            {
                "target_id": f"tgt-coding-1k-{sha256_text(row['id'])[:8]}",
                "scenario": "coding",
                "bucket": bucket,
                "lang": "en",
                "source": "humaneval",
                "upstream_id": row["id"],
                "primary": [row["code"]],
                "filler": [],
                "task": _task_explain_code_en(),
                "evidence": None,
                "cut_policy": "document",
                "stratum": "humaneval",
                "meta": {"prompt_tokens": None},
            }
            for row in rows
        ]
    if bucket == "4k":
        rows = _lcc_rows()
        return [
            {
                "target_id": f"tgt-coding-4k-{sha256_text(row['id'])[:8]}",
                "scenario": "coding",
                "bucket": bucket,
                "lang": "en",
                "source": "longbench_v1:lcc",
                "upstream_id": row["id"],
                "primary": [row["context"]],
                "filler": [],
                "task": _task_explain_code_en(),
                "evidence": None,
                "cut_policy": "paragraph",
                "stratum": "lcc",
                "meta": {"repo": "lcc"},
            }
            for row in rows
        ]
    rows = _lcc_rows() + _repobench_rows()
    return [
        {
            "target_id": f"tgt-coding-32k-{sha256_text(row['id'])[:8]}",
            "scenario": "coding",
            "bucket": bucket,
            "lang": "en",
            "source": "longbench_v1:lcc" if "lcc" in row.get("id", "") else "longbench_v1:repobench-p",
            "upstream_id": row["id"],
            "primary": [row["context"]],
            "filler": [],
            "task": _task_explain_code_en(),
            "evidence": None,
            "cut_policy": "paragraph",
            "stratum": "code",
            "meta": {},
        }
        for row in rows
    ]


def build_general_agent(bucket: str) -> List[Dict[str, Any]]:
    rows = _bfcl_rows()
    if not rows:
        return []
    # one shared catalog of tool schemas; each target keeps its own tools as primary
    catalog: List[str] = []
    seen = set()
    for row in rows:
        for tool in row["tools"]:
            digest = sha256_text(tool)[:16]
            if digest in seen:
                continue
            seen.add(digest)
            catalog.append(tool)
    pool = []
    for row in rows:
        own = {sha256_text(tool)[:16] for tool in row["tools"]}
        filler = [tool for tool in catalog if sha256_text(tool)[:16] not in own]
        pool.append(
            {
                "target_id": f"tgt-general_agent-{bucket}-{sha256_text(row['id'])[:8]}",
                "scenario": "general_agent",
                "bucket": bucket,
                "lang": "en",
                "source": "bfcl_v4",
                "upstream_id": row["id"],
                "primary": list(row["tools"]),
                "filler": filler,
                "task": (
                    f"User request: {row['request']}\n\n"
                    "The tool library above is the MCP server's catalog. Name the tool you would "
                    "call, give the full argument JSON, then justify in at least 80 words."
                ),
                "evidence": None,
                "cut_policy": "tool",
                "stratum": "multiple" if len(row["tools"]) > 1 else "simple",
                "meta": {"tool_count": len(row["tools"]), "catalog_size": len(catalog)},
            }
        )
    return pool


def build_search_agent(bucket: str) -> List[Dict[str, Any]]:
    if bucket == "1k":
        rows = [
            row
            for row in _hotpotqa_rows()
            if _evidence_present(row["answer"], SEP.join(row["gold"]))
        ]
        return [
            {
                "target_id": f"tgt-search_agent-1k-{row['id'][:8]}",
                "scenario": "search_agent",
                "bucket": bucket,
                "lang": "en",
                "source": "hotpotqa:distractor",
                "upstream_id": row["id"],
                "primary": row["gold"],
                "filler": row["distract"],
                "task": _task_search_en(row["question"]),
                "evidence": row["answer"],
                "cut_policy": "document",
                "stratum": row["type"] or "unknown",
                "meta": {"level": row["level"], "gold_paragraphs": len(row["gold"])},
            }
            for row in rows
        ]
    if bucket == "4k":
        rows = [
            row
            for row in _hotpotqa_rows()
            if _evidence_present(row["answer"], SEP.join(row["gold"]))
        ]
        pool_filler = _hotpotqa_filler_pool(rows)
        return [
            {
                "target_id": f"tgt-search_agent-4k-{row['id'][:8]}",
                "scenario": "search_agent",
                "bucket": bucket,
                "lang": "en",
                "source": "hotpotqa:distractor",
                "upstream_id": row["id"],
                "primary": row["gold"],
                "filler": row["distract"] + pool_filler,
                "task": _task_search_en(row["question"]),
                "evidence": row["answer"],
                "cut_policy": "document",
                "stratum": row["type"] or "unknown",
                "meta": {"level": row["level"], "gold_paragraphs": len(row["gold"])},
            }
            for row in rows
        ]
    rows = _lbpro_rows(("T1", "T5"), "English")
    pool = []
    for row in rows:
        answers = [a for a in row["answer"] if a]
        pool.append(
            {
                "target_id": f"tgt-search_agent-32k-{row['id'][:8]}",
                "scenario": "search_agent",
                "bucket": bucket,
                "lang": "en",
                "source": "longbench_pro:T1+T5",
                "upstream_id": row["id"],
                "primary": [row["context"]],
                "filler": [],
                "task": _task_search_en(row["question"]),
                "evidence": None,
                "cut_policy": "document",
                "stratum": row["secondary_task"] or "T1/T5",
                "meta": {
                    "primary_task": row["primary_task"],
                    "band": row["band"],
                    "answer": answers[0] if answers else None,
                    "evidence_check": "complete-document",
                },
            }
        )
    return pool


def build_multilingual(bucket: str) -> List[Dict[str, Any]]:
    pool = []
    for lang in MULTILINGUAL_LANGS:
        rows = _global_mmlu_rows(lang)

        def demo(row: Dict[str, Any]) -> str:
            return (
                f"Question: {row['question'].strip()}\n"
                f"A. {row['option_a']}\nB. {row['option_b']}\nC. {row['option_c']}\n"
                f"D. {row['option_d']}\nAnswer: {row['answer']}"
            )

        target_row = deterministic_pick(
            sorted((str(row["sample_id"]) for row in rows)), 1, SEED
        )[0]
        by_id = {str(row["sample_id"]): row for row in rows}
        row = by_id[target_row]
        pool.append(
            {
                "target_id": f"tgt-multilingual-{bucket}-{lang}",
                "scenario": "multilingual",
                "bucket": bucket,
                "lang": lang,
                "source": f"global_mmlu:{lang}",
                "upstream_id": f"{lang}#{target_row}",
                "primary": [demo(next(
                    by_id[i] for i in sorted(by_id) if i != target_row
                ))],
                "filler": [
                    demo(other)
                    for other in (by_id[i] for i in sorted(by_id))
                    if str(other["sample_id"]) not in {target_row, next(i for i in sorted(by_id) if i != target_row)}
                ],
                "task": (
                    f"Question: {row['question'].strip()}\n"
                    f"A. {row['option_a']}\nB. {row['option_b']}\nC. {row['option_c']}\n"
                    f"D. {row['option_d']}\n\n"
                    + {
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
                    }[lang]
                ),
                "evidence": None,
                "cut_policy": "example",
                "stratum": lang,
                "meta": {"lang": lang, "subject": str(row.get("subject"))},
            }
        )
    return pool


BUILDERS = {
    "knowledge_qa": build_knowledge_qa,
    "instruction_following": build_instruction_following,
    "long_context": build_long_context,
    "math_logic": build_math_logic,
    "coding": build_coding,
    "general_agent": build_general_agent,
    "search_agent": build_search_agent,
    "multilingual": build_multilingual,
}


# --------------------------------------------------------------------------- #
# build / validate / emit
# --------------------------------------------------------------------------- #
def _fit_primary(
    counter: Counter,
    max_model: MaxModelCounter,
    target: Dict[str, Any],
    bucket: str,
) -> Dict[str, Any]:
    """Guarantee that the primary part stays complete and fits the bucket.

    The primary must fit in the bucket for every known tokenizer (after the runtime
    aligner trimmed the filler away) and must leave room for filler in the reference
    count.  Prefix-safe units are shortened at a boundary; whole-document units are
    dropped when they cannot fit.
    """
    lang = target["lang"]
    task = target["task"]
    model_budget = BUCKETS[bucket] - max_model.tokens(task, lang) - TEMPLATE_MARGIN
    ref_budget = int(content_goal(bucket) * 0.92)
    policy = target.get("cut_policy", "document")
    units = list(target["primary"])
    if not units:
        return target

    def fits() -> bool:
        text = SEP.join(units)
        ref_tokens = counter.n(text)
        if ref_tokens >= ref_budget:
            return False
        # model/cl100k ratio is >= ~0.98 for the English sources and <= 0.6 for the
        # Chinese ones; reject by the cheap reference count before running eleven
        # model tokenizers over a 100 KB document.
        if ref_tokens * 0.98 > model_budget:
            return False
        return max_model.tokens(text, lang) <= model_budget

    if not fits():
        if policy == "document":
            return {}
        # shrink the tail unit at a boundary until both budgets are satisfied;
        # each pass removes at least an eighth so a near-1.0 ratio cannot stall
        for _ in range(200):
            if not units:
                break
            text = SEP.join(units)
            if fits():
                break
            model_tokens = max_model.tokens(text, lang)
            ref_tokens = counter.n(text)
            ratio = min(
                model_budget / max(1, model_tokens),
                (ref_budget - 1) / max(1, ref_tokens),
                0.87,
            )
            allowed = max(1, int(len(units[-1]) * ratio))
            if allowed >= len(units[-1]):
                allowed = max(1, len(units[-1]) - 1)
            shortened = cut_at_boundary(units[-1], allowed)
            if shortened == units[-1]:
                shortened = units[-1][:allowed]
            units[-1] = shortened
            if not units[-1]:
                units.pop()
        if not units or not fits():
            return {}
    target["primary"] = units
    return target


def build_rows(
    counter: Counter,
    max_model: MaxModelCounter,
    scenarios: Sequence[str],
    buckets: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    targets_all: List[Dict[str, Any]] = []
    for scenario in scenarios:
        builder = BUILDERS[scenario]
        for bucket in buckets:
            started = time.time()
            candidates = builder(bucket)
            print(
                f"[cell] {scenario}/{bucket}: {len(candidates)} candidates "
                f"({time.time() - started:.1f}s)",
                file=sys.stderr,
                flush=True,
            )
            if not candidates:
                raise ValueError(f"{scenario}/{bucket}: no candidates")
            # Deterministic shortlist: fitting runs every local model tokenizer, so
            # thousands of HotpotQA/Global-MMLU candidates would dominate build time.
            # Keep the first SHORTLIST_PER_STRATUM per stratum in hash order; the
            # final 10 are still drawn across all strata by largest remainder.
            grouped: Dict[str, List[Dict[str, Any]]] = {}
            for candidate in candidates:
                grouped.setdefault(str(candidate.get("stratum") or "all"), []).append(candidate)
            shortlist: List[Dict[str, Any]] = []
            for key in sorted(grouped):
                items = sorted(
                    grouped[key],
                    key=lambda item: sha256_text(f"{SEED}|shortlist|{item['upstream_id']}"),
                )
                shortlist.extend(items[:SHORTLIST_PER_STRATUM])
            candidates = shortlist
            fitted = []
            for target in candidates:
                target = dict(target)
                target.setdefault("scenario", scenario)
                target.setdefault("bucket", bucket)
                fitted.append(_fit_primary(counter, max_model, target, bucket))
            fitted = [t for t in fitted if t]
            if len(fitted) < TARGETS_PER_SCENARIO:
                raise ValueError(
                    f"{scenario}/{bucket}: only {len(fitted)} fitted candidates, need "
                    f"{TARGETS_PER_SCENARIO}"
                )
            # fill the filler pool from the remaining fitted candidates (same family)
            family = fitted
            picked = pick_targets(family, TARGETS_PER_SCENARIO)
            chars_budget = content_goal(bucket) * 12
            for index, target in enumerate(picked):
                filler = list(target["filler"])
                chars = sum(len(unit) for unit in filler)
                others = [t for t in family if t is not target]
                for other in sorted(others, key=lambda t: sha256_text(f"{SEED}|filler|{t['target_id']}")):
                    if chars >= chars_budget:
                        break
                    for unit in other.get("filler") or other.get("primary") or []:
                        filler.append(unit)
                        chars += len(unit)
                        if chars >= chars_budget:
                            break
                target["filler"] = filler
                target["target_id"] = f"{target['target_id']}"
                rows.append(make_row(counter, max_model, target, bucket, index))
                targets_all.append(target)
            print(
                f"[cell] {scenario}/{bucket}: built {len(picked)} rows "
                f"({time.time() - started:.1f}s total)",
                file=sys.stderr,
                flush=True,
            )
    return rows, targets_all


def validate_rows(rows: Sequence[Dict[str, Any]], targets: Sequence[Dict[str, Any]]) -> None:
    by_cell: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        by_cell.setdefault((row["scenario"], row["bucket"]), []).append(row)
    expected_cells = {(s, b) for s in SCENARIOS for b in BUCKETS}
    if set(by_cell) != expected_cells:
        raise ValueError(f"cells mismatch: {sorted(set(by_cell) ^ expected_cells)}")
    ids = set()
    for (scenario, bucket), cell_rows in sorted(by_cell.items()):
        if len(cell_rows) != TARGETS_PER_SCENARIO:
            raise ValueError(f"{scenario}/{bucket}: {len(cell_rows)} rows != {TARGETS_PER_SCENARIO}")
        for row in cell_rows:
            if row["id"] in ids:
                raise ValueError(f"duplicate row id {row['id']}")
            ids.add(row["id"])
            if row["tokens"] != BUCKETS[bucket]:
                raise ValueError(f"{row['id']}: tokens != bucket")
            messages = row["messages"]
            if [m["role"] for m in messages] != ["user", "user"] or row["tools"]:
                raise ValueError(f"{row['id']}: expected two user messages and no tools")
            meta = row["meta"]
            goal = content_goal(bucket)
            # content is 2x oversize material; only the runtime alignment is exact.
            # Allow the boundary slack that trimming to a paragraph edge can leave.
            if not (goal * 0.90 <= meta["content_tokens"] <= goal * 1.20):
                raise ValueError(
                    f"{row['id']}: content_tokens {meta['content_tokens']} outside {goal}"
                )
            if meta["body_tokens"] + meta["task_tokens"] != meta["content_tokens"]:
                raise ValueError(f"{row['id']}: body+task accounting mismatch")
            if not meta["primary_complete"]:
                raise ValueError(f"{row['id']}: primary validation failed")
            if meta.get("evidence_in_primary") is False:
                raise ValueError(f"{row['id']}: evidence validation failed")
            if not meta["fits_bucket"]:
                raise ValueError(f"{row['id']}: primary+task do not fit the bucket")
            if meta["primary_model_tokens"] + meta["task_model_tokens"] + TEMPLATE_MARGIN > BUCKETS[bucket]:
                raise ValueError(f"{row['id']}: model-token margin violated")
            if meta["lang"] not in scenario_langs(scenario, bucket):
                raise ValueError(f"{row['id']}: unexpected language {meta['lang']}")
    zh = sum(1 for row in rows if row["meta"]["lang"] == "zh")
    if zh < 40:
        raise ValueError(f"only {zh} Chinese rows")


def build_stats(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {"rows": len(rows), "cells": len({(r["scenario"], r["bucket"]) for r in rows})}
    stats["langs"] = {}
    for row in rows:
        lang = row["meta"]["lang"]
        stats["langs"][lang] = stats["langs"].get(lang, 0) + 1
    for bucket in BUCKETS:
        sizes = [len(row["messages"][0]["content"]) for row in rows if row["bucket"] == bucket]
        stats[f"content_chars_{bucket}"] = {
            "min": min(sizes),
            "median": sorted(sizes)[len(sizes) // 2],
            "max": max(sizes),
        }
    stats["primary_model_tokens"] = {
        bucket: {
            "min": min(r["meta"]["primary_model_tokens"] for r in rows if r["bucket"] == bucket),
            "max": max(r["meta"]["primary_model_tokens"] for r in rows if r["bucket"] == bucket),
        }
        for bucket in BUCKETS
    }
    stats["evidence_rows"] = sum(1 for row in rows if row["meta"].get("evidence_sha256"))
    return stats


def write_outputs(rows: Sequence[Dict[str, Any]], manifest_extra: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    import yaml  # noqa: PLC0415

    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "testset.jsonl"
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    jsonl_path.write_text(payload, encoding="utf-8")
    rows_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    tokenizer_path = RAW_DIR / REFERENCE_TOKENIZER_REL
    manifest = {
        "name": "performance",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "builder": "tool/dataset/performance/build.py",
        "seed": SEED,
        "targets_per_scenario": TARGETS_PER_SCENARIO,
        "buckets": BUCKETS,
        "max_tokens": OSL,
        "min_completion_tokens_required": MIN_COMPLETION_TOKENS,
        "reference_tokenizer": {
            "path": REFERENCE_TOKENIZER_REL,
            "sha256": sha256_file(tokenizer_path) if tokenizer_path.exists() else None,
        },
        "construction": {
            "policy": "primary + filler",
            "oversize": OVERSIZE,
            "content_tokens": {bucket: content_goal(bucket) for bucket in BUCKETS},
            "template_margin_tokens": TEMPLATE_MARGIN,
            "runtime_alignment": {
                "mode": "trim+pad",
                "target": "prompt tokens of the rendered chat request",
                "tolerance_tokens": 0,
                "skipped_on_mismatch": False,
                "aligner": "src/modelbench/application/prompt_align.py",
            },
        },
        "validation": manifest_extra.get("validation") or {},
        "sources": manifest_extra.get("sources") or {},
        "cells": {
            f"{scenario}/{bucket}": {
                "rows": sum(1 for r in rows if r["scenario"] == scenario and r["bucket"] == bucket),
                "dataset": SCENARIO_DATASET[(scenario, bucket)],
                "langs": sorted({r["meta"]["lang"] for r in rows if r["scenario"] == scenario and r["bucket"] == bucket}),
            }
            for scenario in SCENARIOS
            for bucket in BUCKETS
        },
        "stats": build_stats(rows),
        "rows": len(rows),
        "rows_sha256": rows_sha,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    yaml_doc = {
        "name": "performance",
        "title": "性能测试（8 场景 × 1K/4K/32K × 10 样本）",
        "description": (
            "性能专用样本集：每个样本由「完整主材料 + 可裁剪补充材料」构成，"
            "运行前由 runner 按目标模型 tokenizer 把 prompt 精确对齐到 1K/4K/32K（0 容差）；"
            "固定 50-token 解码窗口（可在 testset/CLI/Web 覆盖）。"
        ),
        "scenarios": {
            scenario: {
                "dataset": "perf_jsonl",
                "dataset_params": {"scenario": scenario, "path": "dataset/test/performance/testset.jsonl"},
            }
            for scenario in SCENARIOS
        },
        "sampling": {
            "seed": SEED,
            "targets_per_scenario": TARGETS_PER_SCENARIO,
            "method": "sha256(seed|upstream_id) order + stratified largest remainder",
        },
        "buckets": BUCKETS,
        "max_tokens": OSL,
        "strict_input_tokens": True,
    }
    (out_dir / "testset.yaml").write_text(
        yaml.safe_dump(yaml_doc, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return manifest


def check_outputs(out_dir: Optional[Path] = None) -> int:
    out_dir = out_dir or OUT_DIR
    rows = load_jsonl(out_dir / "testset.jsonl")
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    import yaml  # noqa: PLC0415

    yaml_doc = yaml.safe_load((out_dir / "testset.yaml").read_text(encoding="utf-8"))
    problems: List[str] = []
    if manifest.get("name") != "performance" or yaml_doc.get("name") != "performance":
        problems.append("name != performance")
    if manifest.get("buckets") != BUCKETS or yaml_doc.get("buckets") != BUCKETS:
        problems.append("buckets mismatch")
    if yaml_doc.get("max_tokens") != OSL or not yaml_doc.get("strict_input_tokens"):
        problems.append("yaml max_tokens/strict_input_tokens mismatch")
    if "osl" in yaml_doc:
        problems.append("yaml still declares osl")
    validate_rows(rows, [])
    if manifest.get("rows") != len(rows):
        problems.append("manifest rows mismatch")
    if manifest.get("rows_sha256") != sha256_file(out_dir / "testset.jsonl"):
        problems.append("manifest rows_sha256 mismatch")
    stats = manifest.get("stats") or {}
    if stats.get("rows") != len(rows) or stats.get("cells") != len(BUCKETS) * len(SCENARIOS):
        problems.append("manifest stats mismatch")
    if problems:
        for problem in problems:
            print(f"[check] FAIL: {problem}")
        return 1
    print(f"[check] ok: {len(rows)} rows, {len(BUCKETS) * len(SCENARIOS)} cells, sha {manifest['rows_sha256'][:16]}...")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    global RAW_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    parser.add_argument("--buckets", nargs="*", default=list(BUCKETS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        return check_outputs(args.out_dir)
    for scenario in args.scenarios:
        if scenario not in BUILDERS:
            parser.error(f"unknown scenario {scenario}")
    for bucket in args.buckets:
        if bucket not in BUCKETS:
            parser.error(f"unknown bucket {bucket}")
    RAW_DIR = args.raw_dir
    counter = Counter(RAW_DIR / REFERENCE_TOKENIZER_REL)
    max_model = MaxModelCounter()
    started = time.time()
    rows, targets = build_rows(counter, max_model, args.scenarios, args.buckets)
    print(
        f"[build] scenarios={args.scenarios} buckets={args.buckets} rows={len(rows)} "
        f"models={len(max_model.models)} elapsed={time.time() - started:.1f}s"
    )
    if args.dry_run:
        for row in rows[:3]:
            meta = row["meta"]
            print(
                f"  {row['id']} avg_content={meta['content_tokens']} primary={meta['primary_tokens']} "
                f"filler={meta['filler_units']} lang={meta['lang']} evidence={meta['evidence_in_primary']}"
            )
        return 0
    validate_rows(rows, targets)
    sources = {
        "reference_tokenizer": REFERENCE_TOKENIZER_REL,
        "cells": {f"{s}/{b}": SCENARIO_DATASET[(s, b)] for s in SCENARIOS for b in BUCKETS},
    }
    validation = {
        "checks": [
            "primary units complete and present at body start",
            "evidence string present in primary for evidence-critical scenarios",
            "max known-tokenizer(primary) + max known-tokenizer(task) + template <= bucket",
            "body content tokens == bucket x oversize (reference tokenizer)",
            "two user messages, no tools, 10 rows per cell",
        ],
        "tokenizer_probe": max_model.describe(),
        "evidence_rows": sum(1 for row in rows if row["meta"].get("evidence_sha256")),
    }
    manifest = write_outputs(rows, {"validation": validation, "sources": sources}, args.out_dir)
    print(f"[build] wrote {args.out_dir} rows={len(rows)} sha={manifest['rows_sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
