#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the accuracy test set: 8 scenarios x 250 rows in ONE jsonl.

Inputs  (gitignored, fetch with ``tool/dataset/accuracy/fetch.py``):
    dataset/raw/...         upstream files listed by ``raw_spec.json``
Outputs:
    dataset/test/accuracy/testset.jsonl    2,000 rows, self-describing judge contract
    dataset/test/accuracy/manifest.json    sampling / provenance / SHA-256 (no timestamps)
    reports/dataset-build/accuracy/selection_report.md    generated sampling report (intermediate)

The accuracy set is intentionally **hidden from the run/testset selector**: it has
no ``testset.yaml`` and the manifest declares ``status=offline-asset`` /
``runner=not wired``.  It is a data asset only until accuracy evaluation is
separately designed and wired.

    python tool/dataset/accuracy/build.py
    python tool/dataset/accuracy/build.py --check   # verify existing outputs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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
    write_json,
    write_jsonl,
)
from tool.common.tokenizer import TOKENIZER_NAME, default_tokenizer  # noqa: E402
from tool.dataset.accuracy import prompts, sources  # noqa: E402
from tool.dataset.accuracy.sampling import allocate, deterministic_pick  # noqa: E402

SEED = 42
PER_SCENARIO = 250
SCENARIOS = [
    "knowledge_qa",
    "instruction_following",
    "long_context",
    "math_logic",
    "coding",
    "general_agent",
    "search_agent",
    "multilingual",
]
JUDGE_FAMILIES = {
    "mc_letter": "extract option letter A-D/J and compare with the gold letter",
    "ifeval": "official IFEval instruction checkers, prompt-level strict accuracy",
    "normalized_match": "SQuAD-style token F1 or numeric-equivalence exact match",
    "code_exec": "run the sampled unit tests in an isolated subprocess (pass@1)",
    "tool_call": "BFCL possible-answer structural match (function name + argument values)",
}
TEST_DIR_ACC = TEST_DIR / "accuracy"
OUT_REPORT = REPORTS_DIR / "accuracy" / "selection_report.md"
TOKENIZER = default_tokenizer()

#: Global-MMLU en/zh pairing.  Upstream zh/ko ``high_school_world_history`` rows
#: are misordered relative to their ``sample_id`` (176/237 zh answer letters
#: differ from en), so a subject with any en/zh mismatch is dropped from the
#: pairing pool instead of silently pairing two different questions.
ALIGNMENT_MAX_SUBJECT_MISMATCH_RATE = 0.0

#: Known upstream data errors excluded from the *pairing* scenarios, with the
#: reason recorded in the manifest.  Both languages of a pair are excluded: the
#: knowledge scenario contract is "1 question x {en,zh}", so keeping only one
#: side would break the pairing design.
KNOWN_UPSTREAM_ERRORS: Dict[str, str] = {
    "abstract_algebra/test/1": (
        "upstream zh machine translation renders 'index' as '幂' (power), so the "
        "zh row reads gold C=24 as '24th power' and invites a wrong answer"
    ),
}

#: Multilingual scenario languages (en/zh are anchors in knowledge_qa instead).
MULTILINGUAL_LANGUAGES = ["fr", "de", "es", "pt", "ru", "ja", "ko", "ar", "hi", "vi"]

#: LongBench v1 tasks used by the long_context scenario, their language and the
#: official metric.  The Chinese tasks use the official LongBench Chinese
#: scorers: dureader -> ROUGE-L, multifieldqa_zh -> jieba token F1.
LONG_CONTEXT_TASK_LANG = {
    "narrativeqa": "en",
    "qasper": "en",
    "hotpotqa": "en",
    "2wikimqa": "en",
    "multifieldqa_en": "en",
    "dureader": "zh",
    "multifieldqa_zh": "zh",
}
LONG_CONTEXT_TASK_METRIC = {
    "narrativeqa": "f1",
    "qasper": "f1",
    "hotpotqa": "f1",
    "2wikimqa": "f1",
    "multifieldqa_en": "f1",
    "dureader": "rouge_zh",
    "multifieldqa_zh": "f1_zh",
}

#: ``judge.metric`` values understood by ``tool/system/judges.py``.
ALLOWED_METRICS = {"f1", "numeric_exact", "f1_zh", "rouge_zh"}

#: Per-scenario generation budget (design §6.1).  long_context uses the official
#: LongBench ``dataset2maxlen`` and is checked separately.
EXPECTED_MAX_NEW_TOKENS = {
    "knowledge_qa": 1536,
    "multilingual": 1536,
    "instruction_following": 1024,
    "math_logic": 512,
    "coding": 768,
    "general_agent": 256,
    "search_agent": 32,
}

#: ``upstream_id`` is dataset-qualified so it is globally unique.  GSM8K and
#: CMATH both use ``test/<index>`` keys, which collided before this prefix.
UPSTREAM_ID_FORMAT = {
    "knowledge_qa": "<sample_id>|<lang>",
    "instruction_following": "<ifeval key>",
    "long_context": "<task>|<context sha1 prefix>",
    "math_logic": "gsm8k/test/<index> | cmath/test/<index>",
    "coding": "HumanEval/<index> | task_id=<mbpp task_id>",
    "general_agent": "<bfcl request id>",
    "search_agent": "<hotpotqa question id>",
    "multilingual": "<sample_id>|<lang>",
}

#: Differences from the upstream/reference protocols that matter when reading
#: the scores (the set is data-only; nothing is scored by the runner yet).
DEVIATIONS = [
    "Global-MMLU is scored generatively from option letters, not with likelihood-based scoring.",
    "Global-MMLU pairing drops a sample_id when en/zh answer letters disagree, drops every subject whose en/zh mismatch rate is above ALIGNMENT_MAX_SUBJECT_MISMATCH_RATE (upstream high_school_world_history zh/ko row misalignment), and drops known upstream translation errors (abstract_algebra/test/1); the excluded subjects/counts are recorded under sampling.alignment_excluded.",
    "LongBench v1 long-context rows are filtered to 4K-32K prompt tokens (no truncation); 16-32K rows are English-only because Chinese documents are shorter.",
    "LongBench v1 rows are deduplicated by (task, input, context); every acceptable gold answer of a duplicate group is unioned into judge.answers in upstream order (e.g. narrativeqa 'A monk.' / 'a priest').",
    "LongBench Chinese tasks use the official Chinese metrics: dureader -> ROUGE-L (metric=rouge_zh), multifieldqa_zh -> jieba token F1 (metric=f1_zh); the five English tasks keep SQuAD token F1.",
    "CMATH non-integer gold answers carry judge.tolerance=relative_1pct (1% relative tolerance); GSM8K stays exact and has no tolerance field.",
    "MBPP prompts name the entry point extracted from the reference solution and quote the first three official tests; judge.entry_point is set and prompt_prefix stays null.",
    "HotpotQA search_agent is a single-request retrieval QA: the ten distractor paragraphs are in the prompt and no tool loop is executed.",
    "Multilingual is 10 languages x 25 paired questions (not the 42-language Global-MMLU profile); en/zh are covered by knowledge_qa instead. Candidates require identical answer letters in all 10 languages and four distinct non-empty options in every language; filtered rows are recorded under sampling.alignment_excluded.",
    "Instruction following uses IFEval; the performance set's reference pool uses the same IFEval source but is never scored.",
    "Coding uses HumanEval + MBPP (not LiveCodeBench); both are easier sets chosen for 1B-class models.",
    "Every row carries judge.max_new_tokens (knowledge/multilingual 1536, instruction 1024, math 512, coding 768, general_agent 256, search 32, long_context the official task maxlen).",
    "Accuracy is a hidden offline data asset: no evaluator is wired into the run and no score may be quoted from it.",
]


def _tokens(text: str) -> int:
    return TOKENIZER.count(text)


def make_row(scenario: str, dataset: str, language: str, upstream_id: str,
             prompt: str, judge: Dict[str, Any], meta: Dict[str, Any],
             tokens: int | None = None) -> Dict[str, Any]:
    return {
        "id": f"{scenario}:{dataset}:{upstream_id}",
        "scenario": scenario,
        "dataset": dataset,
        "language": language,
        "upstream_id": upstream_id,
        "input_tokens": tokens if tokens is not None else _tokens(prompt),
        "messages": [{"role": "user", "content": prompt}],
        "judge": judge,
        "meta": meta,
    }


def _token_stats(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    vals = [r["input_tokens"] for r in rows]
    return {
        "min": min(vals),
        "median": int(statistics.median(vals)),
        "max": max(vals),
        "total": sum(vals),
    }


# ---------------------------------------------------------------- pure helpers


def _mc_row_is_valid(data: Dict[str, List[Any]], i: int) -> bool:
    """True when row ``i`` is a well-formed 4-choice question with an A-D answer."""
    return (
        str(data["answer"][i] or "") in {"A", "B", "C", "D"}
        and all(
            str(data[k][i] or "").strip()
            for k in ("question", "option_a", "option_b", "option_c", "option_d")
        )
    )


def _pair_aligned_sample_ids(
    en_answers: Dict[str, str],
    zh_answers: Dict[str, str],
    subject_by_id: Dict[str, str],
    *,
    max_mismatch_rate: float = ALIGNMENT_MAX_SUBJECT_MISMATCH_RATE,
) -> Tuple[List[str], Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """Pair Global-MMLU en/zh rows by ``sample_id`` when the gold letters agree.

    Returns ``(aligned_ids, subject_stats, excluded_subjects)``.

    ``subject_stats`` records, per subject, the pair count, mismatching pair count
    and mismatch rate.  A subject is removed from the pairing pool entirely when
    its mismatch rate exceeds ``max_mismatch_rate``: upstream zh/ko
    ``high_school_world_history`` rows are misordered, so every row of that
    subject is a different question from its en counterpart even though the
    answer letters sometimes agree by chance.
    """
    common = [sid for sid in en_answers if sid in zh_answers]
    pairs: Counter = Counter()
    mismatches: Counter = Counter()
    for sid in common:
        subject = subject_by_id[sid]
        pairs[subject] += 1
        if en_answers[sid] != zh_answers[sid]:
            mismatches[subject] += 1

    subject_stats: Dict[str, Dict[str, Any]] = {}
    excluded: List[Dict[str, Any]] = []
    for subject in sorted(pairs):
        total = pairs[subject]
        bad = mismatches[subject]
        rate = bad / total if total else 0.0
        subject_stats[subject] = {
            "pairs": total,
            "mismatches": bad,
            "mismatch_rate": round(rate, 4),
        }
        if rate > max_mismatch_rate:
            excluded.append({
                "subject": subject,
                "count": total,
                "reason": (
                    f"en/zh answer mismatch rate {rate:.1%} > "
                    f"{max_mismatch_rate:.0%}; upstream rows are misaligned by sample_id"
                ),
            })

    dropped = {item["subject"] for item in excluded}
    aligned = [
        sid
        for sid in common
        if subject_by_id[sid] not in dropped and en_answers[sid] == zh_answers[sid]
    ]
    return aligned, subject_stats, excluded


def _union_answers(first: Iterable[str], second: Iterable[str]) -> List[str]:
    """Order-preserving union of acceptable gold answers (no duplicates)."""
    merged = list(first)
    for answer in second:
        if answer not in merged:
            merged.append(answer)
    return merged


def _longbench_groups(task: str, templates: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Group one LongBench task's rows by rendered prompt and union their answers.

    LongBench v1 has 20 duplicate ``(task, prompt)`` groups; 18 of them carry
    different acceptable answers (narrativeqa: "A monk." vs "a priest").  The
    representative row is the first occurrence, but ``answers`` keeps every
    upstream answer in file order so the F1/ROUGE scorer can match any of them.
    """
    groups: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(RAW_DIR / f"longbench_v1/data/{task}.jsonl"):
        prompt = templates[task].format(context=row["context"], input=row["input"])
        key = hashlib.sha256((task + "\x00" + prompt).encode("utf-8")).hexdigest()
        group = groups.get(key)
        if group is None:
            groups[key] = {
                "key": key,
                "prompt": prompt,
                "answers": list(row["answers"]),
                "first_answers": len(row["answers"]),
                "source_rows": 1,
            }
        else:
            group["answers"] = _union_answers(group["answers"], row["answers"])
            group["source_rows"] += 1
    return groups


def _multilingual_options_ok(options: Iterable[str]) -> bool:
    """True when a 4-choice row has four non-empty pairwise-distinct options."""
    cleaned = [str(option or "").strip() for option in options]
    return len(cleaned) == 4 and all(cleaned) and len(set(cleaned)) == 4


_MC_OPTION_MARKER = re.compile(r"\n([ABCD])\. ")


def _mc_options_from_prompt(prompt: str) -> List[str]:
    """Parse the A-D option texts back out of a rendered multiple-choice prompt."""
    markers = list(_MC_OPTION_MARKER.finditer(prompt))
    for start in range(len(markers) - 3):
        window = markers[start:start + 4]
        if [marker.group(1) for marker in window] != ["A", "B", "C", "D"]:
            continue
        options: List[str] = []
        for pos, marker in enumerate(window):
            end = window[pos + 1].start() if pos < 3 else len(prompt)
            options.append(prompt[marker.end():end])
        for suffix in ("\n\nAnswer:", "\n\n答案："):
            if options[-1].endswith(suffix):
                options[-1] = options[-1][: -len(suffix)]
                break
        else:
            continue
        return [option.strip() for option in options]
    raise ValueError("could not parse A-D options from prompt")


# ---------------------------------------------------------------- scenario builders


def build_knowledge_qa() -> Tuple[List[Dict[str, Any]], Dict[str, Any], set]:
    import pyarrow.parquet as pq  # noqa: PLC0415

    en = pq.read_table(RAW_DIR / "global_mmlu/en/test.parquet").to_pydict()
    zh = pq.read_table(RAW_DIR / "global_mmlu/zh/test.parquet").to_pydict()
    en_index = {sid: i for i, sid in enumerate(en["sample_id"])}
    zh_index = {sid: i for i, sid in enumerate(zh["sample_id"])}
    assert set(en_index) == set(zh_index), "Global-MMLU en/zh sample_ids are not aligned"

    subject_by_id = {sid: en["subject"][en_index[sid]] for sid in en["sample_id"]}
    en_answers = {sid: str(en["answer"][en_index[sid]] or "") for sid in en["sample_id"]}
    zh_answers = {sid: str(zh["answer"][zh_index[sid]] or "") for sid in zh["sample_id"]}

    # Upstream zh/ko high_school_world_history is misordered by sample_id, so
    # pairing on the id alone is unsafe.  Drop mismatching sample_ids and every
    # subject whose mismatch rate exceeds the threshold, with recorded reasons.
    aligned, subject_alignment, alignment_excluded = _pair_aligned_sample_ids(
        en_answers, zh_answers, subject_by_id
    )

    # A pair is only usable when both rows are structurally valid (four non-empty
    # options, A-D answer); e.g. virology/test/4 has a blank option D in zh.
    valid: List[str] = []
    structural_bad: Counter = Counter()
    for sid in aligned:
        if _mc_row_is_valid(en, en_index[sid]) and _mc_row_is_valid(zh, zh_index[sid]):
            valid.append(sid)
        else:
            structural_bad[subject_by_id[sid]] += 1
    for subject, count in sorted(structural_bad.items()):
        alignment_excluded.append({
            "subject": subject,
            "count": count,
            "reason": "en and/or zh row has an answer outside A-D or an empty option",
        })

    # Documented upstream translation errors break the prompt/gold contract for
    # both languages of the pair.
    known_bad = [sid for sid in valid if sid in KNOWN_UPSTREAM_ERRORS]
    for sid in known_bad:
        alignment_excluded.append({
            "subject": subject_by_id[sid],
            "sample_id": sid,
            "count": 1,
            "reason": KNOWN_UPSTREAM_ERRORS[sid],
        })
    valid = [sid for sid in valid if sid not in KNOWN_UPSTREAM_ERRORS]
    subject = {sid: subject_by_id[sid] for sid in valid}
    quota = allocate(Counter(subject[s] for s in valid), 125, min_each=2, seed=SEED)

    selected: List[str] = []
    for sub in sorted(quota):
        selected.extend(deterministic_pick([s for s in valid if subject[s] == sub], quota[sub], SEED))
    assert len(selected) == 125 and len(set(selected)) == 125

    rows = []
    for sid in sorted(selected):
        for lang in ("en", "zh"):
            src = en if lang == "en" else zh
            i = en_index[sid] if lang == "en" else zh_index[sid]
            template = prompts.MC_EN if lang == "en" else prompts.MC_ZH
            text = template.format(
                q=src["question"][i], a=src["option_a"][i], b=src["option_b"][i],
                c=src["option_c"][i], d=src["option_d"][i],
            )
            rows.append(make_row(
                "knowledge_qa", "global_mmlu", lang, f"{sid}|{lang}", text,
                {
                    "family": "mc_letter",
                    "answer": src["answer"][i],
                    "max_new_tokens": EXPECTED_MAX_NEW_TOKENS["knowledge_qa"],
                },
                {
                    "sample_id": sid,
                    "subject": src["subject"][i],
                    "subject_category": src["subject_category"][i],
                    "stratum": f"subject={src['subject'][i]}|lang={lang}",
                },
            ))
    stats = {
        "pool": len(valid),
        "pool_before_alignment": len(en["sample_id"]),
        "selected_ids": len(selected),
        "subjects": len({subject[s] for s in selected}),
        "subjects_per_category": dict(Counter(en["subject_category"][en_index[s]] for s in selected)),
        "quotas": quota,
        "alignment": {
            "max_subject_mismatch_rate": ALIGNMENT_MAX_SUBJECT_MISMATCH_RATE,
            "subjects": subject_alignment,
        },
        "alignment_excluded": alignment_excluded,
    }
    return rows, stats, set(selected)


def build_instruction_following() -> Tuple[List[Dict[str, Any]], Dict[str, Any], set]:
    rows_in = read_jsonl(RAW_DIR / "ifeval/ifeval_input_data.jsonl")
    usable = [r for r in rows_in if not any(i.startswith("language:") for i in r["instruction_id_list"])]
    strata = Counter(r["instruction_id_list"][0].split(":")[0] for r in usable)
    quota = allocate(strata, PER_SCENARIO, min_each=1, seed=SEED)

    rows = []
    for typ in sorted(quota):
        pool = [r for r in usable if r["instruction_id_list"][0].split(":")[0] == typ]
        for r in deterministic_pick([str(x["key"]) for x in pool], quota[typ], SEED):
            item = next(x for x in pool if str(x["key"]) == r)
            rows.append(make_row(
                "instruction_following", "ifeval", "en", str(item["key"]), item["prompt"],
                {
                    "family": "ifeval",
                    "instruction_id_list": item["instruction_id_list"],
                    "kwargs": item["kwargs"],
                    "max_new_tokens": EXPECTED_MAX_NEW_TOKENS["instruction_following"],
                },
                {"type": typ, "n_instructions": len(item["instruction_id_list"]), "stratum": f"type={typ}"},
            ))
    stats = {
        "pool": len(usable),
        "excluded_language_instructions": len(rows_in) - len(usable),
        "strata": dict(strata),
        "quotas": quota,
    }
    return rows, stats, set()


def build_long_context() -> Tuple[List[Dict[str, Any]], Dict[str, Any], set]:
    templates = read_json(RAW_DIR / "longbench_v1/prompts/dataset2prompt.json")
    maxlen = read_json(RAW_DIR / "longbench_v1/prompts/dataset2maxlen.json")
    targets = {
        ("en", "4-8k"): 50, ("zh", "4-8k"): 50,
        ("en", "8-16k"): 55, ("zh", "8-16k"): 50,
        ("en", "16-32k"): 45,
    }
    pool: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    duplicate_groups = 0
    duplicate_answer_unions = 0
    for task, lang in LONG_CONTEXT_TASK_LANG.items():
        # Group first so duplicate prompts union every upstream acceptable answer;
        # the representative row and therefore the sampled set stay unchanged.
        for group in _longbench_groups(task, templates).values():
            if group["source_rows"] > 1:
                duplicate_groups += 1
                if len(group["answers"]) > group["first_answers"]:
                    duplicate_answer_unions += 1
            n = len(TOKENIZER.encode(group["prompt"]))
            bucket = (
                "4-8k" if 4096 <= n < 8192
                else "8-16k" if 8192 <= n < 16384
                else "16-32k" if 16384 <= n <= 32768
                else None
            )
            if bucket is None:
                continue
            pool[(lang, task, bucket)].append({
                "key": group["key"],
                "prompt": group["prompt"],
                "answers": group["answers"],
                "source_rows": group["source_rows"],
                "tokens": n,
            })

    rows = []
    per_cell = Counter()
    tasks_selected = Counter()
    unioned_selected = 0
    for (lang, bucket), target in sorted(targets.items()):
        tasks = [t for t, l in LONG_CONTEXT_TASK_LANG.items() if l == lang]
        counts = {t: len(pool[(lang, t, bucket)]) for t in tasks}
        if sum(counts.values()) < target:
            raise ValueError(f"long_context cell {lang}/{bucket} pool {counts} < {target}")
        quota = allocate(counts, target, min_each=0, seed=SEED)
        for task in sorted(quota):
            by_key = {item["key"]: item for item in pool[(lang, task, bucket)]}
            for key in deterministic_pick(list(by_key), quota[task], SEED):
                item = by_key[key]
                if item["source_rows"] > 1:
                    unioned_selected += 1
                rows.append(make_row(
                    "long_context", "longbench_v1", lang, f"{task}|{key[:16]}", item["prompt"],
                    {
                        "family": "normalized_match",
                        "metric": LONG_CONTEXT_TASK_METRIC[task],
                        "answers": item["answers"],
                        "max_new_tokens": int(maxlen.get(task, 64)),
                    },
                    {"task": task, "length_bucket": bucket, "stratum": f"task={task}|bucket={bucket}"},
                    tokens=item["tokens"],
                ))
                per_cell[(lang, bucket)] += 1
                tasks_selected[task] += 1
    stats = {
        "pool": {f"{lang}/{task}/{bucket}": len(v) for (lang, task, bucket), v in sorted(pool.items())},
        "targets": {f"{lang}/{bucket}": v for (lang, bucket), v in sorted(targets.items())},
        "selected_buckets": {f"{lang}/{bucket}": v for (lang, bucket), v in sorted(per_cell.items())},
        "selected_tasks": dict(tasks_selected),
        "duplicate_prompt_groups": duplicate_groups,
        "duplicate_answer_unions": duplicate_answer_unions,
        "selected_rows_with_unioned_answers": unioned_selected,
        "task_metrics": LONG_CONTEXT_TASK_METRIC,
    }
    return rows, stats, set()


def build_math_logic() -> Tuple[List[Dict[str, Any]], Dict[str, Any], set]:
    import pyarrow.parquet as pq  # noqa: PLC0415

    gsm = pq.read_table(RAW_DIR / "gsm8k/main/test-00000-of-00001.parquet").to_pydict()
    cmath = read_jsonl(RAW_DIR / "cmath/cmath_test.jsonl")

    def step_bucket(answer: str) -> str:
        n = len(re.findall(r"<<", answer))
        return "0-1" if n <= 1 else "2" if n == 2 else "3" if n == 3 else "4" if n == 4 else "5+"

    rows = []
    gsm_pool = {i: gsm["question"][i].strip() for i in range(len(gsm["question"])) if gsm["question"][i].strip()}
    gsm_strata = Counter(step_bucket(gsm["answer"][i]) for i in gsm_pool)
    gsm_quota = allocate(gsm_strata, 125, min_each=1, seed=SEED)
    for bucket in sorted(gsm_quota):
        ids = [i for i in gsm_pool if step_bucket(gsm["answer"][i]) == bucket]
        for raw_id in deterministic_pick([str(i) for i in ids], gsm_quota[bucket], SEED):
            i = int(raw_id)
            final = prompts.gsm8k_final(gsm["answer"][i])
            rows.append(make_row(
                "math_logic", "gsm8k", "en", f"gsm8k/test/{i}",
                prompts.GSM8K_PROMPT.format(q=gsm_pool[i]),
                {
                    "family": "normalized_match",
                    "metric": "numeric_exact",
                    "answers": [final],
                    "max_new_tokens": EXPECTED_MAX_NEW_TOKENS["math_logic"],
                },
                {"step_bucket": bucket, "stratum": f"step={bucket}"},
            ))

    cmath_strata = Counter(str(r["grade"]) for r in cmath)
    cmath_quota = allocate(cmath_strata, 125, min_each=1, seed=SEED)
    for grade in sorted(cmath_quota):
        ids = [i for i, r in enumerate(cmath) if str(r["grade"]) == grade]
        for raw_id in deterministic_pick([str(i) for i in ids], cmath_quota[grade], SEED):
            i = int(raw_id)
            item = cmath[i]
            rows.append(make_row(
                "math_logic", "cmath", "zh", f"cmath/test/{i}",
                prompts.CMATH_PROMPT.format(q=item["question"]),
                {
                    "family": "normalized_match",
                    "metric": "numeric_exact",
                    "answers": [prompts.cmath_final(item["golden"])],
                    # Non-integer CMATH gold (fractions/decimals) is accepted with
                    # a 1% relative tolerance; integer gold stays exact.
                    "tolerance": "relative_1pct",
                    "max_new_tokens": EXPECTED_MAX_NEW_TOKENS["math_logic"],
                },
                {"grade": int(grade), "stratum": f"grade={grade}"},
            ))
    stats = {
        "pool": {"gsm8k": len(gsm_pool), "cmath": len(cmath)},
        "gsm8k_strata": dict(gsm_strata),
        "gsm8k_quotas": gsm_quota,
        "cmath_strata": dict(cmath_strata),
        "cmath_quotas": cmath_quota,
    }
    return rows, stats, set()


def _quartile_strata(lengths: List[int]):
    ordered = sorted(lengths)
    q1, q2, q3 = ordered[len(ordered) // 4], ordered[len(ordered) // 2], ordered[3 * len(ordered) // 4]

    def which(n: int) -> str:
        return "q1" if n <= q1 else "q2" if n <= q2 else "q3" if n <= q3 else "q4"

    return which


def build_coding() -> Tuple[List[Dict[str, Any]], Dict[str, Any], set]:
    import pyarrow.parquet as pq  # noqa: PLC0415

    humaneval = pq.read_table(RAW_DIR / "humaneval/openai_humaneval/test-00000-of-00001.parquet").to_pydict()
    mbpp = read_json(RAW_DIR / "mbpp/sanitized-mbpp.json")
    rows = []

    he_len = {i: len(humaneval["prompt"][i]) for i in range(len(humaneval["prompt"]))}
    which = _quartile_strata(list(he_len.values()))
    he_quota = allocate(Counter(which(n) for n in he_len.values()), 125, min_each=1, seed=SEED)
    for quartile in sorted(he_quota):
        ids = [i for i in he_len if which(he_len[i]) == quartile]
        for raw_id in deterministic_pick([str(i) for i in ids], he_quota[quartile], SEED):
            i = int(raw_id)
            test_code = humaneval["test"][i] + f"\n\ncheck({humaneval['entry_point'][i]})\n"
            rows.append(make_row(
                "coding", "humaneval", "en", humaneval["task_id"][i],
                prompts.HUMANEVAL_PROMPT.format(prompt=humaneval["prompt"][i]),
                {
                    "family": "code_exec",
                    "prompt_prefix": humaneval["prompt"][i],
                    "test_code": test_code,
                    "entry_point": humaneval["entry_point"][i],
                    "timeout_s": 6,
                    "max_new_tokens": EXPECTED_MAX_NEW_TOKENS["coding"],
                },
                {"quartile": quartile, "stratum": f"prompt_len={quartile}"},
            ))

    # MBPP sanitized task descriptions do not name the function under test, so
    # render prompts from the reference solution's entry point plus the first
    # official tests, and stratify by the *rendered* prompt length (token count,
    # consistent with input_tokens) instead of the raw task sentence.
    mbpp_items = []
    for item in mbpp:
        entry_point = prompts.mbpp_entry_point(item["code"])
        text = prompts.mbpp_prompt(item["prompt"], entry_point, item["test_list"])
        mbpp_items.append({
            "item": item,
            "entry_point": entry_point,
            "prompt": text,
            "tokens": _tokens(text),
        })
    mb_len = {i: prepared["tokens"] for i, prepared in enumerate(mbpp_items)}
    which_mb = _quartile_strata(list(mb_len.values()))
    mb_quota = allocate(Counter(which_mb(n) for n in mb_len.values()), 125, min_each=1, seed=SEED)
    mbpp_rows: List[Dict[str, Any]] = []
    for quartile in sorted(mb_quota):
        ids = [i for i in mb_len if which_mb(mb_len[i]) == quartile]
        for raw_id in deterministic_pick([str(i) for i in ids], mb_quota[quartile], SEED):
            prepared = mbpp_items[int(raw_id)]
            item = prepared["item"]
            test_code = "\n".join(list(item.get("test_imports") or []) + list(item["test_list"])) + "\n"
            mbpp_rows.append(make_row(
                "coding", "mbpp", "en", f"task_id={item['task_id']}",
                prepared["prompt"],
                {
                    "family": "code_exec",
                    "prompt_prefix": None,
                    "test_code": test_code,
                    "entry_point": prepared["entry_point"],
                    "timeout_s": 6,
                    "max_new_tokens": EXPECTED_MAX_NEW_TOKENS["coding"],
                },
                {"quartile": quartile, "stratum": f"prompt_len={quartile}"},
                tokens=prepared["tokens"],
            ))
    rows.extend(mbpp_rows)

    # Post-condition for the 2026-11 audit defect: every MBPP row must tell the
    # model which function name the unit tests call.
    assert len(mbpp_rows) == 125, len(mbpp_rows)
    for row in mbpp_rows:
        entry_point = row["judge"]["entry_point"]
        assert entry_point and entry_point in row["messages"][0]["content"], row["id"]

    stats = {
        "pool": {"humaneval": len(humaneval["prompt"]), "mbpp": len(mbpp)},
        "humaneval_quotas": he_quota,
        "mbpp_quotas": mb_quota,
        "mbpp_quartile_basis": "rendered prompt tokens (template + entry point + 3 official tests)",
        "mbpp_entry_points": len({prepared["entry_point"] for prepared in mbpp_items}),
    }
    return rows, stats, set()


def _normalise_json_schema(node: Any) -> Any:
    type_map = {"dict": "object", "float": "number", "tuple": "array", "list": "array",
                "str": "string", "int": "integer", "bool": "boolean"}
    if isinstance(node, dict):
        return {k: (type_map[v] if k == "type" and isinstance(v, str) and v in type_map else _normalise_json_schema(v))
                for k, v in node.items()}
    if isinstance(node, list):
        return [_normalise_json_schema(v) for v in node]
    return node


def build_general_agent() -> Tuple[List[Dict[str, Any]], Dict[str, Any], set]:
    categories = [
        "simple_python", "simple_java", "simple_javascript", "multiple", "parallel",
        "parallel_multiple", "irrelevance", "live_simple", "live_multiple",
        "live_parallel", "live_parallel_multiple",
    ]
    pools: Dict[str, List[Dict[str, Any]]] = {}
    for category in categories:
        requests = read_jsonl(RAW_DIR / f"bfcl/BFCL_v4_{category}.json")
        answers = {}
        answer_path = RAW_DIR / f"bfcl/possible_answer/BFCL_v4_{category}.json"
        if answer_path.is_file():
            for item in read_jsonl(answer_path):
                answers[item["id"]] = item["ground_truth"]
        for item in requests:
            # irrelevance has no possible-answer file: empty ground truth = "do not call".
            item["_ground_truth"] = answers.get(item["id"], [])
        pools[category] = requests
    quota = allocate({c: len(pools[c]) for c in categories}, PER_SCENARIO, min_each=12, seed=SEED)

    rows = []
    for category in categories:
        by_id = {item["id"]: item for item in pools[category]}
        for sid in deterministic_pick(list(by_id), quota[category], SEED):
            item = by_id[sid]
            functions = _normalise_json_schema(item.get("function") or [])
            turns: List[str] = []
            for turn in item["question"]:
                for message in turn:
                    if message.get("role") == "user":
                        turns.append(message.get("content", ""))
            question = turns[-1] if turns else ""
            text = prompts.BFCL_PROMPT.format(
                functions=json.dumps(functions, ensure_ascii=False), question=question
            )
            rows.append(make_row(
                "general_agent", "bfcl_v4", "en", sid, text,
                {
                    "family": "tool_call",
                    "ground_truth": item["_ground_truth"],
                    "functions": functions,
                    "max_new_tokens": EXPECTED_MAX_NEW_TOKENS["general_agent"],
                },
                {"category": category, "n_functions": len(functions), "stratum": f"category={category}"},
            ))
    stats = {"pool": {c: len(pools[c]) for c in categories}, "quotas": quota}
    return rows, stats, set()


def build_search_agent() -> Tuple[List[Dict[str, Any]], Dict[str, Any], set]:
    import pyarrow.parquet as pq  # noqa: PLC0415

    data = pq.read_table(RAW_DIR / "hotpotqa/distractor/validation-00000-of-00001.parquet").to_pydict()
    items = []
    seen_questions = set()
    for i in range(len(data["question"])):
        question = data["question"][i].strip()
        if not question or question in seen_questions:
            continue
        seen_questions.add(question)
        paragraphs = []
        context = data["context"][i]
        for title, sentences in zip(context["title"], context["sentences"]):
            paragraphs.append(f"[{len(paragraphs) + 1}] {title}: " + "".join(sentences))
        items.append({
            "i": i,
            "question": question,
            "prompt": prompts.SEARCH_PROMPT.format(
                paragraphs="\n".join(paragraphs), question=question
            ),
            "stratum": f"{data['type'][i]}|hop={min(len(data['supporting_facts'][i]['title']), 4)}",
        })
    quota = allocate(Counter(item["stratum"] for item in items), PER_SCENARIO, min_each=1, seed=SEED)

    rows = []
    for stratum in sorted(quota):
        candidates = [item for item in items if item["stratum"] == stratum]
        by_question = {item["question"]: item for item in candidates}
        assert len(by_question) >= quota[stratum]
        for question in deterministic_pick(list(by_question), quota[stratum], SEED):
            item = by_question[question]
            i = item["i"]
            rows.append(make_row(
                "search_agent", "hotpotqa_distractor", "en", data["id"][i], item["prompt"],
                {
                    "family": "normalized_match",
                    "metric": "f1",
                    "answers": [data["answer"][i]],
                    "max_new_tokens": EXPECTED_MAX_NEW_TOKENS["search_agent"],
                },
                {
                    "type": data["type"][i],
                    "n_supporting_facts": len(data["supporting_facts"][i]["title"]),
                    "stratum": item["stratum"],
                },
            ))
    stats = {"pool": len(items), "strata": dict(Counter(item["stratum"] for item in items)), "quotas": quota}
    return rows, stats, set()


def build_multilingual(knowledge_ids: set) -> Tuple[List[Dict[str, Any]], Dict[str, Any], set]:
    import pyarrow.parquet as pq  # noqa: PLC0415

    en = pq.read_table(RAW_DIR / "global_mmlu/en/test.parquet").to_pydict()
    en_index = {sid: i for i, sid in enumerate(en["sample_id"])}
    subject_by_id = {sid: en["subject"][en_index[sid]] for sid in en["sample_id"]}
    category_by_subject = {
        subject_by_id[sid]: en["subject_category"][en_index[sid]] for sid in en["sample_id"]
    }

    # Load every multilingual translation once: candidate filtering needs all 10
    # answer letters and all 10 option sets for each candidate sample_id.
    language_data: Dict[str, Tuple[Dict[str, List[Any]], Dict[str, int]]] = {}
    for language in MULTILINGUAL_LANGUAGES:
        data = pq.read_table(RAW_DIR / f"global_mmlu/{language}/test.parquet").to_pydict()
        lang_index = {sid: i for i, sid in enumerate(data["sample_id"])}
        assert set(lang_index) == set(en_index), f"Global-MMLU {language} sample_ids are not aligned"
        language_data[language] = (data, lang_index)

    def _candidate_exclusion(sid: str) -> str | None:
        """Return the exclusion reason when a candidate fails cross-language checks."""
        answers = set()
        for language in MULTILINGUAL_LANGUAGES:
            data, lang_index = language_data[language]
            answers.add(str(data["answer"][lang_index[sid]] or ""))
        if len(answers) != 1:
            return "answers differ across the 10 languages"
        for language in MULTILINGUAL_LANGUAGES:
            data, lang_index = language_data[language]
            i = lang_index[sid]
            options = [data[k][i] for k in ("option_a", "option_b", "option_c", "option_d")]
            if not _multilingual_options_ok(options):
                return f"options collapse to <4 distinct non-empty choices in {language}"
        return None

    # Build the candidate pool per subject, excluding knowledge_qa ids (mutual
    # exclusion) and rows that fail the cross-language alignment/option checks.
    eligible: Dict[str, List[str]] = defaultdict(list)
    excluded_reasons: Counter = Counter()
    for sid in en["sample_id"]:
        if sid in knowledge_ids:
            continue
        if not _mc_row_is_valid(en, en_index[sid]):
            continue
        reason = _candidate_exclusion(sid)
        if reason is None:
            eligible[subject_by_id[sid]].append(sid)
        else:
            excluded_reasons[(subject_by_id[sid], reason)] += 1

    # Re-allocate the 25-question quota over the *post-filter* category pools so a
    # category that lost candidates (e.g. every ko row of a subject collapsing)
    # does not silently stay over-weighted.
    eligible_subjects = {subject: sorted(ids) for subject, ids in eligible.items()}
    category_pool: Counter = Counter()
    for subject, ids in eligible_subjects.items():
        category_pool[category_by_subject[subject]] += len(ids)
    category_quota = allocate(category_pool, 25, min_each=1, seed=SEED)

    chosen: List[Tuple[str, str]] = []
    for category in sorted(category_quota):
        quota = category_quota[category]
        candidates = sorted(
            subject for subject in eligible_subjects
            if category_by_subject[subject] == category
        )
        if quota > len(candidates):
            raise ValueError(
                f"multilingual category {category}: quota {quota} > subjects {len(candidates)}"
            )
        candidates.sort(key=lambda s: hashlib.sha256(f"{SEED}|subject|{s}".encode()).hexdigest())
        for subject in candidates[:quota]:
            chosen.append((subject, deterministic_pick(eligible_subjects[subject], 1, SEED)[0]))
    assert len(chosen) == 25, len(chosen)
    assert len({sid for _, sid in chosen}) == 25

    rows = []
    for language in MULTILINGUAL_LANGUAGES:
        data, lang_index = language_data[language]
        for _subject, sid in chosen:
            i = lang_index[sid]
            text = prompts.MC_EN.format(
                q=data["question"][i], a=data["option_a"][i], b=data["option_b"][i],
                c=data["option_c"][i], d=data["option_d"][i],
            )
            rows.append(make_row(
                "multilingual", "global_mmlu", language, f"{sid}|{language}", text,
                {
                    "family": "mc_letter",
                    "answer": data["answer"][i],
                    "max_new_tokens": EXPECTED_MAX_NEW_TOKENS["multilingual"],
                },
                {
                    "sample_id": sid,
                    "subject": data["subject"][i],
                    "subject_category": data["subject_category"][i],
                    "stratum": f"lang={language}|category={data['subject_category'][i]}",
                },
            ))
    stats = {
        "pool": {
            "languages": MULTILINGUAL_LANGUAGES,
            "rows_per_language": len(en["sample_id"]),
            "eligible_sample_ids": sum(len(ids) for ids in eligible_subjects.values()),
            "eligible_by_category": dict(category_pool),
        },
        "unique_question_ids": len({sid for _, sid in chosen}),
        "languages": MULTILINGUAL_LANGUAGES,
        "category_quotas": category_quota,
        "alignment_excluded": [
            {"subject": subject, "count": count, "reason": reason}
            for (subject, reason), count in sorted(excluded_reasons.items())
        ],
    }
    return rows, stats, set()


# ---------------------------------------------------------------- outputs


def build_outputs() -> Dict[str, Any]:
    rows_by_scenario: Dict[str, List[Dict[str, Any]]] = {}
    stats: Dict[str, Any] = {}
    knowledge_ids: set = set()
    builders = {
        "knowledge_qa": build_knowledge_qa,
        "instruction_following": build_instruction_following,
        "long_context": build_long_context,
        "math_logic": build_math_logic,
        "coding": build_coding,
        "general_agent": build_general_agent,
        "search_agent": build_search_agent,
    }
    for name, builder in builders.items():
        rows, scenario_stats, ids = builder()
        if name == "knowledge_qa":
            knowledge_ids = ids
        rows.sort(key=lambda r: r["id"])
        rows_by_scenario[name] = rows
        stats[name] = scenario_stats

    rows, scenario_stats, _ = build_multilingual(knowledge_ids)
    rows.sort(key=lambda r: r["id"])
    rows_by_scenario["multilingual"] = rows
    stats["multilingual"] = scenario_stats

    combined: List[Dict[str, Any]] = []
    for scenario in SCENARIOS:
        scenario_rows = rows_by_scenario[scenario]
        if len(scenario_rows) != PER_SCENARIO:
            raise AssertionError(f"{scenario}: {len(scenario_rows)} rows != {PER_SCENARIO}")
        if len({r["id"] for r in scenario_rows}) != PER_SCENARIO:
            raise AssertionError(f"{scenario}: duplicate ids")
        combined.extend(scenario_rows)

    TEST_DIR_ACC.mkdir(parents=True, exist_ok=True)
    testset_path = TEST_DIR_ACC / "testset.jsonl"
    write_jsonl(testset_path, combined)

    scenario_manifest = {}
    for scenario in SCENARIOS:
        scenario_rows = rows_by_scenario[scenario]
        scenario_manifest[scenario] = {
            "rows": len(scenario_rows),
            "datasets": dict(Counter(r["dataset"] for r in scenario_rows)),
            "languages": dict(Counter(r["language"] for r in scenario_rows)),
            "judges": dict(Counter(r["judge"]["family"] for r in scenario_rows)),
            "input_tokens": _token_stats(scenario_rows),
            "sampling": stats[scenario],
        }

    tokenizer_spec = next(f for f in sources.files_for("tokenizer") if f["path"].endswith("tokenizer.json"))
    manifest = {
        "schema_version": "accuracy/v1",
        "name": "accuracy",
        "description": "8 scenarios x 250 rows, small-model accuracy set (bf16 vs quantized); hidden offline data asset",
        "status": "offline-asset",
        "runner": "not wired",
        "hidden": True,
        "rows": len(combined),
        "per_scenario": PER_SCENARIO,
        "seed": SEED,
        "tokenizer": {
            "name": TOKENIZER_NAME,
            "path": sources.raw_path_display(tokenizer_spec["path"]),
            "sha256": tokenizer_spec["sha256"],
            "tokens_semantics": "input_tokens = prompt content tokens; chat-template control tokens excluded",
        },
        "sampling": {
            "method": "sha256(seed|upstream_id) order + stratified largest-remainder quotas",
            "executed_by": "tool/dataset/accuracy/build.py",
        },
        "upstream_id_format": UPSTREAM_ID_FORMAT,
        "judges": JUDGE_FAMILIES,
        "scenarios": scenario_manifest,
        "datasets": sources.dataset_provenance("data"),
        "tooling": sources.dataset_provenance("tooling"),
        "totals": {
            "rows": len(combined),
            "scenarios": len(SCENARIOS),
            "per_scenario": PER_SCENARIO,
        },
        "deviations": DEVIATIONS,
        "files": {
            "testset.jsonl": {
                "rows": len(combined),
                "bytes": testset_path.stat().st_size,
                "sha256": sha256_file(testset_path),
            }
        },
    }
    write_json(TEST_DIR_ACC / "manifest.json", manifest)
    write_report(manifest)
    print(f"[accuracy] wrote {len(combined)} rows -> {testset_path.relative_to(ROOT)}")
    return manifest


JUDGE_SUMMARY = {
    "knowledge_qa": "mc_letter",
    "instruction_following": "ifeval",
    "long_context": "normalized_match (f1)",
    "math_logic": "normalized_match (numeric_exact)",
    "coding": "code_exec",
    "general_agent": "tool_call",
    "search_agent": "normalized_match (f1)",
    "multilingual": "mc_letter",
}


def write_report(manifest: Dict[str, Any]) -> None:
    lines = ["# 精度测试集：构造与抽样报告（自动生成）", ""]
    lines.append(f"- seed={manifest['seed']}；每场景 {manifest['per_scenario']} 条；共 {manifest['rows']} 条（单文件 testset.jsonl）")
    lines.append(f"- 参考 tokenizer：{manifest['tokenizer']['name']}；input_tokens 为 prompt 内容 token（不含 chat template）")
    lines.append(f"- 抽样：{manifest['sampling']['method']}")
    lines.append("")
    lines.append("| 场景 | 数据集 | 语言 | 样本 | 判题族 | input_tokens(min/median/max) |")
    lines.append("|---|---|---|---:|---|---|")
    for scenario in SCENARIOS:
        item = manifest["scenarios"][scenario]
        datasets = ", ".join(f"{k}:{v}" for k, v in item["datasets"].items())
        languages = ", ".join(f"{k}:{v}" for k, v in item["languages"].items())
        tokens = item["input_tokens"]
        lines.append(
            f"| {scenario} | {datasets} | {languages} | {item['rows']} | "
            f"{JUDGE_SUMMARY[scenario]} | {tokens['min']}/{tokens['median']}/{tokens['max']} |"
        )
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
    lines.append("## 数据源")
    lines.append("")
    lines.append("| 数据集 | 上游 | 许可 | 文件数 |")
    lines.append("|---|---|---|---:|")
    for name, info in manifest["datasets"].items():
        lines.append(f"| {name} | {info['source_url']} | {info['license']} | {len(info['files'])} |")
    lines.append("")
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def _check_mc_pairs(
    rows: List[Dict[str, Any]],
    scenario: str,
    expected_groups: int,
    expected_languages: Iterable[str],
    problems: List[str],
    *,
    require_distinct_options: bool = False,
) -> None:
    """Verify a paired multiple-choice scenario: same question id x languages."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["scenario"] == scenario:
            grouped[row["meta"]["sample_id"]].append(row)
    wanted = set(expected_languages)
    if len(grouped) != expected_groups:
        problems.append(f"{scenario}: {len(grouped)} paired sample_ids != {expected_groups}")
        return
    for sample_id in sorted(grouped):
        group = grouped[sample_id]
        languages = {row["language"] for row in group}
        if languages != wanted or len(group) != len(wanted):
            problems.append(f"{scenario}: {sample_id} languages {sorted(languages)}")
            return
        answers = {row["judge"]["answer"] for row in group}
        if len(answers) != 1:
            problems.append(f"{scenario}: {sample_id} answer letters disagree: {sorted(answers)}")
            return
        if require_distinct_options:
            for row in group:
                try:
                    options = _mc_options_from_prompt(row["messages"][0]["content"])
                except ValueError as exc:
                    problems.append(f"{scenario}: {row['id']} options unparsable: {exc}")
                    return
                if not _multilingual_options_ok(options):
                    problems.append(f"{scenario}: {row['id']} options are not 4 distinct non-empty choices")
                    return


def _check_long_context_unions(rows: List[Dict[str, Any]], problems: List[str]) -> int:
    """Recompute every long_context answer union from raw and compare with the output."""
    prompt_path = RAW_DIR / "longbench_v1/prompts/dataset2prompt.json"
    if not prompt_path.is_file():
        problems.append("long_context union check needs raw dataset2prompt.json (run fetch.py)")
        return 0
    templates = read_json(prompt_path)
    expected: Dict[Tuple[str, str], List[str]] = {}
    for task in LONG_CONTEXT_TASK_LANG:
        for group in _longbench_groups(task, templates).values():
            expected[(task, group["prompt"])] = group["answers"]
    unioned = 0
    for row in rows:
        if row["scenario"] != "long_context":
            continue
        key = (row["meta"]["task"], row["messages"][0]["content"])
        wanted = expected.get(key)
        if wanted is None:
            problems.append(f"long_context prompt not found upstream: {row['id']}")
            return unioned
        if row["judge"]["answers"] != wanted:
            problems.append(f"long_context answers are not the full upstream union: {row['id']}")
            return unioned
        if len(wanted) > 1:
            unioned += 1
    if unioned < 7:
        problems.append(f"long_context unioned-answer coverage too low: {unioned} rows (< 7 known rows)")
    return unioned


def _check_judge_contract(rows: List[Dict[str, Any]], problems: List[str]) -> None:
    """Every row must carry a complete, scenario-consistent judge contract."""
    maxlen_path = RAW_DIR / "longbench_v1/prompts/dataset2maxlen.json"
    maxlen = read_json(maxlen_path) if maxlen_path.is_file() else {}
    for row in rows:
        judge = row["judge"]
        row_id = row["id"]
        scenario = row["scenario"]
        max_new = judge.get("max_new_tokens")
        if not isinstance(max_new, int) or max_new <= 0:
            problems.append(f"missing/invalid judge.max_new_tokens: {row_id}")
            continue
        if scenario in EXPECTED_MAX_NEW_TOKENS and max_new != EXPECTED_MAX_NEW_TOKENS[scenario]:
            problems.append(
                f"{scenario}: judge.max_new_tokens={max_new} != {EXPECTED_MAX_NEW_TOKENS[scenario]} ({row_id})"
            )
        if scenario == "long_context":
            task = row["meta"].get("task")
            expected_metric = LONG_CONTEXT_TASK_METRIC.get(task)
            if judge.get("metric") != expected_metric:
                problems.append(
                    f"long_context metric {judge.get('metric')!r} != {expected_metric!r} for task {task} ({row_id})"
                )
            if maxlen and max_new != int(maxlen.get(task, 64)):
                problems.append(
                    f"long_context max_new_tokens {max_new} != dataset2maxlen {maxlen.get(task)} ({row_id})"
                )
        if judge["family"] == "normalized_match":
            metric = judge.get("metric")
            if metric not in ALLOWED_METRICS:
                problems.append(f"unknown judge.metric {metric!r}: {row_id}")
            elif scenario == "math_logic" and metric != "numeric_exact":
                problems.append(f"math_logic metric {metric!r}: {row_id}")
            elif scenario == "search_agent" and metric != "f1":
                problems.append(f"search_agent metric {metric!r}: {row_id}")
        if row["dataset"] == "cmath":
            if judge.get("tolerance") != "relative_1pct":
                problems.append(f"CMATH row must set tolerance=relative_1pct: {row_id}")
        elif judge.get("tolerance") is not None:
            problems.append(f"only CMATH uses tolerance (found {judge.get('tolerance')!r}): {row_id}")


def check_outputs() -> int:
    manifest = read_json(TEST_DIR_ACC / "manifest.json")
    rows = read_jsonl(TEST_DIR_ACC / "testset.jsonl")
    problems: List[str] = []
    if len(rows) != manifest["rows"] or manifest["rows"] != PER_SCENARIO * len(SCENARIOS):
        problems.append(f"row count {len(rows)} != {manifest['rows']}")
    if len({r["id"] for r in rows}) != len(rows):
        problems.append("duplicate ids")
    upstream_ids = [r["upstream_id"] for r in rows]
    if len(set(upstream_ids)) != len(upstream_ids):
        problems.append("duplicate upstream_id values (must be globally unique)")
    if manifest.get("status") != "offline-asset" or manifest.get("runner") != "not wired":
        problems.append("accuracy manifest must stay status=offline-asset / runner=not wired")
    if (TEST_DIR_ACC / "testset.yaml").is_file():
        problems.append("accuracy must stay hidden: remove testset.yaml")
    by_scenario = Counter(r["scenario"] for r in rows)
    for scenario in SCENARIOS:
        if by_scenario[scenario] != PER_SCENARIO:
            problems.append(f"{scenario}: {by_scenario[scenario]} rows")
    language_counts = Counter(r["language"] for r in rows)
    if (
        language_counts.get("en") != 1400
        or language_counts.get("zh") != 350
        or sum(v for k, v in language_counts.items() if k not in ("en", "zh")) != 250
    ):
        problems.append(f"language distribution changed: {dict(language_counts)}")
    allowed_fields = {"id", "scenario", "dataset", "language", "upstream_id", "input_tokens", "messages", "judge", "meta"}
    for row in rows:
        if set(row) - allowed_fields or allowed_fields - set(row):
            problems.append(f"unexpected row fields: {row.get('id')}")
            break
        if row["judge"]["family"] not in JUDGE_FAMILIES:
            problems.append(f"unknown judge family: {row['id']}")
            break
        if not isinstance(row["input_tokens"], int) or row["input_tokens"] <= 0:
            problems.append(f"bad input_tokens: {row['id']}")
            break
        if not row["messages"] or not row["messages"][0]["content"].strip():
            problems.append(f"empty prompt: {row['id']}")
            break

    _check_judge_contract(rows, problems)

    # A) Global-MMLU alignment: no misaligned subject or known-bad pair may ship.
    for row in rows:
        if row["scenario"] != "knowledge_qa":
            continue
        if row["meta"]["subject"] == "high_school_world_history":
            problems.append(f"knowledge_qa still ships misaligned subject: {row['id']}")
            break
        if row["meta"]["sample_id"] in KNOWN_UPSTREAM_ERRORS:
            problems.append(f"knowledge_qa still ships known upstream error: {row['id']}")
            break
    _check_mc_pairs(rows, "knowledge_qa", 125, ("en", "zh"), problems)
    _check_mc_pairs(rows, "multilingual", 25, MULTILINGUAL_LANGUAGES, problems, require_distinct_options=True)

    for scenario in ("knowledge_qa", "multilingual"):
        excluded = manifest["scenarios"][scenario]["sampling"].get("alignment_excluded")
        if not excluded:
            problems.append(f"{scenario}: manifest sampling.alignment_excluded is missing/empty")
        if scenario == "knowledge_qa" and not any(
            item.get("subject") == "high_school_world_history" for item in (excluded or [])
        ):
            problems.append("knowledge_qa: alignment_excluded does not record high_school_world_history")

    # B) LongBench duplicate-prompt answers must be the full upstream union.
    _check_long_context_unions(rows, problems)

    # E) MBPP must name the function under test in the prompt and the judge.
    mbpp_rows = [r for r in rows if r["scenario"] == "coding" and r["dataset"] == "mbpp"]
    if len(mbpp_rows) != 125:
        problems.append(f"mbpp rows {len(mbpp_rows)} != 125")
    for row in mbpp_rows:
        entry_point = row["judge"].get("entry_point")
        if not entry_point:
            problems.append(f"mbpp judge.entry_point is empty: {row['id']}")
            break
        if entry_point not in row["messages"][0]["content"]:
            problems.append(f"mbpp entry_point missing from prompt: {row['id']}")
            break
    for row in rows:
        if row["scenario"] != "coding" or row["dataset"] != "humaneval":
            continue
        entry_point = row["judge"].get("entry_point")
        if not entry_point or entry_point not in (row["judge"].get("prompt_prefix") or ""):
            problems.append(f"humaneval entry_point missing from prompt_prefix: {row['id']}")
            break

    digest = sha256_file(TEST_DIR_ACC / "testset.jsonl")
    if digest != manifest["files"]["testset.jsonl"]["sha256"]:
        problems.append("testset.jsonl sha256 mismatch")
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}")
        return 1
    print(f"[check] PASS rows={len(rows)} per_scenario={PER_SCENARIO} scenarios={len(SCENARIOS)} sha={digest[:12]}")
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verify existing outputs only")
    args = parser.parse_args(argv)
    if args.check:
        return check_outputs()
    for path in (
        RAW_DIR / "tokenizer" / TOKENIZER_NAME / "tokenizer.json",
        RAW_DIR / "global_mmlu/en/test.parquet",
        RAW_DIR / "longbench_v1/prompts/dataset2prompt.json",
    ):
        if not path.is_file():
            print(f"missing raw asset: {path}\nrun: python tool/dataset/accuracy/fetch.py", file=sys.stderr)
            return 1
    build_outputs()
    return check_outputs()


if __name__ == "__main__":
    sys.exit(main())
