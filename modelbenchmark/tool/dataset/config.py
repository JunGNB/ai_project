"""Stable configuration for the benchmark sample sets (design v3).

Everything that decides *which* samples are produced lives here so a rebuild is
deterministic: fixed seed, quotas, sources, sampling strata, construction rules,
template identifiers and provenance inputs.

Design reference: ``dataset/TEST_SAMPLE_SELECTION.md``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

SEED = 20261106

#: Declared context buckets (reference tokenizer counts, messages + tools).
BUCKETS: Dict[str, int] = {"1k": 1024, "8k": 8192, "32k": 32768}
ACCEPT_LOW, ACCEPT_HIGH = 0.95, 1.05
OSL = 50  # run-time default decode length (perf.max_tokens); manifests record it

#: Full performance set: 8 scenarios x 3 buckets x 20 targets = 480 rows.
PERF_PER_BUCKET = 20
#: Quick regression set: same scenarios, 10 targets per bucket = 240 rows.
PERF_SUBSET_PER_BUCKET = 10
#: Default accuracy rows per scenario; three scenarios override this (see
#: ``ACC_SOURCES[*]["n"]``): general_agent = 889, multilingual = 4,200.
ACC_PER_SCENARIO = 250
#: Accuracy long-context samples keep only prompts inside a 4K-32K window
#: (sampling filter, no truncation): below the floor a sample would not exercise
#: long-context behaviour, above the ceiling it would not fit the 32K model.
ACC_MIN_TOKENS = 4096
ACC_MAX_TOKENS = 32768
#: LiveCodeBench hidden tests above this JSON size are omitted (local 10s executor cannot use them).
ACC_MAX_TEST_BYTES = 256 * 1024

#: Performance scenario -> corpus and construction rule.
PERF_SOURCES: Dict[str, Dict[str, Any]] = {
    "knowledge_qa": {
        "dataset": "longbench_v2",
        "workload": "longbench_v2:Single-Document QA (open-book long-doc MCQ)",
        "domain": "Single-Document QA",
        "construction": "document_prefix",
        "tools": None,
    },
    "long_context": {
        "dataset": "longbench_v2",
        "workload": "longbench_v2:Long Structured Data Understanding",
        "domain": "Long Structured Data Understanding",
        "construction": "document_prefix",
        "tools": None,
    },
    "coding": {
        "dataset": "longbench_v2",
        "workload": "longbench_v2:Code Repository Understanding (reading, not generation)",
        "domain": "Code Repository Understanding",
        "construction": "document_prefix",
        "tools": None,
    },
    "search_agent": {
        "dataset": "longbench_v2",
        "workload": "longbench_v2:Multi-Document QA (single request; tool schemas only)",
        "domain": "Multi-Document QA",
        "construction": "document_prefix_with_search_tools",
        "tools": "search_lookup",
    },
    # Not multilingual: LongBench v1 dureader is Chinese-only.  Renamed from
    # ``multilingual`` so the scenario key does not over-claim a capability.
    "chinese_long_context": {
        "dataset": "longbench_v1",
        "workload": "longbench_v1:dureader",
        "task": "dureader",
        "construction": "target_document_first",
        "tools": None,
    },
    "instruction_following": {
        "dataset": "ifeval",
        "workload": "ifeval:target instruction + same-set reference pool",
        "construction": "instruction_pool",
        "tools": None,
    },
    "math_logic": {
        "dataset": "gsm8k",
        "workload": "gsm8k:target problem + solved examples",
        "construction": "solved_examples",
        "tools": None,
    },
    "general_agent": {
        "dataset": "bfcl",
        "workload": "bfcl:single request + whole-function catalogue prefix (target functions first)",
        "construction": "function_catalog",
        "tools": "target_functions",
    },
}

#: LongBench v1 tasks used by the accuracy long-context scenario (all four kept).
LONG_CONTEXT_TASKS: Tuple[str, ...] = ("narrativeqa", "qasper", "hotpotqa", "2wikimqa")
#: Length bands for the accuracy long-context scenario.  A prompt belongs to the
#: band whose ``[floor, ceiling)`` contains its reference token count; the top band
#: is capped by ``ACC_MAX_TOKENS`` instead (floor 16,384, ceiling 32,768 inclusive).
#: Targets: 250 rows total, with the whole 16K-32K pool (51 rows) taken up front.
LONG_CONTEXT_LENGTH_BANDS: Dict[str, int] = {"4k-8k": 100, "8k-16k": 99, "16k-32k": 51}

#: BFCL v4 single-turn categories (11) in deterministic sampling order.
BFCL_CATEGORIES: Tuple[str, ...] = (
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
)
#: Per-category accuracy target; a category whose deduped pool is smaller takes
#: the whole pool (live_parallel 15, live_parallel_multiple 24).
BFCL_CATEGORY_COUNTS: Dict[str, int] = {
    "simple_python": 100,
    "simple_java": 100,
    "simple_javascript": 50,
    "multiple": 100,
    "parallel": 100,
    "parallel_multiple": 100,
    "irrelevance": 100,
    "live_simple": 100,
    "live_multiple": 100,
    "live_parallel": 15,
    "live_parallel_multiple": 24,
}
#: BFCL accuracy rows after global user-question dedupe (sum of category counts).
BFCL_ACC_ROWS = sum(BFCL_CATEGORY_COUNTS.values())

#: Global-MMLU languages available locally: the same 14,042 sample_ids per language.
GLOBAL_MMLU_LANGUAGES: Tuple[str, ...] = (
    "am", "ar", "bn", "cs", "de", "el", "en", "es", "fa", "fil", "fr", "ha", "he",
    "hi", "id", "ig", "it", "ja", "ko", "ky", "lt", "mg", "ms", "ne", "nl", "ny",
    "pl", "pt", "ro", "ru", "si", "sn", "so", "sr", "sv", "sw", "te", "tr", "uk",
    "vi", "yo", "zh",
)
#: Rows per language for the paired multilingual design (same 100 sample_ids everywhere).
GLOBAL_MMLU_PER_LANGUAGE = 100

#: Accuracy scenario -> dataset, judge contract and sampling strata.
ACC_SOURCES: Dict[str, Dict[str, Any]] = {
    "knowledge_qa": {
        "dataset": "mmlu_pro",
        "n": ACC_PER_SCENARIO,
        "judge": "mc_letter",
        "metric": "acc@1",
        "strata": "category (14 subjects), round-robin",
        "prompt": "mmlu_pro",
    },
    "instruction_following": {
        "dataset": "ifbench",
        "n": ACC_PER_SCENARIO,
        "judge": "ifbench",
        "metric": "prompt-level + instruction-level, strict + loose acc",
        "strata": "primary instruction id (58 types), round-robin",
        "prompt": "verbatim",
    },
    "long_context": {
        "dataset": "longbench_v1",
        "n": sum(LONG_CONTEXT_LENGTH_BANDS.values()),
        "judge": "qa_f1",
        "metric": "F1 (max over reference answers) / EM",
        "strata": "length band (4k-8k/8k-16k/16k-32k) x task, round-robin within band; only 4K-32K tokens",
        "tasks": list(LONG_CONTEXT_TASKS),
        "length_bands": dict(LONG_CONTEXT_LENGTH_BANDS),
        "min_tokens": ACC_MIN_TOKENS,
        "max_tokens": ACC_MAX_TOKENS,
    },
    "math_logic": {
        "dataset": "math_500",
        "n": ACC_PER_SCENARIO,
        "judge": "math",
        "metric": "acc (math-verify equivalence)",
        "strata": "subject x level, round-robin",
        "prompt": "math_boxed",
    },
    "coding": {
        "dataset": "livecodebench",
        "n": ACC_PER_SCENARIO,
        "judge": "code_pass1",
        "metric": "pass@1 (hidden stdin/stdout tests)",
        "strata": "newest release window first: all judgeable test6 (2025-01..2025-04), then newest judgeable test5 rows (2024-09..2025-01) to reach 250",
        "prompt": "lcb",
        "window": "2024-09-22..2025-04-06 (release v5 + v6)",
    },
    "general_agent": {
        "dataset": "bfcl",
        "n": BFCL_ACC_ROWS,
        "judge": "bfcl_ast",
        "metric": "AST match acc (possible answers, order-insensitive)",
        "strata": "single_turn category: " + " / ".join(
            f"{category} {count}" for category, count in BFCL_CATEGORY_COUNTS.items()
        ) + "; random within category, global user-question dedupe",
        "prompt": "bfcl",
        "categories": list(BFCL_CATEGORIES),
        "category_counts": dict(BFCL_CATEGORY_COUNTS),
        "category_type": "single_turn",
    },
    "search_agent": {
        "dataset": "hotpotqa",
        "n": ACC_PER_SCENARIO,
        "judge": "qa_f1",
        "metric": "EM / F1",
        "strata": "type (bridge/comparison, proportional to canonical pool), round-robin",
        "prompt": "hotpot_agent",
        "corpus": "paragraphs in meta, retrieval via search/lookup tools",
    },
    "multilingual": {
        "dataset": "global_mmlu",
        "n": GLOBAL_MMLU_PER_LANGUAGE * len(GLOBAL_MMLU_LANGUAGES),
        "judge": "mc_letter",
        "metric": "acc@1 per language + macro",
        "strata": "language (42) x paired sample_id (100 common ids, subject round-robin)",
        "prompt": "global_mmlu",
        "languages": list(GLOBAL_MMLU_LANGUAGES),
        "per_language": GLOBAL_MMLU_PER_LANGUAGE,
        "pairing": "same 100 sample_id values in every language (translated counterparts)",
    },
}

#: LongBench v1 accuracy tasks and their official prompt templates.
LONG_CONTEXT_TEMPLATES: Dict[str, str] = {
    # Official LongBench v1 ``config/dataset2prompt.json`` entries (verbatim,
    # including upstream's "asconcisely" typo in narrativeqa).
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, and a question. "
        "Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\n"
        "Story: {context}\n\n"
        "Now, answer the question based on the story asconcisely as you can, using a single phrase if possible. "
        "Do not provide any explanation.\n\n"
        "Question: {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question as concisely as you can, "
        "using a single phrase or sentence if possible. If the question cannot be answered based on the information "
        "in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or "
        "\"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n "
        "Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. "
        "If the question cannot be answered based on the information in the article, write \"unanswerable\". "
        "If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\n"
        "Question: {input}\n\nAnswer:"
    ),
}

#: Fixed tool schemas for search-style samples.
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

JUDGE_SPECS: Dict[str, Dict[str, Any]] = {
    "mc_letter": {
        "answer": "single option letter (A-J for MMLU-Pro, A-D for Global-MMLU)",
        "extract": "explicit answer phrase -> bracketed letter -> letter-only line -> unique standalone letter; "
                   "letters beyond len(choices) are rejected",
        "correct": "exact letter match",
        "metric": "acc@1",
    },
    "ifbench": {
        "answer": "null; judge inputs are meta.instruction_id_list, meta.kwargs and meta.ifbench_key",
        "extract": "official IFBench evaluation_lib + instructions registry",
        "correct": "prompt-level strict/loose and instruction-level strict/loose accuracy",
        "metric": "four accuracy variants",
    },
    "qa_f1": {
        "answer": "meta.answers (list of reference answers; answer mirrors answers[0])",
        "extract": "SQuAD-style normalisation",
        "correct": "F1 (max over references); EM reported as secondary",
        "metric": "F1 / EM",
    },
    "math": {
        "answer": "reference answer string (LaTeX allowed)",
        "extract": "last \\boxed{...} then answer phrase",
        "correct": "math-verify symbolic/numeric equivalence",
        "metric": "acc",
    },
    "code_pass1": {
        "answer": "null; judge inputs are meta.tests (hidden stdin/stdout, <=256KB each) and meta.public_tests",
        "extract": "run generated program once per test in an isolated process",
        "correct": "all hidden tests pass; pass@1 over the sampled set",
        "metric": "pass@1",
    },
    "bfcl_ast": {
        "answer": "meta.possible_answers (official possible-answer structure)",
        "extract": "normalise predicted tool calls",
        "correct": "official AST match: candidate set, per-argument alternatives, optional params, "
                   "order-insensitive parallel calls",
        "metric": "acc",
    },
}

CONSTRUCTION_RULES: Dict[str, str] = {
    "document_prefix": (
        "one LongBench v2 document per target; 1K/8K/32K take a prefix slice of that document; "
        "question/options are appended unchanged. Workload-shape probe, not answerability-guaranteed."
    ),
    "document_prefix_with_search_tools": (
        "same as document_prefix, plus fixed search/lookup tool schemas counted in tokens; "
        "one request, tools are not executed in performance mode."
    ),
    "target_document_first": (
        "1K slices the target's own dureader document; 8K/32K concatenate the target document first, "
        "then other same-task documents, and slice to the target length."
    ),
    "instruction_pool": (
        "target IFEval instruction is appended after a reference block built from other IFEval prompts "
        "(same dataset, target excluded)."
    ),
    "solved_examples": (
        "target GSM8K problem is appended after solved examples drawn from the same split, "
        "excluding the 20 target problems."
    ),
    "function_catalog": (
        "target BFCL request plus a catalogue of complete, JSON-Schema-normalised function definitions "
        "ordered target-functions-first, then same-category, then the rest; filled by whole functions only "
        "(never truncated mid-JSON). The catalogue is a prefix of a shared 642-function pool, not the full pool."
    ),
}

PROVENANCE_INPUTS: List[Dict[str, Any]] = [
    {
        "dataset": "mmlu_pro",
        "used_by": ['accuracy'],
        "source_url": "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro",
        "license": "MIT",
        "files": [("mmlu_pro/test.parquet", "test.parquet")],
    },
    {
        "dataset": "ifbench",
        "used_by": ['accuracy'],
        "source_url": "https://huggingface.co/datasets/allenai/IFBench_test",
        "license": "ODC-BY-1.0",
        "files": [("ifbench/IFBench_test.jsonl", "IFBench_test.jsonl")],
    },
    {
        "dataset": "longbench_v1",
        "used_by": ['performance', 'accuracy'],
        "source_url": "https://huggingface.co/datasets/zai-org/LongBench",
        "license": "research use (see upstream repo)",
        "files": [("longbench_v1/data.zip", "data.zip")],
    },
    {
        "dataset": "math_500",
        "used_by": ['accuracy'],
        "source_url": "https://huggingface.co/datasets/HuggingFaceH4/MATH-500",
        "license": "MIT",
        "files": [("math_500/test.jsonl", "test.jsonl")],
    },
    {
        "dataset": "livecodebench",
        "used_by": ['accuracy'],
        "source_url": "https://huggingface.co/datasets/livecodebench/code_generation_lite",
        "license": "see upstream repo",
        "files": [
            ("livecodebench/test5.jsonl", "test5.jsonl"),
            ("livecodebench/test6.jsonl", "test6.jsonl"),
        ],
    },
    {
        "dataset": "bfcl",
        "used_by": ['performance', 'accuracy'],
        "source_url": "https://github.com/ShishirPatil/gorilla",
        "license": "Apache-2.0",
        # Accuracy consumes all 11 single-turn request files plus their
        # possible-answer files.  ``irrelevance`` has no possible_answer file: its
        # gold set is the empty answer list (no tool call is correct).
        "files": [
            (f"bfcl/BFCL_v4_{category}.json", f"BFCL_v4_{category}.json")
            for category in BFCL_CATEGORIES
        ] + [
            (
                f"bfcl/possible_answer/BFCL_v4_{category}.json",
                f"possible_answer/BFCL_v4_{category}.json",
            )
            for category in BFCL_CATEGORIES
            if category != "irrelevance"
        ],
        # Performance uses the request files only; possible_answer is judge input.
        "performance_files": [
            ("bfcl/BFCL_v4_simple_python.json", "BFCL_v4_simple_python.json"),
            ("bfcl/BFCL_v4_multiple.json", "BFCL_v4_multiple.json"),
            ("bfcl/BFCL_v4_parallel.json", "BFCL_v4_parallel.json"),
        ],
    },
    {
        "dataset": "hotpotqa",
        "used_by": ['accuracy'],
        "source_url": "https://huggingface.co/datasets/hotpotqa/hotpot_qa",
        "license": "CC BY-SA 4.0",
        "files": [
            (
                "hotpotqa/distractor/validation-00000-of-00001.parquet",
                "distractor/validation.parquet",
            )
        ],
    },
    {
        "dataset": "global_mmlu",
        "used_by": ['accuracy'],
        "source_url": "https://huggingface.co/datasets/CohereLabs/Global-MMLU",
        "license": "see dataset card",
        "files": [
            (f"global_mmlu/{lang}/test.parquet", f"{lang}/test.parquet")
            for lang in GLOBAL_MMLU_LANGUAGES
        ],
    },
    {
        "dataset": "longbench_v2",
        "used_by": ['performance'],
        "source_url": "https://huggingface.co/datasets/zai-org/LongBench-v2",
        "license": "Apache-2.0",
        "files": [("longbench_v2/data.json", "data.json")],
    },
    {
        "dataset": "ifeval",
        "used_by": ['performance'],
        "source_url": "https://huggingface.co/datasets/google/IFEval",
        "license": "Apache-2.0",
        "files": [("ifeval/ifeval_input_data.jsonl", "ifeval_input_data.jsonl")],
    },
    {
        "dataset": "gsm8k",
        "used_by": ['performance'],
        "source_url": "https://huggingface.co/datasets/openai/gsm8k",
        "license": "MIT",
        "files": [("gsm8k/main/test-00000-of-00001.parquet", "main/test.parquet")],
    },
]

def provenance_for(set_name: str) -> List[Dict[str, Any]]:
    """Provenance entries actually used by one set (``performance`` / ``accuracy``)."""
    out: List[Dict[str, Any]] = []
    for entry in PROVENANCE_INPUTS:
        if set_name not in entry.get("used_by", []):
            continue
        files = entry.get(f"{set_name}_files", entry["files"])
        out.append({"dataset": entry["dataset"], "source_url": entry["source_url"],
                    "license": entry["license"], "files": files})
    return out


EXPECTED = {
    "perf_rows": len(PERF_SOURCES) * len(BUCKETS) * PERF_PER_BUCKET,
    "perf_subset_rows": len(PERF_SOURCES) * len(BUCKETS) * PERF_SUBSET_PER_BUCKET,
    # 6 x 250 + BFCL 889 + Global-MMLU 4,200 = 6,589.
    "acc_rows": sum(spec["n"] for spec in ACC_SOURCES.values()),
}
