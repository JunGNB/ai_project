#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Upstream data sources for the performance test set (``performance``).

Every entry is a file that already lives under ``dataset/raw/`` (gitignored
evidence directory) or that :mod:`tool.dataset.performance.fetch` can restore from
the pinned upstream URL.  ``url`` is the canonical Hugging Face path (``hf-mirror``
is used at download time when ``HUGGINGFACE_MIRROR``/``--mirror`` is set).

Selection rationale for every source lives in
``docs/spec/11-performance-testset.md``; this table only records *where the
bytes come from* (plus size / sha256 / license, filled by ``fetch --verify``).
"""
from __future__ import annotations

from typing import Dict, List

HF = "https://huggingface.co/datasets/"

#: scenario -> (source id, upstream files used by the builder)
#: ``path`` is relative to ``dataset/raw``.
SOURCES: Dict[str, Dict] = {
    "reference_tokenizer_blob": {
        "path": "tokenizer/cl100k_base/cl100k_base.tiktoken",
        "url": "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
        "license": "MIT (tiktoken, OpenAI)",
        "note": "通用参考 tokenizer 的 tiktoken 词表；构建期用它精确对齐 prompt token 数，"
                "不代表任何被测模型",
        "required": True,
    },
    "reference_tokenizer": {
        "path": "tokenizer/cl100k_base/tokenizer.json",
        "url": "generated",
        "license": "MIT (tiktoken, OpenAI)",
        "note": "由 derive_cl100k.py 从上面的 tiktoken 词表确定性转换而来（SHA-256 固定）",
        "generated": True,
        "required": True,
    },
    # ------------------------------------------- 知识问答（中文）/ 长上下文 / 搜索 Agent（英文）
    "longbench_pro": {
        "path": "longbench_pro/longbench_pro.json",
        "url": HF + "caskcsg/LongBench-Pro/resolve/main/longbench_pro.json",
        "license": "apache-2.0",
        "note": "LongBench-Pro (2026-01) 1,500 条，750 中/750 英，原生 8K/16K/32K/64K/128K/256K",
        "required": True,
    },
    # ---------------------------------------------------------------- 指令遵循（中文）
    "vcsum": {
        "path": "longbench_v1/data/vcsum.jsonl",
        "url": HF + "zai-org/LongBench/resolve/main/data.zip",
        "license": "apache-2.0",
        "note": "LongBench v1 中文会议纪要，200 篇（0.6K–23K tok），作为长输入载体",
        "required": True,
    },
    "multi_if": {
        "path": "multi_if/multiIF_20241018.csv",
        "url": HF + "facebook/Multi-IF/resolve/main/multiIF_20241018.csv",
        "license": "cc-by-nc-2.0",
        "note": "Multi-IF (2024-10) 中文约束集合（454 条，提供 instruction_id/kwargs）",
        "required": True,
    },
    # ---------------------------------------------------------------- 编程（英文）
    "longbench_v2": {
        "path": "longbench_v2/data.json",
        "url": HF + "zai-org/LongBench-v2/resolve/main/data.json",
        "license": "apache-2.0",
        "note": "LongBench v2 (2024-12) 503 条；本测试集只用 Code Repository Understanding（50 条）",
        "required": True,
    },
    # ---------------------------------------------------------------- 数学与逻辑（英文，2026 题面）
    "aime_2026": {
        "path": "aime2026/data/train-00000-of-00001.parquet",
        "url": HF + "MathArena/aime_2026/resolve/main/data/train-00000-of-00001.parquet",
        "license": "cc-by-nc-sa-4.0",
        "note": "AIME 2026（2026-10-19 发布）30 道 2026 年竞赛题，仅题面（无解答）",
        "required": True,
    },
    "math_500": {
        "path": "math_500/test.jsonl",
        "url": HF + "HuggingFaceH4/MATH-500/resolve/main/test.jsonl",
        "license": "mit",
        "note": "MATH-500 (2025-12) 500 题 + 完整解答，用作 many-shot 示范",
        "required": True,
    },
    # ---------------------------------------------------------------- 通用 Agent（英文）
    "bfcl_v4": {
        "path": "bfcl/BFCL_v4_simple_python.json",
        "url": HF + "gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/BFCL_v4_simple_python.json",
        "license": "apache-2.0",
        "note": "BFCL v4 (gorilla-llm, 2026-04-29 发布)；simple_python 400 行 + multiple 200 行",
        "required": True,
    },
    # ---------------------------------------------------------------- 多语言（10 语种，含中文）
    "global_mmlu": {
        "path": "global_mmlu/{lang}/test.parquet",
        "url": HF + "CohereLabs/Global-MMLU/resolve/main/{lang}/test.parquet",
        "license": "apache-2.0",
        "note": "Global-MMLU (2025-08) 42 语种；本测试集取 zh/en/fr/de/es/pt/ru/ja/ko/ar",
        "required": True,
    },
}

#: scenario -> source id (documented in the design doc / manifest provenance)
SCENARIO_SOURCE: Dict[str, str] = {
    "knowledge_qa": "longbench_pro:T3. Evidence-Grounded QA",
    "instruction_following": "multi_if+vcsum",
    "long_context": "longbench_pro:T4. Summarization & Synthesis",
    "math_logic": "aime_2026+math_500",
    "coding": "longbench_v2:Code Repository Understanding",
    "general_agent": "bfcl_v4:simple_python+multiple",
    "search_agent": "longbench_pro:T1/T5 Retrieval & Attribution",
    "multilingual": "global_mmlu",
}


def raw_paths(raw_dir) -> List[str]:
    """Expand ``{lang}`` placeholders for the multilingual source."""
    out: List[str] = []
    for key, spec in SOURCES.items():
        path = str(spec["path"])
        if "{lang}" in path:
            continue
        out.append(path)
    return out
