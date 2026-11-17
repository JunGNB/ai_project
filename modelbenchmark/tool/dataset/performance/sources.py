#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Upstream data sources for the performance test set (``performance``).

Every entry is a file that already lives under ``dataset/raw/`` (gitignored
evidence directory) or that :mod:`tool.dataset.performance.fetch` can restore from
the pinned upstream URL.  ``url`` is the canonical Hugging Face path (``hf-mirror``
is used at download time when ``HUGGINGFACE_MIRROR``/``--mirror`` is set).

Members of ``longbench_v1/data.zip`` are restored through the ``archive`` /
``member`` pair, so a clean checkout can be rebuilt with a single ``fetch.py``.

Selection rationale for every source lives in
``docs/spec/11-performance-testset.md``; this table only records *where the
bytes come from* (plus size / sha256 / license, filled by ``fetch --verify``).
"""
from __future__ import annotations

from typing import Dict, List

HF = "https://huggingface.co/datasets/"

#: source id -> upstream file used by the builder.  ``path`` is relative to
#: ``dataset/raw``; ``{lang}`` fans out to the ten multilingual languages.
SOURCES: Dict[str, Dict] = {
    "reference_tokenizer_blob": {
        "path": "tokenizer/cl100k_base/cl100k_base.tiktoken",
        "url": "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
        "license": "MIT (tiktoken, OpenAI)",
        "note": "通用参考 tokenizer 的 tiktoken 词表；构建期用它量样本长度，不代表任何被测模型",
        "required": True,
    },
    "reference_tokenizer": {
        "path": "tokenizer/cl100k_base/tokenizer.json",
        "url": "generated",
        "license": "MIT (tiktoken, OpenAI)",
        "note": "由 derive_cl100k.py 从 tiktoken 词表确定性转换（SHA-256 固定）",
        "generated": True,
        "required": True,
    },
    # ------------------------------------------------- 长上下文 / 知识问答 32K / 搜索 32K
    "longbench_pro": {
        "path": "longbench_pro/longbench_pro.json",
        "url": HF + "caskcsg/LongBench-Pro/resolve/main/longbench_pro.json",
        "license": "apache-2.0",
        "note": "LongBench-Pro (2026-01) 1,500 条；750 中/750 英；原生 8K/16K/32K/64K/128K/256K",
        "required": True,
    },
    # ------------------------------------------------- LongBench v1 归档（指令遵循 / 编程）
    "longbench_v1_archive": {
        "path": "longbench_v1/data.zip",
        "url": HF + "zai-org/LongBench/resolve/main/data.zip",
        "license": "apache-2.0",
        "note": "LongBench v1 归档，供下列成员文件还原；只读取使用到的成员",
        "required": True,
    },
    "vcsum": {
        "path": "longbench_v1/data/vcsum.jsonl",
        "url": HF + "zai-org/LongBench/resolve/main/data.zip",
        "member": "data/vcsum.jsonl",
        "archive": "longbench_v1/data.zip",
        "license": "apache-2.0",
        "note": "LongBench v1 中文会议纪要 200 篇（0.6K–23K tok），指令遵循的长输入载体",
        "required": True,
    },
    "lcc": {
        "path": "longbench_v1/data/lcc.jsonl",
        "url": HF + "zai-org/LongBench/resolve/main/data.zip",
        "member": "data/lcc.jsonl",
        "archive": "longbench_v1/data.zip",
        "license": "apache-2.0",
        "note": "LongBench v1 代码补全上下文（Python/Java/C#），编程 4K/32K 输入",
        "required": True,
    },
    "repobench_p": {
        "path": "longbench_v1/data/repobench-p.jsonl",
        "url": HF + "zai-org/LongBench/resolve/main/data.zip",
        "member": "data/repobench-p.jsonl",
        "archive": "longbench_v1/data.zip",
        "license": "apache-2.0",
        "note": "LongBench v1 仓库级补全上下文，编程 32K 的补充片段",
        "required": True,
    },
    # ------------------------------------------------- 知识问答 1K/4K / 搜索 1K/4K（英文）
    "hotpotqa": {
        "path": "hotpotqa/distractor/validation-00000-of-00001.parquet",
        "url": HF + "hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet",
        "license": "cc-by-sa-4.0",
        "note": "HotpotQA distractor 验证集：gold 段落自带答案，distractor 段落可作尾部填充",
        "required": True,
    },
    # ------------------------------------------------- 编程 1K（英文）
    "humaneval": {
        "path": "humaneval/openai_humaneval/test-00000-of-00001.parquet",
        "url": HF + "openai/openai_humaneval/resolve/main/test-00000-of-00001.parquet",
        "license": "mit",
        "note": "HumanEval 164 题（题面 + canonical solution），编程 1K 的完整代码单元",
        "required": True,
    },
    # ------------------------------------------------- 指令遵循（中文）
    "multi_if": {
        "path": "multi_if/multiIF_20241018.csv",
        "url": HF + "facebook/Multi-IF/resolve/main/multiIF_20241018.csv",
        "license": "cc-by-nc-2.0",
        "note": "Multi-IF (2024-10) 中文约束集合（454 条，提供 instruction_id/kwargs）",
        "required": True,
    },
    # ------------------------------------------------- 数学与逻辑（英文）
    "aime_2026": {
        "path": "aime2026/data/train-00000-of-00001.parquet",
        "url": HF + "MathArena/aime_2026/resolve/main/data/train-00000-of-00001.parquet",
        "license": "cc-by-nc-sa-4.0",
        "note": "AIME 2026（2026-10-19 发布）30 道竞赛题，仅题面（无解答）",
        "required": True,
    },
    "math_500": {
        "path": "math_500/test.jsonl",
        "url": HF + "HuggingFaceH4/MATH-500/resolve/main/test.jsonl",
        "license": "mit",
        "note": "MATH-500 (2025-12) 500 题 + 完整解答，用作示例/填充",
        "required": True,
    },
    # ------------------------------------------------- 通用 Agent（英文）
    "bfcl_v4": {
        "path": "bfcl/BFCL_v4_simple_python.json",
        "url": HF + "gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/BFCL_v4_simple_python.json",
        "license": "apache-2.0",
        "note": "BFCL v4 (gorilla-llm, 2026-04-29 发布)；simple_python 400 行 + multiple 200 行",
        "required": True,
    },
    # ------------------------------------------------- 多语言（10 语种，含中文）
    "global_mmlu": {
        "path": "global_mmlu/{lang}/test.parquet",
        "url": HF + "CohereLabs/Global-MMLU/resolve/main/{lang}/test.parquet",
        "license": "apache-2.0",
        "note": "Global-MMLU (2025-08) 42 语种；本测试集取 zh/en/fr/de/es/pt/ru/ja/ko/ar",
        "required": True,
    },
}

#: scenario -> upstream source ids actually used by the builder.  A scenario may
#: use different corpora per length bucket; the per-cell mapping is recorded in
#: the manifest (``cells[*].dataset``).
SCENARIO_SOURCE: Dict[str, str] = {
    "knowledge_qa": "hotpotqa (1K/4K) + longbench_pro:T3 (32K)",
    "instruction_following": "vcsum + multi_if",
    "long_context": "longbench_pro:T4",
    "math_logic": "aime_2026 + math_500",
    "coding": "humaneval (1K) + lcc / repobench-p (4K/32K)",
    "general_agent": "bfcl_v4:simple_python+multiple",
    "search_agent": "hotpotqa (1K/4K) + longbench_pro:T1/T5 (32K)",
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
