#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Raw-asset specification for the accuracy test set (``dataset/test/accuracy``).

``raw_spec.json`` is the committed, auditable list of every upstream file the
accuracy builder needs: relative path under ``dataset/raw/``, download URL, byte
size and SHA-256.  ``dataset/raw/`` itself is gitignored, so a fresh clone fetches
these files with ``tool/dataset/accuracy/fetch.py`` and then rebuilds the test
set with ``tool/dataset/accuracy/build.py``.

This module also owns the *dataset* metadata (human source URL / license) that
ends up in the test-set manifest.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from tool.common.fs import DATA_DIR, RAW_DIR, read_json

SPEC_PATH = Path(__file__).with_name("raw_spec.json")

#: Dataset -> human-facing provenance.  File-level URL/size/SHA-256 live in
#: ``raw_spec.json``; these are the canonical project / model-card links.
DATASETS: Dict[str, Dict[str, str]] = {
    "global_mmlu": {
        "name": "Global-MMLU",
        "source_url": "https://huggingface.co/datasets/CohereLabs/Global-MMLU",
        "license": "Apache-2.0",
        "used_by": ["knowledge_qa", "multilingual"],
    },
    "ifeval": {
        "name": "IFEval",
        "source_url": "https://github.com/google-research/google-research/tree/master/instruction_following_eval",
        "license": "Apache-2.0",
        "used_by": ["instruction_following"],
    },
    "longbench_v1": {
        "name": "LongBench v1",
        "source_url": "https://github.com/THUDM/LongBench",
        "license": "MIT",
        "used_by": ["long_context"],
    },
    "gsm8k": {
        "name": "GSM8K",
        "source_url": "https://huggingface.co/datasets/openai/gsm8k",
        "license": "MIT",
        "used_by": ["math_logic"],
    },
    "cmath": {
        "name": "CMATH",
        "source_url": "https://github.com/XiaoMi/cmath",
        "license": "see upstream repository (research use)",
        "used_by": ["math_logic"],
    },
    "humaneval": {
        "name": "HumanEval",
        "source_url": "https://github.com/openai/human-eval",
        "license": "MIT",
        "used_by": ["coding"],
    },
    "mbpp": {
        "name": "MBPP (sanitized)",
        "source_url": "https://github.com/google-research/google-research/tree/master/mbpp",
        "license": "CC-BY-4.0",
        "used_by": ["coding"],
    },
    "bfcl_v4": {
        "name": "BFCL v4",
        "source_url": "https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard",
        "license": "Apache-2.0",
        "used_by": ["general_agent"],
    },
    "hotpotqa_distractor": {
        "name": "HotpotQA (distractor)",
        "source_url": "https://huggingface.co/datasets/hotpotqa/hotpot_qa",
        "license": "CC-BY-SA-4.0",
        "used_by": ["search_agent"],
    },
}

#: Non-dataset assets required by builders / judges (reference tokenizer, NLTK data).
TOOLING: Dict[str, Dict[str, str]] = {
    "tokenizer": {
        "name": "cl100k_base (OpenAI)",
        "source_url": "https://github.com/openai/tiktoken",
        "license": "MIT (tiktoken) / cl100k_base encoding data",
        "purpose": (
            "generic industry reference tokenizer for input_tokens and "
            "long-context length filtering (model-agnostic)"
        ),
    },
    "nltk": {
        "name": "NLTK punkt / punkt_tab",
        "source_url": "https://github.com/nltk/nltk_data",
        "license": "Apache-2.0 (nltk_data)",
        "purpose": "sentence/word tokenization used by the official IFEval checkers",
    },
}


def load_spec() -> Dict[str, Any]:
    return read_json(SPEC_PATH)


def files_for(dataset: str) -> List[Dict[str, Any]]:
    return [f for f in load_spec()["files"] if f["dataset"] == dataset]


def tokenizer_entry() -> Dict[str, Any]:
    """The tokenizer asset the manifest should describe.

    Prefers the derived ``tokenizer.json`` generated rule (the file the builder
    actually reads) and falls back to the downloaded blob entry.
    """
    spec = load_spec()
    for item in spec.get("generated", []):
        if str(item.get("path", "")).endswith("tokenizer.json"):
            return item
    for item in spec["files"]:
        if item.get("dataset") == "tokenizer":
            return item
    raise KeyError("raw_spec.json has no tokenizer asset entry")


def dataset_provenance(scope: str = "data") -> Dict[str, Any]:
    """Manifest-ready provenance: dataset -> source URL, license, files.

    ``scope`` is ``"data"`` (the nine evaluation datasets) or ``"tooling"``
    (tokenizer / NLTK assets; generated derived files are listed with their
    ``derived_by`` script).
    """
    meta = DATASETS if scope == "data" else TOOLING
    spec = load_spec()
    out: Dict[str, Any] = {}
    for dataset, info in meta.items():
        files = [f for f in spec["files"] if f["dataset"] == dataset]
        entries = [
            {"path": f["path"], "bytes": f["bytes"], "sha256": f["sha256"]}
            for f in files
        ]
        if scope == "tooling":
            for item in spec.get("generated", []):
                if item.get("dataset") == dataset:
                    entry = {
                        "path": item["path"],
                        "bytes": item.get("bytes"),
                        "sha256": item.get("sha256"),
                    }
                    if item.get("generator"):
                        entry["derived_by"] = item["generator"]
                    entries.append(entry)
        if not entries:
            continue
        out[dataset] = {
            "name": info["name"],
            "source_url": info["source_url"],
            "license": info["license"],
            "files": entries,
        }
        if "used_by" in info:
            out[dataset]["used_by"] = info["used_by"]
    return out


def raw_path_display(rel: str) -> str:
    """Stable repo-relative path used in manifests/docs (``dataset/raw/...``)."""
    return str(DATA_DIR.relative_to(Path(__file__).resolve().parents[3]) / "raw" / rel)
