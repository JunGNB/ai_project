#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Raw-asset specification for the performance test set (``dataset/test/performance``).

``raw_spec.json`` is the committed, auditable list of every upstream file the
performance builder needs: relative path under ``dataset/raw/``, download URL, byte
size and SHA-256.  ``dataset/raw/`` itself is gitignored, so a fresh clone fetches
these files with ``tool/dataset/performance/fetch.py`` and then rebuilds the test
set with ``tool/dataset/performance/build.py``.

This module also owns the *dataset* metadata (human source URL / license) that
ends up in the test-set manifest.  Only the five datasets the performance set
actually reads are listed here (no accuracy-only assets).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from tool.common.fs import DATA_DIR, RAW_DIR, read_json

SPEC_PATH = Path(__file__).with_name("raw_spec.json")

#: Dataset -> human-facing provenance.  File-level URL/size/SHA-256 live in
#: ``raw_spec.json``; these are the canonical project / model-card links.
DATASETS: Dict[str, Dict[str, Any]] = {
    "longbench_v2": {
        "name": "LongBench v2",
        "source_url": "https://huggingface.co/datasets/zai-org/LongBench-v2",
        "license": "Apache-2.0",
        "used_by": ["knowledge_qa", "long_context", "coding", "search_agent"],
    },
    "ifeval": {
        "name": "IFEval",
        "source_url": "https://github.com/google-research/google-research/tree/master/instruction_following_eval",
        "license": "Apache-2.0",
        "used_by": ["instruction_following"],
    },
    "gsm8k": {
        "name": "GSM8K",
        "source_url": "https://huggingface.co/datasets/openai/gsm8k",
        "license": "MIT",
        "used_by": ["math_logic"],
    },
    "bfcl_v4": {
        "name": "BFCL v4",
        "source_url": "https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard",
        "license": "Apache-2.0",
        "used_by": ["general_agent"],
    },
    "global_mmlu": {
        "name": "Global-MMLU",
        "source_url": "https://huggingface.co/datasets/CohereLabs/Global-MMLU",
        "license": "Apache-2.0",
        "used_by": ["multilingual"],
    },
}

#: Non-dataset assets required by builders (reference tokenizer).
TOOLING: Dict[str, Dict[str, str]] = {
    "tokenizer": {
        "name": "MiniCPM5-1B tokenizer",
        "source_url": "https://huggingface.co/openbmb/MiniCPM5-1B",
        "license": "see model card",
        "purpose": "reference tokenizer for the declared content-token counts (no chat template)",
    },
}


def load_spec() -> Dict[str, Any]:
    return read_json(SPEC_PATH)


def files_for(dataset: str) -> List[Dict[str, Any]]:
    return [f for f in load_spec()["files"] if f["dataset"] == dataset]


def dataset_provenance(scope: str = "data") -> Dict[str, Any]:
    """Manifest-ready provenance: dataset -> source URL, license, files.

    ``scope`` is ``"data"`` (the five performance datasets) or ``"tooling"``
    (the reference tokenizer).
    """
    meta = DATASETS if scope == "data" else TOOLING
    out: Dict[str, Any] = {}
    for dataset, info in meta.items():
        files = files_for(dataset)
        if not files:
            continue
        out[dataset] = {
            "name": info["name"],
            "source_url": info["source_url"],
            "license": info["license"],
            "files": [
                {"path": f["path"], "bytes": f["bytes"], "sha256": f["sha256"]}
                for f in files
            ],
        }
        if "used_by" in info:
            out[dataset]["used_by"] = info["used_by"]
    return out


def raw_path_display(rel: str) -> str:
    """Stable repo-relative path used in manifests/docs (``dataset/raw/...``)."""
    return str(DATA_DIR.relative_to(Path(__file__).resolve().parents[3]) / "raw" / rel)


def tokenizer_entry() -> Dict[str, Any]:
    """The ``tokenizer/MiniCPM5-1B/tokenizer.json`` raw-spec entry."""
    return next(
        f for f in files_for("tokenizer") if f["path"].endswith("tokenizer.json")
    )


def required_paths() -> List[Path]:
    """Absolute paths of every raw file this builder reads (for the preflight)."""
    return [RAW_DIR / f["path"] for f in load_spec()["files"]]
