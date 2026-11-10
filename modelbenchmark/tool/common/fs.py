#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared filesystem / JSONL / hashing helpers for ``tool/``.

Kept dependency-free (standard library only) so both the offline dataset
builders and the evaluation-side tools can import it before ``.venv`` exists.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List

#: Repository root (``tool/common/fs.py`` -> ``tool/common`` -> ``tool`` -> root).
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "dataset"
RAW_DIR = DATA_DIR / "raw"
TEST_DIR = DATA_DIR / "test"
OUTPUT_DIR = ROOT / "output"
#: Dataset-build intermediate reports (selection/pilot).  They are NOT benchmark
#: results and must not pollute the schema-3.0 ``output/`` evidence tree.
REPORTS_DIR = ROOT / "reports" / "dataset-build"
TOOL_DIR = ROOT / "tool"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    """Atomic JSON write (tmp + replace) so a failed build never leaves a torn file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
