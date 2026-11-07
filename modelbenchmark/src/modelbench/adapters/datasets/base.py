"""Dataset adapter contract.

A dataset adapter has exactly two responsibilities:

1. ``ensure_downloaded()`` — make the raw files exist locally (idempotent,
   resumable, verifying checksums) and return their root path.
2. ``load()`` — normalize raw files into :class:`modelbench.domain.models.Sample` objects.

Adapters must not build prompts or score answers; that belongs to the scenario
layer.  Adapters must be deterministic: repeated ``load()`` calls with the same
config return the same samples in the same order.
"""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from ...domain.models import Sample

from modelbench.paths import PROJECT_ROOT as _PROJECT_ROOT


def resolve_project_path(value: str | Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return path if path.is_absolute() else (_PROJECT_ROOT / path)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    tmp.replace(path)
    return count


class DatasetAdapter(ABC):
    """Base class for all dataset adapters."""

    #: unique dataset id, e.g. ``"longbench_v2"``, ``"mmlu"``
    name: str = ""
    #: default scenario this dataset feeds (may be overridden by scenario config)
    scenario: str = ""
    #: canonical upstream identity for docs / manifest
    source: str = ""
    #: license note for the docs
    license: str = ""

    def __init__(
        self,
        raw_dir: Path,
        processed_dir: Path,
        params: Optional[Dict[str, Any]] = None,
        testset_dir: Optional[Path] = None,
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.params = dict(params or {})
        #: Root of ``dataset/test`` (resolved from ``datasets.testset_dir``).
        self.testset_dir = Path(testset_dir) if testset_dir is not None else None

    # ---- required -------------------------------------------------------

    @abstractmethod
    def ensure_downloaded(self) -> Path:
        """Download/verify raw data; return the raw root directory."""

    @abstractmethod
    def load(self) -> List[Sample]:
        """Load and normalize all requested samples."""

    # ---- optional -------------------------------------------------------

    def plan(self, params: Optional[Dict[str, Any]] = None) -> Optional[List[Dict[str, Any]]]:
        """Describe the cells this adapter would produce without materialising samples.

        Returns ``None`` when the adapter cannot describe its selection cheaply
        (the caller then reports the plan as unknown).  Adapters that do
        implement it return a list of::

            {"scenario": str, "dataset": str, "context_length": int|None, "n": int}

        one row per aggregate cell, in load order.  The result must match what
        ``load()`` produces for the same params (same filtering/limit).
        """
        return None

    def processed_path(self) -> Path:
        return self.processed_dir / f"{self.name}.jsonl"

    def save_processed(self, samples: List[Sample]) -> Path:
        path = self.processed_path()
        write_jsonl(path, [s.model_dump(mode="json") for s in samples])
        return path

    def load_processed(self) -> Optional[List[Sample]]:
        path = self.processed_path()
        if not path.is_file():
            return None
        return [Sample(**row) for row in read_jsonl(path)]

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "scenario": self.scenario,
            "source": self.source,
            "license": self.license,
            "raw_dir": str(self.raw_dir),
            "params": self.params,
        }
