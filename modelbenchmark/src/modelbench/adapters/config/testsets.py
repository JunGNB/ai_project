"""Named test sets under ``dataset/test/``.

A test set is a small YAML overlay: it selects scenarios and may swap the
dataset/params for a scenario (e.g. point ``math_logic`` at a custom JSONL).
The performance samples live under ``dataset/test/performance/``; user-provided
samples can live in any directory under ``dataset/test/<name>/`` and are
referenced by relative path.

The default root is ``dataset/test``; every caller (CLI / validate / Web) reads
it from ``datasets.testset_dir`` in ``default.yaml``.

Example (``dataset/test/currency_test/testset.yaml``)::

    name: currency_test
    title: 货币换算（自定义样本）
    scenarios:
      math_logic:
        dataset: custom
        dataset_params:
          path: dataset/test/currency_test/currency.jsonl
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

from modelbench.paths import PROJECT_ROOT

TESTSET_DIR = PROJECT_ROOT / "dataset" / "test"


class TestSetScenario(BaseModel):
    dataset: Optional[str] = None
    dataset_params: Dict[str, Any] = Field(default_factory=dict)


class TestSet(BaseModel):
    name: str
    title: Optional[str] = None
    description: Optional[str] = None
    scenarios: Dict[str, TestSetScenario] = Field(default_factory=dict)


def load_testsets(directory: Optional[Path] = None) -> Dict[str, TestSet]:
    root = Path(directory or TESTSET_DIR)
    out: Dict[str, TestSet] = {}
    if not root.is_dir():
        return out
    for path in sorted(root.glob("*/testset.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError(f"invalid testset {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"invalid testset {path}: root must be a mapping")
        data.setdefault("name", path.parent.name)
        testset = TestSet(**data)
        if testset.name in out:
            raise ValueError(f"duplicate testset name: {testset.name}")
        out[testset.name] = testset
    return out


def available_testsets(directory: Optional[Path] = None) -> List[str]:
    return sorted(load_testsets(directory))
