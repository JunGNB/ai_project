"""Dataset adapter registry.

Adapters are resolved lazily so that a missing optional dataset never breaks
commands that do not use it (e.g. ``compare``).
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Dict, List, Optional, Type

from .base import DatasetAdapter

#: dataset name -> (module, class name)
_REGISTRY: Dict[str, tuple[str, str]] = {
    # Primary runtime source: the pre-constructed performance test set under
    # dataset/test/performance/ (one scenario slice per configured scenario).
    "perf_jsonl": ("modelbench.adapters.datasets.perf_jsonl", "PerfJSONLAdapter"),
    "longbench_v2": ("modelbench.adapters.datasets.longbench_v2", "LongBenchV2Adapter"),
    "mmlu": ("modelbench.adapters.datasets.mmlu", "MMLUAdapter"),
    "gsm8k": ("modelbench.adapters.datasets.gsm8k", "GSM8KAdapter"),
    "ifeval": ("modelbench.adapters.datasets.ifeval", "IFEvalAdapter"),
    "humaneval": ("modelbench.adapters.datasets.humaneval", "HumanEvalAdapter"),
    "mmmlu": ("modelbench.adapters.datasets.mmmlu", "MMMLUAdapter"),
    "hotpotqa": ("modelbench.adapters.datasets.hotpotqa", "HotpotQAAdapter"),
    "bfcl": ("modelbench.adapters.datasets.bfcl", "BFCLAdapter"),
    "synthetic": ("modelbench.adapters.datasets.synthetic", "SyntheticThroughputAdapter"),
    "custom": ("modelbench.adapters.datasets.custom", "CustomJSONLAdapter"),
}


def available_datasets() -> List[str]:
    return sorted(_REGISTRY)


def adapter_class(name: str) -> Type[DatasetAdapter]:
    try:
        module_name, class_name = _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown dataset '{name}'; available: {', '.join(available_datasets())}"
        ) from exc
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - depends on optional files
        raise RuntimeError(
            f"dataset '{name}' adapter module is missing ({module_name}); "
            "run `python -m modelbench.interfaces.cli validate` for details"
        ) from exc
    return getattr(module, class_name)


def get_adapter(
    name: str,
    raw_dir: Path,
    processed_dir: Path,
    params: Optional[Dict] = None,
    testset_dir: Optional[Path] = None,
) -> DatasetAdapter:
    """Instantiate an adapter with ``raw_dir`` resolved to its own subdirectory.

    Convention: every upstream dataset owns ``dataset/raw/<dataset>/`` and the
    adapter receives that directory (so adapters never hardcode their own name
    twice).  ``testset_dir`` is the configured ``dataset/test`` root and is only
    used by adapters that read pre-constructed test sets (``perf_jsonl``).
    """
    dataset_raw = Path(raw_dir) / name
    dataset_raw.mkdir(parents=True, exist_ok=True)
    return adapter_class(name)(
        raw_dir=dataset_raw,
        processed_dir=processed_dir,
        params=params or {},
        testset_dir=testset_dir,
    )
