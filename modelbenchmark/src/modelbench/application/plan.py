"""Run planning: apply a test set and describe the performance plan.

Pure configuration transformation plus an optional, cached dataset plan
(``DatasetAdapter.plan()``) so the CLI/Web can show how many measured requests
and cells the current selection implies.  There is no strategy layer and no
context-length matrix: the plan is ``models x (scenario, dataset, bucket) cells``
with a configurable sample count per scenario.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from modelbench.adapters.config.settings import ProjectConfig, ScenarioSpec
from modelbench.adapters.config.testsets import TestSet
from modelbench.adapters.datasets.registry import get_adapter
from modelbench.domain.context_windows import describe_windows


def apply_testset(cfg: ProjectConfig, testset: TestSet) -> ProjectConfig:
    """Apply scenario/dataset overrides from a test set in place."""
    for scenario_name, override in testset.scenarios.items():
        if scenario_name not in cfg.scenarios:
            raise ValueError(
                f"testset {testset.name} references unknown scenario: {scenario_name}"
            )
        spec = cfg.scenarios[scenario_name].model_copy(deep=True)
        if override.dataset:
            spec.dataset = override.dataset
        params = dict(spec.params or {})
        dataset_params = dict(params.get("dataset_params") or {})
        dataset_params.update(override.dataset_params)
        if dataset_params:
            params["dataset_params"] = dataset_params
        spec.params = params
        cfg.scenarios[scenario_name] = spec
    if testset.scenarios:
        cfg.selection.scenarios = list(testset.scenarios)
    return cfg


def select(
    cfg: ProjectConfig,
    *,
    models: Optional[List[str]] = None,
    scenarios: Optional[List[str]] = None,
) -> ProjectConfig:
    """Apply explicit model/scenario selection (CLI/Web); mutates and returns cfg."""
    if models:
        cfg.selection.models = list(models)
    if scenarios:
        cfg.selection.scenarios = list(scenarios)
    return cfg


def scenario_label(name: str, spec: ScenarioSpec) -> str:
    """Human-readable dataset label, distinguishing shared-adapter slices.

    Eight performance scenarios all use ``dataset: perf_jsonl``; the meaningful
    identity is the ``dataset_params.scenario`` slice, so the label becomes
    ``knowledge_qa=perf_jsonl[knowledge_qa]`` instead of eight identical names.
    """
    dataset_params = spec.params.get("dataset_params") or {}
    slice_name = dataset_params.get("scenario") or dataset_params.get("scenario_name")
    if slice_name:
        return f"{name}={spec.dataset}[{slice_name}]"
    return f"{name}={spec.dataset}"


def dataset_plan(
    cfg: ProjectConfig,
    spec: ScenarioSpec,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Cached adapter plan for one scenario slice (``None`` when unknown).

    ``extra`` carries run-level selectors (``context_windows``/``samples``/
    ``sample_ids``) so adapters that support planning can compute the exact
    cells the runner will execute under the current UI/CLI selections.
    """
    params = dict(spec.params.get("dataset_params") or {})
    if extra:
        params.update({key: value for key, value in extra.items() if value is not None})
    try:
        adapter = get_adapter(
            spec.dataset,
            raw_dir=Path(cfg.datasets.raw_dir),
            processed_dir=Path(cfg.datasets.processed_dir),
            params=params,
            testset_dir=Path(cfg.datasets.testset_dir),
        )
        return adapter.plan(extra or None)
    except (OSError, ValueError, RuntimeError):
        return None


def scenario_plan_extras(cfg: ProjectConfig, name: str) -> Dict[str, Any]:
    """Run selectors that affect which samples/cells a scenario will execute."""
    explicit = cfg.perf.sample_ids_for(name)
    count = cfg.perf.samples_for(name)
    extra: Dict[str, Any] = {}
    if cfg.perf.context_windows:
        extra["context_windows"] = list(cfg.perf.context_windows)
    if explicit:
        extra["sample_ids"] = list(explicit)
    elif count is not None:
        extra["samples"] = int(count)
    return extra


def plan_overview(cfg: ProjectConfig) -> Dict[str, Any]:
    """Describe the resolved plan; uses dataset plans when sample counts are open."""
    models = cfg.selected_models()
    scenarios = cfg.selected_scenarios()
    samples = cfg.perf.samples_per_dataset
    repeats = int(cfg.run.repeats)
    window_filtered = bool(cfg.perf.context_windows)

    scenario_rows: List[Dict[str, Any]] = []
    plan_groups: List[Dict[str, Any]] = []
    total_samples = 0
    total_groups = 0
    #: True only while every scenario's size/cell split is known analytically.
    cells_known = True
    samples_known = True
    for name, spec in scenarios.items():
        explicit = cfg.perf.sample_ids_for(name)
        count = cfg.perf.samples_for(name)
        row: Dict[str, Any] = {
            "name": name,
            "dataset": spec.dataset,
            "label": scenario_label(name, spec),
            "samples_total": None,
            "cells_total": None,
        }
        groups = dataset_plan(cfg, spec, scenario_plan_extras(cfg, name))
        if groups is not None:
            row["samples_total"] = sum(int(item["n"]) for item in groups)
            row["cells_total"] = len(groups)
            total_samples += row["samples_total"]
            total_groups += len(groups)
            plan_groups.extend({"scenario": name, **item} for item in groups)
        elif explicit:
            row["samples_total"] = len(explicit)
            total_samples += len(explicit)
            cells_known = False
        elif count is not None:
            row["samples_total"] = int(count)
            total_samples += int(count)
            cells_known = False
        else:
            samples_known = False
            cells_known = False
        scenario_rows.append(row)

    if window_filtered and not cells_known:
        # Legacy adapters cannot plan tokenizer-filtered windows.
        samples_known = False
    request_estimate: Optional[int] = None
    if samples_known and total_samples:
        request_estimate = total_samples * len(models) * repeats
    cells_estimate: Optional[int] = None
    if cells_known and total_groups:
        cells_estimate = total_groups * len(models)

    return {
        "mode": "performance",
        "models": [m.label for m in models],
        "scenarios": scenario_rows,
        "groups": plan_groups,
        "cells_estimate": cells_estimate,
        "samples_total": total_samples if samples_known else None,
        "samples_per_dataset": samples,
        "samples_label": "全部可用样本" if samples is None else str(samples),
        "max_tokens": cfg.perf.max_tokens,
        "context_windows": list(cfg.perf.context_windows or []),
        "context_filter": describe_windows(cfg.perf.context_windows or []),
        "repeats": repeats,
        "warmup_requests": cfg.run.warmup_requests,
        "warmup_per_model": cfg.run.warmup_per_model,
        "cache_mode": cfg.run.cache_mode.value,
        "thinking_mode": cfg.run.thinking_mode,
        "request_estimate": request_estimate,
        "max_requests": cfg.run.max_requests,
        "dataset_dir": cfg.datasets.testset_dir,
    }
