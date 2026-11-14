"""Run planning: apply a test set and describe the resolved plan.

Pure configuration transformation plus an optional, cached dataset plan
(``DatasetAdapter.plan()``) so the CLI/Web can show how many measured requests
and cells the current selection implies.  There is no strategy layer and no
context-length matrix: the plan is ``models x scenario cells`` with a
configurable sample count per scenario.

The overview is dispatched through the method registry
(:mod:`modelbench.application.methods`): performance keeps the historical
``(scenario, dataset, context_length)`` cells and ISL filter, accuracy reports
one cell per scenario (``scenario -> n samples``) and no context filter.  Both
return the same key set so the Web/CLI can render either without branching.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from modelbench.adapters.config.settings import ProjectConfig, ScenarioSpec
from modelbench.adapters.config.testsets import TestSet
from modelbench.adapters.datasets.registry import get_adapter
from modelbench.domain.context_windows import describe_windows
from modelbench.domain.method import RunMethod


def apply_testset(
    cfg: ProjectConfig,
    testset: TestSet,
    *,
    explicit_method: Optional[Union[str, RunMethod]] = None,
) -> ProjectConfig:
    """Apply scenario/dataset overrides and the test set's method in place.

    The test set is the declarative owner of the method (an accuracy test set
    always runs the accuracy protocol).  ``explicit_method`` is the value the
    caller *asked for* (CLI ``--method`` / Web card); if the two disagree the
    call fails instead of silently changing the measurement calibre
    (design §4.1).  ``cfg.method`` cannot be used for the check because the
    default value is indistinguishable from an explicit one.

    After the scenario overrides are applied the method's fixed protocol is
    enforced via ``MethodProfile.configure`` (accuracy: cold/repeats=1/...).
    """
    if explicit_method is not None:
        requested = RunMethod.parse(explicit_method)
        if requested is not testset.method:
            raise ValueError(
                f"testset {testset.name!r} is method={testset.method.value!r} but "
                f"method={requested.value!r} was requested; choose a matching "
                "test set or drop the explicit --method"
            )

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

    cfg.method = testset.method
    # A performance test set may declare its generation length (OSL): the samples
    # were built for that answer budget, so the default 50 must not silently apply.
    if testset.max_tokens is not None:
        if testset.method is not RunMethod.PERFORMANCE:
            raise ValueError(
                f"testset {testset.name!r}: max_tokens is only valid for performance "
                "sets (accuracy generation length comes from judge.max_new_tokens)"
            )
        cfg.perf.max_tokens = int(testset.max_tokens)
    # Strict input-token alignment is a performance-only contract: the set
    # guarantees exact bucket targets with the target model tokenizer.  Always
    # assign it (including False) so switching test sets cannot leave a stale
    # strict flag behind on a re-used config.
    if testset.strict_input_tokens and testset.method is not RunMethod.PERFORMANCE:
        raise ValueError(
            f"testset {testset.name!r}: strict_input_tokens is only valid for "
            "performance sets (accuracy prompt length is not a bucket contract)"
        )
    cfg.perf.strict_input_tokens = bool(testset.strict_input_tokens)
    # Deferred import: methods imports the plan module for its ``plan`` hook.
    from modelbench.application.methods import get_method

    get_method(cfg.method).configure(cfg)
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


def _method_of(cfg: ProjectConfig) -> RunMethod:
    """Coerce ``cfg.method`` (typed enum, or a plain string after mutation)."""
    return RunMethod.parse(cfg.method)


def scenario_plan_extras(cfg: ProjectConfig, name: str) -> Dict[str, Any]:
    """Run selectors that affect which samples/cells a scenario will execute."""
    method = _method_of(cfg)
    block = cfg.accuracy if method is RunMethod.ACCURACY else cfg.perf
    explicit = block.sample_ids_for(name)
    count = block.samples_for(name)
    extra: Dict[str, Any] = {}
    if method is RunMethod.PERFORMANCE and cfg.perf.context_windows:
        extra["context_windows"] = list(cfg.perf.context_windows)
    if explicit:
        extra["sample_ids"] = list(explicit)
    elif count is not None:
        extra["samples"] = int(count)
    return extra


# --------------------------------------------------------------------------- #
# Performance plan (historical shape; unchanged behaviour)
# --------------------------------------------------------------------------- #


def plan_overview_performance(cfg: ProjectConfig) -> Dict[str, Any]:
    """Describe the resolved performance plan (unchanged historical shape)."""
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
        "mode": RunMethod.PERFORMANCE.value,
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


# --------------------------------------------------------------------------- #
# Accuracy plan
# --------------------------------------------------------------------------- #


def _accuracy_judge_families(cfg: ProjectConfig) -> List[str]:
    """Judge families the accuracy test set declares (manifest, else all five)."""
    from modelbench.domain.judges import JUDGE_FAMILIES

    testset_dir = Path(str(getattr(cfg.datasets, "testset_dir", "dataset/test")))
    manifest_path = testset_dir / "accuracy" / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return list(JUDGE_FAMILIES)
    families = (manifest or {}).get("judges")
    if isinstance(families, dict) and families:
        return sorted(str(name) for name in families)
    return list(JUDGE_FAMILIES)


def plan_overview_accuracy(cfg: ProjectConfig) -> Dict[str, Any]:
    """Describe the resolved accuracy plan: one cell per scenario, no buckets."""
    models = cfg.selected_models()
    scenarios = cfg.selected_scenarios()
    samples = cfg.accuracy.samples_per_scenario
    repeats = int(cfg.run.repeats)

    scenario_rows: List[Dict[str, Any]] = []
    plan_groups: List[Dict[str, Any]] = []
    total_samples = 0
    total_groups = 0
    samples_known = True
    cells_known = True
    for name, spec in scenarios.items():
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
        else:
            explicit = cfg.accuracy.sample_ids_for(name)
            count = cfg.accuracy.samples_for(name)
            if explicit:
                row["samples_total"] = len(explicit)
                total_samples += len(explicit)
            elif count is not None:
                row["samples_total"] = int(count)
                total_samples += int(count)
            else:
                samples_known = False
            # Legacy/unavailable adapter: one cell per scenario is still the
            # contract for accuracy (no dataset/bucket split exists).
            cells_known = False
        scenario_rows.append(row)

    request_estimate: Optional[int] = None
    if samples_known and total_samples:
        request_estimate = total_samples * len(models) * repeats
    cells_estimate: Optional[int] = None
    if cells_known and total_groups:
        cells_estimate = total_groups * len(models)

    return {
        "mode": RunMethod.ACCURACY.value,
        "models": [m.label for m in models],
        "scenarios": scenario_rows,
        "groups": plan_groups,
        "cells_estimate": cells_estimate,
        "samples_total": total_samples if samples_known else None,
        "samples_per_scenario": samples,
        "samples_label": "全部可用样本" if samples is None else str(samples),
        "judge_families": _accuracy_judge_families(cfg),
        "needs_judges": True,
        "context_windows": [],
        "context_filter": describe_windows([]),
        "repeats": repeats,
        "warmup_requests": cfg.run.warmup_requests,
        "warmup_per_model": cfg.run.warmup_per_model,
        "cache_mode": cfg.run.cache_mode.value,
        "thinking_mode": cfg.run.thinking_mode,
        "request_estimate": request_estimate,
        "max_requests": cfg.run.max_requests,
        "dataset_dir": cfg.datasets.testset_dir,
    }


def plan_overview(cfg: ProjectConfig) -> Dict[str, Any]:
    """Dispatch the overview to the configured method (design §4.2)."""
    from modelbench.application.methods import get_method

    return get_method(_method_of(cfg)).plan(cfg)


__all__ = [
    "apply_testset",
    "dataset_plan",
    "plan_overview",
    "plan_overview_accuracy",
    "plan_overview_performance",
    "scenario_label",
    "scenario_plan_extras",
    "select",
]
