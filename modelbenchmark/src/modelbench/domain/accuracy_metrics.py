"""Accuracy scoring normalization, aggregates and paired comparison.

This module is the *only* place where accuracy scores are aggregated.  Both the
live accuracy orchestrator and ``rebuild``/``verify`` (WP-3) recompute from
``records.jsonl`` through these functions, so the counting rules below are the
contract (AC-A5/A6/A7):

Counting rules
--------------
1. Only ``phase == measured`` records are counted (warmups never are).
2. ``status`` taxonomy: ``ok`` | ``judge_error`` | ``error`` | ``timeout``.
   ``error``/``timeout`` are **not** in the score denominator; they are reported
   separately (``n_error``/``n_timeout``) so the error rate is visible.
3. ``judge_error`` is counted as **0 and stays in the denominator**: dropping it
   would inflate the score, which the accuracy plan forbids.  A record that the
   runner persisted as ``status=ok`` with ``score.detail.judge_error=true`` is
   also forced to 0.  Hence ``n_scored = n_ok + n_status_judge_error``; when the
   runner uses the ``status=ok`` + detail-flag shape this is exactly ``n_ok``.
4. ``n_valid = n_ok + n_status_judge_error`` (measured requests that produced a
   judgment attempt); ``n_missing_score`` counts ``status=ok`` rows with no
   score at all (contract violation, scored as 0 and surfaced).
5. ``n_correct`` counts scored rows whose normalized score is exactly 1.0.
6. A cell is keyed by ``(model, scenario, dataset)``; there is one overall row
   per model.
7. ``macro_score`` = equal-weight mean of scenario scores (each scenario score
   first averages its own scored rows, so a scenario with two datasets cannot
   outweigh a single-dataset scenario).  ``micro_score`` = all-row mean
   (auxiliary only).
8. CIs: Wilson score interval for 0/1 accuracy, normal approximation
   (``mean ± 1.96·std/√n``) for the score; both are null when ``n < 2``.
9. Aggregates are pure functions of the records: every count and mean can be
   recomputed from ``records.jsonl`` (no hidden state, no filtering by score).

Paired comparison (``paired_accuracy``) pairs two runs by ``(scenario,
sample_id)``: per-scenario ``Δ(pp)``, exact McNemar for 0/1 scenarios, a
deterministic paired bootstrap for F1 scenarios, and Holm correction across
scenarios.  Samples present in only one run (``unpaired``) or not scored on both
sides (``unscored``) are listed explicitly — never silently dropped.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from statistics import fmean, stdev
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .judges import BINARY_FAMILIES, BINARY_METRICS, is_binary
from .models import (
    AccuracyCellAggregate,
    AccuracyOverallAggregate,
    AccuracySampleResult,
    AccuracySlice,
)
from .models import RecordStatus

#: Statuses that produced a judgment attempt and are in the score denominator.
_VALID_STATUSES = frozenset({RecordStatus.OK.value, RecordStatus.JUDGE_ERROR.value})
#: Statuses excluded from the denominator (reported separately).
_EXCLUDED_STATUSES = frozenset(
    {RecordStatus.ERROR.value, RecordStatus.TIMEOUT.value,
     RecordStatus.CONTAMINATED.value, RecordStatus.WARMUP.value}
)

_EPS = 1e-9


# --------------------------------------------------------------------------- #
# record access helpers (SampleRecord objects or plain dicts from jsonl)
# --------------------------------------------------------------------------- #


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _phase(record: Any) -> str:
    return str(_enum_value(_get(record, "phase", "measured")))


def _status(record: Any) -> str:
    return str(_enum_value(_get(record, "status", "ok")))


def _detail(record: Any) -> Dict[str, Any]:
    score = _get(record, "score")
    if score is None:
        return {}
    detail = _get(score, "detail", None)
    return dict(detail) if isinstance(detail, Mapping) else {}


def _raw_score(record: Any) -> Optional[float]:
    score = _get(record, "score")
    if score is None:
        return None
    value = _get(score, "score", None)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


def _sample_meta(record: Any) -> Dict[str, Any]:
    meta = _get(record, "sample_meta", None)
    return dict(meta) if isinstance(meta, Mapping) else {}


def judge_meta(record: Any) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(judge_family, metric)`` for a record (detail first, then meta)."""
    detail = _detail(record)
    meta = _sample_meta(record)
    judge = meta.get("judge") if isinstance(meta.get("judge"), Mapping) else {}
    family = detail.get("family") or (judge or {}).get("family")
    family = str(family) if family else None
    metric = detail.get("metric") or (judge or {}).get("metric")
    if family == "normalized_match" and not metric:
        metric = "f1"
    if not metric and family in BINARY_FAMILIES:
        metric = "accuracy"
    return family, (str(metric) if metric else None)


def language_of(record: Any) -> str:
    meta = _sample_meta(record)
    value = meta.get("language") or meta.get("lang")
    return str(value) if value else "unknown"


def dataset_of(record: Any) -> str:
    return str(_get(record, "dataset", "") or "")


def scenario_of(record: Any) -> str:
    return str(_get(record, "scenario", "") or "")


def model_of(record: Any) -> str:
    return str(_get(record, "model", "") or "")


# --------------------------------------------------------------------------- #
# score normalization
# --------------------------------------------------------------------------- #


def normalize_judge_score(family: Optional[str], metric: Optional[str], score: Any) -> float:
    """Clamp a judge score into ``[0, 1]`` (0/1 families become exactly 0 or 1).

    ``None``/NaN/non-numeric values normalize to 0.0 (fail closed).  A 0/1 family
    or metric uses a 0.5 threshold so a float artifact cannot become a false
    "correct"; F1 metrics keep their value, clamped to ``[0, 1]``.
    """
    try:
        value = float(score)
    except (TypeError, ValueError):
        return 0.0
    if value != value:  # NaN
        return 0.0
    if is_binary(family, metric):
        return 1.0 if value >= 0.5 else 0.0
    return min(1.0, max(0.0, value))


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Optional[Tuple[float, float]]:
    """Wilson score interval for a binomial proportion (``None`` when n < 2)."""
    if n is None or n < 2:
        return None
    successes = min(max(int(successes), 0), int(n))
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def normal_interval(values: Sequence[float], z: float = 1.96) -> Optional[Tuple[float, float]]:
    """Normal-approximation CI of the mean (``None`` when n < 2)."""
    n = len(values)
    if n < 2:
        return None
    mean = fmean(values)
    std = stdev(values)
    half = z * std / math.sqrt(n)
    return (max(0.0, mean - half), min(1.0, mean + half))


# --------------------------------------------------------------------------- #
# one measured record -> one classified row
# --------------------------------------------------------------------------- #


@dataclass
class _Measured:
    model: str
    scenario: str
    dataset: str
    sample_id: str
    language: str
    status: str
    family: Optional[str]
    metric: Optional[str]
    binary: bool
    #: Score used by aggregation (None = excluded from denominator).
    value: Optional[float]
    correct: bool
    format_failure: bool
    judge_error: bool
    missing_score: bool


def _measure(record: Any) -> Optional[_Measured]:
    """Classify one record; ``None`` when it is not a measured row."""
    if _phase(record) != "measured":
        return None
    status = _status(record)
    if status in _EXCLUDED_STATUSES:
        value: Optional[float] = None
        correct = False
    else:
        detail = _detail(record)
        family, metric = judge_meta(record)
        judge_error = status == RecordStatus.JUDGE_ERROR.value or bool(detail.get("judge_error"))
        raw = _raw_score(record)
        missing_score = raw is None
        if judge_error or missing_score:
            value = 0.0
        else:
            value = normalize_judge_score(family, metric, raw)
        correct = bool(value is not None and value >= 1.0 - _EPS)
        return _Measured(
            model=model_of(record),
            scenario=scenario_of(record),
            dataset=dataset_of(record),
            sample_id=str(_get(record, "sample_id", "") or ""),
            language=language_of(record),
            status=status,
            family=family,
            metric=metric,
            binary=is_binary(family, metric),
            value=value,
            correct=correct,
            format_failure=bool(detail.get("format_failure")),
            judge_error=bool(judge_error),
            missing_score=bool(missing_score and not judge_error),
        )
    return _Measured(
        model=model_of(record),
        scenario=scenario_of(record),
        dataset=dataset_of(record),
        sample_id=str(_get(record, "sample_id", "") or ""),
        language=language_of(record),
        status=status,
        family=judge_meta(record)[0],
        metric=judge_meta(record)[1],
        binary=False,
        value=value,
        correct=correct,
        format_failure=False,
        judge_error=False,
        missing_score=False,
    )


def _stats(rows: Sequence[_Measured]) -> Dict[str, Any]:
    """Counts/means/CIs for one slice (documented counting rules at module top)."""
    n_samples = len(rows)
    n_ok = sum(1 for row in rows if row.status == RecordStatus.OK.value)
    n_status_judge = sum(1 for row in rows if row.status == RecordStatus.JUDGE_ERROR.value)
    n_error = sum(1 for row in rows if row.status == RecordStatus.ERROR.value)
    n_timeout = sum(1 for row in rows if row.status == RecordStatus.TIMEOUT.value)
    n_judge_error = sum(1 for row in rows if row.judge_error)
    n_format_failure = sum(1 for row in rows if row.format_failure)
    n_missing_score = sum(1 for row in rows if row.missing_score)
    scored = [row for row in rows if row.value is not None]
    values = [float(row.value) for row in scored]
    binary = [row for row in scored if row.binary]
    n_binary = len(binary)
    n_binary_correct = sum(1 for row in binary if row.correct)
    n_correct = sum(1 for row in scored if row.correct)

    mean_score = fmean(values) if values else None
    score_std = stdev(values) if len(values) >= 2 else None
    score_ci = normal_interval(values)
    accuracy = (n_binary_correct / n_binary) if n_binary else None
    accuracy_ci = wilson_interval(n_binary_correct, n_binary)
    return {
        "n_samples": n_samples,
        "n_valid": n_ok + n_status_judge,
        "n_ok": n_ok,
        "n_error": n_error,
        "n_timeout": n_timeout,
        "n_judge_error": n_judge_error,
        "n_format_failure": n_format_failure,
        "n_missing_score": n_missing_score,
        "n_scored": len(scored),
        "n_correct": n_correct,
        "mean_score": mean_score,
        "accuracy": accuracy,
        "score_std": score_std,
        "score_ci95_low": score_ci[0] if score_ci else None,
        "score_ci95_high": score_ci[1] if score_ci else None,
        "accuracy_ci95_low": accuracy_ci[0] if accuracy_ci else None,
        "accuracy_ci95_high": accuracy_ci[1] if accuracy_ci else None,
    }


def _slice(rows: Sequence[_Measured]) -> AccuracySlice:
    return AccuracySlice(**_stats(rows))


def _one_or_mixed(values: Iterable[Optional[str]]) -> Optional[str]:
    unique = sorted({value for value in values if value})
    if not unique:
        return None
    return unique[0] if len(unique) == 1 else "mixed"


# --------------------------------------------------------------------------- #
# aggregates
# --------------------------------------------------------------------------- #


def aggregate_accuracy(
    records: Iterable[Any],
) -> Tuple[List[AccuracyCellAggregate], List[AccuracyOverallAggregate]]:
    """Recompute accuracy cells and overall rows from measured records.

    Returns ``(cells, overall)`` where ``cells`` is keyed by
    ``(model, scenario, dataset)`` sorted in that order and ``overall`` holds one
    row per model.  See the module docstring for the counting rules; every field
    is a pure function of ``records`` (AC-A5).
    """
    measured: List[_Measured] = []
    for record in records:
        row = _measure(record)
        if row is not None:
            measured.append(row)

    cells: Dict[Tuple[str, str, str], List[_Measured]] = {}
    overall: Dict[str, List[_Measured]] = {}
    for row in measured:
        cells.setdefault((row.model, row.scenario, row.dataset), []).append(row)
        overall.setdefault(row.model, []).append(row)

    cell_rows: List[AccuracyCellAggregate] = []
    for (model, scenario, dataset) in sorted(cells):
        rows = cells[(model, scenario, dataset)]
        by_language: Dict[str, List[_Measured]] = {}
        for row in rows:
            by_language.setdefault(row.language, []).append(row)
        cell_rows.append(
            AccuracyCellAggregate(
                model=model,
                scenario=scenario,
                dataset=dataset,
                judge_family=_one_or_mixed(row.family for row in rows),
                metric=_one_or_mixed(row.metric for row in rows),
                by_language={
                    language: _slice(group)
                    for language, group in sorted(by_language.items())
                },
                **_stats(rows),
            )
        )

    overall_rows: List[AccuracyOverallAggregate] = []
    for model in sorted(overall):
        rows = overall[model]
        by_scenario: Dict[str, List[_Measured]] = {}
        by_language: Dict[str, List[_Measured]] = {}
        for row in rows:
            by_scenario.setdefault(row.scenario, []).append(row)
            by_language.setdefault(row.language, []).append(row)
        scenario_scores: Dict[str, float] = {}
        scenario_accuracy: Dict[str, float] = {}
        for scenario, group in sorted(by_scenario.items()):
            stats = _stats(group)
            if stats["mean_score"] is not None:
                scenario_scores[scenario] = float(stats["mean_score"])
            if stats["accuracy"] is not None:
                scenario_accuracy[scenario] = float(stats["accuracy"])
        stats = _stats(rows)
        # ``by_language`` on the overall row is a profile (different row counts
        # per language), never a headline number.
        overall_rows.append(
            AccuracyOverallAggregate(
                model=model,
                n_scenarios_scored=len(scenario_scores),
                scenario_scores=scenario_scores,
                macro_score=(
                    fmean(scenario_scores.values()) if scenario_scores else None
                ),
                micro_score=stats["mean_score"],
                macro_accuracy=(
                    fmean(scenario_accuracy.values()) if scenario_accuracy else None
                ),
                by_language={
                    language: _slice(group)
                    for language, group in sorted(by_language.items())
                },
                judge_families=sorted({row.family for row in rows if row.family}),
                **stats,
            )
        )
    return cell_rows, overall_rows


# --------------------------------------------------------------------------- #
# per-sample table
# --------------------------------------------------------------------------- #


def accuracy_sample_results(records: Iterable[Any]) -> List[AccuracySampleResult]:
    """Build one :class:`AccuracySampleResult` per measured record (CSV shape).

    ``error``/``timeout`` rows keep ``score=None``; ``judge_error`` rows are 0.0
    with ``judge_error=True`` (judged, wrong by contract, still counted).
    """
    out: List[AccuracySampleResult] = []
    for record in records:
        if _phase(record) != "measured":
            continue
        status = _status(record)
        family, metric = judge_meta(record)
        detail = _detail(record)
        meta = _sample_meta(record)
        request = _get(record, "request")
        metrics = _get(record, "metrics")
        response = _get(record, "response")
        usage = _get(response, "usage") if response is not None else None
        cache = _get(record, "cache")
        raw = _raw_score(record)
        if status in _EXCLUDED_STATUSES:
            score: Optional[float] = None
            correct: Optional[bool] = None
            judge_error = False
        else:
            judge_error = status == RecordStatus.JUDGE_ERROR.value or bool(
                detail.get("judge_error")
            )
            score = 0.0 if (judge_error or raw is None) else normalize_judge_score(family, metric, raw)
            correct = bool(score is not None and score >= 1.0 - _EPS)
        out.append(
            AccuracySampleResult(
                model=model_of(record),
                model_id=_get(record, "model", None),
                scenario=scenario_of(record),
                dataset=dataset_of(record),
                language=meta.get("language") or meta.get("lang"),
                sample_id=str(_get(record, "sample_id", "") or ""),
                upstream_id=meta.get("upstream_id"),
                repeat=int(_get(record, "repeat", 0) or 0),
                phase=_phase(record),
                status=status,
                error=_get(record, "error", None),
                judge_family=family,
                judge_metric=metric,
                score=score,
                correct=correct,
                format_failure=bool(detail.get("format_failure")),
                judge_error=bool(judge_error),
                max_new_tokens=_get(request, "max_tokens", None) if request is not None else None,
                completion_tokens=(
                    _get(metrics, "completion_tokens", None) if metrics is not None else None
                ),
                input_context_tokens=(
                    _get(metrics, "input_context_tokens", None) if metrics is not None else None
                ),
                actual_prompt_tokens=(
                    _get(request, "actual_prompt_tokens", None) if request is not None else None
                ),
                ttft_ms=_get(metrics, "ttft_ms", None) if metrics is not None else None,
                e2e_ms=_get(metrics, "e2e_ms", None) if metrics is not None else None,
                finish_reason=(
                    _get(response, "finish_reason", None) if response is not None else None
                ),
                cache_mode=str(_enum_value(_get(cache, "mode", "")) or "") or None
                if cache is not None
                else None,
                cache_cleared=(
                    _get(cache, "cleared_before_request", None) if cache is not None else None
                ),
                cache_polluted=_get(cache, "polluted", None) if cache is not None else None,
                prompt_sha256=(
                    _get(request, "prompt_sha256", None) if request is not None else None
                ),
                timestamp=_get(record, "timestamp", None),
            )
        )
    return out


# --------------------------------------------------------------------------- #
# paired comparison (same testset, two runs)
# --------------------------------------------------------------------------- #


def _mcnemar_exact(b: int, c: int) -> Optional[float]:
    """Exact two-sided McNemar p-value (binomial test, p=0.5)."""
    n = int(b) + int(c)
    if n == 0:
        return None
    k = min(int(b), int(c))
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile of an already sorted sequence."""
    if not sorted_values:
        raise ValueError("quantile of empty sequence")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = q * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(sorted_values) - 1)
    frac = position - lower
    return float(sorted_values[lower] * (1 - frac) + sorted_values[upper] * frac)


def _bootstrap_delta_ci(
    deltas: Sequence[float], n_boot: int, seed: int
) -> Optional[Tuple[float, float, int]]:
    """Deterministic paired bootstrap percentile CI of the mean difference."""
    n = len(deltas)
    if n_boot <= 0 or n < 2:
        return None
    rng = random.Random(seed)
    samples: List[float] = []
    for _ in range(int(n_boot)):
        # Resample pair indexes with replacement (paired, not independent).
        samples.append(fmean(deltas[rng.randrange(n)] for _ in range(n)))
    samples.sort()
    return (_quantile(samples, 0.025), _quantile(samples, 0.975), int(n_boot))


def holm_adjust(pvalues: Mapping[Any, Optional[float]]) -> Dict[Any, Optional[float]]:
    """Holm-Bonferroni step-down adjustment over the non-null p-values."""
    ordered = sorted(
        ((key, float(p)) for key, p in pvalues.items() if p is not None),
        key=lambda item: (item[1], str(item[0])),
    )
    m = len(ordered)
    adjusted: Dict[Any, Optional[float]] = {key: None for key in pvalues}
    running = 0.0
    for index, (key, p) in enumerate(ordered):
        value = min(1.0, (m - index) * p)
        running = max(running, value)
        adjusted[key] = running
    return adjusted


def _measured_index(records: Iterable[Any]) -> Tuple[Dict[Tuple[str, str], _Measured], List[Dict[str, Any]]]:
    index: Dict[Tuple[str, str], _Measured] = {}
    duplicates: List[Dict[str, Any]] = []
    for record in records:
        row = _measure(record)
        if row is None:
            continue
        key = (row.scenario, row.sample_id)
        if key in index:
            duplicates.append({"scenario": key[0], "sample_id": key[1]})
            continue
        index[key] = row
    return index, duplicates


def paired_accuracy(
    records_a: Iterable[Any],
    records_b: Iterable[Any],
    *,
    bootstrap: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Pair two accuracy runs by ``(scenario, sample_id)`` and test Δ.

    ``delta = mean(a) - mean(b)`` (so a positive Δ means run A is better);
    ``delta_pp = 100 * delta``.  Direction and the exact pairing key are part of
    the returned dict.  Per scenario:

    * ``n`` = pairs scored on **both** sides;
    * 0/1 scenarios: 2x2 counts + exact McNemar p-value;
    * F1 scenarios: paired bootstrap percentile CI of ``delta`` (deterministic
      given ``seed``, ``bootstrap`` resamples; ``None`` when ``n < 2``).

    Across scenarios the p-values are Holm-adjusted.  Keys present in only one
    run are returned in ``unpaired``; keys present in both but not scored on both
    sides (error/timeout) are returned in ``unscored``.  A judge-family/metric
    mismatch for the same scenario is listed in ``judge_family_mismatches``.
    """
    index_a, duplicates_a = _measured_index(records_a)
    index_b, duplicates_b = _measured_index(records_b)

    unpaired: List[Dict[str, Any]] = []
    unscored: List[Dict[str, Any]] = []
    per_scenario: Dict[str, List[Tuple[_Measured, _Measured]]] = {}
    family_mismatches: List[Dict[str, Any]] = []

    for key in sorted(set(index_a) | set(index_b)):
        scenario, sample_id = key
        row_a = index_a.get(key)
        row_b = index_b.get(key)
        if row_a is None or row_b is None:
            unpaired.append(
                {
                    "scenario": scenario,
                    "sample_id": sample_id,
                    "in_a": row_a is not None,
                    "in_b": row_b is not None,
                }
            )
            continue
        if row_a.value is None or row_b.value is None:
            unscored.append(
                {
                    "scenario": scenario,
                    "sample_id": sample_id,
                    "status_a": row_a.status,
                    "status_b": row_b.status,
                }
            )
            continue
        if (row_a.family, row_a.metric) != (row_b.family, row_b.metric):
            family_mismatches.append(
                {
                    "scenario": scenario,
                    "sample_id": sample_id,
                    "a": {"family": row_a.family, "metric": row_a.metric},
                    "b": {"family": row_b.family, "metric": row_b.metric},
                }
            )
            continue
        per_scenario.setdefault(scenario, []).append((row_a, row_b))

    scenario_rows: List[Dict[str, Any]] = []
    pvalues: Dict[str, Optional[float]] = {}
    for scenario in sorted(per_scenario):
        pairs = per_scenario[scenario]
        values_a = [float(a.value) for a, _b in pairs]
        values_b = [float(b.value) for _a, b in pairs]
        n = len(pairs)
        mean_a = fmean(values_a)
        mean_b = fmean(values_b)
        delta = mean_a - mean_b
        binary = all(a.binary and b.binary for a, b in pairs)
        record: Dict[str, Any] = {
            "scenario": scenario,
            "n": n,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "delta": delta,
            "delta_pp": 100.0 * delta,
            "relative_loss": (delta / mean_a) if mean_a else None,
            "binary": binary,
            "metric": pairs[0][0].metric,
            "judge_family": pairs[0][0].family,
            "n_judge_error_a": sum(1 for a, _b in pairs if a.judge_error),
            "n_judge_error_b": sum(1 for _a, b in pairs if b.judge_error),
        }
        p_value: Optional[float] = None
        if binary:
            n_11 = sum(1 for a, b in pairs if a.correct and b.correct)
            n_10 = sum(1 for a, b in pairs if a.correct and not b.correct)
            n_01 = sum(1 for a, b in pairs if not a.correct and b.correct)
            n_00 = sum(1 for a, b in pairs if not a.correct and not b.correct)
            p_value = _mcnemar_exact(n_10, n_01)
            record.update(
                {
                    "n_both_correct": n_11,
                    "n_a_correct_b_wrong": n_10,
                    "n_a_wrong_b_correct": n_01,
                    "n_both_wrong": n_00,
                    "mcnemar_p": p_value,
                }
            )
        else:
            boot = _bootstrap_delta_ci(
                [a.value - b.value for a, b in pairs], bootstrap, seed
            )
            record["bootstrap_ci95_low_pp"] = boot[0] * 100.0 if boot else None
            record["bootstrap_ci95_high_pp"] = boot[1] * 100.0 if boot else None
            record["n_bootstrap"] = boot[2] if boot else 0
        pvalues[scenario] = p_value
        scenario_rows.append(record)

    holm = holm_adjust(pvalues)
    for record in scenario_rows:
        p_adj = holm.get(record["scenario"])
        record["mcnemar_p_holm"] = p_adj
        record["significant_holm"] = bool(p_adj is not None and p_adj < alpha)

    deltas = [row["delta"] for row in scenario_rows]
    macro_delta = fmean(deltas) if deltas else None
    return {
        "method": "accuracy",
        "pairing": "(scenario, sample_id)",
        "direction": "delta = mean(a) - mean(b); positive means A is better",
        "alpha": alpha,
        "n_paired": sum(row["n"] for row in scenario_rows),
        "n_scenarios": len(scenario_rows),
        "macro_delta": macro_delta,
        "macro_delta_pp": (100.0 * macro_delta) if macro_delta is not None else None,
        "scenarios": scenario_rows,
        "unpaired": unpaired,
        "unscored": unscored,
        "judge_family_mismatches": family_mismatches,
        "duplicates_a": duplicates_a,
        "duplicates_b": duplicates_b,
    }


__all__ = [
    "aggregate_accuracy",
    "accuracy_sample_results",
    "normalize_judge_score",
    "paired_accuracy",
    "wilson_interval",
    "normal_interval",
    "holm_adjust",
    "judge_meta",
    "language_of",
]
