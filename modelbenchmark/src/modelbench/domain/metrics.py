"""Metric computation and aggregation (performance-only schema 2.1).

Aggregation has three levels:

* ``SampleRecord``      one measured request (authoritative evidence);
* ``CellAggregate``     per model x scenario x dataset x context bucket;
* ``OverallAggregate``  per model x context bucket (the headline averages shown
  to users).  Context buckets are never collapsed: TTFT/prefill differ by an
  order of magnitude between 1K/8K/32K, so a model-wide mean is meaningless.

Accuracy scoring is not part of the performance benchmark, so no accuracy/CI
fields are produced here.  ``wilson_interval`` is kept as a helper for the
retained (currently disconnected) evaluator domain.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .models import (
    CellAggregate,
    Distribution,
    OverallAggregate,
    PerfMetrics,
    RecordStatus,
    SampleRecord,
    SampleResult,
)

#: Decode-rate statistics are only meaningful when enough tokens were generated
#: (oMLX itself uses 16 as the minimum).  Shorter generations report ``None``.
MIN_DECODE_TOKENS = 16


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> Optional[List[float]]:
    """95% Wilson score interval for a binomial proportion.

    Retained for the evaluator domain; not used by the performance benchmark.
    """
    if n <= 0:
        return None
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)]


def distribution(values: Sequence[Optional[float]]) -> Distribution:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    n = len(clean)
    if n == 0:
        return Distribution(n=0)
    clean.sort()
    mean = sum(clean) / n
    median = _percentile(clean, 50)
    var = sum((x - mean) ** 2 for x in clean) / n
    return Distribution(
        n=n,
        mean=round(mean, 3),
        median=round(median, 3),
        std=round(math.sqrt(var), 3),
        min=round(clean[0], 3),
        max=round(clean[-1], 3),
        p95=round(_percentile(clean, 95), 3),
    )


def _percentile(sorted_values: List[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return sorted_values[low]
    weight = rank - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


PERF_FIELDS = (
    "ttft_ms",
    "ttft_visible_ms",
    "tpot_ms",
    "e2e_ms",
    "gen_tps",
    "prefill_tps",
    "total_tps",
    "input_context_tokens",
    "input_context_chars",
    "peak_cpu_pct",
    "mean_cpu_pct",
    "peak_rss_mb",
    "peak_memory_gb",
    "prompt_tokens",
    "completion_tokens",
)


def _valid(records: Iterable[SampleRecord]) -> List[SampleRecord]:
    return [
        r
        for r in records
        if r.phase.value == "measured" and r.status is RecordStatus.OK
    ]


def is_length_capped(record: SampleRecord) -> bool:
    """True when generation stopped at ``perf.max_tokens``.

    A capped generation is a *complete measurement* of the emitted decode
    window: it is never dropped, and ``gen_tps``/``tpot_ms`` use the actual
    ``completion_tokens`` from usage, not the configured cap.
    """
    return (record.response.finish_reason or "").lower() == "length"


def _stability(perf: Dict[str, Distribution]) -> Optional[float]:
    ttft = perf.get("ttft_ms")
    if ttft and ttft.mean and ttft.std is not None and ttft.mean > 0:
        return round(ttft.std / ttft.mean, 3)
    return None


def aggregate_cell(records: List[SampleRecord]) -> CellAggregate:
    if not records:
        raise ValueError("cannot aggregate an empty record list")
    first = records[0]
    cell = CellAggregate(
        model=first.model,
        scenario=first.scenario,
        context_length=first.context_length,
        dataset=first.dataset,
        n_samples=len({r.sample_id for r in records}),
        n_repeats=len({r.repeat for r in records if r.phase.value == "measured"}),
    )
    valid = _valid(records)
    cell.n_valid = len(valid)
    cell.n_warmup = sum(1 for r in records if r.phase.value == "warmup")
    cell.n_contaminated = sum(
        1 for r in records if r.status is RecordStatus.CONTAMINATED
    )
    cell.n_error = sum(
        1
        for r in records
        if r.phase.value == "measured"
        and r.status in (RecordStatus.ERROR, RecordStatus.TIMEOUT)
    )
    cell.n_length_capped = sum(1 for r in valid if is_length_capped(r))

    perf: Dict[str, Distribution] = {}
    for field in PERF_FIELDS:
        perf[field] = distribution([getattr(r.metrics, field) for r in valid])
    cell.perf = perf
    cell.stability_cv = _stability(perf)
    cell.unstable = cell.stability_cv is not None and cell.stability_cv > 0.2
    return cell


def aggregate_records(records: Iterable[SampleRecord]) -> List[CellAggregate]:
    grouped: Dict[tuple, List[SampleRecord]] = {}
    for record in records:
        key = (record.model, record.scenario, record.context_length, record.dataset)
        grouped.setdefault(key, []).append(record)
    cells = [aggregate_cell(rows) for rows in grouped.values()]
    cells.sort(key=lambda c: (c.model, c.scenario, c.dataset, c.context_length or 0))
    return cells


def aggregate_overall(records: Iterable[SampleRecord]) -> List[OverallAggregate]:
    """Per-model x per-context averages over valid measured samples.

    Records are grouped by ``(model, context_length)``; each group is
    aggregated independently.  This is deliberate: 1K/8K/32K cells have
    completely different TTFT/prefill/TG-TPS profiles, and a unified
    "all contexts" mean would be dominated by whichever bucket has more
    samples and cannot be compared across runs.  Legacy records without a
    declared bucket form their own ``context_length=None`` group.
    """
    records = list(records)
    grouped: Dict[tuple, List[SampleRecord]] = {}
    for record in records:
        key = (record.model, record.context_length)
        grouped.setdefault(key, []).append(record)

    overall: List[OverallAggregate] = []
    for (model, context_length), rows in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1] or 0)
    ):
        valid = _valid(rows)
        entry = OverallAggregate(
            model=model,
            context_length=context_length,
            n_datasets=len({(r.scenario, r.dataset) for r in rows if r.phase.value == "measured"}),
            n_samples=len({(r.scenario, r.dataset, r.sample_id) for r in rows}),
            n_valid=len(valid),
            n_warmup=sum(1 for r in rows if r.phase.value == "warmup"),
            n_contaminated=sum(
                1 for r in rows if r.status is RecordStatus.CONTAMINATED
            ),
            n_error=sum(
                1
                for r in rows
                if r.phase.value == "measured"
                and r.status in (RecordStatus.ERROR, RecordStatus.TIMEOUT)
            ),
        )
        perf: Dict[str, Distribution] = {}
        for field in PERF_FIELDS:
            perf[field] = distribution([getattr(r.metrics, field) for r in valid])
        entry.perf = perf
        entry.n_length_capped = sum(1 for r in valid if is_length_capped(r))
        rss_values = [
            r.metrics.peak_rss_mb for r in valid if r.metrics.peak_rss_mb is not None
        ]
        entry.peak_rss_mb_max = round(max(rss_values), 1) if rss_values else None
        mem_values = [
            r.metrics.peak_memory_gb
            for r in valid
            if r.metrics.peak_memory_gb is not None
        ]
        entry.peak_memory_gb_max = round(max(mem_values), 2) if mem_values else None
        entry.stability_cv = _stability(perf)
        entry.unstable = entry.stability_cv is not None and entry.stability_cv > 0.2
        overall.append(entry)
    return overall


def sample_results(records: Iterable[SampleRecord]) -> List[SampleResult]:
    """Project measured records into the per-sample result table."""
    rows: List[SampleResult] = []
    for record in records:
        if record.phase.value != "measured":
            continue
        metrics = record.metrics
        rows.append(
            SampleResult(
                model=record.model,
                scenario=record.scenario,
                dataset=record.dataset,
                context_length=record.context_length,
                sample_id=record.sample_id,
                repeat=record.repeat,
                status=record.status.value,
                input_context_tokens=metrics.input_context_tokens
                if metrics.input_context_tokens is not None
                else metrics.prompt_tokens,
                input_context_chars=metrics.input_context_chars,
                ttft_ms=metrics.ttft_ms,
                ttft_visible_ms=metrics.ttft_visible_ms,
                tpot_ms=metrics.tpot_ms,
                gen_tps=metrics.gen_tps,
                e2e_ms=metrics.e2e_ms,
                prefill_tps=metrics.prefill_tps,
                peak_cpu_pct=metrics.peak_cpu_pct,
                peak_rss_mb=metrics.peak_rss_mb,
                peak_memory_gb=metrics.peak_memory_gb,
                completion_tokens=metrics.completion_tokens,
                finish_reason=record.response.finish_reason,
                length_capped=is_length_capped(record),
                truncated=bool(record.request.truncated),
            )
        )
    rows.sort(key=lambda r: (r.model, r.scenario, r.sample_id, r.repeat))
    return rows


def compute_metrics(
    *,
    ttft_s: Optional[float],
    first_visible_s: Optional[float],
    last_token_s: Optional[float],
    e2e_s: float,
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    usage: Any = None,
    input_context_chars: Optional[int] = None,
) -> PerfMetrics:
    """Turn client-side timings + usage into the normalized metric set."""
    ttft_ms = round(ttft_s * 1000.0, 1) if ttft_s is not None else None
    visible_ms = round(first_visible_s * 1000.0, 1) if first_visible_s is not None else None
    e2e_ms = round(e2e_s * 1000.0, 1)

    tpot_ms = None
    gen_tps = None
    if (
        ttft_s is not None
        and last_token_s is not None
        and completion_tokens is not None
        and completion_tokens >= MIN_DECODE_TOKENS
    ):
        decode_s = last_token_s - ttft_s
        if decode_s > 0:
            tpot_ms = round(decode_s / (completion_tokens - 1) * 1000.0, 2)
            gen_tps = round((completion_tokens - 1) / decode_s, 2)

    prefill_tps = None
    if ttft_s and ttft_s > 0 and prompt_tokens:
        prefill_tps = round(prompt_tokens / ttft_s, 1)

    total_tps = None
    if e2e_s > 0:
        total_tps = round(((prompt_tokens or 0) + (completion_tokens or 0)) / e2e_s, 2)

    return PerfMetrics(
        ttft_ms=ttft_ms,
        ttft_visible_ms=visible_ms,
        tpot_ms=tpot_ms,
        e2e_ms=e2e_ms,
        gen_tps=gen_tps,
        prefill_tps=prefill_tps,
        total_tps=total_tps,
        input_context_tokens=prompt_tokens,
        input_context_chars=input_context_chars,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
