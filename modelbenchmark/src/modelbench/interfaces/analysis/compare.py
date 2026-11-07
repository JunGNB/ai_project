"""Cross-run comparison (performance-only schema 2.x).

Compares the ``overall`` averages per ``(model, context_length)`` from several
run bundles.  Context buckets are kept separate: a single model-wide mean over
1K/8K/32K would compare different workloads and is never produced.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_METRICS = [
    "ttft_ms",
    "gen_tps",
    "tpot_ms",
    "e2e_ms",
    "peak_memory_gb",
    "input_context_tokens",
]


def load_summary(run: Path | str) -> Dict[str, Any]:
    path = Path(run)
    if path.is_dir():
        path = path / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"summary.json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _context_label(context_length: Any) -> str:
    try:
        value = int(context_length)
    except (TypeError, ValueError):
        return "全部"
    return f"{value // 1024}K" if value > 0 else "全部"


def _overall_lookup(
    summary: Dict[str, Any],
) -> Dict[Tuple[str, Optional[int]], Dict[str, Any]]:
    """Key ``overall`` rows by (model, context_length); 2.0 rows have no bucket."""
    lookup: Dict[Tuple[str, Optional[int]], Dict[str, Any]] = {}
    for entry in summary.get("overall", []):
        model = entry.get("model")
        if not model:
            continue
        context = entry.get("context_length")
        lookup[(str(model), int(context) if context is not None else None)] = entry
    return lookup


def _metric_value(entry: Optional[Dict[str, Any]], metric: str) -> Optional[float]:
    if not entry:
        return None
    if metric == "peak_memory_gb_global_max":
        return entry.get("peak_memory_gb_max")
    perf = entry.get("perf", {}).get(metric)
    return perf.get("mean") if isinstance(perf, dict) else None


def compare_summaries(
    summaries: Sequence[Dict[str, Any]],
    metrics: Optional[Sequence[str]] = None,
    only_deltas: bool = False,
) -> str:
    """Render a markdown comparison table for 2+ runs."""
    metrics = list(metrics or DEFAULT_METRICS)
    lookups = [_overall_lookup(s) for s in summaries]
    keys = sorted(
        {key for lookup in lookups for key in lookup},
        key=lambda item: (item[0], item[1] or 0),
    )
    run_labels = [s.get("run_id", f"run{i}") for i, s in enumerate(summaries)]

    header = ["模型", "档位"]
    for label in run_labels:
        for metric in metrics:
            header.append(f"{label}:{metric}")
    if len(summaries) == 2:
        for metric in metrics:
            header.append(f"Δ({run_labels[1]}-{run_labels[0]}):{metric}")

    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for model, context_length in keys:
        entries = [lookup.get((model, context_length)) for lookup in lookups]
        if only_deltas and len(summaries) == 2 and all(e is None for e in entries):
            continue
        row = [model, _context_label(context_length)]
        values: List[List[Optional[float]]] = []
        for entry in entries:
            vals = [_metric_value(entry, metric) for metric in metrics]
            values.append(vals)
            row.extend(_fmt(v) for v in vals)
        if len(summaries) == 2:
            for idx in range(len(metrics)):
                a, b = values[0][idx], values[1][idx]
                row.append(_fmt(b - a) if a is not None and b is not None else "-")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "-"
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.3f}"


def print_comparison(run_paths: Sequence[Path | str], metrics: Optional[Sequence[str]] = None) -> None:
    summaries = [load_summary(p) for p in run_paths]
    print(compare_summaries(summaries, metrics=metrics))
