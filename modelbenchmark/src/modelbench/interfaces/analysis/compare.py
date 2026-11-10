"""Cross-run comparison over schema-3.0 ``aggregates.json``.

Compares the ``overall`` averages per ``(model, context_length)`` from several
run bundles.  Context buckets are kept separate: a single model-wide mean over
1K/8K/32K would compare different workloads and is never produced.

Schema 3.0 deleted ``summary.json``: the single machine-readable form of the
aggregates is ``<run>/aggregates.json`` (``overall[]`` + ``cells[]``), written by
:class:`~modelbench.adapters.storage.writer.RunWriter` and rebuildable from
``records.jsonl``.  The public comparison output (markdown table, same columns,
same ``Δ`` block) is unchanged.
"""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from modelbench.adapters.storage import layout

DEFAULT_METRICS = [
    "ttft_ms",
    "gen_tps",
    "tpot_ms",
    "e2e_ms",
    "peak_memory_gb",
    "input_context_tokens",
]


def _aggregates_path(run: Path | str) -> Path:
    path = Path(run)
    if path.is_dir():
        return path / layout.AGGREGATES_JSON
    if path.is_file() and path.name != layout.AGGREGATES_JSON:
        # Tolerate a sibling file (e.g. run.json) from a tab-completed path.
        sibling = path.parent / layout.AGGREGATES_JSON
        if sibling.is_file():
            return sibling
    return path


def load_aggregates(run: Path | str) -> Dict[str, Any]:
    """Read ``aggregates.json`` for a run directory (or an explicit path).

    Raises ``FileNotFoundError`` when the run has no aggregates yet, so a typo in
    a run id never silently compares an empty table.
    """
    path = _aggregates_path(run)
    if not path.is_file():
        raise FileNotFoundError(f"{layout.AGGREGATES_JSON} not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{layout.AGGREGATES_JSON} is not a JSON object: {path}")
    return payload


#: Backwards-compatible alias: callers used to load ``summary.json``.
def load_summary(run: Path | str) -> Dict[str, Any]:
    """Deprecated alias for :func:`load_aggregates` (schema 3.0)."""
    return load_aggregates(run)


def _context_label(context_length: Any) -> str:
    try:
        value = int(context_length)
    except (TypeError, ValueError):
        return "全部"
    return f"{value // 1024}K" if value > 0 else "全部"


def _overall_lookup(
    aggregates: Dict[str, Any],
) -> Dict[Tuple[str, Optional[int]], Dict[str, Any]]:
    """Key ``overall`` rows by (model, context_length); 2.0 rows have no bucket."""
    lookup: Dict[Tuple[str, Optional[int]], Dict[str, Any]] = {}
    for entry in aggregates.get("overall", []):
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


def _as_aggregates(item: Any) -> Dict[str, Any]:
    """Accept an already-loaded dict or a run path for convenience."""
    if isinstance(item, dict):
        return item
    return load_aggregates(item)


def compare_summaries(
    summaries: Sequence[Dict[str, Any]],
    metrics: Optional[Sequence[str]] = None,
    only_deltas: bool = False,
) -> str:
    """Render a markdown comparison table for 2+ runs.

    ``summaries`` holds aggregates documents (``aggregates.json`` shape) or paths
    to run directories / ``aggregates.json`` files.  The output structure is
    unchanged from schema 2.x.
    """
    metrics = list(metrics or DEFAULT_METRICS)
    documents = [_as_aggregates(item) for item in summaries]
    lookups = [_overall_lookup(document) for document in documents]
    keys = sorted(
        {key for lookup in lookups for key in lookup},
        key=lambda item: (item[0], item[1] or 0),
    )
    run_labels = [
        document.get("run_id") or document.get("batch_id") or f"run{i}"
        for i, document in enumerate(documents)
    ]

    header = ["模型", "档位"]
    for label in run_labels:
        for metric in metrics:
            header.append(f"{label}:{metric}")
    if len(documents) == 2:
        for metric in metrics:
            header.append(f"Δ({run_labels[1]}-{run_labels[0]}):{metric}")

    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for model, context_length in keys:
        entries = [lookup.get((model, context_length)) for lookup in lookups]
        if only_deltas and len(documents) == 2 and all(e is None for e in entries):
            continue
        row = [model, _context_label(context_length)]
        values: List[List[Optional[float]]] = []
        for entry in entries:
            vals = [_metric_value(entry, metric) for metric in metrics]
            values.append(vals)
            row.extend(_fmt(v) for v in vals)
        if len(documents) == 2:
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
    aggregates = [load_aggregates(path) for path in run_paths]
    print(compare_summaries(aggregates, metrics=metrics))


# --------------------------------------------------------------------------- #
# paired comparison (same sample_id across two runs)
# --------------------------------------------------------------------------- #

#: Columns read from ``samples.csv`` for paired comparisons.
PAIRED_METRICS = (
    "ttft_ms",
    "gen_tps",
    "tpot_ms",
    "e2e_ms",
    "peak_memory_gb",
    "input_context_tokens",
)


def _find_samples_csv(run: Path | str) -> Path:
    path = Path(run)
    if path.is_file():
        path = path.parent
    return path / "samples.csv"


def load_samples(run: Path | str) -> List[Dict[str, str]]:
    """Read a run's ``samples.csv`` rows (measured samples, status=ok)."""
    path = _find_samples_csv(run)
    if not path.is_file():
        raise FileNotFoundError(f"samples.csv not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if (row.get("status") or "") == "ok"]


def _pair_key(row: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    return (
        str(row.get("scenario") or ""),
        str(row.get("dataset") or ""),
        str(row.get("context_length") or ""),
        str(row.get("sample_id") or ""),
        str(row.get("repeat") or "0"),
    )


def _as_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bootstrap_ci(values: Sequence[float], *, samples: int = 4000, seed: int = 42) -> Tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(samples):
        means.append(statistics.mean(values[rng.randrange(n)] for _ in range(n)))
    means.sort()
    lo = means[int(0.025 * (samples - 1))]
    hi = means[int(0.975 * (samples - 1))]
    return lo, hi


def paired_comparison(
    run_a: Path | str,
    run_b: Path | str,
    metrics: Optional[Sequence[str]] = None,
    *,
    bootstrap_samples: int = 4000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compare two runs on the **same sample_id** (paired deltas, bootstrap CI).

    Both runs must expose ``samples.csv``.  Pairing uses
    ``(scenario, dataset, context_length, sample_id, repeat)``; only measured
    ``status=ok`` rows with a numeric value on both sides contribute.  The CI is
    a paired bootstrap over the per-sample deltas ``B - A`` (95%).  This is
    still within-run sampling uncertainty: it does not replace a run-to-run
    noise baseline.
    """
    metrics = list(metrics or PAIRED_METRICS)
    rows_a = load_samples(run_a)
    rows_b = load_samples(run_b)
    map_a = {_pair_key(row): row for row in rows_a}
    map_b = {_pair_key(row): row for row in rows_b}
    common = sorted(set(map_a) & set(map_b))
    result: Dict[str, Any] = {
        "run_a": str(run_a),
        "run_b": str(run_b),
        "n_rows_a": len(rows_a),
        "n_rows_b": len(rows_b),
        "n_common": len(common),
        "metrics": {},
        "note": (
            "paired bootstrap 95% CI over per-sample deltas (B-A); "
            "within-run sampling uncertainty, not run-to-run noise"
        ),
    }
    for metric in metrics:
        deltas: List[float] = []
        values_a: List[float] = []
        values_b: List[float] = []
        for key in common:
            a = _as_float(map_a[key].get(metric))
            b = _as_float(map_b[key].get(metric))
            if a is None or b is None:
                continue
            values_a.append(a)
            values_b.append(b)
            deltas.append(b - a)
        if not deltas:
            result["metrics"][metric] = {"n": 0, "note": "no paired values"}
            continue
        mean_a = statistics.mean(values_a)
        mean_b = statistics.mean(values_b)
        delta = statistics.mean(deltas)
        lo, hi = _bootstrap_ci(deltas, samples=bootstrap_samples, seed=seed)
        result["metrics"][metric] = {
            "n": len(deltas),
            "mean_a": round(mean_a, 4),
            "mean_b": round(mean_b, 4),
            "delta": round(delta, 4),
            "ci95_low": round(lo, 4),
            "ci95_high": round(hi, 4),
            "significant": not (lo <= 0.0 <= hi),
        }
    return result


def render_paired(result: Dict[str, Any]) -> str:
    lines = [
        f"# Paired comparison: A={result['run_a']}  B={result['run_b']}",
        "",
        f"- common samples: {result['n_common']} "
        f"(A rows={result['n_rows_a']}, B rows={result['n_rows_b']})",
        f"- {result['note']}",
        "",
        "| metric | n | A mean | B mean | Δ(B-A) | 95% CI | CI 不含 0 |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for metric, item in result["metrics"].items():
        if not item.get("n"):
            lines.append(f"| {metric} | 0 | - | - | - | - | - |")
            continue
        ci = f"[{item['ci95_low']}, {item['ci95_high']}]"
        lines.append(
            f"| {metric} | {item['n']} | {item['mean_a']} | {item['mean_b']} | "
            f"{item['delta']} | {ci} | {'yes' if item['significant'] else 'no'} |"
        )
    return "\n".join(lines) + "\n"
