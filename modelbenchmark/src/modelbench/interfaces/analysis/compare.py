"""Cross-run comparison over schema-3.0 run bundles.

Two independent comparison surfaces live here, selected by
``run.json.method``:

* **performance** (default, unchanged): ``compare_summaries`` compares the
  ``overall`` averages per ``(model, context_length)`` from several run bundles;
  ``paired_comparison`` pairs ``samples.csv`` rows by
  ``(scenario, dataset, context_length, sample_id, repeat)``.  Context buckets
  are kept separate: a single model-wide mean over 1K/8K/32K would compare
  different workloads and is never produced.
* **accuracy**: ``render_accuracy_table`` summarises ``aggregates.json``
  (scenario cells + macro/micro overall rows) and
  ``accuracy_paired_comparison`` pairs two runs by ``(scenario, sample_id)``
  through :func:`modelbench.domain.accuracy_metrics.paired_accuracy`, adding Δpp,
  exact McNemar (0/1), paired bootstrap CI (F1) and Holm correction.

Schema 3.0 deleted ``summary.json``: the single machine-readable form of the
aggregates is ``<run>/aggregates.json`` (``overall[]`` + ``cells[]``), written by
:class:`~modelbench.adapters.storage.writer.RunWriter` and rebuildable from
``records.jsonl``.  The public performance comparison output (markdown table,
same columns, same ``Δ`` block) is unchanged.

Accuracy runs are never mixed with performance runs: ``load_run_method`` reads
``run.json.method`` and callers must reject a mixed comparison instead of
guessing.
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


# --------------------------------------------------------------------------- #
# accuracy comparison (method=accuracy)
# --------------------------------------------------------------------------- #

PERFORMANCE_METHOD = "performance"
ACCURACY_METHOD = "accuracy"
#: Protocol the accuracy plan fixes for every run (``22-accuracy-test-plan.md`` §2).
ACCURACY_PROTOCOL = "cold, repeats=1, temperature=0"


def _read_json_file(path: Path) -> Optional[Dict[str, Any]]:
    """Read a JSON object; ``None`` for a missing/unreadable/non-object file."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _method_from_document(document: Optional[Dict[str, Any]]) -> Optional[str]:
    """Normalize a document's ``method``; unknown/absent values are ``None``.

    Only ``accuracy`` is recognized as the second method; everything else
    (including a legacy document with no ``method``) belongs to the performance
    path, exactly as before the two-method split.
    """
    if not isinstance(document, dict):
        return None
    value = document.get("method")
    if not value:
        return None
    return ACCURACY_METHOD if str(value).strip().lower() == ACCURACY_METHOD else PERFORMANCE_METHOD


def _run_dir(run: Path | str | Dict[str, Any]) -> Path:
    """The run directory for a directory, a run file, or an explicit path."""
    path = Path(run)
    return path.parent if path.is_file() else path


def _run_json_for(run: Path | str | Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if isinstance(run, dict):
        return run
    path = Path(run)
    if path.is_file() and path.name == layout.RUN_JSON:
        return _read_json_file(path)
    return _read_json_file(_run_dir(path) / layout.RUN_JSON)


def load_run_method(run: Path | str | Dict[str, Any]) -> str:
    """Return a run's method: ``"performance"`` or ``"accuracy"``.

    ``run`` may be a run directory, a ``run.json``/``aggregates.json`` path, or an
    already-loaded ``aggregates.json``/``run.json`` document.  ``run.json`` wins;
    aggregates are the fallback.  Unknown or legacy runs (no ``method`` field, or
    an unrecognized value) default to ``"performance"`` so existing callers keep
    their schema-2.x behavior without special-casing old bundles.
    """
    if isinstance(run, dict):
        return _method_from_document(run) or PERFORMANCE_METHOD
    path = Path(run)
    if path.is_file() and path.name == layout.AGGREGATES_JSON:
        return _method_from_document(_read_json_file(path)) or PERFORMANCE_METHOD
    recorded = _method_from_document(_run_json_for(path))
    if recorded is not None:
        return recorded
    aggregates = _read_json_file(_run_dir(path) / layout.AGGREGATES_JSON)
    return _method_from_document(aggregates) or PERFORMANCE_METHOD


def _document_testset_sha256(document: Optional[Dict[str, Any]]) -> Optional[str]:
    """Extract a testset SHA-256 from an aggregates/run/batch document."""
    if not isinstance(document, dict):
        return None
    candidates: List[Any] = [document.get("testset_sha256")]
    testset = document.get("testset")
    if isinstance(testset, dict):
        candidates.append(testset.get("sha256"))
    inputs = document.get("inputs")
    if isinstance(inputs, dict):
        testset = inputs.get("testset")
        if isinstance(testset, dict):
            candidates.append(testset.get("sha256"))
    for value in candidates:
        if value:
            return str(value)
    return None


def _run_testset_sha256(run: Path | str | Dict[str, Any]) -> Optional[str]:
    """Best-effort testset SHA-256: aggregates, then run.json, then batch.json.

    ``batch.json`` lives at ``<output>/batches/<batch_id>.json`` one level above
    ``runs/``; a run directory mounted elsewhere simply yields ``None`` instead
    of a fabricated value.
    """
    if isinstance(run, dict):
        return _document_testset_sha256(run)
    path = Path(run)
    if path.is_file() and path.name == layout.AGGREGATES_JSON:
        aggregates = _read_json_file(path)
    else:
        aggregates = _read_json_file(_run_dir(path) / layout.AGGREGATES_JSON)
    sha = _document_testset_sha256(aggregates)
    if sha:
        return sha
    run_json = _read_json_file(_run_dir(path) / layout.RUN_JSON)
    sha = _document_testset_sha256(run_json)
    if sha:
        return sha
    batch_id = (run_json or {}).get("batch_id")
    if batch_id:
        output_root = _run_dir(path).parent.parent
        sha = _document_testset_sha256(
            _read_json_file(layout.batch_path(output_root, str(batch_id)))
        )
        if sha:
            return sha
    return None


def _run_protocol(run: Path | str | Dict[str, Any]) -> str:
    """Observed accuracy protocol (``cold, repeats=1, temperature=0``)."""
    run_json = _run_json_for(run)
    config = (run_json or {}).get("config") or {}
    run_config = config.get("run") if isinstance(config, dict) else {}
    run_config = run_config if isinstance(run_config, dict) else {}
    cache = str(run_config.get("cache_mode") or "cold")
    repeats = run_config.get("repeats", 1)
    return f"{cache}, repeats={repeats}, temperature=0"


def _fmt_score(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(number):
        return "-"
    return f"{number:.4f}"


def _fmt_ci95(entry: Optional[Dict[str, Any]]) -> str:
    """Score CI bracket (accuracy CI when a slice has no scores)."""
    if not isinstance(entry, dict):
        return "-"
    low = entry.get("score_ci95_low")
    high = entry.get("score_ci95_high")
    if low is None or high is None:
        low = entry.get("accuracy_ci95_low")
        high = entry.get("accuracy_ci95_high")
    if low is None or high is None:
        return "-"
    return f"[{_fmt_score(low)}, {_fmt_score(high)}]"


def _run_label(document: Dict[str, Any], index: int) -> str:
    return str(
        document.get("run_id")
        or document.get("batch_id")
        or f"run{index}"
    )


def _document_models(document: Dict[str, Any]) -> List[str]:
    models = [
        str(row.get("model"))
        for row in document.get("overall") or []
        if isinstance(row, dict) and row.get("model")
    ]
    if not models:
        models = sorted(
            {
                str(cell.get("model"))
                for cell in document.get("cells") or []
                if isinstance(cell, dict) and cell.get("model")
            }
        )
    return models


def render_accuracy_table(
    aggregates_documents: Sequence[Any],
    *,
    run_paths: Optional[Sequence[Any]] = None,
) -> str:
    """Render a markdown accuracy comparison for 1+ accuracy runs.

    ``aggregates_documents`` holds ``aggregates.json`` documents or paths to run
    directories / ``aggregates.json`` files (same convenience as
    :func:`compare_summaries`).  Each scenario cell becomes one row
    (``model | scenario | n | score | accuracy | 95%CI``); when a scenario has
    more than one dataset the dataset is appended to the scenario label so two
    cells never collapse into an ambiguous row.  Per model, one ``整体(macro)``
    row (equal-weight scenario score, the headline metric) and one
    ``整体(micro)`` row (all-row mean, auxiliary only) are emitted.

    ``run_paths`` is an optional parallel list of the original run paths; it is
    used to resolve the testset SHA-256 (via ``batch.json``) and the observed
    protocol for the document header when only loaded dicts are supplied.
    """
    documents = [_as_aggregates(item) for item in aggregates_documents]
    targets = list(run_paths or [])
    labels = [_run_label(document, index) for index, document in enumerate(documents)]

    shas: List[Optional[str]] = []
    protocols: List[Optional[str]] = []
    for index, document in enumerate(documents):
        target = targets[index] if index < len(targets) else None
        sha = _document_testset_sha256(document)
        if not sha and target is not None:
            sha = _run_testset_sha256(target)
        shas.append(sha)
        protocols.append(_run_protocol(target) if target is not None else None)

    known_shas = sorted({sha for sha in shas if sha})
    if not known_shas:
        sha_line = "unknown"
    elif len(known_shas) == 1:
        sha_line = known_shas[0]
    else:
        sha_line = ", ".join(
            f"{labels[index]}={shas[index] or 'unknown'}"
            for index in range(len(documents))
        )

    observed_protocols = sorted({item for item in protocols if item})
    if not observed_protocols:
        protocol_line = ACCURACY_PROTOCOL
    elif len(observed_protocols) == 1:
        protocol_line = observed_protocols[0]
    else:
        protocol_line = "; ".join(
            f"{labels[index]}={protocols[index] or ACCURACY_PROTOCOL}"
            for index in range(len(documents))
        )

    # Disambiguate a model that appears in more than one run.
    model_counts: Dict[str, int] = {}
    for document in documents:
        for model in _document_models(document):
            model_counts[model] = model_counts.get(model, 0) + 1

    lines = [
        "# 精度对比（method=accuracy）",
        "",
        f"- method: {ACCURACY_METHOD}",
        f"- 协议: {protocol_line}",
        f"- testset SHA-256: {sha_line}",
        "",
        "| 模型 | 场景 | n | score | accuracy | 95%CI |",
        "|---|---|---|---:|---:|---|",
    ]
    for index, document in enumerate(documents):
        default_model = (_document_models(document) or [labels[index]])[0]
        cells = [
            cell for cell in document.get("cells") or [] if isinstance(cell, dict)
        ]
        scenario_counts: Dict[str, int] = {}
        for cell in cells:
            scenario = str(cell.get("scenario") or "")
            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
        cells.sort(
            key=lambda cell: (
                str(cell.get("scenario") or ""),
                str(cell.get("dataset") or ""),
            )
        )
        for cell in cells:
            model = str(cell.get("model") or default_model)
            model_label = (
                model
                if model_counts.get(model, 0) <= 1
                else f"{model} [{labels[index]}]"
            )
            scenario = str(cell.get("scenario") or "")
            if scenario_counts.get(scenario, 0) > 1:
                scenario = f"{scenario}/{cell.get('dataset') or '?'}"
            n = cell.get("n_scored")
            if n is None:
                n = cell.get("n_samples", 0)
            lines.append(
                "| "
                + " | ".join(
                    [
                        model_label,
                        scenario,
                        str(n),
                        _fmt_score(cell.get("mean_score")),
                        _fmt_score(cell.get("accuracy")),
                        _fmt_ci95(cell),
                    ]
                )
                + " |"
            )
        for row in document.get("overall") or []:
            if not isinstance(row, dict):
                continue
            model = str(row.get("model") or default_model)
            model_label = (
                model
                if model_counts.get(model, 0) <= 1
                else f"{model} [{labels[index]}]"
            )
            n = row.get("n_scored")
            if n is None:
                n = row.get("n_samples", 0)
            # macro_score has no per-scenario CI of its own: leave the macro CI
            # blank rather than mislabel the all-row (micro) interval as macro.
            lines.append(
                "| "
                + " | ".join(
                    [
                        model_label,
                        "整体(macro)",
                        str(n),
                        _fmt_score(row.get("macro_score")),
                        _fmt_score(row.get("macro_accuracy")),
                        "-",
                    ]
                )
                + " |"
            )
            lines.append(
                "| "
                + " | ".join(
                    [
                        model_label,
                        "整体(micro)",
                        str(n),
                        _fmt_score(row.get("micro_score")),
                        _fmt_score(row.get("accuracy")),
                        _fmt_ci95(row),
                    ]
                )
                + " |"
            )
    lines += [
        "",
        "> score = 场景均分（0/1 与 F1 的统一分母口径）；accuracy = 0/1 判题族准确率；"
        "macro = 场景等权平均（结论口径），micro = 全行平均（辅助）；"
        "cell CI：0/1 场景为 Wilson，F1 场景为正态近似（n<2 时为 \"-\"）；"
        "整体(micro) 行给全行正态近似 CI，macro 行无独立 CI。",
    ]
    return "\n".join(lines) + "\n"


def _load_run_records(run: Path | str | Dict[str, Any]) -> List[Any]:
    """``records.jsonl`` of one run, parsed into :class:`SampleRecord` objects."""
    if isinstance(run, dict):
        raise ValueError(
            "accuracy_paired_comparison needs run directories/paths (records.jsonl), "
            "not a loaded aggregates document"
        )
    from modelbench.adapters.storage import catalog  # local import: perf path stays light

    run_dir = _run_dir(run)
    records_path = run_dir / layout.RECORDS
    if not records_path.is_file():
        raise FileNotFoundError(
            f"{layout.RECORDS} not found: {records_path} (not a complete run bundle)"
        )
    return catalog.load_records(run_dir)


def accuracy_paired_comparison(
    run_a: Path | str,
    run_b: Path | str,
    *,
    bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Pair two accuracy runs by ``(scenario, sample_id)`` and test Δ.

    Both runs must record ``run.json.method == "accuracy"``; comparing a
    performance run with an accuracy run raises :class:`ValueError` (the caller
    must never silently reinterpret a ``score=null`` performance run).  The
    delegation goes to :func:`modelbench.domain.accuracy_metrics.paired_accuracy`,
    which owns the counting/statistics contract.

    Returns that result plus run identity, ``testset_sha256`` (from
    ``batch.json`` / aggregates; ``None`` when unavailable), a ``note`` that pins
    the Δ direction (A − B, positive means A is better), ``ok``/``reason`` and an
    ``explicit`` block that keeps the unpaired/unscored/judge-family-mismatch/
    duplicate lists.  ``ok`` is ``False`` (with ``reason``) for a judge
    family/metric mismatch or a differing testset SHA-256 — those make Δ
    incomparable.  A sample that exists in only one run means the two sample sets
    differ (different testset/skip set), so ``ok`` is also ``False`` there — but
    the lists stay in ``explicit`` and the table is still rendered for diagnosis
    (AC-A10: sample mismatch must be reported, never silently dropped).
    """
    method_a = load_run_method(run_a)
    method_b = load_run_method(run_b)
    if method_a != ACCURACY_METHOD or method_b != ACCURACY_METHOD:
        raise ValueError(
            "cannot compare performance and accuracy runs: "
            f"run_a method={method_a!r}, run_b method={method_b!r}; "
            "accuracy paired comparison needs two method='accuracy' runs"
        )

    from modelbench.domain.accuracy_metrics import paired_accuracy  # local import

    result = paired_accuracy(
        _load_run_records(run_a),
        _load_run_records(run_b),
        bootstrap=bootstrap,
        seed=seed,
    )

    sha_a = _run_testset_sha256(run_a)
    sha_b = _run_testset_sha256(run_b)
    if sha_a and sha_b:
        common_sha = sha_a if sha_a == sha_b else None
    else:
        common_sha = sha_a or sha_b

    unpaired = list(result.get("unpaired") or [])
    unscored = list(result.get("unscored") or [])
    mismatches = list(result.get("judge_family_mismatches") or [])
    duplicates_a = list(result.get("duplicates_a") or [])
    duplicates_b = list(result.get("duplicates_b") or [])

    failures: List[str] = []
    if unpaired:
        failures.append(
            f"{len(unpaired)} sample(s) are present in only one run; the sample "
            "sets differ (see explicit.unpaired)"
        )
    if mismatches:
        failures.append(
            f"judge family/metric mismatch on {len(mismatches)} sample(s); "
            "Δ is not comparable (see explicit.judge_family_mismatches)"
        )
    if sha_a and sha_b and sha_a != sha_b:
        failures.append(
            "testset SHA-256 differs between runs "
            f"(run_a={sha_a}, run_b={sha_b}); scores are not comparable"
        )
    warnings: List[str] = []
    if unscored:
        warnings.append(
            f"{len(unscored)} paired sample(s) unscored on at least one side "
            "(error/timeout; see explicit.unscored)"
        )
    if duplicates_a or duplicates_b:
        warnings.append(
            "duplicate (scenario, sample_id) rows: "
            f"run_a={len(duplicates_a)}, run_b={len(duplicates_b)} "
            "(only the first row contributes to the pairing)"
        )

    result.update(
        {
            "run_a": str(run_a),
            "run_b": str(run_b),
            "testset_sha256": common_sha,
            "testset_sha256_a": sha_a,
            "testset_sha256_b": sha_b,
            "note": (
                "Δ = mean(A) − mean(B); a positive Δ means run A is better; "
                "Δpp = 100 × Δ. 0/1 scenarios use the exact two-sided McNemar "
                "test, F1 scenarios a paired bootstrap 95% CI, and p-values are "
                "Holm-adjusted across scenarios. unpaired / unscored / "
                "judge_family_mismatches are always listed explicitly."
            ),
            "ok": not failures,
            "reason": "; ".join(failures) if failures else None,
            "warnings": warnings,
            "explicit": {
                "unpaired": unpaired,
                "unscored": unscored,
                "judge_family_mismatches": mismatches,
                "duplicates_a": duplicates_a,
                "duplicates_b": duplicates_b,
            },
        }
    )
    return result


def _fmt_pp(value: Any) -> str:
    try:
        return f"{float(value):+.2f}"
    except (TypeError, ValueError):
        return "-"


def _fmt_pp_unsigned(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "-"


def _fmt_pvalue(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(number):
        return "-"
    return f"{number:.4g}"


def _accuracy_verdict(
    row: Dict[str, Any],
    *,
    alpha: float,
    noise_baseline_pp: float,
) -> str:
    """Turn a scenario row into an explicit verdict (never over-claims).

    0/1 scenarios use the Holm-adjusted McNemar p-value; F1 scenarios have no
    p-value, so a paired bootstrap CI that excludes 0 is the significance test.
    Even when significant, a regression/improvement label requires
    ``|Δpp| > noise_baseline_pp``; with the default baseline of 0 only the
    direction is reported ("显著（A 更好）"), never "退化".
    """
    p_holm = row.get("mcnemar_p_holm")
    if p_holm is not None:
        significant = float(p_holm) < alpha
    else:
        low = row.get("bootstrap_ci95_low_pp")
        high = row.get("bootstrap_ci95_high_pp")
        significant = (
            low is not None
            and high is not None
            and not (float(low) <= 0.0 <= float(high))
        )
    if not significant:
        return "不显著"
    delta_pp = float(row.get("delta_pp") or 0.0)
    baseline = float(noise_baseline_pp or 0.0)
    if baseline <= 0:
        if delta_pp > 0:
            return "显著（A 更好）"
        if delta_pp < 0:
            return "显著（B 更好）"
        return "显著（Δ=0）"
    if abs(delta_pp) <= baseline:
        return "显著但未超过噪声基线"
    return "显著提升" if delta_pp > 0 else "显著退化"


def render_accuracy_paired(
    result: Dict[str, Any],
    *,
    noise_baseline_pp: float = 0.0,
) -> str:
    """Render an ``accuracy_paired_comparison`` result as a markdown report.

    Table columns: scenario / n / A mean / B mean / Δpp / 95%CI(pp) /
    p(McNemar) / p(Holm) / verdict.  A verdict is significant only when the
    (Holm-adjusted) p-value — or, for F1 scenarios, the paired bootstrap CI —
    says so **and** ``|Δpp|`` exceeds ``noise_baseline_pp``; the default baseline
    of 0 reports significance only and never claims a regression.  The
    unpaired/unscored/mismatch counts (and the run's incomparability reason, when
    any) are printed below the table so nothing is silently dropped.
    """
    alpha = float(result.get("alpha") or 0.05)
    baseline = float(noise_baseline_pp or 0.0)
    explicit = result.get("explicit") or {}
    unpaired = explicit.get("unpaired") or result.get("unpaired") or []
    unscored = explicit.get("unscored") or result.get("unscored") or []
    mismatches = (
        explicit.get("judge_family_mismatches")
        or result.get("judge_family_mismatches")
        or []
    )
    duplicates_a = explicit.get("duplicates_a") or result.get("duplicates_a") or []
    duplicates_b = explicit.get("duplicates_b") or result.get("duplicates_b") or []

    lines = [
        f"# 精度配对比较: A={result.get('run_a')}  B={result.get('run_b')}",
        "",
        f"- 配对键: {result.get('pairing') or '(scenario, sample_id)'}",
        "- 方向: Δ = mean(A) − mean(B)；正值 = A 更好；Δpp = 100 × Δ",
        f"- method: {result.get('method') or ACCURACY_METHOD}",
        f"- testset SHA-256: {result.get('testset_sha256') or 'unknown'}",
        f"- α: {alpha:g}；噪声基线: {baseline:g} pp",
        "- 判定: 0/1 场景 p(Holm) < α 且 |Δpp| > 噪声基线；F1 场景配对 bootstrap "
        "95%CI 不含 0 且 |Δpp| > 噪声基线；基线为 0 时只报显著方向，不断言退化",
    ]
    if not result.get("ok", True):
        lines.append(f"- 警告: 结果不可直接比较 — {result.get('reason')}")

    lines += [
        "",
        "| 场景 | n | A 均分 | B 均分 | Δpp | 95%CI(pp) | p(McNemar) | p(Holm) | 判定 |",
        "|---|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    scenarios = [row for row in result.get("scenarios") or [] if isinstance(row, dict)]
    for row in scenarios:
        low = row.get("bootstrap_ci95_low_pp")
        high = row.get("bootstrap_ci95_high_pp")
        ci = (
            f"[{_fmt_pp_unsigned(low)}, {_fmt_pp_unsigned(high)}]"
            if low is not None and high is not None
            else "-"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("scenario") or ""),
                    str(int(row.get("n") or 0)),
                    _fmt_score(row.get("mean_a")),
                    _fmt_score(row.get("mean_b")),
                    _fmt_pp(row.get("delta_pp")),
                    ci,
                    _fmt_pvalue(row.get("mcnemar_p")),
                    _fmt_pvalue(row.get("mcnemar_p_holm")),
                    _accuracy_verdict(row, alpha=alpha, noise_baseline_pp=baseline),
                ]
            )
            + " |"
        )

    lines += [
        "",
        f"- unpaired: {len(unpaired)}（仅一侧存在的样本，未进入配对；明细见 JSON `explicit.unpaired`）",
        f"- unscored: {len(unscored)}（两侧都有但 error/timeout 未计分，未进入配对；明细见 `explicit.unscored`）",
        f"- judge_family_mismatches: {len(mismatches)}（同一样本判题族/metric 不一致，Δ 不可比；明细见 `explicit.judge_family_mismatches`）",
        f"- duplicates: run_a={len(duplicates_a)}, run_b={len(duplicates_b)}（同键重复行只取首行参与配对）",
    ]
    if result.get("n_paired") is not None:
        lines.append(
            f"- 配对样本数: {result.get('n_paired')}；场景数: {result.get('n_scenarios')}"
        )
    macro = result.get("macro_delta_pp")
    if macro is not None:
        lines.append(f"- 宏平均 Δpp: {_fmt_pp(macro)}（场景等权）")
    lines += [
        "",
        "> 注：0/1 场景的显著性由精确 McNemar 双侧检验与 Holm 校正给出（表中 CI 为 \"-\"）；"
        "F1 场景无 p 值，95%CI 为配对 bootstrap 百分位区间，CI 不含 0 即视为显著。"
        "噪声基线来自同一 testset、同一协议下的独立重复 run（bf16 基线），默认 0 表示未知。",
    ]
    return "\n".join(lines) + "\n"
