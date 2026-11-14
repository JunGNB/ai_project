"""CLI: compare/analyze run bundles (schema 3.0, performance + accuracy).

Targets may be:

* a run **directory** (``output/runs/<batch>__<method>__<slug>/``) — read from
  its ``aggregates.json`` (or, for paired accuracy, its ``records.jsonl``);
* an explicit ``aggregates.json`` path;
* a single-segment **run_id** (``<batch>__<method>__<model_slug>``), resolved
  against the output root through ``catalog.resolve_run``.

The method is selected by ``run.json.method``: ``--method auto`` (default) uses
the runs' recorded method, while an explicit ``--method`` must agree with it.
Performance runs keep the historical aggregates table / paired ``samples.csv``
output; accuracy runs get the scenario table and, with ``--paired``, the
``(scenario, sample_id)`` Δ / McNemar / Holm comparison.  Mixing the two methods
is a hard error — a ``score=null`` performance run is never reinterpreted as an
accuracy result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .compare import (
    ACCURACY_METHOD,
    DEFAULT_METRICS,
    PERFORMANCE_METHOD,
    accuracy_paired_comparison,
    compare_summaries,
    load_aggregates,
    load_run_method,
    paired_comparison,
    render_accuracy_paired,
    render_accuracy_table,
    render_paired,
)

METHOD_CHOICES = ("auto", PERFORMANCE_METHOD, ACCURACY_METHOD)


def _default_output_root() -> Path:
    try:
        from modelbench.adapters.config.settings import load_config

        return Path(load_config().run.output_dir)
    except Exception:  # noqa: BLE001 - analysis must work without project config
        return Path("output")


def _resolve_target(target: str, output_root: Path) -> Path:
    """Return a run directory / aggregates path for a path or a run_id."""
    candidate = Path(target).expanduser()
    if candidate.exists():
        return candidate
    from modelbench.adapters.storage.catalog import resolve_run

    return resolve_run(output_root, target)


def _select_method(
    targets: List[Path], requested: str
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve the run method, refusing mixed or explicitly contradicted runs.

    Returns ``(method, error)``: exactly one is ``None``.  Legacy/unknown runs
    are ``performance`` (see :func:`~.compare.load_run_method`).
    """
    methods = [load_run_method(target) for target in targets]
    distinct = sorted(set(methods))
    if len(distinct) > 1:
        detail = ", ".join(
            f"{target}={method}" for target, method in zip(targets, methods)
        )
        return None, (
            "cannot compare performance and accuracy runs "
            f"({detail}); compare runs of the same method"
        )
    recorded = distinct[0] if distinct else PERFORMANCE_METHOD
    if requested != "auto" and requested != recorded:
        return None, (
            "cannot compare performance and accuracy runs: "
            f"run(s) record method={recorded!r} but --method {requested} was "
            "requested"
        )
    return recorded, None


def _load_documents(targets: List[Path]) -> List[Dict[str, Any]]:
    return [load_aggregates(target) for target in targets]


def _run_accuracy(args: argparse.Namespace, targets: List[Path]) -> int:
    if args.paired:
        if len(targets) != 2:
            print("ERROR: --paired requires exactly two runs", file=sys.stderr)
            return 2
        try:
            result = accuracy_paired_comparison(targets[0], targets[1])
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        except (KeyError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(
                render_accuracy_paired(
                    result,
                    noise_baseline_pp=float(args.noise_baseline_pp or 0.0),
                )
            )
        if not result.get("ok", True):
            # The numbers are still printed for diagnosis, but an incomparable
            # comparison must never look like a successful analysis.
            print(f"ERROR: {result.get('reason')}", file=sys.stderr)
            return 2
        return 0

    try:
        documents = _load_documents(targets)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except (KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(documents, ensure_ascii=False, indent=2))
        return 0
    print(render_accuracy_table(documents, run_paths=targets))
    return 0


def _run_performance(args: argparse.Namespace, targets: List[Path]) -> int:
    """Historical performance path: same outputs and exit codes as before."""
    try:
        documents = _load_documents(targets)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except (KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.paired:
        if len(targets) != 2:
            print("ERROR: --paired requires exactly two runs", file=sys.stderr)
            return 2
        try:
            result = paired_comparison(
                targets[0],
                targets[1],
                metrics=args.metrics,
            )
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(render_paired(result))
        return 0

    if args.json:
        print(json.dumps(documents, ensure_ascii=False, indent=2))
        return 0
    print(compare_summaries(documents, metrics=args.metrics))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="modelbench-analysis",
        description=(
            "Compare two or more runs (aggregates.json; paired accuracy uses "
            "records.jsonl). Targets are run directories, aggregates.json paths, "
            "or run ids."
        ),
    )
    parser.add_argument(
        "runs", nargs="+", help="run directories, aggregates.json paths, or run ids"
    )
    parser.add_argument(
        "--method",
        choices=METHOD_CHOICES,
        default="auto",
        help=(
            "test method: auto (default; per run.json.method), performance, or "
            "accuracy. Mixing methods is an error."
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="performance metrics to compare (ignored for accuracy runs)",
    )
    parser.add_argument(
        "--noise-baseline-pp",
        type=float,
        default=0.0,
        help=(
            "accuracy paired verdict: |Δpp| must exceed this run-to-run noise "
            "baseline before a regression/improvement label is printed "
            "(default 0 = report significance only)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        help="output root used to resolve run ids (default: config run.output_dir)",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit structured JSON instead of a table"
    )
    parser.add_argument(
        "--paired",
        action="store_true",
        help=(
            "compare exactly two runs on common sample ids: performance uses "
            "samples.csv (paired bootstrap), accuracy uses (scenario, sample_id) "
            "with McNemar/Holm"
        ),
    )
    args = parser.parse_args(argv)

    output_root = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else _default_output_root()
    )
    try:
        targets = [_resolve_target(target, output_root) for target in args.runs]
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except (KeyError, ValueError) as exc:
        # Preserve the historical exit code for unresolved/invalid targets.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    method, error = _select_method(targets, args.method)
    if error is not None:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    if method == ACCURACY_METHOD:
        return _run_accuracy(args, targets)
    return _run_performance(args, targets)
