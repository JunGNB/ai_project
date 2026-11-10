"""CLI: compare/analyze run aggregates (schema 3.0).

Targets may be:

* a run **directory** (``output/runs/2026-11/<batch>/performance/<slug>/``) — read
  from its ``aggregates.json``;
* an explicit ``aggregates.json`` path;
* a single-segment **run_id** (``<batch>__<method>__<model_slug>``), resolved
  against the output root through ``catalog.resolve_run``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .compare import (
    DEFAULT_METRICS,
    compare_summaries,
    load_aggregates,
    paired_comparison,
    render_paired,
)


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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="modelbench-analysis",
        description=(
            "Compare two or more performance runs by aggregates.json. "
            "Targets are run directories, aggregates.json paths, or run ids."
        ),
    )
    parser.add_argument(
        "runs", nargs="+", help="run directories, aggregates.json paths, or run ids"
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
    )
    parser.add_argument(
        "--output-dir",
        help="output root used to resolve run ids (default: config run.output_dir)",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit aggregates as JSON instead of a table"
    )
    parser.add_argument(
        "--paired",
        action="store_true",
        help="compare exactly two runs on common sample_id values (paired bootstrap 95%% CI)",
    )
    args = parser.parse_args(argv)

    output_root = Path(args.output_dir).expanduser() if args.output_dir else _default_output_root()
    documents: List[Dict[str, Any]] = []
    try:
        for target in args.runs:
            documents.append(load_aggregates(_resolve_target(target, output_root)))
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except (KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.paired:
        if len(args.runs) != 2:
            print("ERROR: --paired requires exactly two runs", file=sys.stderr)
            return 2
        try:
            result = paired_comparison(
                _resolve_target(args.runs[0], output_root),
                _resolve_target(args.runs[1], output_root),
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
