"""CLI: compare/analyze run summaries."""

from __future__ import annotations

import argparse
import json

from .compare import DEFAULT_METRICS, compare_summaries, load_summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="modelbench-analysis",
        description="Compare two or more performance run summaries from output/<run_id>/.",
    )
    parser.add_argument("runs", nargs="+", help="run directories or summary.json paths")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
    )
    parser.add_argument("--json", action="store_true", help="emit summaries as JSON instead of a table")
    args = parser.parse_args(argv)

    summaries = [load_summary(path) for path in args.runs]
    if args.json:
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
        return 0
    print(compare_summaries(summaries, metrics=args.metrics))
    return 0
