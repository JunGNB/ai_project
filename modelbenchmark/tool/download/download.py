#!/usr/bin/env python3
"""Unified dataset download + normalisation entry point.

Usage
-----
::

    # everything this layer owns (MMLU, GSM8K, IFEval, HumanEval, MMMLU,
    # HotpotQA, BFCL v4)
    python -m tool.download all

    # one or more datasets, optional params (JSON)
    python -m tool.download mmlu --params '{"subjects": ["philosophy"], "per_subject": 5}'
    python -m tool.download mmlu gsm8k ifeval

    # download only (skip normalisation into data/processed/)
    python -m tool.download all --no-process

    # machine-readable report (files, sizes, SHA-256, sample counts)
    python -m tool.download all --json --report data/raw/download_report.json

Behaviour
---------
* Idempotent / resumable: each adapter's ``ensure_downloaded()`` reuses a file
  whose SHA-256 matches ``data/raw/<dataset>/download_manifest.json`` and
  otherwise resumes the HTTP transfer (see ``src/modelbench/adapters/datasets/_download.py``).
* Every downloaded file is recorded with URL + size + SHA-256 in that manifest;
  docs/DATASETS.md mirrors the same information for reviewers.
* ``--force`` re-downloads even verified files.
* ``longbench_v2`` is intentionally *not* part of ``all``: it is owned by the
  Lead layer (its 465 MB archive is downloaded separately).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modelbench.adapters.datasets._download import MANIFEST_NAME, DownloadError, list_raw_files  # noqa: E402
from modelbench.adapters.datasets.registry import available_datasets, get_adapter  # noqa: E402

#: Datasets owned by the data layer.  ``longbench_v2`` is deliberately absent
#: (Lead-owned, downloaded separately); it can still be named explicitly.
DEFAULT_DATASETS: tuple[str, ...] = (
    "mmlu",
    "gsm8k",
    "ifeval",
    "humaneval",
    "mmmlu",
    "hotpotqa",
    "bfcl",
)

_ALIASES = {"all": DEFAULT_DATASETS}


def _resolve_datasets(names: List[str]) -> List[str]:
    resolved: List[str] = []
    for name in names or ["all"]:
        items = _ALIASES.get(name, (name,))
        for item in items:
            if item not in resolved:
                resolved.append(item)
    return resolved


def _human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GB"


def _breakdown(samples: List[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for sample in samples:
        key = sample.subtask or "(all)"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _dataset_files(raw_dir: Path) -> List[Dict[str, Any]]:
    return list_raw_files(raw_dir)


def download_one(
    name: str,
    raw_root: Path,
    processed_root: Path,
    params: Optional[Dict[str, Any]] = None,
    *,
    process: bool = True,
) -> Dict[str, Any]:
    """Download (and optionally normalise) one dataset; never raises."""

    started = time.time()
    result: Dict[str, Any] = {
        "dataset": name,
        "raw_dir": str(raw_root / name),
        "files": [],
        "samples": None,
        "breakdown": None,
        "processed": None,
        "elapsed_s": None,
        "error": None,
    }
    try:
        # registry.get_adapter() appends the dataset name to the base raw dir
        adapter = get_adapter(name, raw_root, processed_root, params or {})
        adapter.ensure_downloaded()
        result["files"] = _dataset_files(raw_root / name)
        if process and name in DEFAULT_DATASETS:
            samples = adapter.load()
            processed_path = adapter.save_processed(samples)
            result["samples"] = len(samples)
            result["breakdown"] = _breakdown(samples)
            result["processed"] = str(processed_path)
        skipped = getattr(adapter, "skipped_categories", None)
        if skipped:
            result["skipped_categories"] = list(skipped)
        download_errors = getattr(adapter, "download_errors", None)
        if download_errors:
            result["download_errors"] = download_errors
    except (DownloadError, FileNotFoundError, ValueError, RuntimeError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # pragma: no cover - defensive, keep other datasets going
        result["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    result["elapsed_s"] = round(time.time() - started, 2)
    return result


def _print_human(results: List[Dict[str, Any]]) -> None:
    for result in results:
        name = result["dataset"]
        if result["error"]:
            print(f"[FAIL] {name}: {result['error']}")
            continue
        files = result["files"]
        total_bytes = sum(item["size"] for item in files)
        samples = result["samples"]
        print(
            f"[ok]   {name:<10} files={len(files):<2} {_human_bytes(total_bytes):>9}  "
            f"samples={samples if samples is not None else '-':>6}  "
            f"elapsed={result['elapsed_s']:.2f}s"
        )
        for item in files:
            print(f"         {item['relative_path']}  {item['size']} bytes  sha256={item['sha256']}")
        if result["breakdown"]:
            breakdown = result["breakdown"]
            shown = ", ".join(f"{key}={value}" for key, value in list(breakdown.items())[:8])
            if len(breakdown) > 8:
                shown += f", … ({len(breakdown)} groups)"
            print(f"         breakdown: {shown}")
        if result.get("skipped_categories"):
            print(f"         skipped categories: {', '.join(result['skipped_categories'])}")
        if result.get("download_errors"):
            print(
                f"         download errors: {len(result['download_errors'])} "
                f"(see {Path(result['raw_dir']) / 'download_errors.json'})"
            )


def _write_report(path: Path, results: List[Dict[str, Any]], datasets: List[str]) -> None:
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "project_root": str(PROJECT_ROOT),
        "datasets_requested": datasets,
        "results": results,
        "failed": [item["dataset"] for item in results if item["error"]],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nreport written to {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and normalise benchmark datasets (idempotent, resumable, SHA-256 recorded).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help=f"dataset names, or 'all' (default: {', '.join(DEFAULT_DATASETS)})",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="dataset name (repeatable); alternative to the positional form",
    )
    parser.add_argument(
        "--raw-root",
        default=str(PROJECT_ROOT / "data" / "raw"),
        help="raw data root (default: data/raw)",
    )
    parser.add_argument(
        "--processed-root",
        default=str(PROJECT_ROOT / "data" / "processed"),
        help="normalised JSONL root (default: data/processed)",
    )
    parser.add_argument(
        "--params",
        default=None,
        help="JSON object with adapter params applied to every requested dataset",
    )
    parser.add_argument(
        "--no-process",
        action="store_true",
        help="download only; do not load/normalise/save data/processed/*.jsonl",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download files even when their checksum matches the manifest",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list registered datasets and exit",
    )
    parser.add_argument("--json", action="store_true", help="print a JSON report to stdout")
    parser.add_argument("--report", default=None, help="write the JSON report to this path")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        registered = available_datasets()
        for name in registered:
            marker = " (data layer)" if name in DEFAULT_DATASETS else ""
            print(f"{name}{marker}")
        return 0

    raw_root = Path(args.raw_root).expanduser().resolve()
    processed_root = Path(args.processed_root).expanduser().resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    processed_root.mkdir(parents=True, exist_ok=True)

    params: Dict[str, Any] = {}
    if args.params:
        try:
            params = json.loads(args.params)
        except ValueError as exc:
            print(f"error: --params is not valid JSON: {exc}", file=sys.stderr)
            return 2
        if not isinstance(params, dict):
            print("error: --params must be a JSON object", file=sys.stderr)
            return 2
    if args.force:
        params = {**params, "force_download": True}

    datasets = _resolve_datasets(list(args.datasets or []) + list(args.dataset or []))
    unknown = [name for name in datasets if name not in available_datasets()]
    if unknown:
        print(
            f"error: unknown dataset(s): {', '.join(unknown)}; "
            f"available: {', '.join(available_datasets())}",
            file=sys.stderr,
        )
        return 2

    print(
        f"downloading {len(datasets)} dataset(s) -> {raw_root} "
        f"(processed: {processed_root if not args.no_process else 'disabled'})"
    )
    results = [
        download_one(
            name,
            raw_root,
            processed_root,
            params,
            process=not args.no_process,
        )
        for name in datasets
    ]

    if args.json:
        print(json.dumps({"results": results}, indent=2, ensure_ascii=False))
    else:
        _print_human(results)

    failed = [item["dataset"] for item in results if item["error"]]
    if args.report:
        _write_report(Path(args.report).expanduser().resolve(), results, datasets)
    if failed:
        print(f"\nFAILED: {', '.join(failed)}", file=sys.stderr)
        return 1
    total_samples = sum(item["samples"] or 0 for item in results)
    print(f"\nall {len(results)} dataset(s) ready; {total_samples} normalised samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
