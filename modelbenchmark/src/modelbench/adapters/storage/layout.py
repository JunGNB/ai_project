"""Filesystem layout for the schema-3.0 output tree.

Single source of truth for every path under ``output/``.  Writers, the catalog,
the web console, the Excel exporter and the verification tools must all go
through these helpers instead of joining path components by hand; the layout is
the contract described in ``docs/design/31-output-contract.md`` (design rationale in
``docs/archive/OUTPUT_LAYOUT.md``).

The tree is deliberately **flat**: one directory per run under ``runs/``, with
the batch id (which already carries the date), the test method and the model
slug joined into a single name.  No ``YYYY-MM/`` bucket and no ``performance/``
level — ``ls output/runs/`` shows every result at a glance::

    output/
    ├── index.json
    ├── blobs/prompts/<sha[:2]>/<sha>.json      # content-addressed prompt pool
    ├── runs/<run_id>/                          # run_id = batch__method__model
    │   ├── run.json  MANIFEST.json  records.jsonl
    │   ├── samples.csv  aggregates.json  report.md
    │   └── logs/run.log
    ├── batches/<batch_id>.json                 # one action -> one file
    ├── exports/<batch_id>.xlsx
    ├── archive/...
    └── .state/{index.lock,benchmark.lock,tmp/}
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

#: Top-level directory names.
RUNS = "runs"
BATCHES = "batches"
BLOBS = "blobs"
PROMPTS = "prompts"
EXPORTS = "exports"
ARCHIVE = "archive"
STATE = ".state"
TMP = "tmp"

#: Per-run file names.
RUN_JSON = "run.json"
MANIFEST_JSON = "MANIFEST.json"
RECORDS = "records.jsonl"
SAMPLES_CSV = "samples.csv"
AGGREGATES_JSON = "aggregates.json"
REPORT_MD = "report.md"
LOGS_DIR = "logs"
LOG_NAME = "run.log"

#: Batch metadata file name (inside ``batches/``).
BATCH_JSON = "batch.json"

#: Global index.
INDEX_JSON = "index.json"

#: Runtime coordination files.
INDEX_LOCK = "index.lock"
BENCHMARK_LOCK = "benchmark.lock"

#: Separator between the run_id components.  ``run_id`` is a single path
#: segment: it is both the index key and (now that the tree is flat) the run
#: directory name.
RUN_ID_SEP = "__"

_ISO_PREFIX = re.compile(r"^(\d{4})-(\d{2})")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_run_id(batch_id: str, method: str, model_slug: str) -> str:
    """Single-segment stable run identity == the run directory name."""
    return f"{batch_id}{RUN_ID_SEP}{method}{RUN_ID_SEP}{model_slug}"


def parse_run_id(run_id: str) -> Tuple[str, str, str]:
    """Inverse of :func:`make_run_id`; raises ``ValueError`` when malformed."""
    parts = str(run_id).split(RUN_ID_SEP)
    if len(parts) != 3 or not all(parts):
        raise ValueError(
            f"invalid run_id {run_id!r}: expected "
            f"<batch>{RUN_ID_SEP}<method>{RUN_ID_SEP}<model_slug>"
        )
    return parts[0], parts[1], parts[2]


def runs_dir(output_root: Path) -> Path:
    return Path(output_root) / RUNS


def run_dir(output_root: Path, run_id: str) -> Path:
    """``output/runs/<run_id>`` (single segment, no nesting)."""
    return runs_dir(output_root) / str(run_id)


def run_dir_for(
    output_root: Path,
    batch_id: str,
    method: str,
    model_slug: str,
) -> Path:
    return run_dir(output_root, make_run_id(batch_id, method, model_slug))


def batches_dir(output_root: Path) -> Path:
    return Path(output_root) / BATCHES


def batch_path(output_root: Path, batch_id: str) -> Path:
    return batches_dir(output_root) / f"{batch_id}.json"


def blob_path(output_root: Path, sha: str) -> Path:
    sha = str(sha)
    return Path(output_root) / BLOBS / PROMPTS / sha[:2] / f"{sha}.json"


def blobs_root(output_root: Path) -> Path:
    return Path(output_root) / BLOBS / PROMPTS


def exports_dir(output_root: Path) -> Path:
    return Path(output_root) / EXPORTS


def archive_dir(output_root: Path) -> Path:
    return Path(output_root) / ARCHIVE


def state_dir(output_root: Path) -> Path:
    return Path(output_root) / STATE


def index_path(output_root: Path) -> Path:
    return Path(output_root) / INDEX_JSON


def index_lock_path(output_root: Path) -> Path:
    return state_dir(output_root) / INDEX_LOCK


def benchmark_lock_path(output_root: Path) -> Path:
    return state_dir(output_root) / BENCHMARK_LOCK


def tmp_dir(output_root: Path) -> Path:
    return state_dir(output_root) / TMP


def is_safe_rel_path(rel: str) -> bool:
    """True when ``rel`` is a plain relative path that cannot escape its root."""
    if not rel:
        return False
    text = str(rel).replace("\\", "/")
    if text.startswith("/") or re.match(r"^[A-Za-z]:", text):
        return False
    parts = [part for part in text.split("/") if part not in ("", ".")]
    return bool(parts) and ".." not in parts


def resolve_under(root: Path, rel: str) -> Path:
    """Resolve ``rel`` under ``root``, refusing traversal.

    Raises ``ValueError`` for unsafe input and for paths that resolve outside
    ``root`` (defence in depth against symlinks and odd separators).
    """
    if not is_safe_rel_path(rel):
        raise ValueError(f"unsafe relative path: {rel!r}")
    root = Path(root).resolve()
    candidate = (root / str(rel).replace("\\", "/")).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"path escapes output root: {rel!r}")
    return candidate
