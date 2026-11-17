"""Summarize a built performance test set: per-cell sizes, primary share, validity flags.

    .venv/bin/python -m tool.dataset.performance.review

Reads ``dataset/test/performance/testset.jsonl`` + ``manifest.json`` and prints one row
per ``scenario/bucket`` cell. This is a read-only report used when reviewing whether the
constructed samples still satisfy the benchmark conditions (see docs/spec/11 §8).
"""

from __future__ import annotations

import collections
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List

TEST_DIR = Path(__file__).resolve().parents[3] / "dataset" / "test" / "performance"


def load() -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in (TEST_DIR / "testset.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = json.loads((TEST_DIR / "manifest.json").read_text(encoding="utf-8"))
    return rows, manifest


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cells: Dict[tuple, List[Dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        cells[(row["scenario"], row["bucket"])].append(row)
    out = []
    for (scenario, bucket), items in sorted(cells.items()):
        metas = [row["meta"] for row in items]
        out.append(
            {
                "cell": f"{scenario}/{bucket}",
                "n": len(items),
                "content_median": int(statistics.median(m["content_tokens"] for m in metas)),
                "target": int(metas[0]["target_tokens"]),
                "primary_share": round(
                    statistics.mean(float(m["primary_share"]) for m in metas), 3
                ),
                "task_max": max(int(m["task_tokens"]) for m in metas),
                "primary_model_max": max(int(m["primary_model_tokens"]) for m in metas),
                "fits": all(bool(m["fits_bucket"]) for m in metas),
                "primary_complete": all(bool(m["primary_complete"]) for m in metas),
                "evidence_ok": sum(1 for m in metas if m.get("evidence_in_primary") is not False),
                "langs": sorted({m["lang"] for m in metas}),
            }
        )
    return out


def main() -> None:
    rows, manifest = load()
    print(f"rows={len(rows)} buckets={manifest['buckets']} sha={manifest['rows_sha256'][:16]}")
    header = (
        f"{'cell':34s} {'n':>2s} {'content':>7s} {'target':>6s} {'prim%':>6s} "
        f"{'task':>4s} {'pmax':>6s} {'fit':>3s} {'evid':>5s}  langs"
    )
    print(header)
    for item in summarize(rows):
        print(
            f"{item['cell']:34s} {item['n']:2d} {item['content_median']:7d} "
            f"{item['target']:6d} {item['primary_share'] * 100:5.1f}% {item['task_max']:4d} "
            f"{item['primary_model_max']:6d} {str(item['fits']):>3s} "
            f"{item['evidence_ok']:2d}/{item['n']}  {','.join(item['langs'])}"
        )


if __name__ == "__main__":
    main()
