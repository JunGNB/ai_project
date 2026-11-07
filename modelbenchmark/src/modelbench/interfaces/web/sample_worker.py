"""One-shot dataset preview worker for the web sample browser.

Large datasets (e.g. LongBench-v2's 465MB JSON) can briefly use hundreds of MB
to a few GB while parsing. Running that in the long-lived web-server process
would keep the memory reserved for the rest of the session, so the server
spawns this worker for large dataset directories and reads a small JSON preview
from its stdout. The child exits immediately and its memory is reclaimed.

Input (stdin, JSON)::

    {"dataset": "longbench_v2", "params": {...},
     "raw_dir": "...", "processed_dir": "..."}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from modelbench.adapters.datasets.registry import get_adapter

_QUESTION_CHARS = 180


def preview_samples(samples: List[Any]) -> List[Dict[str, Any]]:
    """Small, JSON-serializable metadata rows for the sample browser."""
    return [
        {
            "sample_id": sample.sample_id,
            "subtask": sample.subtask,
            "question": " ".join((sample.question or "").split())[:_QUESTION_CHARS],
            "context_chars": sample.context_chars,
            "est_context_tokens": sample.est_context_tokens,
        }
        for sample in samples
    ]


def build_preview(
    raw_dir: str,
    processed_dir: str,
    dataset: str,
    params: Dict[str, Any] | None = None,
    testset_dir: str | None = None,
) -> Dict[str, Any]:
    adapter = get_adapter(
        dataset,
        raw_dir=Path(raw_dir),
        processed_dir=Path(processed_dir),
        params=dict(params or {}),
        testset_dir=Path(testset_dir) if testset_dir else None,
    )
    samples = adapter.load()
    return {"dataset": dataset, "total": len(samples), "samples": preview_samples(samples)}


def main(argv: None = None) -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        result = build_preview(
            raw_dir=str(payload["raw_dir"]),
            processed_dir=str(payload["processed_dir"]),
            dataset=str(payload["dataset"]),
            params=payload.get("params") or {},
            testset_dir=payload.get("testset_dir"),
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the parent process
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
