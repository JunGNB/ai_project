#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild ``dataset/test/performance/testset.jsonl`` for one target model.

Why this exists
---------------
The committed test set is built with a **generic reference tokenizer**
(``cl100k_base``) so the repository is reproducible without owning a model.  A
real run, however, must present the model an *exact* prompt length: the bucket
targets are 1024 / 8192 / 32768 prompt tokens and the runner runs the set with
``strict_input_tokens: true``, which skips any sample whose measured prompt
token count differs.  Tokenizers disagree, so the rows are rebuilt here with
the tokenizer the model under test actually uses, and every row is verified
through the server's own ``/v1/tokenize`` endpoint (the exact count the server
will report at request time, chat template included).

What is rebuilt and what is not
-------------------------------
Only the *length* of a row changes: the 10 targets per scenario, their upstream
documents, the task block and the sampling order are all fixed by
``build.py`` with ``sha256(seed|id)`` ordering, so the aligned set is the same
design at the same seed.  The alignment is deterministic for a given
(tokenizer, reserve) pair.

Modes
-----
Server mode (production)::

    python tool/dataset/performance/align.py --tokenizer <model>/tokenizer.json \
        --base-url http://127.0.0.1:8000 --model <model-id>

  * derive the chat-template reserve from a probe request,
  * rebuild the 240 rows with ``content = bucket - reserve`` tokens,
  * verify every row through ``/v1/tokenize`` and fail loudly on any mismatch.

Offline mode (pre-flight / CI, no server)::

    python tool/dataset/performance/align.py --tokenizer <model>/tokenizer.json --reserve 9

  * reserve must be supplied by the caller; rows are rebuilt but only locally
    validated (``content + reserve == bucket``), which is a necessary, not a
    sufficient, condition for the server count.  Use server mode before trusting
    results.

Auth: the server's ``/v1/tokenize`` may require a bearer token.  The key is
resolved from ``--api-key``, then ``$OMLX_API_KEY``, then the modelbench config
(``server`` block, same source the runner uses).

Outputs: ``dataset/test/performance/testset.jsonl`` (gitignored) and an
alignment report under ``dataset/.cache/performance/`` (gitignored).  The
committed ``testset.yaml`` / ``manifest.json`` describe the reference build and
are not rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from tool.dataset.performance import build as builder  # noqa: E402

PROBE_TEXT = (
    "Alignment probe. Tokenizers disagree on how text is split into tokens; "
    "this sentence only exists to measure the chat template overhead."
)
MAX_ALIGN_ATTEMPTS = 3


# --------------------------------------------------------------------------- #
# server tokenize
# --------------------------------------------------------------------------- #
def resolve_api_key(base_url: str) -> str:
    """API key for the target server: ``OMLX_API_KEY`` first, then modelbench config."""
    env_key = os.environ.get("OMLX_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        from modelbench.adapters.config.settings import load_config

        cfg = load_config()
        if cfg.server.base_url.rstrip("/") == base_url.rstrip("/"):
            return (cfg.resolve_api_key() or "").strip()
    except Exception:  # noqa: BLE001 - alignment must still work without the package
        pass
    return ""


def post_tokenize(
    base_url: str,
    messages: List[Dict[str, Any]],
    model: Optional[str],
    api_key: str = "",
    timeout_s: float = 60.0,
) -> int:
    payload: Dict[str, Any] = {"messages": messages}
    if model:
        payload["model"] = model
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/tokenize",
        data=data,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise SystemExit(f"tokenize failed: HTTP {exc.code}: {detail}")
    except (urllib.error.URLError, OSError) as exc:
        raise SystemExit(f"tokenize failed: {exc}")
    return int(body["count"])


def measure_reserve(
    base_url: str, model: Optional[str], counter: "builder.Counter", api_key: str = ""
) -> Dict[str, Any]:
    """Chat-template overhead = rendered prompt tokens - raw content tokens."""
    content_tokens = counter.n(PROBE_TEXT)
    total = post_tokenize(base_url, [{"role": "user", "content": PROBE_TEXT}], model, api_key)
    return {
        "probe_content_tokens": content_tokens,
        "probe_prompt_tokens": total,
        "reserve_tokens": total - content_tokens,
    }


# --------------------------------------------------------------------------- #
# rebuild + verify
# --------------------------------------------------------------------------- #
def render(
    counter: "builder.Counter",
    reserve: int,
    scenarios: List[str],
    buckets: List[str],
) -> tuple[List[Dict[str, Any]], Dict[tuple, str], Dict[str, List[Dict[str, Any]]]]:
    builder.CONTENT_RESERVE = reserve
    # Pre-flight alignment renders exact-length rows (the committed test set is
    # built oversized and trimmed by the runner); this keeps the verification
    # semantically identical to a run whose prompts need no trimming.
    builder.OVERSIZE = 1
    targets_by_scenario = builder.build_targets(counter, scenarios)
    rows: List[Dict[str, Any]] = []
    bodies: Dict[tuple, str] = {}
    for scenario in scenarios:
        for index, target in enumerate(targets_by_scenario[scenario]):
            for bucket in buckets:
                row, body = builder.make_row(counter, scenario, target, bucket, index)
                rows.append(row)
                bodies[(row["target_id"], bucket)] = body
    return rows, bodies, targets_by_scenario


def verify_on_server(
    rows: List[Dict[str, Any]],
    base_url: str,
    model: Optional[str],
    api_key: str = "",
) -> List[Dict[str, Any]]:
    bad: List[Dict[str, Any]] = []
    for row in rows:
        expected = row["tokens"]
        got = post_tokenize(base_url, row["messages"], model, api_key)
        if got != expected:
            bad.append(
                {
                    "id": row["id"],
                    "scenario": row["scenario"],
                    "bucket": row["bucket"],
                    "expected": expected,
                    "got": got,
                }
            )
    return bad


def fix_rows_on_server(
    counter: "builder.Counter",
    rows: List[Dict[str, Any]],
    bodies: Dict[tuple, str],
    targets_by_scenario: Dict[str, List[Dict[str, Any]]],
    bad: List[Dict[str, Any]],
    base_url: str,
    model: Optional[str],
    api_key: str,
    reserve: int,
    max_passes: int = 3,
) -> List[Dict[str, Any]]:
    """Correct only the offending rows, one token at a time, and re-verify each.

    A single global reserve cannot fix heterogeneous drift (some rows +1, others
    -2): each row's content budget is shifted by its own measured delta instead.
    Returns the list of applied adjustments.
    """
    position = {row["id"]: pos for pos, row in enumerate(rows)}
    target_of = {
        target["target_id"]: (scenario, index, target)
        for scenario, targets in targets_by_scenario.items()
        for index, target in enumerate(targets)
    }
    applied: List[Dict[str, Any]] = []
    for item in bad:
        original = rows[position[item["id"]]]
        scenario, index, target = target_of[original["target_id"]]
        bucket = original["bucket"]
        delta = item["got"] - item["expected"]
        for _ in range(max_passes):
            try:
                builder.CONTENT_RESERVE = max(0, reserve + delta)
                new_row, new_body = builder.make_row(counter, scenario, target, bucket, index)
            finally:
                builder.CONTENT_RESERVE = reserve
            got = post_tokenize(base_url, new_row["messages"], model, api_key)
            if got == item["expected"]:
                rows[position[item["id"]]] = new_row
                bodies[(new_row["target_id"], bucket)] = new_body
                applied.append(
                    {
                        "id": new_row["id"],
                        "expected": item["expected"],
                        "before": item["got"],
                        "reserve_shift": delta,
                    }
                )
                break
            delta += got - item["expected"]
    return applied


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default=str(builder.RAW_DIR))
    parser.add_argument("--out-dir", default=str(builder.OUT_DIR))
    parser.add_argument("--tokenizer", required=True, help="tokenizer.json of the model under test")
    parser.add_argument("--base-url", default="", help="oMLX server, e.g. http://127.0.0.1:8000")
    parser.add_argument("--model", default="", help="model id on the server")
    parser.add_argument("--reserve", type=int, default=None, help="chat-template token overhead")
    parser.add_argument(
        "--api-key",
        default="",
        help="bearer token for /v1/tokenize; default: $OMLX_API_KEY, else modelbench config",
    )
    parser.add_argument("--report-dir", default=str(PROJECT_ROOT / "dataset" / ".cache" / "performance"))
    parser.add_argument("--keep-report", action="store_true", help="always write the report")
    args = parser.parse_args(argv)

    builder.RAW_DIR = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    counter = builder.Counter(Path(args.tokenizer))
    tokenizer_sha = builder.sha256_file(counter.path)
    scenarios = list(builder.SCENARIOS)
    buckets = list(builder.BUCKETS)

    server_mode = bool(args.base_url)
    if server_mode and not args.model:
        parser.error("--model is required with --base-url")
    if not server_mode and args.reserve is None:
        parser.error("offline mode requires --reserve (see --help)")

    info: Dict[str, Any] = {
        "tokenizer": {"path": str(counter.path), "sha256": tokenizer_sha, "vocab_size": counter._tok.get_vocab_size()},
        "server": {"base_url": args.base_url or None, "model": args.model or None},
        "buckets": buckets,
        "attempts": [],
    }

    reserve = int(args.reserve or 0)
    api_key = ""
    if server_mode:
        api_key = (args.api_key or resolve_api_key(args.base_url)).strip()
        measured = measure_reserve(args.base_url, args.model, counter, api_key)
        reserve = measured["reserve_tokens"]
        info["probe"] = measured
        if reserve < 0:
            raise SystemExit(f"implausible reserve {reserve} (prompt shorter than content)")

    for attempt in range(1, MAX_ALIGN_ATTEMPTS + 1):
        t0 = time.time()
        rows, bodies, targets = render(counter, reserve, scenarios, buckets)
        report = builder.validate_rows(rows, targets, buckets, bodies)
        if report["problems"]:
            raise SystemExit("local validation failed:\n  " + "\n  ".join(report["problems"][:20]))
        bad = verify_on_server(rows, args.base_url, args.model, api_key) if server_mode else []
        applied: List[Dict[str, Any]] = []
        if bad:
            # row-local fixes first: a single global reserve cannot absorb per-row drift
            applied = fix_rows_on_server(
                counter, rows, bodies, targets, bad, args.base_url, args.model, api_key, reserve
            )
            if applied:
                report = builder.validate_rows(rows, targets, buckets, bodies)
                if report["problems"]:
                    raise SystemExit("local validation failed after row fixes:\n  " + "\n  ".join(report["problems"][:20]))
                bad = verify_on_server(rows, args.base_url, args.model, api_key)
        delta = 0
        if bad:
            deltas = [item["got"] - item["expected"] for item in bad]
            deltas.sort()
            delta = deltas[len(deltas) // 2]
        info["attempts"].append(
            {
                "attempt": attempt,
                "reserve": reserve,
                "seconds": round(time.time() - t0, 1),
                "rows": len(rows),
                "mismatches": len(bad),
                "median_delta": delta if bad else 0,
                "row_fixes": applied,
                "first_mismatches": bad[:5],
            }
        )
        if not bad:
            break
        if attempt == MAX_ALIGN_ATTEMPTS:
            raise SystemExit(
                f"strict alignment failed after {MAX_ALIGN_ATTEMPTS} attempts: "
                f"{len(bad)} rows differ, e.g. {bad[:3]}"
            )
        # one extra template token shows up as a constant offset on every row; fold it
        # into the reserve and rebuild (content shrinks by the same amount).
        reserve = max(0, reserve + delta)
        print(f"[align] attempt {attempt}: {len(bad)} mismatches (median delta {delta}) -> reserve {reserve}")

    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    target = out_dir / "testset.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(payload, encoding="utf-8")
    rows_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    info.update(
        {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "reserve_tokens": reserve,
            "rows": len(rows),
            "rows_sha256": rows_sha,
            "cells": {f"{s}/{b}": n for s, per in report["cells"].items() for b, n in per.items()},
            "langs": report["langs"],
            "verified_on_server": bool(server_mode),
            "output": str(target),
        }
    )
    out_dir_report = out_dir / "testset.jsonl"
    print(f"[align] {out_dir_report}  ({len(payload.encode('utf-8')) / 1e6:.1f} MB, {len(rows)} rows)")
    print(f"[align] reserve={reserve}  rows_sha256={rows_sha[:16]}…  verified={server_mode}")
    if server_mode or args.keep_report:
        report_dir = Path(args.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        slug = (args.model or counter.path.parent.name).replace("/", "_") or "model"
        path = report_dir / f"alignment-{slug}.json"
        path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[align] report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
