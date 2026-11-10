#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Difficulty pilot: run a small stratified slice of each scenario on the reference
small model (oMLX /v1/chat/completions), judge locally, and report the accuracy band.

Purpose: prove the selected pools are neither saturated nor at the floor for a
~1-2B model, and prove the judge contracts run end-to-end. This is NOT the
accuracy test itself (which needs bf16 vs quantized runs and the full 2,000 rows).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tool.system import judges  # noqa: E402

TESTSET = ROOT / "dataset" / "test" / "accuracy" / "testset.jsonl"
OUT = ROOT / "reports" / "dataset-build" / "accuracy"
SCENARIOS = [
    "knowledge_qa",
    "instruction_following",
    "long_context",
    "math_logic",
    "coding",
    "general_agent",
    "search_agent",
    "multilingual",
]
SCEN_MAX = {
    "knowledge_qa": 1024,
    "instruction_following": 1024,
    "long_context": 64,
    "math_logic": 512,
    "coding": 768,
    "general_agent": 256,
    "search_agent": 32,
    "multilingual": 1024,
}


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def stratified_slice(rows, limit):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["meta"].get("stratum", ""))].append(r)
    keys = sorted(groups)
    out, level = [], 0
    while len(out) < limit:
        added = False
        for k in keys:
            if level < len(groups[k]):
                out.append(groups[k][level])
                added = True
                if len(out) >= limit:
                    break
        if not added:
            break
        level += 1
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - s) / d, (c + s) / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="MiniCPM-V-4.6")
    ap.add_argument("--api-key", default=os.environ.get("OMLX_API_KEY", ""))
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--scenario", default="all")
    ap.add_argument("--out", default=str(OUT / "pilot"))
    ap.add_argument("--timeout", type=float, default=900)
    args = ap.parse_args()

    rows_by_scenario = {}
    for row in read_jsonl(TESTSET):
        rows_by_scenario.setdefault(row["scenario"], []).append(row)
    if not args.api_key:
        settings = Path.home() / ".omlx" / "settings.json"
        args.api_key = json.load(open(settings))["auth"]["api_key"]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scenarios = SCENARIOS if args.scenario == "all" else [args.scenario]
    summary = {}
    client = httpx.Client(timeout=args.timeout, limits=httpx.Limits(max_connections=1))

    for scen in scenarios:
        rows = rows_by_scenario[scen]
        picks = stratified_slice(rows, args.limit)
        records = []
        t0 = time.time()
        for idx, row in enumerate(picks, 1):
            max_new = max(SCEN_MAX.get(scen, 256), int(row["judge"].get("max_new_tokens", 0)))
            payload = {
                "model": args.model,
                "messages": row["messages"],
                "temperature": 0,
                "top_p": 1,
                "max_tokens": max_new,
            }
            started = time.time()
            try:
                resp = client.post(
                    args.base_url + "/v1/chat/completions",
                    headers={"Authorization": f"Bearer {args.api_key}"},
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"].get("content") or ""
                usage = data.get("usage", {})
            except Exception as exc:  # noqa: BLE001
                text, usage = "", {"error": f"{type(exc).__name__}: {exc}"}
            elapsed = round(time.time() - started, 2)
            judged = judges.evaluate(row, text)
            rec = {
                "id": row["id"],
                "scenario": scen,
                "dataset": row["dataset"],
                "language": row["language"],
                "stratum": row["meta"].get("stratum"),
                "score": judged["score"],
                "pass": judged.get("pass"),
                "judge": judged,
                "latency_s": elapsed,
                "usage": usage,
                "response": text[:2000],
            }
            records.append(rec)
            print(
                f"[{scen}] {idx}/{len(picks)} score={judged['score']:.3f} "
                f"lat={elapsed}s tok={usage.get('completion_tokens')} {row['id'][:60]}",
                flush=True,
            )
        n = len(records)
        correct = sum(1 for r in records if r["score"] >= 0.999)
        mean = statistics.mean(r["score"] for r in records) if n else 0.0
        lo, hi = wilson(correct, n)
        band = "floor" if mean < 0.15 else "ceiling" if mean > 0.85 else "usable"
        summary[scen] = {
            "n": n,
            "score_mean": round(mean, 4),
            "pass_rate": round(correct / n, 4) if n else 0.0,
            "wilson95": [round(lo, 4), round(hi, 4)],
            "band": band,
            "elapsed_s": round(time.time() - t0, 1),
            "datasets": dict(Counter(r["dataset"] for r in records)),
        }
        with open(out / f"{scen}.jsonl", "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[{scen}] mean={mean:.3f} pass={correct}/{n} band={band}", flush=True)

    with open(out / "pilot_summary.json", "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "limit": args.limit, "scenarios": summary}, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
