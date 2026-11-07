"""Derive the quick performance test set: 8 scenarios x 1K/8K/32K x 10 = 240 rows.

The subset is deterministic: per scenario, take every second ``target_id`` of the
full set (10 of 20); each target contributes its three bucket rows, so the
"same target, different length" property is preserved.  Rows are copied
byte-for-byte (dict equality is asserted, not re-generated).

Writes ``dataset/test/performance/testset_10.jsonl`` and ``manifest_10.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as CFG
import ds_common as C

FULL = C.TEST_PERF / "testset.jsonl"
OUT = C.TEST_PERF / "testset_10.jsonl"
MANIFEST = C.TEST_PERF / "manifest_10.json"
GENERATOR_VERSION = "tool/dataset/build_perf_subset.py v4"

#: Manifest keys copied unchanged from ``manifest.json`` (provenance/budget/contract).
SYNCED_KEYS = (
    "seed",
    "tokenizer",
    "tokenizer_cache",
    "buckets",
    "accept_range",
    "accept_bounds",
    "accept_rule",
    "osl",
    "run_defaults",
    "tokens_semantics",
    "bucket_basis",
    "construction_rules",
    "workloads",
    "scenarios",
    "window_requirements",
    "provenance",
)


def cell_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    for row in rows:
        key = f"{row['scenario']}|{row['bucket']}"
        entry = stats.setdefault(
            key,
            {"n": 0, "tokens": [], "constructed": 0, "dataset": row["dataset"], "langs": set()},
        )
        entry["n"] += 1
        entry["tokens"].append(row["tokens"])
        entry["constructed"] += int(row["constructed"])
        lang = (row.get("meta") or {}).get("lang")
        if lang:
            entry["langs"].add(lang)
    for entry in stats.values():
        values = sorted(entry.pop("tokens"))
        entry.update(
            tokens_min=values[0],
            tokens_median=values[len(values) // 2],
            tokens_max=values[-1],
            tokens_mean=round(sum(values) / len(values), 1),
            langs=sorted(entry.pop("langs")),
        )
    return stats


def main() -> int:
    rows = [json.loads(line) for line in FULL.open() if line.strip()]
    by_scenario: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_scenario.setdefault(row["scenario"], []).append(row)

    kept: List[Dict[str, Any]] = []
    selected_targets: Dict[str, List[str]] = {}
    for scenario, items in sorted(by_scenario.items()):
        targets = sorted({row["target_id"] for row in items})
        if len(targets) != CFG.PERF_PER_BUCKET:
            raise SystemExit(f"{scenario}: expected {CFG.PERF_PER_BUCKET} targets, got {len(targets)}")
        chosen = targets[::2][: CFG.PERF_SUBSET_PER_BUCKET]
        if len(chosen) != CFG.PERF_SUBSET_PER_BUCKET:
            raise SystemExit(f"{scenario}: subset selection produced {len(chosen)} targets")
        selected_targets[scenario] = chosen
        chosen_set = set(chosen)
        kept += [row for row in items if row["target_id"] in chosen_set]

    order = {"1k": 0, "8k": 1, "32k": 2}
    kept.sort(key=lambda row: (row["scenario"], order[row["bucket"]], row["id"]))

    # ---- subset invariants (fail loudly instead of writing a broken file) ---- #
    if len(kept) != CFG.EXPECTED["perf_subset_rows"]:
        raise SystemExit(f"subset rows {len(kept)} != {CFG.EXPECTED['perf_subset_rows']}")
    cells: Dict[tuple, int] = {}
    target_buckets: Dict[str, List[str]] = {}
    for row in kept:
        cells[(row["scenario"], row["bucket"])] = cells.get((row["scenario"], row["bucket"]), 0) + 1
        target_buckets.setdefault(row["target_id"], []).append(row["bucket"])
    if len(cells) != 24 or set(cells.values()) != {CFG.PERF_SUBSET_PER_BUCKET}:
        raise SystemExit(f"subset cells invalid: {cells}")
    if len(target_buckets) != 80 or any(sorted(v) != ["1k", "32k", "8k"] for v in target_buckets.values()):
        raise SystemExit("subset target triples invalid")
    full_by_id = {row["id"]: row for row in rows}
    for row in kept:
        if full_by_id.get(row["id"]) != row:
            raise SystemExit(f"subset row {row['id']} differs from the full set")
    for scenario, chosen in selected_targets.items():
        actual = sorted({row["target_id"] for row in kept if row["scenario"] == scenario})
        if actual != sorted(chosen):
            raise SystemExit(f"subset selection mismatch for {scenario}")

    C.write_jsonl(OUT, kept)

    full_manifest = json.loads((C.TEST_PERF / "manifest.json").read_text(encoding="utf-8"))
    missing = [key for key in SYNCED_KEYS if key not in full_manifest]
    if missing:
        raise SystemExit(f"full manifest missing synced keys: {missing}")
    manifest = {
        "schema_version": 3,
        "generator_version": GENERATOR_VERSION,
        "subset_of": FULL.name,
        "selection": (
            "per scenario: sorted target_id[::2], first "
            f"{CFG.PERF_SUBSET_PER_BUCKET} (each keeps its 1k/8k/32k rows)"
        ),
        "files": [OUT.name],
        **{key: full_manifest[key] for key in SYNCED_KEYS},
        "sampling": full_manifest.get("sampling"),
        "sampling_note": (
            "sampling/skip report belongs to the full performance selection; "
            "performance_10 keeps the first 10 sorted targets per scenario from it"
        ),
        "selected_targets": selected_targets,
        "stats": cell_stats(kept),
        "tokens_mean": round(sum(row["tokens"] for row in kept) / len(kept), 1),
        "totals": {
            "rows": len(kept),
            "cells": len(cells),
            "per_cell": CFG.PERF_SUBSET_PER_BUCKET,
            "measured_requests": len(kept),
            "targets": len(target_buckets),
        },
        "default_run": {
            "measured_requests": len(kept),
            "cells": len(cells),
        },
        "note": "same underlying samples as performance/; not an independent evidence set",
    }
    if manifest["provenance"] != full_manifest["provenance"]:
        raise SystemExit("subset manifest provenance differs from the full set")
    if manifest["totals"]["rows"] != 240 or manifest["totals"]["cells"] != 24:
        raise SystemExit("subset manifest totals invalid")
    C.write_json(MANIFEST, manifest)
    print(f"[perf-10] rows={len(kept)} targets={len(target_buckets)} cells={len(cells)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
