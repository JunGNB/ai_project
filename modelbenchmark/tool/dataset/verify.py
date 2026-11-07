"""Read-only verification of the rebuilt test sets (design v3, optimization pass).

Checks the invariants from ``dataset/TEST_SAMPLE_SELECTION.md`` §6, the
manifest/provenance claims, and the fixes made in the 2026-11 optimization
pass (empty scenario rename, dedupe, official prompts, upstream ids,
provenance-per-set).  Exits non-zero on the first failed section.

    .venv/bin/python tool/dataset/verify.py
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as CFG
import ds_common as C

FAILS: List[str] = []
CHECKS = 0

TOOL_SCENARIOS = {"general_agent", "search_agent"}
DOC_SCENARIOS = {"knowledge_qa", "long_context", "coding", "search_agent"}
PERF_MARKERS = {
    "knowledge_qa": "\n\nQuestion: ",
    "long_context": "\n\nQuestion: ",
    "coding": "\n\nQuestion: ",
    "search_agent": "\n\nQuestion: ",
    "chinese_long_context": "\n\n问题：",
    "instruction_following": "\n\nInstruction: ",
    "math_logic": "\n\nQuestion: ",
    "general_agent": "\n\nUser request:\n",
}
DOC_BODY_PREFIXES = (
    "Read the document and answer the question. You may use the provided tools.\n\n",
    "Read the document and answer the question.\n\n",
)


def check(condition: bool, message: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILS.append(message)


def load_rows(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def schema_types(node: Any) -> List[str]:
    out: List[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "type" and isinstance(value, str):
                out.append(value)
            else:
                out.extend(schema_types(value))
    elif isinstance(node, list):
        for item in node:
            out.extend(schema_types(item))
    return out


def bucket_ok(tokens: int, target: int) -> bool:
    """Integer token counts use the documented floor(±5%) tolerance."""
    return math.floor(CFG.ACCEPT_LOW * target) <= tokens <= math.floor(
        CFG.ACCEPT_HIGH * target
    )


def last_user_text(row: Dict[str, Any]) -> str:
    for message in reversed(row.get("messages") or []):
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


# --------------------------------------------------------------------------- #
# performance
# --------------------------------------------------------------------------- #
def verify_perf() -> None:
    rows = load_rows(C.TEST_PERF / "testset.jsonl")
    manifest = json.loads((C.TEST_PERF / "manifest.json").read_text(encoding="utf-8"))
    check(len(rows) == 480, f"perf rows {len(rows)} != 480")
    check(len({row["id"] for row in rows}) == len(rows), "perf duplicate ids")
    check({row["scenario"] for row in rows} == set(CFG.PERF_SOURCES),
          f"perf scenario set {sorted({row['scenario'] for row in rows})}")
    cells = Counter((row["scenario"], row["bucket"]) for row in rows)
    check(len(cells) == 24 and all(value == 20 for value in cells.values()), f"perf cells {cells}")
    targets = Counter(row["target_id"] for row in rows)
    check(len(targets) == 160 and all(value == 3 for value in targets.values()), "perf target triples")
    check(all(row["tokens"] == row["messages_tokens"] + row["tools_tokens"] for row in rows),
          "perf tokens != messages + tools")
    for row in rows:
        target = CFG.BUCKETS[row["bucket"]]
        check(bucket_ok(row["tokens"], target),
              f"perf {row['id']} tokens {row['tokens']} outside bucket {target}")
        check(bool(row["messages"]) and row["messages"][-1]["role"] == "user",
              f"perf {row['id']} messages shape")
        source = CFG.PERF_SOURCES[row["scenario"]]
        check(row["meta"].get("workload") == source["workload"],
              f"perf {row['id']} workload label")
        check(row["meta"].get("lang"), f"perf {row['id']} missing lang")
    for row in rows:
        if not row["tools"]:
            continue
        bad = [value for value in schema_types(row["tools"])
               if value not in ("object", "array", "string", "number", "integer", "boolean", "null", "function")]
        check(not bad, f"perf {row['id']} invalid tool schema types {bad[:3]}")

    for row in rows:
        expected = row["scenario"] in TOOL_SCENARIOS
        check(bool(row["tools"]) == expected, f"perf {row['id']} tools presence")

    # constructed rules
    for row in rows:
        scenario, bucket = row["scenario"], row["bucket"]
        if scenario in DOC_SCENARIOS:
            expected = False
        elif scenario == "chinese_long_context":
            expected = bucket in ("8k", "32k")
        else:
            expected = True
        check(row["constructed"] == expected, f"perf {row['id']} constructed flag")

    # same target => identical suffix
    by_target: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_target[row["target_id"]][row["bucket"]] = row
    for target_id, buckets in by_target.items():
        marker = PERF_MARKERS[buckets["1k"]["scenario"]]
        suffixes = {}
        for bucket, row in buckets.items():
            content = row["messages"][0]["content"]
            index = content.rfind(marker)
            check(index >= 0, f"perf {row['id']} marker missing")
            suffixes[bucket] = content[index:] if index >= 0 else ""
        check(len(set(suffixes.values())) == 1, f"perf {target_id} suffix differs across buckets")

    # document scenarios: 1K body prefix of 8K prefix of 32K; 1K bodies unique
    for target_id, buckets in by_target.items():
        scenario = buckets["1k"]["scenario"]
        if scenario not in DOC_SCENARIOS:
            continue
        marker = PERF_MARKERS[scenario]
        bodies = {}
        for bucket, row in buckets.items():
            content = row["messages"][0]["content"]
            index = content.rfind(marker)
            body = content[:index]
            for candidate in DOC_BODY_PREFIXES:
                if body.startswith(candidate):
                    body = body[len(candidate):]
                    break
            bodies[bucket] = body
        check(bodies["8k"].startswith(bodies["1k"]), f"perf {target_id} 1K not prefix of 8K")
        check(bodies["32k"].startswith(bodies["8k"]), f"perf {target_id} 8K not prefix of 32K")
    for scenario in DOC_SCENARIOS:
        bodies_1k = {
            row["messages"][0]["content"][: row["messages"][0]["content"].rfind(PERF_MARKERS[scenario])]
            for row in rows
            if row["scenario"] == scenario and row["bucket"] == "1k"
        }
        if scenario == "long_context":
            # LongBench v2 Long Structured Data docs share a 2,651-char boilerplate
            # prefix; the 33-row domain pool cannot yield 20 unique 1K bodies.
            # The collision count is recorded explicitly in manifest.sampling.
            note = manifest.get("sampling", {}).get("long_context", {}).get(
                "known_1k_prefix_collisions", {})
            check(note.get("unique_1k_bodies") == len(bodies_1k),
                  f"perf long_context documented unique_1k_bodies {note.get('unique_1k_bodies')} != {len(bodies_1k)}")
            check(note.get("selected_targets") == 20, "perf long_context selected_targets")
            check(len(bodies_1k) >= 15, f"perf long_context 1K bodies {len(bodies_1k)} < 15")
        else:
            check(len(bodies_1k) == 20, f"perf {scenario} duplicate 1K document bodies ({len(bodies_1k)}/20)")

    # chinese_long_context: target document present in every bucket; 1K contexts differ
    dureader = {
        row["input"]: row["context"]
        for row in C.load_zip_jsonl(C.RAW / "longbench_v1" / "data.zip", "data/dureader.jsonl")
    }
    zh_1k = set()
    for target_id, buckets in by_target.items():
        if buckets["1k"]["scenario"] != "chinese_long_context":
            continue
        check(buckets["1k"]["meta"].get("lang") == "zh", f"perf {target_id} Chinese lang tag")
        content = buckets["1k"]["messages"][0]["content"]
        question = content.rsplit("\n\n问题：", 1)[1].rsplit("\n答案：", 1)[0]
        own = dureader[question]
        for bucket, row in buckets.items():
            check(own[:200] in row["messages"][0]["content"],
                  f"perf {row['id']} missing own document")
        zh_1k.add(content.rsplit("\n\n问题：", 1)[0])
    check(len(zh_1k) == 20, f"perf chinese_long_context 1K unique contexts {len(zh_1k)} != 20")

    # agent catalogue: valid JSON, target-functions-first metadata, category quotas
    prefix = "You are a function-calling assistant. API reference:\n"
    for row in rows:
        if row["scenario"] != "general_agent":
            continue
        content = row["messages"][0]["content"]
        index = content.rfind("\n\nUser request:\n")
        try:
            catalogue = json.loads(content[len(prefix):index])
            check(isinstance(catalogue, list), f"perf {row['id']} agent catalogue is not a list")
        except json.JSONDecodeError:
            check(False, f"perf {row['id']} agent catalogue is not valid JSON")
            continue
        check(row["meta"].get("category") in ("simple", "multiple", "parallel"),
              f"perf {row['id']} BFCL category")
        check(isinstance(row["meta"].get("catalog_functions"), int),
              f"perf {row['id']} catalog_functions meta")
        check(isinstance(row["meta"].get("own_functions_in_catalog"), bool),
              f"perf {row['id']} own_functions_in_catalog meta")
        if row["meta"].get("own_functions_in_catalog"):
            names = {item.get("function", {}).get("name") for item in catalogue}
            own_names = {tool.get("function", {}).get("name") for tool in row["tools"] or []}
            check(own_names and own_names <= names,
                  f"perf {row['id']} own function missing from catalogue")
    categories_1k = Counter(row["meta"]["category"] for row in rows
                            if row["scenario"] == "general_agent" and row["bucket"] == "1k")
    check(categories_1k == Counter({"simple": 10, "multiple": 5, "parallel": 5}),
          f"perf BFCL category quotas {categories_1k}")

    # ifeval shape / math leakage / search tools
    for row in rows:
        content = row["messages"][0]["content"]
        if row["scenario"] == "instruction_following":
            check(content.startswith("Reference instructions:\n") and "\n\nInstruction: " in content,
                  f"perf {row['id']} ifeval shape")
        if row["scenario"] == "search_agent":
            check([tool["function"]["name"] for tool in row["tools"]] == ["search", "lookup"],
                  f"perf {row['id']} search tools")
    for target_id, buckets in by_target.items():
        if buckets["1k"]["scenario"] != "math_logic":
            continue
        content = buckets["1k"]["messages"][0]["content"]
        index = content.rfind("\n\nQuestion: ")
        problem = content[index + len("\n\nQuestion: "):].rsplit("\nAnswer:", 1)[0]
        check(problem not in content[:index], f"perf {target_id} target problem leaked into examples")

    # subset strictness
    subset = load_rows(C.TEST_PERF / "testset_10.jsonl")
    full = {row["id"]: row for row in rows}
    check(len(subset) == 240, f"perf subset rows {len(subset)} != 240")
    check(all(row["id"] in full and full[row["id"]] == row for row in subset),
          "perf subset not identical rows")
    sub_targets = Counter(row["target_id"] for row in subset)
    check(len(sub_targets) == 80 and all(value == 3 for value in sub_targets.values()),
          "perf subset target triples")
    for scenario in CFG.PERF_SOURCES:
        targets_for_scenario = sorted({row["target_id"] for row in rows if row["scenario"] == scenario})
        expected = targets_for_scenario[::2][: CFG.PERF_SUBSET_PER_BUCKET]
        actual = sorted({row["target_id"] for row in subset if row["scenario"] == scenario})
        check(actual == sorted(expected), f"perf subset selection {scenario}")
    subset_manifest = json.loads((C.TEST_PERF / "manifest_10.json").read_text(encoding="utf-8"))
    check(subset_manifest["totals"]["rows"] == 240, "perf subset manifest rows")
    check(subset_manifest.get("selected_targets") is not None, "perf subset manifest targets")
    check(subset_manifest.get("provenance") == manifest.get("provenance"),
          "perf subset provenance differs")
    for key in ("run_defaults", "window_requirements", "sampling", "workloads", "osl"):
        check(key in subset_manifest, f"manifest_10 missing {key}")

    check(manifest["totals"]["rows"] == 480 and manifest["totals"]["cells"] == 24,
          "perf manifest totals")
    check(manifest["tokenizer"]["mode"] in ("local", "omlx"), "perf manifest tokenizer mode")
    check(manifest.get("osl") == CFG.OSL, f"perf manifest osl {manifest.get('osl')} != {CFG.OSL}")
    check(manifest.get("run_defaults", {}).get("max_tokens") == CFG.OSL,
          "perf manifest run_defaults.max_tokens")
    check(isinstance(manifest.get("window_requirements"), dict), "perf manifest window_requirements")
    check(set(CFG.PERF_SOURCES) <= set(manifest.get("sampling", {})),
          "perf manifest sampling scenarios")
    check(manifest.get("workloads"), "perf manifest workloads")
    expected_provenance = {entry["dataset"]: entry for entry in CFG.provenance_for("performance")}
    provenance = manifest.get("provenance", {})
    check(set(provenance) == set(expected_provenance),
          f"perf provenance datasets {sorted(provenance)} != {sorted(expected_provenance)}")
    check("livecodebench" not in provenance, "perf provenance must not contain livecodebench")
    bfcl_files = {item["logical_path"] for item in provenance.get("bfcl", {}).get("files", [])}
    check(len(bfcl_files) == 3 and not any("possible_answer" in name for name in bfcl_files),
          "perf bfcl provenance must only list request files")


# --------------------------------------------------------------------------- #
# accuracy
# --------------------------------------------------------------------------- #
def verify_acc() -> None:
    rows = load_rows(C.TEST_ACC / "testset.jsonl")
    manifest = json.loads((C.TEST_ACC / "manifest.json").read_text(encoding="utf-8"))
    expected_rows = CFG.EXPECTED["acc_rows"]
    check(len(rows) == expected_rows, f"acc rows {len(rows)} != {expected_rows}")
    check(len({row["id"] for row in rows}) == len(rows), "acc duplicate ids")
    counts = Counter(row["scenario"] for row in rows)
    check(set(counts) == set(CFG.ACC_SOURCES)
          and all(counts[name] == spec["n"] for name, spec in CFG.ACC_SOURCES.items()),
          f"acc scenario counts {counts}")
    fields = {"id", "scenario", "dataset", "judge", "tokens", "messages", "choices", "answer", "tools", "meta"}
    check(all(set(row) == fields for row in rows), "acc field set")
    check(manifest.get("status") == "offline-asset", "acc manifest status")
    check("not wired" in str(manifest.get("runner")), "acc manifest runner status")
    for row in rows:
        spec = CFG.ACC_SOURCES[row["scenario"]]
        check(row["judge"] == spec["judge"], f"acc {row['id']} judge {row['judge']} != {spec['judge']}")
        check(isinstance(row["tokens"], int) and row["tokens"] > 0, f"acc {row['id']} tokens")
        check(isinstance(row["meta"].get("upstream_id"), str) and row["meta"]["upstream_id"],
              f"acc {row['id']} missing upstream_id")
        text = last_user_text(row)
        for marker in ("$LETTER", "{context}", "{input}", "{question}"):
            check(marker not in text, f"acc {row['id']} unrendered placeholder {marker}")
        if row["tools"]:
            bad = [value for value in schema_types(row["tools"])
                   if value not in ("object", "array", "string", "number", "integer", "boolean", "null", "function")]
            check(not bad, f"acc {row['id']} invalid tool schema types {bad[:3]}")
    check(len({row["meta"]["upstream_id"] for row in rows}) == len(rows),
          "acc duplicate upstream_id")

    by_scenario: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scenario[row["scenario"]].append(row)

    for scenario, scenario_rows in by_scenario.items():
        prompts = [last_user_text(row) for row in scenario_rows]
        check(len(set(prompts)) == len(prompts), f"acc {scenario} duplicate prompts")

    for row in by_scenario["knowledge_qa"]:
        check(isinstance(row["choices"], list) and len(row["choices"]) >= 3,
              f"acc {row['id']} mmlu choices")
        check(isinstance(row["answer"], str) and row["answer"] in "ABCDEFGHIJ"[: len(row["choices"])],
              f"acc {row['id']} mmlu answer out of range")
    for row in by_scenario["multilingual"]:
        check(isinstance(row["answer"], str) and row["answer"] in "ABCD",
              f"acc {row['id']} gmmu answer")
        check(len(row["choices"]) == 4, f"acc {row['id']} gmmu choices")
        check(row["meta"].get("is_en_reference") == (row["meta"].get("lang") == "en"),
              f"acc {row['id']} is_en_reference")
    check({row["meta"]["lang"] for row in by_scenario["multilingual"]}
          == set(CFG.ACC_SOURCES["multilingual"]["languages"]), "acc multilingual languages")
    per_language = Counter(row["meta"]["lang"] for row in by_scenario["multilingual"])
    check(set(per_language.values()) == {CFG.ACC_SOURCES["multilingual"]["per_language"]},
          f"acc multilingual per-language counts {per_language}")
    check(sum(1 for row in by_scenario["multilingual"] if row["meta"].get("is_en_reference"))
          == CFG.ACC_SOURCES["multilingual"]["per_language"], "acc multilingual en reference count")
    paired = defaultdict(set)
    for row in by_scenario["multilingual"]:
        paired[row["meta"]["lang"]].add(row["meta"]["sample_id"])
    first_ids = next(iter(paired.values()))
    check(all(ids == first_ids for ids in paired.values()), "acc multilingual paired sample_ids")

    for row in by_scenario["instruction_following"]:
        check(row["meta"].get("ifbench_key") is not None, f"acc {row['id']} missing ifbench key")
        check(isinstance(row["meta"].get("instruction_id_list"), list), f"acc {row['id']} ids")
        check(isinstance(row["meta"].get("kwargs"), list), f"acc {row['id']} kwargs")

    for row in by_scenario["coding"]:
        tests = row["meta"].get("tests")
        check(isinstance(tests, list) and bool(tests), f"acc {row['id']} hidden tests")
        check(all(len(json.dumps(test, ensure_ascii=False)) <= CFG.ACC_MAX_TEST_BYTES for test in tests),
              f"acc {row['id']} oversized hidden test")
        total = row["meta"].get("tests_total")
        dropped = row["meta"].get("tests_dropped_large")
        check(isinstance(total, int) and isinstance(dropped, int) and total == len(tests) + dropped,
              f"acc {row['id']} test accounting")
        check(isinstance(row["meta"].get("public_tests"), list), f"acc {row['id']} public tests")
        testtype = row["meta"].get("testtype")
        has_functional = any(test.get("testtype") == "functional" for test in tests)
        check(testtype in ("stdin", "functional", "mixed"), f"acc {row['id']} testtype")
        if has_functional:
            check(testtype in ("functional", "mixed"), f"acc {row['id']} functional testtype")
            check(bool(row["meta"].get("fn_name")), f"acc {row['id']} functional row missing fn_name")
            check("starter code" in last_user_text(row), f"acc {row['id']} missing starter prompt")
        else:
            check(testtype == "stdin", f"acc {row['id']} stdin testtype")
    testtypes = Counter(row["meta"]["testtype"] for row in by_scenario["coding"])
    check(testtypes == Counter({"stdin": 150, "functional": 100}),
          f"acc coding testtypes {testtypes}")

    bfcl_categories = CFG.ACC_SOURCES["general_agent"]["categories"]
    for row in by_scenario["general_agent"]:
        check(isinstance(row["meta"].get("possible_answers"), list),
              f"acc {row['id']} possible answers")
        check(row["meta"].get("category") in bfcl_categories, f"acc {row['id']} category")
        check(row["meta"].get("category_type") == "single_turn", f"acc {row['id']} category_type")
        if row["meta"].get("category") == "irrelevance":
            check(row["meta"].get("possible_answers") == [], f"acc {row['id']} irrelevance gold")
        else:
            check(bool(row["meta"].get("possible_answers")), f"acc {row['id']} empty possible_answers")
    categories = Counter(row["meta"]["category"] for row in by_scenario["general_agent"])
    expected_categories = CFG.ACC_SOURCES["general_agent"]["category_counts"]
    check(categories == Counter(expected_categories),
          f"acc bfcl categories {categories} != {expected_categories}")
    check(len({last_user_text(row) for row in by_scenario["general_agent"]})
          == len(by_scenario["general_agent"]), "acc bfcl duplicate user questions")

    for row in by_scenario["search_agent"]:
        check(len(row["meta"]["paragraphs"]) == 10, f"acc {row['id']} paragraphs")
        check(len(row["meta"]["supporting_facts"]) == 2, f"acc {row['id']} supporting facts")
        check(row["tools"] and [tool["function"]["name"] for tool in row["tools"]] == ["search", "lookup"],
              f"acc {row['id']} search tools")
        check(isinstance(row["meta"].get("corpus_tokens"), int), f"acc {row['id']} corpus_tokens")
    types = Counter(row["meta"]["type"] for row in by_scenario["search_agent"])
    check(types == Counter({"bridge": 194, "comparison": 56}), f"acc hotpot types {types}")

    bands = CFG.ACC_SOURCES["long_context"]["length_bands"]
    band_counts = Counter()
    for row in by_scenario["long_context"]:
        check(isinstance(row["meta"].get("answers"), list) and row["meta"]["answers"],
              f"acc {row['id']} answers")
        check(CFG.ACC_MIN_TOKENS <= row["tokens"] <= CFG.ACC_MAX_TOKENS,
              f"acc {row['id']} tokens outside {CFG.ACC_MIN_TOKENS}-{CFG.ACC_MAX_TOKENS}")
        task = row["meta"]["task"]
        template = CFG.LONG_CONTEXT_TEMPLATES[task]
        head, tail = template.split("{context}")
        body = last_user_text(row)
        check(body.startswith(head), f"acc {row['id']} not official template head ({task})")
        check(tail.split("{input}")[1] in body, f"acc {row['id']} not official template tail ({task})")
        band = row["meta"].get("length_band")
        check(band in bands, f"acc {row['id']} bad length_band {band}")
        band_counts[band] += 1
    check(band_counts == Counter(bands), f"acc longctx band counts {band_counts} != {bands}")
    check({row["meta"]["task"] for row in by_scenario["long_context"]}
          == set(CFG.ACC_SOURCES["long_context"]["tasks"]), "acc longctx tasks")

    # cross-scenario HotpotQA duplication must be zero
    lc_hot_ids = {
        row["meta"]["upstream_id"].split(":")[-1]
        for row in by_scenario["long_context"] if row["meta"]["task"] == "hotpotqa"
    }
    search_ids = {row["meta"]["upstream_id"].split(":")[-1] for row in by_scenario["search_agent"]}
    check(not (lc_hot_ids & search_ids),
          f"acc hotpot cross-scenario duplicates {sorted(lc_hot_ids & search_ids)[:5]}")

    # manifest stats parity
    for scenario, stats in manifest["stats"].items():
        scenario_rows = by_scenario[scenario]
        tokens = sorted(row["tokens"] for row in scenario_rows)
        check(
            (stats["n"], stats["tokens_min"], stats["tokens_median"], stats["tokens_max"])
            == (len(scenario_rows), tokens[0], tokens[len(tokens) // 2], tokens[-1]),
            f"acc manifest stats {scenario}",
        )
    check(manifest["totals"]["rows"] == expected_rows, "acc manifest totals rows")
    check(manifest["totals"]["scenarios"] == len(CFG.ACC_SOURCES), "acc manifest totals scenarios")
    check(manifest["totals"]["per_scenario"] == {name: spec["n"] for name, spec in CFG.ACC_SOURCES.items()},
          "acc manifest totals per_scenario")
    check(manifest["tokenizer"]["mode"] in ("local", "omlx"), "acc manifest tokenizer mode")
    check(set(manifest.get("strata_counts", {})) == set(CFG.ACC_SOURCES),
          "acc manifest strata_counts scenarios")
    check(set(CFG.ACC_SOURCES) <= set(manifest.get("sampling", {})),
          "acc manifest sampling scenarios")
    check(isinstance(manifest.get("deviations"), list) and manifest["deviations"],
          "acc manifest deviations")
    for judge_entry in manifest.get("judges", {}).values():
        check("implementation_status" in judge_entry, "acc manifest judge implementation_status")
    expected_provenance = {entry["dataset"]: entry for entry in CFG.provenance_for("accuracy")}
    check(set(manifest.get("provenance", {})) == set(expected_provenance),
          f"acc provenance datasets {sorted(manifest.get('provenance', {}))}")


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #
def verify_provenance() -> None:
    for set_name, manifest_path in (
        ("performance", C.TEST_PERF / "manifest.json"),
        ("accuracy", C.TEST_ACC / "manifest.json"),
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {entry["dataset"]: entry for entry in CFG.provenance_for(set_name)}
        provenance = manifest.get("provenance", {})
        check(set(provenance) == set(expected),
              f"{manifest_path.name} provenance datasets {sorted(provenance)} != {sorted(expected)}")
        for dataset, entry in expected.items():
            recorded = provenance.get(dataset, {})
            check(recorded.get("source_url") == entry["source_url"],
                  f"{manifest_path.name} {dataset} source_url")
            check(recorded.get("license") == entry["license"],
                  f"{manifest_path.name} {dataset} license")
            recorded_files = {item["logical_path"]: item for item in recorded.get("files", [])}
            expected_files = {logical for _relative, logical in entry["files"]}
            check(set(recorded_files) == expected_files,
                  f"{manifest_path.name} {dataset} files {sorted(recorded_files)}")
            for relative, logical in entry["files"]:
                path = C.RAW / relative
                check(path.is_file(), f"{manifest_path.name} raw missing {path}")
                if not path.is_file():
                    continue
                digest = (hashlib.sha256(path.read_bytes()).hexdigest()
                          if path.stat().st_size < 8 << 20 else C.sha256_file(path))
                item = recorded_files.get(logical, {})
                check(item.get("bytes") == path.stat().st_size,
                      f"{manifest_path.name} {logical} bytes")
                check(item.get("sha256") == digest, f"{manifest_path.name} {logical} sha256")


def main() -> int:
    verify_perf()
    verify_acc()
    verify_provenance()
    if FAILS:
        print(f"FAILED ({len(FAILS)} of {CHECKS} checks)")
        for message in FAILS[:40]:
            print("  -", message)
        return 1
    print(f"PASS: {CHECKS} checks across performance / accuracy / provenance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
