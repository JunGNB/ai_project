#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gold-answer validation of the five judge families.

For every judge family we feed the *known-correct* reference answer and require
a pass, plus at least one negative control where meaningful. This does not test
the model; it tests that the grader itself is wired correctly and deterministic.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tool.system import judges  # noqa: E402

TESTSET = ROOT / "dataset" / "test" / "accuracy" / "testset.jsonl"
RAW = ROOT / "dataset" / "raw"


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def load_testset():
    """Return {scenario: [rows]} from the single combined testset.jsonl."""
    by_scenario = {}
    for row in read_jsonl(TESTSET):
        by_scenario.setdefault(row["scenario"], []).append(row)
    return by_scenario


def load_humaneval():
    import pyarrow.parquet as pq

    t = pq.read_table(RAW / "humaneval/openai_humaneval/test-00000-of-00001.parquet").to_pydict()
    return {t["task_id"][i]: {"prompt": t["prompt"][i], "code": t["canonical_solution"][i]} for i in range(len(t["task_id"]))}


def load_mbpp():
    rows = json.load(open(RAW / "mbpp/sanitized-mbpp.json", encoding="utf-8"))
    return {f"task_id={r['task_id']}": r["code"] for r in rows}


def main():
    global SCENARIOS
    SCENARIOS = load_testset()
    fmt = "%-22s %-8s %s"
    failures = 0
    print(fmt % ("scenario", "n", "check"))

    # mc_letter: gold passes, a wrong letter fails
    for scen in ("knowledge_qa", "multilingual"):
        rows = SCENARIOS[scen]
        bad = 0
        for r in rows:
            gold = r["judge"]["answer"]
            pos = judges.evaluate(r, f"The answer is {gold}.")
            wrong = "A" if gold != "A" else "B"
            neg = judges.evaluate(r, f"The answer is {wrong}.")
            if not pos["pass"] or neg["pass"]:
                bad += 1
        failures += bad
        print(fmt % (scen, len(rows), f"gold_pass/wrong_fail bad={bad}"))

    # normalized_match: f1 gold -> 1.0 ; numeric gold -> pass
    for scen in ("long_context", "search_agent"):
        rows = SCENARIOS[scen]
        bad = 0
        for r in rows:
            res = judges.evaluate(r, str(r["judge"]["answers"][0]))
            if res["score"] < 0.999:
                bad += 1
        failures += bad
        print(fmt % (scen, len(rows), f"gold_f1=1.0 bad={bad}"))

    rows = SCENARIOS["math_logic"]
    bad = 0
    for r in rows:
        res = judges.evaluate(r, "推理过程略。\n#### " + r["judge"]["answers"][0])
        if not res["pass"]:
            bad += 1
    failures += bad
    print(fmt % ("math_logic", len(rows), f"gold_numeric_pass bad={bad}"))

    # code_exec: canonical solutions must pass their own tests
    he = load_humaneval()
    mb = load_mbpp()
    rows = SCENARIOS["coding"]
    bad = 0
    for r in rows:
        if r["dataset"] == "humaneval":
            code = he[r["upstream_id"]]["code"]
        else:
            code = mb[r["upstream_id"]]
        res = judges.evaluate(r, "```python\n" + code + "\n```")
        if not res["pass"]:
            bad += 1
            if bad <= 3:
                print("  coding gold failed:", r["id"], res["error"][:200])
    failures += bad
    print(fmt % ("coding", len(rows), f"canonical_pass bad={bad}"))

    # tool_call: ground truth itself must match; irrelevance must accept []
    rows = SCENARIOS["general_agent"]
    bad = 0
    for r in rows:
        gt = r["judge"]["ground_truth"]
        if gt:
            calls = []
            for call in gt:
                (name, args), = call.items()
                picked = {}
                for arg, acceptable in args.items():
                    for acc in acceptable:
                        if acc != "":
                            picked[arg] = acc
                            break
                    else:
                        picked[arg] = acceptable[0]
                calls.append({"name": name, "arguments": picked})
            resp = json.dumps(calls, ensure_ascii=False)
        else:
            resp = "[]"
        res = judges.evaluate(r, resp)
        if not res["pass"]:
            bad += 1
            if bad <= 3:
                print("  bfcl gold failed:", r["id"], res.get("reason"))
    failures += bad
    print(fmt % ("general_agent", len(rows), f"ground_truth_pass bad={bad}"))

    # ifeval: every selected instruction id is known; empty response fails all
    rows = SCENARIOS["instruction_following"]
    ev = judges._load_ifeval()
    registry = ev.instructions_registry.INSTRUCTION_DICT
    unknown = sorted({i for r in rows for i in r["judge"]["instruction_id_list"] if i not in registry})
    bad = sum(1 for r in rows if judges.evaluate(r, "")["pass"])
    failures += len(unknown) + bad
    print(fmt % ("instruction_following", len(rows), f"unknown_ids={len(unknown)} empty_pass={bad}"))

    # negative controls: the checks above only prove "gold is not rejected"; they
    # cannot catch false positives.  This block is the regression net for the P0
    # defects found in the 2026-11 audit (see test/test_judges.py for unit-level
    # coverage).
    neg = 0
    mc = SCENARIOS["knowledge_qa"][0]
    gold = mc["judge"]["answer"]
    wrong = "A" if gold != "A" else "B"

    def _expect(condition: bool) -> None:
        nonlocal neg
        if not condition:
            neg += 1

    _expect(judges.evaluate(mc, f"答案是 {gold}。")["pass"] is True)
    _expect(judges.evaluate(mc, f"答案：{gold.lower()}")["pass"] is True)
    _expect(judges.evaluate(mc, "A. first\nB. second\nC. third\nD. fourth\n答案：" + gold)["pass"] is True)
    _expect(judges.evaluate(mc, f"The answer is {wrong}.")["pass"] is False)
    _expect(judges.evaluate(mc, "Cannot determine from the given options.")["pred"] is None)
    ga_irr = next(r for r in SCENARIOS["general_agent"] if r["judge"]["ground_truth"] == [])
    _expect(judges.evaluate(ga_irr, "I cannot answer that with the given functions.")["pass"] is True)
    _expect(judges.parse_calls('{"name": "f", "arguments": {"xs": [1, 2]}}') is not None)
    code_row = next(r for r in SCENARIOS["coding"] if r["dataset"] == "humaneval")
    _expect(judges.evaluate(code_row, "```python\ndef nope():\n    return 0\n```")["pass"] is False)
    _expect(judges.evaluate(code_row, "```python\nimport sys\nsys.exit(0)\n```")["pass"] is False)
    math_row = SCENARIOS["math_logic"][0]
    wrong_number = int(float(math_row["judge"]["answers"][0])) + 1
    _expect(judges.evaluate(math_row, "#### " + str(wrong_number))["pass"] is False)
    failures += neg
    print(fmt % ("negative controls", 10, f"false_positive_or_negative={neg}"))

    print()
    print("RESULT:", "PASS" if failures == 0 else f"FAIL ({failures} problems)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
