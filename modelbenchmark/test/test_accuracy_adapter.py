#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the accuracy test set adapter (``accuracy_jsonl``).

Most cases use a tiny synthetic jsonl so the contract is checked on a clean
clone; the real 2,000-row ``dataset/test/accuracy/testset.jsonl`` (gitignored) is
exercised too when present.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modelbench.adapters.datasets.accuracy_jsonl import (
    AccuracyJSONLAdapter,
    stratified_select,
)
from modelbench.adapters.datasets.registry import available_datasets, get_adapter
from modelbench.domain.models import AnswerType, Sample

TESTSET = Path("dataset/test/accuracy/testset.jsonl")
SCENARIOS = (
    "knowledge_qa",
    "instruction_following",
    "long_context",
    "math_logic",
    "coding",
    "general_agent",
    "search_agent",
    "multilingual",
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _row(
    sample_id: str,
    *,
    scenario: str = "knowledge_qa",
    dataset: str = "global_mmlu",
    language: str = "en",
    stratum: str = "subject=abstract_algebra|lang=en",
    judge: dict | None = None,
    input_tokens: int = 11,
    content: str | None = None,
) -> dict:
    return {
        "id": sample_id,
        "scenario": scenario,
        "dataset": dataset,
        "language": language,
        "upstream_id": f"{sample_id}|{language}",
        "input_tokens": input_tokens,
        "messages": [{"role": "user", "content": content or f"question {sample_id}"}],
        "judge": judge or {"family": "mc_letter", "answer": "B", "max_new_tokens": 1536},
        "meta": {"sample_id": sample_id, "stratum": stratum, "subject": "abstract_algebra"},
    }


def _write(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _adapter(tmp_path: Path, path: Path, params: dict) -> AccuracyJSONLAdapter:
    return AccuracyJSONLAdapter(
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        params=params,
        testset_dir=tmp_path / "test",
    )


@pytest.fixture
def synthetic(tmp_path: Path) -> tuple[Path, Path]:
    """10 rows in two scenarios, five strata of 2 rows inside knowledge_qa."""
    rows = []
    for index in range(10):
        stratum = f"subject=s{index // 2}|lang=en"
        rows.append(_row(f"knowledge_qa:{index}", stratum=stratum, input_tokens=100 + index))
    for index in range(3):
        rows.append(_row(f"math_logic:{index}", scenario="math_logic", dataset="gsm8k"))
    path = _write(tmp_path / "test" / "accuracy" / "testset.jsonl", rows)
    return path, tmp_path


# --------------------------------------------------------------------------- #
# loading + row contract
# --------------------------------------------------------------------------- #


def test_loads_only_the_requested_scenario_slice(synthetic):
    path, tmp_path = synthetic
    adapter = _adapter(tmp_path, path, {"scenario": "knowledge_qa"})
    samples = adapter.load()
    assert len(samples) == 10
    assert {sample.scenario for sample in samples} == {"knowledge_qa"}
    assert all(isinstance(sample, Sample) for sample in samples)


def test_sample_metadata_and_judge_contract(synthetic):
    path, tmp_path = synthetic
    adapter = _adapter(tmp_path, path, {"scenario": "knowledge_qa"})
    sample = adapter.load()[0]
    assert sample.sample_id == "knowledge_qa:0"
    assert sample.dataset == "global_mmlu"
    assert sample.scenario == "knowledge_qa"
    assert sample.question == "question knowledge_qa:0"
    assert sample.context is None and sample.subtask is None and sample.answer is None
    assert sample.answer_type == AnswerType.MC_LETTER
    assert sample.context_chars == len("question knowledge_qa:0")
    assert sample.est_context_tokens == 100
    metadata = sample.metadata
    assert metadata["prebuilt"] is True
    assert metadata["judge"] == {"family": "mc_letter", "answer": "B", "max_new_tokens": 1536}
    assert metadata["language"] == "en"
    assert metadata["upstream_id"] == "knowledge_qa:0|en"
    assert metadata["dataset_meta"]["stratum"] == "subject=s0|lang=en"
    assert metadata["messages"] == [{"role": "user", "content": "question knowledge_qa:0"}]
    assert metadata["input_tokens"] == 100


@pytest.mark.parametrize(
    "judge,expected",
    [
        ({"family": "mc_letter", "answer": "A"}, AnswerType.MC_LETTER),
        ({"family": "ifeval", "instruction_id_list": [], "kwargs": []}, AnswerType.CONSTRAINTS),
        ({"family": "normalized_match", "metric": "numeric_exact", "answers": ["2"]}, AnswerType.NUMERIC),
        ({"family": "normalized_match", "metric": "f1", "answers": ["x"]}, AnswerType.EXACT),
        ({"family": "code_exec", "test_code": "", "entry_point": "f"}, AnswerType.CODE),
        ({"family": "tool_call", "ground_truth": [], "functions": []}, AnswerType.TOOL_CALL),
    ],
)
def test_answer_type_follows_judge_family(tmp_path, judge, expected):
    path = _write(tmp_path / "test" / "accuracy" / "testset.jsonl", [_row("k:1", judge=judge)])
    adapter = _adapter(tmp_path, path, {"scenario": "knowledge_qa"})
    assert adapter.load()[0].answer_type == expected


# --------------------------------------------------------------------------- #
# selection + plan
# --------------------------------------------------------------------------- #


def test_plan_counts_scenario_slice_and_honours_limits(synthetic):
    path, tmp_path = synthetic
    adapter = _adapter(tmp_path, path, {"scenario": "knowledge_qa"})
    assert adapter.plan() == [
        {"scenario": "knowledge_qa", "dataset": "accuracy_jsonl", "n": 10}
    ]
    assert adapter.plan({"samples": 5}) == [
        {"scenario": "knowledge_qa", "dataset": "accuracy_jsonl", "n": 5}
    ]
    ids = [sample.sample_id for sample in adapter.load()[:2]]
    assert adapter.plan({"sample_ids": ids})[0]["n"] == 2
    # plan must match what load()+select_samples actually yields.
    selection = {"samples": 5}
    assert adapter.plan(selection)[0]["n"] == len(
        adapter.select_samples(adapter.load(), selection)
    )


def test_stratified_selection_is_deterministic_and_covers_strata(synthetic):
    path, tmp_path = synthetic
    adapter = _adapter(tmp_path, path, {"scenario": "knowledge_qa"})
    samples = adapter.load()
    selected = stratified_select(samples, 5)
    assert len(selected) == 5
    assert [sample.sample_id for sample in selected] == sorted(
        sample.sample_id for sample in selected
    )
    # exactly one sample from each of the five strata (no collapsing)
    strata = [sample.metadata["dataset_meta"]["stratum"] for sample in selected]
    assert len(set(strata)) == 5
    assert [sample.sample_id for sample in selected] == [
        sample.sample_id for sample in stratified_select(samples, 5)
    ]
    assert stratified_select(samples, None) == sorted(samples, key=lambda s: s.sample_id)


def test_explicit_sample_ids_win_and_invalid_ids_are_absent(synthetic):
    path, tmp_path = synthetic
    adapter = _adapter(tmp_path, path, {"scenario": "knowledge_qa"})
    samples = adapter.load()
    picked = stratified_select(
        samples, 9, ["knowledge_qa:1", "knowledge_qa:4", "missing:id"]
    )
    assert [sample.sample_id for sample in picked] == ["knowledge_qa:1", "knowledge_qa:4"]


def test_strata_reports_selected_counts(synthetic):
    path, tmp_path = synthetic
    adapter = _adapter(tmp_path, path, {"scenario": "knowledge_qa"})
    rows = adapter.strata({"samples": 5})
    assert rows is not None and len(rows) == 5
    assert sum(row["n"] for row in rows) == 5
    assert all(set(row) == {"stratum", "language", "dataset", "n"} for row in rows)


# --------------------------------------------------------------------------- #
# errors + registry
# --------------------------------------------------------------------------- #


def test_missing_file_raises_filenotfound_with_build_hint(tmp_path):
    adapter = _adapter(tmp_path, tmp_path / "nope" / "testset.jsonl", {"scenario": "knowledge_qa"})
    with pytest.raises(FileNotFoundError) as excinfo:
        adapter.load()
    assert "tool/dataset/accuracy/build.py" in str(excinfo.value)


def test_scenario_param_is_required(synthetic):
    path, tmp_path = synthetic
    adapter = _adapter(tmp_path, path, {})
    with pytest.raises(ValueError, match="params.scenario"):
        adapter.load()


def test_unknown_judge_family_is_rejected(tmp_path):
    judge = {"family": "llm_judge", "prompt": "grade it"}
    path = _write(tmp_path / "test" / "accuracy" / "testset.jsonl", [_row("k:1", judge=judge)])
    adapter = _adapter(tmp_path, path, {"scenario": "knowledge_qa"})
    with pytest.raises(ValueError, match="unknown judge family"):
        adapter.load()


def test_adapter_is_registered_and_instantiable(tmp_path):
    assert "accuracy_jsonl" in available_datasets()
    adapter = get_adapter(
        "accuracy_jsonl",
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        params={"scenario": "knowledge_qa"},
        testset_dir=tmp_path / "test",
    )
    assert isinstance(adapter, AccuracyJSONLAdapter)
    assert adapter.plan() is None  # absent file -> plan unknown, never a crash


# --------------------------------------------------------------------------- #
# the real 2,000-row asset (skipped on a clean clone)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not TESTSET.is_file(), reason="dataset/test/accuracy/testset.jsonl not built")
def test_real_accuracy_testset_has_2000_rows_across_8_scenarios():
    adapter = AccuracyJSONLAdapter(
        raw_dir=Path("dataset/raw"),
        processed_dir=Path("dataset/processed"),
        params={"scenario": "knowledge_qa"},
        testset_dir=Path("dataset/test"),
    )
    total = 0
    families = set()
    for scenario in SCENARIOS:
        adapter.params = {"scenario": scenario}
        plan = adapter.plan()
        assert plan is not None and plan[0]["scenario"] == scenario
        assert plan[0]["n"] == 250, (scenario, plan)
        total += plan[0]["n"]
        samples = adapter.load()
        assert len(samples) == 250
        assert {sample.scenario for sample in samples} == {scenario}
        assert all(sample.metadata["prebuilt"] for sample in samples)
        assert all(sample.metadata["judge"]["max_new_tokens"] >= 1 for sample in samples)
        families.update(sample.metadata["judge"]["family"] for sample in samples)
    assert total == 2000
    assert families == {
        "mc_letter",
        "ifeval",
        "normalized_match",
        "code_exec",
        "tool_call",
    }
