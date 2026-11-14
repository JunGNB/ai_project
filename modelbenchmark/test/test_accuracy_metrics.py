#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for accuracy aggregation, CIs and paired comparison.

Hand-written :class:`SampleRecord` fixtures keep this suite offline and fast:
every counting rule from ``docs/spec/22-accuracy-test-plan.md`` §4 is asserted
against records that could be read back from ``records.jsonl`` (AC-A5/A6).
"""

from __future__ import annotations

import math

import pytest

from modelbench.domain.accuracy_metrics import (
    accuracy_sample_results,
    aggregate_accuracy,
    holm_adjust,
    normalize_judge_score,
    normal_interval,
    paired_accuracy,
    wilson_interval,
)
from modelbench.domain.models import (
    Phase,
    RecordStatus,
    RequestSpec,
    SampleRecord,
    Score,
)


# --------------------------------------------------------------------------- #
# fixture helpers
# --------------------------------------------------------------------------- #


def _record(
    *,
    model: str = "bf16",
    scenario: str = "knowledge_qa",
    dataset: str = "global_mmlu",
    sid: str = "s1",
    status: RecordStatus = RecordStatus.OK,
    score: float | None = 1.0,
    family: str = "mc_letter",
    metric: str = "accuracy",
    language: str = "en",
    detail: dict | None = None,
    phase: Phase = Phase.MEASURED,
) -> SampleRecord:
    detail = dict(detail or {})
    detail.setdefault("family", family)
    detail.setdefault("metric", metric)
    score_obj = (
        None
        if score is None
        else Score(correct=bool(float(score) >= 1.0), score=float(score), detail=detail)
    )
    return SampleRecord(
        run_id="run-1",
        model=model,
        scenario=scenario,
        dataset=dataset,
        sample_id=sid,
        phase=phase,
        status=status,
        request=RequestSpec(
            model=model, messages=[{"role": "user", "content": "q"}], max_tokens=16
        ),
        score=score_obj,
        sample_meta={"language": language, "upstream_id": sid, "judge": {"family": family}},
    )


def _f1(sid: str, score: float, **kwargs) -> SampleRecord:
    return _record(
        sid=sid, score=score, family="normalized_match", metric="f1", **kwargs
    )


# --------------------------------------------------------------------------- #
# normalization
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "family,metric,value,expected",
    [
        ("mc_letter", "accuracy", 1.0, 1.0),
        ("mc_letter", "accuracy", 0.0, 0.0),
        ("mc_letter", "accuracy", 0.7, 1.0),      # 0/1 family: threshold at 0.5
        ("mc_letter", "accuracy", 0.4, 0.0),
        ("normalized_match", "f1", 0.62, 0.62),
        ("normalized_match", "rouge_zh", 1.5, 1.0),   # clamp
        ("normalized_match", "f1_zh", -0.2, 0.0),
        ("normalized_match", "numeric_exact", 1.0, 1.0),
        ("normalized_match", "numeric_exact", 0.9, 1.0),
        ("tool_call", "accuracy", None, 0.0),
        ("tool_call", "accuracy", "nan", 0.0),
        ("ifeval", "accuracy", float("nan"), 0.0),
    ],
)
def test_normalize_judge_score(family, metric, value, expected):
    assert normalize_judge_score(family, metric, value) == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# cells / overall counting rules
# --------------------------------------------------------------------------- #


def test_cell_counts_error_and_judge_error_are_explicit():
    records = [
        _record(sid="ok1", score=1.0),
        _record(sid="ok2", score=1.0),
        _record(sid="ok3", score=0.0),
        _record(sid="judge", status=RecordStatus.JUDGE_ERROR, score=0.0,
                detail={"judge_error": True}),
        _record(sid="err", status=RecordStatus.ERROR, score=None),
        _record(sid="to", status=RecordStatus.TIMEOUT, score=None),
    ]
    cells, overall = aggregate_accuracy(records)
    assert len(cells) == 1 and len(overall) == 1
    cell = cells[0]
    assert cell.model == "bf16" and cell.scenario == "knowledge_qa"
    assert cell.dataset == "global_mmlu" and cell.judge_family == "mc_letter"
    assert cell.n_samples == 6
    assert cell.n_ok == 3
    assert cell.n_error == 1 and cell.n_timeout == 1
    assert cell.n_judge_error == 1
    # judge_error counts as 0 AND stays in the denominator (never inflates).
    assert cell.n_valid == 4 and cell.n_scored == 4
    assert cell.n_correct == 2
    assert cell.mean_score == pytest.approx(0.5)
    assert cell.accuracy == pytest.approx(0.5)
    assert cell.score_ci95_low is not None and cell.score_ci95_high is not None
    row = overall[0]
    assert row.n_samples == 6
    assert row.macro_score == pytest.approx(0.5)
    assert row.micro_score == pytest.approx(0.5)
    assert row.judge_families == ["mc_letter"]


def test_ok_record_with_judge_error_detail_is_zero_and_counted():
    records = [
        _record(sid="a", score=1.0),
        _record(sid="b", score=1.0, detail={"judge_error": True}),
    ]
    cells, _ = aggregate_accuracy(records)
    cell = cells[0]
    assert cell.n_ok == 2
    assert cell.n_judge_error == 1
    assert cell.n_scored == 2
    assert cell.mean_score == pytest.approx(0.5)


def test_f1_slice_reports_mean_score_but_no_accuracy():
    records = [_f1("f1", 1.0), _f1("f2", 0.5), _f1("f3", 0.0)]
    cells, overall = aggregate_accuracy(records)
    cell = cells[0]
    assert cell.metric == "f1"
    assert cell.mean_score == pytest.approx(0.5)
    assert cell.n_correct == 1
    assert cell.accuracy is None
    assert cell.accuracy_ci95_low is None
    assert overall[0].macro_accuracy is None
    assert overall[0].macro_score == pytest.approx(0.5)


def test_macro_score_is_scenario_equal_weight_not_row_weight():
    # scenario A: 2 rows at 1.0; scenario B: 3 rows at 0.0.
    records = [_record(sid=f"a{i}", scenario="alpha", score=1.0) for i in range(2)]
    records += [_record(sid=f"b{i}", scenario="beta", score=0.0) for i in range(3)]
    _, overall = aggregate_accuracy(records)
    row = overall[0]
    assert row.scenario_scores == {"alpha": 1.0, "beta": 0.0}
    assert row.macro_score == pytest.approx(0.5)     # (1 + 0) / 2
    assert row.micro_score == pytest.approx(0.4)     # 2 / 5
    assert row.n_scenarios_scored == 2


def test_format_failure_and_language_profile_are_counted():
    records = [
        _record(sid="en", score=1.0, language="en"),
        _record(sid="zh", score=0.0, language="zh",
                detail={"format_failure": True}),
        _record(sid="zh2", score=1.0, language="zh"),
    ]
    cells, overall = aggregate_accuracy(records)
    cell = cells[0]
    assert cell.n_format_failure == 1
    assert set(cell.by_language) == {"en", "zh"}
    assert cell.by_language["en"].n_samples == 1
    assert cell.by_language["zh"].n_samples == 2
    assert cell.by_language["zh"].mean_score == pytest.approx(0.5)
    assert set(overall[0].by_language) == {"en", "zh"}


def test_warmup_and_non_measured_records_are_ignored():
    records = [
        _record(sid="measured", score=1.0),
        _record(sid="warm", score=1.0, phase=Phase.WARMUP),
    ]
    cells, overall = aggregate_accuracy(records)
    assert cells[0].n_samples == 1
    assert overall[0].n_samples == 1


def test_aggregates_are_recomputable_and_deterministic():
    records = [
        _record(sid="1", score=1.0),
        _record(sid="2", score=0.0, dataset="global_mmlu_alt"),
        _f1("3", 0.4),
    ]
    first = aggregate_accuracy(records)
    second = aggregate_accuracy(list(records))
    assert [c.model_dump() for c in first[0]] == [c.model_dump() for c in second[0]]
    assert [o.model_dump() for o in first[1]] == [o.model_dump() for o in second[1]]
    # cells are keyed by model x scenario x dataset (2 distinct datasets here).
    assert len(first[0]) == 2
    assert first[0][0].judge_family == "mixed"


# --------------------------------------------------------------------------- #
# CIs
# --------------------------------------------------------------------------- #


def test_wilson_interval_matches_reference_and_handles_small_n():
    low, high = wilson_interval(8, 10)
    assert low == pytest.approx(0.49016, abs=1e-4)
    assert high == pytest.approx(0.94332, abs=1e-4)
    assert wilson_interval(1, 1) is None       # n < 2 -> null per spec
    assert wilson_interval(0, 0) is None


def test_normal_interval_and_null_for_small_n():
    low, high = normal_interval([0.2, 0.4, 0.6])
    half = 1.96 * 0.2 / math.sqrt(3)
    assert low == pytest.approx(0.4 - half)
    assert high == pytest.approx(0.4 + half)
    assert normal_interval([0.5]) is None


# --------------------------------------------------------------------------- #
# per-sample table
# --------------------------------------------------------------------------- #


def test_accuracy_sample_results_cover_ok_error_and_judge_error():
    records = [
        _f1("ok", 0.8),
        _record(sid="err", status=RecordStatus.ERROR, score=None),
        _record(sid="je", status=RecordStatus.JUDGE_ERROR, score=0.0,
                detail={"judge_error": True, "format_failure": True}),
    ]
    rows = accuracy_sample_results(records)
    assert [row.sample_id for row in rows] == ["ok", "err", "je"]
    ok, err, je = rows
    assert ok.score == pytest.approx(0.8) and ok.correct is False
    assert ok.judge_family == "normalized_match" and ok.judge_metric == "f1"
    assert ok.language == "en" and ok.upstream_id == "ok"
    assert ok.max_new_tokens == 16 and ok.phase == "measured"
    assert err.status == "error" and err.score is None and err.correct is None
    assert je.judge_error is True and je.score == 0.0 and je.correct is False
    assert je.format_failure is True


# --------------------------------------------------------------------------- #
# paired comparison
# --------------------------------------------------------------------------- #


def _paired_runs():
    a: list[SampleRecord] = []
    b: list[SampleRecord] = []
    # knowledge_qa: A 5/6 correct, B 1/6 correct -> McNemar 4 vs 0.
    for index in range(1, 7):
        a.append(_record(sid=f"k{index}", score=1.0 if index <= 5 else 0.0))
        b.append(_record(sid=f"k{index}", score=1.0 if index == 2 else 0.0))
    # instruction_following: 2/4 each, one discordant pair in each direction.
    for index in range(1, 5):
        a.append(_record(sid=f"i{index}", scenario="instruction_following",
                         score=1.0 if index <= 2 else 0.0))
        b.append(_record(sid=f"i{index}", scenario="instruction_following",
                         score=1.0 if index in (1, 3) else 0.0))
    # long_context: F1, paired bootstrap CI.
    for sid, sa, sb in (("l1", 1.0, 0.5), ("l2", 0.5, 0.5), ("l3", 0.0, 0.0)):
        a.append(_f1(sid, sa, scenario="long_context", dataset="longbench_v1"))
        b.append(_f1(sid, sb, scenario="long_context", dataset="longbench_v1"))
    # unpaired (only in A) and unscored (B errored).
    a.append(_record(sid="solo", score=1.0))
    a.append(_record(sid="bad", scenario="instruction_following", score=1.0))
    b.append(_record(sid="bad", scenario="instruction_following", score=None,
                     status=RecordStatus.ERROR))
    return a, b


def test_paired_accuracy_reports_delta_mcnemar_and_unpaired():
    a, b = _paired_runs()
    result = paired_accuracy(a, b, bootstrap=200)
    assert result["pairing"] == "(scenario, sample_id)"
    by_scenario = {row["scenario"]: row for row in result["scenarios"]}
    assert set(by_scenario) == {"knowledge_qa", "instruction_following", "long_context"}

    knowledge = by_scenario["knowledge_qa"]
    assert knowledge["n"] == 6
    assert knowledge["delta"] == pytest.approx(5 / 6 - 1 / 6)
    assert knowledge["n_a_correct_b_wrong"] == 4
    assert knowledge["n_a_wrong_b_correct"] == 0
    assert knowledge["mcnemar_p"] == pytest.approx(2 * (1 / 2 ** 4))
    assert knowledge["binary"] is True

    instruction = by_scenario["instruction_following"]
    assert instruction["delta"] == pytest.approx(0.0)
    assert instruction["mcnemar_p"] == pytest.approx(1.0)

    long_context = by_scenario["long_context"]
    assert long_context["binary"] is False
    assert long_context["n"] == 3
    assert long_context["delta"] == pytest.approx(0.5 - 1 / 3)
    assert long_context["bootstrap_ci95_low_pp"] is not None
    assert long_context["bootstrap_ci95_high_pp"] >= long_context["bootstrap_ci95_low_pp"]

    assert result["n_paired"] == 13
    assert result["unpaired"] == [
        {"scenario": "knowledge_qa", "sample_id": "solo", "in_a": True, "in_b": False}
    ]
    assert result["unscored"] == [
        {
            "scenario": "instruction_following",
            "sample_id": "bad",
            "status_a": "ok",
            "status_b": "error",
        }
    ]
    assert not result["judge_family_mismatches"]
    # Holm-adjusted p-values are never smaller than the raw ones.
    for row in result["scenarios"]:
        if row.get("mcnemar_p") is not None:
            assert row["mcnemar_p_holm"] >= row["mcnemar_p"]


def test_paired_accuracy_is_deterministic_for_the_bootstrap():
    a, b = _paired_runs()
    first = paired_accuracy(a, b, bootstrap=200, seed=7)
    second = paired_accuracy(a, b, bootstrap=200, seed=7)
    assert first == second


def test_paired_accuracy_flags_judge_family_mismatch():
    a = [_record(sid="x", scenario="general_agent", family="tool_call")]
    b = [_record(sid="x", scenario="general_agent", family="mc_letter")]
    result = paired_accuracy(a, b)
    assert result["scenarios"] == []
    assert result["judge_family_mismatches"][0]["scenario"] == "general_agent"


def test_holm_adjust_step_down():
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": None})
    assert adjusted["a"] == pytest.approx(0.02)   # 2 * 0.01
    assert adjusted["b"] == pytest.approx(0.04)   # max(0.02, 1 * 0.04)
    assert adjusted["c"] is None
