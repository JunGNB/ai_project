#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the accuracy side of the analysis surface (WP-5).

These fixtures are hand-written run bundles (``run.json`` + ``records.jsonl`` +
``aggregates.json`` + ``batches/<id>.json``) built directly from
:class:`~modelbench.domain.models.SampleRecord`; nothing here depends on the
accuracy runner/writer (WP-2/WP-3), so the suite stays offline and fast while
still exercising the real ``catalog.load_records`` readback path.

Covered contract:

* ``load_run_method`` — run.json/aggregates precedence, legacy default;
* ``accuracy_paired_comparison`` — Δ direction (A − B), exact McNemar p,
  Holm adjustment, explicit unpaired/unscored lists, method/testset/judge-family
  failures;
* ``render_accuracy_paired`` — table + verdict rules (noise baseline);
* ``render_accuracy_table`` — scenario cells, macro/micro overall rows, header
  (testset SHA / method / cold protocol);
* analysis CLI — accuracy table/paired/JSON, ``--method``, exit codes, and the
  unchanged performance path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pytest

from modelbench.adapters.storage import catalog, layout
from modelbench.domain.accuracy_metrics import aggregate_accuracy
from modelbench.domain.models import (
    Phase,
    RecordStatus,
    RequestSpec,
    SampleRecord,
    Score,
)
from modelbench.interfaces.analysis import cli as analysis_cli
from modelbench.interfaces.analysis.compare import (
    accuracy_paired_comparison,
    load_run_method,
    render_accuracy_paired,
    render_accuracy_table,
)

TEST_SHA = "a" * 64
OTHER_SHA = "b" * 64


# --------------------------------------------------------------------------- #
# fixture helpers: minimal schema-3.0 accuracy bundles
# --------------------------------------------------------------------------- #


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sample(
    *,
    run_id: str,
    model: str,
    scenario: str,
    sample_id: str,
    score: Optional[float],
    dataset: str = "global_mmlu",
    family: str = "mc_letter",
    metric: str = "accuracy",
    status: RecordStatus = RecordStatus.OK,
    language: str = "en",
) -> SampleRecord:
    detail = {"family": family, "metric": metric}
    score_obj = (
        None
        if score is None
        else Score(
            correct=float(score) >= 1.0,
            score=float(score),
            detail=dict(detail),
        )
    )
    return SampleRecord(
        run_id=run_id,
        model=model,
        scenario=scenario,
        dataset=dataset,
        sample_id=sample_id,
        phase=Phase.MEASURED,
        status=status,
        request=RequestSpec(
            model=model,
            messages=[{"role": "user", "content": "question"}],
            max_tokens=16,
            temperature=0.0,
            top_p=1.0,
            prompt_sha256=TEST_SHA,
        ),
        score=score_obj,
        sample_meta={
            "language": language,
            "upstream_id": sample_id,
            "judge": {"family": family, "metric": metric},
        },
    )


def _write_records(run_dir: Path, records: Iterable[SampleRecord]) -> None:
    # Mirror ``RunWriter.write_record``: the prompt body is dropped from the
    # persisted record (the prompt lives in the blob pool by sha256).
    lines = [
        json.dumps(
            record.model_dump(
                mode="json", exclude={"request": {"messages", "tools"}}
            ),
            ensure_ascii=False,
        )
        for record in records
    ]
    (run_dir / layout.RECORDS).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_accuracy_aggregates(
    run_dir: Path, run_id: str, records: List[SampleRecord]
) -> Dict[str, Any]:
    """Aggregate exactly like a real bundle would (WP-1 domain function)."""
    cells, overall = aggregate_accuracy(records)
    payload = {
        "schema_version": "3.0",
        "method": "accuracy",
        "run_id": run_id,
        "generated_at": "2026-11-11T00:00:00+00:00",
        "n_records": len(records),
        "n_samples": len(records),
        "n_cells": len(cells),
        "overall": [entry.model_dump(mode="json") for entry in overall],
        "cells": [cell.model_dump(mode="json") for cell in cells],
    }
    _write_json(run_dir / layout.AGGREGATES_JSON, payload)
    return payload


def _make_run(
    output_root: Path,
    run_id: str,
    *,
    method: str = "accuracy",
    batch_id: str = "batch-1",
    records: Optional[List[SampleRecord]] = None,
    aggregates: bool = True,
    protocol: Optional[Dict[str, Any]] = None,
) -> Path:
    run_dir = layout.run_dir(output_root, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = {"repeats": 1, "cache_mode": "cold"}
    run_config.update(protocol or {})
    _write_json(
        run_dir / layout.RUN_JSON,
        {
            "schema_version": "3.0",
            "run_id": run_id,
            "batch_id": batch_id,
            "method": method,
            "created_at": "2026-11-11T00:00:00+00:00",
            "config": {"run": run_config},
        },
    )
    if records is not None:
        _write_records(run_dir, records)
    if aggregates:
        _write_accuracy_aggregates(run_dir, run_id, list(records or []))
    return run_dir


def _write_batch(
    output_root: Path, batch_id: str = "batch-1", *, sha256: str = TEST_SHA
) -> Path:
    path = layout.batch_path(output_root, batch_id)
    _write_json(
        path,
        {
            "schema_version": "3.0",
            "batch_id": batch_id,
            "method": "accuracy",
            "inputs": {"testset": {"path": "dataset/test/accuracy/testset.jsonl", "sha256": sha256}},
        },
    )
    return path


def _paired_runs(output_root: Path, *, extras: bool):
    """Two accuracy runs: 6 discordant (p=0.03125) + 2 discordant (p=0.5).

    ``extras=True`` adds one sample only in A (unpaired) and one sample errored
    in B (unscored), so the explicit lists are non-empty; ``extras=False`` is a
    clean pair whose sample sets match exactly.
    """
    run_a_id = "batch-1__accuracy__bf16"
    run_b_id = "batch-1__accuracy__quant"
    records_a: List[SampleRecord] = []
    records_b: List[SampleRecord] = []
    for index in range(6):
        sid = f"k{index}"
        records_a.append(
            _sample(run_id=run_a_id, model="bf16", scenario="knowledge_qa",
                    sample_id=sid, score=1.0)
        )
        records_b.append(
            _sample(run_id=run_b_id, model="quant", scenario="knowledge_qa",
                    sample_id=sid, score=0.0)
        )
    for index in range(2):
        sid = f"m{index}"
        records_a.append(
            _sample(run_id=run_a_id, model="bf16", scenario="math",
                    sample_id=sid, score=1.0, dataset="gsm8k")
        )
        records_b.append(
            _sample(run_id=run_b_id, model="quant", scenario="math",
                    sample_id=sid, score=0.0, dataset="gsm8k")
        )
    if extras:
        # Only in A.
        records_a.append(
            _sample(run_id=run_a_id, model="bf16", scenario="knowledge_qa",
                    sample_id="k_only_a", score=1.0)
        )
        # In both, but B errored -> unscored.
        records_a.append(
            _sample(run_id=run_a_id, model="bf16", scenario="knowledge_qa",
                    sample_id="k_unscored", score=1.0)
        )
        records_b.append(
            _sample(run_id=run_b_id, model="quant", scenario="knowledge_qa",
                    sample_id="k_unscored", score=None, status=RecordStatus.ERROR)
        )
    _write_batch(output_root)
    run_a = _make_run(output_root, run_a_id, records=records_a)
    run_b = _make_run(output_root, run_b_id, records=records_b)
    return run_a, run_b, records_a, records_b


def _clean_paired_runs(output_root: Path):
    run_a, run_b, _records_a, _records_b = _paired_runs(output_root, extras=False)
    return run_a, run_b


def _paired_fixture(output_root: Path):
    return _paired_runs(output_root, extras=True)


# --------------------------------------------------------------------------- #
# load_run_method
# --------------------------------------------------------------------------- #


def test_load_run_method_reads_run_json_and_defaults_to_performance(tmp_path):
    accuracy_dir = _make_run(tmp_path, "b1__accuracy__m", records=[])
    performance_dir = _make_run(
        tmp_path, "b1__performance__m", method="performance", records=[]
    )
    legacy_dir = layout.run_dir(tmp_path, "b1__performance__legacy")
    legacy_dir.mkdir(parents=True)

    assert load_run_method(accuracy_dir) == "accuracy"
    assert load_run_method(accuracy_dir / layout.RUN_JSON) == "accuracy"
    assert load_run_method(performance_dir) == "performance"
    # No run.json: an aggregates document is the fallback, and its method wins.
    assert load_run_method(legacy_dir) == "performance"
    # Loaded documents are accepted directly.
    assert load_run_method({"method": "accuracy"}) == "accuracy"
    assert load_run_method({"schema_version": "3.0"}) == "performance"
    # Unknown method values are legacy/unknown -> performance (never guessed).
    assert load_run_method({"method": "mystery"}) == "performance"


def test_load_run_method_uses_aggregates_when_run_json_lacks_method(tmp_path):
    run_dir = _make_run(tmp_path, "b1__accuracy__aggonly", records=[], method="")
    aggregates = json.loads(
        (run_dir / layout.AGGREGATES_JSON).read_text(encoding="utf-8")
    )
    assert aggregates["method"] == "accuracy"
    run_json = json.loads((run_dir / layout.RUN_JSON).read_text(encoding="utf-8"))
    run_json.pop("method")
    _write_json(run_dir / layout.RUN_JSON, run_json)
    assert load_run_method(run_dir) == "accuracy"
    # run.json itself is authoritative when it does carry a (different) method.
    run_json["method"] = "performance"
    _write_json(run_dir / layout.RUN_JSON, run_json)
    assert load_run_method(run_dir) == "performance"


# --------------------------------------------------------------------------- #
# paired accuracy: Δ / McNemar / Holm / explicit lists
# --------------------------------------------------------------------------- #


def test_accuracy_paired_comparison_delta_mcnemar_holm(tmp_path):
    run_a, run_b, _records_a, _records_b = _paired_fixture(tmp_path)

    result = accuracy_paired_comparison(run_a, run_b)

    # The sample sets differ (one key only in A), so the comparison explicitly
    # reports incomparability; the numbers and lists are still available.
    assert result["ok"] is False
    assert "sample sets differ" in result["reason"]
    assert result["method"] == "accuracy"
    assert result["run_a"] == str(run_a)
    assert result["run_b"] == str(run_b)
    assert result["testset_sha256"] == TEST_SHA
    assert result["testset_sha256_a"] == TEST_SHA
    assert result["testset_sha256_b"] == TEST_SHA
    assert "A is better" in result["note"] or "A" in result["note"]
    assert result["pairing"] == "(scenario, sample_id)"
    assert result["n_paired"] == 8
    assert result["n_scenarios"] == 2

    scenarios = {row["scenario"]: row for row in result["scenarios"]}
    knowledge = scenarios["knowledge_qa"]
    assert knowledge["n"] == 6
    assert knowledge["mean_a"] == pytest.approx(1.0)
    assert knowledge["mean_b"] == pytest.approx(0.0)
    assert knowledge["delta_pp"] == pytest.approx(100.0)
    assert knowledge["n_a_correct_b_wrong"] == 6
    assert knowledge["n_a_wrong_b_correct"] == 0
    # Exact McNemar: 2 * (1/2)^6 = 0.03125.
    assert knowledge["mcnemar_p"] == pytest.approx(0.03125)
    # Holm across two scenarios: 2 * 0.03125 = 0.0625 -> not significant.
    assert result["alpha"] == pytest.approx(0.05)
    assert knowledge["mcnemar_p_holm"] == pytest.approx(0.0625)
    assert knowledge["significant_holm"] is False

    math_row = scenarios["math"]
    assert math_row["n"] == 2
    # 2 * (1/2)^2 = 0.5; Holm step-down keeps 0.5 for the largest p-value.
    assert math_row["mcnemar_p"] == pytest.approx(0.5)
    assert math_row["mcnemar_p_holm"] == pytest.approx(0.5)

    assert result["macro_delta_pp"] == pytest.approx(100.0)

    # Samples/keys that cannot be paired are explicit, never silently dropped.
    assert len(result["unpaired"]) == 1
    assert result["unpaired"][0]["sample_id"] == "k_only_a"
    assert result["unpaired"][0]["in_a"] is True
    assert result["unpaired"][0]["in_b"] is False
    assert len(result["unscored"]) == 1
    assert result["unscored"][0]["sample_id"] == "k_unscored"
    assert result["unscored"][0]["status_b"] == "error"
    assert len(result["explicit"]["unpaired"]) == 1
    assert len(result["explicit"]["unscored"]) == 1
    assert result["explicit"]["judge_family_mismatches"] == []
    assert result["duplicates_a"] == [] and result["duplicates_b"] == []
    assert any("unscored" in warning for warning in result["warnings"])


def test_render_accuracy_paired_table_and_explicit_counts(tmp_path):
    run_a, run_b, _ra, _rb = _paired_fixture(tmp_path)
    result = accuracy_paired_comparison(run_a, run_b)

    text = render_accuracy_paired(result)

    assert "精度配对比较" in text
    assert "knowledge_qa" in text and "math" in text
    assert "p(McNemar)" in text and "p(Holm)" in text and "Δpp" in text
    assert "0.03125" in text      # exact McNemar p, 0/1 scenario
    assert "0.0625" in text       # Holm-adjusted p
    assert "0.5" in text          # math scenario p
    assert "+100.00" in text      # Δpp(A - B)
    assert "不显著" in text        # p_holm 0.0625 >= 0.05
    assert f"testset SHA-256: {TEST_SHA}" in text
    assert "unpaired: 1" in text
    assert "unscored: 1" in text
    assert "judge_family_mismatches: 0" in text
    assert "配对样本数: 8" in text
    assert "结果不可直接比较" in text
    assert "sample sets differ" in text


# --------------------------------------------------------------------------- #
# paired accuracy: incomparable inputs must fail loudly
# --------------------------------------------------------------------------- #


def test_accuracy_paired_rejects_performance_runs(tmp_path):
    performance_run = _make_run(
        tmp_path, "b1__performance__m", method="performance", records=[], aggregates=False
    )
    accuracy_run = _make_run(tmp_path, "b1__accuracy__m", records=[])

    with pytest.raises(ValueError, match="cannot compare performance and accuracy runs"):
        accuracy_paired_comparison(performance_run, accuracy_run)
    with pytest.raises(ValueError, match="cannot compare performance and accuracy runs"):
        accuracy_paired_comparison(performance_run, performance_run)


def test_accuracy_paired_flags_judge_family_mismatch(tmp_path):
    run_a_id = "batch-1__accuracy__bf16"
    run_b_id = "batch-1__accuracy__quant"
    records_a = [
        _sample(run_id=run_a_id, model="bf16", scenario="knowledge_qa",
                sample_id="s1", score=1.0, family="mc_letter", metric="accuracy")
    ]
    records_b = [
        _sample(run_id=run_b_id, model="quant", scenario="knowledge_qa",
                sample_id="s1", score=1.0, family="ifeval", metric="accuracy")
    ]
    _write_batch(tmp_path)
    run_a = _make_run(tmp_path, run_a_id, records=records_a)
    run_b = _make_run(tmp_path, run_b_id, records=records_b)

    result = accuracy_paired_comparison(run_a, run_b)

    assert result["ok"] is False
    assert "judge family/metric mismatch" in result["reason"]
    assert len(result["explicit"]["judge_family_mismatches"]) == 1
    assert result["scenarios"] == []
    text = render_accuracy_paired(result)
    assert "结果不可直接比较" in text
    assert "judge_family_mismatches: 1" in text


def test_accuracy_paired_flags_testset_sha_mismatch(tmp_path):
    run_a_id = "batch-a__accuracy__bf16"
    run_b_id = "batch-b__accuracy__quant"
    records_a = [
        _sample(run_id=run_a_id, model="bf16", scenario="knowledge_qa",
                sample_id="s1", score=1.0)
    ]
    records_b = [
        _sample(run_id=run_b_id, model="quant", scenario="knowledge_qa",
                sample_id="s1", score=0.0)
    ]
    _write_batch(tmp_path, "batch-a", sha256=TEST_SHA)
    _write_batch(tmp_path, "batch-b", sha256=OTHER_SHA)
    run_a = _make_run(tmp_path, run_a_id, batch_id="batch-a", records=records_a)
    run_b = _make_run(tmp_path, run_b_id, batch_id="batch-b", records=records_b)

    result = accuracy_paired_comparison(run_a, run_b)

    assert result["ok"] is False
    assert "testset SHA-256 differs" in result["reason"]
    assert result["testset_sha256"] is None
    assert result["testset_sha256_a"] == TEST_SHA
    assert result["testset_sha256_b"] == OTHER_SHA


def test_accuracy_paired_missing_records_raises_file_not_found(tmp_path):
    run_a = _make_run(tmp_path, "b1__accuracy__a", records=[], aggregates=True)
    run_b = _make_run(tmp_path, "b1__accuracy__b", records=[], aggregates=True)
    (run_a / layout.RECORDS).unlink(missing_ok=True)
    with pytest.raises(FileNotFoundError, match="records.jsonl"):
        accuracy_paired_comparison(run_a, run_b)


# --------------------------------------------------------------------------- #
# paired accuracy: verdict rules and F1 bootstrap CI
# --------------------------------------------------------------------------- #


def _discordant_run_dirs(
    output_root: Path, *, a_score: float, b_score: float, count: int = 6
):
    run_a_id = "batch-1__accuracy__bf16"
    run_b_id = "batch-1__accuracy__quant"
    records_a = [
        _sample(run_id=run_a_id, model="bf16", scenario="knowledge_qa",
                sample_id=f"s{index}", score=a_score)
        for index in range(count)
    ]
    records_b = [
        _sample(run_id=run_b_id, model="quant", scenario="knowledge_qa",
                sample_id=f"s{index}", score=b_score)
        for index in range(count)
    ]
    _write_batch(output_root)
    run_a = _make_run(output_root, run_a_id, records=records_a)
    run_b = _make_run(output_root, run_b_id, records=records_b)
    return run_a, run_b


def _discordant_runs(output_root: Path, *, a_score: float, b_score: float, count: int = 6):
    run_a, run_b = _discordant_run_dirs(
        output_root, a_score=a_score, b_score=b_score, count=count
    )
    return accuracy_paired_comparison(run_a, run_b)


def test_render_accuracy_paired_verdict_respects_noise_baseline(tmp_path):
    # A correct / B wrong six times: Holm p = 0.03125 (< alpha), Δpp = +100.
    result = _discordant_runs(tmp_path, a_score=1.0, b_score=0.0)
    row = result["scenarios"][0]
    assert row["mcnemar_p_holm"] == pytest.approx(0.03125)
    assert row["significant_holm"] is True

    default_text = render_accuracy_paired(result)
    assert "显著（A 更好）" in default_text
    # Baseline 0 reports significance only: it must not claim a regression.
    assert "显著退化" not in default_text
    assert "显著提升" not in default_text

    # |Δpp| = 100 > 50 -> an improvement label is allowed.
    improved = render_accuracy_paired(result, noise_baseline_pp=50.0)
    assert "显著提升" in improved
    # |Δpp| = 100 <= 200 -> significant but below the noise baseline.
    below = render_accuracy_paired(result, noise_baseline_pp=200.0)
    assert "显著但未超过噪声基线" in below


def test_render_accuracy_paired_reports_regression_direction(tmp_path):
    result = _discordant_runs(tmp_path, a_score=0.0, b_score=1.0)
    text = render_accuracy_paired(result, noise_baseline_pp=50.0)
    assert "显著退化" in text
    assert "-100.00" in text
    default_text = render_accuracy_paired(result)
    assert "显著（B 更好）" in default_text


def test_accuracy_paired_f1_scenario_uses_bootstrap_ci(tmp_path):
    run_a_id = "batch-1__accuracy__bf16"
    run_b_id = "batch-1__accuracy__quant"
    records_a = [
        _sample(run_id=run_a_id, model="bf16", scenario="long_context",
                sample_id=f"f{index}", score=0.9, family="normalized_match",
                metric="f1", dataset="longbench")
        for index in range(10)
    ]
    records_b = [
        _sample(run_id=run_b_id, model="quant", scenario="long_context",
                sample_id=f"f{index}", score=0.7, family="normalized_match",
                metric="f1", dataset="longbench")
        for index in range(10)
    ]
    _write_batch(tmp_path)
    run_a = _make_run(tmp_path, run_a_id, records=records_a)
    run_b = _make_run(tmp_path, run_b_id, records=records_b)

    result = accuracy_paired_comparison(run_a, run_b)
    row = result["scenarios"][0]
    assert row["binary"] is False
    assert row.get("mcnemar_p") is None
    assert row["delta_pp"] == pytest.approx(20.0)
    # All paired deltas are identical, so the bootstrap CI collapses on 20pp.
    assert row["bootstrap_ci95_low_pp"] == pytest.approx(20.0)
    assert row["bootstrap_ci95_high_pp"] == pytest.approx(20.0)

    text = render_accuracy_paired(result)
    assert "20.00" in text and "[20.00, 20.00]" in text
    assert "显著（A 更好）" in text


# --------------------------------------------------------------------------- #
# accuracy aggregates table
# --------------------------------------------------------------------------- #


def _table_runs(output_root: Path):
    run_a_id = "batch-1__accuracy__bf16"
    run_b_id = "batch-1__accuracy__quant"
    records_a = [
        _sample(run_id=run_a_id, model="bf16", scenario="knowledge_qa",
                sample_id="k0", score=1.0),
        _sample(run_id=run_a_id, model="bf16", scenario="knowledge_qa",
                sample_id="k1", score=1.0),
        _sample(run_id=run_a_id, model="bf16", scenario="knowledge_qa",
                sample_id="k2", score=1.0),
        _sample(run_id=run_a_id, model="bf16", scenario="knowledge_qa",
                sample_id="k3", score=0.0),
        _sample(run_id=run_a_id, model="bf16", scenario="math",
                sample_id="g0", score=0.8, dataset="gsm8k",
                family="normalized_match", metric="f1"),
        _sample(run_id=run_a_id, model="bf16", scenario="math",
                sample_id="g1", score=0.6, dataset="gsm8k",
                family="normalized_match", metric="f1"),
        _sample(run_id=run_a_id, model="bf16", scenario="math",
                sample_id="c0", score=0.5, dataset="cmath",
                family="normalized_match", metric="f1"),
        _sample(run_id=run_a_id, model="bf16", scenario="math",
                sample_id="c1", score=0.7, dataset="cmath",
                family="normalized_match", metric="f1"),
    ]
    records_b = [
        _sample(run_id=run_b_id, model="quant", scenario="knowledge_qa",
                sample_id="k0", score=0.0),
        _sample(run_id=run_b_id, model="quant", scenario="knowledge_qa",
                sample_id="k1", score=0.0),
        _sample(run_id=run_b_id, model="quant", scenario="knowledge_qa",
                sample_id="k2", score=0.0),
        _sample(run_id=run_b_id, model="quant", scenario="knowledge_qa",
                sample_id="k3", score=0.0),
        _sample(run_id=run_b_id, model="quant", scenario="math",
                sample_id="g0", score=0.4, dataset="gsm8k",
                family="normalized_match", metric="f1"),
        _sample(run_id=run_b_id, model="quant", scenario="math",
                sample_id="g1", score=0.4, dataset="gsm8k",
                family="normalized_match", metric="f1"),
        _sample(run_id=run_b_id, model="quant", scenario="math",
                sample_id="c0", score=0.5, dataset="cmath",
                family="normalized_match", metric="f1"),
        _sample(run_id=run_b_id, model="quant", scenario="math",
                sample_id="c1", score=0.5, dataset="cmath",
                family="normalized_match", metric="f1"),
    ]
    _write_batch(output_root)
    run_a = _make_run(output_root, run_a_id, records=records_a)
    run_b = _make_run(output_root, run_b_id, records=records_b)
    return run_a, run_b


def test_render_accuracy_table_paths_and_documents(tmp_path):
    run_a, run_b = _table_runs(tmp_path)

    text = render_accuracy_table([run_a, run_b], run_paths=[run_a, run_b])
    assert "# 精度对比" in text
    assert "method: accuracy" in text
    assert f"testset SHA-256: {TEST_SHA}" in text
    assert "cold, repeats=1" in text
    assert "| 模型 | 场景 | n | score | accuracy | 95%CI |" in text
    assert "knowledge_qa" in text
    # A scenario with two datasets shows each cell explicitly.
    assert "math/gsm8k" in text and "math/cmath" in text
    assert "整体(macro)" in text and "整体(micro)" in text
    # bf16 knowledge_qa = 3/4, macro over scenarios, etc.
    assert "0.7500" in text
    assert "0.7000" in text
    assert "[" in text and "]" in text

    # Loaded documents alone still render; the SHA is explicitly unknown.
    documents = [json.loads((run / layout.AGGREGATES_JSON).read_text()) for run in (run_a, run_b)]
    doc_text = render_accuracy_table(documents)
    assert "testset SHA-256: unknown" in doc_text
    assert "knowledge_qa" in doc_text


# --------------------------------------------------------------------------- #
# analysis CLI: accuracy + performance paths
# --------------------------------------------------------------------------- #


def test_analysis_cli_accuracy_table_paired_json_and_methods(tmp_path, capsys):
    run_a, run_b = _clean_paired_runs(tmp_path)

    # Non-paired accuracy: scenario table with macro/micro.
    assert analysis_cli.main([str(run_a), str(run_b)]) == 0
    out = capsys.readouterr().out
    assert "精度对比" in out and "knowledge_qa" in out
    assert "整体(macro)" in out
    assert f"testset SHA-256: {TEST_SHA}" in out

    # --json on the non-paired path emits the aggregates documents.
    assert analysis_cli.main([str(run_a), str(run_b), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list) and len(payload) == 2
    assert all(doc["method"] == "accuracy" for doc in payload)

    # Paired accuracy table.
    assert analysis_cli.main([str(run_a), str(run_b), "--paired"]) == 0
    out = capsys.readouterr().out
    assert "p(McNemar)" in out and "p(Holm)" in out
    assert "+100.00" in out and "unpaired: 0" in out

    # Paired accuracy JSON is machine-readable and keeps the explicit lists.
    assert analysis_cli.main([str(run_a), str(run_b), "--paired", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is True
    assert result["n_paired"] == 8
    assert result["testset_sha256"] == TEST_SHA
    assert result["ok"] is True
    assert result["explicit"]["unpaired"] == []

    # Explicit --method accuracy agrees with the runs.
    assert (
        analysis_cli.main(
            [str(run_a), str(run_b), "--paired", "--method", "accuracy"]
        )
        == 0
    )
    capsys.readouterr()

    # Noise baseline is forwarded to the verdict (Δpp=100 <= 200).  Use a
    # single-scenario fixture: Holm correction keeps the two-scenario Δ above.
    noise_a, noise_b = _discordant_run_dirs(tmp_path / "noise", a_score=1.0, b_score=0.0)
    assert (
        analysis_cli.main(
            [
                str(noise_a),
                str(noise_b),
                "--paired",
                "--noise-baseline-pp",
                "200",
            ]
        )
        == 0
    )
    assert "显著但未超过噪声基线" in capsys.readouterr().out


def test_analysis_cli_rejects_mixed_methods(tmp_path, capsys):
    run_a, run_b = _clean_paired_runs(tmp_path)
    performance_run = _make_run(
        tmp_path,
        "batch-9__performance__m",
        method="performance",
        records=[],
        aggregates=False,
    )

    # Explicit --method contradicts the recorded methods.
    assert (
        analysis_cli.main([str(run_a), str(run_b), "--method", "performance"])
        == 2
    )
    err = capsys.readouterr().err
    assert "cannot compare performance and accuracy runs" in err

    # auto detects a mixed target list.
    assert analysis_cli.main([str(run_a), str(performance_run), "--paired"]) == 2
    err = capsys.readouterr().err
    assert "cannot compare performance and accuracy runs" in err

    # Accuracy paired needs exactly two runs.
    run_c = _make_run(tmp_path, "batch-1__accuracy__third", records=[])
    assert analysis_cli.main([str(run_a), str(run_b), str(run_c), "--paired"]) == 2
    assert "exactly two runs" in capsys.readouterr().err


def test_analysis_cli_flags_unpaired_sample_sets(tmp_path, capsys):
    run_a, run_b, _ra, _rb = _paired_fixture(tmp_path)

    assert analysis_cli.main([str(run_a), str(run_b), "--paired"]) == 2
    captured = capsys.readouterr()
    assert "unpaired: 1" in captured.out
    assert "sample sets differ" in captured.err

    assert analysis_cli.main([str(run_a), str(run_b), "--paired", "--json"]) == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["ok"] is False
    assert len(payload["explicit"]["unpaired"]) == 1
    assert "sample sets differ" in captured.err


def test_analysis_cli_accuracy_error_exit_codes(tmp_path, capsys):
    # records.jsonl missing -> FileNotFoundError path -> exit 1.
    run_a = _make_run(tmp_path, "b1__accuracy__a", records=[], aggregates=True)
    run_b = _make_run(tmp_path, "b1__accuracy__b", records=[], aggregates=True)
    (run_a / layout.RECORDS).unlink(missing_ok=True)
    assert analysis_cli.main([str(run_a), str(run_b), "--paired"]) == 1
    assert "ERROR" in capsys.readouterr().err

    # aggregates.json missing on the non-paired path -> exit 1.
    (run_b / layout.AGGREGATES_JSON).unlink(missing_ok=True)
    assert analysis_cli.main([str(run_a), str(run_b)]) == 1
    assert "ERROR" in capsys.readouterr().err


def test_analysis_cli_incomparable_accuracy_result_exits_nonzero(tmp_path, capsys):
    run_a_id = "batch-1__accuracy__bf16"
    run_b_id = "batch-1__accuracy__quant"
    records_a = [
        _sample(run_id=run_a_id, model="bf16", scenario="knowledge_qa",
                sample_id="s1", score=1.0, family="mc_letter")
    ]
    records_b = [
        _sample(run_id=run_b_id, model="quant", scenario="knowledge_qa",
                sample_id="s1", score=1.0, family="ifeval")
    ]
    _write_batch(tmp_path)
    run_a = _make_run(tmp_path, run_a_id, records=records_a)
    run_b = _make_run(tmp_path, run_b_id, records=records_b)

    # The table is still printed for diagnosis, but the exit code is non-zero
    # and the reason goes to stderr (judge contracts differ).
    assert analysis_cli.main([str(run_a), str(run_b), "--paired"]) == 2
    captured = capsys.readouterr()
    assert "judge_family_mismatches" in captured.out
    assert "judge family/metric mismatch" in captured.err


# --------------------------------------------------------------------------- #
# performance path regression (must keep its historical output/exit codes)
# --------------------------------------------------------------------------- #


def _make_performance_run(output_root: Path, run_id: str, *, ttft: float) -> Path:
    run_dir = layout.run_dir(output_root, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / layout.AGGREGATES_JSON,
        {
            "schema_version": "3.0",
            "run_id": run_id,
            "generated_at": "2026-11-01T00:00:00+00:00",
            "overall": [
                {
                    "model": "MiniCPM5-1B",
                    "context_length": 1024,
                    "n_datasets": 1,
                    "n_samples": 5,
                    "perf": {"ttft_ms": {"n": 5, "mean": ttft}},
                }
            ],
            "cells": [],
        },
    )
    header = (
        "scenario,dataset,context_length,sample_id,repeat,status,ttft_ms,"
        "gen_tps,tpot_ms,e2e_ms,peak_memory_gb,input_context_tokens\n"
    )
    (run_dir / layout.SAMPLES_CSV).write_text(
        header
        + "coding,d,1024,s1,0,ok,100,50,10,200,1.0,1024\n"
        + "coding,d,1024,s2,0,ok,120,40,12,240,1.2,1024\n"
        + "coding,d,1024,s3,0,ok,110,45,11,220,1.1,1024\n",
        encoding="utf-8",
    )
    return run_dir


def test_performance_analysis_path_does_not_regress(tmp_path, capsys):
    run_a = _make_performance_run(tmp_path, "batchA__performance__m", ttft=10.0)
    run_b = _make_performance_run(tmp_path, "batchB__performance__m", ttft=12.0)

    assert analysis_cli.main([str(run_a), str(run_b), "--metrics", "ttft_ms"]) == 0
    out = capsys.readouterr().out
    assert "batchA__performance__m:ttft_ms" in out
    assert "Δ(batchB__performance__m-batchA__performance__m):ttft_ms" in out
    assert "| MiniCPM5-1B | 1K | 10.000 | 12.000 | 2.000 |" in out

    assert (
        analysis_cli.main([str(run_a), str(run_b), "--paired", "--metrics", "ttft_ms"])
        == 0
    )
    out = capsys.readouterr().out
    assert "Paired comparison" in out and "ttft_ms" in out

    assert analysis_cli.main([str(run_a), str(run_b), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [doc["run_id"] for doc in payload] == [
        "batchA__performance__m",
        "batchB__performance__m",
    ]

    # Explicit --method performance keeps working (and legacy runs default to it).
    assert (
        analysis_cli.main([str(run_a), str(run_b), "--method", "performance"])
        == 0
    )
    capsys.readouterr()

    # Unresolved run id keeps the historical exit code 1.
    assert analysis_cli.main(["missing-run", "--output-dir", str(tmp_path)]) == 1
    assert "ERROR" in capsys.readouterr().err


def test_catalog_load_records_reads_accuracy_fixture(tmp_path):
    """The fixture must round-trip through the real catalog reader."""
    run_a, _run_b, records_a, _rb = _paired_fixture(tmp_path)
    loaded = catalog.load_records(run_a)
    assert len(loaded) == len(records_a)
    assert {record.sample_id for record in loaded} >= {"k0", "k5", "k_only_a"}
    assert all(record.score is not None for record in loaded)
    # Persisted records may drop prompt messages; the reader restores a placeholder.
    assert loaded[0].request.messages == []
