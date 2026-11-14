"""Accuracy run projections (``method=accuracy``) — schema 3.0.

Pure functions shared by
:class:`~modelbench.adapters.storage.writer.AccuracyRunWriter` and
``rebuild``/``verify`` (:mod:`modelbench.adapters.storage.tools`).  The live
writer and a rebuilt bundle must produce semantically identical derived files
(``aggregates.json`` / ``samples.csv`` / ``report.md``, AC-A5/A6), so the
projections are computed **only** here instead of being duplicated per caller.

The aggregation itself is :mod:`modelbench.domain.accuracy_metrics`; this module
only serialises it, renders the human report and derives the accuracy counters
that go into ``run.json.outcome`` (design 32 §5.4/§5.5/§5.6).

Counting contract (see ``domain.accuracy_metrics``): ``judge_error`` is 0 and
stays in the score denominator, ``error``/``timeout`` are reported separately
and never enter it, and every field can be recomputed from ``records.jsonl``.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from modelbench.adapters.storage import layout
from modelbench.domain.accuracy_metrics import (
    accuracy_sample_results,
    aggregate_accuracy,
)
from modelbench.domain.models import (
    SCHEMA_VERSION,
    AccuracyCellAggregate,
    AccuracyOverallAggregate,
    AccuracySampleResult,
)

#: Method tag written to ``aggregates.json.method`` / outcome / report.
ACCURACY_METHOD = "accuracy"

#: Field order of the accuracy ``samples.csv`` — frozen, one-to-one with
#: :class:`~modelbench.domain.models.AccuracySampleResult` and with the design's
#: §5.5 column list (29 columns; the design text is the source of truth).
SAMPLES_CSV_FIELDS_ACCURACY = (
    "model",
    "model_id",
    "scenario",
    "dataset",
    "language",
    "sample_id",
    "upstream_id",
    "repeat",
    "phase",
    "status",
    "error",
    "judge_family",
    "judge_metric",
    "score",
    "correct",
    "format_failure",
    "judge_error",
    "max_new_tokens",
    "completion_tokens",
    "input_context_tokens",
    "actual_prompt_tokens",
    "ttft_ms",
    "e2e_ms",
    "finish_reason",
    "cache_mode",
    "cache_cleared",
    "cache_polluted",
    "prompt_sha256",
    "timestamp",
)

#: Outcome counters added on top of the perf-compatible block (design §5.6).
ACCURACY_OUTCOME_FIELDS = (
    "n_scored",
    "n_correct",
    "n_judge_error",
    "n_format_failure",
    "n_scenarios_scored",
)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(tmp, path)


def _unique_samples(samples: Iterable[AccuracySampleResult]) -> int:
    """Distinct ``(model, scenario, sample_id)`` measured samples."""
    return len(
        {(row.model, row.scenario, row.sample_id) for row in samples}
    )


# --------------------------------------------------------------------------- #
# aggregates.json / samples.csv
# --------------------------------------------------------------------------- #


def write_accuracy_aggregates(
    run_dir: Path,
    records: Sequence[Any],
    *,
    run_id: str,
) -> Dict[str, Any]:
    """Write the accuracy ``aggregates.json`` and return the projections.

    Shape (design 32 §5.4)::

        {"schema_version": "3.0", "method": "accuracy", "run_id": ...,
         "n_records": ..., "n_samples": ..., "n_cells": ...,
         "overall": [AccuracyOverallAggregate...], "cells": [AccuracyCellAggregate...]}

    ``cells``/``overall`` are :meth:`model_dump(mode="json")` payloads; the
    returned dict additionally carries the live objects (plus the per-sample
    rows) so the writer can render the report and write ``samples.csv`` without
    recomputing anything.
    """
    run_dir = Path(run_dir)
    rows = list(records)
    cells, overall = aggregate_accuracy(rows)
    samples = accuracy_sample_results(rows)
    n_samples = _unique_samples(samples)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": ACCURACY_METHOD,
        "run_id": str(run_id),
        "generated_at": _utc_now(),
        "n_records": len(rows),
        "n_samples": n_samples,
        "n_cells": len(cells),
        "overall": [entry.model_dump(mode="json") for entry in overall],
        "cells": [cell.model_dump(mode="json") for cell in cells],
    }
    _write_json(run_dir / layout.AGGREGATES_JSON, payload)
    return {
        "aggregates": payload,
        "cells": cells,
        "overall": overall,
        "samples": samples,
        "n_records": len(rows),
        "n_samples": n_samples,
        "n_cells": len(cells),
    }


def write_accuracy_samples_csv(
    path: Path,
    records: Sequence[Any],
    *,
    samples: Optional[Sequence[AccuracySampleResult]] = None,
) -> List[AccuracySampleResult]:
    """Write ``samples.csv`` with :data:`SAMPLES_CSV_FIELDS_ACCURACY` order.

    ``samples`` may carry an already computed projection (the writer gets it
    from :func:`write_accuracy_aggregates`); when omitted it is recomputed from
    ``records``.  Returns the rows that were written.
    """
    rows = list(samples) if samples is not None else accuracy_sample_results(records)
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLES_CSV_FIELDS_ACCURACY)
        writer.writeheader()
        for row in rows:
            payload = row.model_dump(mode="json")
            writer.writerow(
                {key: payload.get(key) for key in SAMPLES_CSV_FIELDS_ACCURACY}
            )
    return rows


# --------------------------------------------------------------------------- #
# outcome counters (run.json)
# --------------------------------------------------------------------------- #


def accuracy_outcome(
    records: Sequence[Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Accuracy counters to merge into the perf-compatible ``run.json.outcome``.

    The writer first computes the shared block with
    :meth:`RunWriter._finalize_outcome` (``n_selected``/``n_measured``/``n_ok``/
    ``n_records``/``n_samples``/``n_cells``/``skipped``/``warnings``/
    ``duration_s``/``memory_errors``/``cancelled``/``status``) and then updates
    it with these five counters (design 32 §5.6).  ``extra`` mirrors the dict the
    writer receives; an explicit value there (e.g. an orchestrator that already
    counted) wins over the recomputation, and every counter stays derivable from
    ``records`` (AC-A5).

    Definitions (see ``domain.accuracy_metrics`` for the counting rules):

    * ``n_scored`` — rows in the score denominator (``ok`` + ``judge_error``);
    * ``n_correct`` — rows whose normalized score is exactly 1.0;
    * ``n_judge_error`` — rows the judge failed on (0 and still counted);
    * ``n_format_failure`` — rows flagged as a format/parse failure;
    * ``n_scenarios_scored`` — distinct scenarios with a score denominator.
    """
    extra = dict(extra or {})
    cells, _overall = aggregate_accuracy(list(records))
    counts: Dict[str, Any] = {
        "n_scored": sum(int(cell.n_scored) for cell in cells),
        "n_correct": sum(int(cell.n_correct) for cell in cells),
        "n_judge_error": sum(int(cell.n_judge_error) for cell in cells),
        "n_format_failure": sum(int(cell.n_format_failure) for cell in cells),
        "n_scenarios_scored": len(
            {cell.scenario for cell in cells if int(cell.n_scored) > 0}
        ),
    }
    for key in ACCURACY_OUTCOME_FIELDS:
        value = extra.get(key)
        if value is not None:
            counts[key] = int(value)
    return counts


# --------------------------------------------------------------------------- #
# report.md
# --------------------------------------------------------------------------- #


def _num(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _ci(low: Optional[float], high: Optional[float], digits: int = 3) -> str:
    if low is None or high is None:
        return "-"
    return f"[{float(low):.{digits}f}, {float(high):.{digits}f}]"


def _rate(numerator: int, denominator: int) -> str:
    if not denominator:
        return "-"
    return f"{100.0 * float(numerator) / float(denominator):.1f}%"


def _judge_families(cells: Sequence[AccuracyCellAggregate]) -> List[str]:
    return sorted({cell.judge_family for cell in cells if cell.judge_family})


def _testset_block(summary: Dict[str, Any]) -> Dict[str, Any]:
    testset = summary.get("testset") or {}
    return testset if isinstance(testset, Mapping) else {}


def render_accuracy_report(
    *,
    run_id: str,
    cells: Sequence[AccuracyCellAggregate],
    overall: Sequence[AccuracyOverallAggregate],
    samples: Sequence[AccuracySampleResult],
    summary: Optional[Dict[str, Any]] = None,
    hardware: Optional[Dict[str, Any]] = None,
    omlx: Optional[Dict[str, Any]] = None,
) -> str:
    """Render ``report.md`` for one accuracy run (Chinese, protocol first).

    Sections: run protocol (cold / temperature=0 / no decode guard / repeats=1),
    testset SHA, environment, per-scenario score table, per-model macro average,
    language profile, error statistics and judge dependencies/warnings.
    ``summary`` is the merged ``run.json.config`` view (``accuracy``/``run``/
    ``testset``/``warnings``/``judge_dependencies``); every block degrades to an
    explicit "-" instead of disappearing when a caller has less information.
    """
    summary = dict(summary or {})
    hw = dict(hardware or {})
    om = dict(omlx or {})
    chip = " ".join(
        str(part)
        for part in (hw.get("chip_name") or hw.get("chip"), hw.get("chip_variant"))
        if part
    )
    run_cfg = summary.get("run") or {}
    acc_cfg = summary.get("accuracy") or {}
    testset = _testset_block(summary)
    warnings = list(summary.get("warnings") or [])
    dependencies = summary.get("judge_dependencies") or []
    skipped = list(summary.get("skipped") or [])

    lines: List[str] = []
    lines.append(f"# Run `{run_id}` 精度测试摘要\n")
    lines.append(
        "> 本报告只描述**精度口径**（`method=accuracy`）：回答对不对。"
        "不得与性能 run（TTFT/TPS/内存）横比，也不合成综合分。"
    )
    lines.append("")

    # -- protocol ------------------------------------------------------- #
    lines.append("## 运行协议（固定口径）\n")
    samples_per_scenario = acc_cfg.get("samples_per_scenario")
    lines.append(
        f"- 方法：精度测试（`method=accuracy`）；样本选择：每场景 "
        f"{samples_per_scenario if samples_per_scenario is not None else '全部（250）'}"
        " 条（`accuracy.samples_per_scenario`；固定样本集，不做 per-model 裁剪）"
    )
    lines.append(
        "- 缓存：**cold**（每个 measured 请求前清 hot+SSD，禁写缓存，"
        "`cache.cached_tokens==0`）；warm/none 不适用于本方法"
    )
    lines.append(
        "- 解码：`temperature=0`、`top_p=1`、无重复惩罚、**无 decode guard**"
        "（精度不追加长输出指令，不改变被评测的回答本身）"
    )
    lines.append(
        "- 生成长度：逐行 `judge.max_new_tokens`（记录于 `request.max_tokens`）；"
        "轮数 `repeats="
        f"{run_cfg.get('repeats', 1)}`（每样本一次响应 → 一条判分记录）"
    )
    lines.append(
        "- 重试：传输/HTTP 错误重试 1 次；仍失败记 `status=error`，"
        "该样本不计判分但单列错误率"
    )
    lines.append(
        f"- 思考模式：`{run_cfg.get('thinking_mode')}`；"
        f"模型级预热（丢弃、不判分）：{run_cfg.get('warmup_per_model', 1)} 次/模型"
    )
    lines.append(
        f"- 记录数：{summary.get('n_records')}；样本数：{summary.get('n_samples')}；"
        f"测试单元（cell）：{summary.get('n_cells')}；"
        f"耗时：{summary.get('duration_s', '-')}s"
    )
    lines.append(
        "- 测试集："
        f"`{testset.get('path') or '-'}`；SHA-256："
        f"`{testset.get('sha256') or '-'}`；行数：{testset.get('rows', '-')}"
        "（同一 testset SHA 的 run 才可配对比较）"
    )
    lines.append("")

    # -- environment ---------------------------------------------------- #
    lines.append("## 环境\n")
    lines.append(
        f"- 硬件：{chip or '-'}；内存 {hw.get('memory_gb')} GB；"
        f"GPU {hw.get('gpu_cores')} 核"
    )
    lines.append(
        f"- 系统：{hw.get('os_version')}；"
        f"oMLX：{om.get('omlx_version') or om.get('version') or hw.get('omlx_version')}"
    )
    flags = om.get("feature_flags") or []
    lines.append(f"- 加速特性（应为空 = 默认配置）：{flags if flags else '无'}")
    lines.append("")

    # -- per-scenario scores -------------------------------------------- #
    lines.append("## 场景得分\n")
    lines.append(
        "> 每行一个 `(模型, 场景, 数据集)` cell。`mean_score` 为判分行均值"
        "（判题失败计 0 且保留在分母；error/timeout 不进分母）；"
        "`accuracy` 仅 0/1 判题族有值，F1 场景为 `-`。"
    )
    lines.append("")
    header = [
        "模型",
        "场景",
        "数据集",
        "判题族",
        "指标",
        "样本",
        "有效",
        "mean_score",
        "accuracy",
        "score 95%CI",
        "accuracy 95%CI",
        "判题失败",
        "格式失败",
        "错误",
        "超时",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for cell in cells:
        lines.append(
            "| {model} | {scenario} | {dataset} | {family} | {metric} | {n} | {valid} | "
            "{mean} | {acc} | {score_ci} | {acc_ci} | {judge_error} | {fmt} | "
            "{error} | {timeout} |".format(
                model=cell.model,
                scenario=cell.scenario,
                dataset=cell.dataset,
                family=cell.judge_family or "-",
                metric=cell.metric or "-",
                n=cell.n_samples,
                valid=cell.n_valid,
                mean=_num(cell.mean_score),
                acc=_num(cell.accuracy),
                score_ci=_ci(cell.score_ci95_low, cell.score_ci95_high),
                acc_ci=_ci(cell.accuracy_ci95_low, cell.accuracy_ci95_high),
                judge_error=cell.n_judge_error,
                fmt=cell.n_format_failure,
                error=cell.n_error,
                timeout=cell.n_timeout,
            )
        )
    if not cells:
        lines.append("| - | - | - | - | - | 0 | 0 | - | - | - | - | 0 | 0 | 0 | 0 |")
    lines.append("")

    # -- macro average -------------------------------------------------- #
    lines.append("## 总体（宏平均）\n")
    lines.append(
        "> `macro_score` 是**结论指标**：场景等权平均（每场景先聚合数据集再平均，"
        "避免行数/场景权重影响）；`micro_score` 是全行平均，仅辅助；"
        "`macro_accuracy` 只统计 0/1 场景。"
    )
    lines.append("")
    header = [
        "模型",
        "macro_score",
        "micro_score",
        "macro_accuracy",
        "场景数",
        "样本",
        "有效",
        "判分分母",
        "正确",
        "判题失败",
        "格式失败",
        "错误",
        "超时",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for entry in overall:
        lines.append(
            "| {model} | {macro} | {micro} | {macro_acc} | {scenarios} | {n} | {valid} | "
            "{scored} | {correct} | {judge_error} | {fmt} | {error} | {timeout} |".format(
                model=entry.model,
                macro=_num(entry.macro_score),
                micro=_num(entry.micro_score),
                macro_acc=_num(entry.macro_accuracy),
                scenarios=entry.n_scenarios_scored,
                n=entry.n_samples,
                valid=entry.n_valid,
                scored=entry.n_scored,
                correct=entry.n_correct,
                judge_error=entry.n_judge_error,
                fmt=entry.n_format_failure,
                error=entry.n_error,
                timeout=entry.n_timeout,
            )
        )
    if not overall:
        lines.append("| - | - | - | - | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |")
    lines.append("")

    # -- language profile ----------------------------------------------- #
    lines.append("## 语言剖面\n")
    lines.append(
        "> 按 `sample_meta.language` 分组；单语小样本只用于定位退化来源，不做排名。"
    )
    lines.append("")
    header = ["模型", "语言", "样本", "有效", "mean_score", "accuracy", "判题失败"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for entry in overall:
        by_language = entry.by_language or {}
        for language in sorted(by_language):
            slice_ = by_language[language]
            lines.append(
                "| {model} | {language} | {n} | {valid} | {mean} | {acc} | {judge} |".format(
                    model=entry.model,
                    language=language,
                    n=slice_.n_samples,
                    valid=slice_.n_valid,
                    mean=_num(slice_.mean_score),
                    acc=_num(slice_.accuracy),
                    judge=slice_.n_judge_error,
                )
            )
    lines.append("")

    # -- error statistics ----------------------------------------------- #
    lines.append("## 错误统计\n")
    header = [
        "模型",
        "样本",
        "有效",
        "ok",
        "error",
        "timeout",
        "判题失败",
        "格式失败",
        "判分失败率",
        "格式失败率",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for entry in overall:
        lines.append(
            "| {model} | {n} | {valid} | {ok} | {error} | {timeout} | {judge} | {fmt} | "
            "{judge_rate} | {fmt_rate} |".format(
                model=entry.model,
                n=entry.n_samples,
                valid=entry.n_valid,
                ok=entry.n_ok,
                error=entry.n_error,
                timeout=entry.n_timeout,
                judge=entry.n_judge_error,
                fmt=entry.n_format_failure,
                judge_rate=_rate(entry.n_judge_error, entry.n_samples),
                fmt_rate=_rate(entry.n_format_failure, entry.n_samples),
            )
        )
    lines.append("")
    mismatched = [
        cell
        for cell in cells
        if cell.n_samples and cell.n_format_failure / cell.n_samples > 0.20
    ]
    if mismatched:
        lines.append(
            "- **格式不适配**（单场景格式失败率 >20%）："
            + "、".join(f"{cell.model}/{cell.scenario}" for cell in mismatched)
            + "（响应无法解析，计 0 分；建议检查 prompt/判题契约）"
        )
    failing = [row for row in samples if row.judge_error]
    if failing:
        examples = "、".join(
            f"{row.scenario}/{row.sample_id}" for row in failing[:5]
        )
        lines.append(
            f"- `judge_error` 共 {len(failing)} 条（计 0 分并保留在分母，"
            f"分数不会虚高）；示例：{examples}"
        )
    if skipped:
        lines.append(
            f"- 跳过（`outcome.skipped`）共 {len(skipped)} 条："
            "超窗/样本缺失等原因，不计入分母"
        )
    if not cells:
        lines.append("- 无判分记录（本次 run 没有产生 measured 判分结果）")
    lines.append("")

    # -- judge dependencies / warnings ---------------------------------- #
    lines.append("## 判题依赖与警告\n")
    lines.append(
        "- 判题族：" + ("、".join(_judge_families(cells)) or "-")
        + "（确定性判题：mc_letter / ifeval / normalized_match / code_exec / tool_call）"
    )
    if isinstance(dependencies, Mapping):
        dep_text = "、".join(f"{key}={value}" for key, value in sorted(dependencies.items()))
    elif isinstance(dependencies, (list, tuple)):
        dep_text = "、".join(str(item) for item in dependencies)
    else:
        dep_text = str(dependencies)
    lines.append(f"- 判题依赖（缺失的 [eval] 包等）：{dep_text or '未记录'}")
    if warnings:
        for warning in warnings:
            lines.append(f"- ⚠️ {warning}")
    else:
        lines.append("- 警告：无")
    lines.append("")

    lines.append(
        "> 判题失败（`judge_error`）与格式失败（`format_failure`）不会从分母剔除："
        "依赖缺失/异常时对应样本计 0 分，由错误率与计数暴露质量（不可静默给分）。"
    )
    lines.append(
        "> 目录结构（schema 3.0）：`records.jsonl` 为权威逐样本证据（含 `score`），"
        "`aggregates.json`/`samples.csv`/`report.md` 是可由其复算的派生投影；"
        "`./modelbench rebuild` 可重建本目录的派生文件。"
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "ACCURACY_METHOD",
    "ACCURACY_OUTCOME_FIELDS",
    "SAMPLES_CSV_FIELDS_ACCURACY",
    "accuracy_outcome",
    "render_accuracy_report",
    "write_accuracy_aggregates",
    "write_accuracy_samples_csv",
]
