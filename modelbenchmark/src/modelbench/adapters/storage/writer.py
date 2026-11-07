"""Run bundle writer: records, environment, summary, CSVs and report.

Bundle layout (schema 2.1, performance-only)::

    manifest.json   config snapshot, hardware, oMLX snapshot, models, git
    hardware.json   host/OS/dependencies + oMLX device info
    omlx.json       oMLX version, global settings, per-model settings/flags
    models.json     /v1/models/status snapshot
    records.jsonl   one authoritative record per request
    summary.json    per (model, context) averages + cells + per-sample rows
    summary.csv     per (model, context) averages
    cells.csv       per model x scenario x dataset x context
    samples.csv     per-sample results (incl. input context size)
    report.md       human-readable summary
"""

from __future__ import annotations

import csv
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from modelbench.adapters.storage.catalog import upsert_run
from modelbench.domain.context_windows import describe_windows
from modelbench.domain.metrics import (
    aggregate_overall,
    aggregate_records,
    sample_results,
)
from modelbench.domain.models import (
    SCHEMA_VERSION,
    CellAggregate,
    OverallAggregate,
    RunManifest,
    SampleRecord,
    SampleResult,
)

#: Metrics rendered in the human-readable overall/markdown tables.
_REPORT_METRICS = (
    ("ttft_ms", "TTFT(ms)"),
    ("gen_tps", "TG TPS"),
    ("tpot_ms", "TPOT(ms)"),
    ("e2e_ms", "E2E(ms)"),
    ("peak_memory_gb", "峰值内存(GB)"),
    ("input_context_tokens", "输入token"),
)


def sanitize_name(value: str) -> str:
    """Filesystem-safe, human-recognizable component."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "unnamed"


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class RunWriter:
    """Owns one ``output/<run_id>/`` directory."""

    def __init__(
        self,
        output_root: Path,
        run_id: Optional[str] = None,
        overwrite: bool = False,
        batch: Optional[Dict[str, Any]] = None,
    ):
        self.output_root = Path(output_root)
        self.run_id = run_id or new_run_id()
        self.batch = dict(batch or {})
        self.run_dir = self.output_root / self.run_id
        if self.run_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"run directory already exists: {self.run_dir} (choose another --run-id)"
                )
            shutil.rmtree(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "raw").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)
        self._records_path = self.run_dir / "records.jsonl"
        self._records_fh = None
        self._count = 0
        self._manifest: Optional[RunManifest] = None
        self._hardware: Dict[str, Any] = {}
        self._omlx: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #

    def initialize(
        self,
        manifest: RunManifest,
        hardware: Dict[str, Any],
        models: Dict[str, Any],
        omlx: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._manifest = manifest
        self._hardware = hardware
        self._omlx = omlx or {}
        self._write_json("manifest.json", manifest.model_dump(mode="json"))
        self._write_json("hardware.json", hardware)
        self._write_json("models.json", models)
        self.write_omlx(self._omlx)

    def write_omlx(self, omlx: Dict[str, Any]) -> None:
        if omlx:
            self._omlx = omlx
        self._write_json("omlx.json", self._omlx)

    def _write_json(self, name: str, payload: Any) -> None:
        (self.run_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------------ #

    def write_record(self, record: SampleRecord) -> None:
        if self._records_fh is None:
            self._records_fh = self._records_path.open("a", encoding="utf-8")
        payload = record.model_dump(mode="json")
        self._records_fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._records_fh.flush()
        self._count += 1

        ctx = f"{record.context_length // 1024}k" if record.context_length else "native"
        raw_dir = (
            self.run_dir
            / "raw"
            / sanitize_name(record.model)
            / sanitize_name(record.scenario)
            / ctx
        )
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / f"{sanitize_name(record.sample_id)}__r{record.repeat}.json"
        raw_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def close(self) -> None:
        if self._records_fh is not None:
            self._records_fh.close()
            self._records_fh = None

    # ------------------------------------------------------------------ #

    def finalize(
        self,
        records: List[SampleRecord],
        extra: Optional[Dict[str, Any]] = None,
        perf: Optional[Dict[str, Any]] = None,
        run_config: Optional[Dict[str, Any]] = None,
        omlx: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.close()
        if omlx is not None:
            self._omlx = omlx
            self.write_omlx(omlx)
            if self._manifest is not None:
                self._manifest.omlx = omlx
                self._write_json("manifest.json", self._manifest.model_dump(mode="json"))
        cells = aggregate_records(records)
        overall = aggregate_overall(records)
        samples = sample_results(records)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "mode": "performance",
            "batch": self.batch,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "perf": perf or {},
            "run": run_config or {},
            "n_records": len(records),
            "n_samples": len(samples),
            "n_cells": len(cells),
            "overall": [entry.model_dump(mode="json") for entry in overall],
            "cells": [cell.model_dump(mode="json") for cell in cells],
            "samples": [row.model_dump(mode="json") for row in samples],
            "environment": {
                "hardware": "hardware.json",
                "omlx": "omlx.json",
                "models": "models.json",
            },
        }
        if extra:
            summary.update(extra)
        self._write_json("summary.json", summary)
        self._write_overall_csv(overall)
        self._write_cells_csv(cells)
        self._write_samples_csv(samples)
        (self.run_dir / "report.md").write_text(
            self._render_report(overall, cells, samples, summary), encoding="utf-8"
        )
        self._write_analysis(summary, samples)
        index_path = self._update_catalog(summary)
        result = {
            "run_dir": str(self.run_dir),
            "summary_json": str(self.run_dir / "summary.json"),
            "summary_csv": str(self.run_dir / "summary.csv"),
            "cells_csv": str(self.run_dir / "cells.csv"),
            "samples_csv": str(self.run_dir / "samples.csv"),
            "report_md": str(self.run_dir / "report.md"),
            "records_jsonl": str(self._records_path),
            "analysis_dir": str(self.run_dir / "analysis"),
            "index_json": str(index_path),
            "n_records": len(records),
            "n_samples": len(samples),
            "n_cells": len(cells),
        }
        if extra:
            for key, value in extra.items():
                result.setdefault(key, value)
        return result

    # ------------------------------------------------------------------ #
    # analysis projections + catalog
    # ------------------------------------------------------------------ #

    def _write_analysis(
        self, summary: Dict[str, Any], samples: Sequence[SampleResult]
    ) -> None:
        """Write small analysis-friendly projections next to the raw evidence."""
        analysis_dir = self.run_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        manifest = self._manifest.model_dump(mode="json") if self._manifest else {}
        selected = [
            str(model.get("alias") or model.get("id"))
            for model in (manifest.get("models") or {}).get("selected") or []
        ]
        run_meta = {
            "run_id": self.run_id,
            "batch": summary.get("batch") or {},
            "schema_version": summary.get("schema_version"),
            "mode": summary.get("mode"),
            "generated_at": summary.get("generated_at"),
            "created_at": manifest.get("created_at"),
            "duration_s": summary.get("duration_s"),
            "perf": summary.get("perf") or {},
            "run": summary.get("run") or {},
            "models_selected": selected,
            "models_measured": sorted(
                {entry.get("model") for entry in summary.get("overall", []) if entry.get("model")}
            ),
            "skipped": summary.get("skipped") or [],
            "warnings": summary.get("warnings") or [],
            "cancelled": bool(summary.get("cancelled")),
            "n_records": summary.get("n_records"),
            "n_samples": summary.get("n_samples"),
            "n_cells": summary.get("n_cells"),
            "git_commit": manifest.get("git_commit"),
            "environment": summary.get("environment") or {},
        }
        self._write_json("analysis/run.json", run_meta)
        environment = {
            "hardware": {
                "upload_fields": (self._hardware or {}).get("upload_fields"),
                "chip": (self._hardware or {}).get("chip"),
                "cpu": (self._hardware or {}).get("cpu"),
                "memory": (self._hardware or {}).get("memory"),
                "os": (self._hardware or {}).get("os"),
            },
            "omlx": {
                "version": (self._omlx or {}).get("omlx_version"),
                "feature_flags": (self._omlx or {}).get("feature_flags"),
                "default_config": (self._omlx or {}).get("default_config"),
                "models": {
                    model_id: {
                        "engine_type": info.get("engine_type"),
                        "model_type": info.get("model_type"),
                        "quantization": info.get("quantization"),
                        "context_length": info.get("context_length"),
                        "settings": info.get("settings"),
                        "feature_flags": info.get("feature_flags"),
                    }
                    for model_id, info in ((self._omlx or {}).get("models") or {}).items()
                },
            },
            "perf": summary.get("perf") or {},
            "run": summary.get("run") or {},
        }
        self._write_json("analysis/environment.json", environment)
        self._write_samples_long_csv(analysis_dir / "samples_long.csv", samples)
        (analysis_dir / "README.md").write_text(
            self._render_analysis_readme(), encoding="utf-8"
        )

    def _write_samples_long_csv(self, path: Path, samples: Sequence[SampleResult]) -> None:
        metric_fields = (
            "ttft_ms",
            "ttft_visible_ms",
            "tpot_ms",
            "gen_tps",
            "e2e_ms",
            "prefill_tps",
            "peak_cpu_pct",
            "peak_rss_mb",
            "peak_memory_gb",
            "input_context_tokens",
            "input_context_chars",
            "completion_tokens",
        )
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "run_id",
                    "model",
                    "scenario",
                    "dataset",
                    "context_length",
                    "sample_id",
                    "repeat",
                    "status",
                    "finish_reason",
                    "length_capped",
                    "metric",
                    "value",
                ],
            )
            writer.writeheader()
            for row in samples:
                for metric in metric_fields:
                    writer.writerow(
                        {
                            "run_id": self.run_id,
                            "model": row.model,
                            "scenario": row.scenario,
                            "dataset": row.dataset,
                            "context_length": row.context_length,
                            "sample_id": row.sample_id,
                            "repeat": row.repeat,
                            "status": row.status,
                            "finish_reason": row.finish_reason,
                            "length_capped": row.length_capped,
                            "metric": metric,
                            "value": getattr(row, metric),
                        }
                    )

    def _render_analysis_readme(self) -> str:
        return (
            f"# Analysis files for run `{self.run_id}`\n\n"
            "本目录是给二次分析用的小体积投影，原始证据仍在 run 根目录。\n\n"
            "| 文件 | 内容 |\n|---|---|\n"
            "| `run.json` | 运行配置/模型/跳过项/计数（不含逐样本大表） |\n"
            "| `environment.json` | 硬件、oMLX 配置、feature flags、perf/run 参数 |\n"
            "| `samples_long.csv` | 长表：每行一个 (sample, metric, value)，含上下文档位，便于 pivot |\n\n"
            "原始文件：`records.jsonl`（逐请求权威证据）、`summary.json`、`summary.csv`、"
            "`cells.csv`、`samples.csv`、`report.md`。\n\n"
            "## pandas 示例\n\n"
            "```python\n"
            "import pandas as pd\n"
            f"run = \"output/{self.run_id}\"\n"
            "samples = pd.read_csv(f\"{run}/samples.csv\")\n"
            "# IMPORTANT: never average across context_length; 1K/8K/32K do not mix.\n"
            "print(samples.groupby([\"model\", \"context_length\"])[[\"ttft_ms\", \"gen_tps\", \"peak_memory_gb\"]].mean())\n"
            "\n"
            "long = pd.read_csv(f\"{run}/analysis/samples_long.csv\")\n"
            "pivot = long.pivot_table(index=[\"model\", \"context_length\", \"dataset\", \"sample_id\"], "
            "columns=\"metric\", values=\"value\")\n"
            "```\n"
        )

    def _update_catalog(self, summary: Dict[str, Any]) -> Path:
        manifest = self._manifest.model_dump(mode="json") if self._manifest else {}
        upsert_run(self.output_root, self.run_dir, summary, manifest)
        return self.output_root / "index.json"

    # ------------------------------------------------------------------ #
    # CSV
    # ------------------------------------------------------------------ #

    def _write_overall_csv(self, overall: Sequence[OverallAggregate]) -> None:
        fields = [
            "model",
            "context_length",
            "n_datasets",
            "n_samples",
            "n_valid",
            "gen_tps_n",
            "n_contaminated",
            "n_error",
            "n_length_capped",
            "unstable",
        ]
        for metric in (
            "ttft_ms",
            "ttft_visible_ms",
            "tpot_ms",
            "e2e_ms",
            "gen_tps",
            "prefill_tps",
            "input_context_tokens",
            "input_context_chars",
            "peak_cpu_pct",
            "peak_memory_gb",
        ):
            fields += [
                f"{metric}_mean",
                f"{metric}_median",
                f"{metric}_p95",
                f"{metric}_max",
            ]
        fields.append("peak_memory_gb_global_max")
        rows: List[Dict[str, Any]] = []
        for entry in overall:
            row: Dict[str, Any] = {
                "model": entry.model,
                "context_length": entry.context_length,
                "n_datasets": entry.n_datasets,
                "n_samples": entry.n_samples,
                "n_valid": entry.n_valid,
                "gen_tps_n": (
                    entry.perf["gen_tps"].n if "gen_tps" in entry.perf else 0
                ),
                "n_contaminated": entry.n_contaminated,
                "n_error": entry.n_error,
                "n_length_capped": entry.n_length_capped,
                "unstable": entry.unstable,
                "peak_memory_gb_global_max": entry.peak_memory_gb_max,
            }
            for metric, dist in entry.perf.items():
                row[f"{metric}_mean"] = dist.mean
                row[f"{metric}_median"] = dist.median
                row[f"{metric}_p95"] = dist.p95
                row[f"{metric}_max"] = dist.max
            rows.append(row)
        self._write_csv("summary.csv", fields, rows)

    def _write_cells_csv(self, cells: Sequence[CellAggregate]) -> None:
        fields = [
            "model",
            "scenario",
            "dataset",
            "context_length",
            "n_samples",
            "n_repeats",
            "n_valid",
            "gen_tps_n",
            "n_contaminated",
            "n_error",
            "n_length_capped",
            "unstable",
        ]
        for metric in (
            "ttft_ms",
            "ttft_visible_ms",
            "tpot_ms",
            "e2e_ms",
            "gen_tps",
            "prefill_tps",
            "input_context_tokens",
            "input_context_chars",
            "peak_cpu_pct",
            "peak_memory_gb",
        ):
            fields += [f"{metric}_mean", f"{metric}_median", f"{metric}_p95", f"{metric}_max"]
        rows: List[Dict[str, Any]] = []
        for cell in cells:
            row: Dict[str, Any] = {
                "model": cell.model,
                "scenario": cell.scenario,
                "dataset": cell.dataset,
                "context_length": cell.context_length,
                "n_samples": cell.n_samples,
                "n_repeats": cell.n_repeats,
                "n_valid": cell.n_valid,
                "gen_tps_n": cell.perf["gen_tps"].n if "gen_tps" in cell.perf else 0,
                "n_contaminated": cell.n_contaminated,
                "n_error": cell.n_error,
                "n_length_capped": cell.n_length_capped,
                "unstable": cell.unstable,
            }
            for metric, dist in cell.perf.items():
                row[f"{metric}_mean"] = dist.mean
                row[f"{metric}_median"] = dist.median
                row[f"{metric}_p95"] = dist.p95
                row[f"{metric}_max"] = dist.max
            rows.append(row)
        self._write_csv("cells.csv", fields, rows)

    def _write_samples_csv(self, samples: Sequence[SampleResult]) -> None:
        fields = list(SampleResult.model_fields)
        rows = [row.model_dump(mode="json") for row in samples]
        self._write_csv("samples.csv", fields, rows)

    def _write_csv(self, name: str, fields: List[str], rows: List[Dict[str, Any]]) -> None:
        path = self.run_dir / name
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fields})

    # ------------------------------------------------------------------ #
    # report
    # ------------------------------------------------------------------ #

    def _render_report(
        self,
        overall: Sequence[OverallAggregate],
        cells: Sequence[CellAggregate],
        samples: Sequence[SampleResult],
        summary: Dict[str, Any],
    ) -> str:
        lines: List[str] = []
        lines.append(f"# Run `{self.run_id}` 性能测试摘要\n")
        perf = summary.get("perf") or {}
        run_cfg = summary.get("run") or {}
        lines.append("## 运行配置\n")
        lines.append(
            f"- 模式：性能测试（逐样本）；样本数/数据集："
            f"{perf.get('samples_per_dataset') if perf.get('samples_per_dataset') is not None else '全部'}"
            f"；生成长度：{perf.get('max_tokens')}；重复：{run_cfg.get('repeats')}"
        )
        guard = perf.get("decode_guard")
        lines.append(
            f"- 解码窗口保障（decode guard）：{'开启' if guard else '关闭' if guard is False else '未知'}"
            "（性能模式追加固定长输出指令，使每个样本都能测 TG TPS；响应不判分，"
            "指令原文见 records.jsonl 的 request.messages）"
        )
        lines.append(
            f"- 上下文筛选：{describe_windows(perf.get('context_windows') or [])}；"
            f"模型级预热（丢弃）：{run_cfg.get('warmup_per_model', 0)} 次/模型；"
            f"per-cell 预热：{run_cfg.get('warmup_requests')} 次/cell"
        )
        lines.append(
            f"- 缓存模式：{run_cfg.get('cache_mode')}；思考模式：{run_cfg.get('thinking_mode')}；"
            f"记录：{summary.get('n_records')}；样本：{summary.get('n_samples')}；"
            f"测试单元（cell）：{summary.get('n_cells')}"
        )
        lines.append(f"- 耗时：{summary.get('duration_s', '-')}s\n")

        lines.append("## 环境\n")
        hw = self._hardware or {}
        omlx = self._omlx or {}
        lines.append(
            f"- 硬件：{hw.get('chip_name') or hw.get('chip')} "
            f"{hw.get('chip_variant') or ''}；内存 {hw.get('memory_gb')} GB；"
            f"GPU {hw.get('gpu_cores')} 核"
        )
        lines.append(
            f"- 系统：{hw.get('os_version')}；oMLX：{omlx.get('omlx_version') or hw.get('omlx_version')}"
        )
        flags = omlx.get("feature_flags") or []
        lines.append(
            f"- 加速特性（应为空 = 默认配置）：{flags if flags else '无'}"
        )
        for model_id, info in (omlx.get("models") or {}).items():
            settings = info.get("settings") or {}
            memory = info.get("memory") or {}
            memory_note = ""
            if memory:
                base = memory.get("baseline_gb")
                idle = memory.get("idle_gb")
                load_gb = memory.get("load_gb")
                if base is None:
                    memory_note = (
                        f"；进程内存：模型在运行前已加载，加载后 {idle} GB"
                    )
                else:
                    note = f"；进程内存基线 {base} GB → 加载后 {idle} GB"
                    if load_gb is not None:
                        note += f"（加载增量 {load_gb} GB，含引擎缓冲）"
                    memory_note = note
            lines.append(
                f"  - `{model_id}`：engine={info.get('engine_type')} "
                f"quant={info.get('quantization')} "
                f"moe_offload={settings.get('moe_expert_offload_enabled')}"
                f"{memory_note}"
            )
        lines.append("")

        lines.append("## 整体结果（按上下文档位分别平均）\n")
        lines.append(
            "> 1K/8K/32K 的 TTFT/预填充相差一个数量级，因此**不跨档位合并**；"
            "每行是一个模型的一个档位。TG TPS 后括号为参与统计的样本数/有效样本数"
            "（`completion_tokens < 16` 的样本不参与）。\n"
        )
        header = ["模型", "档位", "数据集", "样本", "有效", "错误", "触顶(max_tokens)"]
        header += [label for _, label in _REPORT_METRICS]
        header += ["峰值内存全局最大(GB)", "状态"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for entry in overall:
            row = [
                entry.model,
                _bucket_label(entry.context_length),
                str(entry.n_datasets),
                str(entry.n_samples),
                str(entry.n_valid),
                str(entry.n_error),
                str(entry.n_length_capped),
            ]
            for metric, _ in _REPORT_METRICS:
                dist = entry.perf.get(metric)
                value = _num(dist.mean if dist else None)
                if metric == "gen_tps" and dist is not None and dist.n < entry.n_valid:
                    value = f"{value} ({dist.n}/{entry.n_valid})"
                row.append(value)
            row.append(_num(entry.peak_memory_gb_max))
            row.append("unstable" if entry.unstable else "ok")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

        lines.append("## 数据集结果\n")
        header = [
            "模型",
            "场景",
            "数据集",
            "档位",
            "有效/样本",
            "触顶",
            "TTFT(ms)",
            "TG TPS",
            "峰值内存(GB)",
            "输入token",
        ]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for cell in cells:
            ttft = cell.perf.get("ttft_ms")
            tps = cell.perf.get("gen_tps")
            rss = cell.perf.get("peak_memory_gb")
            ctx = cell.perf.get("input_context_tokens")
            lines.append(
                "| {model} | {scenario} | {dataset} | {bucket} | {valid}/{total} | {capped} | {ttft} | {tps} | {rss} | {ctx} |".format(
                    model=cell.model,
                    scenario=cell.scenario,
                    dataset=cell.dataset,
                    bucket=_bucket_label(cell.context_length),
                    valid=cell.n_valid,
                    total=cell.n_samples * max(cell.n_repeats, 1),
                    capped=cell.n_length_capped,
                    ttft=_num(ttft.mean if ttft else None),
                    tps=(
                        f"{_num(tps.mean if tps else None)}"
                        + (
                            f" ({tps.n}/{cell.n_valid})"
                            if tps is not None and tps.n < cell.n_valid
                            else ""
                        )
                    ),
                    rss=_num(rss.max if rss else None),
                    ctx=_num(ctx.mean if ctx else None),
                )
            )
        lines.append("")

        lines.append("## 逐样本结果\n")
        header = [
            "模型",
            "数据集",
            "档位",
            "样本",
            "r",
            "输入token",
            "输入字符",
            "TTFT(ms)",
            "TG TPS",
            "峰值内存(GB)",
            "状态",
        ]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for row in samples:
            lines.append(
                "| {model} | {dataset} | {bucket} | {sample} | {rep} | {ctx_tok} | {ctx_chars} | "
                "{ttft} | {tps} | {rss} | {status}{trunc} |".format(
                    model=row.model,
                    dataset=row.dataset,
                    bucket=_bucket_label(row.context_length),
                    sample=row.sample_id,
                    rep=row.repeat,
                    ctx_tok=row.input_context_tokens if row.input_context_tokens is not None else "-",
                    ctx_chars=row.input_context_chars if row.input_context_chars is not None else "-",
                    ttft=_num(row.ttft_ms),
                    tps=_num(row.gen_tps),
                    rss=_num(row.peak_memory_gb),
                    status=row.status,
                    trunc=",truncated" if row.truncated else "",
                )
            )
        lines.append("")
        lines.append("> 指标说明：TTFT 为客户端首 token 延迟；TG TPS 为解码窗口内 token/s")
        lines.append(
            "> （`completion_tokens < 16` 时为 null）。性能模式默认开启 decode guard，"
            "在原始样本后追加长输出指令，避免以 `Answer:` 结尾的样本 1 token 结束；"
            "若仍为 null，括号里的 n/有效数给出真实覆盖，不做插补。"
        )
        lines.append(
            "> 输出触顶（`finish_reason=length`，达到 `perf.max_tokens`）**不是丢弃**："
            "样本照常进入 TTFT/内存/吞吐聚合，`gen_tps`/`tpot_ms` 用实际生成的 "
            "`completion_tokens` 计算到停止前；仅 `completion_tokens < 16` 的样本"
            "不参与 TG TPS 统计（TTFT/内存等仍统计）。"
        )
        lines.append(
            "> 峰值内存为 oMLX 服务进程树 "
            "phys_footprint 峰值（GiB，含 Metal 分配）。注意：oMLX 自带 benchmark 的 "
            "PeakMem 是 `mlx.core.get_peak_memory()` 的 Metal 峰值，通常比本值低 "
            "约 0.4–0.9 GB，两者不可逐位比较。"
        )
        lines.append("> 冷缓存模式下每次请求前均清理热缓存与 SSD 缓存；污染记录剔除聚合但保留在 records.jsonl。")
        lines.append("> 输入 token/字符为每个样本实际发送的上下文大小；超窗样本会标记 truncated。")
        return "\n".join(lines) + "\n"


def _num(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def _bucket_label(context_length: Optional[int]) -> str:
    """Render a declared context bucket (1024/8192/32768) as ``1K``/``8K``/``32K``."""
    return f"{context_length // 1024}K" if context_length else "native"
