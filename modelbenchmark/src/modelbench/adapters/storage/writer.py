"""Run bundle writer for schema 3.0.

Layout (one model run)::

    output/runs/<YYYY-MM>/<batch_id>/<method>/<model_slug>/
      run.json         run metadata + the single environment snapshot
      MANIFEST.json    file inventory with sha256 (written last)
      records.jsonl    authoritative per-request evidence (prompt by digest)
      samples.csv      per-sample table (projection)
      aggregates.json  overall + cells (projection)
      report.md        human summary (projection)
      logs/run.log     operational log

Prompt bodies are *not* copied into ``records.jsonl``: the writer stores the
messages once in the content-addressed blob pool and the record keeps
``request.prompt_sha256``.  This is what removes the previous ~32x duplication of
the same prompt text across models, batches and repeats.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from modelbench.adapters.storage import layout
from modelbench.adapters.storage.blobstore import PromptBlobStore, prompt_chars
from modelbench.domain.context_windows import describe_windows
from modelbench.domain.metrics import (
    aggregate_overall,
    aggregate_records,
    sample_results,
)
from modelbench.domain.metrics import isl_band
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

#: Field order of ``samples.csv`` — frozen (see the implementation contract).
SAMPLES_CSV_FIELDS = (
    "model",
    "model_id",
    "scenario",
    "dataset",
    "subtask",
    "context_length",
    "sample_id",
    "repeat",
    "phase",
    "status",
    "finish_reason",
    "length_capped",
    "input_context_tokens",
    "input_context_chars",
    "actual_prompt_tokens",
    "isl_band",
    "ttft_ms",
    "ttft_visible_ms",
    "tpot_ms",
    "gen_tps",
    "e2e_ms",
    "prefill_tps",
    "total_tps",
    "completion_tokens",
    "cached_tokens",
    "cache_mode",
    "cache_cleared",
    "cache_polluted",
    "peak_memory_gb",
    "peak_rss_mb",
    "peak_cpu_pct",
    "mean_cpu_pct",
    "prompt_sha256",
    "truncated",
    "error",
    "timestamp",
)


def sanitize_name(value: str) -> str:
    """Filesystem-safe, human-recognizable component."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "unnamed"


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_env_digest(entry: Dict[str, Any]) -> str:
    """Stable 16-hex key for a ``model_env`` block."""
    body = json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


class RunWriter:
    """Owns one model run directory under ``output/runs``."""

    def __init__(
        self,
        output_root: Path,
        run_id: Optional[str] = None,
        overwrite: bool = False,
        batch: Optional[Dict[str, Any]] = None,
        *,
        batch_id: Optional[str] = None,
        method: str = "performance",
        model_slug: Optional[str] = None,
        created_at: Optional[str] = None,
        model: Optional[Dict[str, Any]] = None,
    ):
        self.output_root = Path(output_root)
        self.method = sanitize_name(str(method or "performance"))
        self.batch = dict(batch or {})
        self.batch_id, self.model_slug = self._resolve_identity(
            run_id, batch_id, model_slug
        )
        self.created_at = created_at or _utc_now()
        self.run_id = layout.make_run_id(self.batch_id, self.method, self.model_slug)
        # Flat layout: the run directory name *is* the run id.
        self.run_dir = layout.run_dir(self.output_root, self.run_id)
        self.batch_dir = layout.batches_dir(self.output_root)
        if self.run_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"run directory already exists: {self.run_dir} "
                    "(choose another --run-id or pass overwrite=True)"
                )
            shutil.rmtree(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / layout.LOGS_DIR).mkdir(parents=True, exist_ok=True)
        self.batch_dir.mkdir(parents=True, exist_ok=True)

        self.blob_store = PromptBlobStore(self.output_root)
        self._records_path = self.run_dir / layout.RECORDS
        self._records_fh = None
        self._count = 0
        self._manifest: Optional[RunManifest] = None
        self._hardware: Dict[str, Any] = {}
        self._omlx: Dict[str, Any] = {}
        self._model = dict(model or {})

    # ------------------------------------------------------------------ #
    # identity
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_identity(
        run_id: Optional[str],
        batch_id: Optional[str],
        model_slug: Optional[str],
    ):
        """Accept either a schema-3 ``run_id`` or a free-form label.

        Production code passes ``batch_id``/``method``/``model_slug`` explicitly.
        Tests and scripts that only have a free-form label may keep passing
        ``run_id="something"``: it is then used as the batch id with a single
        default model slug so the flat ``runs/<batch_id>__method__slug`` layout
        still applies.
        """
        if run_id:
            text = str(run_id)
            if layout.RUN_ID_SEP in text:
                parsed_batch, _parsed_method, parsed_slug = layout.parse_run_id(text)
                return (
                    sanitize_name(batch_id or parsed_batch),
                    sanitize_name(model_slug or parsed_slug),
                )
            return sanitize_name(text), sanitize_name(model_slug or "model")
        if not batch_id:
            raise ValueError("RunWriter needs either run_id or batch_id")
        return sanitize_name(str(batch_id)), sanitize_name(model_slug or "model")

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / layout.LOGS_DIR

    @property
    def log_path(self) -> Path:
        return self.logs_dir / layout.LOG_NAME

    # ------------------------------------------------------------------ #
    # initialization
    # ------------------------------------------------------------------ #

    def initialize(
        self,
        manifest: RunManifest,
        hardware: Optional[Dict[str, Any]] = None,
        models: Optional[Dict[str, Any]] = None,
        omlx: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist ``run.json``.

        ``hardware``/``models``/``omlx`` are accepted for backwards
        compatibility with callers that still pass the old separate snapshots;
        they are folded into ``manifest.env`` when it is empty.
        """
        env = dict(manifest.env or {})
        if hardware and not env.get("hardware"):
            env["hardware"] = hardware
        if omlx and not env.get("omlx"):
            env["omlx"] = omlx
        if models and not env.get("model_env"):
            env["model_env"] = self._model_env_from_models(models, manifest)
        manifest.env = env
        self._hardware = env.get("hardware") or {}
        self._omlx = env.get("omlx") or {}
        self._model = dict(manifest.model or self._model or {})
        self._manifest = manifest
        self.write_run_json()

    @staticmethod
    def _model_env_from_models(
        models: Dict[str, Any], manifest: RunManifest
    ) -> Dict[str, Dict[str, Any]]:
        """Best-effort ``model_env`` map for callers passing the old snapshot."""
        selected = {
            str(item.get("alias") or item.get("id"))
            for item in (models.get("selected") or [])
            if isinstance(item, dict)
        }
        candidates = models.get("server") or models.get("models") or {}
        if isinstance(candidates, dict):
            candidates = list(candidates.values())
        blocks: Dict[str, Dict[str, Any]] = {}
        for entry in candidates or []:
            if not isinstance(entry, dict):
                continue
            label = str(entry.get("id") or entry.get("alias") or "")
            if selected and label not in selected:
                continue
            blocks[model_env_digest(entry)] = entry
        if not blocks:
            blocks[model_env_digest({})] = {}
        return blocks

    def register_model_env(self, snapshot: Dict[str, Any]) -> str:
        """Record a model-configuration snapshot once and return its digest.

        The digest is what ``SampleRecord.model_env_ref`` points at, so the
        snapshot is stored once per run instead of being repeated on every
        record.  A mid-run reload legitimately adds a second block.
        """
        digest = model_env_digest(snapshot or {})
        if self._manifest is not None:
            env = dict(self._manifest.env or {})
            blocks = dict(env.get("model_env") or {})
            if blocks.get(digest) != snapshot:
                blocks[digest] = dict(snapshot or {})
                env["model_env"] = blocks
                self._manifest.env = env
                self.write_run_json()
        return digest

    def write_run_json(self) -> None:
        if self._manifest is None:
            return
        self._manifest.run_id = self.run_id
        self._manifest.batch_id = self.batch_id
        self._manifest.method = self.method
        self._manifest.schema_version = SCHEMA_VERSION
        self._write_json(layout.RUN_JSON, self._manifest.model_dump(mode="json"))

    def write_omlx(self, omlx: Dict[str, Any]) -> None:
        """Update the oMLX portion of the environment snapshot in place."""
        if omlx:
            self._omlx = omlx
        if self._manifest is not None:
            env = dict(self._manifest.env or {})
            env["omlx"] = self._omlx
            self._manifest.env = env
            self.write_run_json()

    def _write_json(self, name: str, payload: Any) -> None:
        path = self.run_dir / name
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)

    # ------------------------------------------------------------------ #
    # per-record
    # ------------------------------------------------------------------ #

    def write_record(
        self,
        record: SampleRecord,
        prompt_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append one record, storing its prompt body in the blob pool first."""
        request = record.request
        if prompt_payload is not None:
            digest = self.blob_store.put(prompt_payload)
            request.prompt_sha256 = digest
            if request.prompt_chars is None:
                request.prompt_chars = int(prompt_payload.get("chars") or 0)
        elif not request.prompt_sha256:
            if not request.messages:
                raise ValueError(
                    "record has neither request.messages nor request.prompt_sha256"
                )
            request.prompt_sha256 = self.blob_store.put_messages(
                request.messages,
                tools=request.tools,
                chat_template_kwargs=request.chat_template_kwargs,
            )
            if request.prompt_chars is None:
                request.prompt_chars = prompt_chars(request.messages)

        payload = record.model_dump(
            mode="json",
            exclude={"request": {"messages", "tools"}},
        )
        if self._records_fh is None:
            self._records_fh = self._records_path.open("a", encoding="utf-8")
        self._records_fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._records_fh.flush()
        self._count += 1

    def close(self) -> None:
        if self._records_fh is not None:
            self._records_fh.close()
            self._records_fh = None

    # ------------------------------------------------------------------ #
    # finalize
    # ------------------------------------------------------------------ #

    def finalize(
        self,
        records: List[SampleRecord],
        extra: Optional[Dict[str, Any]] = None,
        perf: Optional[Dict[str, Any]] = None,
        run_config: Optional[Dict[str, Any]] = None,
        omlx: Optional[Dict[str, Any]] = None,
        outcome: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.close()
        if omlx is not None:
            self._omlx = omlx
        extra = dict(extra or {})

        cells = aggregate_records(records)
        overall = aggregate_overall(records)
        samples = sample_results(records)
        aggregates = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "generated_at": _utc_now(),
            "n_records": len(records),
            "n_samples": len(samples),
            "n_cells": len(cells),
            "overall": [entry.model_dump(mode="json") for entry in overall],
            "cells": [cell.model_dump(mode="json") for cell in cells],
        }
        self._write_json(layout.AGGREGATES_JSON, aggregates)
        self._write_samples_csv(samples)

        merged_outcome = self._finalize_outcome(records, samples, extra, outcome)
        merged_config = dict((self._manifest.config if self._manifest else {}) or {})
        if perf:
            merged_config["perf"] = perf
        if run_config:
            merged_config["run"] = run_config
        if self._manifest is not None:
            self._manifest.outcome = merged_outcome
            self._manifest.config = merged_config
            if self._hardware:
                env = dict(self._manifest.env or {})
                env.setdefault("hardware", self._hardware)
                env["omlx"] = self._omlx or env.get("omlx") or {}
                self._manifest.env = env
            self.write_run_json()

        report_summary = {
            "perf": merged_config.get("perf") or {},
            "run": merged_config.get("run") or {},
            "n_records": aggregates["n_records"],
            "n_samples": aggregates["n_samples"],
            "n_cells": aggregates["n_cells"],
            "duration_s": merged_outcome.get("duration_s"),
            "n_isl_in_band": sum(entry.n_isl_in_band for entry in overall),
            "n_isl_out_of_band": sum(entry.n_isl_out_of_band for entry in overall),
        }
        (self.run_dir / layout.REPORT_MD).write_text(
            self._render_report(overall, cells, samples, report_summary),
            encoding="utf-8",
        )
        manifest = self._write_manifest()
        self._update_catalog()
        result: Dict[str, Any] = {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "run_json": str(self.run_dir / layout.RUN_JSON),
            "manifest_json": str(self.run_dir / layout.MANIFEST_JSON),
            "records_jsonl": str(self._records_path),
            "samples_csv": str(self.run_dir / layout.SAMPLES_CSV),
            "aggregates_json": str(self.run_dir / layout.AGGREGATES_JSON),
            "report_md": str(self.run_dir / layout.REPORT_MD),
            "n_records": len(records),
            "n_samples": len(samples),
            "n_cells": len(cells),
            "n_blobs": manifest["blob_refs"]["prompts"],
            "duration_s": merged_outcome.get("duration_s"),
            "bytes_total": manifest["totals"]["bytes"],
        }
        result.update({key: value for key, value in extra.items() if key not in result})
        return result

    def _finalize_outcome(
        self,
        records: Sequence[SampleRecord],
        samples: Sequence[SampleResult],
        extra: Dict[str, Any],
        outcome: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        measured = [r for r in records if r.phase.value == "measured"]
        ok_measured = [r for r in measured if r.status.value == "ok"]
        errored = [r for r in records if r.status.value in ("error", "timeout")]
        base = dict(self._manifest.outcome if self._manifest else {}) or {}
        skipped = list(extra.get("skipped") or base.get("skipped") or [])
        cancelled = bool(extra.get("cancelled", base.get("cancelled", False)))
        # A run always ends in a terminal state: ``running`` is the initial
        # value written when the orchestrator starts and never survives finalize.
        if cancelled:
            status = "cancelled"
        elif errored and not ok_measured:
            status = "error"
        elif errored:
            status = "partial"
        else:
            status = "ok"
        # ``n_selected`` = samples the plan picked.  The orchestrator does not
        # track it separately, so derive it from the traceability identity:
        # every selected sample is either measured-ok or explicitly skipped
        # (AGENTS.md invariant 13).
        # ``n_measured`` counts ALL measured-phase records (error/timeout
        # included) so failed attempts stay in the accounting; ``n_ok`` is the
        # ok subset used by ratios such as test-plan AC-2 (ok/n_measured).
        # ``n_selected`` = samples the plan picked; every selected sample is
        # measured (ok/error/timeout) or explicitly skipped (invariant 13).
        n_measured = len(measured)
        n_ok = len(ok_measured)
        n_selected = int(extra.get("n_selected") or 0) or (n_measured + len(skipped))
        base.update(
            {
                "status": status,
                "cancelled": cancelled,
                "memory_errors": int(
                    extra.get("memory_errors", base.get("memory_errors", 0)) or 0
                ),
                "n_selected": n_selected,
                "n_measured": n_measured,
                "n_ok": n_ok,
                "n_records": len(records),
                "n_samples": len(samples),
                "n_cells": len(
                    {
                        (r.model, r.scenario, r.dataset, r.context_length)
                        for r in measured
                    }
                ),
                # Measurement-validity counters: visible instead of being
                # buried in per-sample records.
                "n_gen_tps_invalid": sum(
                    1 for r in ok_measured if r.metrics.gen_tps_invalid
                ),
                "n_isl_in_band": sum(1 for r in ok_measured if isl_band(r)),
                "n_isl_out_of_band": sum(
                    1 for r in ok_measured if not isl_band(r)
                ),
                "skipped": skipped,
                "warnings": list(extra.get("warnings") or base.get("warnings") or []),
                "duration_s": extra.get("duration_s", base.get("duration_s")),
            }
        )
        if outcome:
            base.update(outcome)
        return base

    def _write_samples_csv(self, samples: Sequence[SampleResult]) -> None:
        rows = [row.model_dump(mode="json") for row in samples]
        with (self.run_dir / layout.SAMPLES_CSV).open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=SAMPLES_CSV_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in SAMPLES_CSV_FIELDS})

    def _write_manifest(self) -> Dict[str, Any]:
        entries: List[Dict[str, Any]] = []
        for name in (
            layout.RUN_JSON,
            layout.RECORDS,
            layout.SAMPLES_CSV,
            layout.AGGREGATES_JSON,
            layout.REPORT_MD,
            f"{layout.LOGS_DIR}/{layout.LOG_NAME}",
        ):
            path = self.run_dir / name
            if not path.is_file():
                continue
            entries.append(
                {
                    "path": name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
        entries.sort(key=lambda item: item["path"])
        shas = set(_iter_record_prompt_shas(self._records_path))
        missing = self.blob_store.missing(shas)
        pool = self.blob_store.stats()
        referenced_bytes = 0
        for sha in shas:
            try:
                referenced_bytes += self.blob_store.path_for(sha).stat().st_size
            except OSError:
                pass
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "generated_at": _utc_now(),
            "algorithm": "sha256",
            "files": entries,
            "totals": {
                "files": len(entries),
                "bytes": sum(item["bytes"] for item in entries),
            },
            "blob_refs": {
                # Bytes this run actually references (shared blobs are shared, so
                # this is NOT an addition to the on-disk total).
                "prompts": len(shas),
                "bytes": referenced_bytes,
                # Whole-pool size, useful for "how big is output/".
                "pool_bytes": pool["bytes"],
                "missing": len(missing),
            },
        }
        self._write_json(layout.MANIFEST_JSON, manifest)
        return manifest

    def _update_catalog(self) -> None:
        """Refresh ``output/index.json``; the index is derived, so stay tolerant."""
        try:
            from modelbench.adapters.storage import catalog
        except Exception as exc:  # pragma: no cover - import cycle/partial install
            print(
                f"WARNING: catalog unavailable ({exc}); index not updated",
                file=sys.stderr,
            )
            return
        upsert = getattr(catalog, "upsert_run", None)
        if upsert is None:  # pragma: no cover - during staggered rollout
            print("WARNING: catalog.upsert_run missing; index not updated", file=sys.stderr)
            return
        try:
            upsert(self.output_root, self.run_dir)
        except TypeError:  # pragma: no cover - legacy two/three-arg signature
            try:
                upsert(self.output_root, self.run_dir, self._manifest)
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: index update failed ({exc})", file=sys.stderr)
        except Exception as exc:  # pragma: no cover - index is rebuildable
            print(
                f"WARNING: index update failed ({exc}); run output is complete",
                file=sys.stderr,
            )

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
            "指令原文见 `output/blobs/prompts/` 中该记录 `request.prompt_sha256` 指向的 blob）"
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
        lines.append(f"- 耗时：{summary.get('duration_s', '-')}s")
        if summary.get("n_isl_out_of_band"):
            lines.append(
                f"- **ISL 带宽**：入带 {summary.get('n_isl_in_band', 0)} 个、"
                f"出带 {summary.get('n_isl_out_of_band', 0)} 个"
                "（出带样本仍保留在 records/samples.csv，但不参与平均值）"
            )
        lines.append("")

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
        lines.append(f"- 加速特性（应为空 = 默认配置）：{flags if flags else '无'}")
        for model_id, info in (omlx.get("models") or {}).items():
            settings = info.get("settings") or {}
            memory = info.get("memory") or {}
            memory_note = ""
            if memory:
                base = memory.get("baseline_gb")
                idle = memory.get("idle_gb")
                load_gb = memory.get("load_gb")
                if base is None:
                    memory_note = f"；进程内存：模型在运行前已加载，加载后 {idle} GB"
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
        header = [
            "模型",
            "档位",
            "数据集",
            "样本",
            "有效",
            "错误",
            "触顶(max_tokens)",
            "TG无效",
            "ISL入带",
            "ISL出带",
        ]
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
                str(entry.n_gen_tps_invalid),
                str(entry.n_isl_in_band),
                str(entry.n_isl_out_of_band),
            ]
            for metric, _ in _REPORT_METRICS:
                dist = entry.perf.get(metric)
                value = _num(dist.mean if dist else None)
                if (
                    metric == "gen_tps"
                    and dist is not None
                    and dist.n < entry.n_isl_in_band
                ):
                    value = f"{value} ({dist.n}/{entry.n_isl_in_band})"
                row.append(value)
            row.append(_num(entry.peak_memory_gb_max))
            status = "unstable" if entry.unstable else "ok"
            if entry.insufficient_samples:
                status += " insufficient_samples"
            row.append(status)
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
            "TG无效",
            "ISL入带",
            "ISL出带",
            "TTFT(ms)",
            "TG TPS",
            "峰值内存(GB)",
            "输入token",
            "状态",
        ]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for cell in cells:
            ttft = cell.perf.get("ttft_ms")
            tps = cell.perf.get("gen_tps")
            rss = cell.perf.get("peak_memory_gb")
            ctx = cell.perf.get("input_context_tokens")
            lines.append(
                "| {model} | {scenario} | {dataset} | {bucket} | {valid}/{total} | {capped} | {tginvalid} | {isl_in} | {isl_out} | {ttft} | {tps} | {rss} | {ctx} | {status} |".format(
                    model=cell.model,
                    scenario=cell.scenario,
                    dataset=cell.dataset,
                    bucket=_bucket_label(cell.context_length),
                    valid=cell.n_valid,
                    total=cell.n_samples * max(cell.n_repeats, 1),
                    capped=cell.n_length_capped,
                    tginvalid=cell.n_gen_tps_invalid,
                    isl_in=cell.n_isl_in_band,
                    isl_out=cell.n_isl_out_of_band,
                    ttft=_num(ttft.mean if ttft else None),
                    tps=(
                        f"{_num(tps.mean if tps else None)}"
                        + (
                            f" ({tps.n}/{cell.n_isl_in_band})"
                            if tps is not None and tps.n < cell.n_isl_in_band
                            else ""
                        )
                    ),
                    rss=_num(rss.max if rss else None),
                    ctx=_num(ctx.mean if ctx else None),
                    status=(
                        "insufficient_samples"
                        if cell.insufficient_samples
                        else "ok"
                    ),
                )
            )
        lines.append("")

        if any(getattr(cell, "insufficient_samples", False) for cell in cells):
            lines.append(
                "> `insufficient_samples`：入带样本 <8（默认每 cell 10 条）。"
                "`--samples <8` 的冒烟 run 必然触发，不代表数据异常；"
                "正式结论需要每 cell ≥8 条 actual 在 target ±10% 内的样本。\n"
            )
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
                    ctx_tok=row.input_context_tokens
                    if row.input_context_tokens is not None
                    else "-",
                    ctx_chars=row.input_context_chars
                    if row.input_context_chars is not None
                    else "-",
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
        lines.append(
            "> 冷缓存模式下每次请求前均清理热缓存与 SSD 缓存；"
            "污染记录剔除聚合但保留在 records.jsonl。"
        )
        lines.append(
            "> 输入 token/字符为每个样本实际发送的上下文大小；超窗样本会标记 truncated。"
        )
        lines.append(
            "> 目录结构（schema 3.0）：records.jsonl 为权威逐样本证据，prompt 正文按 "
            "`request.prompt_sha256` 存于 `output/blobs/prompts/`（全库一份），"
            "可用 `./modelbench rebuild` 从 records 重建本目录的派生文件。"
        )
        return "\n".join(lines) + "\n"


def _iter_record_prompt_shas(records_path: Path) -> Iterable[str]:
    if not records_path.is_file():
        return []
    shas: List[str] = []
    with records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            sha = ((payload.get("request") or {}).get("prompt_sha256")) or ""
            if sha:
                shas.append(sha)
    return shas


def _num(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def _bucket_label(context_length: Optional[int]) -> str:
    """Render a declared context bucket (1024/8192/32768) as ``1K``/``8K``/``32K``."""
    return f"{context_length // 1024}K" if context_length else "native"
