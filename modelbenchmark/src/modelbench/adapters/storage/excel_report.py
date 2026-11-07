"""Excel result analysis for a completed batch.

Directory layout::

    output/result_analysis/<batch_id>_<method>/
      summary.xlsx     # sheet1 hardware, sheet2 per-model averages, sheet3 details
      batch.json       # batch metadata copy
      README.md        # sheet/column description

The workbook is generated automatically by :class:`BatchRunner` after a batch
finishes, and can be regenerated for existing batches with
``./modelbench export-analysis``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from modelbench.adapters.storage.catalog import load_index
from modelbench.adapters.storage.writer import sanitize_name

#: Excel cell limit is 32767 characters; keep a safety margin.
_MAX_CELL_CHARS = 30_000


def _sanitize_excel_text(text: str) -> str:
    """Drop ASCII control characters and neutralise formula payloads.

    openpyxl raises ``IllegalCharacterError`` when a cell contains e.g. BEL
    (``\\x07``) or VT (``\\x0b``), which real dataset/prompt text can carry
    (observed: dureader/spec prompts).  ``\\t\\n\\r`` are legal and kept.

    Model output is untrusted text: a leading ``=``/``+``/``-``/``@`` would be
    interpreted by Excel as a formula (CSV/Excel injection).  Prefixing with an
    apostrophe forces a literal string while Excel hides the apostrophe.
    """
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

    cleaned = ILLEGAL_CHARACTERS_RE.sub("", str(text))
    if cleaned[:1] in ("=", "+", "-", "@"):
        cleaned = "'" + cleaned
    return cleaned


def _clip(value: Any, limit: int = _MAX_CELL_CHARS) -> str:
    text = "" if value is None else _sanitize_excel_text(str(value))
    if len(text) <= limit:
        return text
    head = (limit - 40) // 2
    tail = limit - 40 - head
    return (
        text[:head]
        + f"\n...[内容已截断，原长度 {len(text)} 字符]...\n"
        + text[-tail:]
    )


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _request_input_text(request: Dict[str, Any]) -> str:
    messages = request.get("messages") or []
    parts: List[str] = []
    for message in messages:
        role = message.get("role") or "user"
        content = message.get("content")
        if content:
            parts.append(f"[{role}]\n{content}")
        tool_calls = message.get("tool_calls")
        if tool_calls:
            parts.append(f"[{role}:tool_calls]\n{json.dumps(tool_calls, ensure_ascii=False)}")
    return "\n\n".join(parts)


def _response_output_text(response: Dict[str, Any]) -> str:
    text = response.get("text") or ""
    if text:
        return text
    calls = response.get("tool_calls") or []
    if calls:
        return "\n".join(
            f"tool_call: {call.get('name')}({call.get('arguments')})"
            for call in calls
        )
    finish = response.get("finish_reason")
    return f"(text empty; finish_reason={finish})"


def _iter_records(run_dir: Path) -> Iterator[Dict[str, Any]]:
    path = run_dir / "records.jsonl"
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _number(value: Any) -> Any:
    """Return a number when possible so Excel can sort/format it."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return value
        return int(number) if number.is_integer() else number
    return value


# --------------------------------------------------------------------------- #
# workbook building
# --------------------------------------------------------------------------- #

SHEET_HARDWARE = "硬件配置"
SHEET_AVERAGES = "模型性能平均"
SHEET_DETAILS = "详细数据"


def _header(ws: Any, headers: List[str]) -> None:
    from openpyxl.styles import Alignment, Font, PatternFill

    ws.append(headers)
    fill = PatternFill("solid", fgColor="DCE6F1")
    font = Font(bold=True, color="1B3A5C")
    for cell in ws[1]:
        cell.fill = fill
        cell.font = font
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    ws.row_dimensions[1].height = 28


def _set_widths(ws: Any, widths: Dict[int, float]) -> None:
    from openpyxl.utils import get_column_letter

    for index, width in widths.items():
        ws.column_dimensions[get_column_letter(index)].width = width


def _flatten_settings(settings: Dict[str, Any], prefix: str = "") -> List[tuple[str, str]]:
    rows: List[tuple[str, str]] = []
    for key, value in (settings or {}).items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            rows.extend(_flatten_settings(value, name))
        else:
            rows.append((name, _clip(json.dumps(value, ensure_ascii=False) if isinstance(value, (list, tuple)) else value, 500)))
    return rows


def _sheet_hardware(wb: Any, batch: Dict[str, Any]) -> None:
    ws = wb.create_sheet(SHEET_HARDWARE)
    ws.append(["配置项", "值"])
    rows: List[tuple[Any, Any]] = [
        ("批次 ID", batch.get("batch_id")),
        ("测试方法", batch.get("method")),
        ("创建时间", batch.get("created_at")),
        ("模型列表", ", ".join(batch.get("models_selected") or [])),
        ("模型数量", batch.get("n_models_tested")),
        ("总耗时(s)", batch.get("duration_s")),
        ("总样本数", sum(int(item.get("n_samples") or 0) for item in batch.get("model_runs") or [])),
    ]
    first_run = (batch.get("model_runs") or [{}])[0]
    run_dir = Path(str(first_run.get("run_dir") or ""))
    hardware = _load_json(run_dir / "hardware.json")
    omlx = _load_json(run_dir / "omlx.json")
    manifest = _load_json(run_dir / "manifest.json")
    upload = hardware.get("upload_fields") or {}
    memory = hardware.get("memory") or {}
    cpu = hardware.get("cpu") or {}
    rows += [
        ("芯片", upload.get("chip_name") or hardware.get("chip")),
        ("芯片变体", upload.get("chip_variant")),
        ("内存(GB)", upload.get("memory_gb") or memory.get("total_gb")),
        ("内存可用(GB)", memory.get("available_gb")),
        ("内存已用(GB)", memory.get("used_gb")),
        ("内存使用率(%)", memory.get("percent")),
        ("Swap 已用(GB)", memory.get("swap_used_gb")),
        ("GPU 核心数", upload.get("gpu_cores")),
        ("操作系统", upload.get("os_version")),
        ("oMLX 版本", upload.get("omlx_version") or omlx.get("omlx_version")),
        ("CPU 逻辑/物理核心", f"{cpu.get('logical_cores')}/{cpu.get('physical_cores')}"),
        ("主机名", hardware.get("hostname")),
        ("架构", hardware.get("arch")),
        ("Python", (hardware.get("python") or {}).get("version")),
        ("git commit", manifest.get("git_commit")),
    ]
    perf = (manifest.get("config_snapshot") or {}).get("perf") or {}
    run_cfg = (manifest.get("config_snapshot") or {}).get("run") or {}
    rows += [
        ("样本数/数据集", perf.get("samples_per_dataset") if perf.get("samples_per_dataset") is not None else "全部"),
        ("max_tokens", perf.get("max_tokens")),
        ("decode_guard（长输出保障）", perf.get("decode_guard")),
        ("repeats", run_cfg.get("repeats")),
        ("cache_mode", run_cfg.get("cache_mode")),
        ("thinking_mode", run_cfg.get("thinking_mode")),
        ("加速 feature flags", ", ".join(omlx.get("feature_flags") or []) or "无（默认配置）"),
    ]
    for key, value in rows:
        ws.append([key, _clip(value, 500)])
    # Model configuration block
    ws.append([])
    ws.append(["模型配置", "值"])
    for model_id, info in (omlx.get("models") or {}).items():
        ws.append([f"{model_id} · engine", info.get("engine_type")])
        ws.append([f"{model_id} · quantization", info.get("quantization")])
        ws.append([f"{model_id} · context", info.get("context_length")])
        ws.append([f"{model_id} · settings", _clip(json.dumps(info.get("settings") or {}, ensure_ascii=False), 2000)])
        ws.append([f"{model_id} · feature_flags", ", ".join(
            flag.get("key", str(flag)) for flag in (info.get("feature_flags") or [])
        ) or "无"])
    if not (omlx.get("models") or {}):
        ws.append(["模型配置", "无（缺少 omlx.json）"])
    # Global settings (important sections only)
    globals_ = omlx.get("global_settings") or {}
    for section in ("memory", "scheduler", "cache", "sampling"):
        for key, value in _flatten_settings(globals_.get(section) or {}, section):
            ws.append([f"oMLX.{key}", _clip(value, 500)])
    _set_widths(ws, {1: 40, 2: 90})
    ws.freeze_panes = "A2"


def _sheet_averages(wb: Any, batch: Dict[str, Any]) -> None:
    ws = wb.create_sheet(SHEET_AVERAGES)
    headers = [
        "测试模型", "模型ID", "上下文档位", "状态", "数据集数", "样本数", "有效数", "错误数",
        "触顶数(max_tokens)", "TTFT均值(ms)", "TTFT中位数(ms)", "TTFT P95(ms)",
        "TG TPS均值", "TG TPS有效数", "TPOT均值(ms)", "E2E均值(ms)", "Prefill TPS均值",
        "峰值内存均值(GB)", "峰值内存最大(GB)", "峰值CPU均值(%)",
        "输入token均值", "输入字符均值", "输出token均值", "耗时(s)",
        "跳过数", "内存错误数",
    ]
    _header(ws, headers)
    for run in batch.get("model_runs") or []:
        summary = _load_json(Path(str(run.get("run_dir") or "")) / "summary.json")
        overall_rows = summary.get("overall") or []
        skipped = summary.get("skipped") or []
        status = "ok"
        if summary.get("cancelled"):
            status = "cancelled"
        elif not overall_rows:
            status = "skipped/failed"
        # Schema 2.1: one row per (model, context_length); never collapse 1K/8K/32K.
        # A run with no valid records still gets one status row (per model).
        for overall in (overall_rows or [{}]):
            perf = overall.get("perf") or {}
            def mean(key: str) -> Any:
                dist = perf.get(key) or {}
                return dist.get("mean")
            def median(key: str) -> Any:
                dist = perf.get(key) or {}
                return dist.get("median")
            def p95(key: str) -> Any:
                dist = perf.get(key) or {}
                return dist.get("p95")
            def count(key: str) -> Any:
                dist = perf.get(key) or {}
                return dist.get("n")
            row = [
                run.get("model"),
                run.get("model_id"),
                _bucket_label(overall.get("context_length")),
                status,
                overall.get("n_datasets"),
                overall.get("n_samples"),
                overall.get("n_valid"),
                overall.get("n_error"),
                overall.get("n_length_capped"),
                mean("ttft_ms"), median("ttft_ms"), p95("ttft_ms"),
                mean("gen_tps"), count("gen_tps"), mean("tpot_ms"), mean("e2e_ms"), mean("prefill_tps"),
                mean("peak_memory_gb"), (perf.get("peak_memory_gb") or {}).get("max"),
                mean("peak_cpu_pct"),
                mean("input_context_tokens"), mean("input_context_chars"), mean("completion_tokens"),
                run.get("duration_s"),
                len(skipped), run.get("memory_errors") or 0,
            ]
            ws.append([_number(value) for value in row])
    if ws.max_row == 1:
        ws.append(["无模型结果"])
    _set_widths(ws, {
        1: 28, 2: 28, 3: 12, 4: 12, 5: 10, 6: 9, 7: 9, 8: 9, 9: 16,
        10: 14, 11: 14, 12: 12, 13: 12, 14: 12, 15: 12, 16: 12, 17: 14,
        18: 15, 19: 15, 20: 14, 21: 13, 22: 13, 23: 12, 24: 10, 25: 9, 26: 11,
    })


def _bucket_label(context_length: Any) -> str:
    """Render a declared context bucket (1024/8192/32768) as 1K/8K/32K."""
    try:
        value = int(context_length)
    except (TypeError, ValueError):
        return "全部/未标注"
    return f"{value // 1024}K" if value > 0 else "全部/未标注"


def _sheet_details(wb: Any, batch: Dict[str, Any]) -> None:
    ws = wb.create_sheet(SHEET_DETAILS)
    headers = [
        "测试模型", "模型ID", "样本来源", "样本所属场景", "样本ID", "重复",
        "API 输入内容", "输入 prompt token 大小", "输入字符数",
        "输出结果", "输出 token 大小", "输出字符数", "峰值内存占用(GB)",
        "TTFT(ms)", "TG TPS", "TPOT(ms)", "E2E(ms)", "Prefill TPS", "Total TPS",
        "峰值CPU(%)", "均值CPU(%)", "cached tokens", "finish_reason（length=触顶）", "状态",
        "上下文截断", "错误/备注", "时间戳",
    ]
    _header(ws, headers)
    row_count = 0
    for run in batch.get("model_runs") or []:
        run_dir = Path(str(run.get("run_dir") or ""))
        for record in _iter_records(run_dir):
            if record.get("phase") != "measured":
                continue
            request = record.get("request") or {}
            response = record.get("response") or {}
            metrics = record.get("metrics") or {}
            input_tokens = metrics.get("input_context_tokens")
            if input_tokens is None:
                input_tokens = metrics.get("prompt_tokens") or request.get("actual_prompt_tokens")
            output_text = _response_output_text(response)
            input_text = _request_input_text(request)
            ws.append([
                run.get("model"),
                run.get("model_id"),
                record.get("dataset"),
                record.get("scenario"),
                record.get("sample_id"),
                record.get("repeat"),
                _clip(input_text),
                _number(input_tokens),
                _number(metrics.get("input_context_chars")),
                _clip(output_text),
                _number(metrics.get("completion_tokens")),
                len(output_text),
                _number(metrics.get("peak_memory_gb")),
                _number(metrics.get("ttft_ms")),
                _number(metrics.get("gen_tps")),
                _number(metrics.get("tpot_ms")),
                _number(metrics.get("e2e_ms")),
                _number(metrics.get("prefill_tps")),
                _number(metrics.get("total_tps")),
                _number(metrics.get("peak_cpu_pct")),
                _number(metrics.get("mean_cpu_pct")),
                _number(metrics.get("cached_tokens")),
                response.get("finish_reason"),
                record.get("status"),
                bool(request.get("truncated")),
                _clip(record.get("error"), 1000),
                record.get("timestamp"),
            ])
            row_count += 1
    if row_count == 0:
        ws.append(["无逐样本记录"])
    _set_widths(ws, {
        1: 24, 2: 24, 3: 14, 4: 16, 5: 30, 6: 7,
        7: 70, 8: 14, 9: 12, 10: 55, 11: 12, 12: 11, 13: 14,
        14: 11, 15: 11, 16: 11, 17: 11, 18: 12, 19: 12,
        20: 11, 21: 11, 22: 12, 23: 14, 24: 12, 25: 10, 26: 40, 27: 24,
    })
    # Keep long text readable but avoid enormous row heights.
    from openpyxl.styles import Alignment

    for row in ws.iter_rows(min_row=2, min_col=7, max_col=10):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=False)


def _render_workbook(batch: Dict[str, Any]) -> Any:
    try:
        from openpyxl import Workbook
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise RuntimeError(
            "openpyxl is required for Excel result analysis; "
            "install it with `uv pip install openpyxl` or `pip install openpyxl`"
        ) from exc
    wb = Workbook()
    wb.remove(wb.active)
    _sheet_hardware(wb, batch)
    _sheet_averages(wb, batch)
    _sheet_details(wb, batch)
    wb.properties.title = f"Result analysis {batch.get('batch_id')} ({batch.get('method')})"
    wb.properties.creator = "modelbench"
    return wb


def analysis_dir_for(output_root: Path, batch: Dict[str, Any]) -> Path:
    parts = [
        sanitize_name(batch.get("batch_id") or "batch"),
        sanitize_name(batch.get("method") or "performance"),
    ]
    models = [
        run.get("model") or run.get("model_id")
        for run in (batch.get("model_runs") or [])
    ]
    model_part = "_".join(
        sanitize_name(str(model)) for model in models if model
    )
    if model_part:
        parts.append(model_part)
    return Path(output_root) / "result_analysis" / "_".join(parts)


def _sanitize_workbook(workbook: Any) -> None:
    """Belt-and-braces: strip illegal XML characters from every string cell."""
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

    for ws in workbook.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                if isinstance(cell.value, str) and ILLEGAL_CHARACTERS_RE.search(cell.value):
                    cell.value = ILLEGAL_CHARACTERS_RE.sub("", cell.value)


def write_batch_excel(output_root: Path, batch: Dict[str, Any]) -> Dict[str, Any]:
    """Generate ``summary.xlsx`` for one batch; returns the artifact paths."""
    directory = analysis_dir_for(output_root, batch)
    directory.mkdir(parents=True, exist_ok=True)
    workbook = _render_workbook(batch)
    _sanitize_workbook(workbook)
    workbook_path = directory / "summary.xlsx"
    tmp = workbook_path.with_suffix(".xlsx.tmp")
    workbook.save(tmp)
    os.replace(tmp, workbook_path)
    (directory / "batch.json").write_text(
        json.dumps(batch, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (directory / "README.md").write_text(
        _render_readme(batch, workbook_path), encoding="utf-8"
    )
    return {
        "dir": str(directory),
        "workbook": str(workbook_path),
        "sheets": [SHEET_HARDWARE, SHEET_AVERAGES, SHEET_DETAILS],
    }


def _render_readme(batch: Dict[str, Any], workbook_path: Path) -> str:
    return (
        f"# 结果分析：{batch.get('batch_id')} ({batch.get('method')})\n\n"
        f"- 工作簿：`{workbook_path.name}`\n"
        "- 页签 1 `硬件配置`：机器/系统/oMLX/模型生效配置与测试参数。\n"
        "- 页签 2 `模型性能平均`：每个模型的样本测试性能平均值。\n"
        "- 页签 3 `详细数据`：逐样本 API 输入、输入/输出 token 大小、输出结果、峰值内存与其它性能指标。\n\n"
        "> 长文本（API 输入/输出）超过 Excel 单元格上限时会做头尾截断并标注原长度。\n"
    )


# --------------------------------------------------------------------------- #
# existing batches / CLI
# --------------------------------------------------------------------------- #


def _batch_from_index(index: Dict[str, Any], batch_id: str) -> Optional[Dict[str, Any]]:
    for batch in index.get("batches") or []:
        if batch.get("batch_id") != batch_id:
            continue
        entries = {entry.get("run_id"): entry for entry in index.get("runs") or []}
        model_runs = []
        for item in batch.get("models") or []:
            entry = entries.get(item.get("run_id")) or {}
            model_runs.append(
                {
                    "model": item.get("model") or entry.get("model"),
                    "model_id": entry.get("model") or item.get("model"),
                    "run_id": item.get("run_id"),
                    "run_dir": entry.get("path"),
                    "summary_path": str(Path(str(entry.get("path") or "")) / "summary.json"),
                    "n_samples": item.get("n_samples"),
                    "duration_s": item.get("duration_s"),
                    "memory_errors": item.get("memory_errors") or 0,
                    "skipped": item.get("skipped") or [],
                }
            )
        return {
            "batch_id": batch_id,
            "method": batch.get("method") or "performance",
            "created_at": batch.get("created_at"),
            "models_selected": [run.get("model") for run in model_runs],
            "n_models_tested": len(model_runs),
            "duration_s": batch.get("duration_s"),
            "model_runs": model_runs,
        }
    return None


def _batch_for_regeneration(output_root: Path, batch_id: str) -> Optional[Dict[str, Any]]:
    """Prefer the authoritative per-batch metadata written by BatchRunner.

    ``output/batches/<id>.json`` keeps the model *labels* (aliases) used for the
    result_analysis directory name.  Rebuilding from ``index.json`` would use the
    canonical model id and create a second, differently-named directory.
    """
    meta_path = output_root / "batches" / f"{batch_id}.json"
    if meta_path.is_file():
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict) and payload.get("model_runs"):
            return payload
    index = load_index(output_root)
    if index is None:
        from modelbench.adapters.storage.catalog import rebuild_index

        index = rebuild_index(output_root)
    return _batch_from_index(index, batch_id)


def generate_for_batch(output_root: Path, batch_id: str) -> Dict[str, Any]:
    output_root = Path(output_root)
    batch = _batch_for_regeneration(output_root, batch_id)
    if batch is None:
        raise FileNotFoundError(f"batch not found: {batch_id}")
    return write_batch_excel(output_root, batch)


def generate_all(output_root: Path) -> List[Dict[str, Any]]:
    output_root = Path(output_root)
    index = load_index(output_root)
    if index is None:
        from modelbench.adapters.storage.catalog import rebuild_index

        index = rebuild_index(output_root)
    results: List[Dict[str, Any]] = []
    for batch in index.get("batches") or []:
        batch_id = batch.get("batch_id")
        if not batch_id:
            continue
        batch_obj = _batch_for_regeneration(output_root, batch_id)
        if batch_obj is None:
            continue
        results.append(write_batch_excel(output_root, batch_obj))
    return results
