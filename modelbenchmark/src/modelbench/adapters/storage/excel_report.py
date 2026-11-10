"""Excel result analysis for a completed batch (schema 3.0).

The workbook is a *derived deliverable*: it lives in ``output/exports/`` and can
be deleted and regenerated at any time (``./modelbench export-analysis``); it is
no longer written next to the run bundles and no longer carries copies of
``batch.json`` / ``README.md``.

Data sources (the schema-3.0 tree)::

    output/
      runs/<run_id>/                                flat: dir name == run_id
        run.json                                    hardware / omlx / model env
        aggregates.json                             overall[] / cells[]
        records.jsonl                               per-sample evidence
      batches/<batch_id>.json                       batch metadata (one file)
      blobs/prompts/<sha[:2]>/<sha>.json            prompt bodies (content addressed)
      exports/<batch_id>.xlsx                       <- this module writes here

``run_id`` is ``<batch_id>__<method>__<model_slug>``; ``batches/<batch_id>.json``
lists its runs as ``{"run_id": ..., "run_path": "runs/<run_id>"}``.

``records.jsonl`` no longer embeds the prompt text; each record carries
``request.prompt_sha256`` and the body is resolved from
:class:`~modelbench.adapters.storage.blobstore.PromptBlobStore`.

Three sheets, names and headers unchanged:

1. ``硬件配置``    machine / OS / oMLX / model configuration and run parameters
2. ``模型性能平均`` one row per (model, context bucket)
3. ``详细数据``    one row per measured sample

Excel cells are limited to 32,767 characters.  The prompt bodies now live in the
blob store in full, so the head/tail clipping in :func:`_clip` is *more* important
than before, not less: it keeps the workbook writable without silently dropping
evidence (the marker records the original length).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from modelbench.adapters.storage import layout
from modelbench.adapters.storage.blobstore import PromptBlobStore

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
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _iter_records(run_dir: Optional[Path]) -> Iterator[Dict[str, Any]]:
    if run_dir is None:
        return
    path = Path(run_dir) / layout.RECORDS
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


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


def _flag_names(value: Any) -> List[str]:
    """Render ``feature_flags`` whether it is a list of str/dicts or a mapping."""
    if not value:
        return []
    if isinstance(value, dict):
        enabled = [str(key) for key, flag in value.items() if flag]
        return enabled or [str(key) for key in value]
    names: List[str] = []
    for item in value:
        if isinstance(item, dict):
            names.append(str(item.get("key") or item.get("name") or item))
        else:
            names.append(str(item))
    return names


def _gib(size: Any) -> Optional[float]:
    try:
        return round(float(size) / (1024 ** 3), 2)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# schema-3.0 readers
# --------------------------------------------------------------------------- #


def _safe_filename(value: Any) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._")
    return name or "batch"


def excel_path_for(output_root: Path, batch_id: str) -> Path:
    """``output/exports/<batch_id>.xlsx`` (no filesystem change)."""
    return layout.exports_dir(Path(output_root)) / f"{_safe_filename(batch_id)}.xlsx"


def _load_batch(batch_path: Path, batch_meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Read ``batches/<batch_id>.json`` (or use caller-supplied metadata).

    The flattened layout has no batch *directory*: the metadata is one file.  A
    directory argument is a wiring bug (it silently exported an empty workbook in
    the nested layout), so it fails loudly instead of guessing a ``batch.json``.
    """
    if isinstance(batch_meta, dict) and batch_meta:
        return dict(batch_meta)
    path = Path(batch_path)
    if path.is_dir():
        raise ValueError(
            "batch metadata must be a file ("
            f"{layout.BATCHES}/<batch_id>.json), got a directory: {path}"
        )
    payload = _load_json(path)
    if not payload:
        raise FileNotFoundError(f"batch metadata not found: {path}")
    return payload


def _resolve_run_path(output_root: Path, entry: Dict[str, Any]) -> Optional[Path]:
    """Resolve one ``model_runs[]`` entry to its run directory.

    ``batches/<id>.json`` stores ``run_path`` relative to the output root
    (``runs/<run_id>``) so the batch file stays movable; absolute values are
    honoured as-is and a missing ``run_path`` falls back to ``run_id``.
    """
    raw = entry.get("run_path")
    if raw:
        path = Path(str(raw))
        return path if path.is_absolute() else Path(output_root) / path
    run_id = entry.get("run_id")
    if run_id:
        return layout.run_dir(Path(output_root), str(run_id))
    return None


def _scan_run_dirs(output_root: Path, batch: Dict[str, Any]) -> List[Path]:
    """Fallback when a batch file has no ``model_runs`` (incomplete batch).

    The flat ``runs/`` tree has no shared parent directory, so runs belonging to
    this batch are identified by their own ``run.json.batch_id``.
    """
    batch_id = str(batch.get("batch_id") or "")
    if not batch_id:
        return []
    found: List[Path] = []
    root = layout.runs_dir(Path(output_root))
    if not root.is_dir():
        return found
    for run_path in sorted(root.iterdir()):
        if not (run_path / layout.RUN_JSON).is_file():
            continue
        if str(_load_json(run_path / layout.RUN_JSON).get("batch_id") or "") == batch_id:
            found.append(run_path)
    return found


def _collect_runs(output_root: Path, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Load ``run.json`` + ``aggregates.json`` for every run of the batch."""
    entries: List[Dict[str, Any]] = []
    for entry in batch.get("model_runs") or []:
        if isinstance(entry, dict):
            entries.append(entry)
    run_paths: List[tuple[Optional[Path], Dict[str, Any]]] = [
        (_resolve_run_path(output_root, entry), entry) for entry in entries
    ]
    if not run_paths:
        run_paths = [(path, {}) for path in _scan_run_dirs(output_root, batch)]
    runs: List[Dict[str, Any]] = []
    for run_path, entry in run_paths:
        runs.append(
            {
                "path": run_path,
                "entry": entry,
                # An entry without run_path/run_id still gets a row so the model
                # is visible; the status column says why there is no data.
                "run": _load_json(run_path / layout.RUN_JSON) if run_path else {},
                "aggregates": (
                    _load_json(run_path / layout.AGGREGATES_JSON) if run_path else {}
                ),
            }
        )
    return runs


def _run_model(run: Dict[str, Any]) -> Dict[str, Any]:
    model = (run.get("run") or {}).get("model") or {}
    return model if isinstance(model, dict) else {}


def _model_label(run: Dict[str, Any]) -> Any:
    """Human-facing model label; ``batch.json`` keeps the authoritative label."""
    model = _run_model(run)
    entry = run.get("entry") or {}
    return entry.get("model") or model.get("alias") or model.get("id") or entry.get("model_id")


def _model_id(run: Dict[str, Any]) -> Any:
    model = _run_model(run)
    entry = run.get("entry") or {}
    return model.get("id") or entry.get("model_id") or entry.get("model")


def _model_size_gb(run: Dict[str, Any]) -> Optional[float]:
    """Model size in GiB from ``run.json.env.model_env[*].estimated_size``.

    ``model_size_gb`` in ``batch.json`` is the cheap copy (computed by the batch
    runner); the environment snapshot is the authoritative per-run source.
    """
    env = (run.get("run") or {}).get("env") or {}
    model_env = env.get("model_env") or {}
    omlx_models = (env.get("omlx") or {}).get("models") or {}
    blocks: List[Dict[str, Any]] = []
    if isinstance(model_env, dict):
        blocks.extend(info for info in model_env.values() if isinstance(info, dict))
    if isinstance(omlx_models, dict):
        blocks.extend(info for info in omlx_models.values() if isinstance(info, dict))
    model_id = _model_id(run)
    for info in blocks:
        size = info.get("estimated_size")
        if model_id and info.get("id") == model_id and size is not None:
            return _gib(size)
    for info in blocks:
        size = info.get("estimated_size")
        if size is not None:
            return _gib(size)
    entry = run.get("entry") or {}
    size = entry.get("model_size_gb")
    return size if isinstance(size, (int, float)) else None


def _prompt_text(
    request: Dict[str, Any],
    store: PromptBlobStore,
    cache: Optional[Dict[str, str]] = None,
) -> str:
    """Render the exact messages sent, resolving the body from the blob store.

    The same prompt is re-sent for every repeat and every model in a batch, so a
    digest-keyed render cache keeps a large export from re-parsing the same blob
    thousands of times.
    """
    messages = request.get("messages") or []
    if messages:
        return _messages_input_text(messages)
    sha = request.get("prompt_sha256")
    if not sha:
        return ""
    key = str(sha)
    if cache is not None and key in cache:
        return cache[key]
    payload = store.get(key)
    if not payload:
        # Never silently drop the evidence: the workbook has to say why the cell
        # is empty (AGENTS no-silent-degradation invariant).
        text = f"(prompt blob 缺失: {key})"
    else:
        text = _messages_input_text(payload.get("messages") or [])
    if cache is not None:
        cache[key] = text
    return text


def _messages_input_text(messages: Sequence[Any]) -> str:
    parts: List[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
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


def _sheet_hardware(wb: Any, batch: Dict[str, Any], runs: List[Dict[str, Any]]) -> None:
    ws = wb.create_sheet(SHEET_HARDWARE)
    ws.append(["配置项", "值"])
    entries = [run.get("entry") or {} for run in runs]
    durations = [
        run.get("run", {}).get("duration_s", entry.get("duration_s"))
        for run, entry in zip(runs, entries)
    ]
    durations = [value for value in durations if isinstance(value, (int, float))]
    n_samples = [
        run.get("run", {}).get("outcome", {}).get("n_samples", entry.get("n_samples"))
        for run, entry in zip(runs, entries)
    ]
    n_samples_total = sum(value for value in n_samples if isinstance(value, int))
    models = list(batch.get("models_selected") or []) or [
        _model_label(run) for run in runs
    ]
    rows: List[tuple[Any, Any]] = [
        ("批次 ID", batch.get("batch_id")),
        ("测试方法", batch.get("method")),
        ("创建时间", batch.get("created_at")),
        ("模型列表", ", ".join(str(model) for model in models if model)),
        ("模型数量", len(models)),
        ("总耗时(s)", round(sum(durations), 1) if durations else None),
        ("总样本数", n_samples_total or None),
        ("来源", batch.get("source")),
        ("结束时间", batch.get("finished_at")),
    ]
    first = runs[0]["run"] if runs else {}
    env = first.get("env") or {}
    hardware = env.get("hardware") or {}
    omlx = env.get("omlx") or {}
    upload = hardware.get("upload_fields") or {}
    memory = hardware.get("memory") or {}
    cpu = hardware.get("cpu") or {}
    os_info = hardware.get("os") or {}
    python_info = hardware.get("python") or {}
    rows += [
        ("芯片", upload.get("chip_name") or hardware.get("chip")),
        ("芯片变体", upload.get("chip_variant")),
        ("内存(GB)", upload.get("memory_gb") or memory.get("total_gb")),
        ("内存可用(GB)", memory.get("available_gb")),
        ("内存已用(GB)", memory.get("used_gb")),
        ("内存使用率(%)", memory.get("percent")),
        ("Swap 已用(GB)", memory.get("swap_used_gb")),
        ("GPU 核心数", upload.get("gpu_cores")),
        ("操作系统", upload.get("os_version") or os_info.get("version")),
        ("oMLX 版本", upload.get("omlx_version") or omlx.get("version") or omlx.get("omlx_version")),
        ("oMLX 默认配置", omlx.get("default_config")),
        ("CPU 逻辑/物理核心", f"{cpu.get('logical_cores')}/{cpu.get('physical_cores')}"),
        ("主机名", hardware.get("hostname")),
        ("架构", hardware.get("arch")),
        ("Python", python_info.get("version")),
        ("git commit", (first.get("git") or {}).get("commit")),
    ]
    config = first.get("config") or {}
    perf = config.get("perf") or {}
    run_cfg = config.get("run") or {}
    rows += [
        ("样本数/数据集", perf.get("samples_per_dataset") if perf.get("samples_per_dataset") is not None else "全部"),
        ("max_tokens", perf.get("max_tokens")),
        ("decode_guard（长输出保障）", perf.get("decode_guard")),
        ("repeats", run_cfg.get("repeats")),
        ("cache_mode", run_cfg.get("cache_mode")),
        ("thinking_mode", run_cfg.get("thinking_mode")),
        ("加速 feature flags", ", ".join(_flag_names(omlx.get("feature_flags"))) or "无（默认配置）"),
    ]
    for key, value in rows:
        ws.append([key, _clip(value, 500)])
    # Model configuration block: one entry per run.json env.model_env snapshot.
    ws.append([])
    ws.append(["模型配置", "值"])
    seen: set = set()
    model_rows = 0
    for run in runs:
        env = (run.get("run") or {}).get("env") or {}
        model_env = env.get("model_env") or {}
        # ``env.omlx.models`` carries the same per-model facts in the live
        # snapshot; use it to fill gaps when the model_env block is thinner.
        omlx_models = (env.get("omlx") or {}).get("models") or {}
        blocks = dict(model_env) if isinstance(model_env, dict) else {}
        if not blocks and isinstance(omlx_models, dict):
            blocks = dict(omlx_models)
        for key, info in blocks.items():
            if not isinstance(info, dict):
                continue
            fallback = (
                omlx_models.get(info.get("id"))
                if isinstance(omlx_models, dict)
                else None
            ) or {}
            model_id = info.get("id") or fallback.get("id") or _model_id(run)
            identity = (model_id, key)
            if identity in seen:
                continue
            seen.add(identity)
            estimated = info.get("estimated_size", fallback.get("estimated_size"))
            ws.append([
                f"{model_id} · engine",
                info.get("engine_type") or fallback.get("engine_type"),
            ])
            ws.append([
                f"{model_id} · quantization",
                info.get("quantization") or fallback.get("quantization"),
            ])
            ws.append([
                f"{model_id} · context",
                info.get("context_length")
                or fallback.get("context_length")
                or fallback.get("max_context_window"),
            ])
            ws.append([f"{model_id} · 模型大小(GB)", _gib(estimated)])
            settings = info.get("settings") or fallback.get("settings") or {}
            ws.append([
                f"{model_id} · settings",
                _clip(json.dumps(settings, ensure_ascii=False), 2000),
            ])
            ws.append([
                f"{model_id} · feature_flags",
                ", ".join(_flag_names(info.get("feature_flags") or fallback.get("feature_flags"))) or "无",
            ])
            model_rows += 1
    if not model_rows:
        ws.append(["模型配置", "无（缺少 run.json env.model_env）"])
    # Global settings (important sections only)
    globals_ = omlx.get("global_settings") or {}
    for section in ("memory", "scheduler", "cache", "sampling"):
        for key, value in _flatten_settings(globals_.get(section) or {}, section):
            ws.append([f"oMLX.{key}", _clip(value, 500)])
    _set_widths(ws, {1: 40, 2: 90})
    ws.freeze_panes = "A2"


def _sheet_averages(wb: Any, runs: List[Dict[str, Any]]) -> None:
    ws = wb.create_sheet(SHEET_AVERAGES)
    headers = [
        "测试模型", "模型ID", "模型大小(GB)", "上下文档位", "状态", "数据集数", "样本数", "有效数", "错误数",
        "触顶数(max_tokens)", "TTFT均值(ms)", "TTFT中位数(ms)", "TTFT P95(ms)",
        "TG TPS均值", "TG TPS有效数", "TPOT均值(ms)", "E2E均值(ms)", "Prefill TPS均值",
        "峰值内存均值(GB)", "峰值内存最大(GB)", "峰值CPU均值(%)",
        "输入token均值", "输入字符均值", "输出token均值", "耗时(s)",
        "跳过数", "内存错误数", "TG无效数", "ISL入带数", "ISL出带数", "样本量标记",
    ]
    _header(ws, headers)
    for run in runs:
        run_json = run.get("run") or {}
        entry = run.get("entry") or {}
        outcome = run_json.get("outcome") or {}
        aggregates = run.get("aggregates") or {}
        overall_rows = aggregates.get("overall") or []
        skipped = outcome.get("skipped") or []
        status = str(run_json.get("status") or outcome.get("status") or "ok")
        if outcome.get("cancelled") or run_json.get("cancelled"):
            status = "cancelled"
        elif not overall_rows:
            status = "skipped/failed"
        # One row per (model, context_length); never collapse 1K/8K/32K.
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
                _model_label(run),
                _model_id(run),
                _model_size_gb(run),
                _bucket_label(overall.get("context_length")),
                status,
                overall.get("n_datasets"),
                overall.get("n_samples"),
                overall.get("n_valid"),
                overall.get("n_error"),
                overall.get("n_length_capped"),
                mean("ttft_ms"), median("ttft_ms"), p95("ttft_ms"),
                mean("gen_tps"), count("gen_tps"), mean("tpot_ms"), mean("e2e_ms"), mean("prefill_tps"),
                mean("peak_memory_gb"), overall.get("peak_memory_gb_max"),
                mean("peak_cpu_pct"),
                mean("input_context_tokens"), mean("input_context_chars"), mean("completion_tokens"),
                run_json.get("duration_s", entry.get("duration_s")),
                len(skipped), outcome.get("memory_errors") or 0,
                overall.get("n_gen_tps_invalid") or 0,
                overall.get("n_isl_in_band") or 0,
                overall.get("n_isl_out_of_band") or 0,
                "insufficient_samples" if overall.get("insufficient_samples") else "ok",
            ]
            ws.append([_number(value) for value in row])
    if ws.max_row == 1:
        ws.append(["无模型结果"])
    _set_widths(ws, {
        1: 28, 2: 28, 3: 12, 4: 12, 5: 12, 6: 10, 7: 9, 8: 9, 9: 9,
        10: 16, 11: 14, 12: 14, 13: 12, 14: 12, 15: 12, 16: 12, 17: 14,
        18: 15, 19: 15, 20: 14, 21: 13, 22: 13, 23: 12, 24: 10, 25: 9, 26: 11, 27: 11,
        28: 12, 29: 12, 30: 12, 31: 22,
    })


def _bucket_label(context_length: Any) -> str:
    """Render a declared context bucket (1024/8192/32768) as 1K/8K/32K."""
    try:
        value = int(context_length)
    except (TypeError, ValueError):
        return "全部/未标注"
    return f"{value // 1024}K" if value > 0 else "全部/未标注"


def _sheet_details(wb: Any, runs: List[Dict[str, Any]], store: PromptBlobStore) -> None:
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
    prompt_cache: Dict[str, str] = {}
    for run in runs:
        run_dir = run.get("path")
        model_label = _model_label(run)
        model_id = _model_id(run)
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
            input_text = _prompt_text(request, store, prompt_cache)
            ws.append([
                model_label,
                model_id,
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


def _sanitize_workbook(workbook: Any) -> None:
    """Belt-and-braces: strip illegal XML characters from every string cell."""
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

    for ws in workbook.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                if isinstance(cell.value, str) and ILLEGAL_CHARACTERS_RE.search(cell.value):
                    cell.value = ILLEGAL_CHARACTERS_RE.sub("", cell.value)


# --------------------------------------------------------------------------- #
# public entry points
# --------------------------------------------------------------------------- #


def write_batch_excel(
    output_root: Path,
    batch_path: Path,
    batch_meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """Render ``output/exports/<batch_id>.xlsx`` for one batch and return its path.

    ``batch_path`` is the batch metadata **file** (``batches/<batch_id>.json``);
    ``batch_meta`` lets a caller that just finished the batch pass the metadata
    without a round-trip through that file.  Run directories are resolved from
    ``model_runs[].run_path`` relative to ``output_root``.
    """
    output_root = Path(output_root)
    batch_path = Path(batch_path)
    batch = _load_batch(batch_path, batch_meta)
    batch_id = str(batch.get("batch_id") or batch_path.stem)
    runs = _collect_runs(output_root, batch)
    store = PromptBlobStore(output_root)
    workbook = _render_workbook(batch, runs, store)
    _sanitize_workbook(workbook)
    directory = layout.exports_dir(output_root)
    directory.mkdir(parents=True, exist_ok=True)
    workbook_path = excel_path_for(output_root, batch_id)
    tmp = workbook_path.with_name(f".{workbook_path.name}.tmp-{os.getpid()}")
    workbook.save(tmp)
    os.replace(tmp, workbook_path)
    return workbook_path


def _render_workbook(
    batch: Dict[str, Any], runs: List[Dict[str, Any]], store: PromptBlobStore
) -> Any:
    try:
        from openpyxl import Workbook
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise RuntimeError(
            "openpyxl is required for Excel result analysis; "
            "install it with `uv pip install openpyxl` or `pip install openpyxl`"
        ) from exc
    wb = Workbook()
    wb.remove(wb.active)
    _sheet_hardware(wb, batch, runs)
    _sheet_averages(wb, runs)
    _sheet_details(wb, runs, store)
    wb.properties.title = f"Result analysis {batch.get('batch_id')} ({batch.get('method')})"
    wb.properties.creator = "modelbench"
    return wb


def _find_batch_path(output_root: Path, batch_id: str) -> Optional[Path]:
    """Locate ``batches/<batch_id>.json`` for regeneration.

    The batch file is the authoritative batch metadata; a scan of ``batches/``
    covers a misnamed file, and ``index.json`` is only a fallback for a tree
    whose metadata has not been written yet.
    """
    root = Path(output_root)
    canonical = layout.batch_path(root, batch_id)
    if canonical.is_file():
        return canonical
    for meta_path in sorted(layout.batches_dir(root).glob("*.json")):
        meta = _load_json(meta_path)
        if str(meta.get("batch_id") or "") == batch_id or meta_path.stem == batch_id:
            return meta_path
    try:
        from modelbench.adapters.storage.catalog import load_index
    except ImportError:  # pragma: no cover - catalog is an optional collaborator
        return None
    try:
        index = load_index(root)
    except Exception:  # noqa: BLE001 - a broken index must not block export
        index = None
    if not isinstance(index, dict):
        return None
    for entry in index.get("batches") or []:
        if not isinstance(entry, dict) or str(entry.get("batch_id") or "") != batch_id:
            continue
        rel = entry.get("path")
        if not rel:
            continue
        candidate = Path(str(rel))
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.is_file():
            return candidate
    return None


def generate_for_batch(output_root: Path, batch_id: str) -> Path:
    """Regenerate ``exports/<batch_id>.xlsx`` from an existing batch file."""
    output_root = Path(output_root)
    batch_path = _find_batch_path(output_root, batch_id)
    if batch_path is None:
        raise FileNotFoundError(f"batch not found: {batch_id}")
    return write_batch_excel(output_root, batch_path)


def generate_all(output_root: Path) -> List[Path]:
    """Regenerate the workbook for every batch file under ``output/batches``."""
    output_root = Path(output_root)
    results: List[Path] = []
    for batch_path in sorted(layout.batches_dir(output_root).glob("*.json")):
        try:
            results.append(write_batch_excel(output_root, batch_path))
        except FileNotFoundError:
            continue
    return results
