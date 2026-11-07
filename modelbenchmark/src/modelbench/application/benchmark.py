"""Run orchestration: model lifecycle, cache hygiene, per-sample measurement.

The benchmark is performance-only: every selected scenario slice is sampled once
(``perf.samples_per_dataset`` or all samples), each sample is measured with its
pre-built/native prompt (no padding/alignment), and results are persisted as
per-sample records plus overall/cell aggregates.  Cells are keyed by
``(model, scenario, context_length, dataset)``: the declared 1K/8K/32K bucket is
a real cell dimension, so one scenario can produce several cells.

Accuracy scoring is intentionally not invoked (``SampleRecord.score`` is
always ``None``); the evaluator domain is retained but disconnected.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from modelbench import __version__
from modelbench.adapters.omlx.client import OMLXClient, OMLXError, StreamResult
from modelbench.adapters.datasets.registry import get_adapter
from modelbench.adapters.omlx.environment import (
    collect_omlx_environment,
    current_model_settings,
    default_settings_repair,
)
from modelbench.adapters.storage.writer import RunWriter, sanitize_name
from modelbench.domain.models import (
    CacheMode,
    CacheState,
    ChatMessage,
    PerfMetrics,
    Phase,
    RawResponse,
    RecordStatus,
    RequestSpec,
    RunManifest,
    Sample,
    SampleRecord,
    ToolCall,
    Usage,
)
from modelbench.domain.context_windows import describe_windows, in_context_windows
from modelbench.domain.scenarios.base import (
    ScenarioAdapter,
    deterministic_sample,
    stratified_sample,
)
from modelbench.domain.scenarios.registry import build_scenario
from modelbench.adapters.config.settings import ProjectConfig, ScenarioSpec
from modelbench.adapters.omlx.cache import CacheController
from modelbench.adapters.host.hardware import collect_hardware_profile
from modelbench.adapters.host.process_memory import bytes_to_gib
from modelbench.adapters.host.sampler import (
    ResourceReport,
    ResourceSampler,
    process_tree_memory,
)
from modelbench.domain.metrics import compute_metrics

#: Extra slack beyond the documented 242-token reserve (max_tokens + safety
#: margin + decode-guard reserve).  The reserve alone is sufficient: the guard
#: is measured at ~39 tokens (<= 64 reserve) and the 128-token safety margin
#: covers tokenizer drift, so no undocumented extra slop is added here.  Keep
#: the constant for the overflow comparison (room < 0 means the request cannot
#: fit) so docs/manifest (242) and runtime agree.
_MIN_GENERATION_ROOM = 0

#: Measurement-protocol suffix appended to every performance request when
#: ``perf.decode_guard`` is enabled.  Dataset prompts in the performance set
#: often end with ``Answer:`` (LongBench v2 / multiple choice), so the model
#: legitimately emits one token and stops; ``completion_tokens < 16`` then
#: makes ``gen_tps``/``tpot_ms`` null and TG TPS unmeasurable for that sample.
#: This instruction asks for a long deterministic continuation *in addition to*
#: the task, which guarantees a decode window without changing the input
#: context shape.  The response content is not scored (performance-only run).
DECODE_GUARD_INSTRUCTION = (
    "This request is a performance measurement. Ignore the question above. "
    "Instead, write a long detailed technical summary of the provided material "
    "in at least 250 words, in continuous prose."
)

#: oMLX-style model warmup: one short request after model load to trigger JIT,
#: Metal shader compilation and KV-cache initialisation.  Mirrors oMLX's
#: ``QUICK`` warmup (short prompt, ``max_tokens=8``); never recorded/aggregated.
MODEL_WARMUP_PROMPT = (
    "This is a short warmup request for the benchmark. "
    "Ignore any task semantics; reply with the single word: ready."
)
MODEL_WARMUP_MAX_TOKENS = 8

#: Conservative token reserve for the guard in the model-window check.  The
#: authoritative prompt size still comes from server usage
#: (``metrics.input_context_tokens``); this is only headroom so the guard can
#: never push a prebuilt prompt past the model window.
DECODE_GUARD_TOKEN_RESERVE = 64


class RunCancelled(RuntimeError):
    """Raised when the caller requests a cooperative stop."""


@dataclass
class RunStats:
    records: List[SampleRecord] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    memory_errors: int = 0
    cancelled: bool = False
    started_at: float = field(default_factory=time.time)

    @property
    def elapsed_s(self) -> float:
        return time.time() - self.started_at


class Orchestrator:
    def __init__(
        self,
        cfg: ProjectConfig,
        client: OMLXClient,
        writer: RunWriter,
        *,
        progress_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        cancel_event: Any = None,
    ):
        self.cfg = cfg
        self.client = client
        self.writer = writer
        self.progress_cb = progress_cb
        self.cancel_event = cancel_event
        self.controller = CacheController(
            client,
            mode=cfg.run.cache_mode,
            settle_ms=cfg.run.cache_settle_ms,
            max_contamination_retries=cfg.run.max_contamination_retries,
        )
        self.stats = RunStats()
        self.server_pid: Optional[int] = None
        self._dataset_cache: Dict[str, List[Sample]] = {}
        self._adapter_cache: Dict[str, Any] = {}
        self._request_cache: Dict[Tuple[str, str, str], RequestSpec] = {}
        self._loaded_by_run: set = set()
        self._model_memory: Dict[str, Dict[str, Any]] = {}
        self._log_fh = None
        self._omlx_env: Dict[str, Any] = {}
        self._cap_warned = False
        self._cap_hit = False
        self._cell_abort_reason: Optional[str] = None
        self._model_seq = 0
        self._cell_seq = 0
        self._cell_total = 0
        self._memory_warning_issued = False
        self._tool_schema_warning_issued = False
        self._contamination_warning_issued = False

    # ------------------------------------------------------------------ #
    # logging / manifest
    # ------------------------------------------------------------------ #

    def _check_cancel(self) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise RunCancelled("cancelled by user")

    def _progress(self, event: str, payload: Dict[str, Any]) -> None:
        if self.progress_cb is None:
            return
        try:
            self.progress_cb(event, payload)
        except Exception:  # noqa: BLE001 - progress must never break a run
            pass

    def _log(self, message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, file=sys.stderr, flush=True)
        if self._log_fh is None:
            self._log_fh = (self.writer.run_dir / "logs" / "run.log").open("a", encoding="utf-8")
        self._log_fh.write(line + "\n")
        self._log_fh.flush()

    @staticmethod
    def _is_tool_schema_error(error: Optional[str]) -> bool:
        """True when oMLX rejects the request's tool JSON schema (dataset quality)."""
        text = (error or "").lower()
        return any(
            token in text
            for token in (
                "invalid tools",
                "tools schema",
                "invalid tool",
                "function.parameters",
                "tool definition",
                "tools must",
                "invalid parameter type",
                "json schema",
            )
        )

    @staticmethod
    def _is_timeout(error: Optional[str]) -> bool:
        text = (error or "").lower()
        return any(
            token in text
            for token in ("timeout", "timed out", "readtimeout", "connecttimeout")
        )

    @staticmethod
    def _is_memory_error(error: Optional[str]) -> bool:
        text = (error or "").lower()
        return any(
            token in text
            for token in (
                "prefill_memory_exceeded",
                "dynamic memory ceiling",
                "memory guard",
                "http 507",
            )
        )

    @staticmethod
    def _git_state() -> Tuple[Optional[str], Optional[bool]]:
        """Record the project commit so a run bundle is traceable."""
        import subprocess

        try:
            root = Path(__file__).resolve().parents[2]
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root, capture_output=True, text=True, timeout=5,
            )
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root, capture_output=True, text=True, timeout=5,
            )
            if commit.returncode != 0:
                return None, None
            return commit.stdout.strip() or None, bool(status.stdout.strip())
        except Exception:  # noqa: BLE001 - git is optional
            return None, None

    def _models_snapshot(self) -> Dict[str, Any]:
        try:
            status = self.client.models_status()
        except Exception as exc:  # pragma: no cover - live server only
            status = {"error": str(exc)}
        return {
            "server": status,
            "selected": [m.model_dump(mode="json") for m in self.cfg.selected_models()],
        }

    def _refresh_omlx_env(self, warn_flags: bool = True) -> None:
        """Re-read the oMLX configuration snapshot (after a model loads)."""
        try:
            self._omlx_env = collect_omlx_environment(
                self.client, [m.id for m in self.cfg.selected_models()]
            )
        except Exception as exc:  # noqa: BLE001 - environment is advisory
            self._log(f"WARNING: oMLX environment snapshot failed: {exc}")
            return
        self.writer.write_omlx(self._omlx_env)
        for note in self._omlx_env.get("notes") or []:
            if note not in self.stats.warnings:
                self.stats.warnings.append(note)
        flags = self._omlx_env.get("feature_flags") or []
        if flags and warn_flags:
            self._log(
                "WARNING: non-default acceleration features still enabled: "
                f"{flags} (expected empty for a default-config benchmark)"
            )
        elif flags:
            self._log(
                "pre-run config has non-default acceleration features: "
                f"{flags} (will be reset to defaults before measuring)"
            )

    def prepare(self) -> RunManifest:
        self.server_pid = self.client.discover_server_pid()
        git_commit, git_dirty = self._git_state()
        hardware = collect_hardware_profile(self.client)
        self._refresh_omlx_env(warn_flags=False)
        memory = hardware.get("memory") or {}
        health = hardware.get("omlx_health") or {}
        ceiling = (health.get("engine_pool") or {}).get("final_ceiling")
        ceiling_text = f"{ceiling / 1e9:.2f} GB" if ceiling else "unknown"
        self._log(
            "system memory: "
            f"total={memory.get('total_gb')}GB available={memory.get('available_gb')}GB "
            f"used={memory.get('used_gb')}GB ({memory.get('percent')}%) "
            f"swap={memory.get('swap_used_gb')}GB | oMLX dynamic ceiling={ceiling_text}"
        )
        manifest = RunManifest(
            run_id=self.writer.run_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            project_version=__version__,
            mode="performance",
            batch=dict(getattr(self.writer, "batch", {}) or {}),
            config_snapshot=self.cfg.redacted_snapshot(),
            hardware=hardware,
            omlx=self._omlx_env,
            server={
                "base_url": self.cfg.server.base_url,
                "pid": self.server_pid,
                "cache_mode": self.cfg.run.cache_mode.value,
            },
            models=self._models_snapshot(),
            scenarios=list(self.cfg.selected_scenarios()),
            host={"python": sys.version.split()[0], "cwd": str(Path.cwd())},
            git_commit=git_commit,
            git_dirty=git_dirty,
        )
        self.writer.initialize(manifest, hardware, manifest.models, self._omlx_env)
        self._log(
            f"run {self.writer.run_id} | mode=performance "
            f"models={[m.label for m in self.cfg.selected_models()]} "
            f"datasets={list(self.cfg.selected_scenarios())} "
            f"samples_per_dataset={self.cfg.perf.samples_per_dataset or 'all'} "
            f"max_tokens={self.cfg.perf.max_tokens} repeats={self.cfg.run.repeats} "
            f"warmup_per_model={self.cfg.run.warmup_per_model}"
        )
        estimate = self._estimate_requests()
        cap = self.cfg.run.max_requests
        if estimate is not None and cap and estimate > cap:
            self._log(
                f"WARNING: plan needs ~{estimate} measured requests but max_requests={cap}; "
                "the run will be truncated (raise max_requests or narrow the plan)"
            )
        if self.server_pid:
            self._log(f"oMLX server pid={self.server_pid} (CPU/RSS sampling enabled)")
        else:
            self._log("WARNING: oMLX server pid not found; CPU/RSS sampling disabled")
        return manifest

    # ------------------------------------------------------------------ #
    # main loop
    # ------------------------------------------------------------------ #

    def run(self) -> Dict[str, Any]:
        self._progress("run_start", {"run_id": self.writer.run_id})
        selected_models: List[Any] = []
        started_models: set = set()
        try:
            self.prepare()
            selected_models = self.cfg.selected_models()
            for index, model_spec in enumerate(selected_models):
                self._check_cancel()
                if self._cap_reached():
                    for remaining in selected_models[index:]:
                        self.stats.skipped.append(
                            f"model {remaining.id}: not measured "
                            "(max_requests cap reached)"
                        )
                    break
                started_models.add(model_spec.id)
                try:
                    self._run_model(model_spec)
                except RunCancelled:
                    raise
                except Exception as exc:  # noqa: BLE001 - keep the run alive unless fail_fast
                    self._log(f"ERROR: model {model_spec.id} aborted: {type(exc).__name__}: {exc}")
                    self.stats.skipped.append(f"model {model_spec.id}: {exc}")
                    if self.cfg.run.fail_fast:
                        self._log("fail_fast=true: stopping the run on the first failure")
                        raise
        except RunCancelled:
            self.stats.cancelled = True
            self.stats.skipped.append("cancelled by user")
            # Models that never started have no sample records; make the skip
            # explicit at model level (AGENTS §9/§13).  Samples of a started
            # model are recorded per-sample by _run_model_impl/_run_cell_impl.
            for pending in selected_models:
                if pending.id not in started_models:
                    self.stats.skipped.append(
                        f"model {pending.id}: not measured (cancelled by user)"
                    )
            self._log("run cancelled by user")
        finally:
            cleanup_report = self.controller.cleanup()
            if isinstance(cleanup_report, dict) and cleanup_report.get("error"):
                message = (
                    "cache cleanup failed after run: "
                    f"{cleanup_report.get('error')}"
                )
                self.stats.warnings.append(message)
                self._log(f"WARNING: {message}")
            if self.cfg.run.restore_state:
                self._restore_models()
            if self._omlx_env:
                self.writer.write_omlx(self._omlx_env)
            self.writer.close()
            if self._log_fh:
                self._log_fh.close()
                self._log_fh = None
        extra = {
            "duration_s": round(self.stats.elapsed_s, 1),
            "skipped": list(self.stats.skipped),
            "warnings": list(self.stats.warnings),
            "memory_errors": self.stats.memory_errors,
            "cancelled": self.stats.cancelled,
        }
        summary = self.writer.finalize(
            self.stats.records,
            extra=extra,
            perf=self.cfg.perf.model_dump(mode="json"),
            omlx=self._omlx_env,
            run_config={
                "repeats": self.cfg.run.repeats,
                "warmup_requests": self.cfg.run.warmup_requests,
                "warmup_per_model": self.cfg.run.warmup_per_model,
                "cache_mode": self.cfg.run.cache_mode.value,
                "thinking_mode": self.cfg.run.thinking_mode,
                "max_requests": self.cfg.run.max_requests,
            },
        )
        self._log(
            f"done: {summary['n_records']} records, {summary['n_samples']} samples, "
            f"{summary['n_cells']} cells, {summary['duration_s']}s -> {summary['run_dir']}"
        )
        return summary

    def _estimate_requests(self) -> Optional[int]:
        """Measured-request count when every selected scenario has a fixed size.

        Uses ``DatasetAdapter.plan()`` for scenario slices whose sample count is
        only known after filtering the dataset (e.g. ``perf_jsonl``); adapters
        without a plan keep the run estimate unknown.
        """
        total = 0
        for scenario_name, spec in self.cfg.selected_scenarios().items():
            explicit = self.cfg.perf.sample_ids_for(scenario_name)
            count = self.cfg.perf.samples_for(scenario_name)
            extra: Dict[str, Any] = {}
            if self.cfg.perf.context_windows:
                extra["context_windows"] = list(self.cfg.perf.context_windows)
            if explicit:
                extra["sample_ids"] = list(explicit)
            elif count is not None:
                extra["samples"] = int(count)
            plan = self._dataset_plan(spec, extra)
            if plan is not None:
                total += sum(int(item["n"]) for item in plan)
                continue
            # Legacy adapter without a plan: fixed counts are still knowable.
            if explicit:
                total += len(explicit)
            elif count is not None:
                total += int(count)
            else:
                return None
        return total * len(self.cfg.selected_models()) * int(self.cfg.run.repeats)

    def _dataset_plan(
        self, spec: ScenarioSpec, extra: Optional[Dict[str, Any]] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Cell plan for one scenario slice (``None`` when unknown)."""
        adapter = self._adapter_for(spec)
        plan = getattr(adapter, "plan", None)
        if plan is None:
            return None
        try:
            return plan(extra or None)
        except Exception as exc:  # noqa: BLE001 - estimate must never break a run
            self._log(f"WARNING: cannot plan dataset {spec.dataset}: {exc}")
            return None

    def _restore_models(self) -> None:
        """Unload models that this run loaded (restore_state).

        oMLX may have already evicted the model under memory pressure; skip it
        instead of producing a misleading "Model not loaded" warning.
        """
        try:
            currently_loaded = set(self.client.loaded_models())
        except Exception:  # noqa: BLE001 - best effort
            currently_loaded = set(self._loaded_by_run)
        for model_id in sorted(self._loaded_by_run):
            if model_id not in currently_loaded:
                self._log(
                    f"restore_state: {model_id} already unloaded/evicted; nothing to do"
                )
                continue
            try:
                self._log(f"restore_state: unloading {model_id}")
                self.client.unload_model(model_id)
                self.client.wait_until_unloaded(model_id, timeout_s=120)
            except Exception as exc:  # noqa: BLE001 - best effort
                self._log(f"WARNING: restore_state unload failed for {model_id}: {exc}")
        self._loaded_by_run.clear()

    def _cap_reached(self) -> bool:
        cap = self.cfg.run.max_requests
        if cap:
            measured = sum(1 for record in self.stats.records if record.phase is Phase.MEASURED)
            if measured >= cap:
                message = (
                    f"max_requests cap reached ({cap} measured); "
                    "remaining planned samples were not measured"
                )
                self._log(message)
                if not self._cap_warned:
                    self.stats.warnings.append(message)
                    self._cap_warned = True
                self._cap_hit = True
                return True
        return False

    # ------------------------------------------------------------------ #
    # model lifecycle
    # ------------------------------------------------------------------ #

    def _ensure_model_ready(self, model_spec) -> Dict[str, Any]:
        model_id = model_spec.id
        self._log(f"=== model {model_id} ===")
        try:
            snapshot = self.client.model_config(model_id)
        except OMLXError:
            snapshot = {"id": model_id}

        if self.cfg.run.isolate_models:
            try:
                for loaded in self.client.loaded_models():
                    if loaded != model_id:
                        self._log(f"unloading {loaded} (isolate_models)")
                        self.client.unload_model(loaded)
                        self.client.wait_until_unloaded(loaded, timeout_s=120)
            except OMLXError as exc:
                self._log(f"WARNING: unloading others failed: {exc}")

        try:
            load_cfg = getattr(model_spec, "load", None)
            configured = dict(getattr(load_cfg, "settings", {}) or {})
            # Enforce the default-config contract: persisted TurboQuant/DFlash/
            # MTP/SpecPrefill/ANE toggles left over from earlier runs are reset
            # to False; the only intentional deviation is MoE expert offload.
            current = current_model_settings(self.client, model_id)
            settings = default_settings_repair(current, configured)
            if settings:
                repaired = sorted(set(settings) - set(configured))
                if repaired:
                    self._log(
                        f"resetting non-default acceleration settings for {model_id}: "
                        f"{ {key: settings[key] for key in repaired} }"
                    )
                if configured:
                    self._log(f"applying MoE load settings for {model_id}: {configured}")
                try:
                    if snapshot.get("loaded") and getattr(load_cfg, "unload_before_apply", True):
                        self.client.unload_model(model_id)
                        self.client.wait_until_unloaded(model_id, timeout_s=120)
                        snapshot = {"id": model_id, "loaded": False}
                    apply = getattr(self.client, "apply_model_settings", None)
                    if apply is None:
                        result = {
                            "applied": True,
                            "response": self.client.update_model_settings(model_id, settings),
                        }
                    else:
                        result = apply(
                            model_id,
                            settings,
                            optional=bool(getattr(load_cfg, "optional", True)),
                        )
                    if not result.get("applied", True):
                        message = (
                            f"model {model_id}: load settings not applied: {result.get('error')}"
                        )
                        self.stats.warnings.append(message)
                        self._log(f"WARNING: {message}")
                    else:
                        self._log(f"load settings applied: {result.get('response')}")
                except Exception as exc:  # noqa: BLE001
                    if getattr(load_cfg, "optional", True):
                        message = f"model {model_id}: load settings failed: {exc}"
                        self.stats.warnings.append(message)
                        self._log(f"WARNING: {message}")
                    else:
                        raise
            memory_baseline = process_tree_memory(self.server_pid)
            was_loaded = bool(snapshot.get("loaded"))
            if not snapshot.get("loaded"):
                self._log("loading model ...")
                self.client.load_model(model_id)
                self._loaded_by_run.add(model_id)
                if not self.client.wait_until_loaded(model_id, timeout_s=600):
                    self._log(f"WARNING: model {model_id} did not report loaded in time")
            snapshot = self.client.model_config(model_id)
            self._refresh_omlx_env()
            memory_idle = process_tree_memory(self.server_pid)
            memory = {
                "baseline_gb": memory_baseline.get("footprint_gb") if not was_loaded else None,
                "idle_gb": memory_idle.get("footprint_gb"),
                "baseline_rss_mb": memory_baseline.get("rss_mb") if not was_loaded else None,
                "idle_rss_mb": memory_idle.get("rss_mb"),
            }
            if memory["baseline_gb"] is not None and memory["idle_gb"] is not None:
                memory["load_gb"] = round(memory["idle_gb"] - memory["baseline_gb"], 2)
            self._model_memory[model_id] = memory
            if self._omlx_env:
                self._omlx_env.setdefault("models", {}).setdefault(model_id, {})[
                    "memory"
                ] = memory
                self.writer.write_omlx(self._omlx_env)
            model_env = (self._omlx_env.get("models") or {}).get(model_id, {})
            self._log(
                f"model ready: ctx={snapshot.get('max_context_window')} "
                f"max_tokens={snapshot.get('max_tokens')} size={snapshot.get('estimated_size')} "
                f"quantization={model_env.get('quantization')} "
                f"feature_flags={[f['key'] for f in model_env.get('feature_flags') or []]} "
                f"memory={memory.get('baseline_gb')}->{memory.get('idle_gb')}GB"
                + (f" (load +{memory['load_gb']}GB)" if memory.get("load_gb") is not None else "")
            )
        except OMLXError as exc:
            self._log(f"ERROR: cannot load {model_id}: {exc}")
            raise
        return snapshot

    # ------------------------------------------------------------------ #
    # datasets / requests
    # ------------------------------------------------------------------ #

    def _adapter_for(self, spec: ScenarioSpec):
        params = dict(spec.params.get("dataset_params") or {})
        key = spec.dataset + "|" + json.dumps(params, sort_keys=True)
        adapter = self._adapter_cache.get(key)
        if adapter is None:
            adapter = get_adapter(
                spec.dataset,
                raw_dir=Path(self.cfg.datasets.raw_dir),
                processed_dir=Path(self.cfg.datasets.processed_dir),
                params=params,
                testset_dir=Path(self.cfg.datasets.testset_dir),
            )
            self._adapter_cache[key] = adapter
        return adapter

    def _load_samples(self, spec: ScenarioSpec) -> List[Sample]:
        params = dict(spec.params.get("dataset_params") or {})
        key = spec.dataset + "|" + json.dumps(params, sort_keys=True)
        if key not in self._dataset_cache:
            adapter = self._adapter_for(spec)
            samples = adapter.load()
            self._log(f"dataset {spec.dataset}: {len(samples)} samples available")
            self._dataset_cache[key] = samples
        return self._dataset_cache[key]

    def _tokenizer(self, model_id: str) -> Callable[[List[ChatMessage], Any], int]:
        def _count(messages: List[ChatMessage], tools: Any = None) -> int:
            return self.client.tokenize_messages(messages, model=model_id, tools=tools)

        return _count

    def _build_request(
        self,
        scenario: ScenarioAdapter,
        spec: ScenarioSpec,
        model_id: str,
        sample: Sample,
        server_ctx: Optional[int],
    ) -> RequestSpec:
        # Tokenizer counts/truncation are model-specific: the cache key must
        # include the model so a multi-model run cannot reuse another model's
        # prompt preparation.
        key = (model_id, sample.sample_id, spec.name)
        if key in self._request_cache:
            return self._request_cache[key]
        max_tokens = self.cfg.perf.max_tokens
        request = scenario.build_request(
            model_id,
            sample,
            None,  # performance mode: native context, no target length
            self._tokenizer(model_id),
            max_tokens=max_tokens,
            temperature=0.0,
            seed=self.cfg.run.seed,
        )
        thinking = self.cfg.run.thinking_mode
        if thinking in ("disabled", "enabled"):
            request.chat_template_kwargs = {"enable_thinking": thinking == "enabled"}
        guard_reserve = (
            DECODE_GUARD_TOKEN_RESERVE if self.cfg.perf.decode_guard else 0
        )
        if request.context_meta.get("prebuilt"):
            # Pre-constructed samples are sent verbatim except for the optional
            # decode guard appended below.  They must NEVER be silently
            # truncated: an overflowing prompt raises ValueError so the existing
            # per-sample skip path records the reason instead.
            declared = request.actual_prompt_tokens
            if server_ctx and declared is not None:
                room = (
                    int(server_ctx)
                    - int(declared)
                    - max_tokens
                    - self.cfg.run.context_safety_margin_tokens
                    - guard_reserve
                )
                request.context_meta["model_context_window"] = int(server_ctx)
                if room < _MIN_GENERATION_ROOM:
                    reserve = (
                        max_tokens
                        + self.cfg.run.context_safety_margin_tokens
                        + guard_reserve
                    )
                    raise ValueError(
                        f"prebuilt prompt too long for {model_id}: "
                        f"declared {declared} + generation {max_tokens} + "
                        f"safety_margin {self.cfg.run.context_safety_margin_tokens} + "
                        f"decode_guard_reserve {guard_reserve} = "
                        f"{declared + reserve} exceeds context window {server_ctx} "
                        f"(not-testable; no silent truncation)"
                    )
            self._apply_decode_guard(request)
            self._request_cache[key] = request
            return request
        # Keep prompt + generation inside the model's context window.  The
        # uniform generation length is preserved; only oversized prompts are
        # truncated (head+tail) so the sample still fits.
        if server_ctx and request.actual_prompt_tokens:
            room = (
                server_ctx
                - request.actual_prompt_tokens
                - max_tokens
                - self.cfg.run.context_safety_margin_tokens
                - guard_reserve
            )
            if room < _MIN_GENERATION_ROOM:
                target = (
                    server_ctx
                    - max_tokens
                    - self.cfg.run.context_safety_margin_tokens
                    - guard_reserve
                )
                if target < 128:
                    raise ValueError(
                        f"prompt too long for {model_id}: {request.actual_prompt_tokens} tokens"
                    )
                request = scenario.build_request(
                    model_id,
                    sample,
                    target,
                    self._tokenizer(model_id),
                    max_tokens=max_tokens,
                    temperature=0.0,
                    seed=self.cfg.run.seed,
                )
                request.target_context_tokens = None
                request.truncated = True
                request.context_meta["native_truncated"] = True
                request.context_meta["model_context_window"] = server_ctx
                if thinking in ("disabled", "enabled"):
                    request.chat_template_kwargs = {
                        "enable_thinking": thinking == "enabled"
                    }
        self._apply_decode_guard(request)
        self._request_cache[key] = request
        return request

    def _apply_decode_guard(self, request: RequestSpec) -> None:
        """Append the fixed long-output suffix so TG TPS is measurable.

        The suffix is part of the *measurement protocol*, not the dataset: it
        is recorded in the request (and thus ``records.jsonl``) and controlled
        by ``perf.decode_guard``.  It is applied to warmup and measured
        requests alike, so the decode path is warmed identically.
        """
        if not self.cfg.perf.decode_guard:
            return
        if request.context_meta.get("decode_guard"):
            return
        request.messages.append(
            ChatMessage(role="user", content=DECODE_GUARD_INSTRUCTION)
        )
        request.context_meta["decode_guard"] = True
        request.context_meta["decode_guard_instruction"] = DECODE_GUARD_INSTRUCTION
        request.context_meta["decode_guard_tokens_reserved"] = DECODE_GUARD_TOKEN_RESERVE

    # ------------------------------------------------------------------ #
    # cell execution
    # ------------------------------------------------------------------ #

    def _run_model(self, model_spec) -> None:
        """Wrapper that reports model lifecycle progress to the UI."""
        self._model_seq += 1
        self._progress(
            "model_start",
            {
                "model": model_spec.id,
                "index": self._model_seq,
                "total": len(self.cfg.selected_models()),
            },
        )
        try:
            self._run_model_impl(model_spec)
        finally:
            self._progress(
                "model_done",
                {"model": model_spec.id, "index": self._model_seq},
            )

    def _warmup_model(self, model_id: str) -> None:
        """oMLX-style model warmup: one short request, discarded and unrecorded.

        oMLX's built-in benchmark sends a single short request after loading a
        model (default ``QUICK``: 32 prompt tokens, ``max_tokens=8``) before any
        measured test, so JIT/Metal compilation is not charged to the first
        measured sample.  It does not warm the prefix cache; the performance
        run's cold-cache policy is unchanged (the request is sent with
        ``cache_store=False``).
        """
        count = int(self.cfg.run.warmup_per_model or 0)
        if count <= 0:
            return
        if self.cfg.run.cache_mode is not CacheMode.NONE:
            state = self.controller.clear()
            if not state.cleared_before_request:
                message = f"model warmup {model_id}: cache clear failed: {state.note}"
                self._log(f"WARNING: {message}")
                self.stats.warnings.append(message)
        request = RequestSpec(
            model=model_id,
            messages=[ChatMessage(role="user", content=MODEL_WARMUP_PROMPT)],
            max_tokens=MODEL_WARMUP_MAX_TOKENS,
            temperature=0.0,
            context_meta={"warmup": True},
        )
        if self.cfg.run.thinking_mode in ("disabled", "enabled"):
            request.chat_template_kwargs = {
                "enable_thinking": self.cfg.run.thinking_mode == "enabled"
            }
        for index in range(1, count + 1):
            started = time.perf_counter()
            try:
                stream = self.client.chat_stream(request, cache_store=False)
            except Exception as exc:  # noqa: BLE001 - warmup must not kill the run
                message = (
                    f"model warmup failed for {model_id} "
                    f"({index}/{count}): {type(exc).__name__}: {exc}"
                )
                self._log(f"WARNING: {message}")
                self.stats.warnings.append(message)
                return
            wall_ms = (time.perf_counter() - started) * 1000.0
            ttft = stream.first_token_ts
            ttft_ms = round(ttft * 1000.0, 1) if ttft is not None else None
            completion = stream.usage.completion_tokens
            self._log(
                f"model warmup {model_id} {index}/{count}: "
                f"ttft={ttft_ms}ms wall={round(wall_ms, 1)}ms "
                f"completion={completion}"
            )

    def _run_model_impl(self, model_spec) -> None:
        model_id = model_spec.id
        try:
            snapshot = self._ensure_model_ready(model_spec)
        except OMLXError as exc:
            self.stats.skipped.append(f"model {model_id}: {exc}")
            if self.cfg.run.fail_fast:
                raise
            return
        server_ctx = snapshot.get("max_context_window") or snapshot.get("model_context_length")
        if model_spec.max_context:
            server_ctx = min(server_ctx or model_spec.max_context, model_spec.max_context)

        # Load/filter/select every scenario first so the cell total (one cell per
        # sample.dataset x context_length group, matching ``aggregate_records``)
        # is known before the first progress event is emitted.
        prepared: List[Tuple[str, ScenarioAdapter, ScenarioSpec, Dict[Tuple[str, Optional[int]], List[Sample]]]] = []
        groups_total = 0
        scenario_specs = self.cfg.selected_scenarios()
        prepared_names = set()
        try:
            for name, spec in scenario_specs.items():
                self._check_cancel()
                if self._cap_reached():
                    return
                scenario = build_scenario(name, spec.params)
                try:
                    samples = self._load_samples(spec)
                except Exception as exc:
                    self._log(f"ERROR: cannot load dataset {spec.dataset}: {exc}")
                    self.stats.skipped.append(f"{name}/{spec.dataset}: {exc}")
                    if self.cfg.run.fail_fast:
                        raise
                    continue
                samples = self._filter_by_context(
                    name, scenario, spec, model_id, samples, server_ctx
                )
                samples = self._select_samples(name, samples)
                if not samples:
                    self._log(f"WARNING: scenario {name} has no samples after selection")
                    continue
                groups = self._group_samples(samples)
                prepared.append((name, scenario, spec, groups))
                prepared_names.add(name)
                groups_total += len(groups)
        except RunCancelled:
            # Cancelled while preparing scenarios: selected samples that were
            # already prepared get per-sample skipped entries; later scenarios
            # were never loaded, so they get an explicit scenario-level entry.
            for name, scenario, spec, groups in prepared:
                for (dataset, context_length), cell_samples in groups.items():
                    self._record_abort_skips(
                        scenario,
                        self._cell_label(dataset, context_length),
                        cell_samples,
                        start_index=0,
                        start_repeat=0,
                        repeats=self.cfg.run.repeats,
                        reason="cancelled by user",
                    )
            for name in scenario_specs:
                if name not in prepared_names:
                    self.stats.skipped.append(
                        f"{name}: not measured (cancelled during sample preparation)"
                    )
            raise
        self._cell_total = groups_total * len(self.cfg.selected_models())
        self._log(
            f"cell plan: {self._cell_total} cell(s) = {groups_total} "
            f"(scenario x dataset x context) groups x "
            f"{len(self.cfg.selected_models())} model(s)"
        )

        cells = [
            (scenario, spec, dataset, context_length, cell_samples)
            for name, scenario, spec, groups in prepared
            for (dataset, context_length), cell_samples in groups.items()
        ]
        current_cell_index = -1
        try:
            if cells:
                self._warmup_model(model_id)

            for index, (scenario, spec, dataset, context_length, cell_samples) in enumerate(cells):
                self._check_cancel()
                if self._cap_reached():
                    self._record_cap_skips(
                        scenario.name,
                        [(dataset, context_length, cell_samples)],
                    )
                    self._record_cap_skips(
                        scenario.name,
                        [
                            (d, c, samples)
                            for _s, _sp, d, c, samples in cells[index + 1 :]
                        ],
                    )
                    return
                current_cell_index = index
                self._run_cell(
                    model_id,
                    snapshot,
                    scenario,
                    spec,
                    cell_samples,
                    server_ctx,
                    dataset=dataset,
                    context_length=context_length,
                )
        except RunCancelled:
            # The current cell (if _run_cell started) already recorded its own
            # remaining samples.  Record every later cell so a cancel between
            # cells / during model warmup still satisfies AGENTS §13.
            for _scenario, _spec, dataset, context_length, cell_samples in cells[
                current_cell_index + 1 :
            ]:
                self._record_abort_skips(
                    _scenario,
                    self._cell_label(dataset, context_length),
                    cell_samples,
                    start_index=0,
                    start_repeat=0,
                    repeats=self.cfg.run.repeats,
                    reason="cancelled by user",
                )
            raise

    @staticmethod
    def _context_length(sample: Sample) -> Optional[int]:
        """Declared bucket size (1024/8192/32768); ``None`` for legacy samples."""
        value = (sample.metadata or {}).get("context_length")
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _cell_label(cls, dataset: str, context_length: Optional[int]) -> str:
        return f"{dataset}@{context_length // 1024}k" if context_length else dataset

    def _group_samples(
        self, samples: List[Sample]
    ) -> Dict[Tuple[str, Optional[int]], List[Sample]]:
        """Group one scenario slice into cells keyed like ``aggregate_records``."""
        groups: Dict[Tuple[str, Optional[int]], List[Sample]] = {}
        for sample in samples:
            key = (sample.dataset, self._context_length(sample))
            groups.setdefault(key, []).append(sample)
        return groups

    def _record_abort_skips(
        self,
        scenario: ScenarioAdapter,
        label: str,
        samples: List[Sample],
        *,
        start_index: int,
        start_repeat: int,
        repeats: int,
        reason: str,
    ) -> None:
        """Record samples an aborted cell never measured (auditability contract).

        A cell that aborts (e.g. cache clear fails because oMLX restarted) must
        not silently drop its remaining selected samples: every selected sample
        needs either a measured record or a ``summary.skipped`` entry (AGENTS
        invariant 13).  Covers the current sample's pending repeats and every
        later sample of the cell.
        """
        for offset, sample in enumerate(samples[start_index:]):
            first_repeat = start_repeat if offset == 0 else 0
            for pending_repeat in range(first_repeat, repeats):
                self.stats.skipped.append(
                    f"{scenario.name}/{label}/{sample.sample_id}/r{pending_repeat}: "
                    f"not measured (cell aborted: {reason})"
                )

    def _record_cap_skips(
        self,
        scenario_name: str,
        groups: List[Tuple[str, Optional[int], List[Sample]]],
    ) -> None:
        """Record every unmeasured selected sample when ``max_requests`` stops a run."""
        cap = self.cfg.run.max_requests
        for dataset, context_length, samples in groups:
            label = self._cell_label(dataset, context_length)
            for sample in samples:
                self.stats.skipped.append(
                    f"{scenario_name}/{label}/{sample.sample_id}: not measured "
                    f"(max_requests cap reached, cap={cap})"
                )

    def _filter_by_context(
        self,
        scenario_name: str,
        scenario: ScenarioAdapter,
        spec: ScenarioSpec,
        model_id: str,
        samples: List[Sample],
        server_ctx: Optional[int],
    ) -> List[Sample]:
        """Keep only samples whose prompt tokens fall inside the selected windows.

        Pre-built performance samples use their declared ``metadata.tokens``
        (reference tokenizer, the same value that defines their 1K/8K/32K bucket);
        legacy samples use the model tokenizer (``/v1/tokenize`` via
        ``_build_request``).  The filter only narrows the sample set -- it never
        changes the cell key: cells always include ``context_length``.
        """
        windows = list(self.cfg.perf.context_windows or [])
        if not windows:
            return samples
        explicit = self.cfg.perf.sample_ids_for(scenario_name)
        wanted = set(explicit) if explicit else None
        ordered = sorted(samples, key=lambda sample: sample.sample_id)
        kept: List[Sample] = []
        examined = 0
        dropped = 0
        for sample in ordered:
            if wanted is not None and sample.sample_id not in wanted:
                continue
            examined += 1
            tokens: Optional[int] = None
            metadata = sample.metadata or {}
            if metadata.get("prebuilt") and metadata.get("tokens") is not None:
                # Pre-built samples are filtered by the declared (reference
                # tokenizer) token count that defines their bucket -- not by the
                # target model tokenizer.  See docs/METHODOLOGY.md §5.
                tokens = int(metadata["tokens"])
            else:
                try:
                    request = self._build_request(
                        scenario, spec, model_id, sample, server_ctx
                    )
                    tokens = request.actual_prompt_tokens
                except Exception as exc:  # noqa: BLE001 - skip unfilterable sample
                    self._log(
                        f"WARNING: context filter cannot tokenize {sample.sample_id}: {exc}"
                    )
            if tokens is not None and in_context_windows(int(tokens), windows):
                kept.append(sample)
            else:
                dropped += 1
        label = describe_windows(windows)
        self._log(
            f"dataset {scenario_name}: context windows {label} -> kept "
            f"{len(kept)}/{examined} examined samples"
            + (f" ({dropped} outside range)" if dropped else "")
        )
        if not kept:
            message = (
                f"scenario {scenario_name}: no samples inside selected context "
                f"windows ({label}); {examined} candidates checked"
            )
            self.stats.skipped.append(message)
            self._log(f"WARNING: {message}")
        return kept

    def _select_samples(self, scenario_name: str, samples: List[Sample]) -> List[Sample]:
        """Apply explicit sample ids, then per-scenario count, then global count."""
        explicit = self.cfg.perf.sample_ids_for(scenario_name)
        if explicit:
            by_id = {sample.sample_id: sample for sample in samples}
            missing = [sample_id for sample_id in explicit if sample_id not in by_id]
            if missing:
                message = (
                    f"scenario {scenario_name}: {len(missing)} selected sample id(s) "
                    f"not found: {missing[:5]}"
                )
                self._log(f"WARNING: {message}")
                self.stats.warnings.append(message)
            chosen = [by_id[sample_id] for sample_id in explicit if sample_id in by_id]
            self._log(
                f"dataset {scenario_name}: {len(chosen)} explicitly selected samples "
                f"(requested {len(explicit)})"
            )
            if not chosen:
                message = (
                    f"scenario {scenario_name}: none of the {len(explicit)} explicitly "
                    "selected sample ids exist in the dataset"
                )
                self.stats.skipped.append(message)
                self._log(f"WARNING: {message}")
            return chosen
        count = self.cfg.perf.samples_for(scenario_name)
        prebuilt = any((sample.metadata or {}).get("prebuilt") for sample in samples)
        if prebuilt:
            chosen = stratified_sample(samples, count)
            how = (
                "all"
                if count is None or count >= len(samples)
                else f"first {count} stratified by context_length/lang"
            )
        else:
            chosen = deterministic_sample(samples, count)
            how = "all" if count is None else f"first {count}"
        buckets: Dict[Any, int] = {}
        for sample in chosen:
            key = (sample.metadata or {}).get("context_length")
            buckets[key] = buckets.get(key, 0) + 1
        self._log(
            f"dataset {scenario_name}: selected {len(chosen)} of {len(samples)} samples "
            f"({how}) strata={buckets if prebuilt else 'legacy'}"
        )
        return chosen

    def _run_cell(
        self,
        model_id: str,
        model_snapshot: Dict[str, Any],
        scenario: ScenarioAdapter,
        spec: ScenarioSpec,
        samples: List[Sample],
        server_ctx: Optional[int],
        *,
        dataset: Optional[str] = None,
        context_length: Optional[int] = None,
    ) -> None:
        """Wrapper that reports per-cell progress (one aggregate cell)."""
        self._cell_seq += 1
        repeats = self.cfg.run.repeats
        cell_total = self._cell_total or (
            len(self.cfg.selected_models()) * len(self.cfg.selected_scenarios())
        )
        cell_dataset = dataset if dataset is not None else spec.dataset
        payload = {
            "model": model_id,
            "scenario": scenario.name,
            "dataset": cell_dataset,
            "context_length": context_length,
            "samples": len(samples),
            "repeats": repeats,
            "cell_index": self._cell_seq,
            "cell_total": cell_total,
        }
        self._progress("cell_start", dict(payload))
        try:
            self._run_cell_impl(
                model_id,
                model_snapshot,
                scenario,
                spec,
                samples,
                server_ctx,
                dataset=cell_dataset,
                context_length=context_length,
            )
        finally:
            self._progress("cell_done", dict(payload))

    def _run_cell_impl(
        self,
        model_id: str,
        model_snapshot: Dict[str, Any],
        scenario: ScenarioAdapter,
        spec: ScenarioSpec,
        samples: List[Sample],
        server_ctx: Optional[int],
        *,
        dataset: Optional[str] = None,
        context_length: Optional[int] = None,
    ) -> None:
        repeats = self.cfg.run.repeats
        cell_dataset = dataset if dataset is not None else spec.dataset
        label = self._cell_label(cell_dataset, context_length)
        self._log(
            f"--- cell {model_id} | {scenario.name} | {label} | "
            f"{len(samples)} samples x {repeats} repeats"
            f" (warmup {self.cfg.run.warmup_requests} per cell)"
        )
        self._check_cancel()
        self.controller.start_cell()
        self._cell_abort_reason = None
        first_sample = samples[0]
        warmup = self.cfg.run.warmup_requests
        if warmup:
            try:
                request = self._build_request(scenario, spec, model_id, first_sample, server_ctx)
            except Exception as exc:
                self._log(f"ERROR building warmup request: {exc}")
                self.stats.skipped.append(f"{scenario.name}/{label}/warmup: {exc}")
                self._record_abort_skips(
                    scenario,
                    label,
                    samples,
                    start_index=0,
                    start_repeat=0,
                    repeats=repeats,
                    reason=f"warmup request build failed: {exc}",
                )
                return
            try:
                for _ in range(warmup):
                    self._execute(
                        model_id,
                        model_snapshot,
                        scenario,
                        sample=first_sample,
                        request=request,
                        repeat=-1,
                        phase=Phase.WARMUP,
                    )
            except RunCancelled:
                # A cancel during the optional per-cell warmup means no measured
                # sample was taken: record every selected sample (AGENTS §13).
                self._record_abort_skips(
                    scenario,
                    label,
                    samples,
                    start_index=0,
                    start_repeat=0,
                    repeats=repeats,
                    reason="cancelled by user",
                )
                raise

        current_sample_index = 0
        current_repeat = 0
        try:
            for sample_index, sample in enumerate(samples):
                current_sample_index = sample_index
                current_repeat = 0
                self._check_cancel()
                if self._cap_reached():
                    self._record_cap_skips(
                        scenario.name, [(cell_dataset, context_length, samples[sample_index:])]
                    )
                    return
                try:
                    request = self._build_request(scenario, spec, model_id, sample, server_ctx)
                except Exception as exc:
                    self._log(f"ERROR building request for {sample.sample_id}: {exc}")
                    self.stats.skipped.append(
                        f"{scenario.name}/{label}/{sample.sample_id}: "
                        f"not-testable ({exc})"
                    )
                    if self.cfg.run.fail_fast:
                        raise
                    continue
                self._write_sample(scenario, sample)
                for repeat in range(repeats):
                    current_repeat = repeat
                    self._check_cancel()
                    if self._cap_reached():
                        for pending_repeat in range(repeat, repeats):
                            self.stats.skipped.append(
                                f"{scenario.name}/{label}/{sample.sample_id}/r{pending_repeat}: "
                                f"not measured (max_requests cap reached, "
                                f"cap={self.cfg.run.max_requests})"
                            )
                        return
                    self._execute(
                        model_id,
                        model_snapshot,
                        scenario,
                        sample=sample,
                        request=request,
                        repeat=repeat,
                        phase=Phase.MEASURED,
                    )
                    if self._cell_abort_reason:
                        message = (
                            f"{model_id}/{scenario.name}: cell aborted: "
                            f"{self._cell_abort_reason}"
                        )
                        self._log(f"WARNING: {message}")
                        self.stats.warnings.append(message)
                        self._record_abort_skips(
                            scenario,
                            label,
                            samples,
                            start_index=sample_index,
                            start_repeat=repeat + 1,
                            repeats=repeats,
                            reason=str(self._cell_abort_reason),
                        )
                        return
        except RunCancelled:
            # User cancel: every selected sample that was not measured must
            # still have a per-sample ``summary.skipped`` entry (AGENTS §13).
            # The top-level handler only adds a generic "cancelled by user"
            # marker, which does not identify which samples were dropped.
            self._record_abort_skips(
                scenario,
                label,
                samples,
                start_index=current_sample_index,
                start_repeat=current_repeat,
                repeats=repeats,
                reason="cancelled by user",
            )
            raise

    # ------------------------------------------------------------------ #
    # one measured request
    # ------------------------------------------------------------------ #

    def _execute(
        self,
        model_id: str,
        model_snapshot: Dict[str, Any],
        scenario: ScenarioAdapter,
        *,
        sample: Sample,
        request: RequestSpec,
        repeat: int,
        phase: Phase,
    ) -> SampleRecord:
        self._check_cancel()
        max_retries = self.cfg.run.max_retries
        # Transport retries and cold-cache contamination retries have separate
        # budgets (docs/METHODOLOGY.md §8).  They used to share one loop counter,
        # which meant a contamination retry could be silently skipped after a
        # couple of transport errors -- the log said "retrying" but no request
        # was made.  ``transport_retries`` only counts transient failures.
        transport_retries = 0
        stream: Optional[StreamResult] = None
        report: Optional[ResourceReport] = None
        state = CacheState(mode=self.cfg.run.cache_mode)
        error: Optional[str] = None
        trace: List[Dict[str, Any]] = []
        # Warm mode measures steady-state cache reuse, so prompts MUST be allowed
        # to populate the prefix cache between the cell's first clear and the
        # measured requests.  Cold mode keeps ``cache_store=False``.
        cache_store = self.cfg.run.cache_mode is CacheMode.WARM
        contamination_attempts = 0

        while True:
            state = self.controller.prepare() if phase is Phase.MEASURED else CacheState(
                mode=self.cfg.run.cache_mode
            )
            if phase is Phase.MEASURED and self._clear_required(state):
                error = f"cache clear failed: {state.note}"
                self._log(f"WARNING: {error}; aborting cell to avoid warm-cache data")
                self._cell_abort_reason = error
                break
            sampler = ResourceSampler(self.server_pid, self.cfg.run.cpu_sample_interval_ms).start()
            try:
                if scenario.agentic and not request.context_meta.get("prebuilt"):
                    stream, trace = self._run_agentic(
                        scenario, sample, request, cache_store=cache_store
                    )
                else:
                    # Pre-built requests are sent exactly once, even for agentic
                    # scenarios: the dataset already contains the full prompt and
                    # the tool loop would change what is being measured.
                    stream = self.client.chat_stream(
                        request, cache_store=cache_store
                    )
                report = sampler.stop()
                error = None
            except Exception as exc:  # noqa: BLE001 - record and retry
                report = sampler.stop()
                error = f"{type(exc).__name__}: {exc}"
                if self._is_memory_error(error):
                    # Memory-guard rejection is deterministic for the current
                    # system state: retrying wastes time and cannot succeed.
                    self.stats.memory_errors += 1
                    if not self._memory_warning_issued:
                        message = (
                            f"oMLX memory guard rejected requests for model {model_id} "
                            f"({sample.sample_id}): free system memory, raise oMLX "
                            "memory_guard_tier, or run a smaller model/shorter samples. "
                            "See docs/USER_GUIDE.md (内存不足)."
                        )
                        self.stats.warnings.append(message)
                        self._log(f"WARNING: {message}")
                        self._memory_warning_issued = True
                    break
                if self._is_tool_schema_error(error) and request.tools:
                    # Dataset quality issue (e.g. non-standard function.parameters
                    # types such as dict/float/tuple).  Retrying cannot help and
                    # rebuilding the dataset needs explicit human approval.
                    if not self._tool_schema_warning_issued:
                        message = (
                            "oMLX rejected the dataset's tools schema for "
                            f"{sample.sample_id} ({len(request.tools)} tool(s)): {error}. "
                            "This is a dataset quality issue (non-standard JSON-schema "
                            "parameter types); retry is pointless. Rebuilding the "
                            "dataset requires human approval (AGENTS §8)."
                        )
                        self.stats.warnings.append(message)
                        self._log(f"WARNING: {message}")
                        self._tool_schema_warning_issued = True
                    break
                if transport_retries < max_retries:
                    transport_retries += 1
                    self._log(
                        f"retry {transport_retries}/{max_retries} after error: {error}"
                    )
                    time.sleep(self.cfg.run.retry_backoff_s * transport_retries)
                    continue
                break

            state = self.controller.evaluate(state, stream.usage)
            if (
                phase is Phase.MEASURED
                and state.polluted
                and contamination_attempts < self.controller.max_contamination_retries
            ):
                contamination_attempts += 1
                if state.cached_tokens is None:
                    reason = "cached_tokens missing in cold mode"
                else:
                    reason = f"cache hit {state.cached_tokens} tokens in cold mode"
                self._log(
                    f"{reason}; retrying with fresh cache "
                    f"clear ({contamination_attempts}/"
                    f"{self.controller.max_contamination_retries})"
                )
                time.sleep(self.cfg.run.retry_backoff_s)
                continue
            break

        status = RecordStatus.OK
        if error is not None:
            status = RecordStatus.TIMEOUT if self._is_timeout(error) else RecordStatus.ERROR
        elif phase is Phase.WARMUP:
            status = RecordStatus.WARMUP
        elif state.polluted:
            status = RecordStatus.CONTAMINATED

        if status is RecordStatus.CONTAMINATED and not self._contamination_warning_issued:
            detail = (
                "cached_tokens missing in cold mode"
                if state.cached_tokens is None
                else f"cache hit {state.cached_tokens} tokens in cold mode"
            )
            message = (
                f"cold-cache contamination kept for {model_id}/{sample.sample_id}: "
                f"{detail}. The sample is excluded from aggregates (fail-closed); "
                "check oMLX cache clearing/admin credentials before comparing runs."
            )
            self.stats.warnings.append(message)
            self._log(f"WARNING: {message}")
            self._contamination_warning_issued = True

        input_context_chars = sum(len(m.content or "") for m in request.messages)
        metrics = PerfMetrics()
        if stream is not None:
            usage = stream.usage
            completion = usage.completion_tokens
            prompt = usage.prompt_tokens
            if prompt is None:
                prompt = request.actual_prompt_tokens
            metrics = compute_metrics(
                ttft_s=stream.first_token_ts,
                first_visible_s=stream.first_visible_ts,
                last_token_s=stream.last_token_ts,
                e2e_s=stream.elapsed_s,
                prompt_tokens=prompt,
                completion_tokens=completion,
                usage=usage,
                input_context_chars=input_context_chars,
            )
            if report is not None:
                metrics.peak_cpu_pct = report.peak_cpu_pct
                metrics.mean_cpu_pct = report.mean_cpu_pct
                metrics.cpu_seconds = report.cpu_seconds
                metrics.peak_rss_mb = report.peak_rss_mb
                metrics.peak_memory_gb = bytes_to_gib(
                    report.peak_phys_footprint_bytes
                )
                if metrics.peak_memory_gb is None and report.peak_rss_mb is not None:
                    # Non-macOS fallback: RSS is the only observable value.
                    metrics.peak_memory_gb = round(report.peak_rss_mb / 1024.0, 2)
                metrics.cpu_samples = report.samples
                metrics.cpu_sample_window_ms = report.window_ms
            metrics.cached_tokens = usage.cached_tokens

        record = SampleRecord(
            run_id=self.writer.run_id,
            model=model_id,
            model_config_snapshot=model_snapshot,
            scenario=scenario.name,
            context_length=self._context_length(sample),
            dataset=sample.dataset,
            sample_id=sample.sample_id,
            repeat=repeat,
            phase=phase,
            status=status,
            error=error,
            request=request,
            response=self._to_raw_response(stream) if stream else RawResponse(),
            metrics=metrics,
            score=None,  # performance mode: no accuracy scoring
            cache=state,
            sample_meta=self._sample_meta(scenario, sample, request, trace),
        )
        self.writer.write_record(record)
        self.stats.records.append(record)
        self._progress(
            "record",
            {
                "model": model_id,
                "scenario": scenario.name,
                "dataset": sample.dataset,
                "context_length": self._context_length(sample),
                "sample_id": sample.sample_id,
                "repeat": repeat,
                "status": status.value,
                "input_context_tokens": metrics.input_context_tokens,
                "ttft_ms": metrics.ttft_ms,
                "gen_tps": metrics.gen_tps,
                "peak_rss_mb": metrics.peak_rss_mb,
                "peak_memory_gb": metrics.peak_memory_gb,
            },
        )

        if status is RecordStatus.OK:
            self._log(
                f"{scenario.name}/{sample.sample_id} r{repeat} "
                f"ctx_tokens={metrics.input_context_tokens} | "
                f"ttft={metrics.ttft_ms}ms e2e={metrics.e2e_ms}ms "
                f"tps={metrics.gen_tps} mem={metrics.peak_memory_gb}GB"
            )
        elif status is RecordStatus.WARMUP:
            self._log(
                f"warmup {scenario.name} | ctx_tokens={metrics.input_context_tokens} "
                f"ttft={metrics.ttft_ms}ms"
            )
        else:
            self._log(f"{status.value} {scenario.name} {sample.sample_id} r{repeat}: {error or state.note}")
        if self.cfg.run.fail_fast and status in (RecordStatus.ERROR, RecordStatus.TIMEOUT):
            raise RuntimeError(
                f"fail_fast: {scenario.name}/{sample.sample_id} ended with {status.value}: {error}"
            )
        return record

    # ------------------------------------------------------------------ #
    # agentic loop (search agent)
    # ------------------------------------------------------------------ #

    def _run_agentic(
        self,
        scenario: ScenarioAdapter,
        sample: Sample,
        request: RequestSpec,
        *,
        cache_store: bool = False,
    ) -> Tuple[StreamResult, List[Dict[str, Any]]]:
        messages = list(request.messages)
        trace: List[Dict[str, Any]] = []
        t_start = time.perf_counter()
        total_prompt = 0
        total_completion = 0
        total_cached = 0
        cached_missing = False
        first_ttft: Optional[float] = None
        first_visible: Optional[float] = None
        decode_total = 0.0
        reasoning_parts: List[str] = []
        last_text = ""
        finish_answer: Optional[str] = None
        finish_reason: Optional[str] = None
        max_steps = int(scenario.params.get("max_steps", 3))

        for step in range(max_steps):
            spec = request.model_copy(deep=True)
            spec.messages = messages
            stream = self.client.chat_stream(spec, cache_store=cache_store)
            usage = stream.usage
            total_prompt += usage.prompt_tokens or 0
            total_completion += usage.completion_tokens or 0
            if usage.cached_tokens is None:
                cached_missing = True
            else:
                total_cached += usage.cached_tokens
            if first_ttft is None:
                first_ttft = stream.first_token_ts
                first_visible = stream.first_visible_ts
            if stream.first_token_ts is not None and stream.last_token_ts is not None:
                decode_total += max(stream.last_token_ts - stream.first_token_ts, 0.0)
            reasoning_parts.append(stream.reasoning_text)
            last_text = stream.text or last_text
            finish_reason = stream.finish_reason or finish_reason

            calls = stream.tool_calls
            trace.append(
                {
                    "step": step,
                    "text": (stream.text or "")[:500],
                    "tool_calls": [
                        {"name": c.name, "arguments": c.arguments[:500]} for c in calls
                    ],
                    "ttft_ms": round((stream.first_token_ts or 0) * 1000, 1),
                    "usage": usage.model_dump(exclude_none=True),
                }
            )
            if not calls:
                break
            assistant = ChatMessage(
                role="assistant",
                content=stream.text or None,
                tool_calls=[
                    {
                        "id": c.id or f"call_{step}_{idx}",
                        "type": "function",
                        "function": {"name": c.name, "arguments": c.arguments},
                    }
                    for idx, c in enumerate(calls)
                ],
            )
            messages.append(assistant)
            for idx, call in enumerate(calls):
                call_id = call.id or f"call_{step}_{idx}"
                output = scenario.execute_tool(sample, call)
                if call.name == "finish":
                    try:
                        payload = json.loads(call.arguments or "{}")
                    except json.JSONDecodeError:
                        payload = {}
                    answer = payload.get("answer")
                    if isinstance(answer, str) and answer.strip():
                        finish_answer = answer.strip()
                messages.append(
                    ChatMessage(
                        role="tool",
                        content=str(output)[:4000],
                        tool_call_id=call_id,
                        name=call.name,
                    )
                )

        elapsed = time.perf_counter() - t_start
        combined_usage = Usage(
            prompt_tokens=total_prompt or None,
            completion_tokens=total_completion or None,
            total_tokens=(total_prompt + total_completion) or None,
            cached_tokens=None if cached_missing else total_cached,
        )
        combined = StreamResult(
            text=finish_answer or last_text,
            reasoning_text="".join(reasoning_parts),
            tool_calls=[],
            finish_reason=finish_reason,
            usage=combined_usage,
            num_chunks=0,
            first_token_ts=first_ttft,
            first_visible_ts=first_visible,
            last_token_ts=(first_ttft + decode_total) if first_ttft is not None else None,
            elapsed_s=elapsed,
        )
        trace.append({"final_answer": combined.text[:500]})
        return combined, trace

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_raw_response(stream: StreamResult) -> RawResponse:
        return RawResponse(
            text=stream.text,
            reasoning_text=stream.reasoning_text,
            tool_calls=stream.tool_calls,
            finish_reason=stream.finish_reason,
            usage=stream.usage,
            num_chunks=stream.num_chunks,
            first_token_ts=stream.first_token_ts,
            first_visible_ts=stream.first_visible_ts,
            last_token_ts=stream.last_token_ts,
        )

    def _sample_meta(
        self,
        scenario: ScenarioAdapter,
        sample: Sample,
        request: RequestSpec,
        trace: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        keep = {
            k: v
            for k, v in (sample.metadata or {}).items()
            if k
            in {
                "domain",
                "sub_domain",
                "difficulty",
                "length_category",
                "language",
                "lang",
                "category",
                "subject",
                "bucket",
            }
        }
        meta: Dict[str, Any] = {
            "dataset_meta": keep,
            "subtask": sample.subtask,
            "target_context_tokens": None,
            "input_context": {
                "prompt_tokens": request.actual_prompt_tokens,
                "chars": sum(len(m.content or "") for m in request.messages),
                "truncated": request.truncated,
                "truncation_ratio": request.context_meta.get("truncation_ratio"),
            },
            "raw_sample_path": self._sample_relpath(scenario, sample),
        }
        if trace:
            meta["agent_trace"] = trace
        return meta

    def _sample_relpath(self, scenario: ScenarioAdapter, sample: Sample) -> str:
        # Samples are context-length independent at this level: the original
        # (untruncated) sample is written once and every record points at it.
        return str(
            Path("raw")
            / "samples"
            / sanitize_name(sample.dataset)
            / sanitize_name(scenario.name)
            / f"{sanitize_name(sample.sample_id)}.json"
        )

    def _clear_required(self, state: CacheState) -> bool:
        """A cold measurement is only valid if the cache clear actually happened.

        In ``warm`` mode only the first request of a cell is cleared; later
        requests legitimately report ``cleared_before_request=False`` because
        they measure cache reuse.
        """
        if self.cfg.run.cache_mode is CacheMode.COLD:
            return not state.cleared_before_request
        if self.cfg.run.cache_mode is CacheMode.WARM:
            return not state.cleared_before_request and not self.controller.warm_cleared
        return False

    def _write_sample(self, scenario: ScenarioAdapter, sample: Sample) -> None:
        rel = self._sample_relpath(scenario, sample)
        path = self.writer.run_dir / rel
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(sample.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
