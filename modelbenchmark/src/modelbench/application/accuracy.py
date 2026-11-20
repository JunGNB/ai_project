"""Accuracy orchestrator: reuse the measurement kernel, add judging.

``AccuracyOrchestrator`` inherits the whole performance kernel from
:class:`~modelbench.application.benchmark.Orchestrator` — model lifecycle,
default-config repair, cold-cache governance, single-request execution/retries,
resource sampling, cancellation, progress and the run bundle — and overrides
only the method hooks declared in ``docs/design/32-method-architecture.md`` §7:

* :meth:`_scenario_for` — a lightweight accuracy scenario (no perf rendering);
* :meth:`_filter_by_context` — accuracy has no 1K/8K/32K ISL buckets;
* :meth:`_select_samples` — ``cfg.accuracy`` selectors (stratified; all by default);
* :meth:`_group_samples` — one cell per scenario;
* :meth:`_build_request` — pre-built ``messages`` verbatim, ``max_tokens`` from
  ``judge.max_new_tokens``, ``temperature=0``/``top_p=1``, window check outside
  the measured window, **no decode guard**;
* :meth:`_postprocess` — deterministic judge, ``status=judge_error`` on failure;
* :meth:`_sample_meta` / :meth:`_record_log` — judging evidence fields.

Run protocol is fixed by ``docs/spec/22-accuracy-test-plan.md`` §2 and enforced
by ``application.methods`` (:class:`~modelbench.application.methods.MethodProfile`).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from modelbench.adapters.config.settings import ScenarioSpec
from modelbench.application.benchmark import Orchestrator, thinking_control
from modelbench.domain.judges import (
    JUDGE_FAMILIES,
    evaluate,
    missing_assets,
    missing_dependencies,
)
from modelbench.domain.models import (
    ChatMessage,
    PerfMetrics,
    RecordStatus,
    RequestSpec,
    Sample,
    Score,
)

#: Group/cell key for the single accuracy cell of a scenario.  The base kernel
#: uses the group key as the cell dataset label in run.log/progress/skip
#: reasons; accuracy groups by scenario (dataset splits live in the aggregates),
#: so a stable method label avoids the duplicated ``scenario/scenario`` wording.
ACCURACY_CELL_DATASET = "accuracy"


class AccuracyScenario:
    """Minimal scenario stand-in for the accuracy method.

    The accuracy test set already contains the exact chat request; the scenario
    layer only supplies a name and the ``agentic=False`` flag the kernel reads.
    Instantiating a performance scenario here would be misleading (and could try
    to render the pre-built prompt again).
    """

    dataset_name = "accuracy_jsonl"
    agentic = False
    scoring = True

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        self.name = name
        self.params = dict(params or {})

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "dataset": self.dataset_name, "params": self.params}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<AccuracyScenario {self.name}>"


class AccuracyOrchestrator(Orchestrator):
    """Per-sample deterministic judging on the shared measurement kernel."""

    method_name = "accuracy"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._judge_error_warned = False
        self._judge_dependencies: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # planning / logging hooks
    # ------------------------------------------------------------------ #

    def _plan_log_fields(self) -> str:
        return (
            f"samples_per_scenario="
            f"{self.cfg.accuracy.samples_per_scenario or 'all'} "
        )

    def prepare(self):
        """Prepare the run, then surface missing optional judge dependencies."""
        manifest = super().prepare()
        self._warn_missing_judge_dependencies()
        return manifest

    def _judge_families(self) -> List[str]:
        """Judge families the accuracy test set declares (manifest, else all)."""
        testset_dir = Path(str(getattr(self.cfg.datasets, "testset_dir", "dataset/test")))
        manifest_path = testset_dir / "accuracy" / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return list(JUDGE_FAMILIES)
        families = (manifest or {}).get("judges")
        if isinstance(families, dict) and families:
            return sorted(str(name) for name in families)
        return list(JUDGE_FAMILIES)

    def _warn_missing_judge_dependencies(self) -> None:
        """Write an explicit warning when a used judge family is unavailable.

        A missing judge dependency or on-disk asset must not silently drop the
        scenario from the denominator: the affected samples become
        ``judge_error`` (0 points, counted) and the run carries both lists
        (docs/spec/22 §3.2).
        """
        families = self._judge_families()
        try:
            missing = missing_dependencies(families)
            # NOTE: do not shadow the imported ``missing_assets`` helper here —
            # assigning to the same name makes Python treat it as a local for the
            # whole function and the call above raises UnboundLocalError (that
            # bug surfaced as a "probe failed" warning in the 2026-11-20 run).
            missing_asset_names = missing_assets(families)
        except Exception as exc:  # noqa: BLE001 - advisory probe must not abort a run
            # Never swallow the probe failure: keep the used families and put
            # the error into the persisted judge-dependency block + warning.
            self._judge_dependencies = {
                "used": list(families),
                "missing": [],
                "missing_assets": [],
                "probe_error": f"{type(exc).__name__}: {exc}",
            }
            message = (
                "accuracy judge dependency probe failed: "
                f"{type(exc).__name__}: {exc}; affected samples may become "
                "status=judge_error"
            )
            if message not in self.stats.warnings:
                self.stats.warnings.append(message)
                self._log(f"WARNING: {message}")
            return
        self._judge_dependencies = {
            "used": list(families),
            "missing": list(missing),
            "missing_assets": list(missing_asset_names),
        }
        if not missing and not missing_asset_names:
            return
        parts: List[str] = []
        if missing:
            parts.append("modules: " + ", ".join(missing))
        if missing_asset_names:
            parts.append("assets: " + ", ".join(missing_asset_names))
        message = (
            "accuracy judge dependencies missing ("
            + "; ".join(parts)
            + "); affected samples will be status=judge_error. Fix: reinstall "
            "with ./modelbench setup (or pip install -e .): the judge deps are "
            "core dependencies and the IFEval checker + English punkt data ship "
            "inside the package; only a testset rebuild needs "
            "python tool/dataset/accuracy/fetch.py"
        )
        if message not in self.stats.warnings:
            self.stats.warnings.append(message)
            self._log(f"WARNING: {message}")

    def _finalize_extra(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        """Add the accuracy evidence the writer's report needs (testset/judges)."""
        out = dict(extra)
        try:
            from modelbench.application.methods import get_method

            out["testset"] = get_method(self.cfg.method).testset_fingerprint(self.cfg)
        except Exception as exc:  # noqa: BLE001 - report evidence is advisory
            self._log(f"WARNING: cannot fingerprint accuracy test set: {exc}")
        if self._judge_dependencies:
            out["judge_dependencies"] = dict(self._judge_dependencies)
        return out

    def _estimate_requests(self) -> Optional[int]:
        """Measured-request estimate from the accuracy selectors."""
        from modelbench.application.plan import scenario_plan_extras

        total = 0
        for name, spec in self.cfg.selected_scenarios().items():
            plan = self._dataset_plan(spec, scenario_plan_extras(self.cfg, name))
            if plan is not None:
                total += sum(int(item["n"]) for item in plan)
                continue
            explicit = self.cfg.accuracy.sample_ids_for(name)
            count = self.cfg.accuracy.samples_for(name)
            if explicit:
                total += len(explicit)
            elif count is not None:
                total += int(count)
            else:
                return None
        return total * len(self.cfg.selected_models()) * int(self.cfg.run.repeats)

    # ------------------------------------------------------------------ #
    # scenario / sample selection
    # ------------------------------------------------------------------ #

    def _scenario_for(self, name: str, spec: ScenarioSpec) -> AccuracyScenario:
        """Accuracy never renders prompts: a lightweight named scenario."""
        return AccuracyScenario(name, spec.params)

    def _filter_by_context(
        self,
        scenario_name: str,
        scenario: AccuracyScenario,
        spec: ScenarioSpec,
        model_id: str,
        samples: List[Sample],
        server_ctx: Optional[int],
    ) -> List[Sample]:
        """Accuracy has no ISL bucket filter: every selected sample is measured."""
        if self.cfg.perf.context_windows:
            self._log(
                "WARNING: perf.context_windows is ignored by method=accuracy "
                "(accuracy has no 1K/8K/32K buckets)"
            )
        return samples

    def _selected_spec(self, scenario_name: str) -> Optional[ScenarioSpec]:
        spec = self.cfg.selected_scenarios().get(scenario_name)
        if spec is None:
            spec = self.cfg.scenarios.get(scenario_name)
        return spec

    def _select_samples(self, scenario_name: str, samples: List[Sample]) -> List[Sample]:
        """Apply ``cfg.accuracy`` selectors (explicit ids, then stratified count)."""
        explicit = self.cfg.accuracy.sample_ids_for(scenario_name)
        count = self.cfg.accuracy.samples_for(scenario_name)
        params: Dict[str, Any] = {}
        if explicit:
            params["sample_ids"] = list(explicit)
        elif count is not None:
            params["samples"] = int(count)
        spec = self._selected_spec(scenario_name)
        selector = None
        if spec is not None:
            selector = getattr(self._adapter_for(spec), "select_samples", None)
        if selector is None:
            return list(samples)
        try:
            chosen = selector(samples, params)
        except (TypeError, ValueError) as exc:
            # Fail closed: a broken selector must not silently measure everything.
            raise ValueError(
                f"scenario {scenario_name}: accuracy sample selection failed: {exc}"
            ) from exc
        if explicit:
            missing = [
                sample_id
                for sample_id in explicit
                if sample_id not in {sample.sample_id for sample in chosen}
            ]
            if missing:
                message = (
                    f"scenario {scenario_name}: {len(missing)} selected sample id(s) "
                    f"not found: {missing[:5]}"
                )
                self._log(f"WARNING: {message}")
                self.stats.warnings.append(message)
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
        how = "all" if count is None or count >= len(samples) else f"first {count} stratified"
        self._log(
            f"dataset {scenario_name}: selected {len(chosen)} of {len(samples)} samples "
            f"({how})"
        )
        return chosen

    @classmethod
    def _cell_label(cls, dataset: str, context_length: Optional[int]) -> str:
        """Accuracy has one cell per scenario; label it by method, not twice."""
        return "accuracy"

    def _group_samples(
        self, samples: List[Sample]
    ) -> Dict[Any, List[Sample]]:
        """Accuracy aggregates one cell per scenario (no dataset/bucket split)."""
        if not samples:
            return {}
        return {(ACCURACY_CELL_DATASET, None): list(samples)}

    # ------------------------------------------------------------------ #
    # request construction (see docs/design/32-method-architecture.md §7)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _messages_for(sample: Sample) -> List[ChatMessage]:
        """Convert the pre-built JSON messages to the domain chat model verbatim."""
        rows = (sample.metadata or {}).get("messages") or []
        if not isinstance(rows, list) or not rows:
            raise ValueError(
                f"accuracy sample {sample.sample_id} has no pre-built messages"
            )
        messages: List[ChatMessage] = []
        for row in rows:
            content = row.get("content")
            if content is not None and not isinstance(content, str):
                # Multimodal content is out of scope for the accuracy set; keep
                # the exact JSON text rather than dropping the message.
                content = json.dumps(content, ensure_ascii=False)
            messages.append(
                ChatMessage(
                    role=str(row.get("role") or "user"),
                    content=content,
                    tool_call_id=row.get("tool_call_id"),
                    name=row.get("name"),
                    tool_calls=row.get("tool_calls"),
                )
            )
        return messages

    def _build_request(
        self,
        scenario: AccuracyScenario,
        spec: ScenarioSpec,
        model_id: str,
        sample: Sample,
        server_ctx: Optional[int],
    ) -> RequestSpec:
        """Build the judged request: messages verbatim, per-row generation cap."""
        key = (model_id, sample.sample_id, spec.name)
        cached = self._request_cache.get(key)
        if cached is not None:
            return cached

        metadata = sample.metadata or {}
        judge = metadata.get("judge")
        if not isinstance(judge, dict) or not judge:
            raise ValueError(f"accuracy sample {sample.sample_id} has no judge contract")
        try:
            max_tokens = int(judge.get("max_new_tokens"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"accuracy sample {sample.sample_id}: judge.max_new_tokens must be "
                f"an integer (got {judge.get('max_new_tokens')!r})"
            ) from exc
        if max_tokens < 1:
            raise ValueError(
                f"accuracy sample {sample.sample_id}: judge.max_new_tokens must be >= 1"
            )
        family = str(judge.get("family") or "")

        request = RequestSpec(
            model=model_id,
            messages=self._messages_for(sample),
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            stream=True,
            tools=None,
            # No target bucket, no decode guard: the exact pre-built prompt is
            # what gets scored (a guard would change the answer being judged).
            target_context_tokens=None,
            actual_prompt_tokens=(
                int(metadata["input_tokens"])
                if metadata.get("input_tokens") is not None
                else None
            ),
            context_meta={
                "prebuilt": True,
                "judge_family": family,
                "max_new_tokens": max_tokens,
                # Accuracy does not use the ±10% ISL band; the key stays null so
                # readers can tell "not applicable" from "missing".
                "isl_tolerance": None,
            },
        )
        thinking_kwargs, thinking_label = thinking_control(self.cfg, model_id)
        if thinking_kwargs:
            request.chat_template_kwargs = dict(thinking_kwargs)
        request.context_meta["thinking_control"] = thinking_label

        # Tokenize the final prompt before the measured window starts: the count
        # decides whether the row is testable at all (over-window rows are
        # skipped explicitly, never truncated).
        tokenized: Optional[int] = None
        tokenize_started = time.perf_counter()
        try:
            tokenized = int(
                self.client.tokenize_messages(
                    request.messages, model=model_id, tools=request.tools
                )
            )
            request.tokenized_prompt_tokens = tokenized
            request.context_meta["tokenized_prompt_tokens"] = tokenized
        except Exception as exc:  # noqa: BLE001 - fall back to the declared count
            request.context_meta["tokenize_error"] = f"{type(exc).__name__}: {exc}"
            if not self._tokenize_warning_issued:
                message = (
                    f"target tokenizer unavailable for {model_id}: {exc}; falling "
                    "back to the declared prompt tokens for the accuracy window check"
                )
                self.stats.warnings.append(message)
                self._log(f"WARNING: {message}")
                self._tokenize_warning_issued = True
        request.context_meta["tokenize_latency_ms"] = round(
            (time.perf_counter() - tokenize_started) * 1000.0, 2
        )

        if server_ctx:
            window = int(server_ctx)
            request.context_meta["model_context_window"] = window
            if tokenized is not None:
                basis, counted = "tokenized", tokenized
            else:
                basis, counted = "declared", request.actual_prompt_tokens
            if counted is not None:
                request.context_meta["window_check_basis"] = basis
                room = (
                    window
                    - int(counted)
                    - max_tokens
                    - int(self.cfg.run.context_safety_margin_tokens)
                )
                if room < 0:
                    raise ValueError(
                        f"over_window: {basis} prompt {int(counted)} + "
                        f"max_new_tokens {max_tokens} + safety "
                        f"{int(self.cfg.run.context_safety_margin_tokens)} exceeds "
                        f"model context window {window} (not truncating; sample is "
                        "not-testable)"
                    )

        self._request_cache[key] = request
        return request

    # ------------------------------------------------------------------ #
    # scoring hooks
    # ------------------------------------------------------------------ #

    def _postprocess(
        self,
        *,
        scenario: AccuracyScenario,
        sample: Sample,
        stream: Any,
        error: Optional[str],
        status: RecordStatus,
    ):
        """Judge one response; judge failures become ``status=judge_error``."""
        if status is not RecordStatus.OK or stream is None:
            return None, status
        metadata = sample.metadata or {}
        judge = metadata.get("judge")
        if not isinstance(judge, dict) or not judge:
            return (
                Score(
                    correct=False,
                    score=0.0,
                    detail={
                        "family": None,
                        "metric": None,
                        "pass": False,
                        "format_failure": False,
                        "judge_error": True,
                        "reason": "missing judge contract",
                    },
                ),
                RecordStatus.JUDGE_ERROR,
            )
        row = {"judge": judge}
        try:
            result = evaluate(row, stream.text)
        except Exception as exc:  # noqa: BLE001 - a judge must never abort the run
            if not self._judge_error_warned:
                message = (
                    f"judge raised for {sample.sample_id}; counted as judge_error "
                    f"(0 points, kept in the denominator): "
                    f"{type(exc).__name__}: {exc}"
                )
                self.stats.warnings.append(message)
                self._log(f"WARNING: {message}")
                self._judge_error_warned = True
            return (
                Score(
                    correct=False,
                    score=0.0,
                    detail={
                        "family": judge.get("family"),
                        "metric": judge.get("metric", "accuracy"),
                        "pass": False,
                        "format_failure": False,
                        "judge_error": True,
                        "reason": f"{type(exc).__name__}: {exc}",
                    },
                ),
                RecordStatus.JUDGE_ERROR,
            )

        detail = dict(result.detail or {})
        score = Score(
            correct=bool(result.passed),
            score=float(result.score),
            detail=detail,
        )
        if detail.get("judge_error"):
            return score, RecordStatus.JUDGE_ERROR
        return score, status

    # ------------------------------------------------------------------ #
    # record metadata / log line
    # ------------------------------------------------------------------ #

    def _sample_meta(
        self,
        scenario: AccuracyScenario,
        sample: Sample,
        request: RequestSpec,
        trace: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        metadata = sample.metadata or {}
        judge = metadata.get("judge") or {}
        return {
            "target_context_tokens": None,
            "input_context": {
                "prompt_tokens": request.actual_prompt_tokens,
                "tokenized_prompt_tokens": request.tokenized_prompt_tokens,
                "chars": sum(len(message.content or "") for message in request.messages),
                "truncated": False,
                "truncation_ratio": None,
            },
            "dataset_meta": dict(metadata.get("dataset_meta") or {}),
            "language": metadata.get("language"),
            "upstream_id": metadata.get("upstream_id"),
            "judge": {
                "family": judge.get("family"),
                "metric": judge.get("metric"),
                "max_new_tokens": judge.get("max_new_tokens"),
            },
            "subtask": None,
        }

    def _record_log(
        self,
        *,
        scenario: AccuracyScenario,
        sample: Sample,
        repeat: int,
        status: RecordStatus,
        error: Optional[str],
        metrics: PerfMetrics,
        state: Any,
        score: Optional[Score],
    ) -> None:
        if score is not None:
            detail = score.detail or {}
            self._log(
                f"{scenario.name}/{sample.sample_id} r{repeat} "
                f"score={score.score} judge={detail.get('family')} "
                f"format_failure={bool(detail.get('format_failure'))} "
                f"judge_error={bool(detail.get('judge_error'))}"
                + (f" status={status.value}" if status is not RecordStatus.OK else "")
            )
        elif status is RecordStatus.WARMUP:
            self._log(
                f"warmup {scenario.name} | ctx_tokens={metrics.input_context_tokens}"
            )
        else:
            self._log(
                f"{status.value} {scenario.name} {sample.sample_id} r{repeat}: "
                f"{error or state.note}"
            )


__all__ = ["AccuracyOrchestrator", "AccuracyScenario"]
