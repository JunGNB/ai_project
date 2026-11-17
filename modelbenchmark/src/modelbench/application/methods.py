"""Test-method registry: one measurement kernel, two declared methods.

``performance`` (speed/memory, ``score`` stays null) and ``accuracy``
(deterministic per-sample judging) share the runner, writer, locks and
environment evidence; they differ only in *what is measured, how it is scored
and how it is aggregated*.  Everything method-specific is declared here so the
runner, batch layer, Web and CLI dispatch through one registry instead of
sprinkling ``if method == ...`` across the code base.

Design contract: ``docs/design/32-method-architecture.md`` §4.2.  The accuracy
run protocol (cold cache, ``repeats=1``, one transport retry, no per-cell
warmup) is fixed by ``docs/spec/22-accuracy-test-plan.md`` §2 and is enforced by
:meth:`MethodProfile.configure` — a caller asking for a protocol the method does
not support gets a ``ValueError`` instead of a silently different calibre.

Import discipline: this module is imported by the composition roots (``plan``,
``batch``, CLI/Web), so the heavy pieces (orchestrator, writer, plan) are
imported lazily inside the profile methods.  That keeps the registry importable
without pulling in oMLX adapters and breaks the ``plan <-> methods`` cycle.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from modelbench.adapters.config.settings import AccuracyConfig, ProjectConfig
from modelbench.adapters.storage.blobstore import DECODE_GUARD_VERSION
from modelbench.domain.method import RunMethod
from modelbench.domain.models import CacheMode

#: Accuracy run protocol (fixed; documented in docs/spec/22 §2).
ACCURACY_REPEATS = 1
ACCURACY_MAX_RETRIES = 1
ACCURACY_WARMUP_REQUESTS = 0

#: Repository root, used to persist portable (relative) test-set paths.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: Web parameter-panel metadata schema (``params`` in :meth:`method_metadata`):
#: ``{"name", "label", "type", "default", "hint"}`` where ``type`` is one of
#: ``int``/``bool``/``enum``/``multi_enum``/``str``.
_PERF_PARAMS: Tuple[Dict[str, Any], ...] = (
    {
        # Payload/CLI name; the config field is ``perf.samples_per_dataset``.
        "name": "samples",
        "label": "每个场景样本数",
        "type": "int",
        "default": None,
        "hint": "null = 全部可用样本（性能测试集共 240 条）",
    },
    {
        # Leave blank: the test set's declared value applies (performance = 50).
        # The config default (50) only applies to test sets that declare nothing.
        "name": "max_tokens",
        "label": "生成长度 (OSL)",
        "type": "int",
        "default": None,
        "hint": "留空 = 测试集声明（performance 集为 50）；填写则覆盖生成长度，"
        "不影响输入对齐，触顶样本不丢弃",
    },
    {
        "name": "repeats",
        "label": "重复次数",
        "type": "int",
        "default": 1,
        "hint": "同一 cell 内重复测量次数",
    },
    {
        "name": "cache_mode",
        "label": "缓存模式",
        "type": "enum",
        "default": "cold",
        "options": ["cold", "warm", "none"],
        "hint": "cold = 每个 measured 请求前清 hot+SSD 缓存",
    },
    {
        "name": "warmup_per_model",
        "label": "模型预热次数",
        "type": "int",
        "default": 1,
        "hint": "模型加载后的一次性短请求（丢弃，不聚合）",
    },
    {
        "name": "thinking_mode",
        "label": "思考模式",
        "type": "enum",
        "default": "disabled",
        "options": ["disabled", "auto", "enabled"],
        "hint": "默认关闭思考，与 oMLX 默认口径一致",
    },
    {
        "name": "context_windows",
        "label": "上下文档位",
        "type": "multi_enum",
        "default": [],
        "options": [1024, 4096, 32768],
        "hint": "空 = 不筛选；仅选择样本，不改变 cell 键",
    },
)

_ACCURACY_PARAMS: Tuple[Dict[str, Any], ...] = (
    {
        # Payload/CLI name; the config field is ``accuracy.samples_per_scenario``.
        "name": "samples",
        "label": "每个场景样本数",
        "type": "int",
        "default": None,
        "hint": "null = 全部 250 条/场景（固定 8×250 样本集，不裁剪）",
    },
    {
        "name": "warmup_per_model",
        "label": "模型级预热",
        "type": "int",
        "default": 1,
        "hint": "模型加载后的一次性短请求（丢弃、不判分）",
    },
    {
        "name": "thinking_mode",
        "label": "思考模式",
        "type": "enum",
        "default": "disabled",
        "options": ["disabled", "auto", "enabled"],
        "hint": "默认关闭思考；开启可能耗尽 max_new_tokens 计入 format_failure",
    },
)


# --------------------------------------------------------------------------- #
# Fingerprint helpers
# --------------------------------------------------------------------------- #


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_rows(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for _line in handle:
            count += 1
    return count


def _display_path(path: Path) -> Optional[str]:
    """Project-relative POSIX path, or ``None`` when the file is outside the repo.

    ``batch.json`` must stay movable, so absolute machine-specific paths are
    never persisted; for an external test set only its digest is kept.
    """
    try:
        return path.resolve().relative_to(_PROJECT_ROOT).as_posix()
    except (ValueError, OSError):
        return None


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def judge_contract_version(manifest: Optional[Dict[str, Any]]) -> str:
    """Stable version tag for the judge contract a run was scored with.

    The accuracy ``manifest.json`` describes the five families instead of
    carrying an explicit contract version, so the version is derived from that
    declaration: ``judges-<sha256(judges)[:16]>``.  Two runs are comparable only
    when this tag and the test-set SHA agree (AC-A8).
    """
    from modelbench.domain.judges import JUDGE_FAMILIES

    families = (manifest or {}).get("judges") or list(JUDGE_FAMILIES)
    body = json.dumps(families, sort_keys=True, ensure_ascii=False)
    return "judges-" + hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# MethodProfile
# --------------------------------------------------------------------------- #


class MethodProfile:
    """Declarative description of one test method (see design §4.2)."""

    #: Canonical method name; also the ``run_id``/``batch.json`` method segment.
    name: RunMethod = RunMethod.PERFORMANCE
    #: Short human title for the Web method card.
    title: str = ""
    #: Default named test set for this method.
    default_testset: str = ""
    #: One-paragraph user-facing description.
    description: str = ""
    #: True when the method scores responses with domain judges.
    needs_judges: bool = False
    #: Config block this method reads/writes (``"perf"`` / ``"accuracy"``).
    config_block: str = "perf"
    #: Web parameter-panel metadata (schema documented above).
    params: Tuple[Dict[str, Any], ...] = ()
    #: Ordered Web result tabs for this method (exposed via ``result_views()``).
    _result_views: Tuple[str, ...] = ()

    # ---- behaviour -----------------------------------------------------

    def configure(self, cfg: ProjectConfig) -> None:
        """Apply the method's fixed defaults/protocol to ``cfg`` (in place)."""
        raise NotImplementedError

    def make_orchestrator(self, cfg: ProjectConfig, client: Any, writer: Any, **kwargs: Any):
        """Instantiate the method's orchestrator for one model run."""
        raise NotImplementedError

    def make_writer(self, *args: Any, **kwargs: Any):
        """Instantiate the method's run writer (same signature as ``RunWriter``)."""
        raise NotImplementedError

    def plan(self, cfg: ProjectConfig) -> Dict[str, Any]:
        """Return the method's :func:`application.plan.plan_overview` payload."""
        raise NotImplementedError

    def testset_fingerprint(self, cfg: ProjectConfig) -> Dict[str, Any]:
        """Identify the test set (and judge contract) a run consumes."""
        raise NotImplementedError

    def result_views(self) -> List[str]:
        """Ordered Web result tabs for this method (design §9.2)."""
        return list(self._result_views)

    # ---- metadata ------------------------------------------------------

    def metadata(self) -> Dict[str, Any]:
        """Web/CLI method card payload (stable keys, see design §9.1)."""
        return {
            "name": self.name.value,
            "title": self.title,
            "description": self.description,
            "default_testset": self.default_testset,
            "needs_judges": self.needs_judges,
            "config_block": self.config_block,
            "params": [dict(item) for item in self.params],
            "result_views": self.result_views(),
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<MethodProfile {self.name.value}>"


class _PerformanceMethod(MethodProfile):
    """Speed/memory method: the historical behaviour, unchanged."""

    name = RunMethod.PERFORMANCE
    title = "性能测试"
    default_testset = "performance"
    description = (
        "测量吞吐与内存：8 场景 × 1K/8K/32K，冷缓存逐样本测量 TTFT/TG TPS/峰值内存。"
        "不判分（score 恒为 null），指标口径见 docs/spec/21-test-plan.md。"
    )
    needs_judges = False
    config_block = "perf"
    params = _PERF_PARAMS
    _result_views = ("overall", "cells", "samples", "environment", "report")

    def configure(self, cfg: ProjectConfig) -> None:
        """Performance behaviour is the default: nothing to force."""
        return None

    def make_orchestrator(self, cfg: ProjectConfig, client: Any, writer: Any, **kwargs: Any):
        from modelbench.application.benchmark import Orchestrator

        return Orchestrator(cfg, client, writer, **kwargs)

    def make_writer(self, *args: Any, **kwargs: Any):
        from modelbench.adapters.storage.writer import RunWriter

        return RunWriter(*args, **kwargs)

    def plan(self, cfg: ProjectConfig) -> Dict[str, Any]:
        from modelbench.application.plan import plan_overview_performance

        return plan_overview_performance(cfg)

    def testset_fingerprint(self, cfg: ProjectConfig) -> Dict[str, Any]:
        testset_dir = Path(str(getattr(cfg.datasets, "testset_dir", "dataset/test")))
        candidate = testset_dir / self.default_testset / "testset.jsonl"
        info: Dict[str, Any] = {
            "path": _display_path(candidate),
            "sha256": None,
            "rows": None,
            "context_windows": list(cfg.perf.context_windows or []),
            "decode_guard_version": DECODE_GUARD_VERSION,
        }
        try:
            if candidate.is_file():
                info["sha256"] = _sha256_file(candidate)
                info["rows"] = _count_rows(candidate)
        except OSError:
            pass
        return info


class _AccuracyMethod(MethodProfile):
    """Per-sample deterministic judging method (docs/spec/22)."""

    name = RunMethod.ACCURACY
    title = "精度测试"
    default_testset = "accuracy"
    description = (
        "测量答得对不对：8 场景 × 250 条固定样本，temperature=0/冷缓存/单次生成，"
        "逐行确定性判题（5 族），输出场景分、宏平均与配对 Δ。"
    )
    needs_judges = True
    config_block = "accuracy"
    params = _ACCURACY_PARAMS
    _result_views = (
        "overall",
        "cells",
        "samples",
        "errors",
        "environment",
        "report",
    )

    def configure(self, cfg: ProjectConfig) -> None:
        """Enforce the fixed accuracy run protocol; reject incompatible calibres."""
        run = cfg.run
        if run.cache_mode is not CacheMode.COLD:
            raise ValueError(
                "method=accuracy requires run.cache_mode='cold' (got "
                f"{run.cache_mode.value!r}): accuracy conclusions require a "
                "deterministic, cache-free request path "
                "(docs/spec/22-accuracy-test-plan.md §2)"
            )
        if int(run.repeats) > ACCURACY_REPEATS:
            raise ValueError(
                "method=accuracy fixes run.repeats=1 (got "
                f"{run.repeats}): each sample gets exactly one judged response; "
                "repeatability baselines use separate runs"
            )
        # Validate method parameters even when constructed programmatically
        # (pydantic validators only run at model construction).
        try:
            cfg.accuracy = AccuracyConfig(**cfg.accuracy.model_dump(mode="python"))
        except ValueError as exc:
            raise ValueError(f"invalid accuracy configuration: {exc}") from exc

        cfg.method = RunMethod.ACCURACY
        run.cache_mode = CacheMode.COLD
        run.repeats = ACCURACY_REPEATS
        run.max_retries = ACCURACY_MAX_RETRIES
        run.warmup_requests = ACCURACY_WARMUP_REQUESTS

    def make_orchestrator(self, cfg: ProjectConfig, client: Any, writer: Any, **kwargs: Any):
        from modelbench.application.accuracy import AccuracyOrchestrator

        return AccuracyOrchestrator(cfg, client, writer, **kwargs)

    def make_writer(self, *args: Any, **kwargs: Any):
        """Accuracy writer (WP-3); imported lazily so WP-1/2 stay testable."""
        try:
            from modelbench.adapters.storage.writer import AccuracyRunWriter
        except ImportError as exc:  # pragma: no cover - integration-time contract
            raise RuntimeError(
                "method=accuracy needs AccuracyRunWriter "
                "(modelbench.adapters.storage.writer); the accuracy output layer "
                "is not installed in this checkout"
            ) from exc
        return AccuracyRunWriter(*args, **kwargs)

    def plan(self, cfg: ProjectConfig) -> Dict[str, Any]:
        from modelbench.application.plan import plan_overview_accuracy

        return plan_overview_accuracy(cfg)

    def testset_fingerprint(self, cfg: ProjectConfig) -> Dict[str, Any]:
        testset_dir = Path(str(getattr(cfg.datasets, "testset_dir", "dataset/test")))
        root = testset_dir / self.default_testset
        candidate = root / "testset.jsonl"
        manifest_path = root / "manifest.json"
        manifest = _read_json(manifest_path)
        info: Dict[str, Any] = {
            "path": _display_path(candidate),
            "sha256": None,
            "rows": None,
            "method": self.name.value,
            "manifest_sha256": None,
            "judge_contract_version": judge_contract_version(manifest),
        }
        try:
            if candidate.is_file():
                info["sha256"] = _sha256_file(candidate)
                info["rows"] = _count_rows(candidate)
        except OSError:
            pass
        try:
            if manifest_path.is_file():
                info["manifest_sha256"] = _sha256_file(manifest_path)
        except OSError:
            pass
        return info


#: Registry in Web card order (performance first: the historical default).
METHODS: Dict[RunMethod, MethodProfile] = {
    RunMethod.PERFORMANCE: _PerformanceMethod(),
    RunMethod.ACCURACY: _AccuracyMethod(),
}


def get_method(name: Union[str, RunMethod, None]) -> MethodProfile:
    """Resolve a method name/``RunMethod`` to its :class:`MethodProfile`."""
    try:
        method = RunMethod.parse(name)
    except ValueError as exc:
        raise ValueError(
            f"unknown test method {name!r}; available: "
            f"{[item.value for item in METHODS]}"
        ) from exc
    try:
        return METHODS[method]
    except KeyError as exc:  # pragma: no cover - registry/parity guarantee
        raise ValueError(
            f"test method {method.value!r} has no registered profile; available: "
            f"{[item.value for item in METHODS]}"
        ) from exc


def method_metadata() -> List[Dict[str, Any]]:
    """Web/CLI list of method cards (``/api/overview`` 的 ``methods`` 字段)."""
    return [profile.metadata() for profile in METHODS.values()]


__all__ = [
    "ACCURACY_MAX_RETRIES",
    "ACCURACY_REPEATS",
    "ACCURACY_WARMUP_REQUESTS",
    "METHODS",
    "MethodProfile",
    "get_method",
    "judge_contract_version",
    "method_metadata",
]
