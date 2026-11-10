"""Configuration loading and validation.

Priority (highest last):
  1. ``src/modelbench/resources/config/default.yaml`` (or ``--config``)
  2. explicit CLI/Web overrides
  3. environment variables (API keys, base URL)

The resolved object is a plain pydantic model so it can be dumped verbatim into
``manifest.json`` for reproducibility (secrets are redacted).

Model loading contract: only ``moe_expert_offload_enabled`` may be set on a
model.  Everything else (TurboQuant, DFlash, MTP, SpecPrefill, ANE, ...) is
rejected at validation time so the benchmark always measures oMLX defaults.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import yaml
from pydantic import BaseModel, Field, field_validator

from modelbench.domain.context_windows import normalize_windows
from modelbench.domain.models import CacheMode

from modelbench.paths import PROJECT_ROOT, RESOURCES

DEFAULT_CONFIG = RESOURCES / "config" / "default.yaml"
DEFAULT_MODELS = RESOURCES / "config" / "models.yaml"
DEFAULT_SCENARIOS = RESOURCES / "config" / "scenarios.yaml"

#: The only per-model load setting the benchmark is allowed to change.
ALLOWED_LOAD_SETTINGS = frozenset({"moe_expert_offload_enabled"})

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


# --------------------------------------------------------------------------- #
# Config models
# --------------------------------------------------------------------------- #


class ServerConfig(BaseModel):
    base_url: str = "http://127.0.0.1:8000"
    api_key: Optional[str] = None
    api_key_env: str = "OMLX_API_KEY"
    api_key_from_omlx_settings: bool = True
    omlx_settings_path: str = "~/.omlx/settings.json"
    admin_api_key: Optional[str] = None
    admin_api_key_env: str = "OMLX_ADMIN_API_KEY"
    request_timeout_s: float = 900.0
    connect_timeout_s: float = 10.0
    pid_file: Optional[str] = None
    verify_ssl: bool = True


class PerfConfig(BaseModel):
    """The performance benchmark plan (the only test mode)."""

    #: Samples taken from each selected dataset; ``None`` = every available sample.
    samples_per_dataset: Optional[int] = None
    #: Per-scenario sample-count override (Web/API selection).
    samples_by_scenario: Dict[str, int] = Field(default_factory=dict)
    #: Explicit sample ids per scenario (Web/API selection; highest priority).
    sample_ids: Dict[str, List[str]] = Field(default_factory=dict)
    #: Context-window filters in tokens (1024 = 1K); empty = every sample.
    context_windows: List[int] = Field(default_factory=list)
    #: Uniform generation length (OSL).  50 is the default required by the
    #: product spec; CLI ``--max-tokens`` / Web input can override it.  Hitting
    #: the cap is a normal measured sample (``finish_reason=length``): TG TPS is
    #: computed over the actually emitted decode window, never filtered out.
    max_tokens: int = 50
    #: Append a fixed long-output instruction to every performance request.
    #: Without it, samples whose prompt ends with "Answer:" (LongBench v2 /
    #: choices) stop after one token, so ``completion_tokens < 16`` and TG TPS
    #: is null.  The guard is a measurement-protocol suffix, not dataset
    #: content: it is recorded verbatim in ``records.jsonl`` and can be turned
    #: off for diagnostics or historical comparisons.
    decode_guard: bool = True

    @field_validator("samples_per_dataset")
    @classmethod
    def _positive_samples(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and int(v) < 1:
            raise ValueError("perf.samples_per_dataset must be >= 1 (or null for all)")
        return None if v is None else int(v)

    @field_validator("samples_by_scenario")
    @classmethod
    def _positive_scenario_samples(cls, v: Dict[str, int]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for key, value in (v or {}).items():
            if value is None:
                continue
            if int(value) < 1:
                raise ValueError(
                    f"perf.samples_by_scenario['{key}'] must be >= 1 (or omitted)"
                )
            out[str(key)] = int(value)
        return out

    @field_validator("context_windows")
    @classmethod
    def _context_windows(cls, v: List[int]) -> List[int]:
        return normalize_windows(v)

    @field_validator("sample_ids")
    @classmethod
    def _sample_ids(cls, v: Dict[str, List[str]]) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for key, values in (v or {}).items():
            ids = [str(item) for item in (values or []) if str(item).strip()]
            if ids:
                out[str(key)] = list(dict.fromkeys(ids))
        return out

    @field_validator("max_tokens")
    @classmethod
    def _positive_max_tokens(cls, v: int) -> int:
        if int(v) < 1:
            raise ValueError("perf.max_tokens must be >= 1")
        return int(v)

    def samples_for(self, scenario: str) -> Optional[int]:
        """Per-scenario sample count, falling back to the global setting."""
        if scenario in self.samples_by_scenario:
            return self.samples_by_scenario[scenario]
        return self.samples_per_dataset

    def sample_ids_for(self, scenario: str) -> List[str]:
        return list(self.sample_ids.get(scenario) or [])


class RunConfig(BaseModel):
    output_dir: str = "output"
    repeats: int = 1
    #: Per-cell warmup requests (legacy/diagnostic; oMLX-style model warmup is
    #: controlled by ``warmup_per_model`` and is the default path).
    warmup_requests: int = 0
    #: oMLX-style short warmup requests after each model load, before any
    #: measured cell; discarded and excluded from aggregates.  Mirrors oMLX's
    #: single 32-token / max_tokens=8 JIT warmup.
    warmup_per_model: int = 1
    max_retries: int = 2
    retry_backoff_s: float = 2.0
    cache_mode: CacheMode = CacheMode.COLD
    cache_settle_ms: int = 800
    isolate_models: bool = True
    restore_state: bool = True
    cpu_sample_interval_ms: int = 50
    seed: Optional[int] = None
    # "auto" = leave the model/chat-template default (thinking models think).
    # "disabled" = send chat_template_kwargs.enable_thinking=false.
    # "enabled"  = force enable_thinking=true.
    thinking_mode: Literal["auto", "disabled", "enabled"] = "disabled"
    fail_fast: bool = False
    # Safety margin so prompt + max_tokens never overflows the context window.
    context_safety_margin_tokens: int = 128
    # Auto-retry a cold measurement when a cache hit is detected.
    max_contamination_retries: int = 1
    # Global safety cap on measured requests (None = unlimited).
    max_requests: Optional[int] = None

    @field_validator("repeats")
    @classmethod
    def _positive_repeats(cls, v: int) -> int:
        if v < 1:
            raise ValueError("repeats must be >= 1")
        return v

    @field_validator("warmup_requests", "warmup_per_model")
    @classmethod
    def _non_negative_warmup(cls, v: int) -> int:
        if v < 0:
            raise ValueError("warmup must be >= 0")
        return v

    @field_validator("max_requests")
    @classmethod
    def _positive_max_requests(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and int(v) < 1:
            raise ValueError("max_requests must be >= 1 (or null)")
        return None if v is None else int(v)


class DatasetPaths(BaseModel):
    #: Named test sets (``<testset_dir>/<name>/testset.yaml``).
    testset_dir: str = "dataset/test"
    #: Raw upstream files and normalised legacy caches (runtime roots).
    raw_dir: str = "dataset/raw"
    processed_dir: str = "dataset/processed"


class ModelLoadConfig(BaseModel):
    """Per-model load-time settings applied before the model is loaded.

    Only ``moe_expert_offload_enabled`` is allowed: the benchmark measures the
    model as downloaded/installed.  ``optional`` downgrades an unsupported
    setting to a warning instead of aborting the model.
    """

    settings: Dict[str, Any] = Field(default_factory=dict)
    optional: bool = True
    unload_before_apply: bool = True

    @field_validator("settings")
    @classmethod
    def _moe_only(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        unknown = sorted(set(value) - ALLOWED_LOAD_SETTINGS)
        if unknown:
            raise ValueError(
                "model load settings are restricted to "
                f"{sorted(ALLOWED_LOAD_SETTINGS)} (oMLX defaults otherwise); "
                f"rejected: {unknown}"
            )
        if "moe_expert_offload_enabled" in value and not isinstance(
            value["moe_expert_offload_enabled"], bool
        ):
            raise ValueError("moe_expert_offload_enabled must be a boolean")
        return dict(value)


class ModelSpec(BaseModel):
    """A model to benchmark. ``id`` is the oMLX model id."""

    id: str
    alias: Optional[str] = None
    max_context: Optional[int] = None      # override; else from /v1/models
    notes: Optional[str] = None
    enabled: bool = True
    load: ModelLoadConfig = Field(default_factory=ModelLoadConfig)

    @property
    def label(self) -> str:
        return self.alias or self.id


class ScenarioSpec(BaseModel):
    """A dataset-backed test target (scenario = dataset + params)."""

    name: str
    dataset: str
    params: Dict[str, Any] = Field(default_factory=dict)


class SelectionSpec(BaseModel):
    """Models/scenarios selected for this run (filled by CLI/Web/testset)."""

    models: List[str] = Field(default_factory=list)
    scenarios: List[str] = Field(default_factory=list)


class ProjectConfig(BaseModel):
    """Fully resolved configuration for one run."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    run: RunConfig = Field(default_factory=RunConfig)
    perf: PerfConfig = Field(default_factory=PerfConfig)
    datasets: DatasetPaths = Field(default_factory=DatasetPaths)
    models: List[ModelSpec] = Field(default_factory=list)
    scenarios: Dict[str, ScenarioSpec] = Field(default_factory=dict)
    selection: SelectionSpec = Field(default_factory=SelectionSpec)

    # ---- helpers ---------------------------------------------------------

    def resolve_api_key(self) -> Optional[str]:
        server = self.server
        if server.api_key:
            return server.api_key
        env_val = os.environ.get(server.api_key_env)
        if env_val:
            return env_val
        if server.api_key_from_omlx_settings:
            path = Path(server.omlx_settings_path).expanduser()
            if path.is_file():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    key = (data.get("auth") or {}).get("api_key")
                    if key:
                        return key
                except (OSError, ValueError):
                    pass
        return None

    def resolve_admin_api_key(self) -> Optional[str]:
        env_val = os.environ.get(self.server.admin_api_key_env)
        if env_val:
            return env_val
        if self.server.admin_api_key:
            return self.server.admin_api_key
        # oMLX uses the same main key for admin login.
        return self.resolve_api_key()

    def selected_models(self) -> List[ModelSpec]:
        if not self.selection.models:
            enabled = [m for m in self.models if m.enabled]
            if not enabled:
                raise ValueError("no enabled models configured")
            return [enabled[0]]
        wanted = {m.lower() for m in self.selection.models}
        picked = [
            m
            for m in self.models
            if m.enabled
            and (m.id.lower() in wanted or (m.alias or "").lower() in wanted)
        ]
        missing = wanted - {m.id.lower() for m in picked} - {
            (m.alias or "").lower() for m in picked
        }
        if missing:
            raise ValueError(f"selection references unknown models: {sorted(missing)}")
        return picked

    def selected_scenarios(self) -> Dict[str, ScenarioSpec]:
        if not self.selection.scenarios:
            return dict(self.scenarios)
        missing = [s for s in self.selection.scenarios if s not in self.scenarios]
        if missing:
            raise ValueError(f"selection references unknown scenarios: {missing}")
        return {s: self.scenarios[s] for s in self.selection.scenarios}

    def redacted_snapshot(self) -> Dict[str, Any]:
        data = self.model_dump(mode="json")
        style = data.setdefault("server", {})
        for key in ("api_key", "admin_api_key"):
            if style.get(key):
                style[key] = "***redacted***"
        return data


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _expand_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` strings."""
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            var, default = match.group(1), match.group(2)
            return os.environ.get(var, default if default is not None else "")

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return _expand_env(data)


# --------------------------------------------------------------------------- #
# Model-catalog sync (models.yaml <-> live GET /v1/models)
# --------------------------------------------------------------------------- #

#: Plain YAML scalars for ids/aliases; anything else is single-quoted.
_YAML_PLAIN_RE = re.compile(r"^[A-Za-z0-9_./+-]+$")
_MODEL_ENTRY_RE = re.compile(
    r"^(?P<indent>[ \t]*)-[ \t]+id:[ \t]+"
    r"(?P<id>\"[^\"]*\"|'[^']*'|[^#\s]+)"
    r"(?P<rest>[ \t]*(?:#.*)?)$"
)
_MODELS_KEY_RE = re.compile(r"^(?P<indent>[ \t]*)models:[ \t]*(?:#.*)?$")
#: Sync stamp may be a whole comment line or the tail of an existing comment.
_LAST_SYNCED_RE = re.compile(r"Last synced \d{4}-\d{2}-\d{2} \(\d+ models\)")


def _yaml_scalar(value: str) -> str:
    return value if _YAML_PLAIN_RE.match(value) else "'" + value.replace("'", "''") + "'"


def _yaml_unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1].replace("''", "'")
    if len(value) >= 2 and value[0] == value[-1] == '"':
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value[1:-1]
    return value


def _scan_model_entries(lines: List[str]) -> Tuple[int, List[Dict[str, Any]]]:
    """Locate the ``models:`` list and its ``- id:`` entries.

    Returns ``(models_line_index, entries)`` where each entry carries ``id``,
    ``indent``, ``start`` and ``end`` (exclusive) line indexes.
    """
    models_index = -1
    key_indent = 0
    for index, line in enumerate(lines):
        match = _MODELS_KEY_RE.match(line)
        if match:
            models_index = index
            key_indent = len(match.group("indent"))
            break
    if models_index < 0:
        raise ValueError("models.yaml has no top-level `models:` list")

    entries: List[Dict[str, Any]] = []
    section_end = len(lines)
    index = models_index + 1
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            index += 1
            continue
        indent = len(line) - len(line.lstrip(" \t"))
        if indent <= key_indent:
            section_end = index
            break
        match = _MODEL_ENTRY_RE.match(line)
        if match:
            entries.append(
                {
                    "id": _yaml_unquote(match.group("id")),
                    "indent": match.group("indent"),
                    "start": index,
                    "end": None,
                }
            )
        index += 1

    for position, entry in enumerate(entries):
        end = entries[position + 1]["start"] if position + 1 < len(entries) else section_end
        # Trailing blank/comment lines belong to the *following* entry (or the
        # file footer), so they must survive a removal of this entry.
        while end - 1 > entry["start"] and (
            not lines[end - 1].strip() or lines[end - 1].lstrip().startswith("#")
        ):
            end -= 1
        entry["end"] = end
    return models_index, entries


def _unique_alias(model_id: str, taken: set[str]) -> str:
    base = re.sub(r"[^a-z0-9._+-]+", "-", model_id.lower()).strip("-") or "model"
    alias = base
    suffix = 2
    while alias in taken:
        alias = f"{base}-{suffix}"
        suffix += 1
    taken.add(alias)
    return alias


def sync_models_file(
    path: Path,
    server_ids: Iterable[str],
    *,
    dry_run: bool = False,
    today: Optional[str] = None,
) -> Dict[str, Any]:
    """Reconcile ``models.yaml`` with the live oMLX catalog (text-level edit).

    Entries for ids the server no longer reports are removed and missing ids are
    appended; everything else (header comments, aliases, notes, load settings)
    is kept verbatim.  The edit is verified by re-parsing before an atomic
    replace, so a malformed rewrite can never corrupt the catalog.
    """
    wanted = sorted({str(model_id) for model_id in server_ids if str(model_id).strip()})
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    if not wanted:
        # Fail closed: an empty/rejected server catalog must never wipe the
        # versioned model list (that would leave a bare ``models:``/null that
        # load_config cannot read).
        raise ValueError(
            "oMLX reported an empty model catalog; refusing to empty models.yaml"
        )
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    models_index, entries = _scan_model_entries(lines)
    current_ids = {entry["id"] for entry in entries}
    wanted_set = set(wanted)
    added = [model_id for model_id in wanted if model_id not in current_ids]
    removed = [entry["id"] for entry in entries if entry["id"] not in wanted_set]

    result: Dict[str, Any] = {
        "path": str(path),
        "server_model_count": len(wanted),
        "added": added,
        "removed": removed,
        "aliases": {},
        "changed": bool(added or removed),
        "applied": False,
    }
    if not result["changed"]:
        return result

    parsed = yaml.safe_load(original) or {}
    taken = {
        str(item.get("alias")).lower()
        for item in parsed.get("models") or []
        if isinstance(item, dict) and item.get("alias")
    }
    indent = entries[0]["indent"] if entries else "  "
    new_lines: List[str] = []
    for model_id in added:
        alias = _unique_alias(model_id, taken)
        result["aliases"][model_id] = alias
        new_lines.append(f"{indent}- id: {_yaml_scalar(model_id)}\n")
        new_lines.append(f"{indent}  alias: {_yaml_scalar(alias)}\n")

    insert_at = models_index + 1
    removed_set = set(removed)
    for entry in reversed(entries):
        if entry["id"] not in removed_set:
            insert_at = entry["end"]
            break

    drop: set[int] = set()
    for entry in entries:
        if entry["id"] in removed_set:
            drop.update(range(entry["start"], entry["end"]))

    output: List[str] = []
    for index, line in enumerate(lines):
        if index == insert_at:
            output.extend(new_lines)
        if index in drop:
            continue
        output.append(line)
    if insert_at >= len(lines):
        output.extend(new_lines)

    # Keep the human-maintained sync stamp truthful when one is present,
    # whether it is its own comment line or the tail of an existing comment.
    if today is None:
        today = date.today().isoformat()
    stamp = f"Last synced {today} ({len(wanted)} models)"
    for index, line in enumerate(output[:80]):
        if _LAST_SYNCED_RE.search(line):
            output[index] = _LAST_SYNCED_RE.sub(stamp, line, count=1)
            break

    new_text = "".join(output)
    if not new_text.endswith("\n"):
        new_text += "\n"

    try:
        parsed_new = yaml.safe_load(new_text) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"models.yaml sync produced invalid YAML: {exc}") from exc
    new_entries = [item for item in parsed_new.get("models") or [] if isinstance(item, dict)]
    new_ids = [str(item.get("id")) for item in new_entries]
    if sorted(new_ids) != wanted or len(new_ids) != len(set(new_ids)):
        raise ValueError(
            f"models.yaml sync verification failed: {sorted(new_ids)} != {wanted}"
        )
    try:
        for item in new_entries:
            ModelSpec(**item)
    except Exception as exc:  # noqa: BLE001 - surface as a config error
        raise ValueError(f"models.yaml sync produced invalid model entries: {exc}") from exc

    if dry_run:
        return result

    parent = path.parent
    handle_fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(parent)
    )
    try:
        with os.fdopen(handle_fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(new_text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            shutil.copymode(path, tmp_name)
        except OSError:
            pass
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    result["applied"] = True
    return result


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(
    config_path: Optional[Path] = None,
    models_path: Optional[Path] = None,
    scenarios_path: Optional[Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> ProjectConfig:
    """Load, merge and validate project configuration."""
    raw = _read_yaml(
        resolve_project_path(str(config_path)) if config_path else DEFAULT_CONFIG
    )

    models_file = Path(raw.pop("models_file", "")) if raw.get("models_file") else None
    models_file = Path(models_path) if models_path else (models_file or DEFAULT_MODELS)
    scenarios_file = (
        Path(scenarios_path) if scenarios_path
        else (Path(raw.pop("scenarios_file")) if raw.get("scenarios_file") else DEFAULT_SCENARIOS)
    )
    raw.pop("scenarios_file", None)
    raw.pop("models_file", None)
    # models.yaml/scenarios.yaml live next to the package by default, but the
    # YAML may override them with relative paths.  Resolve those against the
    # project root too -- otherwise ``./modelbench`` only works when the
    # process CWD happens to be the repository root.
    models_file = resolve_project_path(str(models_file))
    scenarios_file = resolve_project_path(str(scenarios_file))

    models_raw = _read_yaml(models_file)
    scenarios_raw = _read_yaml(scenarios_file)

    merged: Dict[str, Any] = dict(raw)
    if overrides:
        merged = _deep_merge(merged, overrides)

    models = [ModelSpec(**m) for m in models_raw.get("models", [])]
    scenarios = {
        name: ScenarioSpec(name=name, **spec)
        for name, spec in (scenarios_raw.get("scenarios") or {}).items()
    }

    cfg = ProjectConfig(
        server=ServerConfig(**(merged.get("server") or {})),
        run=RunConfig(**(merged.get("run") or {})),
        perf=PerfConfig(**(merged.get("perf") or {})),
        datasets=DatasetPaths(**(merged.get("datasets") or {})),
        models=models,
        scenarios=scenarios,
    )

    # Resolve relative paths against the project root so the CLI works from
    # any working directory.
    cfg.datasets.testset_dir = str(resolve_project_path(cfg.datasets.testset_dir))
    cfg.datasets.raw_dir = str(resolve_project_path(cfg.datasets.raw_dir))
    cfg.datasets.processed_dir = str(resolve_project_path(cfg.datasets.processed_dir))
    cfg.run.output_dir = str(resolve_project_path(cfg.run.output_dir))
    return cfg


def models_config_path(config_path: Optional[Path] = None) -> Path:
    """Resolve the ``models.yaml`` path the same way :func:`load_config` does."""
    raw = _read_yaml(
        resolve_project_path(str(config_path)) if config_path else DEFAULT_CONFIG
    )
    configured = raw.get("models_file")
    path = Path(str(configured)) if configured else DEFAULT_MODELS
    return resolve_project_path(str(path))


def resolve_project_path(value: str) -> Path:
    """Resolve ``value`` against :data:`PROJECT_ROOT` (absolute paths untouched)."""
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    return path if path.is_absolute() else (PROJECT_ROOT / path)
