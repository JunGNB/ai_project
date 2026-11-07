"""oMLX configuration snapshot for the run bundle.

Mirrors what oMLX's built-in benchmark attaches to its community-leaderboard
uploads: hardware identity (collected in ``host/hardware.py``), oMLX version,
the effective per-model settings (projected onto oMLX's upload allowlist) and
the derived acceleration feature flags.

This module is read-only with respect to oMLX: it only calls admin GET
endpoints and never changes settings.
"""

from __future__ import annotations

import platform

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from modelbench.adapters.host.hardware import omlx_version, redact_omlx_settings

#: Performance-relevant ModelSettings fields that oMLX uploads (allowlist, not
#: denylist).  Keep this in sync with the installed oMLX release; unknown
#: fields are ignored rather than leaking local-only/free-text values.
UPLOADED_SETTING_FIELDS = (
    "max_context_window",
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "presence_penalty",
    "force_sampling",
    "enable_thinking",
    "thinking_budget_enabled",
    "thinking_budget_tokens",
    "reasoning_parser",
    "guided_grammar_enabled",
    "model_type_override",
    "index_cache_freq",
    "turboquant_kv_enabled",
    "turboquant_kv_scheme",
    "turboquant_kv_bits",
    "turboquant_skip_last",
    "qwen35_oq_a8_enabled",
    "qwen35_oq_a8_min_tokens",
    "moe_expert_offload_enabled",
    "moe_expert_offload_resident_fraction",
    "specprefill_enabled",
    "specprefill_draft_model",
    "specprefill_keep_pct",
    "specprefill_threshold",
    "dflash_enabled",
    "dflash_draft_model",
    "dflash_draft_quant_enabled",
    "dflash_draft_quant_weight_bits",
    "dflash_draft_quant_group_size",
    "dflash_draft_window_size",
    "dflash_block_size",
    "dflash_verify_mode",
    "mtp_enabled",
    "mtp_adaptive_max_depth",
    "mtp_fixed_depth",
    "vlm_mtp_enabled",
    "vlm_mtp_draft_model",
    "vlm_mtp_draft_block_size",
    "qwen35_ane_prefill_enabled",
    "qwen35_ane_prefill_sequence_length",
    "qwen35_ane_prefill_tail_padding_min_tokens",
    "qwen35_ane_prefill_fraction",
    "qwen35_ane_prefill_shared_fraction",
    "qwen35_ane_prefill_fused_down",
    "qwen35_ane_prefill_max_layers",
    "qwen35_ane_prefill_dual_ane",
    "qwen35_ane_prefill_gdn",
    "qwen35_ane_prefill_gdn_fraction",
    "qwen35_ane_prefill_gdn_max_layers",
    "qwen35_ane_prefill_cpu_enabled",
    "qwen35_ane_prefill_cpu_fraction",
    "qwen35_ane_prefill_cpu_down_fraction",
    "qwen35_ane_prefill_cpu_gdn_fraction",
    "qwen35_ane_prefill_cpu_threads",
    "qwen35_ane_prefill_cpu_shared_resource",
)

_PATH_VALUED_SETTING_FIELDS = frozenset(
    {"specprefill_draft_model", "dflash_draft_model", "vlm_mtp_draft_model"}
)

#: Acceleration toggles surfaced by oMLX; the benchmark expects all of them to
#: be false (default configuration) except MoE offload where explicitly needed.
FEATURE_FLAG_TOGGLES = (
    ("dflash_enabled", "dflash", "DFlash"),
    ("specprefill_enabled", "specprefill", "SpecPrefill"),
    ("turboquant_kv_enabled", "turboquant_kv", "TurboQuant KV"),
    ("mtp_enabled", "lightning_mtp", "Lightning MTP"),
    ("vlm_mtp_enabled", "vlm_mtp", "VLM MTP"),
    ("qwen35_ane_prefill_enabled", "qwen35_ane_prefill", "Qwen ANE Prefill"),
)


def filter_model_settings(settings: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Project effective model settings onto oMLX's upload allowlist."""
    if not settings:
        return {}
    filtered: Dict[str, Any] = {}
    for key in UPLOADED_SETTING_FIELDS:
        if key not in settings:
            continue
        value = settings[key]
        if key in _PATH_VALUED_SETTING_FIELDS and isinstance(value, str):
            value = Path(value.rstrip("/")).name or value
        filtered[key] = value
    return filtered


def derive_feature_flags(settings: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Return active acceleration features ([] means oMLX default config)."""
    flags: List[Dict[str, str]] = []
    for attr, key, label in FEATURE_FLAG_TOGGLES:
        if not (settings or {}).get(attr):
            continue
        if attr == "turboquant_kv_enabled":
            scheme = (settings or {}).get("turboquant_kv_scheme", "turboquant")
            if scheme in ("affine4", "affine8"):
                key, label = f"{scheme}_kv", f"Affine{scheme[-1]} KV"
        flags.append({"key": key, "label": label})
    return flags


def detect_quantization(model_path: Optional[str]) -> str:
    """Detect quantization from config.json or the model directory name."""
    if not model_path:
        return "unknown"
    config_path = Path(model_path) / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            bits = (config.get("quantization_config") or {}).get("bits")
            if bits is not None:
                return f"{bits}bit"
        except (OSError, ValueError):
            pass
    match = re.search(
        r"(2bit|3bit|4bit|6bit|8bit|fp16|bf16|MXFP4|NVFP4)",
        Path(model_path).name,
        re.IGNORECASE,
    )
    return match.group(1).lower() if match else "unknown"


#: Attributes that must be false for a default-config benchmark.  If a model
#: still carries a persisted non-default value from an earlier run, the runner
#: resets it before loading (see Orchestrator._ensure_model_ready).
ACCELERATION_SETTING_ATTRS = tuple(
    attr for attr, _key, _label in FEATURE_FLAG_TOGGLES
) + ("qwen35_oq_a8_enabled",)


def current_model_settings(client: Any, model_id: str) -> Dict[str, Any]:
    """Return the effective settings oMLX currently has for one model."""
    try:
        entries = client.admin_models().get("models", [])
    except Exception:  # noqa: BLE001 - settings are advisory
        return {}
    for entry in entries:
        if str(entry.get("id")) == str(model_id):
            return dict(entry.get("settings") or {})
    return {}


def default_settings_repair(
    current: Dict[str, Any],
    configured: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Settings to apply so a model is measured under the default config.

    * any persisted acceleration toggle that is still ``True`` is reset to
      ``False`` (TurboQuant/DFlash/MTP/SpecPrefill/ANE);
    * the only intentional deviation is ``moe_expert_offload_enabled`` for MoE
      models, provided by ``configured`` (require_offload).

    Returns an empty dict when the model is already in the expected state.
    """
    repair: Dict[str, Any] = {}
    for attr in ACCELERATION_SETTING_ATTRS:
        if current.get(attr):
            repair[attr] = False
    for key, value in (configured or {}).items():
        if current.get(key) != value:
            repair[key] = value
    return repair


def collect_omlx_environment(
    client: Any,
    model_ids: Iterable[str] = (),
) -> Dict[str, Any]:
    """Collect the oMLX configuration snapshot used alongside every run."""
    wanted = {str(m) for m in model_ids}
    env: Dict[str, Any] = {
        "base_url": getattr(client, "base_url", None),
        "omlx_version": omlx_version(),
        "mode": "performance",
        "default_config": True,  # recomputed from feature_flags below
        "global_settings": {},
        "device": {},
        "os": {"version": platform.mac_ver()[0] or platform.platform()},
        "models": {},
        "feature_flags": [],
        "notes": [],
    }

    try:
        env["device"] = client.device_info()
    except Exception as exc:  # pragma: no cover - live server only
        env["notes"].append(f"device-info unavailable: {type(exc).__name__}: {exc}")

    try:
        env["global_settings"] = redact_omlx_settings(client.global_settings())
    except Exception as exc:  # pragma: no cover - live server only
        env["notes"].append(f"global-settings unavailable: {type(exc).__name__}: {exc}")

    admin_models: Dict[str, Dict[str, Any]] = {}
    try:
        for entry in client.admin_models().get("models", []):
            admin_models[str(entry.get("id"))] = entry
    except Exception as exc:  # pragma: no cover - live server only
        env["notes"].append(f"admin models unavailable: {type(exc).__name__}: {exc}")

    status_models: Dict[str, Dict[str, Any]] = {}
    try:
        for entry in client.models_status().get("models", []):
            status_models[str(entry.get("id"))] = entry
    except Exception as exc:  # pragma: no cover
        env["notes"].append(f"models-status unavailable: {type(exc).__name__}: {exc}")

    selected = wanted or set(admin_models) or set(status_models)
    all_flags = set()
    for model_id in sorted(selected):
        admin = admin_models.get(model_id, {})
        status = status_models.get(model_id, {})
        raw_settings = admin.get("settings") or {}
        settings = filter_model_settings(raw_settings)
        flags = derive_feature_flags(raw_settings)
        all_flags.update(flag["key"] for flag in flags)
        model_path = admin.get("model_path") or status.get("model_path")
        env["models"][model_id] = {
            "model_name": Path(model_path).name if model_path else model_id,
            "model_path": model_path,
            "engine_type": admin.get("engine_type") or status.get("engine_type"),
            "model_type": admin.get("model_type") or status.get("model_type"),
            "quantization": detect_quantization(model_path),
            "context_length": admin.get("model_context_length")
            or status.get("model_context_length"),
            "max_context_window": admin.get("model_context_length")
            or status.get("max_context_window")
            or status.get("model_context_length"),
            "max_tokens": admin.get("max_tokens") or status.get("max_tokens"),
            "estimated_size": admin.get("estimated_size") or status.get("estimated_size"),
            "moe_expert_offload_supported": admin.get("moe_expert_offload_supported"),
            "settings": settings,
            "feature_flags": flags,
        }
    env["feature_flags"] = sorted(all_flags)
    env["default_config"] = not env["feature_flags"]
    env["upload_projection"] = upload_projection(env)
    return env


def upload_projection(environment: Dict[str, Any]) -> Dict[str, Any]:
    """Project the snapshot into the shape oMLX uploads to its leaderboard.

    Local-only output (written to ``omlx.json``); nothing is uploaded by this
    project.  Useful for comparing our results with oMLX's own benchmark.
    """
    device = environment.get("device") or {}
    return {
        "chip_name": device.get("chip_name"),
        "chip_variant": device.get("chip_variant"),
        "memory_gb": device.get("memory_gb"),
        "gpu_cores": device.get("gpu_cores"),
        "os_version": (environment.get("os") or {}).get("version"),
        "omlx_version": environment.get("omlx_version"),
        "models": [
            {
                "model_name": info.get("model_name"),
                "quantization": info.get("quantization"),
                "context_length": info.get("context_length"),
                "model_settings": info.get("settings"),
                "feature_flags": info.get("feature_flags"),
            }
            for info in (environment.get("models") or {}).values()
        ],
    }
