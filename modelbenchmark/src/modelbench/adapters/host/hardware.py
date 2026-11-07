"""Hardware / OS / runtime environment snapshot.

Collected once per run and stored in the run bundle so results are always
interpretable after the fact (hardware, OS, oMLX version, settings).

The top-level ``upload_fields`` mirror the local hardware parameters oMLX's
built-in benchmark attaches when uploading results to the community
leaderboard (chip name/variant, memory, GPU cores, oMLX version, OS version).
"""

from __future__ import annotations

import json
import platform
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import psutil

OMLX_APP_INFO = Path("/Applications/oMLX.app/Contents/Info.plist")


def _sysctl(key: str) -> Optional[str]:
    try:
        out = subprocess.check_output(["sysctl", "-n", key], text=True, timeout=5)
        return out.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def _sw_vers() -> Dict[str, str]:
    values: Dict[str, str] = {}
    try:
        text = subprocess.check_output(["sw_vers"], text=True, timeout=5)
        for line in text.splitlines():
            key, _, value = line.partition(":")
            values[key.strip()] = value.strip()
    except (subprocess.SubprocessError, OSError):
        pass
    return values


def omlx_version() -> Optional[str]:
    """oMLX app version from the macOS bundle Info.plist (when available)."""
    try:
        with OMLX_APP_INFO.open("rb") as fh:
            info = plistlib.load(fh)
        return info.get("CFBundleShortVersionString") or info.get("CFBundleVersion")
    except (OSError, plistlib.InvalidFileException):
        return None


# Backwards-compatible private alias.
_omlx_version = omlx_version


def os_version() -> str:
    """Human-readable OS version, e.g. ``macOS 15.7.7``."""
    versions = _sw_vers()
    product = versions.get("ProductName") or platform.system()
    release = versions.get("ProductVersion") or platform.release()
    return f"{product} {release}".strip()


def _package_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def collect_hardware_profile(
    client: Any = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a JSON-serializable profile of the machine and runtime."""

    total_mem = psutil.virtual_memory().total
    try:
        disk = shutil.disk_usage(str(Path.home()))
        disk_info = {
            "total_gb": round(disk.total / 1e9, 1),
            "free_gb": round(disk.free / 1e9, 1),
        }
    except OSError:
        disk_info = {}

    profile: Dict[str, Any] = {
        "hostname": platform.node(),
        "machine_model": _sysctl("hw.model"),
        "chip": _sysctl("machdep.cpu.brand_string") or platform.processor(),
        "arch": platform.machine(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "sw_vers": _sw_vers(),
        },
        "cpu": {
            "logical_cores": psutil.cpu_count(logical=True),
            "physical_cores": psutil.cpu_count(logical=False),
            "performance_cores": _int_or_none(_sysctl("hw.perflevel0.physicalcpu")),
            "efficiency_cores": _int_or_none(_sysctl("hw.perflevel1.physicalcpu")),
            "perflevel_names": [_sysctl(f"hw.perflevel{i}.name") for i in range(2)],
        },
        "memory": {
            "total_gb": round(total_mem / 1e9, 1),
            "total_bytes": total_mem,
            **_memory_detail(),
        },
        "gpu": {
            "cores": None,
            "note": "filled from oMLX /admin/api/device-info when available",
        },
        "disk": disk_info,
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
            "httpx": _package_version("httpx"),
            "psutil": _package_version("psutil"),
            "pydantic": _package_version("pydantic"),
            "numpy": _package_version("numpy"),
        },
        "omlx_app_version": omlx_version(),
    }

    if client is not None:
        try:
            health = client.health()
            profile["omlx_health"] = health
        except Exception as exc:  # pragma: no cover - depends on live server
            profile["omlx_health_error"] = str(exc)
        try:
            device = client.device_info()
            profile["gpu"]["cores"] = device.get("gpu_cores")
            for key in ("chip_name", "chip_variant", "memory_gb"):
                if device.get(key) is not None:
                    profile["gpu"][key] = device[key]
            profile["omlx_device"] = device
        except Exception as exc:  # pragma: no cover
            profile["omlx_device_error"] = str(exc)
        try:
            profile["omlx_global_settings"] = redact_omlx_settings(client.global_settings())
        except Exception as exc:  # pragma: no cover
            profile["omlx_global_settings_error"] = str(exc)

    # Top-level fields aligned with oMLX's benchmark upload payload.
    device = profile.get("omlx_device") or {}
    chip_string = device.get("chip_name") or profile.get("chip") or ""
    chip_name, chip_variant = _parse_chip(chip_string)
    # oMLX device-info reports the variant separately (M4 / Pro); prefer it.
    profile["chip_name"] = device.get("chip_name") or chip_name
    profile["chip_variant"] = device.get("chip_variant") or chip_variant
    profile["memory_gb"] = device.get("memory_gb") or profile["memory"].get("total_gb")
    profile["gpu_cores"] = device.get("gpu_cores") or profile["gpu"].get("cores")
    profile["os_version"] = os_version()
    profile["omlx_version"] = profile.get("omlx_app_version") or omlx_version()
    profile["upload_fields"] = {
        "chip_name": profile["chip_name"],
        "chip_variant": profile["chip_variant"],
        "memory_gb": profile["memory_gb"],
        "gpu_cores": profile["gpu_cores"],
        "os_version": profile["os_version"],
        "omlx_version": profile["omlx_version"],
    }

    if extra:
        profile.update(extra)
    return profile


def _memory_detail() -> Dict[str, Any]:
    """Live memory/swap snapshot (helps explain oMLX memory-guard failures)."""
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "available_gb": round(memory.available / 1e9, 2),
        "used_gb": round(memory.used / 1e9, 2),
        "percent": memory.percent,
        "swap_used_gb": round(swap.used / 1e9, 2),
        "swap_total_gb": round(swap.total / 1e9, 2),
    }


def _parse_chip(value: str) -> tuple[Optional[str], Optional[str]]:
    text = (value or "").replace("Apple ", "").strip()
    if not text:
        return None, None
    parts = text.split()
    if len(parts) == 1:
        return parts[0], None
    return parts[0], " ".join(parts[1:])


def _int_or_none(value: Optional[str]) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def redact_omlx_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Keep reproducibility-relevant oMLX settings, drop secrets."""
    keep_sections = {
        "server",
        "model",
        "memory",
        "scheduler",
        "cache",
        "sampling",
        "system",
        "external_engines",
    }
    redacted: Dict[str, Any] = {}
    for section, values in (settings or {}).items():
        if section not in keep_sections or not isinstance(values, dict):
            continue
        redacted[section] = {
            k: v
            for k, v in values.items()
            if k not in {"api_key", "secret_key", "sub_keys", "hf_token"}
        }
    return redacted


def dumps_profile(profile: Dict[str, Any]) -> str:
    return json.dumps(profile, ensure_ascii=False, indent=2)
