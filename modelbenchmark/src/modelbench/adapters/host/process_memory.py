"""Authoritative per-process memory measurement on macOS (phys_footprint).

Why: the benchmark used to report only ``psutil`` RSS.  On Apple Silicon RSS can
under- or over-report the memory the kernel actually charges to a process, and
it does not include IOAccelerator/Metal allocations consistently.  macOS
jetsam (and oMLX's own dashboard for "model memory") uses the kernel ledger
``ri_phys_footprint`` from ``proc_pid_rusage(RUSAGE_INFO_V4)``; it includes
anonymous memory, dirty file-backed pages and Metal-backed allocations.

This is the closest externally observable value to oMLX's own memory gauge.
oMLX's *benchmark* ``PeakMem`` column is a different metric: it is
``mlx.core.get_peak_memory()`` (Metal allocation peak inside the engine), which
excludes Python/tokenizer/runtime overhead and is typically ~0.4-0.9 GiB lower.

Non-macOS platforms return ``None`` so callers can fall back to RSS.
"""

from __future__ import annotations

import ctypes
import os
import sys
from typing import Optional

# rusage_info_v4 from /usr/include/sys/resource.h (macOS).
class _RusageInfoV4(ctypes.Structure):
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
        ("ri_child_user_time", ctypes.c_uint64),
        ("ri_child_system_time", ctypes.c_uint64),
        ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_child_interrupt_wkups", ctypes.c_uint64),
        ("ri_child_pageins", ctypes.c_uint64),
        ("ri_child_elapsed_abstime", ctypes.c_uint64),
        ("ri_diskio_bytesread", ctypes.c_uint64),
        ("ri_diskio_byteswritten", ctypes.c_uint64),
        ("ri_cpu_time_qos_default", ctypes.c_uint64),
        ("ri_cpu_time_qos_maintenance", ctypes.c_uint64),
        ("ri_cpu_time_qos_background", ctypes.c_uint64),
        ("ri_cpu_time_qos_utility", ctypes.c_uint64),
        ("ri_cpu_time_qos_legacy", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_initiated", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_interactive", ctypes.c_uint64),
        ("ri_billed_system_time", ctypes.c_uint64),
        ("ri_serviced_system_time", ctypes.c_uint64),
        ("ri_logical_writes", ctypes.c_uint64),
        ("ri_lifetime_max_phys_footprint", ctypes.c_uint64),
        ("ri_instructions", ctypes.c_uint64),
        ("ri_cycles", ctypes.c_uint64),
        ("ri_billed_energy", ctypes.c_uint64),
        ("ri_serviced_energy", ctypes.c_uint64),
        ("ri_interval_max_phys_footprint", ctypes.c_uint64),
        ("ri_runnable_time", ctypes.c_uint64),
    ]


_RUSAGE_INFO_V4 = 4
_DARWIN = sys.platform == "darwin"

_proc_pid_rusage = None
if _DARWIN:
    try:
        _libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        _proc_pid_rusage = _libproc.proc_pid_rusage
        _proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        _proc_pid_rusage.restype = ctypes.c_int
    except OSError:  # pragma: no cover - only on unusual macOS installs
        _proc_pid_rusage = None


def get_phys_footprint(pid: Optional[int] = None) -> Optional[int]:
    """Return ``ri_phys_footprint`` of ``pid`` in bytes, or None if unavailable."""
    if _proc_pid_rusage is None:
        return None
    info = _RusageInfoV4()
    target = os.getpid() if pid is None else int(pid)
    rc = _proc_pid_rusage(target, _RUSAGE_INFO_V4, ctypes.byref(info))
    if rc != 0:
        return None
    return int(info.ri_phys_footprint)


def bytes_to_gib(value: Optional[int]) -> Optional[float]:
    """Convert bytes to GiB rounded to 2 decimals (same convention as oMLX)."""
    if value is None:
        return None
    return round(int(value) / (1024 ** 3), 2)


#: Convert bytes to GiB without rounding (used where a float is needed).
GIB = 1024 ** 3
