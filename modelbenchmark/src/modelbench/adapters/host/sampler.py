"""Process resource sampling for peak CPU / RSS measurement.

The oMLX server is sampled as a **process tree** (the server plus any helper
children), because tokenization and cache maintenance can happen off the main
process.  CPU percentages follow psutil/Unix convention: 100% == one core
fully busy, so on a 12-core machine the theoretical maximum is 1200%.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import psutil

from .process_memory import get_phys_footprint


@dataclass
class ResourceReport:
    peak_cpu_pct: Optional[float] = None
    mean_cpu_pct: Optional[float] = None
    cpu_seconds: Optional[float] = None
    peak_rss_mb: Optional[float] = None
    #: Peak macOS ``phys_footprint`` of the server process tree, in bytes.
    #: Includes Metal/IOAccelerator allocations; None off macOS.
    peak_phys_footprint_bytes: Optional[int] = None
    samples: int = 0
    window_ms: Optional[float] = None
    pid: Optional[int] = None
    pids_seen: List[int] = field(default_factory=list)
    error: Optional[str] = None


def process_tree_memory(pid: Optional[int]) -> Dict[str, Optional[float]]:
    """Single-shot memory snapshot of the oMLX process tree.

    Returns ``rss_mb`` (process tree RSS) and ``footprint_gb``
    (sum of macOS ``phys_footprint`` in GiB, None off macOS).
    Used to attribute the gross request peak to the server baseline and to
    the model-load step instead of showing one opaque number.
    """
    if not pid:
        return {"rss_mb": None, "footprint_gb": None}
    try:
        root = psutil.Process(pid)
        tree = [root] + list(root.children(recursive=True))
    except psutil.Error:
        return {"rss_mb": None, "footprint_gb": None}
    rss = 0
    footprint = 0
    for proc in tree:
        try:
            rss += proc.memory_info().rss
        except psutil.Error:
            pass
        value = get_phys_footprint(proc.pid)
        if value:
            footprint += value
    return {
        "rss_mb": round(rss / 1e6, 1) if rss else None,
        "footprint_gb": round(footprint / (1024 ** 3), 2) if footprint else None,
    }


class ResourceSampler:
    """Sample a process tree's CPU/RSS while a request is in flight."""

    def __init__(self, pid: Optional[int], interval_ms: int = 50):
        self.pid = pid
        self.interval_s = max(interval_ms, 10) / 1000.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._root: Optional[psutil.Process] = None
        self._windows: List[tuple[float, float]] = []
        self._peak_cpu = 0.0
        self._peak_rss = 0
        self._peak_footprint = 0
        self._cpu_seconds = 0.0
        self._samples = 0
        self._pids_seen = set()
        self._error: Optional[str] = None
        self._started_at: Optional[float] = None
        self._stopped_at: Optional[float] = None

    # ------------------------------------------------------------------ #

    def start(self) -> "ResourceSampler":
        self._started_at = time.perf_counter()
        if not self.pid:
            self._error = "server pid unavailable; CPU sampling skipped"
            return self
        try:
            self._root = psutil.Process(self.pid)
        except psutil.Error as exc:
            self._error = f"cannot attach to server pid {self.pid}: {exc}"
            return self
        self._thread = threading.Thread(target=self._run, name="cpu-sampler", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> ResourceReport:
        self._stopped_at = time.perf_counter()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        mean_cpu = None
        if self._windows:
            total_work = sum(work for work, _ in self._windows)
            total_time = sum(span for _, span in self._windows)
            if total_time > 0:
                mean_cpu = round(total_work / total_time * 100.0, 1)
        window_ms = None
        if self._started_at is not None and self._stopped_at is not None:
            window_ms = round((self._stopped_at - self._started_at) * 1000.0, 1)
        return ResourceReport(
            peak_cpu_pct=round(self._peak_cpu, 1) if self._samples else None,
            mean_cpu_pct=mean_cpu,
            cpu_seconds=round(self._cpu_seconds, 3) if self._samples else None,
            peak_rss_mb=round(self._peak_rss / 1e6, 1) if self._peak_rss else None,
            peak_phys_footprint_bytes=self._peak_footprint or None,
            samples=self._samples,
            window_ms=window_ms,
            pid=self.pid,
            pids_seen=sorted(self._pids_seen),
            error=self._error,
        )

    # ------------------------------------------------------------------ #

    def _tree(self) -> List[psutil.Process]:
        procs: List[psutil.Process] = []
        if self._root is None:
            return procs
        procs.append(self._root)
        try:
            procs.extend(self._root.children(recursive=True))
        except psutil.Error:
            pass
        return procs

    def _run(self) -> None:
        # Prime CPU counters; the first delta needs a baseline.
        baseline_time = time.perf_counter()
        baseline: dict[int, float] = {}
        for proc in self._tree():
            try:
                times = proc.cpu_times()
                baseline[proc.pid] = times.user + times.system
                self._pids_seen.add(proc.pid)
            except psutil.Error:
                continue
        rss = 0
        footprint = 0
        for proc in self._tree():
            try:
                rss += proc.memory_info().rss
            except psutil.Error:
                pass
            value = get_phys_footprint(proc.pid)
            if value:
                footprint += value
        self._peak_rss = max(self._peak_rss, rss)
        self._peak_footprint = max(self._peak_footprint, footprint)

        while not self._stop.wait(self.interval_s):
            now = time.perf_counter()
            span = now - baseline_time
            if span <= 0:
                continue
            work = 0.0
            current: dict[int, float] = {}
            tree = self._tree()
            for proc in tree:
                try:
                    times = proc.cpu_times()
                    total = times.user + times.system
                    current[proc.pid] = total
                    self._pids_seen.add(proc.pid)
                    if proc.pid in baseline:
                        work += max(total - baseline[proc.pid], 0.0)
                except psutil.Error:
                    continue
            if not current:
                continue
            self._cpu_seconds += work
            cpu_pct = work / span * 100.0
            self._peak_cpu = max(self._peak_cpu, cpu_pct)
            self._windows.append((work, span))
            self._samples += 1
            try:
                rss = 0
                footprint = 0
                for proc in tree:
                    try:
                        rss += proc.memory_info().rss
                    except psutil.Error:
                        pass
                    value = get_phys_footprint(proc.pid)
                    if value:
                        footprint += value
                self._peak_rss = max(self._peak_rss, rss)
                self._peak_footprint = max(self._peak_footprint, footprint)
            except psutil.Error:
                pass
            baseline = current
            baseline_time = now
