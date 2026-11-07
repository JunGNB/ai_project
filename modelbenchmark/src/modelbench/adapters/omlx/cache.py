"""Cache hygiene controller for oMLX benchmarks.

Why this exists
---------------
oMLX keeps a prefix/KV cache (hot tier + SSD tier).  A repeated prompt can be
served from that cache, which changes TTFT by more than an order of magnitude
(measured on this machine: 0.19 s cached vs 4.73 s cold for a 10.8K prompt).
A benchmark that does not control this measures cache luck, not model speed.

Modes
-----
``cold`` (default)
    Clear hot + SSD caches before every *measured* request, then wait for the
    clear to settle.  Requests also carry ``X-oMLX-Cache-Store: off`` so the
    benchmark never seeds the cache.  If the response still reports
    ``cached_tokens > 0`` the sample is flagged contaminated and retried.
``warm``
    Clear once per cell, then measure repeatedly: this is the steady-state /
    cache-reuse performance, reported separately and never mixed with cold.
``none``
    No management (diagnostics only).
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from modelbench.domain.models import CacheMode, CacheState, Usage


class CacheController:
    def __init__(
        self,
        client: Any,
        mode: CacheMode = CacheMode.COLD,
        settle_ms: int = 800,
        max_contamination_retries: int = 1,
    ):
        self.client = client
        self.mode = CacheMode(mode)
        self.settle_ms = max(0, settle_ms)
        self.max_contamination_retries = max(0, max_contamination_retries)
        self._warm_cell_prepared = False
        self._last_clear_report: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #

    def start_cell(self) -> None:
        """Called once before a new (model, scenario, context) cell."""
        self._warm_cell_prepared = False

    def prepare(self) -> CacheState:
        """Clear caches if the current mode requires it before a request."""
        if self.mode is CacheMode.NONE:
            return CacheState(mode=self.mode, cleared_before_request=False)
        if self.mode is CacheMode.WARM and self._warm_cell_prepared:
            return CacheState(mode=self.mode, cleared_before_request=False)
        state = self.clear()
        # Only a successful clear makes the rest of the warm cell valid.
        self._warm_cell_prepared = bool(state.cleared_before_request)
        return state

    def clear(self) -> CacheState:
        started = time.perf_counter()
        try:
            report = self.client.clear_caches()
            self._last_clear_report = report
            latency = report.get("latency_ms")
            if self.settle_ms:
                time.sleep(self.settle_ms / 1000.0)
            return CacheState(
                mode=self.mode,
                cleared_before_request=True,
                clear_latency_ms=latency,
                note=f"hot+ssd cleared (settle {self.settle_ms}ms)",
            )
        except Exception as exc:  # pragma: no cover - depends on live admin API
            return CacheState(
                mode=self.mode,
                cleared_before_request=False,
                clear_latency_ms=round((time.perf_counter() - started) * 1000.0, 1),
                note=f"cache clear failed: {exc}",
            )

    # ------------------------------------------------------------------ #

    def evaluate(self, state: CacheState, usage: Usage) -> CacheState:
        """Attach the observed cache hit count and flag cold-mode pollution."""
        cached = usage.cached_tokens
        state.cached_tokens = cached
        if self.mode is CacheMode.COLD and state.cleared_before_request:
            if cached is None:
                # Fail closed: without the server's cached-token count we cannot
                # prove the request was cold, so do not count it as clean.
                state.polluted = True
                state.note = (state.note or "") + " | cached_tokens missing in cold mode"
            elif cached > 0:
                state.polluted = True
                state.note = (state.note or "") + f" | cache hit {cached} tokens in cold mode"
        return state

    def cleanup(self) -> Optional[Dict[str, Any]]:
        """Best-effort cleanup after a run (or when switching models)."""
        if self.mode is CacheMode.NONE:
            return None
        try:
            report = self.client.clear_caches()
            self._warm_cell_prepared = False
            return report
        except Exception as exc:  # pragma: no cover
            return {"error": str(exc)}

    @property
    def last_clear_report(self) -> Dict[str, Any]:
        return self._last_clear_report

    @property
    def warm_cleared(self) -> bool:
        """True when the current warm cell has already been cleared once."""
        return bool(self._warm_cell_prepared)
