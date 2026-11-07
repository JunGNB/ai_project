"""Cross-process single-instance guard for benchmark runs.

The Web :class:`~modelbench.interfaces.web.jobs.JobManager` only serialises
jobs inside one process.  Two processes (or a CLI run plus a Web job) could
still benchmark the same oMLX instance concurrently, evicting each other's
models and producing untrustworthy numbers (AGENTS invariant 10).

This module implements a small advisory file lock (``flock``) under the run
output root.  Acquisition is non-blocking: the second runner fails fast with a
clear ``RunLockError`` instead of silently corrupting an active run.  The lock
file is left on disk (it is a coordination artifact, not a result), but the OS
releases the lock when the process exits or closes the file.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:  # POSIX (macOS/Linux); this project targets macOS.
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore[assignment]


class RunLockError(RuntimeError):
    """Raised when another benchmark process already holds the run lock."""


class RunLock:
    """Advisory exclusive lock on ``<output_root>/.benchmark.lock``."""

    def __init__(self, output_root: Path | str, filename: str = ".benchmark.lock") -> None:
        self.path = Path(output_root) / filename
        self._fh = None

    def __enter__(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        if fcntl is None:  # pragma: no cover - non-POSIX
            self._fh = handle
            return self
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.seek(0)
            holder = handle.read().strip() or "unknown"
            handle.close()
            raise RunLockError(
                "another benchmark run is active for this oMLX instance "
                f"(lock file: {self.path}; holder: {holder}). "
                "Wait for it to finish or stop it before starting a new run "
                "(AGENTS invariant 10: single-instance serial execution)."
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(
            f"pid={os.getpid()} started={datetime.now(timezone.utc).isoformat()}\n"
        )
        handle.flush()
        self._fh = handle
        return self

    def __exit__(self, *exc_info) -> None:
        handle: Optional[object] = self._fh
        self._fh = None
        if handle is None:
            return
        try:
            if fcntl is not None:  # pragma: no cover - covered via context use
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # type: ignore[union-attr]
        finally:
            handle.close()  # type: ignore[union-attr]
