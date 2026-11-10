"""Cross-process single-instance guards for benchmark runs.

The Web :class:`~modelbench.interfaces.web.jobs.JobManager` only serialises
jobs inside one process.  Two processes (or a CLI run plus a Web job) could
still benchmark the same oMLX instance concurrently, evicting each other's
models and producing untrustworthy numbers (AGENTS invariant 10).

A lock keyed on the output root is not enough: two runs with different
``--output-dir`` values drive the *same* oMLX instance, which is the state that
must be serialised.  :class:`RunLock` therefore takes **two** advisory
exclusive locks, in this fixed order:

1. **oMLX endpoint lock** -- keyed on the normalised
   :attr:`~modelbench.adapters.config.settings.ServerConfig.base_url`
   (scheme + host + port, default port filled in, path/query/fragment dropped)
   and stored under the *system temporary directory* so it is shared across
   processes and output roots::

       <tempfile.gettempdir()>/modelbench-locks/<sha256[:16]>.lock

2. **output root lock** -- the existing ``<output_root>/.state/benchmark.lock``
   so two runs writing the same output tree cannot interleave.

Acquisition is non-blocking and ordered (endpoint first, then output root).
If the second acquisition fails, the first lock is released immediately so a
failed run never leaves a lock behind.  Release happens in reverse order.

The lock files stay on disk (they are coordination artifacts, not results), but
the OS releases the lock when the process exits or closes the handle.  Locking
that the OS/filesystem cannot provide is fail-closed: :class:`RunLockError` is
raised instead of pretending the run is protected.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, List, Optional, Tuple
from urllib.parse import urlsplit

from modelbench.adapters.host import filelock
from modelbench.adapters.storage import layout

#: Directory (under the system temporary directory) holding endpoint locks.
ENDPOINT_LOCK_DIRNAME = "modelbench-locks"

#: Default well-known ports, used when ``base_url`` omits an explicit port.
_DEFAULT_PORTS = {"http": 80, "https": 443}

#: Lock record stored in the lock file so a blocked run can name the holder.
_HOLDER_TEMPLATE = "pid={pid} started={started} lock={lock} target={target}\n"


class RunLockError(RuntimeError):
    """Raised when another benchmark process already holds a run lock."""


def normalize_base_url(base_url: str) -> str:
    """Return the canonical identity of one oMLX endpoint.

    Only the parts that change *which server* is addressed are kept: the
    lower-cased scheme, the lower-cased host and the port (default port filled
    in).  Path, query string and fragment are dropped because they do not
    change the instance being driven; userinfo is dropped as well.

    >>> normalize_base_url("HTTP://LocalHost:8000/v1?x=1#frag")
    'http://localhost:8000'
    >>> normalize_base_url("https://omlx.local")
    'https://omlx.local:443'
    """
    raw = str(base_url).strip()
    if not raw:
        raise ValueError("invalid oMLX base_url: value is empty")
    parts = urlsplit(raw)
    scheme = parts.scheme.lower()
    if scheme not in _DEFAULT_PORTS:
        raise ValueError(
            f"invalid oMLX base_url {raw!r}: unsupported or missing scheme "
            f"{parts.scheme!r} (expected http or https)"
        )
    host = parts.hostname
    if not host:
        raise ValueError(f"invalid oMLX base_url {raw!r}: missing host")
    host = host.lower()
    if ":" in host:  # IPv6 literal; keep it unambiguous in the canonical form.
        host = f"[{host}]"
    try:
        port = parts.port
    except ValueError as exc:
        raise ValueError(f"invalid oMLX base_url {raw!r}: {exc}") from exc
    if port is None:
        port = _DEFAULT_PORTS[scheme]
    return f"{scheme}://{host}:{port}"


def _endpoint_lock_path(normalized: str) -> Path:
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / ENDPOINT_LOCK_DIRNAME / f"{digest}.lock"


def endpoint_lock_key(base_url: str) -> str:
    """``sha256(normalized_base_url)[:16]`` -- the stable endpoint lock key."""
    normalized = normalize_base_url(base_url)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def endpoint_lock_path(base_url: str) -> Path:
    """Lock file shared by every run driving the same oMLX endpoint."""
    return _endpoint_lock_path(normalize_base_url(base_url))


class RunLock:
    """Dual advisory lock: oMLX endpoint first, then the output root.

    ``RunLock(output_root)`` keeps the historical behaviour (output root lock
    only).  Passing ``base_url`` additionally takes the shared endpoint lock, so
    two processes using *different* output roots against the same oMLX instance
    are mutually exclusive.  ``filename`` still overrides the output lock file
    name (resolved under ``output_root``).
    """

    def __init__(
        self,
        output_root: Path | str,
        base_url: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> None:
        root = Path(output_root)
        self.output_root = root
        #: Output-root lock path; kept as ``path`` for backwards compatibility.
        self.path = (
            layout.benchmark_lock_path(root) if filename is None else root / filename
        )
        self.base_url = base_url
        if base_url is None:
            #: No endpoint dimension requested (legacy callers / tests).
            self.normalized_base_url: Optional[str] = None
            self.endpoint_path: Optional[Path] = None
        else:
            self.normalized_base_url = normalize_base_url(base_url)
            self.endpoint_path = _endpoint_lock_path(self.normalized_base_url)
        self._handles: List[Tuple[str, Path, IO]] = []

    # ------------------------------------------------------------------ #
    # acquisition / release
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "RunLock":
        acquired: List[Tuple[str, Path, IO]] = []
        try:
            if self.endpoint_path is not None:
                assert self.normalized_base_url is not None
                acquired.append(
                    (
                        "oMLX endpoint",
                        self.endpoint_path,
                        self._acquire(
                            lock_kind="oMLX endpoint",
                            path=self.endpoint_path,
                            target=f"normalized base_url {self.normalized_base_url}",
                            hint=(
                                "Wait for it to finish or stop it before starting "
                                "a new run; note that a different --output-dir does "
                                "NOT isolate runs against the same oMLX instance."
                            ),
                        ),
                    )
                )
            acquired.append(
                (
                    "output root",
                    self.path,
                    self._acquire(
                        lock_kind="output root",
                        path=self.path,
                        target=f"output root {self.output_root}",
                        hint=(
                            "Wait for it to finish or stop it before starting "
                            "a new run against this output directory."
                        ),
                    ),
                )
            )
        except BaseException:
            # Never leak a partially acquired lock chain (endpoint lock first).
            self._release(reversed(acquired))
            raise
        self._handles = acquired
        return self

    def _acquire(
        self,
        *,
        lock_kind: str,
        path: Path,
        target: str,
        hint: str,
    ) -> IO:
        """Take one non-blocking exclusive lock or raise :class:`RunLockError`."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a+", encoding="utf-8")
        except OSError as exc:
            raise RunLockError(
                f"cannot acquire the {lock_kind} lock "
                f"({lock_kind} lock file: {path}; {target}): "
                f"{type(exc).__name__}: {exc}. Fix: ensure the lock directory "
                "is writable by this user, then retry. Refusing to start a run "
                "(AGENTS invariant 10: single-instance serial execution)."
            ) from exc
        try:
            status = filelock.try_lock(handle)
        except OSError as exc:
            handle.close()
            raise RunLockError(
                f"cannot acquire the {lock_kind} lock "
                f"({lock_kind} lock file: {path}; {target}): "
                f"{type(exc).__name__}: {exc}. Fix: ensure advisory file locks "
                "work on this filesystem/platform, then retry. Refusing to "
                "start a run (AGENTS invariant 10: single-instance serial "
                "execution)."
            ) from exc
        if status == filelock.BUSY:
            handle.seek(0)
            holder = handle.read().strip() or "unknown"
            handle.close()
            raise RunLockError(
                f"another benchmark run is active for this {lock_kind} "
                f"({lock_kind} lock file: {path}; {target}; holder: {holder}). "
                f"{hint} (AGENTS invariant 10: single-instance serial "
                "execution)."
            )
        if status == filelock.UNSUPPORTED:
            handle.close()
            raise RunLockError(
                "cannot guarantee single-instance benchmark execution: OS file "
                f"locking is unsupported on this platform/filesystem "
                f"({lock_kind} lock file: {path}; {target}). Fix: run on a "
                "filesystem that supports advisory locks. Refusing to start a "
                "run (AGENTS invariant 10: single-instance serial execution)."
            )
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(
                _HOLDER_TEMPLATE.format(
                    pid=os.getpid(),
                    started=datetime.now(timezone.utc).isoformat(),
                    lock=lock_kind,
                    target=target,
                )
            )
            handle.flush()
        except OSError as exc:
            # The lock was taken but its holder record could not be written:
            # release it instead of leaking the lock into this process.
            try:
                filelock.unlock(handle)
            except OSError:  # pragma: no cover - close below still releases
                pass
            handle.close()
            raise RunLockError(
                f"cannot record the {lock_kind} lock holder "
                f"({lock_kind} lock file: {path}; {target}): "
                f"{type(exc).__name__}: {exc}. Fix: ensure the lock file is "
                "writable, then retry. Refusing to start a run (AGENTS "
                "invariant 10: single-instance serial execution)."
            ) from exc
        return handle

    @staticmethod
    def _release(handles) -> None:
        """Release handles in the given order; closing always frees the OS lock."""
        for _label, _path, handle in handles:
            try:
                filelock.unlock(handle)
            except OSError:  # closing the handle below still drops the lock
                pass
            finally:
                handle.close()

    def __exit__(self, *exc_info) -> None:
        handles, self._handles = self._handles, []
        # Reverse acquisition order: output root first, endpoint last.
        self._release(reversed(handles))


__all__ = [
    "ENDPOINT_LOCK_DIRNAME",
    "RunLock",
    "RunLockError",
    "endpoint_lock_key",
    "endpoint_lock_path",
    "normalize_base_url",
]
