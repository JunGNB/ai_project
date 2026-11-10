"""Cross-platform advisory file locks.

The benchmark needs one exclusive lock per output root so two runs cannot
evict each other's oMLX models (AGENTS invariant 10).  The POSIX implementation
uses ``fcntl.flock``; Windows uses ``msvcrt.locking`` (which locks a byte range
from the current file position, hence the ``seek(0)``).

Callers open the lock file themselves (``"a+"``) and pass the handle::

    from modelbench.adapters.host import filelock

    handle = path.open("a+")
    try:
        status = filelock.try_lock(handle)
        if status == filelock.BUSY:
            ...
    finally:
        if status == filelock.LOCKED:
            filelock.unlock(handle)
        handle.close()

Return values are explicit strings rather than silent no-ops:

* ``LOCKED`` / ``UNLOCKED`` -- the operation succeeded;
* ``BUSY`` -- another process holds a conflicting lock (non-blocking only);
* ``UNSUPPORTED`` -- this platform/filesystem cannot provide the lock.  Callers
  that rely on single-instance semantics must refuse to continue instead of
  pretending the lock was acquired.
"""

from __future__ import annotations

import errno
import os
import time
from typing import IO, Optional, Set

try:  # POSIX (macOS/Linux)
    import fcntl
except ImportError:  # pragma: no cover - depends on the host platform
    fcntl = None  # type: ignore[assignment]

try:  # Windows
    import msvcrt
except ImportError:  # pragma: no cover - depends on the host platform
    msvcrt = None  # type: ignore[assignment]

#: Successful exclusive lock.
LOCKED = "locked"
#: A different holder owns a conflicting lock (only returned by ``try_lock``).
BUSY = "busy"
#: The host or filesystem does not support advisory locks.
UNSUPPORTED = "unsupported"
#: A held lock was released successfully.
UNLOCKED = "unlocked"

#: Platform selector; ``os.name`` at import time.  Read through a module
#: constant so the Windows backend can be unit-tested on macOS/Linux.
_PLATFORM = os.name

#: Byte range locked by the Windows backend (``msvcrt.locking`` needs a count).
_WINDOWS_LOCK_BYTES = 1

_BUSY_ERRNOS: Set[int] = {
    value
    for value in (
        errno.EACCES,
        getattr(errno, "EAGAIN", None),
        getattr(errno, "EWOULDBLOCK", None),
        getattr(errno, "EDEADLK", None),
        getattr(errno, "EDEADLOCK", None),
    )
    if value is not None
}

_UNSUPPORTED_ERRNOS: Set[int] = {
    value
    for value in (
        getattr(errno, "ENOTSUP", None),
        getattr(errno, "EOPNOTSUPP", None),
        getattr(errno, "ENOSYS", None),
    )
    if value is not None
}


def backend() -> str:
    """Return ``"posix"``, ``"windows"`` or ``"unsupported"`` for this host."""
    if _PLATFORM == "nt":
        return "windows" if msvcrt is not None else "unsupported"
    return "posix" if fcntl is not None else "unsupported"


def _fileno(handle: IO) -> int:
    try:
        return handle.fileno()
    except (AttributeError, OSError) as exc:  # pragma: no cover - misuse
        raise TypeError(
            "filelock expects an open file handle with a valid file descriptor"
        ) from exc


def _classify(exc: OSError) -> Optional[str]:
    """Map an OS error to ``BUSY``/``UNSUPPORTED``; ``None`` means raise it."""
    code = exc.errno
    if code in _BUSY_ERRNOS:
        return BUSY
    if code in _UNSUPPORTED_ERRNOS:
        return UNSUPPORTED
    return None


# --------------------------------------------------------------------------- #
# POSIX (fcntl.flock)
# --------------------------------------------------------------------------- #


def _posix_try(fd: int) -> str:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # type: ignore[union-attr]
    except OSError as exc:
        status = _classify(exc)
        if status is not None:
            return status
        raise
    return LOCKED


def _posix_lock(fd: int) -> str:
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)  # type: ignore[union-attr]
            return LOCKED
        except OSError as exc:
            if exc.errno == errno.EINTR:
                continue
            status = _classify(exc)
            if status is not None:
                return status
            raise


def _posix_unlock(fd: int) -> str:
    fcntl.flock(fd, fcntl.LOCK_UN)  # type: ignore[union-attr]
    return UNLOCKED


# --------------------------------------------------------------------------- #
# Windows (msvcrt.locking)
# --------------------------------------------------------------------------- #


def _windows_try(handle: IO, fd: int) -> str:
    # ``msvcrt.locking`` locks a range starting at the current file position.
    handle.seek(0)
    try:
        msvcrt.locking(fd, msvcrt.LK_NBLCK, _WINDOWS_LOCK_BYTES)  # type: ignore[union-attr]
    except OSError as exc:
        status = _classify(exc)
        if status is not None:
            return status
        raise
    return LOCKED


def _windows_lock(handle: IO, fd: int) -> str:
    # ``LK_LOCK`` only retries ~10s, so emulate a blocking lock with a poll loop
    # to keep POSIX-compatible waiting semantics.
    while True:
        status = _windows_try(handle, fd)
        if status != BUSY:
            return status
        time.sleep(0.1)


def _windows_unlock(handle: IO, fd: int) -> str:
    handle.seek(0)
    msvcrt.locking(fd, msvcrt.LK_UNLCK, _WINDOWS_LOCK_BYTES)  # type: ignore[union-attr]
    return UNLOCKED


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #


def try_lock(handle: IO) -> str:
    """Try to take an exclusive non-blocking lock on *handle*.

    Returns ``LOCKED``, ``BUSY`` (someone else holds it) or ``UNSUPPORTED``.
    """
    fd = _fileno(handle)
    backend_name = backend()
    if backend_name == "unsupported":
        return UNSUPPORTED
    if backend_name == "windows":
        return _windows_try(handle, fd)
    return _posix_try(fd)


def lock(handle: IO) -> str:
    """Take an exclusive blocking lock on *handle*.

    Returns ``LOCKED`` or ``UNSUPPORTED``; unexpected OS errors propagate.
    """
    fd = _fileno(handle)
    backend_name = backend()
    if backend_name == "unsupported":
        return UNSUPPORTED
    if backend_name == "windows":
        return _windows_lock(handle, fd)
    return _posix_lock(fd)


def unlock(handle: IO) -> str:
    """Release a lock previously taken through this module."""
    fd = _fileno(handle)
    backend_name = backend()
    if backend_name == "unsupported":
        return UNSUPPORTED
    if backend_name == "windows":
        return _windows_unlock(handle, fd)
    return _posix_unlock(fd)


__all__ = [
    "LOCKED",
    "BUSY",
    "UNSUPPORTED",
    "UNLOCKED",
    "backend",
    "try_lock",
    "lock",
    "unlock",
]
