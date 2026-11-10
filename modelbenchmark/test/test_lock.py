"""Tests for the dual-scope :class:`~modelbench.application.locking.RunLock`.

AGENTS invariant 10 requires one oMLX instance to be driven by at most one
benchmark at a time.  An output-root-only lock misses the case where two
processes use *different* ``--output-dir`` values against the same server, so
``RunLock`` now takes a normalised-``base_url`` endpoint lock (in the system
temporary directory) **and** the legacy output-root lock.

Covered here:

* ``base_url`` normalisation and the stable, namespaced endpoint lock path;
* backwards-compatible output-root-only locking (``base_url=None``/``filename``);
* in-process scope collisions and the release of a partially acquired chain;
* real cross-process exclusion through a subprocess holding the lock, including
  the fail-closed ``UNSUPPORTED`` path.

Subprocess children are always stopped in ``finally`` with a timeout, and every
temporary path comes from ``tmp_path``; the endpoint key embeds a per-session
UUID so parallel test sessions cannot collide on the shared system temp dir.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest

from modelbench.adapters.host import filelock
from modelbench.adapters.storage import layout
from modelbench.application.locking import (
    ENDPOINT_LOCK_DIRNAME,
    RunLock,
    RunLockError,
    endpoint_lock_key,
    endpoint_lock_path,
    normalize_base_url,
)

ROOT = Path(__file__).resolve().parents[1]

#: One unique endpoint per test session: the endpoint lock lives in the shared
#: system temp directory, so a fixed fake URL could collide with a concurrent
#: test run on the same machine.
_UNIQUE = uuid.uuid4().hex[:12]
ENDPOINT_A_HOST = f"lock-test-{_UNIQUE}-a.invalid"
ENDPOINT_B_HOST = f"lock-test-{_UNIQUE}-b.invalid"
ENDPOINT_A = f"http://{ENDPOINT_A_HOST}:18080"
ENDPOINT_B = f"http://{ENDPOINT_B_HOST}:18081"

#: Child process body: take the lock, signal readiness, hold until stdin EOF.
_CHILD_SCRIPT = """
import sys
from pathlib import Path
from modelbench.application.locking import RunLock

root = Path(sys.argv[1])
base_url = None if sys.argv[2] == "-" else sys.argv[2]
ready = Path(sys.argv[3])
with RunLock(root, base_url):
    ready.write_text("ready", encoding="utf-8")
    sys.stdin.read()
"""


# --------------------------------------------------------------------------- #
# base_url normalisation: scheme + host + port only
# --------------------------------------------------------------------------- #


def test_normalize_base_url_canonicalizes_scheme_host_port():
    assert normalize_base_url("http://127.0.0.1:8000") == "http://127.0.0.1:8000"
    assert normalize_base_url("HTTP://LocalHost:8000/") == "http://localhost:8000"
    assert (
        normalize_base_url("http://LOCALHOST:8000/v1/chat/completions?x=1#frag")
        == "http://localhost:8000"
    )
    assert normalize_base_url("  http://localhost:8000  ") == "http://localhost:8000"
    assert normalize_base_url("http://[::1]:8000/v1") == "http://[::1]:8000"


def test_normalize_base_url_fills_in_default_ports():
    assert normalize_base_url("http://localhost") == "http://localhost:80"
    assert normalize_base_url("https://localhost") == "https://localhost:443"
    assert normalize_base_url("https://localhost:8443") == "https://localhost:8443"
    # Same host/port on a different scheme is a different endpoint.
    assert normalize_base_url("http://localhost:8000") != normalize_base_url(
        "https://localhost:8000"
    )


def test_normalize_base_url_rejects_unusable_values():
    for raw in ("", "   ", "not a url", "ftp://localhost:21", "localhost:8000", "http://"):
        with pytest.raises(ValueError):
            normalize_base_url(raw)


# --------------------------------------------------------------------------- #
# endpoint lock path: shared system temp dir, sha256[:16] file name
# --------------------------------------------------------------------------- #


def test_endpoint_lock_path_is_stable_namespaced_and_per_endpoint():
    expected_dir = Path(tempfile.gettempdir()) / ENDPOINT_LOCK_DIRNAME
    first = endpoint_lock_path(ENDPOINT_A)
    assert first.parent == expected_dir
    assert first.name == f"{endpoint_lock_key(ENDPOINT_A)}.lock"
    key = endpoint_lock_key(ENDPOINT_A)
    assert len(key) == 16 and all(char in "0123456789abcdef" for char in key)
    # Equivalent spellings map to the same lock file; different endpoints do not.
    assert endpoint_lock_path(f"HTTP://{ENDPOINT_A_HOST.upper()}:18080/v1?probe=1") == first
    assert endpoint_lock_path(ENDPOINT_B) != first
    assert endpoint_lock_path(f"https://{ENDPOINT_A_HOST}:18080") != first


# --------------------------------------------------------------------------- #
# backwards-compatible construction
# --------------------------------------------------------------------------- #


def test_run_lock_without_base_url_locks_only_the_output_root(tmp_path):
    lock = RunLock(tmp_path)
    assert lock.endpoint_path is None
    assert lock.normalized_base_url is None
    assert lock.path == layout.benchmark_lock_path(tmp_path)

    with lock:
        assert lock.path.is_file()
        assert lock.path.parent == layout.state_dir(tmp_path)

    with RunLock(tmp_path, filename="custom.lock") as custom:
        assert custom.path == tmp_path / "custom.lock"
        assert custom.path.is_file()


def test_run_lock_rejects_an_unusable_base_url_before_taking_any_lock(tmp_path):
    with pytest.raises(ValueError):
        RunLock(tmp_path, "ftp://localhost:21")
    assert not (tmp_path / layout.STATE).exists()


# --------------------------------------------------------------------------- #
# in-process scope collisions
# --------------------------------------------------------------------------- #


def test_endpoint_lock_is_shared_across_output_roots_in_process(tmp_path):
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    endpoint_file = endpoint_lock_path(ENDPOINT_A)

    with RunLock(out1, ENDPOINT_A):
        assert endpoint_file.parent.is_dir()
        holder_text = endpoint_file.read_text(encoding="utf-8")
        assert f"pid={os.getpid()}" in holder_text
        assert "normalized base_url" in holder_text

        # Same endpoint, different output root: blocked by the endpoint lock.
        with pytest.raises(RunLockError, match="another benchmark run is active") as info:
            with RunLock(out2, ENDPOINT_A):
                pass  # pragma: no cover - must not be reached
        message = str(info.value)
        assert "oMLX endpoint" in message
        assert str(endpoint_file) in message
        assert f"pid={os.getpid()}" in message  # holder content is reported

        # Different endpoint, different output root: independent.
        with RunLock(out2, ENDPOINT_B):
            pass

        # Same output root, different endpoint: still blocked by the root lock.
        with pytest.raises(RunLockError, match="output root"):
            with RunLock(out1, ENDPOINT_B):
                pass  # pragma: no cover - must not be reached

    # Release is symmetric: the endpoint is acquirable again afterwards.
    with RunLock(out2, ENDPOINT_A):
        pass


def test_failed_output_lock_releases_the_endpoint_lock(tmp_path):
    """A failed second acquisition must not leak the endpoint lock it took."""
    out1 = tmp_path / "out1"
    with RunLock(out1):  # legacy output-root-only holder
        with pytest.raises(RunLockError, match="output root"):
            with RunLock(out1, ENDPOINT_A):
                pass  # pragma: no cover - must not be reached
        # Endpoint A must be free again even though the output lock stayed busy.
        with RunLock(tmp_path / "out2", ENDPOINT_A):
            pass


def test_run_lock_refuses_endpoint_lock_when_os_locking_is_unsupported(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(filelock, "_PLATFORM", "posix")
    monkeypatch.setattr(filelock, "fcntl", None)
    with pytest.raises(RunLockError, match="unsupported") as info:
        with RunLock(tmp_path, ENDPOINT_A):
            pass  # pragma: no cover - acquisition must fail first
    assert str(endpoint_lock_path(ENDPOINT_A)) in str(info.value)


# --------------------------------------------------------------------------- #
# real cross-process exclusion
# --------------------------------------------------------------------------- #


def _child_env() -> dict:
    env = os.environ.copy()
    src = str(ROOT / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src if not existing else f"{src}{os.pathsep}{existing}"
    return env


@contextmanager
def _child_holding_lock(tmp_path: Path, output_root: Path, base_url=None):
    """Start a child process that holds *output_root* (and *base_url*) open."""
    ready = tmp_path / f"ready-{uuid.uuid4().hex}.flag"
    command = [
        sys.executable,
        "-c",
        _CHILD_SCRIPT,
        str(output_root),
        base_url if base_url is not None else "-",
        str(ready),
    ]
    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        env=_child_env(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 60
        while not ready.exists():
            if process.poll() is not None:
                stderr = process.stderr.read() if process.stderr is not None else ""
                pytest.fail(
                    "lock child exited before acquiring the lock "
                    f"(returncode={process.returncode}):\n{stderr}"
                )
            if time.monotonic() > deadline:
                pytest.fail("timed out waiting for the lock child to acquire the lock")
            time.sleep(0.02)
        yield process
    finally:
        try:
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()  # EOF -> the child releases and exits
        except OSError:  # pragma: no cover - already closed
            pass
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover - broken child
            process.kill()
            process.wait(timeout=30)
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()


@pytest.mark.skipif(
    filelock.backend() == "unsupported",
    reason="OS advisory file locks are unavailable on this host/filesystem",
)
def test_subprocess_endpoint_lock_blocks_a_second_process(tmp_path):
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    endpoint_file = endpoint_lock_path(ENDPOINT_A)

    with _child_holding_lock(tmp_path, out1, ENDPOINT_A) as child:
        # Same oMLX endpoint + different output root: the new key requirement.
        with pytest.raises(RunLockError, match="another benchmark run is active") as info:
            with RunLock(out2, ENDPOINT_A):
                pass  # pragma: no cover - must not be reached
        message = str(info.value)
        assert "oMLX endpoint" in message
        assert str(endpoint_file) in message
        assert f"pid={child.pid}" in message  # holder content names the child

        # Same endpoint through equivalent spellings is still the same lock.
        equivalent = f"HTTP://{ENDPOINT_A_HOST.upper()}:18080/v1/chat?probe=1"
        with pytest.raises(RunLockError, match="oMLX endpoint"):
            with RunLock(out2, equivalent):
                pass  # pragma: no cover - must not be reached

        # Different endpoint + different output root: allowed concurrently.
        with RunLock(out2, ENDPOINT_B):
            pass

        # Same output root, different endpoint: the root lock still collides.
        with pytest.raises(RunLockError, match="output root") as root_info:
            with RunLock(out1, ENDPOINT_B):
                pass  # pragma: no cover - must not be reached
        assert f"pid={child.pid}" in str(root_info.value)

    # Child released on exit: both scopes are available again.
    with RunLock(out2, ENDPOINT_A):
        pass


@pytest.mark.skipif(
    filelock.backend() == "unsupported",
    reason="OS advisory file locks are unavailable on this host/filesystem",
)
def test_legacy_child_output_lock_blocks_new_style_lock(tmp_path):
    """A ``base_url=None`` holder still protects its output root from new callers."""
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"

    with _child_holding_lock(tmp_path, out1, None) as child:
        with pytest.raises(RunLockError, match="output root") as info:
            with RunLock(out1, ENDPOINT_A):
                pass  # pragma: no cover - must not be reached
        assert f"pid={child.pid}" in str(info.value)
        # The endpoint lock taken first by the failed attempt was released.
        with RunLock(out2, ENDPOINT_A):
            pass


# --------------------------------------------------------------------------- #
# BatchRunner wiring: the runner locks the configured oMLX endpoint
# --------------------------------------------------------------------------- #


def test_batch_runner_uses_the_configured_endpoint_lock(tmp_path):
    from modelbench.adapters.config.settings import load_config
    from modelbench.application.batch import BatchRunner

    cfg = load_config()
    cfg.selection.models = ["minicpm5-1b"]
    cfg.selection.scenarios = ["knowledge_qa"]
    cfg.server.base_url = ENDPOINT_A

    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    with RunLock(out1, ENDPOINT_A):
        with pytest.raises(RunLockError, match="oMLX endpoint"):
            BatchRunner(cfg, None, out2, batch_id="lock-scope-test").run()
    # Endpoint released by the failed BatchRunner attempt (and by the holder).
    with RunLock(out2, ENDPOINT_A):
        pass
