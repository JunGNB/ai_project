"""Tests for the stdlib-only bootstrap and the cross-platform file lock.

Bootstrap tests only use ``--dry-run``: they never create a virtual
environment and never install anything over the network.  The POSIX lock is
exercised for real on macOS/Linux; the Windows ``msvcrt`` branch is simulated
with monkeypatch so no real Windows host is required.
"""

from __future__ import annotations

import errno
import importlib.util
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "tool" / "bootstrap.py"
IS_WINDOWS = os.name == "nt"


def _load_bootstrap_module():
    spec = importlib.util.spec_from_file_location("modelbench_tool_bootstrap", BOOTSTRAP)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves field annotations through sys.modules for the
    # defining module, so register it while executing the module body.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_bootstrap(*args, env=None, timeout=180):
    return subprocess.run(
        [sys.executable, str(BOOTSTRAP), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def _open_lock(tmp_path, name="lock"):
    return (tmp_path / name).open("a+")


# --------------------------------------------------------------------------- #
# bootstrap: plan/command derivation
# --------------------------------------------------------------------------- #


def test_dry_run_plans_venv_creation_and_dev_install(tmp_path):
    venv = tmp_path / "fresh-venv"
    proc = _run_bootstrap("--venv", str(venv), "--dry-run")
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    module = _load_bootstrap_module()
    assert "bootstrap: venv python:" in out
    assert str(module.venv_python(venv)) in out
    assert "[dry-run] command:" in out
    assert "-m venv" in out
    assert "--upgrade pip" in out
    assert "-m pip install" in out
    assert ".[dev]" in out
    assert "dry-run complete" in out
    assert not venv.exists(), "--dry-run must not create the venv"


def test_dry_run_no_dev_installs_package_only(tmp_path):
    proc = _run_bootstrap("--venv", str(tmp_path / "v"), "--dry-run", "--no-dev")
    assert proc.returncode == 0, proc.stderr
    assert ".[dev]" not in proc.stdout
    assert "-e ." in proc.stdout


def test_dry_run_extras_selects_requested_extra(tmp_path):
    proc = _run_bootstrap("--venv", str(tmp_path / "v"), "--dry-run", "--extras", "datasets")
    assert proc.returncode == 0, proc.stderr
    assert ".[datasets]" in proc.stdout
    assert ".[dev]" not in proc.stdout


def test_no_dev_and_extras_are_mutually_exclusive(tmp_path):
    proc = _run_bootstrap(
        "--venv", str(tmp_path / "v"), "--dry-run", "--no-dev", "--extras", "datasets"
    )
    assert proc.returncode == 1
    assert "cannot be combined" in proc.stderr


def test_dry_run_passes_index_url_to_each_pip_install(tmp_path):
    proc = _run_bootstrap(
        "--venv",
        str(tmp_path / "v"),
        "--dry-run",
        "--index-url",
        "https://mirror.example/simple",
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.count("--index-url") == 2
    assert "https://mirror.example/simple" in proc.stdout


def test_dry_run_inherits_pip_index_url_from_environment(tmp_path):
    env = dict(os.environ)
    env["PIP_INDEX_URL"] = "https://mirror.example/simple"
    env["PIP_TRUSTED_HOST"] = "mirror.example"
    proc = _run_bootstrap("--venv", str(tmp_path / "v"), "--dry-run", env=env)
    assert proc.returncode == 0, proc.stderr
    assert "env PIP_INDEX_URL detected" in proc.stdout
    assert "env PIP_TRUSTED_HOST detected" in proc.stdout
    command_lines = [line for line in proc.stdout.splitlines() if "command:" in line]
    assert command_lines
    assert all("--index-url" not in line for line in command_lines)


def test_relative_venv_resolves_against_repo_root(tmp_path):
    proc = _run_bootstrap("--venv", "bootstrap-test-relative-venv", "--dry-run")
    assert proc.returncode == 0, proc.stderr
    assert str(ROOT / "bootstrap-test-relative-venv") in proc.stdout


def test_venv_python_path_is_platform_aware():
    module = _load_bootstrap_module()
    venv = Path("project") / ".venv"
    assert module.venv_python(venv, platform="posix") == venv / "bin" / "python"
    assert module.venv_python(venv, platform="nt") == venv / "Scripts" / "python.exe"
    expected = "Scripts/python.exe" if IS_WINDOWS else "bin/python"
    assert module.venv_python(venv) == venv / expected


def test_missing_python_interpreter_fails_with_actionable_error(tmp_path):
    proc = _run_bootstrap(
        "--venv",
        str(tmp_path / "v"),
        "--dry-run",
        "--python",
        str(tmp_path / "definitely-missing-python"),
    )
    assert proc.returncode == 1
    assert "python interpreter not found" in proc.stderr
    assert "--python" in proc.stderr


@pytest.mark.skipif(IS_WINDOWS, reason="uses a POSIX shebang fake interpreter")
def test_old_python_version_is_rejected(tmp_path):
    fake = tmp_path / "python-fake-3.9"
    fake.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "-c" ]; then\n'
        "  echo '3.9.17'\n"
        "  exit 0\n"
        "fi\n"
        "exit 1\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    proc = _run_bootstrap("--venv", str(tmp_path / "v"), "--dry-run", "--python", str(fake))
    assert proc.returncode == 1
    assert "3.9.17" in proc.stderr
    assert "too old" in proc.stderr
    assert "3.10" in proc.stderr


@pytest.mark.skipif(IS_WINDOWS, reason="POSIX venv layout with a shell fake interpreter")
def test_dry_run_detects_existing_venv_and_usable_pip(tmp_path):
    venv = tmp_path / "existing-venv"
    (venv / "bin").mkdir(parents=True)
    venv_python = venv / "bin" / "python"
    venv_python.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "-m" ] && [ "$2" = "pip" ] && [ "$3" = "--version" ]; then\n'
        "  echo 'pip 24.0 from /fake/site-packages/pip'\n"
        "  exit 0\n"
        "fi\n"
        "exit 1\n",
        encoding="utf-8",
    )
    venv_python.chmod(0o755)
    proc = _run_bootstrap("--venv", str(venv), "--dry-run")
    assert proc.returncode == 0, proc.stderr
    assert "venv python:" in proc.stdout
    assert "pip in venv: present" in proc.stdout
    assert "-m venv" not in proc.stdout
    assert "ensurepip" not in proc.stdout


@pytest.mark.skipif(IS_WINDOWS, reason="POSIX venv layout with a shell fake interpreter")
def test_existing_venv_tolerates_old_base_interpreter(tmp_path):
    """A usable venv must not be blocked by a stale system python3 on PATH."""
    venv = tmp_path / "old-base-venv"
    (venv / "bin").mkdir(parents=True)
    venv_python = venv / "bin" / "python"
    venv_python.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "-m" ] && [ "$2" = "pip" ] && [ "$3" = "--version" ]; then\n'
        "  echo 'pip 24.0'\n"
        "  exit 0\n"
        "fi\n"
        "exit 1\n",
        encoding="utf-8",
    )
    venv_python.chmod(0o755)
    old_python = tmp_path / "system-python-3.9"
    old_python.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "-c" ]; then\n'
        "  echo '3.9.17'\n"
        "  exit 0\n"
        "fi\n"
        "exit 1\n",
        encoding="utf-8",
    )
    old_python.chmod(0o755)
    proc = _run_bootstrap(
        "--venv", str(venv), "--dry-run", "--python", str(old_python)
    )
    assert proc.returncode == 0, proc.stderr
    assert "warning:" in proc.stdout
    assert "3.9.17" in proc.stdout
    assert "-m venv" not in proc.stdout
    assert "pip in venv: present" in proc.stdout


def test_bootstrap_uses_only_the_standard_library():
    text = BOOTSTRAP.read_text(encoding="utf-8")
    imported = set(re.findall(r"^\s*(?:import|from)\s+([A-Za-z_][\w.]*)", text, re.M))
    for module in imported:
        if module == "__future__":
            continue
        assert module.split(".")[0] in sys.stdlib_module_names, f"non-stdlib import: {module}"


# --------------------------------------------------------------------------- #
# file lock: POSIX real lock, Windows branch simulated
# --------------------------------------------------------------------------- #


class _FakeMsvcrt:
    """Minimal stand-in for the ``msvcrt`` module on non-Windows hosts."""

    LK_NBLCK = 1
    LK_UNLCK = 2

    def __init__(self) -> None:
        self.calls = []
        self.busy = False
        self.busy_calls = 0

    def locking(self, fd, mode, nbytes):  # noqa: ANN001 - mirrors msvcrt API
        self.calls.append((mode, nbytes))
        if mode != self.LK_NBLCK:
            return
        if self.busy or self.busy_calls > 0:
            if self.busy_calls > 0:
                self.busy_calls -= 1
            raise OSError(errno.EACCES, "locked by another process")


@pytest.mark.skipif(IS_WINDOWS, reason="POSIX flock path")
def test_posix_flock_is_exclusive_and_releases(tmp_path):
    from modelbench.adapters.host import filelock

    assert filelock.backend() == "posix"
    first = _open_lock(tmp_path)
    second = _open_lock(tmp_path)
    try:
        assert filelock.try_lock(first) == filelock.LOCKED
        assert filelock.try_lock(second) == filelock.BUSY
        assert filelock.unlock(first) == filelock.UNLOCKED
        assert filelock.try_lock(second) == filelock.LOCKED
        assert filelock.unlock(second) == filelock.UNLOCKED
    finally:
        first.close()
        second.close()


@pytest.mark.skipif(IS_WINDOWS, reason="POSIX flock path")
def test_posix_blocking_lock_waits_for_release(tmp_path):
    from modelbench.adapters.host import filelock

    first = _open_lock(tmp_path)
    second = _open_lock(tmp_path)
    try:
        assert filelock.lock(first) == filelock.LOCKED
        state = {}

        def worker():
            state["status"] = filelock.lock(second)
            if state["status"] == filelock.LOCKED:
                filelock.unlock(second)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        time.sleep(0.2)
        assert thread.is_alive(), "blocking lock must wait while another handle holds it"
        assert filelock.unlock(first) == filelock.UNLOCKED
        thread.join(5)
        assert not thread.is_alive()
        assert state["status"] == filelock.LOCKED
    finally:
        first.close()
        second.close()


def test_windows_backend_uses_msvcrt_locking(monkeypatch, tmp_path):
    from modelbench.adapters.host import filelock

    fake = _FakeMsvcrt()
    monkeypatch.setattr(filelock, "_PLATFORM", "nt")
    monkeypatch.setattr(filelock, "msvcrt", fake)
    handle = _open_lock(tmp_path)
    try:
        assert filelock.backend() == "windows"
        handle.seek(7)
        assert filelock.try_lock(handle) == filelock.LOCKED
        # msvcrt.locking locks from the current position, so it must seek(0).
        assert handle.tell() == 0
        assert fake.calls[-1] == (fake.LK_NBLCK, 1)

        fake.busy = True
        assert filelock.try_lock(handle) == filelock.BUSY
        fake.busy = False

        handle.seek(5)
        assert filelock.unlock(handle) == filelock.UNLOCKED
        assert handle.tell() == 0
        assert fake.calls[-1] == (fake.LK_UNLCK, 1)
    finally:
        handle.close()


def test_windows_blocking_lock_retries_until_available(monkeypatch, tmp_path):
    from modelbench.adapters.host import filelock

    fake = _FakeMsvcrt()
    fake.busy_calls = 1  # the first attempt reports BUSY, the next succeeds
    monkeypatch.setattr(filelock, "_PLATFORM", "nt")
    monkeypatch.setattr(filelock, "msvcrt", fake)
    handle = _open_lock(tmp_path)
    try:
        assert filelock.lock(handle) == filelock.LOCKED
        assert len(fake.calls) == 2
        assert filelock.unlock(handle) == filelock.UNLOCKED
    finally:
        handle.close()


def test_windows_backend_without_msvcrt_is_unsupported(monkeypatch, tmp_path):
    from modelbench.adapters.host import filelock

    monkeypatch.setattr(filelock, "_PLATFORM", "nt")
    monkeypatch.setattr(filelock, "msvcrt", None)
    handle = _open_lock(tmp_path)
    try:
        assert filelock.backend() == "unsupported"
        assert filelock.try_lock(handle) == filelock.UNSUPPORTED
        assert filelock.unlock(handle) == filelock.UNSUPPORTED
    finally:
        handle.close()


def test_posix_backend_without_fcntl_is_unsupported(monkeypatch, tmp_path):
    from modelbench.adapters.host import filelock

    monkeypatch.setattr(filelock, "_PLATFORM", "posix")
    monkeypatch.setattr(filelock, "fcntl", None)
    handle = _open_lock(tmp_path)
    try:
        assert filelock.backend() == "unsupported"
        assert filelock.try_lock(handle) == filelock.UNSUPPORTED
        assert filelock.lock(handle) == filelock.UNSUPPORTED
        assert filelock.unlock(handle) == filelock.UNSUPPORTED
    finally:
        handle.close()


def test_run_lock_refuses_instead_of_silently_skipping_when_unsupported(
    monkeypatch, tmp_path
):
    from modelbench.adapters.host import filelock
    from modelbench.application.locking import RunLock, RunLockError

    monkeypatch.setattr(filelock, "_PLATFORM", "posix")
    monkeypatch.setattr(filelock, "fcntl", None)
    with pytest.raises(RunLockError, match="unsupported"):
        with RunLock(tmp_path):
            pass  # pragma: no cover - acquisition must fail first


def test_catalog_lock_refuses_instead_of_silently_skipping_when_unsupported(
    monkeypatch, tmp_path
):
    from modelbench.adapters.host import filelock
    from modelbench.adapters.storage import catalog

    monkeypatch.setattr(filelock, "_PLATFORM", "posix")
    monkeypatch.setattr(filelock, "fcntl", None)
    with pytest.raises(RuntimeError, match="unsupported"):
        catalog.rebuild_index(tmp_path)


# --------------------------------------------------------------------------- #
# mirror selection (Windows launchers rely on the UV_DEFAULT_INDEX fallback)
# --------------------------------------------------------------------------- #


def _bootstrap_dry_run(env_extra):
    env = dict(os.environ)
    for key in ("PIP_INDEX_URL", "UV_DEFAULT_INDEX", "PIP_TRUSTED_HOST"):
        env.pop(key, None)
    env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(BOOTSTRAP), "--dry-run"],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )


def test_uv_default_index_env_is_used_when_pip_index_is_unset():
    proc = _bootstrap_dry_run({"UV_DEFAULT_INDEX": "https://mirror.example/simple"})
    assert proc.returncode == 0, proc.stderr
    assert "index url: https://mirror.example/simple" in proc.stdout
    assert "--index-url https://mirror.example/simple" in proc.stdout


def test_pip_index_url_takes_precedence_over_uv_default():
    proc = _bootstrap_dry_run(
        {
            "PIP_INDEX_URL": "https://pip.example/simple",
            "UV_DEFAULT_INDEX": "https://uv.example/simple",
        }
    )
    assert proc.returncode == 0, proc.stderr
    assert "PIP_INDEX_URL detected" in proc.stdout
    assert "--index-url" not in proc.stdout
