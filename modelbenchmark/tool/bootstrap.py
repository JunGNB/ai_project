#!/usr/bin/env python3
"""Portable project bootstrap: create ``.venv`` and install the package.

This module uses **only the Python standard library** (Python 3.10+) so it can
run before the project environment exists.  It is the shared implementation
behind every launcher:

* POSIX: ``./modelbench setup``
* Windows cmd: ``modelbench.cmd setup``
* Windows PowerShell: ``.\\modelbench.ps1 setup``

Typical use::

    python tool/bootstrap.py                    # create .venv + pip install -e ".[dev]"
    python tool/bootstrap.py --dry-run          # print planned commands, change nothing
    python tool/bootstrap.py --no-dev           # install the package without extras
    python tool/bootstrap.py --extras datasets  # install a specific extra set
    python tool/bootstrap.py --venv .venv-312 --python /usr/bin/python3.12
    python tool/bootstrap.py --index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip automatically honours ``PIP_INDEX_URL`` / ``PIP_TRUSTED_HOST``; an explicit
``--index-url`` is passed through to pip as well (the CLI flag wins over the
environment, exactly like pip itself).
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]

#: Minimum interpreter version; keep in sync with ``pyproject.toml``.
MIN_VERSION: Tuple[int, int] = (3, 10)

#: Extras installed by default (pytest/pytest-timeout).
DEFAULT_EXTRAS: Tuple[str, ...] = ("dev",)

#: pip/venv commands can be slow on a cold cache or mirror.
COMMAND_TIMEOUT_S = 1800

_EXTRAS_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class BootstrapError(RuntimeError):
    """A bootstrap step failed; the message contains an actionable hint."""


@dataclass(frozen=True)
class BootstrapPlan:
    """Everything the bootstrap is going to do, resolved before any effect."""

    python: str
    version: str
    venv: Path
    venv_python: Path
    create_venv: bool
    ensure_pip: bool
    spec: str
    index_url: Optional[str]
    warning: Optional[str] = None

    def commands(self) -> List[List[str]]:
        """Return the commands in execution order (also used by ``--dry-run``)."""
        commands: List[List[str]] = []
        if self.create_venv:
            # ``venv`` bootstraps pip through ensurepip unless --without-pip is given.
            commands.append([self.python, "-m", "venv", str(self.venv)])
        if self.ensure_pip:
            commands.append([str(self.venv_python), "-m", "ensurepip", "--upgrade"])
        index_args = ["--index-url", self.index_url] if self.index_url else []
        commands.append(
            [str(self.venv_python), "-m", "pip", "install", *index_args, "--upgrade", "pip"]
        )
        commands.append(
            [str(self.venv_python), "-m", "pip", "install", *index_args, "-e", self.spec]
        )
        return commands


# --------------------------------------------------------------------------- #
# platform helpers
# --------------------------------------------------------------------------- #


def venv_python(venv: Path, platform: Optional[str] = None) -> Path:
    """Return the interpreter path inside *venv* for the given platform.

    Windows virtual environments use ``Scripts/python.exe``; POSIX ones use
    ``bin/python``.  ``platform`` is ``os.name`` style (``"nt"``/``"posix"``)
    and defaults to the running platform; the parameter exists so the
    derivation is unit-testable on every host.
    """
    name = os.name if platform is None else platform
    if name == "nt":
        return Path(venv) / "Scripts" / "python.exe"
    return Path(venv) / "bin" / "python"


def _launcher_name() -> str:
    return "modelbench.cmd" if os.name == "nt" else "./modelbench"


def _format_command(command: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(command))
    return shlex.join(list(command))


def _next_steps() -> List[str]:
    launcher = _launcher_name()
    return [
        f"{launcher} web",
        f"{launcher} run --models minicpm5-1b",
        f"{launcher} validate --offline",
    ]


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #


def _probe_interpreter(python: str) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(version, None)`` or ``(None, actionable_error)``."""
    code = "import sys; print('%d.%d.%d' % sys.version_info[:3])"
    try:
        proc = subprocess.run(
            [python, "-c", code], capture_output=True, text=True, timeout=60
        )
    except FileNotFoundError:
        return None, (
            f"python interpreter not found: {python!r}. Install Python "
            f"{MIN_VERSION[0]}.{MIN_VERSION[1]}+ or pass --python /path/to/python "
            "(POSIX launcher: set PYTHON=/path/to/python; Windows: set MB_PYTHON)."
        )
    except PermissionError:
        return None, (
            f"python interpreter is not executable: {python!r}. "
            "Fix its permissions or pass --python /path/to/python."
        )
    except subprocess.TimeoutExpired:
        return None, f"python interpreter {python!r} did not answer within 60s."
    except OSError as exc:
        return None, f"cannot run python interpreter {python!r}: {exc}"
    if proc.returncode != 0:
        lines = (proc.stderr or proc.stdout or "").strip().splitlines()
        detail = lines[-1] if lines else f"exit code {proc.returncode}"
        return None, (
            f"python interpreter {python!r} failed to start: {detail}. "
            "Pass --python /path/to/python (Windows: set MB_PYTHON)."
        )
    version = (proc.stdout or "").strip()
    if not version:
        return None, f"python interpreter {python!r} did not report a version."
    return version, None


def _parse_version(version: str) -> Optional[Tuple[int, ...]]:
    try:
        return tuple(int(part) for part in version.strip().split(".")[:3])
    except ValueError:
        return None


def _venv_pip_available(venv_py: Path) -> bool:
    """Best-effort offline probe: does ``<venv python> -m pip`` work?"""
    if not venv_py.exists():
        return False
    try:
        proc = subprocess.run(
            [str(venv_py), "-m", "pip", "--version"],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


# --------------------------------------------------------------------------- #
# plan construction
# --------------------------------------------------------------------------- #


def _resolve_extras(no_dev: bool, extras: Optional[str]) -> Tuple[Optional[List[str]], Optional[str]]:
    """Return ``(extras, error)``; ``None`` means "no extras"."""
    if no_dev and extras:
        return None, "--no-dev and --extras cannot be combined (use one or the other)."
    if extras is None:
        return ([] if no_dev else list(DEFAULT_EXTRAS)), None
    names = [item.strip() for item in extras.split(",") if item.strip()]
    if not names:
        return None, "--extras must name at least one extra (e.g. --extras datasets)."
    invalid = [name for name in names if not _EXTRAS_NAME_RE.match(name)]
    if invalid:
        return None, f"--extras contains an invalid name: {invalid[0]!r}."
    return names, None


def _build_plan(args: argparse.Namespace) -> BootstrapPlan:
    venv = Path(args.venv).expanduser()
    if not venv.is_absolute():
        venv = (ROOT / venv).resolve()
    venv_py = venv_python(venv)
    create_venv = not venv_py.exists()

    version, error = _probe_interpreter(args.python)
    parsed = _parse_version(version) if version is not None else None
    too_old = parsed is not None and tuple(parsed[:2]) < MIN_VERSION
    warning: Optional[str] = None
    if version is not None and parsed is None:
        error = (
            f"cannot parse Python version {version!r} from {args.python!r}; "
            "pass --python /path/to/python."
        )
    if create_venv:
        # The base interpreter is required to create the virtual environment.
        if error is not None:
            raise BootstrapError(error)
        if too_old:
            required = f"{MIN_VERSION[0]}.{MIN_VERSION[1]}"
            raise BootstrapError(
                f"Python {version} is too old: Python {required}+ is required "
                f"(interpreter: {args.python}). Install a newer Python or select it "
                "with --python (POSIX: PYTHON=/path/to/python; Windows: MB_PYTHON)."
            )
    elif error is not None or too_old:
        # An existing venv is usable on its own; a stale system interpreter must
        # not block `setup` from installing into it.
        warning = error or (
            f"base interpreter {args.python!r} is Python {version} (< "
            f"{MIN_VERSION[0]}.{MIN_VERSION[1]}); the existing venv is used as-is."
        )
        version = "unknown"

    extras, extras_error = _resolve_extras(args.no_dev, args.extras)
    if extras_error is not None:
        raise BootstrapError(extras_error)
    assert extras is not None
    spec = "." if not extras else ".[" + ",".join(extras) + "]"

    ensure_pip = False if create_venv else not _venv_pip_available(venv_py)
    # Mirror precedence mirrors pip itself: an explicit --index-url wins, then
    # PIP_INDEX_URL (pip reads it natively), then UV_DEFAULT_INDEX (the POSIX
    # launcher exports it; the Windows launchers rely on this env fallback).
    index_url = args.index_url
    if not index_url and not os.environ.get("PIP_INDEX_URL"):
        index_url = os.environ.get("UV_DEFAULT_INDEX")
    return BootstrapPlan(
        python=args.python,
        version=version if version is not None else "unknown",
        venv=venv,
        venv_python=venv_py,
        create_venv=create_venv,
        ensure_pip=ensure_pip,
        spec=spec,
        index_url=index_url,
        warning=warning,
    )


# --------------------------------------------------------------------------- #
# reporting / execution
# --------------------------------------------------------------------------- #


def _report_plan(plan: BootstrapPlan, dry_run: bool) -> None:
    print(f"bootstrap: repo root: {ROOT}")
    if plan.version == "unknown":
        print(
            f"bootstrap: interpreter: {plan.python} "
            "(not used: virtual environment already exists)"
        )
    else:
        print(f"bootstrap: interpreter: {plan.python} (Python {plan.version})")
    if plan.warning:
        print(f"bootstrap: warning: {plan.warning}")
    print(
        "bootstrap: venv: "
        f"{plan.venv} ({'present' if plan.venv.is_dir() else 'missing'})"
    )
    print(
        "bootstrap: venv python: "
        f"{plan.venv_python} ({'present' if plan.venv_python.exists() else 'missing'})"
    )
    if not plan.create_venv:
        print(f"bootstrap: pip in venv: {'missing (ensurepip will run)' if plan.ensure_pip else 'present'}")
    print(f"bootstrap: install target: -e {plan.spec}")
    if plan.index_url:
        print(f"bootstrap: index url: {plan.index_url}")
    if os.environ.get("PIP_INDEX_URL") and not plan.index_url:
        print("bootstrap: env PIP_INDEX_URL detected (inherited by pip)")
    if os.environ.get("PIP_TRUSTED_HOST") and not plan.index_url:
        print("bootstrap: env PIP_TRUSTED_HOST detected (inherited by pip)")
    if dry_run:
        for command in plan.commands():
            print(f"[dry-run] command: {_format_command(command)}")


def _run_command(command: List[str]) -> None:
    print(f"[run] command: {_format_command(command)}")
    sys.stdout.flush()
    try:
        proc = subprocess.run(command, cwd=str(ROOT), timeout=COMMAND_TIMEOUT_S)
    except FileNotFoundError as exc:
        raise BootstrapError(f"command not found: {command[0]!r}") from exc
    except subprocess.TimeoutExpired as exc:
        raise BootstrapError(
            f"command timed out after {COMMAND_TIMEOUT_S}s: {_format_command(command)}"
        ) from exc
    if proc.returncode != 0:
        raise BootstrapError(
            f"command failed with exit code {proc.returncode}: {_format_command(command)}\n"
            "  hints: check network/proxy access to PyPI and pass --index-url <mirror>\n"
            "         or set PIP_INDEX_URL; on Windows install Python 3.10+ (the 'py'\n"
            "         launcher) or set MB_PYTHON to a full python.exe path."
        )


def _apply(plan: BootstrapPlan) -> None:
    if plan.create_venv:
        # A partially created venv directory is repaired by running venv again.
        _run_command([plan.python, "-m", "venv", str(plan.venv)])
        if not plan.venv_python.exists():
            raise BootstrapError(
                f"virtual environment creation did not produce {plan.venv_python}; "
                f"check that {plan.python!r} is a full CPython install with the "
                "'venv' module (Debian/Ubuntu: apt install python3-venv)."
            )
    if plan.ensure_pip:
        _run_command([str(plan.venv_python), "-m", "ensurepip", "--upgrade"])
        if not _venv_pip_available(plan.venv_python):
            raise BootstrapError(
                f"{plan.venv_python} has no usable pip after ensurepip; "
                "delete the venv and rerun setup."
            )
    index_args = ["--index-url", plan.index_url] if plan.index_url else []
    _run_command(
        [str(plan.venv_python), "-m", "pip", "install", *index_args, "--upgrade", "pip"]
    )
    _run_command(
        [str(plan.venv_python), "-m", "pip", "install", *index_args, "-e", plan.spec]
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python tool/bootstrap.py",
        description=(
            "Create .venv and install the oMLX modelbench package. "
            "Standard library only; safe to run before the environment exists."
        ),
    )
    parser.add_argument(
        "--venv",
        default=".venv",
        metavar="PATH",
        help="virtual environment directory (default: .venv under the repo root)",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        metavar="PATH",
        help="base interpreter used to create the venv (default: this interpreter)",
    )
    parser.add_argument(
        "--index-url",
        default=None,
        metavar="URL",
        help="pip index URL; pip also honours PIP_INDEX_URL/PIP_TRUSTED_HOST",
    )
    parser.add_argument(
        "--no-dev",
        action="store_true",
        help="install the package without the dev extra",
    )
    parser.add_argument(
        "--extras",
        default=None,
        metavar="A,B",
        help="comma-separated extras to install instead of the default dev extra",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print commands and detection results; create/install nothing",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        plan = _build_plan(args)
    except BootstrapError as exc:
        print(f"bootstrap: ERROR: {exc}", file=sys.stderr)
        return 1

    _report_plan(plan, dry_run=args.dry_run)
    if args.dry_run:
        print("bootstrap: dry-run complete (nothing created or installed)")
        return 0

    try:
        _apply(plan)
    except BootstrapError as exc:
        print(f"bootstrap: ERROR: {exc}", file=sys.stderr)
        return 1

    print()
    print(f"bootstrap: venv ready at {plan.venv}")
    print("environment ready. next:")
    for step in _next_steps():
        print(f"  {step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
