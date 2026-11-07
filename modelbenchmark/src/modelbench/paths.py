"""Filesystem locations for the installed package and the project checkout."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
RESOURCES = PACKAGE_ROOT / "resources"


def _find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


#: Repository root in a source checkout; current directory for a wheel install.
PROJECT_ROOT = _find_project_root()
