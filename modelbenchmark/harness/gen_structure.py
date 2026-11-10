"""Generate the canonical source-tree inventory in docs/design/33-project-structure.md.

    python -m harness.gen_structure --write   # update the file
    python -m harness.gen_structure --check   # fail if stale (CI/gate)

Keeping the tree generated prevents documentation drift after refactors.
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "docs" / "design" / "33-project-structure.md"
PACKAGE = ROOT / "src" / "modelbench"

HEADER = """# 项目结构清单（自动生成）

> 由 `python -m harness.gen_structure --write`（或 `make docs-write`）生成，**请勿手改**。
> 检查：`python -m harness.docs_contract`（或 `make docs`）。
"""


def _tree_lines(root: Path) -> list[str]:
    """Render every directory and file, indented by real parent depth."""
    lines = [f"{root.relative_to(ROOT).as_posix()}/"]
    entries = []
    for path in root.rglob("*"):
        if "__pycache__" in path.parts:
            continue
        entries.append(path.relative_to(root))
    # Lexicographic parent-before-child ordering keeps the tree readable.
    entries.sort(key=lambda rel: rel.as_posix())
    for rel in entries:
        indent = "  " * (len(rel.parts) - 1)
        lines.append(f"{indent}{rel.name}{'/' if (root / rel).is_dir() else ''}")
    return lines


def render() -> str:
    body = "\n".join(_tree_lines(PACKAGE))
    return f"{HEADER}\n```text\n{body}\n```\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m harness.gen_structure")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    expected = render()
    if args.write:
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        TARGET.write_text(expected, encoding="utf-8")
        print(f"wrote {TARGET.relative_to(ROOT)}")
        return 0

    if not TARGET.is_file():
        print(f"missing {TARGET.relative_to(ROOT)}; run --write")
        return 1
    actual = TARGET.read_text(encoding="utf-8")
    if actual != expected:
        print(f"stale {TARGET.relative_to(ROOT)}; run --write")
        return 1
    print("structure inventory up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
