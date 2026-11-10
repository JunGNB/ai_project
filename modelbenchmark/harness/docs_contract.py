"""Documentation contract: keep docs aligned with code and filesystem.

Checks (used by ``harness.gate`` and ``pytest``):

* documented ``python -m`` modules are importable;
* backticked repo paths point at existing files/directories;
* documented ``make`` targets exist;
* the single root launcher ``./modelbench`` exists and is executable;
* documented ``./modelbench <cmd> --flag`` flags exist in ``--help``;
* ``docs/design/33-project-structure.md`` matches the current package tree;
* removed schema-2.x/old-layout artifact names do not reappear outside an
  explicit whitelist (historical sections of the output contract / archived
  OUTPUT_LAYOUT, the V&V policy document that defines the rule, and
  ``docs/archive/``).

Scanned documents: root ``README.md``/``AGENTS.md``, ``harness/README.md``,
``harness/prompts/*.md``, ``docs/**/*.md`` (except ``docs/archive/``),
``dataset/README.md`` and ``tool/README.md``.

Checks are offline, read-only and ignore ``docs/archive/``.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]


def iter_docs(root: Path = ROOT) -> List[Path]:
    """Every Markdown document that participates in the docs contract."""
    paths: List[Path] = [
        root / "README.md",
        root / "AGENTS.md",
        root / "harness" / "README.md",
    ]
    docs_dir = root / "docs"
    if docs_dir.is_dir():
        paths += sorted(
            p for p in docs_dir.rglob("*.md") if "archive" not in p.parts
        )
    prompts_dir = root / "harness" / "prompts"
    if prompts_dir.is_dir():
        paths += sorted(prompts_dir.glob("*.md"))
    paths += [root / "dataset" / "README.md", root / "tool" / "README.md"]
    # De-duplicate while preserving order (docs/README.md etc. can appear once).
    unique = list(dict.fromkeys(p for p in paths))
    return [p for p in unique if p.is_file()]


DOCS = iter_docs()

REPO_PREFIXES = ("src/", "dataset/", "output/", "tool/", "test/", "docs/", "harness/", "config/")
PLACEHOLDERS = ("{", "}", "<", ">", "*", "$", "...", "|", " ")
#: Generated/untracked artifacts that docs describe for operators but that do
#: not exist in a clean checkout (they are gitignored by design).  Their
#: absence is not a documentation error; gate's ``dataset_integrity`` check
#: validates them when present.
GENERATED_PREFIXES = (
    "output/",
    "dataset/raw/",
    "dataset/.cache/",
    "harness/reports/",
    # Local-only task journal/evidence; gitignored, absent in a clean clone.
    "harness/state/",
)
GENERATED_TEST_PAYLOADS = re.compile(r"^dataset/test/.+\.jsonl$")
MODULE_RE = re.compile(r"python\s+-m\s+([A-Za-z_][\w\.]*)")
MAKE_RE = re.compile(r"\bmake\s+([a-z][\w-]*)")
CLI_RE = re.compile(
    r"(?:\./)?modelbench\s+"
    r"(run|validate|hardware|list|download|analyze|runs|index|export-analysis|web"
    r"|gc|verify|rebuild|pack)\b([^\n`]*)"
)
FLAG_RE = re.compile(r"--[a-z0-9][a-z0-9-]*")
BACKTICK_RE = re.compile(r"`([^`\n]+)`")
def _backtick_tokens(text: str) -> List[str]:
    return [token.strip() for token in BACKTICK_RE.findall(text)]


def _clean_path(token: str) -> Optional[str]:
    if not token or not token.startswith(REPO_PREFIXES):
        return None
    if any(marker in token for marker in PLACEHOLDERS):
        return None
    candidate = token.split("#", 1)[0]
    candidate = candidate.split("::", 1)[0]
    # strip :line / :line-line
    candidate = re.sub(r":\d+(-\d+)?$", "", candidate)
    candidate = candidate.rstrip(".,:;)")
    return candidate or None


def _is_generated_artifact(candidate: str) -> bool:
    return candidate.startswith(GENERATED_PREFIXES) or bool(
        GENERATED_TEST_PAYLOADS.match(candidate)
    )


def check_paths() -> List[str]:
    issues: List[str] = []
    for doc in DOCS:
        for token in _backtick_tokens(doc.read_text(encoding="utf-8")):
            candidate = _clean_path(token)
            if candidate is None:
                continue
            if not (ROOT / candidate).exists():
                if _is_generated_artifact(candidate):
                    continue
                issues.append(f"{doc.relative_to(ROOT)}: path not found: {candidate}")
    return issues


def check_modules() -> List[str]:
    issues: List[str] = []
    seen: Set[str] = set()
    for doc in DOCS:
        for module in MODULE_RE.findall(doc.read_text(encoding="utf-8")):
            if module in seen or module == "pytest":
                continue
            seen.add(module)
            try:
                found = importlib.util.find_spec(module) is not None
            except Exception:  # noqa: BLE001
                found = False
            if not found:
                issues.append(f"{doc.relative_to(ROOT)}: module not importable: {module}")
    return issues


def check_make_targets() -> List[str]:
    makefile = ROOT / "Makefile"
    if not makefile.is_file():
        return ["Makefile missing"]
    text = makefile.read_text(encoding="utf-8")
    issues: List[str] = []
    for doc in DOCS:
        for target in MAKE_RE.findall(doc.read_text(encoding="utf-8")):
            if target in {"build", "clean"} and target not in text:
                continue
            if not re.search(rf"^{re.escape(target)}:", text, re.M):
                issues.append(f"{doc.relative_to(ROOT)}: make target not found: {target}")
    return issues


def _cli_help(subcommand: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-m", "modelbench.interfaces.cli", subcommand, "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return (proc.stdout or "") + (proc.stderr or "")


def check_cli_flags() -> List[str]:
    issues: List[str] = []
    helps: Dict[str, str] = {}
    for doc in DOCS:
        for match in CLI_RE.finditer(doc.read_text(encoding="utf-8")):
            subcommand, rest = match.group(1), match.group(2)
            if subcommand not in helps:
                helps[subcommand] = _cli_help(subcommand)
            help_text = helps[subcommand]
            for flag in FLAG_RE.findall(rest):
                if len(flag) < 4:
                    continue
                if flag not in help_text:
                    issues.append(
                        f"{doc.relative_to(ROOT)}: flag not in '{subcommand} --help': {flag}"
                    )
    return issues


LEGACY_TERMS = re.compile(
    r"(?<![\w])bench/|(?<![\w])src/(?:core|config|app_test|system|cli|web|strategy|result_analysis)(?![\w])|(?<![\w])results/|(?<![\w])configs/"
)

#: Artifact names that no longer exist after the schema-3.0 flat layout
#: (``runs/<run_id>/`` + ``batches/<batch_id>.json``).  They may only appear in
#: explicitly whitelisted historical/policy documents.
#: ``(?<![\w])`` keeps e.g. ``pilot_summary.json`` from matching ``summary.json``.
DEAD_ARTIFACTS = re.compile(
    r"(?<![\w])summary\.json"
    r"|(?<![\w])hardware\.json"
    r"|(?<![\w])omlx\.json"
    r"|(?<![\w])models\.json"
    r"|(?<![\w])summary\.csv"
    r"|(?<![\w])cells\.csv"
    r"|(?<![\w])result_analysis(?![\w])"
    r"|output/<run_id>"
    r"|output/\.benchmark\.lock"
)

#: Explicit whitelist: historical sections of the output contract design record,
#: the current contract's migration notes, and the V&V policy that defines the
#: blacklist (it must be able to name the banned artifacts). ``docs/archive/``
#: is excluded from ``iter_docs`` and therefore also exempt.
DEAD_ARTIFACT_WHITELIST: Tuple[Path, ...] = (
    ROOT / "docs" / "design" / "31-output-contract.md",
    ROOT / "docs" / "archive" / "OUTPUT_LAYOUT.md",
    ROOT / "docs" / "process" / "50-validation-plan.md",
)


def _dead_artifact_exempt(path: Path) -> bool:
    try:
        resolved = path.resolve()
    except OSError:  # pragma: no cover - defensive only
        return False
    if "archive" in path.parts:
        return True
    for allowed in DEAD_ARTIFACT_WHITELIST:
        try:
            if resolved == allowed.resolve():
                return True
        except OSError:  # pragma: no cover - defensive only
            continue
    return False


def check_dead_artifacts(docs: Optional[Iterable[Path]] = None) -> List[str]:
    """Flag removed artifact names in non-whitelisted documents."""
    issues: List[str] = []
    for doc in docs if docs is not None else DOCS:
        path = Path(doc)
        if _dead_artifact_exempt(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:  # pragma: no cover - unreadable file is reported elsewhere
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            match = DEAD_ARTIFACTS.search(line)
            if match:
                try:
                    label = path.resolve().relative_to(ROOT).as_posix()
                except (OSError, ValueError):
                    label = str(path)
                issues.append(
                    f"{label}:{line_no}: removed artifact name "
                    f"'{match.group(0)}' (schema 3.0 flat layout)"
                )
    return issues


def check_legacy_terms() -> List[str]:
    issues: List[str] = []
    for doc in DOCS:
        for line_no, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), start=1):
            if LEGACY_TERMS.search(line):
                issues.append(f"{doc.relative_to(ROOT)}:{line_no}: legacy path term")
    return issues


def check_entrypoints() -> List[str]:
    issues: List[str] = []
    setup = ROOT / "modelbench"
    if not setup.is_file():
        issues.append("modelbench missing at repo root")
    elif not os.access(setup, os.X_OK):
        issues.append("modelbench is not executable")
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    if "./modelbench" not in makefile:
        issues.append("Makefile setup target does not call ./modelbench")
    return issues


def check_structure() -> List[str]:
    target = ROOT / "docs" / "design" / "33-project-structure.md"
    try:
        from harness.gen_structure import render

        expected = render()
    except Exception as exc:  # noqa: BLE001
        return [f"structure generator failed: {exc}"]
    if not target.is_file():
        return ["docs/design/33-project-structure.md missing; run `make docs-write`"]
    if target.read_text(encoding="utf-8") != expected:
        return ["docs/design/33-project-structure.md stale; run `make docs-write`"]
    return []


def check_all() -> Dict[str, object]:
    checks = {
        "paths": check_paths(),
        "modules": check_modules(),
        "make_targets": check_make_targets(),
        "cli_flags": check_cli_flags(),
        "structure": check_structure(),
        "entrypoints": check_entrypoints(),
        "legacy_terms": check_legacy_terms(),
        "dead_artifacts": check_dead_artifacts(),
    }
    issues = [f"[{name}] {issue}" for name, found in checks.items() for issue in found]
    return {"ok": not issues, "issues": issues, "checks": checks}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m harness.docs_contract")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = check_all()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for issue in report["issues"]:
            print(issue)
        print("docs contract:", "PASS" if report["ok"] else f"FAIL ({len(report['issues'])})")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
