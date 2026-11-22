"""Deterministic quality gate for the benchmark project.

The gate is the single source of truth for "is this change safe to accept?".
It is designed to be run by humans, AI agents, and CI with identical results:

    .venv/bin/python -m harness.gate              # full check
    .venv/bin/python -m harness.gate --ci         # CI mode (datasets may be absent)
    .venv/bin/python -m harness.gate --only tests,docs_links
    .venv/bin/python -m harness.gate --live       # additionally run a real smoke run
    .venv/bin/python -m harness.gate --json       # machine-readable report

Every run writes ``harness/reports/latest.json`` plus a timestamped copy.
Checks are deterministic, offline by default, and safe to run repeatedly.

Adding a check
--------------
1. Write ``def check_xxx(ctx: Context) -> Result`` below.
2. Append ``Check("xxx", "one-line description", check_xxx)`` to ``CHECKS``.
3. A check must be read-only (no writes outside ``harness/reports``), fast (<120s),
   and must not require a live oMLX server unless it is the opt-in ``live_smoke``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "harness" / "reports"
REPORT_LATEST = REPORTS_DIR / "latest.json"

SECRET_PATTERNS = {
    "openai-style key": re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    "aws access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private key block": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "hardcoded oMLX key": re.compile(r"\bomlx-local-[A-Za-z0-9]{4,}\b"),
    "assigned credential": re.compile(
        r"""(?i)\b(api_key|admin_api_key|access_token|secret_key)\s*[:=]\s*["']([^"'{}\s]{12,})["']"""
    ),
}
_CREDENTIAL_ALLOW = ("null", "none", "redacted", "fake", "test", "example", "your", "xxx", "changeme")

REQUIRED_GITIGNORE = (
    ".venv/",
    "dataset/raw/",
    "dataset/.cache/",
    "dataset/processed/",
    "dataset/test/**/testset.jsonl",
    "output/",
    "harness/reports/",
    "harness/state/journal.jsonl",
    "*.egg-info/",
)

FORBIDDEN_TRACKED_PREFIXES = (
    "dataset/raw/",
    "dataset/.cache/",
    "dataset/processed/",
    "output/",
    ".venv/",
    ".uv-cache/",
    "harness/reports/",
    "omlx_modelbenchmark.egg-info/",
)

#: Generated test-set payloads are never committed (rebuilt by tool/dataset).
FORBIDDEN_TRACKED_TEST_SETS = ("dataset/test/", "testset.jsonl")
FORBIDDEN_TRACKED_PARTS = ("__pycache__", ".pytest_cache")
FORBIDDEN_TRACKED_SUFFIXES = (".pyc", ".DS_Store", ".log")


@dataclass
class Context:
    root: Path
    py: str
    ci: bool = False
    live: bool = False


@dataclass
class Result:
    name: str = ""
    ok: bool = True
    detail: str = ""
    data: Dict = field(default_factory=dict)
    duration_ms: int = 0


@dataclass
class Check:
    name: str
    description: str
    func: Callable[[Context], Result]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _run(cmd: List[str], timeout: int = 600, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd or ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _last_line(text: str, limit: int = 400) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return (lines[-1] if lines else "")[:limit]


def _git(*args: str) -> str:
    try:
        proc = _run(["git", *args], timeout=10)
        return proc.stdout.strip() if proc.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


def _git_files() -> List[str]:
    out = _git("ls-files", "-z")
    if out:
        return [p for p in out.split("\0") if p]
    files: List[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if any(rel.startswith(prefix) for prefix in FORBIDDEN_TRACKED_PREFIXES):
            continue
        if ".venv" in rel or ".uv-cache" in rel or ".git/" in rel:
            continue
        files.append(rel)
    return files


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #


def check_imports(ctx: Context) -> Result:
    code = (
        "import modelbench, harness\n"
        "from modelbench.adapters.datasets.registry import available_datasets\n"
        "from modelbench.domain.scenarios.registry import available_scenarios\n"
        "from modelbench.application.benchmark import Orchestrator\n"
        "from modelbench.adapters.storage import catalog, tools\n"
        "ds = available_datasets(); sc = available_scenarios()\n"
        "from modelbench.adapters.config.settings import load_config\n"
        "cfg = load_config()\n"
        "print(f'v{modelbench.__version__} datasets={len(ds)} scenarios={len(sc)} "
        "max_tokens={cfg.perf.max_tokens}')\n"
        "assert len(ds) >= 11, ds\n"
        "assert len(sc) >= 8, sc\n"
        "assert hasattr(Orchestrator, 'run')\n"
        "assert hasattr(catalog, 'rebuild_index')\n"
        "assert hasattr(catalog, 'resolve_run')\n"
        "assert hasattr(catalog, 'VIEWABLE_FILES')\n"
        "assert hasattr(tools, 'verify_output')\n"
        "assert cfg.perf.max_tokens >= 1\n"
    )
    proc = _run([ctx.py, "-c", code], timeout=120)
    detail = (proc.stdout or proc.stderr).strip().replace("\n", " | ")
    ok = proc.returncode == 0
    if not ok:
        detail = _last_line(proc.stderr) or detail
    return Result(ok=ok, detail=detail)


def check_compile(ctx: Context) -> Result:
    proc = _run(
        [ctx.py, "-m", "compileall", "-q", "src", "harness", "tool", "test"],
        timeout=180,
    )
    ok = proc.returncode == 0
    return Result(ok=ok, detail=_last_line(proc.stderr or proc.stdout) or "compiled")


def check_tests(ctx: Context) -> Result:
    proc = _run([ctx.py, "-m", "pytest", "test/"], timeout=900)
    text = (proc.stdout or "") + (proc.stderr or "")
    ok = proc.returncode == 0
    match = re.search(r"(\d+) passed", text)
    passed = int(match.group(1)) if match else 0
    failed = re.findall(r"(\d+) failed", text)
    detail = f"{passed} passed" + (f", {failed[0]} failed" if failed else "")
    if not ok:
        detail = _last_line(text) or detail
    return Result(ok=ok, detail=detail, data={"passed": passed, "failed": int(failed[0]) if failed else 0})


def check_validate(ctx: Context) -> Result:
    cmd = [ctx.py, "-m", "modelbench.interfaces.cli", "validate", "--offline"]
    if ctx.ci:
        cmd += ["--allow-missing", "--no-checksum"]
    proc = _run(cmd, timeout=180)
    text = (proc.stdout or "") + (proc.stderr or "")
    ok = proc.returncode == 0
    if ok:
        detail = _last_line(text) or "validation passed"
    else:
        fails = [line.strip() for line in text.splitlines() if "[FAIL]" in line]
        detail = "; ".join(fails[:4]) or _last_line(text)
    return Result(ok=ok, detail=detail, data={"failures": len(text.split("[FAIL]")) - 1})


def check_dataset_integrity(ctx: Context) -> Result:
    """Validate the new dataset layout (build --check + raw SHA-256 verify).

    Dataset inputs are gitignored on purpose, so a clean checkout / CI has
    nothing to verify.  That is reported as an explicit skip instead of a
    silent pass; locally (and in any checkout with data) the checks must pass.
    The legacy ``tool/dataset/verify.py`` pipeline was removed together with the
    v3 test sets; the new builders own their contract checks.
    """
    testset = ctx.root / "dataset" / "test" / "performance" / "testset.jsonl"
    raw_dir = ctx.root / "dataset" / "raw"
    if not testset.is_file() or not raw_dir.is_dir():
        if ctx.ci:
            return Result(
                ok=True,
                detail="skipped: datasets not present (ci mode)",
                data={"skipped": "datasets absent", "ci": True},
            )
        return Result(
            ok=False,
            detail="datasets absent: rebuild with tool/dataset/performance/build.py",
            data={"skipped": "datasets absent"},
        )

    commands = [
        [ctx.py, str(ctx.root / "tool" / "dataset" / "performance" / "fetch.py"), "--verify"],
        [ctx.py, str(ctx.root / "tool" / "dataset" / "performance" / "build.py"), "--check"],
    ]
    accuracy_testset = ctx.root / "dataset" / "test" / "accuracy" / "testset.jsonl"
    if accuracy_testset.is_file():
        commands.extend(
            [
                [ctx.py, str(ctx.root / "tool" / "dataset" / "accuracy" / "fetch.py"), "--verify"],
                [ctx.py, str(ctx.root / "tool" / "dataset" / "accuracy" / "build.py"), "--check"],
            ]
        )

    failures = []
    for command in commands:
        proc = _run(command, timeout=300)
        text = (proc.stdout or "") + (proc.stderr or "")
        label = Path(command[1]).parent.name if len(command) > 2 else "dataset"
        if proc.returncode != 0:
            failures.append(f"{label}: {_last_line(text) or 'failed'}")
    if failures:
        return Result(ok=False, detail="; ".join(failures[:3]), data={"failures": failures})
    return Result(ok=True, detail="dataset build checks + raw SHA-256 passed", data={"checks": len(commands)})


def check_perf_plan(ctx: Context) -> Result:
    """The performance plan must resolve from config without a matrix/strategy."""
    proc = _run(
        [
            ctx.py,
            "-m",
            "modelbench.interfaces.cli",
            "run",
            "--dry-run",
            "--models",
            "minicpm5-1b",
            "--samples",
            "2",
        ],
        timeout=120,
    )
    text = (proc.stdout or "") + (proc.stderr or "")
    ok = (
        proc.returncode == 0
        and "mode: performance" in text
        and "samples_per_dataset: 2" in text
        and "estimated measured requests" in text
    )
    detail = _last_line(text) or "perf plan resolved"
    return Result(ok=ok, detail=detail, data={"dry_run_ok": proc.returncode == 0})


def check_docs_links(ctx: Context) -> Result:
    files = [ROOT / "README.md", ROOT / "AGENTS.md"]
    # Keep link checking aligned with docs_contract: all current docs,
    # including docs/process/*.md, but not archived historical docs.
    files += sorted(
        p for p in (ROOT / "docs").rglob("*.md") if "archive" not in p.parts
    )
    files += sorted(p for p in (ROOT / "harness").rglob("*.md") if "reports" not in p.parts)
    files = [p for p in files if p.is_file()]
    link_re = re.compile(r"!?\[[^\]]*\]\(([^)\n]+)\)")
    missing: List[str] = []
    absolute: List[str] = []
    checked = 0
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        for raw in link_re.findall(text):
            target = raw.strip()
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1].strip()
            if not target or target.startswith(("http://", "https://", "mailto:", "tel:", "#")):
                continue
            target = target.split("#", 1)[0]
            if not target:
                continue
            if target.startswith("/"):
                absolute.append(f"{path.relative_to(ROOT)} -> {target}")
                continue
            checked += 1
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                missing.append(f"{path.relative_to(ROOT)} -> {target}")
    problems = [f"missing: {m}" for m in missing[:8]] + [f"absolute: {a}" for a in absolute[:8]]
    ok = not problems
    return Result(
        ok=ok,
        detail="; ".join(problems) if problems else f"{checked} relative links checked",
        data={"files": len(files), "links": checked, "missing": missing, "absolute": absolute},
    )


def check_docs_contract(ctx: Context) -> Result:
    from harness.docs_contract import check_all

    report = check_all()
    ok = bool(report["ok"])
    detail = "; ".join(report["issues"][:5]) if not ok else "paths/modules/scripts/flags/structure ok"
    return Result(ok=ok, detail=detail, data={"issues": report["issues"]})


def check_secrets(ctx: Context) -> Result:
    findings: List[str] = []
    scanned = 0
    for rel in _git_files():
        path = ROOT / rel
        if not path.is_file() or path.stat().st_size > 1_000_000:
            continue
        if path.suffix.lower() not in {".py", ".yaml", ".yml", ".toml", ".md", ".json", ".txt", ".sh", ".cfg", ".ini"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        scanned += 1
        for line_no, line in enumerate(text.splitlines(), start=1):
            for label, pattern in SECRET_PATTERNS.items():
                match = pattern.search(line)
                if not match:
                    continue
                if label == "assigned credential":
                    value = match.group(2).lower()
                    if any(token in value for token in _CREDENTIAL_ALLOW):
                        continue
                findings.append(f"{rel}:{line_no} ({label})")
    return Result(
        ok=not findings,
        detail="; ".join(findings[:6]) if findings else f"{scanned} files scanned",
        data={"findings": findings, "scanned": scanned},
    )


def check_hygiene(ctx: Context) -> Result:
    problems: List[str] = []

    gitignore = ROOT / ".gitignore"
    if not gitignore.is_file():
        problems.append(".gitignore missing")
    else:
        lines = {line.strip() for line in gitignore.read_text(encoding="utf-8").splitlines()}
        for entry in REQUIRED_GITIGNORE:
            if entry not in lines:
                problems.append(f".gitignore missing entry: {entry}")

    for rel in _git_files():
        if any(rel.startswith(prefix) for prefix in FORBIDDEN_TRACKED_PREFIXES):
            problems.append(f"tracked generated/evidence path: {rel}")
        if rel.startswith(FORBIDDEN_TRACKED_TEST_SETS[0]) and rel.endswith(
            FORBIDDEN_TRACKED_TEST_SETS[1]
        ):
            problems.append(f"tracked generated test set: {rel}")
        if any(part in rel for part in FORBIDDEN_TRACKED_PARTS):
            problems.append(f"tracked cache path: {rel}")
        if rel.endswith(FORBIDDEN_TRACKED_SUFFIXES):
            problems.append(f"tracked junk file: {rel}")

    pyproject = ROOT / "pyproject.toml"
    if not pyproject.is_file():
        problems.append("pyproject.toml missing")
    else:
        text = pyproject.read_text(encoding="utf-8")
        for needle in ('where = ["src"]', "modelbench*", "resources/config/*.yaml"):
            if needle not in text:
                problems.append(f"pyproject.toml missing: {needle}")

    for required in (
        "AGENTS.md",
        "docs/design/30-architecture.md",
        "docs/guides/41-user-guide.md",
        "docs/spec/21-test-plan.md",
        "docs/design/31-output-contract.md",
        "src/modelbench/resources/config/default.yaml",
        "src/modelbench/adapters/omlx/environment.py",
        "src/modelbench/interfaces/web/static/index.html",
        "dataset/test/performance/testset.yaml",
        "harness/README.md",
        "harness/tasks",
    ):
        if not (ROOT / required).exists():
            problems.append(f"missing required path: {required}")

    return Result(
        ok=not problems,
        detail="; ".join(problems[:6]) if problems else "repo layout/gitignore/packaging ok",
        data={"problems": problems},
    )


def _read_json_file(path: Path) -> Dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _is_run_dir(path: Path) -> bool:
    return (Path(path) / "run.json").is_file()


def resolve_live_run_dir(root: Path, run_id: str) -> Optional[Path]:
    """Locate the run directory produced by a live smoke run (schema 3.0).

    The smoke run is started with ``--run-id <batch_id>`` and the flat layout
    puts each model in ``output/runs/<batch_id>__<method>__<model_slug>/`` (the
    directory name *is* the run id).

    Resolution order:

    1. ``output/index.json`` — ``runs[].run_path`` for an entry whose
       ``run_id`` or ``batch_id`` matches;
    2. glob ``output/runs/*/run.json`` (the index may not have been refreshed
       for a just-finished run, or may be missing entirely), matching either the
       full run id or the batch id prefix before ``__``.

    The retired nested/2.x layouts are deliberately not searched: archived
    bundles live under ``output/archive/`` and are not schema-3.0 evidence.
    """
    output = Path(root) / "output"
    candidates: List[Path] = []

    index = _read_json_file(output / "index.json")
    for entry in index.get("runs") or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("run_id") != run_id and entry.get("batch_id") != run_id:
            continue
        rel = entry.get("run_path")
        if not rel:
            continue
        text = str(rel).replace("\\", "/")
        if text.startswith("/") or ".." in text.split("/"):
            continue
        candidates.append(output / text)

    runs_root = output / "runs"
    if runs_root.is_dir():
        batch_prefix = str(run_id).split("__", 1)[0]
        for run_json_path in sorted(runs_root.glob("*/run.json")):
            run_dir = run_json_path.parent
            name = run_dir.name
            run_json = _read_json_file(run_json_path)
            declared_batch = str(run_json.get("batch_id") or "")
            if name == run_id or declared_batch == run_id:
                candidates.append(run_dir)
            elif not declared_batch and name.split("__", 1)[0] == batch_prefix:
                # Defensive: a half-written run.json without batch_id.
                candidates.append(run_dir)

    for candidate in candidates:
        if _is_run_dir(candidate):
            return candidate
    return None


def _live_smoke_evidence(run_dir: Optional[Path]) -> Dict:
    """Evaluate the schema-3.0 smoke-run invariants from a run directory.

    Reads only ``run.json`` + ``aggregates.json`` + ``records.jsonl`` (the
    retired ``summary.json``/``omlx.json`` are gone).  Returns
    ``{"ok", "detail", "data"}`` so the check stays unit-testable without a
    live oMLX server.
    """
    if run_dir is None:
        return {"ok": False, "detail": "no run directory found", "data": {}}
    run_dir = Path(run_dir)
    run_json_path = run_dir / "run.json"
    if not run_json_path.is_file():
        return {"ok": False, "detail": f"no run.json under {run_dir}", "data": {}}
    run_json = _read_json_file(run_json_path)
    aggregates = _read_json_file(run_dir / "aggregates.json")
    records_path = run_dir / "records.jsonl"
    try:
        records = [
            json.loads(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except OSError as exc:
        return {"ok": False, "detail": f"records.jsonl unreadable: {exc}", "data": {}}
    measured = [r for r in records if r.get("phase") == "measured"]
    bad_cache = [
        r
        for r in measured
        if not (
            r.get("cache", {}).get("cleared_before_request")
            and r.get("cache", {}).get("cached_tokens") == 0
        )
    ]
    errors = [r for r in records if r.get("status") in ("error", "timeout")]
    contaminated = [r for r in records if r.get("status") == "contaminated"]
    missing_context = [
        r for r in measured if r.get("metrics", {}).get("input_context_tokens") is None
    ]
    scored = [r for r in measured if r.get("score") is not None]
    outcome = run_json.get("outcome") or {}
    # One environment snapshot per run now lives in run.json (the old separate
    # omlx.json is gone), and the aggregate shape lives in aggregates.json.
    env_ok = bool(run_json.get("env"))
    shape_ok = (
        isinstance(aggregates.get("overall"), list)
        and outcome.get("n_measured") == len(measured)
    )
    ok = (
        not errors
        and not contaminated
        and not bad_cache
        and not missing_context
        and not scored
        and env_ok
        and shape_ok
        and len(measured) > 0
    )
    detail = (
        f"{len(records)} records, {len(measured)} measured, "
        f"{len(errors)} errors, {len(contaminated)} contaminated, "
        f"{len(bad_cache)} cache issues, {len(missing_context)} missing context, "
        f"{len(scored)} scored, run.json={env_ok}, shape_ok={shape_ok}"
    )
    return {
        "ok": ok,
        "detail": detail,
        "data": {
            "run_dir": str(run_dir),
            "records": len(records),
            "measured": len(measured),
            "shape_ok": shape_ok,
            "env_ok": env_ok,
        },
    }


def check_live_smoke(ctx: Context) -> Result:
    run_id = "gate-live-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    proc = _run(
        [
            ctx.py,
            "-m",
            "modelbench.interfaces.cli",
            "run",
            "--models",
            "minicpm5-1b",
            "--samples",
            "1",
            "--scenarios",
            "knowledge_qa",
            "math_logic",
            "--run-id",
            run_id,
        ],
        timeout=900,
    )
    text = (proc.stdout or "") + (proc.stderr or "")
    run_dir = resolve_live_run_dir(ROOT, run_id)
    if proc.returncode != 0:
        return Result(
            ok=False,
            detail=f"smoke run failed: {_last_line(text)} (batch {run_id})",
        )
    evidence = _live_smoke_evidence(run_dir)
    if not evidence["ok"]:
        return Result(
            ok=False,
            detail=f"{evidence['detail']} (batch {run_id})",
            data=evidence["data"],
        )
    return Result(ok=True, detail=evidence["detail"], data=evidence["data"])


CHECKS: List[Check] = [
    Check("imports", "import all packages; registries have >=11 datasets / >=8 scenarios", check_imports),
    Check("compile", "compileall src/ harness/ tool/ test/", check_compile),
    Check("tests", "pytest test/ (unit + fake-server integration)", check_tests),
    Check("validate", "modelbench.interfaces.cli validate --offline (config + dataset parsing)", check_validate),
    Check(
        "dataset_integrity",
        "tool/dataset/*/build.py --check + fetch.py --verify (skipped when data absent in --ci)",
        check_dataset_integrity,
    ),
    Check("perf_plan", "performance plan resolves (config/testset/models, no strategy/matrix)", check_perf_plan),
    Check("docs_links", "all relative markdown links resolve; no absolute links", check_docs_links),
    Check(
        "docs_contract",
        "documented modules/paths/flags/scripts and generated structure match code",
        check_docs_contract,
    ),
    Check("secrets", "no credential-like literals in tracked text files", check_secrets),
    Check("hygiene", "gitignore/layout/packaging invariants", check_hygiene),
    Check("live_smoke", "real oMLX smoke run with clean-cache evidence", check_live_smoke),
]
DEFAULT_CHECKS = [c for c in CHECKS if c.name != "live_smoke"]


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #


def _report_meta(ctx: Context) -> Dict:
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(ctx.root),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "ci": ctx.ci,
        "live": ctx.live,
        "git_commit": _git("rev-parse", "HEAD") or None,
        "git_dirty": bool(_git("status", "--porcelain")),
    }


def run_checks(ctx: Context, selected: List[Check]) -> Dict:
    started = time.perf_counter()
    results: List[Result] = []
    for check in selected:
        t0 = time.perf_counter()
        try:
            result = check.func(ctx)
        except subprocess.TimeoutExpired:
            result = Result(ok=False, detail="timeout")
        except Exception as exc:  # noqa: BLE001 - a failing check must not crash the gate
            result = Result(ok=False, detail=f"{type(exc).__name__}: {exc}")
        result.name = check.name
        result.duration_ms = int((time.perf_counter() - t0) * 1000)
        results.append(result)

    duration_ms = int((time.perf_counter() - started) * 1000)
    meta = _report_meta(ctx)
    report = {
        **meta,
        "duration_ms": duration_ms,
        "checks": [asdict(r) for r in results],
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if r.ok),
            "failed": sum(1 for r in results if not r.ok),
            "ok": all(r.ok for r in results),
        },
    }
    return report


def write_report(report: Dict) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = REPORTS_DIR / f"gate-{stamp}.json"
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    path.write_text(payload, encoding="utf-8")
    REPORT_LATEST.write_text(payload, encoding="utf-8")
    return path


def _print_table(report: Dict) -> None:
    print(f"gate: {report['root']}")
    print(
        f"python {report['python']} | commit {str(report['git_commit'])[:12]}"
        f"{' (dirty)' if report['git_dirty'] else ''} | ci={report['ci']} live={report['live']}"
    )
    width = max(len(c["name"]) for c in report["checks"])
    for check in report["checks"]:
        status = "PASS" if check["ok"] else "FAIL"
        print(f"  [{status}] {check['name']:<{width}}  {check['duration_ms']:>6} ms  {check['detail']}")
    summary = report["summary"]
    print(
        f"result: {'PASS' if summary['ok'] else 'FAIL'} "
        f"({summary['passed']}/{summary['total']} checks, {report['duration_ms']} ms)"
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="modelbench-gate", description=__doc__)
    parser.add_argument("--list", action="store_true", help="list checks and exit")
    parser.add_argument("--only", help="comma-separated check names")
    parser.add_argument("--ci", action="store_true", help="CI mode: datasets may be absent, no checksum")
    parser.add_argument("--live", action="store_true", help="also run a real oMLX smoke run")
    parser.add_argument("--json", action="store_true", help="print the JSON report only")
    args = parser.parse_args(argv)

    selected = list(DEFAULT_CHECKS)
    if args.live:
        selected = list(CHECKS)
    if args.only:
        wanted = [name.strip() for name in args.only.split(",") if name.strip()]
        known = {check.name for check in CHECKS}
        unknown = [name for name in wanted if name not in known]
        if unknown:
            parser.error(f"unknown check(s): {', '.join(unknown)} (known: {', '.join(sorted(known))})")
        selected = [check for check in CHECKS if check.name in wanted]
        if not selected:
            parser.error("--only selected no checks")

    if args.list:
        for check in CHECKS:
            flag = " (opt-in --live)" if check.name == "live_smoke" else ""
            print(f"{check.name:<14} {check.description}{flag}")
        return 0

    ctx = Context(root=ROOT, py=sys.executable, ci=args.ci, live=args.live)
    report = run_checks(ctx, selected)
    report_path = write_report(report)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_table(report)
        print(f"report: {report_path}")
    return 0 if report["summary"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
