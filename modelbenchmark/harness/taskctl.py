"""Deterministic task board for AI-driven evolution of the benchmark project.

Tasks are plain JSON files under ``harness/tasks/`` (tracked by git).  State
transitions are explicit, journaled, and validated so multiple AI agents (or a
human + an agent) cannot silently lose or overwrite work:

    python -m harness.taskctl list
    python -m harness.taskctl show T-001
    python -m harness.taskctl claim T-001 --owner planner
    python -m harness.taskctl complete T-001 --note "..." --gate
    python -m harness.taskctl new --title "Add RULER dataset" --priority P1
    python -m harness.taskctl journal --limit 10

Design rules
------------
* One file per task; the file is the single source of truth (git-diffable).
* Mutations are atomic (write temp file + ``os.replace``) and append a line to
  ``harness/state/journal.jsonl`` (gitignored evidence).
* ``complete --gate`` refuses to close a task unless the latest gate report
  exists and passed; the report path is stored in the journal entry.
* No network, no external dependencies, deterministic ordering.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]


def _tasks_dir() -> Path:
    return Path(os.environ.get("MB_TASKS_DIR", str(ROOT / "harness" / "tasks")))


def _state_dir() -> Path:
    return Path(os.environ.get("MB_STATE_DIR", str(ROOT / "harness" / "state")))


def _journal_path() -> Path:
    return _state_dir() / "journal.jsonl"


def _report_latest() -> Path:
    return Path(os.environ.get("MB_GATE_REPORT", str(ROOT / "harness" / "reports" / "latest.json")))

STATUSES = ("pending", "in_progress", "blocked", "done")
PRIORITIES = ("P0", "P1", "P2")
ID_PATTERN = "T-"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> Optional[str]:
    try:
        import subprocess

        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10
        )
        return proc.stdout.strip() or None if proc.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def _fail(message: str) -> "NoReturn":
    print(f"taskctl: {message}", file=sys.stderr)
    raise SystemExit(1)


def _task_path(task_id: str) -> Path:
    return _tasks_dir() / f"{task_id}.json"


def load_tasks() -> List[Dict]:
    if not _tasks_dir().is_dir():
        _fail(f"tasks directory missing: {_tasks_dir()}")
    tasks: List[Dict] = []
    seen = set()
    for path in sorted(_tasks_dir().glob("*.json")):
        try:
            task = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _fail(f"cannot read {path.name}: {exc}")
        problems = validate_task(task, path)
        if problems:
            _fail(f"invalid {path.name}: " + "; ".join(problems))
        if task["id"] in seen:
            _fail(f"duplicate task id: {task['id']}")
        seen.add(task["id"])
        tasks.append(task)
    return sorted(tasks, key=lambda t: (t["priority"], t["id"]))


def validate_task(task: Dict, path: Optional[Path] = None) -> List[str]:
    problems: List[str] = []
    if not isinstance(task, dict):
        return ["task must be a JSON object"]
    for key, typ in (
        ("id", str),
        ("title", str),
        ("status", str),
        ("priority", str),
        ("scope", list),
        ("acceptance", list),
        ("notes", list),
        ("created_at", str),
        ("updated_at", str),
    ):
        if key not in task:
            problems.append(f"missing field: {key}")
        elif not isinstance(task[key], typ):
            problems.append(f"field {key} must be {typ.__name__}")
    if problems:
        return problems
    if not task["id"].startswith(ID_PATTERN) or not task["id"][2:].isdigit():
        problems.append("id must look like T-001")
    if path is not None and task["id"] != path.stem:
        problems.append(f"id {task['id']} != filename {path.name}")
    if task["status"] not in STATUSES:
        problems.append(f"invalid status: {task['status']}")
    if task["priority"] not in PRIORITIES:
        problems.append(f"invalid priority: {task['priority']}")
    if task.get("owner") is not None and not isinstance(task["owner"], str):
        problems.append("owner must be a string or null")
    if not task["title"].strip():
        problems.append("title must not be empty")
    if not task["acceptance"]:
        problems.append("acceptance must not be empty")
    return problems


def save_task(task: Dict) -> None:
    task["updated_at"] = _now()
    _tasks_dir().mkdir(parents=True, exist_ok=True)
    payload = json.dumps(task, ensure_ascii=False, indent=2) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=".task-", suffix=".tmp", dir=str(_tasks_dir()))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp_name, _task_path(task["id"]))
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def journal(action: str, task: Dict, **extra) -> None:
    _state_dir().mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": _now(),
        "action": action,
        "task": task["id"],
        "status": task["status"],
        "owner": task.get("owner"),
        "commit": _git_commit(),
        **extra,
    }
    with _journal_path().open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _load_one(task_id: str) -> Dict:
    path = _task_path(task_id)
    if not path.is_file():
        _fail(f"unknown task: {task_id}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"cannot read {path.name}: {exc}")


def _require_status(task: Dict, allowed: tuple, action: str) -> None:
    if task["status"] not in allowed:
        _fail(
            f"cannot {action} {task['id']}: status is '{task['status']}', "
            f"expected one of {allowed}"
        )


def _print_task(task: Dict) -> None:
    print(f"{task['id']}  [{task['priority']}]  {task['status']}  owner={task.get('owner') or '-'}")
    print(f"title: {task['title']}")
    if task.get("context"):
        print(f"context: {task['context']}")
    if task.get("scope"):
        print("scope:")
        for item in task["scope"]:
            print(f"  - {item}")
    print("acceptance:")
    for item in task["acceptance"]:
        print(f"  - {item}")
    if task.get("links"):
        print("links: " + ", ".join(task["links"]))
    if task.get("notes"):
        print("notes:")
        for note in task["notes"]:
            print(f"  - {note}")


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #


def cmd_list(args: argparse.Namespace) -> int:
    tasks = load_tasks()
    if args.status:
        wanted = {s.strip() for s in args.status.split(",") if s.strip()}
        unknown = wanted - set(STATUSES)
        if unknown:
            _fail(f"unknown status: {', '.join(sorted(unknown))}")
        tasks = [t for t in tasks if t["status"] in wanted]
    if args.owner:
        tasks = [t for t in tasks if (t.get("owner") or "") == args.owner]
    if args.json:
        print(json.dumps(tasks, ensure_ascii=False, indent=2))
        return 0
    if not tasks:
        print("(no tasks)")
        return 0
    print(f"{'id':<7} {'pri':<4} {'status':<12} {'owner':<12} title")
    for task in tasks:
        title = task["title"] if len(task["title"]) <= 70 else task["title"][:67] + "..."
        print(f"{task['id']:<7} {task['priority']:<4} {task['status']:<12} {(task.get('owner') or '-'):<12} {title}")
    counts = {status: sum(1 for t in tasks if t["status"] == status) for status in STATUSES}
    print("\n" + "  ".join(f"{k}={v}" for k, v in counts.items()))
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    task = _load_one(args.task_id)
    if args.json:
        print(json.dumps(task, ensure_ascii=False, indent=2))
    else:
        _print_task(task)
    return 0


def cmd_claim(args: argparse.Namespace) -> int:
    task = _load_one(args.task_id)
    _require_status(task, ("pending",), "claim")
    owner = args.owner or os.environ.get("AGENT_NAME") or "agent"
    task["status"] = "in_progress"
    task["owner"] = owner
    save_task(task)
    journal("claim", task, note=args.note)
    print(f"{task['id']} claimed by {owner}")
    return 0


def cmd_complete(args: argparse.Namespace) -> int:
    task = _load_one(args.task_id)
    _require_status(task, ("in_progress",), "complete")
    if not args.note.strip():
        _fail("complete requires --note describing what changed and the evidence")
    report_path: Optional[Path] = None
    if args.gate:
        if not _report_latest().is_file():
            _fail("--gate requested but harness/reports/latest.json is missing; run `python -m harness.gate`")
        try:
            report = json.loads(_report_latest().read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _fail(f"cannot read gate report: {exc}")
        if not report.get("summary", {}).get("ok"):
            _fail("--gate requested but the latest gate report did not pass")
        report_path = _report_latest()
    task["status"] = "done"
    task["notes"].append({"at": _now(), "owner": task.get("owner"), "note": args.note})
    save_task(task)
    journal(
        "complete",
        task,
        note=args.note,
        gate=str(report_path) if report_path else None,
        gate_commit=(json.loads(report_path.read_text(encoding="utf-8")).get("git_commit") if report_path else None),
    )
    print(f"{task['id']} completed")
    return 0


def cmd_block(args: argparse.Namespace) -> int:
    task = _load_one(args.task_id)
    _require_status(task, ("in_progress",), "block")
    if not args.reason.strip():
        _fail("block requires --reason (the concrete blocking condition)")
    task["status"] = "blocked"
    task["notes"].append({"at": _now(), "owner": task.get("owner"), "note": f"BLOCKED: {args.reason}"})
    save_task(task)
    journal("block", task, note=args.reason)
    print(f"{task['id']} blocked: {args.reason}")
    return 0


def cmd_reopen(args: argparse.Namespace) -> int:
    task = _load_one(args.task_id)
    _require_status(task, ("blocked", "done"), "reopen")
    task["status"] = "pending"
    task["owner"] = None
    save_task(task)
    journal("reopen", task, note=args.note)
    print(f"{task['id']} reopened")
    return 0


def cmd_new(args: argparse.Namespace) -> int:
    tasks = load_tasks()
    numbers = [int(t["id"][2:]) for t in tasks] or [0]
    task_id = f"{ID_PATTERN}{max(numbers) + 1:03d}"
    task = {
        "id": task_id,
        "title": args.title.strip(),
        "status": "pending",
        "priority": args.priority,
        "owner": None,
        "created_at": _now(),
        "updated_at": _now(),
        "scope": list(args.scope or []),
        "context": args.context or "",
        "acceptance": list(args.acceptance or ["python -m harness.gate passes"]),
        "notes": [],
        "links": list(args.link or []),
    }
    problems = validate_task(task)
    if problems:
        _fail("; ".join(problems))
    save_task(task)
    journal("new", task, note=args.context)
    print(f"created {task_id}: {task['title']}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    tasks = load_tasks()
    problems: List[str] = []
    for task in tasks:
        if task["status"] == "in_progress" and not task.get("owner"):
            problems.append(f"{task['id']}: in_progress without owner")
        if task["status"] == "done" and not task.get("notes"):
            problems.append(f"{task['id']}: done without notes")
        if task["status"] == "blocked" and not any(
            isinstance(n, dict) and str(n.get("note", "")).startswith("BLOCKED:")
            for n in task.get("notes", [])
        ):
            problems.append(f"{task['id']}: blocked without BLOCKED note")
    if problems:
        print("task board problems:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"task board ok ({len(tasks)} tasks)")
    return 0


def cmd_journal(args: argparse.Namespace) -> int:
    if not _journal_path().is_file():
        print("(journal empty)")
        return 0
    lines = _journal_path().read_text(encoding="utf-8").splitlines()
    for line in lines[-args.limit :]:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        print(json.dumps(entry, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="modelbench-task", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", help="list tasks")
    p.add_argument("--status", help="comma-separated statuses")
    p.add_argument("--owner")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("show", help="show one task")
    p.add_argument("task_id")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("claim", help="claim a pending task")
    p.add_argument("task_id")
    p.add_argument("--owner")
    p.add_argument("--note", default="")
    p.set_defaults(func=cmd_claim)

    p = sub.add_parser("complete", help="complete an in-progress task")
    p.add_argument("task_id")
    p.add_argument("--note", required=True)
    p.add_argument("--gate", action="store_true", help="require a passing harness/reports/latest.json")
    p.set_defaults(func=cmd_complete)

    p = sub.add_parser("block", help="block an in-progress task")
    p.add_argument("task_id")
    p.add_argument("--reason", required=True)
    p.set_defaults(func=cmd_block)

    p = sub.add_parser("reopen", help="reopen a blocked/done task")
    p.add_argument("task_id")
    p.add_argument("--note", default="")
    p.set_defaults(func=cmd_reopen)

    p = sub.add_parser("new", help="create a task")
    p.add_argument("--title", required=True)
    p.add_argument("--priority", choices=PRIORITIES, default="P1")
    p.add_argument("--scope", action="append", help="file/dir expected to change (repeatable)")
    p.add_argument("--acceptance", action="append", help="acceptance criterion (repeatable)")
    p.add_argument("--context", default="")
    p.add_argument("--link", action="append", help="reference path/URL (repeatable)")
    p.set_defaults(func=cmd_new)

    p = sub.add_parser("check", help="validate the task board")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("journal", help="show recent journal entries")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_journal)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
