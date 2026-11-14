"""``code_exec`` judge: execute the official unit tests (pass@1, 0/1).

A generated solution is extracted from a fenced code block, concatenated with
the official prompt prefix and the sample's tests, then run in an isolated
subprocess (CPU/handles/file-size limits, wall-clock timeout, sentinel marker).
Algorithm moved verbatim from ``tool/system/judges.py``; this is **not** a
security sandbox (documented limitation in the accuracy test plan).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

_CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(text: str) -> str:
    if not text:
        return ""
    blocks = _CODE_FENCE.findall(text)
    if blocks:
        return max(blocks, key=len).strip("\n")
    return text.strip()


def _limits(timeout_s: int):
    def _apply():
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (timeout_s + 1, timeout_s + 2))
        # RLIMIT_AS is not settable on macOS (current limit exceeds hard max); the
        # subprocess timeout + CPU limit are the safety bounds here.
        try:  # pragma: no cover - platform specific
            resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
        except (ValueError, OSError):
            pass
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        resource.setrlimit(resource.RLIMIT_FSIZE, (8 << 20, 8 << 20))

    return _apply


def judge_code_exec(response: str, judge: dict) -> dict:
    code = extract_code(response)
    prefix = judge.get("prompt_prefix")
    # Official HumanEval protocol: prompt prefix + generated completion. The model
    # often repeats the signature; a duplicate def is harmless and keeps the imports.
    if prefix:
        code = prefix.rstrip() + "\n" + code
    marker = "__TESTS_PASS__" + uuid.uuid4().hex
    script = (
        code
        + "\n\n"
        + judge["test_code"]
        + f"\n\nprint({marker!r})\n"
    )
    timeout = int(judge.get("timeout_s", 6))
    with tempfile.TemporaryDirectory(prefix="acc_code_") as td:
        path = Path(td) / "candidate.py"
        path.write_text(script, encoding="utf-8")
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": td,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
        }
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=td,
                env=env,
                preexec_fn=_limits(timeout) if os.name == "posix" else None,
            )
            # A test run only passes when the sentinel is reached: sys.exit(0) /
            # os._exit(0) in generated code no longer yields a false pass.
            ok = proc.returncode == 0 and marker in (proc.stdout or "")
            err = "" if ok else (proc.stderr or proc.stdout or "")[-1200:]
            if not ok and proc.returncode == 0 and not err:
                err = "tests did not reach completion sentinel (exit()/os._exit()?)"
        except subprocess.TimeoutExpired:
            # The generated code hung: a deterministic failed solution (0), not a
            # judge failure (the judge itself ran correctly).
            ok, err = False, f"timeout>{timeout}s"
        except Exception as exc:  # noqa: BLE001
            # Infrastructure failure (missing interpreter, fork/exec error, ...)
            # is a *judge* failure: ``evaluate`` converts this raise into
            # status=judge_error (0 points, kept in the denominator) instead of
            # silently scoring it as a wrong answer.
            raise RuntimeError(
                f"code_exec infrastructure error: {type(exc).__name__}: {exc}"
            ) from exc
    return {"score": 1.0 if ok else 0.0, "pass": ok, "error": err}
