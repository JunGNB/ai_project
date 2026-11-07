"""HumanEval-style execution evaluator (pass@1).

The model completion is glued onto the dataset prompt, the official unit test is
appended, and the resulting program is executed as a child process.  The sample
passes when the process exits with status ``0`` before the timeout.

Security / safety boundary
--------------------------
This is the *only* evaluator that executes bytes derived from a model response.
Mitigations in place:

* a fresh :class:`tempfile.TemporaryDirectory` is used as both ``cwd`` and
  ``TMPDIR`` and is deleted afterwards;
* a 10 s wall-clock timeout, after which the whole process group is killed
  (``start_new_session=True`` + ``os.killpg``);
* POSIX resource limits: ``RLIMIT_CPU`` (10 s) and ``RLIMIT_FSIZE`` (4 MiB);
* isolated interpreter (``python -I -S``): no user site, no ``PYTHON*``
  environment inheritance, no script/cwd directory on ``sys.path``;
* stdin is ``/dev/null``; stdout/stderr are bounded files and only their first
  500 characters are kept in ``Score.detail``.

Known limitations (documented, not solved here):

* this is **not** a security sandbox -- generated code can still read/write any
  file the current user can access and can open network connections;
* ``-S`` disables site-packages, so completions importing third-party modules
  fail even when the module is installed (stdlib imports keep working);
* fork bombs / memory exhaustion are only partially mitigated (CPU + file-size
  limits, no address-space limit because CPython needs a large virtual space);
* a hard 10 s timeout makes pathological-but-correct solutions count as wrong.

Run untrusted models only on a disposable or containerised host.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..models import RawResponse, Sample, Score

#: Wall-clock limit for one program execution (seconds).  Kept as a module
#: attribute so unit tests can lower it without waiting ten seconds.
EXEC_TIMEOUT_S: float = 10.0

#: Hard cap on how many bytes of stdout/stderr are copied into ``Score.detail``.
_OUTPUT_CHARS = 500

#: Hard cap on the bytes a generated program may write to a single file.
_FSIZE_LIMIT = 4 * 1024 * 1024


class _ExecutionError(Exception):
    """Internal marker for malformed samples (never escapes the module)."""


_FENCE_RE = re.compile(r"```[ \t]*(?:[A-Za-z0-9_+.\-]*)[ \t]*\r?\n(.*?)```", re.S)


def _strip_fences(text: str, entry_point: str) -> str:
    """Return the code inside markdown fences (preferring the function block)."""
    blocks = [match.group(1) for match in _FENCE_RE.finditer(text or "")]
    if not blocks:
        return text or ""
    if entry_point:
        preferred = [block for block in blocks if f"def {entry_point}" in block]
        if preferred:
            return max(preferred, key=len)
    return "\n\n".join(blocks)


def _def_pattern(entry_point: str) -> "re.Pattern[str]":
    return re.compile(
        r"^[ \t]*(?:async[ \t]+)?def[ \t]+" + re.escape(entry_point) + r"[ \t]*\(",
        re.M,
    )


def clean_code_completion(text: str, entry_point: str, prompt: str = "") -> str:
    """Turn raw model output into a code fragment suitable for ``prompt + code``.

    * markdown fences are removed (the block containing the entry point wins),
    * chatter before the first ``def <entry_point>(`` is dropped,
    * a re-printed copy of the dataset prompt is dropped when the model repeated
      the whole signature,
    * surrounding blank lines are trimmed.

    The result may either be a *body* (indented statements) or a *complete
    function*; :func:`build_program` decides how to splice it.
    """
    code = _strip_fences(text or "", entry_point)
    if not code.strip():
        return ""
    if entry_point:
        match = _def_pattern(entry_point).search(code)
        if match:
            code = code[match.start():]
    if prompt:
        stripped_prompt = prompt.strip()
        if stripped_prompt:
            candidate = code.lstrip()
            if candidate.startswith(stripped_prompt):
                code = candidate[len(stripped_prompt):]
    return code.strip("\n") + "\n"


def _split_prompt(prompt: str, entry_point: str) -> Tuple[str, bool]:
    """Split the prompt into (imports/prefix, has_signature).

    For HumanEval the prompt is ``<imports>\\n\\ndef f(...):\\n    \"\"\"...\"\"\"``;
    the prefix keeps the imports so a completion that re-defines the whole
    function can replace the prompt's signature.
    """
    if not entry_point:
        return prompt or "", False
    match = _def_pattern(entry_point).search(prompt or "")
    if match:
        return (prompt or "")[:match.start()], True
    return prompt or "", False


def _indent_body(body: str) -> str:
    """Indent a de-indented function body so it can be appended to a signature."""
    lines = body.splitlines()
    if not lines:
        return body
    first_code = next((line for line in lines if line.strip()), "")
    if not first_code or first_code[:1] in " \t":
        return body
    if re.match(r"^(?:def |class |import |from |@|#)", first_code):
        return body
    return "\n".join(("    " + line) if line.strip() else line for line in lines)


def _append_tests(program: str, test: str, entry_point: str) -> str:
    program = program.rstrip("\n") + "\n\n"
    if test.strip():
        program += test.strip("\n") + "\n\n"
    program += f"check({entry_point})\n"
    return program


def build_program(sample: Sample, completion: str) -> Tuple[str, Dict[str, Any]]:
    """Build the executable program for one completion.

    Returns ``(program, info)``; ``info`` records how the completion was spliced
    and whether the prompt signature had to be bypassed.
    """
    metadata = sample.metadata or {}
    prompt = metadata.get("prompt") or ""
    test = metadata.get("test") or ""
    entry_point = metadata.get("entry_point") or ""
    if not entry_point:
        raise _ExecutionError("sample.metadata['entry_point'] is missing")
    if not test:
        raise _ExecutionError("sample.metadata['test'] is missing")

    cleaned = completion
    prefix, has_signature = _split_prompt(prompt, entry_point)
    completion_defines = _def_pattern(entry_point).search(cleaned) is not None

    if completion_defines:
        # The model re-printed the whole function: drop the prompt's own
        # (now duplicate) signature and keep only the imports.
        program = prefix + cleaned
        mode = "replaced_signature"
    else:
        program = prompt.rstrip("\n") + "\n" + cleaned
        mode = "appended_body"
    program = _append_tests(program, test, entry_point)
    return program, {"mode": mode, "has_prompt_signature": has_signature}


def _compiles(source: str) -> Tuple[bool, Optional[str]]:
    """Syntax-check ``source`` without executing it."""
    try:
        compile(source, "<human_eval_program>", "exec")
        return True, None
    except SyntaxError as exc:
        return False, f"{exc.msg} (line {exc.lineno})"
    except (ValueError, TypeError) as exc:  # null bytes, bad encoding, ...
        return False, f"{type(exc).__name__}: {exc}"


def _resource_limits() -> None:
    """Apply POSIX rlimits inside the child process (best effort)."""
    try:
        import resource  # local import: unavailable on Windows

        cpu_limit = max(1, int(EXEC_TIMEOUT_S) + 2)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
        resource.setrlimit(resource.RLIMIT_FSIZE, (_FSIZE_LIMIT, _FSIZE_LIMIT))
    except Exception:  # pragma: no cover - resource module not available
        pass


def _kill_process_group(proc: "subprocess.Popen[Any]") -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:  # pragma: no cover - already gone
            pass


def _read_head(path: str, limit: int = _OUTPUT_CHARS) -> str:
    try:
        with open(path, "rb") as handle:
            data = handle.read(limit * 8)
    except OSError:  # pragma: no cover - defensive
        return ""
    # Keep the first ``limit`` characters after lossy UTF-8 decoding.
    return data.decode("utf-8", errors="replace")[:limit]


def run_program(program: str, timeout_s: Optional[float] = None) -> Dict[str, Any]:
    """Execute ``program`` in a temporary directory and report the outcome."""
    timeout = float(EXEC_TIMEOUT_S if timeout_s is None else timeout_s)
    workdir = tempfile.mkdtemp(prefix="humaneval_")
    try:
        script_path = os.path.join(workdir, "solution.py")
        with open(script_path, "w", encoding="utf-8") as handle:
            handle.write(program)
        out_path = os.path.join(workdir, "stdout.bin")
        err_path = os.path.join(workdir, "stderr.bin")
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": workdir,
            "TMPDIR": workdir,
            "PYTHONIOENCODING": "utf-8",
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "LC_ALL": "C.UTF-8",
            "LANG": "C.UTF-8",
        }
        cmd = [sys.executable, "-I", "-S", script_path]
        timed_out = False
        with open(out_path, "wb") as out_handle, open(err_path, "wb") as err_handle:
            proc = subprocess.Popen(
                cmd,
                cwd=workdir,
                stdin=subprocess.DEVNULL,
                stdout=out_handle,
                stderr=err_handle,
                env=env,
                start_new_session=(os.name == "posix"),
                preexec_fn=_resource_limits if os.name == "posix" else None,
            )
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill_process_group(proc)
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:  # pragma: no cover - stuck child
                    _kill_process_group(proc)
        return {
            "exit_code": proc.returncode,
            "timed_out": timed_out,
            "stdout": _read_head(out_path),
            "stderr": _read_head(err_path),
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def score_code(sample: Sample, response: RawResponse) -> Score:
    """Score one HumanEval sample by executing the code against its unit test.

    ``Score.correct`` is ``True`` iff the program exits ``0`` before the
    timeout.  ``detail`` contains ``stdout``/``stderr`` (first 500 chars each),
    the exit code, the timeout flag and the splice mode used.
    """
    try:
        metadata = sample.metadata or {}
        entry_point = metadata.get("entry_point") or ""
        if not entry_point:
            raise _ExecutionError("sample.metadata['entry_point'] is missing")
        if not (metadata.get("test") or ""):
            raise _ExecutionError("sample.metadata['test'] is missing")

        text = (response.text or "") if response is not None else ""
        completion = clean_code_completion(text, entry_point, metadata.get("prompt") or "")
        if not completion.strip():
            return Score(
                correct=False,
                score=0.0,
                detail={"error": "empty completion", "entry_point": entry_point},
            )

        program, info = build_program(sample, completion)
        ok, syntax_error = _compiles(program)
        used_fallback = False
        if not ok:
            # Common failure mode: the model emitted an unindented body.
            prefix, has_signature = _split_prompt(
                metadata.get("prompt") or "", entry_point
            )
            if has_signature and _def_pattern(entry_point).search(completion) is None:
                fallback_completion = _indent_body(completion)
                fallback_program, fallback_info = build_program(
                    sample, fallback_completion
                )
                fallback_ok, fallback_error = _compiles(fallback_program)
                if fallback_ok:
                    program, info, used_fallback = (
                        fallback_program,
                        fallback_info,
                        True,
                    )
                    ok, syntax_error = True, None
                else:
                    syntax_error = fallback_error or syntax_error
            if not ok:
                return Score(
                    correct=False,
                    score=0.0,
                    detail={
                        "error": "syntax_error",
                        "syntax_error": syntax_error,
                        "entry_point": entry_point,
                        "mode": info.get("mode"),
                        "first_500_chars": text[:500],
                    },
                )

        result = run_program(program)
        passed = (not result["timed_out"]) and result["exit_code"] == 0
        detail: Dict[str, Any] = {
            "passed": passed,
            "exit_code": result["exit_code"],
            "timed_out": result["timed_out"],
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "entry_point": entry_point,
            "mode": info.get("mode"),
            "used_indent_fallback": used_fallback,
            "program_chars": len(program),
        }
        if result["timed_out"]:
            detail["error"] = f"timeout after {EXEC_TIMEOUT_S:g}s"
        elif result["exit_code"] != 0:
            detail["error"] = f"non-zero exit code {result['exit_code']}"
        return Score(correct=passed, score=1.0 if passed else 0.0, detail=detail)
    except _ExecutionError as exc:
        return Score(
            correct=False,
            score=0.0,
            detail={"error": str(exc)},
        )
    except Exception as exc:  # pragma: no cover - defensive: never raise upstream
        return Score(
            correct=False,
            score=0.0,
            detail={"error": f"{type(exc).__name__}: {exc}"},
        )


def score_code_batch(
    samples: Sequence[Sample], responses: Sequence[RawResponse]
) -> List[Score]:
    """Convenience wrapper: score paired samples/responses without raising."""
    scores: List[Score] = []
    for sample, response in zip(samples, responses):
        try:
            scores.append(score_code(sample, response))
        except Exception as exc:  # pragma: no cover - score_code is defensive
            scores.append(
                Score(
                    correct=False,
                    score=0.0,
                    detail={"error": f"{type(exc).__name__}: {exc}"},
                )
            )
    return scores
