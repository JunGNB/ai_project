"""``ifeval`` judge: official Google IFEval prompt-level strict accuracy (0/1).

The official checker package (``dataset/raw/ifeval/instruction_following_eval``)
is imported lazily; configuration errors stay loud inside this function and are
converted to a ``judge_error`` by :func:`modelbench.domain.judges.evaluate`.
Algorithm moved verbatim from ``tool/system/judges.py``.
"""

from __future__ import annotations

import os
import sys

from .base import IFE_PKG, NLTK_DATA

_ifeval_mod = None


def _load_ifeval():
    global _ifeval_mod
    if _ifeval_mod is None:
        if not IFE_PKG.is_dir():
            raise RuntimeError(
                f"IFEval checker not found at {IFE_PKG}; expected dataset/raw/ifeval "
                "(set MB_RAW_DIR for a deployed layout)"
            )
        os.environ["NLTK_DATA"] = str(NLTK_DATA)
        sys.path.insert(0, str(IFE_PKG))
        try:
            from instruction_following_eval import evaluation_lib  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover - environment problem
            raise RuntimeError(
                f"IFEval checker failed to import from {IFE_PKG}: {type(exc).__name__}: {exc}"
            ) from exc
        _ifeval_mod = evaluation_lib
    return _ifeval_mod


def judge_ifeval(response: str, judge: dict) -> dict:
    ev = _load_ifeval()  # configuration errors must stay loud, never score 0
    inp = ev.InputExample(
        key=0,
        instruction_id_list=judge["instruction_id_list"],
        prompt="__prompt__",
        kwargs=judge["kwargs"],
    )
    try:
        out = ev.test_instruction_following_strict(inp, {"__prompt__": response})
        ok = bool(out.follow_all_instructions)
        detail = out.follow_instruction_list
    except Exception as exc:  # checker misuse -> treat as not followed, keep evidence
        ok = False
        detail = f"checker_error: {type(exc).__name__}: {exc}"
    return {"score": 1.0 if ok else 0.0, "pass": ok, "follow": detail}
