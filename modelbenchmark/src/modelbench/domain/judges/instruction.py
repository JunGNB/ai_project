"""``ifeval`` judge: official Google IFEval prompt-level strict accuracy (0/1).

The official checker is imported lazily from one of two places:

1. ``dataset/raw/ifeval/instruction_following_eval`` (``$MB_RAW_DIR`` aware) —
   the offline asset fetched by ``tool/dataset/accuracy/fetch.py``; a checkout
   can pin/override the checker here;
2. the vendored copy shipped with the package at
   ``modelbench/domain/judges/_vendor/instruction_following_eval`` — always
   present once installed, so a fresh install never loses this scenario.

Configuration errors stay loud inside this function and are converted to a
``judge_error`` by :func:`modelbench.domain.judges.evaluate`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple

from .base import (
    IFE_PKG,
    NLTK_DATA,
    VENDORED_IFE_DIR,
    VENDORED_IFE_PKG,
    VENDORED_NLTK_DATA,
)

_ifeval_mod = None


def _checker_present(root: Path) -> bool:
    """True when *root* contains a usable ``instruction_following_eval`` package."""
    return (root / "instruction_following_eval" / "evaluation_lib.py").is_file()


def checker_root() -> Tuple[Path, str]:
    """Return ``(import_root, source_label)`` for the official checker.

    ``dataset/raw/ifeval`` wins when present (explicit override / pinned copy);
    otherwise the vendored package is used.  A missing/incomplete checker raises
    instead of silently importing a half-written directory, so the failure shows
    up as ``judge_error`` with evidence.
    """
    if _checker_present(IFE_PKG):
        return IFE_PKG, "dataset/raw/ifeval"
    if _checker_present(VENDORED_IFE_DIR):
        return VENDORED_IFE_DIR, "vendored package"
    raise RuntimeError(
        "IFEval checker not found: expected "
        "dataset/raw/ifeval/instruction_following_eval/evaluation_lib.py (set "
        f"MB_RAW_DIR for a deployed layout) or the vendored copy at {VENDORED_IFE_PKG}"
    )


def _ensure_nltk_data() -> None:
    """Pick a usable NLTK data dir and fail loudly when punkt is absent.

    ``dataset/raw/nltk_data`` wins when present (explicit override); otherwise
    the English punkt data vendored inside the package is used, so a plain
    install judges IFEval offline with no extra download.
    """
    data_dir: Path | None = None
    if (NLTK_DATA / "tokenizers" / "punkt").is_dir():
        data_dir = NLTK_DATA
    elif (VENDORED_NLTK_DATA / "tokenizers" / "punkt").is_dir():
        data_dir = VENDORED_NLTK_DATA
    if data_dir is not None:
        os.environ["NLTK_DATA"] = str(data_dir)
    try:
        import nltk  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - core dependency
        raise RuntimeError(
            "IFEval requires the 'nltk' package; re-run `./modelbench setup` "
            "(nltk is a core dependency now)"
        ) from exc
    if data_dir is not None and str(data_dir) not in nltk.data.path:
        # nltk.data.path is built at import time; make the choice apply even when
        # another module imported nltk before this judge ran.
        nltk.data.path.insert(0, str(data_dir))
    for resource in ("tokenizers/punkt_tab", "tokenizers/punkt"):
        try:
            nltk.data.find(resource)
            return
        except LookupError:
            continue
    raise RuntimeError(
        "IFEval requires the NLTK punkt tokenizer data; expected "
        f"{NLTK_DATA} (or a system NLTK data path). Fix: "
        "python tool/dataset/accuracy/fetch.py --only nltk"
    )


def _load_ifeval():
    global _ifeval_mod
    if _ifeval_mod is None:
        root, label = checker_root()
        _ensure_nltk_data()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            from instruction_following_eval import evaluation_lib  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover - environment problem
            raise RuntimeError(
                f"IFEval checker failed to import from {root} ({label}): "
                f"{type(exc).__name__}: {exc}"
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
