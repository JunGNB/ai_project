"""Shared contract for the deterministic accuracy judges (single source of truth).

Five judge families, no LLM judge, no network, no rubric.  Every sample carries
its whole grading contract in the jsonl ``judge`` field; a judge only executes
that contract.  ``tool/system/judges.py`` is a thin compatibility forwarder to
this package.

The raw-data root is ``dataset/raw`` (older revisions used ``data/raw``); the
``MB_RAW_DIR`` environment variable overrides it for deployed layouts (the
IFEval official checker and ``nltk_data`` live there).
"""

from __future__ import annotations

import importlib.util
import os
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from modelbench.paths import PROJECT_ROOT

#: Repository root (kept as ``ROOT`` for the legacy ``tool/system/judges.py``).
ROOT = PROJECT_ROOT
RAW_DIR = Path(os.environ.get("MB_RAW_DIR", str(ROOT / "dataset" / "raw")))
IFE_PKG = RAW_DIR / "ifeval"
NLTK_DATA = RAW_DIR / "nltk_data"

#: Vendored official IFEval checker + the English NLTK punkt data it needs,
#: shipped inside the installed package.  Used when ``dataset/raw`` assets are
#: absent so a fresh install can always judge the ``instruction_following``
#: scenario offline (see ``_vendor/README.md``).
VENDORED_IFE_DIR = Path(__file__).resolve().parent / "_vendor"
VENDORED_IFE_PKG = VENDORED_IFE_DIR / "instruction_following_eval"
VENDORED_NLTK_DATA = VENDORED_IFE_DIR / "nltk_data"

#: Locations probed for the NLTK punkt tokenizer data; NLTK also has its own
#: search path, this list only feeds the advisory preflight check.
_NLTK_DATA_DIRS: Tuple[Path, ...] = (
    NLTK_DATA,
    VENDORED_NLTK_DATA,
    Path.home() / "nltk_data",
    Path("/usr/local/share/nltk_data"),
    Path("/usr/share/nltk_data"),
)

#: There are exactly five judge families; anything else is a contract error.
JUDGE_FAMILIES: Tuple[str, ...] = (
    "mc_letter",
    "ifeval",
    "normalized_match",
    "code_exec",
    "tool_call",
)

#: Families whose score is 0/1 (as opposed to an F1 in [0, 1]).
BINARY_FAMILIES = frozenset({"mc_letter", "ifeval", "code_exec", "tool_call"})
#: ``normalized_match`` metrics that are 0/1 rather than F1.
BINARY_METRICS = frozenset({"accuracy", "numeric_exact"})

#: Optional ``pip install -e ".[eval]"`` dependencies per family.  Only the
#: *modules actually needed* are reported so an English-only F1 run does not
#: warn about jieba/rouge.
_FAMILY_DEPENDENCIES: Dict[str, Tuple[str, ...]] = {
    "ifeval": ("absl", "nltk", "langdetect", "immutabledict"),
    "normalized_match": ("jieba", "rouge"),
}
_METRIC_DEPENDENCIES: Dict[str, Tuple[str, ...]] = {
    "f1_zh": ("jieba",),
    "rouge_zh": ("jieba", "rouge"),
}


@dataclass
class JudgeResult:
    """One deterministic judgment.

    ``passed`` is the 0/1 verdict (F1: ``score >= 1.0``); ``detail`` carries the
    evidence persisted under ``records[].score.detail`` (family/metric/pred/
    gold/reason/format_failure/judge_error/...).
    """

    score: float
    passed: bool
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Flat dict in the legacy ``judge_mc_letter(...)`` shape.

        ``score``/``pass`` stay top-level (existing callers read them) and
        ``passed`` mirrors ``pass`` for new code; ``detail`` is merged flat so
        ``format_failure``/``judge_error``/``pred``/``error`` remain reachable.
        """
        out: Dict[str, Any] = dict(self.detail)
        out["score"] = float(self.score)
        out["pass"] = bool(self.passed)
        out["passed"] = bool(self.passed)
        return out


def nfkc(text: str) -> str:
    """NFKC-normalize (full-width letters/digits -> ASCII) and tolerate None."""
    return unicodedata.normalize("NFKC", text or "")


#: Legacy private alias (``tool/system/judges.py`` re-exports ``_nfkc``).
_nfkc = nfkc


def clamp01(value: Any) -> float:
    """Coerce a judge output to ``[0, 1]``; non-numeric/None -> 0.0."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number:  # NaN
        return 0.0
    return min(1.0, max(0.0, number))


def is_binary(family: Optional[str], metric: Optional[str]) -> bool:
    """True when ``(family, metric)`` is a 0/1 judge (not F1)."""
    if family in BINARY_FAMILIES:
        return True
    if family == "normalized_match":
        return str(metric or "") in BINARY_METRICS
    return False


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError, ModuleNotFoundError):
        return False


def missing_dependencies(families: Iterable[Any]) -> List[str]:
    """Return the optional dependency *modules* missing for ``families``.

    ``families`` accepts either family names (``"ifeval"``) or judge contracts
    (``{"family": "normalized_match", "metric": "rouge_zh"}``).  Passing the
    contract narrows ``normalized_match`` to the metric actually used; passing a
    bare family name is conservative and requires the Chinese stack too.
    Detection uses :func:`importlib.util.find_spec`, so a missing heavy module is
    reported without importing it.
    """
    missing: List[str] = []
    seen: set[str] = set()

    def _add(name: str) -> None:
        if name in seen:
            return
        seen.add(name)
        if not _module_available(name):
            missing.append(name)

    for family, metric in _family_specs(families):
        if family == "normalized_match" and metric in _METRIC_DEPENDENCIES:
            deps = _METRIC_DEPENDENCIES[str(metric)]
        else:
            deps = _FAMILY_DEPENDENCIES.get(family, ())
        for dep in deps:
            _add(dep)
    return missing


def _family_specs(families: Iterable[Any]) -> List[Tuple[str, Optional[str]]]:
    """Normalize family names / judge contracts into ``(family, metric)`` pairs."""
    specs: List[Tuple[str, Optional[str]]] = []
    for item in families:
        if isinstance(item, str):
            specs.append((item, None))
        elif isinstance(item, dict):
            specs.append((str(item.get("family") or ""), item.get("metric")))
        else:  # e.g. a pydantic model / judge contract object
            specs.append(
                (
                    str(getattr(item, "family", "") or ""),
                    getattr(item, "metric", None),
                )
            )
    return specs


def has_nltk_punkt() -> bool:
    """True when NLTK ``punkt``/``punkt_tab`` data is discoverable on disk."""
    for base in _NLTK_DATA_DIRS:
        for name in ("punkt_tab", "punkt"):
            if (base / "tokenizers" / name).is_dir():
                return True
    return False


def missing_assets(families: Iterable[Any]) -> List[str]:
    """Return the *on-disk assets* missing for ``families`` (not pip modules).

    Module probing alone is not enough: the IFEval judge also needs the checker
    code and the NLTK punkt tokenizer data.  The checker ships vendored, but a
    checkout that has never run ``tool/dataset/accuracy/fetch.py`` would still
    fail inside the judge — this makes that failure visible up front.
    """
    missing: List[str] = []
    names = {family for family, _ in _family_specs(families)}
    if "ifeval" in names:
        if not (IFE_PKG / "instruction_following_eval").is_dir() and not (
            VENDORED_IFE_PKG.is_dir()
        ):
            missing.append("IFEval checker (dataset/raw/ifeval or vendored package)")
        if not has_nltk_punkt():
            missing.append("NLTK punkt data (dataset/raw/nltk_data)")
    return missing
