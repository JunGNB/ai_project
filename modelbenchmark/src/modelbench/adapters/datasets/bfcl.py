"""BFCL v4 (Berkeley Function Calling Leaderboard) dataset adapter.

Source (GitHub, ``ShishirPatil/gorilla`` main branch)::

    berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_<category>.json
    berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v4_<category>.json

Four categories are supported: ``simple``, ``multiple``, ``parallel`` and
``multi_turn`` (file ``BFCL_v4_multi_turn_base.json``).  The question files are
actually JSONL (one object per line) despite the ``.json`` suffix, and they do
**not** contain the ground truth: the expected calls live in the sibling
``possible_answer/`` files, joined by sample ``id``.  Both files are downloaded
and verified; if a category's files cannot be fetched it is skipped (and
recorded in ``data/raw/bfcl/download_errors.json``) rather than fabricated.

Ground-truth shapes differ by category and are normalised to
``metadata["gold_calls"] = [{"name": ..., "arguments": {...}}, ...]``:

* simple / multiple / parallel — ``ground_truth`` is a list of
  ``{function_name: {argument: [acceptable values]}}`` dicts; single-element
  alternative lists are unwrapped to scalars, multi-value lists are kept as
  acceptable alternatives.
* multi_turn — ``ground_truth`` is a list of turns, each turn a list of Python
  call expressions such as ``"cd(folder='document')"``; these are parsed with
  :mod:`ast` into ``name`` / ``arguments`` / ``positional_arguments`` and
  tagged with the turn number.

Samples are single-request flavoured: ``question`` is the concatenation of the
user turns, ``tools`` is the sample's OpenAI-style function list (for
``multi_turn`` the upstream file carries no per-sample function list, so
``tools`` stays ``None`` and ``metadata["involved_classes"]`` is set instead).
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ...domain.models import AnswerType, Sample, ToolSpec
from ._download import DownloadError, download_file
from .base import DatasetAdapter

_URL_BASE = (
    "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/"
    "berkeley-function-call-leaderboard/bfcl_eval/data"
)

#: category id -> question/ground-truth file name
CATEGORY_FILES: Dict[str, str] = {
    "simple": "BFCL_v4_simple_python.json",
    "multiple": "BFCL_v4_multiple.json",
    "parallel": "BFCL_v4_parallel.json",
    "multi_turn": "BFCL_v4_multi_turn_base.json",
}

CATEGORY_ORDER: tuple[str, ...] = ("simple", "multiple", "parallel", "multi_turn")

_CATEGORY_ALIASES: Dict[str, str] = {
    "simple": "simple",
    "simple_python": "simple",
    "multiple": "multiple",
    "parallel": "parallel",
    "multi_turn": "multi_turn",
    "multi_turn_base": "multi_turn",
}

DEFAULT_PER_CATEGORY = 10
GROUND_TRUTH_DIR = "possible_answer"
ERRORS_FILE = "download_errors.json"


def normalize_category(value: str) -> str:
    category = _CATEGORY_ALIASES.get(str(value).strip().lower())
    if category is None:
        raise ValueError(
            f"unknown BFCL category {value!r}; use one of {list(CATEGORY_ORDER)} "
            "(aliases: simple_python, multi_turn_base)"
        )
    return category


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read BFCL's JSONL files (one JSON object per line)."""

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {path}: {exc}") from exc
    return rows


def _flatten_turns(turns: List[List[Dict[str, Any]]]) -> str:
    """Concatenate the user messages of all turns into one question string."""

    parts: List[str] = []
    for turn in turns or []:
        for message in turn or []:
            if str(message.get("role", "")) == "user" and message.get("content"):
                parts.append(str(message["content"]))
    return "\n".join(parts)


def _normalize_turns(turns: List[List[Dict[str, Any]]]) -> List[List[Dict[str, str]]]:
    """Project every message onto ``{"role", "content"}`` for JSON-safe metadata."""

    normalized: List[List[Dict[str, str]]] = []
    for turn in turns or []:
        normalized.append(
            [
                {
                    "role": str(message.get("role", "")),
                    "content": str(message.get("content", "")),
                }
                for message in turn or []
            ]
        )
    return normalized


def _unwrap_alternatives(value: Any) -> Tuple[Any, bool]:
    """Unwrap BFCL acceptable-value lists.

    ``[10]`` becomes ``10``; ``["units", ""]`` stays a list because both are
    acceptable values.  The second return value flags the multi-value case.
    """

    if isinstance(value, list):
        if len(value) == 1:
            unwrapped, _ = _unwrap_alternatives(value[0])
            return unwrapped, False
        items = [_unwrap_alternatives(item) for item in value]
        return [item for item, _ in items], True
    if isinstance(value, dict):
        result: Dict[str, Any] = {}
        multiple = False
        for key, item in value.items():
            result[key], has_alternatives = _unwrap_alternatives(item)
            multiple = multiple or has_alternatives
        return result, multiple
    return value, False


def _parse_call_expression(expression: str) -> Dict[str, Any]:
    """Parse a BFCL multi-turn call string like ``cd(folder='document')``."""

    text = str(expression).strip()
    call: Dict[str, Any] = {"name": text, "arguments": {}, "positional_arguments": []}
    try:
        node = ast.parse(text, mode="eval").body
    except SyntaxError as exc:
        call["parse_error"] = f"SyntaxError: {exc.msg}"
        return call
    if not isinstance(node, ast.Call):
        call["parse_error"] = f"not a call expression: {type(node).__name__}"
        return call
    call["name"] = ast.unparse(node.func)
    for keyword in node.keywords:
        if keyword.arg is None:
            call.setdefault("parse_error", "call uses **kwargs")
            continue
        try:
            call["arguments"][keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, TypeError):
            call["arguments"][keyword.arg] = ast.unparse(keyword.value)
    for argument in node.args:
        try:
            call["positional_arguments"].append(ast.literal_eval(argument))
        except (ValueError, TypeError):
            call["positional_arguments"].append(ast.unparse(argument))
    return call


def _calls_from_single_turn_ground_truth(ground_truth: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for call_map in ground_truth:
        if not isinstance(call_map, dict):
            continue
        for name, arguments in call_map.items():
            if not isinstance(arguments, dict):
                continue
            parsed: Dict[str, Any] = {}
            has_alternatives = False
            for argument_name, value in arguments.items():
                parsed[argument_name], multiple = _unwrap_alternatives(value)
                has_alternatives = has_alternatives or multiple
            call: Dict[str, Any] = {"name": str(name), "arguments": parsed}
            if has_alternatives:
                call["has_alternatives"] = True
            calls.append(call)
    return calls


def _calls_from_multi_turn_ground_truth(ground_truth: List[Any]) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for turn_index, turn in enumerate(ground_truth):
        if not isinstance(turn, list):
            continue
        for item in turn:
            if isinstance(item, str):
                call = _parse_call_expression(item)
                call["turn"] = turn_index
                calls.append(call)
            elif isinstance(item, dict):
                for name, arguments in item.items():
                    parsed: Dict[str, Any] = {}
                    has_alternatives = False
                    for argument_name, value in (arguments or {}).items():
                        parsed[argument_name], multiple = _unwrap_alternatives(value)
                        has_alternatives = has_alternatives or multiple
                    call = {"name": str(name), "arguments": parsed, "turn": turn_index}
                    if has_alternatives:
                        call["has_alternatives"] = True
                    calls.append(call)
    return calls


def parse_ground_truth(ground_truth: Any) -> List[Dict[str, Any]]:
    """Normalise either BFCL ground-truth shape into a list of gold calls."""

    if not isinstance(ground_truth, list) or not ground_truth:
        return []
    if all(isinstance(item, list) for item in ground_truth):
        return _calls_from_multi_turn_ground_truth(ground_truth)
    return _calls_from_single_turn_ground_truth(ground_truth)


class BFCLAdapter(DatasetAdapter):
    """BFCL v4 function-calling samples (questions + ground truth)."""

    name = "bfcl"
    scenario = "general_agent"
    source = (
        "https://github.com/ShishirPatil/gorilla (main branch, "
        "berkeley-function-call-leaderboard/bfcl_eval/data, BFCL v4)"
    )
    license = "Apache-2.0"

    def __init__(self, raw_dir: Path, processed_dir: Path, params: Optional[Dict[str, Any]] = None):
        super().__init__(raw_dir, processed_dir, params)
        #: populated by :meth:`ensure_downloaded` (also written to disk)
        self.download_errors: List[Dict[str, Any]] = []
        #: categories that could not be loaded because files are missing
        self.skipped_categories: List[str] = []

    # ---- paths ----------------------------------------------------------

    def categories(self) -> List[str]:
        wanted = self.params.get("categories")
        if wanted is None:
            return list(CATEGORY_ORDER)
        if isinstance(wanted, str):
            wanted = [part for part in wanted.split(",") if part.strip()]
        ordered: List[str] = []
        for item in wanted:
            category = normalize_category(item)
            if category not in ordered:
                ordered.append(category)
        return ordered

    def question_file(self, category: str) -> Path:
        return self.raw_dir / CATEGORY_FILES[category]

    def ground_truth_file(self, category: str) -> Path:
        return self.raw_dir / GROUND_TRUTH_DIR / CATEGORY_FILES[category]

    # ---- download -------------------------------------------------------

    def ensure_downloaded(self) -> Path:
        force = bool(self.params.get("force_download", False))
        errors: List[Dict[str, Any]] = []
        available = 0
        for category in self.categories():
            file_name = CATEGORY_FILES[category]
            for kind, url, dest in (
                ("questions", f"{_URL_BASE}/{file_name}", self.question_file(category)),
                (
                    "ground_truth",
                    f"{_URL_BASE}/{GROUND_TRUTH_DIR}/{file_name}",
                    self.ground_truth_file(category),
                ),
            ):
                try:
                    download_file(
                        url,
                        dest,
                        raw_root=self.raw_dir,
                        min_bytes=10_000,
                        force=force,
                    )
                except DownloadError as exc:
                    errors.append(
                        {
                            "category": category,
                            "kind": kind,
                            "url": url,
                            "path": str(dest),
                            "error": str(exc),
                        }
                    )
            if self.question_file(category).is_file() and self.ground_truth_file(category).is_file():
                available += 1
        self.download_errors = errors
        _write_download_errors(self.raw_dir, errors)
        if available == 0:
            raise DownloadError(
                "no BFCL category could be downloaded; see "
                f"{self.raw_dir / ERRORS_FILE} for details"
            )
        return self.raw_dir

    # ---- load -----------------------------------------------------------

    def load(self) -> List[Sample]:
        limit = _optional_int(self.params.get("limit"), "limit")
        per_category = self.params.get(
            "per_category", self.params.get("limit_per_category")
        )
        per_category = _optional_int(per_category, "per_category")
        if limit is None and per_category is None:
            per_category = DEFAULT_PER_CATEGORY

        buckets: Dict[str, List[Sample]] = {}
        self.skipped_categories = []
        for category in self.categories():
            question_path = self.question_file(category)
            truth_path = self.ground_truth_file(category)
            if not question_path.is_file() or not truth_path.is_file():
                self.skipped_categories.append(category)
                print(
                    f"[bfcl] skipping category {category!r}: missing "
                    f"{question_path.name if not question_path.is_file() else truth_path.name}",
                    file=sys.stderr,
                )
                continue
            truth_by_id = {
                str(row.get("id")): row.get("ground_truth")
                for row in _read_jsonl(truth_path)
            }
            bucket: List[Sample] = []
            for row in _read_jsonl(question_path):
                if per_category is not None and len(bucket) >= per_category:
                    break
                bucket.append(self._row_to_sample(category, row, truth_by_id))
            buckets[category] = bucket

        if not buckets:
            raise FileNotFoundError(
                "BFCL: no raw files available for the requested categories "
                f"{self.categories()}; run `modelbenchmark download --dataset bfcl`"
            )

        if limit is None:
            return [
                sample
                for category in self.categories()
                for sample in buckets.get(category) or []
            ]

        # An overall cap is filled round-robin so a small limit spans categories
        # instead of taking only the first category.
        selected: List[Sample] = []
        depth = max((len(bucket) for bucket in buckets.values()), default=0)
        for level in range(depth):
            for category in self.categories():
                bucket = buckets.get(category) or []
                if level < len(bucket):
                    selected.append(bucket[level])
                    if len(selected) >= limit:
                        return selected
        return selected

    @staticmethod
    def _row_to_sample(
        category: str,
        row: Dict[str, Any],
        truth_by_id: Dict[str, Any],
    ) -> Sample:
        raw_id = str(row.get("id", ""))
        turns = row.get("question") or []
        if not isinstance(turns, list):
            raise ValueError(f"BFCL {category} sample {raw_id} has malformed 'question'")

        functions = row.get("function") or []
        tools: Optional[List[ToolSpec]] = None
        if functions:
            tools = [ToolSpec(function=dict(function)) for function in functions]

        gold_calls = parse_ground_truth(truth_by_id.get(raw_id))
        metadata: Dict[str, Any] = {
            "gold_calls": gold_calls,
            "category": category,
            "raw_id": raw_id,
            "num_turns": len(turns),
            "turns": _normalize_turns(turns),
            "num_functions": len(functions),
        }
        if truth_by_id.get(raw_id) is None:
            metadata["missing_ground_truth"] = True
        # multi_turn_base has no per-sample function list; retrieval happens in
        # the scenario layer via these hints.
        for extra_key in ("involved_classes", "initial_config", "path", "excluded_function"):
            if extra_key in row and row[extra_key] is not None:
                metadata[extra_key] = row[extra_key]

        return Sample(
            sample_id=f"bfcl-{raw_id}",
            dataset="bfcl",
            scenario="general_agent",
            subtask=category,
            question=_flatten_turns(turns),
            answer=None,
            answer_type=AnswerType.TOOL_CALL,
            tools=tools,
            metadata=metadata,
        )


def _optional_int(value: Any, name: str) -> Optional[int]:
    if value is None:
        return None
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _write_download_errors(raw_dir: Path, errors: List[Dict[str, Any]]) -> None:
    path = Path(raw_dir) / ERRORS_FILE
    if not errors:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")


__all__ = [
    "BFCLAdapter",
    "CATEGORY_FILES",
    "CATEGORY_ORDER",
    "parse_ground_truth",
    "normalize_category",
]
