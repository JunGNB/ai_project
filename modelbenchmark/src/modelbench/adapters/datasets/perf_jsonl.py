"""Pre-constructed performance test set adapter (``dataset/test/performance``).

Reads ``testset.jsonl`` (one JSON object per line, 480 samples) that is built by
``tool/dataset`` from the upstream corpora.  Each row is already a fully
materialised chat request::

    {"id": "perf-knowledge_qa-1k-en-000",
     "scenario": "knowledge_qa", "bucket": "1k", "dataset": "mmlu",
     "tokens": 1108, "messages": [{"role": "user", "content": "..."}],
     "tools": [], "constructed": true,
     "meta": {"lang": "en", "subject": "...", "gold": "A"}}

The adapter filters rows by ``params.scenario`` (required), optional
``params.buckets`` (``1k``/``8k``/``32k``) and ``params.limit`` and normalises
each row into a :class:`~modelbench.domain.models.Sample` whose metadata carries
the pre-constructed request (``messages``/``tools``/``tokens``/``bucket``/
``context_length``/``prebuilt``).

Two hard requirements shape this module:

* parsing is streaming line-by-line, and only matching rows are retained
  (never a full-file dict);
* :meth:`plan` answers "how many samples per (scenario, dataset, bucket) cell"
  from a cached, tiny index of the file, so CLI/Web estimates never re-scan the
  25 MB JSONL on every keystroke.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...domain.context_windows import in_context_windows
from ...domain.models import AnswerType, Sample, ToolSpec
from ...domain.scenarios.base import largest_remainder_allocation
from .base import DatasetAdapter, read_jsonl, resolve_project_path

#: Bucket label -> declared context length in tokens.
BUCKET_TOKENS: Dict[str, int] = {"1k": 1024, "8k": 8192, "32k": 32768}
_TOKENS_TO_BUCKET = {tokens: bucket for bucket, tokens in BUCKET_TOKENS.items()}

#: Scenario -> evaluator answer type (scoring is not wired into run mode).
_ANSWER_TYPES: Dict[str, AnswerType] = {
    "knowledge_qa": AnswerType.MC_LETTER,
    "multilingual": AnswerType.MC_LETTER,
    "chinese_long_context": AnswerType.MC_LETTER,
    "instruction_following": AnswerType.CONSTRAINTS,
    "math_logic": AnswerType.NUMERIC,
    "coding": AnswerType.CODE,
    "general_agent": AnswerType.TOOL_CALL,
    "search_agent": AnswerType.SEARCH_ANSWER,
    "speculative_decoding": AnswerType.EXACT,
}

#: path -> (mtime_ns, size, [(id, scenario, dataset, bucket, tokens, lang), ...])
#: A lightweight index (480 rows) is enough to answer plan/estimate questions
#: without materialising the 25 MB JSONL into Sample objects.
_INDEX_CACHE: Dict[str, Tuple[int, int, List[Tuple[str, str, str, str, int, str]]]] = {}

#: Row keys copied into ``Sample.metadata`` unchanged when not already present.
_META_KEYS = (
    "subject",
    "task",
    "category",
    "question_id",
    "bfcl_id",
    "instruction_ids",
    "docs",
    "grade",
    "depth",
    "arms",
    "answer",
)


class PerfJSONLAdapter(DatasetAdapter):
    name = "perf_jsonl"
    scenario = ""
    source = "dataset/test/performance/testset.jsonl (built by tool/dataset)"
    license = "see dataset/test/performance/manifest.json provenance"

    # ------------------------------------------------------------------ #
    # paths / params
    # ------------------------------------------------------------------ #

    def _merged(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        merged = dict(self.params)
        if params:
            merged.update(params)
            nested = dict(self.params.get("dataset_params") or {})
            nested.update(params)
            # Callers may pass either the dataset_params mapping or an overlay.
            merged.setdefault("dataset_params", nested)
        return merged

    def _file(self, params: Optional[Dict[str, Any]] = None) -> Path:
        merged = self._merged(params)
        value = merged.get("path") or (merged.get("dataset_params") or {}).get("path")
        if value:
            return resolve_project_path(value)
        root = self.testset_dir
        if root is None:
            from modelbench.paths import PROJECT_ROOT

            root = PROJECT_ROOT / "dataset" / "test"
        return Path(root) / "performance" / "testset.jsonl"

    @staticmethod
    def _scenario_name(params: Dict[str, Any]) -> str:
        nested = params.get("dataset_params") or {}
        value = (
            params.get("scenario")
            or nested.get("scenario")
            or params.get("scenario_name")
            or nested.get("scenario_name")
        )
        if not value:
            raise ValueError(
                "perf_jsonl adapter requires params.dataset_params.scenario "
                "(one scenario slice per configured scenario)"
            )
        return str(value)

    @staticmethod
    def _bucket_filter(params: Dict[str, Any]) -> Optional[set]:
        nested = params.get("dataset_params") or {}
        raw = params.get("buckets") or nested.get("buckets")
        if raw is None:
            return None
        if isinstance(raw, (str, int)):
            raw = [raw]
        if not isinstance(raw, (list, tuple)):
            raise ValueError("perf_jsonl params.buckets must be a list of 1k/8k/32k")
        wanted: set = set()
        for item in raw:
            if isinstance(item, bool):
                raise ValueError("perf_jsonl params.buckets entries must be 1k/8k/32k or tokens")
            if isinstance(item, int):
                bucket = _TOKENS_TO_BUCKET.get(int(item))
                if bucket is None:
                    raise ValueError(
                        f"perf_jsonl params.buckets: unknown token count {item}; "
                        f"known: {sorted(BUCKET_TOKENS.values())}"
                    )
            else:
                bucket = str(item).strip().lower()
                if bucket not in BUCKET_TOKENS:
                    raise ValueError(
                        f"perf_jsonl params.buckets: unknown bucket '{item}'; "
                        f"known: {sorted(BUCKET_TOKENS)}"
                    )
            wanted.add(bucket)
        return wanted or None

    @staticmethod
    def _limit(params: Dict[str, Any]) -> Optional[int]:
        nested = params.get("dataset_params") or {}
        raw = params.get("limit", nested.get("limit"))
        if raw is None or raw == "":
            return None
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("perf_jsonl params.limit must be an integer") from exc
        if value < 1:
            raise ValueError("perf_jsonl params.limit must be >= 1")
        return value

    # ------------------------------------------------------------------ #
    # DatasetAdapter API
    # ------------------------------------------------------------------ #

    def ensure_downloaded(self) -> Path:
        path = self._file()
        if not path.is_file():
            raise FileNotFoundError(
                f"performance test set not found: {path}; "
                "rebuild it with tool/dataset (see dataset/README.md)"
            )
        return path

    def load(self) -> List[Sample]:
        path = self.ensure_downloaded()
        merged = self._merged()
        scenario = self._scenario_name(merged)
        buckets = self._bucket_filter(merged)
        limit = self._limit(merged)
        samples: List[Sample] = []
        for row in read_jsonl(path):
            if str(row.get("scenario")) != scenario:
                continue
            bucket = str(row.get("bucket") or "").strip().lower()
            if buckets is not None and bucket not in buckets:
                continue
            samples.append(self._to_sample(row, scenario, bucket))
            if limit is not None and len(samples) >= limit:
                break
        if not samples:
            scope = scenario + (f" buckets={sorted(buckets)}" if buckets else "")
            raise ValueError(
                f"perf_jsonl: no samples for {scope} in {path} "
                "(check dataset_params.scenario/buckets)"
            )
        return samples

    @staticmethod
    def _extra_rows(merged: Dict[str, Any]) -> Dict[str, Any]:
        """Read plan-only selectors from flat or nested ``dataset_params``."""
        nested = merged.get("dataset_params") or {}
        return {
            "context_windows": merged.get("context_windows")
            if merged.get("context_windows") is not None
            else nested.get("context_windows"),
            "samples": merged.get("samples")
            if merged.get("samples") is not None
            else nested.get("samples"),
            "sample_ids": merged.get("sample_ids")
            if merged.get("sample_ids") is not None
            else nested.get("sample_ids"),
        }

    def _selected_rows(
        self, path: Path, params: Optional[Dict[str, Any]] = None
    ) -> Optional[List[Tuple[str, str, str, str, int, str]]]:
        """Rows that ``load()+filter+selection`` would keep, using the index only.

        Mirrors the runner exactly: dataset limit -> context-window filter ->
        explicit sample ids -> stratified sample count (largest remainder on
        ``(context_length, lang)``).  ``None`` means the file is unavailable.
        """
        merged = self._merged(params)
        if not path.is_file():
            return None
        scenario = self._scenario_name(merged)
        buckets = self._bucket_filter(merged)
        limit = self._limit(merged)
        extra = self._extra_rows(merged)

        rows = [
            row
            for row in self._index(path)
            if row[1] == scenario
            and (buckets is None or row[3] in buckets)
        ]
        if limit is not None:
            rows = rows[: int(limit)]
        windows = extra["context_windows"] or []
        if windows:
            rows = [row for row in rows if in_context_windows(int(row[4]), windows)]
        sample_ids = extra["sample_ids"] or []
        if sample_ids:
            wanted = {str(item) for item in sample_ids}
            return [row for row in rows if row[0] in wanted]
        samples_n = extra["samples"]
        if samples_n is None or int(samples_n) <= 0 or int(samples_n) >= len(rows):
            return rows
        ordered = sorted(rows, key=lambda row: row[0])
        strata: Dict[Tuple[int, str], List[Tuple[str, str, str, str, int, str]]] = {}
        for row in ordered:
            strata.setdefault((BUCKET_TOKENS[row[3]], row[5]), []).append(row)
        allocation = largest_remainder_allocation(
            {key: len(group) for key, group in strata.items()}, int(samples_n)
        )
        chosen: List[Tuple[str, str, str, str, int, str]] = []
        for key in sorted(strata, key=lambda item: (str(item[0]), item[1])):
            chosen.extend(strata[key][: allocation.get(key, 0)])
        return sorted(chosen, key=lambda row: row[0])

    def strata(
        self, params: Optional[Dict[str, Any]] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Per ``(context_length, lang)`` sizes for the current selection."""
        path = self._file(params)
        rows = self._selected_rows(path, params)
        if rows is None:
            return None
        counts: Dict[Tuple[int, str], int] = {}
        for row in rows:
            key = (BUCKET_TOKENS[row[3]], row[5])
            counts[key] = counts.get(key, 0) + 1
        return [
            {"context_length": context_length, "lang": lang, "n": n}
            for (context_length, lang), n in sorted(
                counts.items(), key=lambda item: (str(item[0][0]), item[0][1])
            )
        ]

    def plan(
        self, params: Optional[Dict[str, Any]] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Aggregate-cell counts for an estimate, using the cached index only.

        Honours the same selectors as :meth:`_selected_rows` (dataset limit,
        context windows, explicit ids, stratified sample count), so CLI/Web
        estimates match the cells the runner will actually execute.  Each row
        also carries the ``(context_length, lang)`` stratum sizes that produced
        it.
        """
        merged = self._merged(params)
        path = self._file(params)
        rows = self._selected_rows(path, params)
        if rows is None:
            return None
        scenario = self._scenario_name(merged)
        cells: Dict[Tuple[str, int], int] = {}
        cell_strata: Dict[Tuple[str, int], Dict[Any, int]] = {}
        order: List[Tuple[str, int]] = []
        for _row_id, _scenario, dataset, bucket, _tokens, lang in rows:
            context_length = BUCKET_TOKENS[bucket]
            key = (dataset, context_length)
            if key not in cells:
                cells[key] = 0
                cell_strata[key] = {}
                order.append(key)
            cells[key] += 1
            strat_key = (context_length, lang)
            cell_strata[key][strat_key] = cell_strata[key].get(strat_key, 0) + 1
        return [
            {
                "scenario": scenario,
                "dataset": dataset,
                "context_length": context_length,
                "n": cells[(dataset, context_length)],
                "strata": [
                    {"context_length": ctx, "lang": lang, "n": n}
                    for (ctx, lang), n in sorted(
                        cell_strata[(dataset, context_length)].items(),
                        key=lambda item: (str(item[0][0]), item[0][1]),
                    )
                ],
            }
            for dataset, context_length in order
        ]

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #

    def _index(self, path: Path) -> List[Tuple[str, str, str, str, int, str]]:
        stat = path.stat()
        key = str(path)
        cached = _INDEX_CACHE.get(key)
        if cached is not None and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
            return cached[2]
        rows: List[Tuple[str, str, str, str, int, str]] = []
        for row in read_jsonl(path):
            bucket = str(row.get("bucket") or "").strip().lower()
            meta = row.get("meta") or {}
            rows.append(
                (
                    str(row.get("id") or ""),
                    str(row.get("scenario") or ""),
                    str(row.get("dataset") or ""),
                    bucket,
                    int(row.get("tokens") or 0),
                    str(meta.get("lang") or ""),
                )
            )
        _INDEX_CACHE.clear()
        _INDEX_CACHE[key] = (stat.st_mtime_ns, stat.st_size, rows)
        return rows

    def _to_sample(self, row: Dict[str, Any], scenario: str, bucket: str) -> Sample:
        sample_id = row.get("id")
        if not sample_id:
            raise ValueError(f"perf_jsonl: row missing 'id' (scenario={scenario})")
        if bucket not in BUCKET_TOKENS:
            raise ValueError(
                f"perf_jsonl: sample {sample_id} has unknown bucket '{bucket}'; "
                f"known: {sorted(BUCKET_TOKENS)}"
            )
        context_length = BUCKET_TOKENS[bucket]
        messages = row.get("messages") or []
        if not isinstance(messages, list):
            raise ValueError(f"perf_jsonl: sample {sample_id} messages must be a list")
        tools_raw = row.get("tools") or []
        if not isinstance(tools_raw, list):
            raise ValueError(f"perf_jsonl: sample {sample_id} tools must be a list")

        question = ""
        for message in messages:
            if str(message.get("role")) == "user":
                question = str(message.get("content") or "")
                break
        if not question:
            question = str(row.get("question") or "")

        row_meta = dict(row.get("meta") or {})
        gold = row_meta.get("gold", row_meta.get("answer", row.get("answer")))
        metadata: Dict[str, Any] = {
            "messages": messages,
            "tools": tools_raw,
            "tokens": int(row.get("tokens") or 0),
            "messages_tokens": (
                int(row["messages_tokens"]) if row.get("messages_tokens") is not None else None
            ),
            "tools_tokens": (
                int(row["tools_tokens"]) if row.get("tools_tokens") is not None else None
            ),
            "bucket": bucket,
            "context_length": context_length,
            "constructed": bool(row.get("constructed")),
            "lang": row_meta.get("lang"),
            "gold": gold,
            #: The scenario layer sends messages/tools verbatim instead of
            #: rendering a prompt for these samples.
            "prebuilt": bool(messages),
        }
        for key in _META_KEYS:
            if key in row_meta:
                metadata.setdefault(key, row_meta[key])

        tools = [
            ToolSpec(**{k: v for k, v in tool.items() if k in ToolSpec.model_fields})
            for tool in tools_raw
        ] or None
        content_chars = sum(
            len(str(message.get("content") or "")) for message in messages
        )
        return Sample(
            sample_id=str(sample_id),
            dataset=str(row.get("dataset") or ""),
            scenario=scenario,
            subtask=row_meta.get("task") or row_meta.get("category"),
            question=question,
            context=None,
            choices=None,
            answer=str(gold) if gold is not None else None,
            answer_type=_ANSWER_TYPES.get(scenario, AnswerType.EXACT),
            tools=tools,
            metadata=metadata,
            context_chars=content_chars or None,
            est_context_tokens=int(row.get("tokens") or 0) or None,
        )
