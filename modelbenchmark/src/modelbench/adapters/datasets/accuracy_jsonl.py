"""Pre-constructed accuracy test set adapter (``dataset/test/accuracy``).

Reads ``testset.jsonl`` (one JSON object per line, 2,000 rows = 8 scenarios x
250) built by ``tool/dataset/accuracy/build.py``.  Each row is already a fully
materialised chat request plus its grading contract::

    {"id": "knowledge_qa:global_mmlu:abstract_algebra/test/4|en",
     "scenario": "knowledge_qa", "dataset": "global_mmlu",
     "language": "en", "upstream_id": "abstract_algebra/test/4|en",
     "input_tokens": 129,
     "messages": [{"role": "user", "content": "..."}],
     "judge": {"family": "mc_letter", "answer": "B", "max_new_tokens": 1536},
     "meta": {"sample_id": "...", "subject": "...", "stratum": "..."}}

The adapter filters rows by ``params.scenario`` (required).  Prompt construction
and scoring belong to the accuracy orchestrator / ``domain.judges``; this module
only normalises rows into :class:`~modelbench.domain.models.Sample` objects whose
``metadata`` carries the pre-built request and judge contract verbatim.

Two hard requirements shape this module:

* parsing is streaming line-by-line, and only matching rows are retained
  (never a full-file dict of 2,000 long-context prompts);
* :meth:`plan` answers "how many selected samples does this scenario imply" from
  a cached, tiny index of the file, so CLI/Web estimates do not re-scan the
  14 MB JSONL on every keystroke.  Plan honours the same selectors as the
  runner: explicit ``sample_ids`` then deterministic stratified ``samples``
  (largest remainder over ``meta.stratum``), matching
  :func:`stratified_select` exactly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ...domain.judges import JUDGE_FAMILIES
from ...domain.models import AnswerType, Sample
from ...domain.scenarios.base import largest_remainder_allocation
from .base import DatasetAdapter, read_jsonl, resolve_project_path

#: Judge family -> evaluator answer type (scoring lives in ``domain.judges``).
_ANSWER_TYPES: Dict[str, AnswerType] = {
    "mc_letter": AnswerType.MC_LETTER,
    "ifeval": AnswerType.CONSTRAINTS,
    "code_exec": AnswerType.CODE,
    "tool_call": AnswerType.TOOL_CALL,
}

#: path -> (mtime_ns, size, [(id, scenario, dataset, language, stratum, family, metric), ...])
#: A lightweight index (2,000 short rows) answers plan/estimate questions without
#: materialising the 14 MB JSONL (long-context prompts) into Sample objects.
_INDEX_CACHE: Dict[
    str, Tuple[int, int, List[Tuple[str, str, str, str, str, str, str]]]
] = {}


def _answer_type(judge: Dict[str, Any]) -> AnswerType:
    family = str(judge.get("family") or "")
    if family == "normalized_match":
        return (
            AnswerType.NUMERIC
            if judge.get("metric") == "numeric_exact"
            else AnswerType.EXACT
        )
    return _ANSWER_TYPES.get(family, AnswerType.EXACT)


def _stratum(sample: Sample) -> str:
    metadata = sample.metadata or {}
    dataset_meta = metadata.get("dataset_meta") or {}
    value = dataset_meta.get("stratum") or metadata.get("stratum") or ""
    return str(value)


def stratified_select(
    samples: Sequence[Sample],
    n: Optional[int] = None,
    sample_ids: Optional[Iterable[str]] = None,
) -> List[Sample]:
    """Deterministic accuracy sample selection (shared by runner and ``plan``).

    * explicit ``sample_ids`` win (invalid ids are simply absent);
    * otherwise ``n`` is allocated across ``meta.stratum`` with the same
      largest-remainder helper used by the performance side, keeping the first
      ``k`` samples by ``sample_id`` inside each stratum;
    * the result is always sorted by ``sample_id`` so run-to-run selection is
      byte-identical.
    """
    ordered = sorted(samples, key=lambda sample: sample.sample_id)
    if sample_ids:
        wanted = {str(item) for item in sample_ids}
        return [sample for sample in ordered if sample.sample_id in wanted]
    if n is None or int(n) <= 0 or int(n) >= len(ordered):
        return ordered
    by_stratum: Dict[str, List[Sample]] = {}
    for sample in ordered:
        by_stratum.setdefault(_stratum(sample), []).append(sample)
    allocation = largest_remainder_allocation(
        {stratum: len(group) for stratum, group in by_stratum.items()}, int(n)
    )
    chosen: List[Sample] = []
    for stratum in sorted(by_stratum):
        quota = int(allocation.get(stratum, 0))
        if quota > 0:
            chosen.extend(by_stratum[stratum][:quota])
    return sorted(chosen, key=lambda sample: sample.sample_id)


class AccuracyJSONLAdapter(DatasetAdapter):
    name = "accuracy_jsonl"
    scenario = ""
    source = "dataset/test/accuracy/testset.jsonl (built by tool/dataset/accuracy)"
    license = "see dataset/test/accuracy/manifest.json provenance"

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
        return Path(root) / "accuracy" / "testset.jsonl"

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
                "accuracy_jsonl adapter requires params.scenario "
                "(one scenario slice per configured scenario)"
            )
        return str(value)

    @staticmethod
    def _limit(params: Dict[str, Any]) -> Optional[int]:
        nested = params.get("dataset_params") or {}
        raw = params.get("limit", nested.get("limit"))
        if raw is None or raw == "":
            return None
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("accuracy_jsonl params.limit must be an integer") from exc
        if value < 1:
            raise ValueError("accuracy_jsonl params.limit must be >= 1")
        return value

    @staticmethod
    def _selection(params: Dict[str, Any]) -> Tuple[Optional[int], List[str]]:
        nested = params.get("dataset_params") or {}
        samples = params.get("samples", nested.get("samples"))
        ids = params.get("sample_ids", nested.get("sample_ids")) or []
        if isinstance(ids, (str, int)):
            ids = [ids]
        sample_ids = [str(item) for item in ids if str(item).strip()]
        return (int(samples) if samples is not None and str(samples) != "" else None), sample_ids

    # ------------------------------------------------------------------ #
    # DatasetAdapter API
    # ------------------------------------------------------------------ #

    def ensure_downloaded(self) -> Path:
        path = self._file()
        if not path.is_file():
            raise FileNotFoundError(
                f"accuracy test set not found: {path}; build it with "
                "`python tool/dataset/accuracy/build.py` after "
                "`python tool/dataset/accuracy/fetch.py` (see dataset/README.md)"
            )
        return path

    def load(self) -> List[Sample]:
        path = self.ensure_downloaded()
        merged = self._merged()
        scenario = self._scenario_name(merged)
        limit = self._limit(merged)
        samples: List[Sample] = []
        for row in read_jsonl(path):
            if str(row.get("scenario")) != scenario:
                continue
            samples.append(self._to_sample(row))
            if limit is not None and len(samples) >= limit:
                break
        if not samples:
            raise ValueError(
                f"accuracy_jsonl: no samples for scenario {scenario!r} in {path} "
                "(check params.scenario)"
            )
        return samples

    def select_samples(
        self,
        samples: Sequence[Sample],
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Sample]:
        """Apply the configured accuracy selectors to already-loaded samples."""
        merged = self._merged(params)
        count, sample_ids = self._selection(merged)
        return stratified_select(samples, count, sample_ids)

    # ------------------------------------------------------------------ #
    # planning (index only; no prompt materialisation)
    # ------------------------------------------------------------------ #

    def _index(
        self, path: Path
    ) -> List[Tuple[str, str, str, str, str, str, str]]:
        stat = path.stat()
        key = str(path)
        cached = _INDEX_CACHE.get(key)
        if cached is not None and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
            return cached[2]
        rows: List[Tuple[str, str, str, str, str, str, str]] = []
        for row in read_jsonl(path):
            meta = row.get("meta") or {}
            judge = row.get("judge") or {}
            rows.append(
                (
                    str(row.get("id") or ""),
                    str(row.get("scenario") or ""),
                    str(row.get("dataset") or ""),
                    str(row.get("language") or ""),
                    str(meta.get("stratum") or ""),
                    str(judge.get("family") or ""),
                    str(judge.get("metric") or "accuracy"),
                )
            )
        _INDEX_CACHE.clear()
        _INDEX_CACHE[key] = (stat.st_mtime_ns, stat.st_size, rows)
        return rows

    def _selected_index_rows(
        self, path: Path, params: Optional[Dict[str, Any]] = None
    ) -> Optional[List[Tuple[str, str, str, str, str, str, str]]]:
        merged = self._merged(params)
        if not path.is_file():
            return None
        scenario = self._scenario_name(merged)
        limit = self._limit(merged)
        count, sample_ids = self._selection(merged)
        rows = [row for row in self._index(path) if row[1] == scenario]
        if limit is not None:
            rows = rows[:limit]
        rows = sorted(rows, key=lambda row: row[0])
        if sample_ids:
            wanted = set(sample_ids)
            return [row for row in rows if row[0] in wanted]
        if count is None or int(count) <= 0 or int(count) >= len(rows):
            return rows
        by_stratum: Dict[str, List[Tuple[str, str, str, str, str, str, str]]] = {}
        for row in rows:
            by_stratum.setdefault(row[4], []).append(row)
        allocation = largest_remainder_allocation(
            {stratum: len(group) for stratum, group in by_stratum.items()}, int(count)
        )
        chosen: List[Tuple[str, str, str, str, str, str, str]] = []
        for stratum in sorted(by_stratum):
            quota = int(allocation.get(stratum, 0))
            if quota > 0:
                chosen.extend(by_stratum[stratum][:quota])
        return sorted(chosen, key=lambda row: row[0])

    def strata(self, params: Optional[Dict[str, Any]] = None) -> Optional[List[Dict[str, Any]]]:
        """Per-stratum counts for the current selection (``None`` if file absent)."""
        path = self._file(params)
        rows = self._selected_index_rows(path, params)
        if rows is None:
            return None
        counts: Dict[Tuple[str, str, str], int] = {}
        for _row_id, _scenario, dataset, language, stratum, _family, _metric in rows:
            key = (stratum, language, dataset)
            counts[key] = counts.get(key, 0) + 1
        return [
            {"stratum": stratum, "language": language, "dataset": dataset, "n": n}
            for (stratum, language, dataset), n in sorted(counts.items())
        ]

    def plan(
        self, params: Optional[Dict[str, Any]] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """One ``{"scenario", "dataset", "n"}`` row for the accuracy estimate.

        ``n`` is the number of samples the runner will execute for this scenario
        under the same ``samples``/``sample_ids`` selectors (stratified largest
        remainder).  ``None`` means the file is not available.
        """
        merged = self._merged(params)
        path = self._file(params)
        rows = self._selected_index_rows(path, params)
        if rows is None:
            return None
        return [
            {
                "scenario": self._scenario_name(merged),
                "dataset": self.name,
                "n": len(rows),
            }
        ]

    # ------------------------------------------------------------------ #
    # row -> Sample
    # ------------------------------------------------------------------ #

    def _to_sample(self, row: Dict[str, Any]) -> Sample:
        sample_id = row.get("id")
        if not sample_id:
            raise ValueError("accuracy_jsonl: row missing 'id'")
        scenario = str(row.get("scenario") or "")
        judge = row.get("judge") or {}
        family = str(judge.get("family") or "")
        if family not in JUDGE_FAMILIES:
            raise ValueError(
                f"accuracy_jsonl: sample {sample_id} has unknown judge family {family!r}; "
                f"known: {list(JUDGE_FAMILIES)}"
            )
        messages = row.get("messages") or []
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"accuracy_jsonl: sample {sample_id} messages must be a non-empty list")
        question = ""
        for message in messages:
            if str(message.get("role")) == "user":
                question = str(message.get("content") or "")
                break
        if not question:
            raise ValueError(f"accuracy_jsonl: sample {sample_id} has no user message")
        input_tokens = int(row.get("input_tokens") or 0)
        content_chars = sum(
            len(str(message.get("content") or "")) for message in messages
        )
        metadata: Dict[str, Any] = {
            # The accuracy orchestrator sends ``messages`` verbatim; the scenario
            # layer must not re-render anything for these samples.
            "prebuilt": True,
            "judge": dict(judge),
            "language": row.get("language"),
            "upstream_id": row.get("upstream_id"),
            "dataset_meta": dict(row.get("meta") or {}),
            "messages": messages,
            "input_tokens": input_tokens,
        }
        return Sample(
            sample_id=str(sample_id),
            dataset=str(row.get("dataset") or ""),
            scenario=scenario,
            subtask=None,
            context=None,
            question=question,
            choices=None,
            answer=None,
            answer_type=_answer_type(judge),
            tools=None,
            metadata=metadata,
            context_chars=content_chars or None,
            est_context_tokens=input_tokens or None,
        )


__all__ = ["AccuracyJSONLAdapter", "stratified_select"]
