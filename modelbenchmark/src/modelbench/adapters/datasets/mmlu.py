"""MMLU (Massive Multitask Language Understanding) dataset adapter.

Source: ``cais/mmlu`` on HuggingFace, ``all`` configuration, ``test`` split
(14,042 rows, 57 subjects).  Each row has an integer ``answer`` index into the
four ``choices``; it is normalised to the letter ``A``-``D`` so evaluators only
need one multiple-choice representation.

The default load is a deterministic *subset* (12 subjects x 20 rows = 240
samples) rather than the full test set: the benchmark measures a handful of
models against a fixed context length, and 14k samples would make a run
take days.  Passing ``subjects: null`` (or ``[]``) means "all 57 subjects".
Override with e.g.::

    MMLUAdapter(raw, processed, {"subjects": ["college_physics"], "per_subject": 5})
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ...domain.models import AnswerType, Sample
from ._download import DownloadError, download_file
from .base import DatasetAdapter

_URL = (
    "https://hf-mirror.com/datasets/cais/mmlu/resolve/main/"
    "all/test-00000-of-00001.parquet"
)
_RELATIVE_PATH = Path("all") / "test-00000-of-00001.parquet"

#: 12 subjects spanning STEM / humanities / social sciences, used when the
#: caller does not pass ``subjects``.  Chosen once and frozen so that
#: ``sample_id`` sets stay comparable across runs.
DEFAULT_SUBJECTS: tuple[str, ...] = (
    "abstract_algebra",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_physics",
    "high_school_macroeconomics",
    "high_school_psychology",
    "human_aging",
    "machine_learning",
    "philosophy",
    "professional_law",
    "sociology",
)

DEFAULT_PER_SUBJECT = 20

_CHOICE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _read_parquet_rows(path: Path) -> List[Dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment issue
        raise RuntimeError(
            "pyarrow is required to read parquet datasets; install it with "
            "`uv pip install --python .venv/bin/python pyarrow`"
        ) from exc
    return pq.read_table(path).to_pylist()


class MMLUAdapter(DatasetAdapter):
    """MMLU multiple-choice knowledge questions (English)."""

    name = "mmlu"
    scenario = "knowledge_qa"
    source = "https://huggingface.co/datasets/cais/mmlu (config=all, split=test)"
    license = "MIT"

    # ---- download -------------------------------------------------------

    def raw_file(self) -> Path:
        return self.raw_dir / _RELATIVE_PATH

    def ensure_downloaded(self) -> Path:
        download_file(
            _URL,
            self.raw_file(),
            raw_root=self.raw_dir,
            min_bytes=1_000_000,
            force=bool(self.params.get("force_download", False)),
        )
        return self.raw_dir

    # ---- load -----------------------------------------------------------

    def _resolve_subjects(self, available: List[str]) -> List[str]:
        # ``subjects: null`` / ``[]`` explicitly means "every subject in the
        # file"; omitting the key entirely keeps the curated 12-subject default.
        if "subjects" in self.params and self.params["subjects"] is None:
            return sorted(set(available))
        wanted = self.params.get("subjects")
        if wanted is None:
            subjects = list(DEFAULT_SUBJECTS)
        elif isinstance(wanted, str):
            subjects = [part.strip() for part in wanted.split(",") if part.strip()]
        else:
            subjects = [str(item) for item in wanted]
        if not subjects:  # empty list means "every subject in the file"
            return sorted(set(available))
        missing = [subject for subject in subjects if subject not in set(available)]
        if missing:
            raise ValueError(
                f"MMLU subjects not present in dataset: {missing}; "
                f"available examples: {sorted(set(available))[:5]} …"
            )
        # de-duplicate but preserve caller order
        seen: set[str] = set()
        ordered = []
        for subject in subjects:
            if subject not in seen:
                seen.add(subject)
                ordered.append(subject)
        return ordered

    def load(self) -> List[Sample]:
        path = self.raw_file()
        if not path.is_file():
            raise FileNotFoundError(
                f"MMLU raw file not found: {path}; call ensure_downloaded() or "
                "`modelbenchmark download --dataset mmlu` first"
            )
        rows = _read_parquet_rows(path)
        subjects = self._resolve_subjects([str(row.get("subject", "")) for row in rows])

        per_subject = self.params.get("per_subject", DEFAULT_PER_SUBJECT)
        if per_subject is not None:
            per_subject = int(per_subject)
            if per_subject < 0:
                raise ValueError("per_subject must be >= 0")
        limit: Optional[int] = self.params.get("limit")
        if limit is not None:
            limit = int(limit)

        buckets: Dict[str, List[Sample]] = {subject: [] for subject in subjects}
        for row_index, row in enumerate(rows):
            subject = str(row.get("subject", ""))
            if subject not in buckets:
                continue
            if per_subject is not None and len(buckets[subject]) >= per_subject:
                continue
            buckets[subject].append(self._row_to_sample(row, row_index, subject))

        if limit is None:
            return [sample for subject in subjects for sample in buckets[subject]]

        # With an overall cap, round-robin across subjects so a small limit is
        # still representative instead of "first subjects only".
        selected: List[Sample] = []
        depth = max((len(bucket) for bucket in buckets.values()), default=0)
        for level in range(depth):
            for subject in subjects:
                bucket = buckets[subject]
                if level < len(bucket):
                    selected.append(bucket[level])
                    if len(selected) >= limit:
                        return selected
        return selected

    @staticmethod
    def _row_to_sample(row: Dict[str, Any], row_index: int, subject: str) -> Sample:
        choices = [str(choice) for choice in row.get("choices") or []]
        if len(choices) != 4:
            raise ValueError(
                f"MMLU row {row_index} ({subject}) has {len(choices)} choices, expected 4"
            )
        answer_index = row.get("answer")
        if not isinstance(answer_index, int) or not 0 <= answer_index < len(choices):
            raise ValueError(
                f"MMLU row {row_index} ({subject}) has invalid answer index: {answer_index!r}"
            )
        return Sample(
            sample_id=f"mmlu-{subject}-{row_index:05d}",
            dataset="mmlu",
            scenario="knowledge_qa",
            subtask=subject,
            question=str(row.get("question", "")),
            choices=choices,
            answer=_CHOICE_LETTERS[answer_index],
            answer_type=AnswerType.MC_LETTER,
            metadata={"subject": subject, "row_index": row_index, "split": "test"},
        )


__all__ = ["MMLUAdapter", "DEFAULT_SUBJECTS", "DEFAULT_PER_SUBJECT"]
