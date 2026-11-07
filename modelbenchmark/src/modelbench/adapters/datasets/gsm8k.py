"""GSM8K (grade-school math) dataset adapter.

Source: ``openai/gsm8k`` on HuggingFace, ``main`` configuration, ``test`` split
(1,319 rows).  ``answer`` holds the full chain-of-thought solution terminated by
``#### <number>``.  The Sample keeps the full solution (for evidence / debugging)
and exposes the final number as ``metadata["gold_number"]`` (commas removed, so
``1,450,000`` becomes ``1450000``) for the numeric evaluator.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...domain.models import AnswerType, Sample
from ._download import download_file
from .base import DatasetAdapter

_URL = (
    "https://hf-mirror.com/datasets/openai/gsm8k/resolve/main/"
    "main/test-00000-of-00001.parquet"
)
_RELATIVE_PATH = Path("main") / "test-00000-of-00001.parquet"

_MARKER = "####"
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_gold_number(answer: str) -> str:
    """Extract the canonical numeric answer from a GSM8K solution.

    Raises :class:`ValueError` when the ``####`` marker is missing or the tail
    is not a plain number, so corrupted data fails loudly instead of producing
    silently unscorable samples.
    """

    if _MARKER not in answer:
        raise ValueError(f"GSM8K answer has no '{_MARKER}' marker: {answer!r}")
    tail = answer.rsplit(_MARKER, 1)[1].strip()
    cleaned = tail.replace(",", "").replace("$", "").strip()
    if not _NUMBER_RE.fullmatch(cleaned):
        raise ValueError(f"cannot parse GSM8K numeric answer from {tail!r}")
    return cleaned


def _read_parquet_rows(path: Path) -> List[Dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment issue
        raise RuntimeError(
            "pyarrow is required to read parquet datasets; install it with "
            "`uv pip install --python .venv/bin/python pyarrow`"
        ) from exc
    return pq.read_table(path).to_pylist()


class GSM8KAdapter(DatasetAdapter):
    """GSM8K math word problems with final-number grading."""

    name = "gsm8k"
    scenario = "math_logic"
    source = "https://huggingface.co/datasets/openai/gsm8k (config=main, split=test)"
    license = "MIT"

    # ---- download -------------------------------------------------------

    def raw_file(self) -> Path:
        return self.raw_dir / _RELATIVE_PATH

    def ensure_downloaded(self) -> Path:
        download_file(
            _URL,
            self.raw_file(),
            raw_root=self.raw_dir,
            min_bytes=100_000,
            force=bool(self.params.get("force_download", False)),
        )
        return self.raw_dir

    # ---- load -----------------------------------------------------------

    def load(self) -> List[Sample]:
        path = self.raw_file()
        if not path.is_file():
            raise FileNotFoundError(
                f"GSM8K raw file not found: {path}; call ensure_downloaded() or "
                "`modelbenchmark download --dataset gsm8k` first"
            )
        rows = _read_parquet_rows(path)
        limit: Optional[int] = self.params.get("limit")
        if limit is not None:
            limit = int(limit)
        samples: List[Sample] = []
        for row_index, row in enumerate(rows):
            if limit is not None and len(samples) >= limit:
                break
            question = str(row.get("question", ""))
            solution = str(row.get("answer", ""))
            samples.append(
                Sample(
                    sample_id=f"gsm8k-{row_index:05d}",
                    dataset="gsm8k",
                    scenario="math_logic",
                    question=question,
                    answer=solution,
                    answer_type=AnswerType.NUMERIC,
                    metadata={
                        "gold_number": extract_gold_number(solution),
                        "row_index": row_index,
                        "split": "test",
                    },
                )
            )
        return samples


__all__ = ["GSM8KAdapter", "extract_gold_number"]
