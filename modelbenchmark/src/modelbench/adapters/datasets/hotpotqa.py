"""HotpotQA (multi-hop search QA) dataset adapter.

Source: ``hotpotqa/hotpot_qa`` on HuggingFace, ``distractor`` configuration,
``validation`` split (7,405 questions, 10 paragraphs each).

The scenario layer needs a *retrieval corpus* plus a question for the local
search agent, so the adapter flattens the ten context paragraphs into a
deterministic corpus::

    Title 1: paragraph text 1

    Title 2: paragraph text 2
    ...

The gold answer stays in ``answer``; ``metadata`` keeps the paragraph list
(``[title, text]``) and the supporting-fact sentence ids so evaluation can check
both answer correctness and retrieval quality.  Default ``limit`` is 30
questions (a run over 7k long prompts would take hours).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ...domain.models import AnswerType, Sample
from ._download import download_file
from .base import DatasetAdapter

_URL = (
    "https://hf-mirror.com/datasets/hotpotqa/hotpot_qa/resolve/main/"
    "distractor/validation-00000-of-00001.parquet"
)
_RELATIVE_PATH = Path("distractor") / "validation-00000-of-00001.parquet"

DEFAULT_LIMIT = 30
_PARAGRAPH_SEPARATOR = "\n\n"


def _read_parquet_rows(path: Path) -> List[Dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment issue
        raise RuntimeError(
            "pyarrow is required to read parquet datasets; install it with "
            "`uv pip install --python .venv/bin/python pyarrow`"
        ) from exc
    return pq.read_table(path).to_pylist()


def build_corpus(paragraphs: List[List[str]]) -> str:
    """Render ``[[title, text], ...]`` as a deterministic retrieval corpus."""

    return _PARAGRAPH_SEPARATOR.join(f"{title}: {text}" for title, text in paragraphs)


class HotpotQAAdapter(DatasetAdapter):
    """HotpotQA distractor questions with a concatenated retrieval corpus."""

    name = "hotpotqa"
    scenario = "search_agent"
    source = "https://huggingface.co/datasets/hotpotqa/hotpot_qa (distractor, validation)"
    license = "CC BY-SA 4.0"

    # ---- download -------------------------------------------------------

    def raw_file(self) -> Path:
        return self.raw_dir / _RELATIVE_PATH

    def ensure_downloaded(self) -> Path:
        download_file(
            _URL,
            self.raw_file(),
            raw_root=self.raw_dir,
            min_bytes=10_000_000,
            force=bool(self.params.get("force_download", False)),
        )
        return self.raw_dir

    # ---- load -----------------------------------------------------------

    def load(self) -> List[Sample]:
        path = self.raw_file()
        if not path.is_file():
            raise FileNotFoundError(
                f"HotpotQA raw file not found: {path}; call ensure_downloaded() or "
                "`modelbenchmark download --dataset hotpotqa` first"
            )
        rows = _read_parquet_rows(path)
        limit = self.params.get("limit", DEFAULT_LIMIT)
        if limit is not None:
            limit = int(limit)
            if limit < 0:
                raise ValueError("limit must be >= 0")

        samples: List[Sample] = []
        for row_index, row in enumerate(rows):
            if limit is not None and len(samples) >= limit:
                break
            samples.append(self._row_to_sample(row, row_index))
        return samples

    @staticmethod
    def _row_to_sample(row: Dict[str, Any], row_index: int) -> Sample:
        context = row.get("context") or {}
        titles = list(context.get("title") or [])
        sentence_groups = list(context.get("sentences") or [])
        if len(titles) != len(sentence_groups):
            raise ValueError(
                f"HotpotQA row {row_index} has {len(titles)} titles but "
                f"{len(sentence_groups)} sentence groups"
            )
        paragraphs: List[List[str]] = [
            [str(title), " ".join(str(sentence) for sentence in sentences)]
            for title, sentences in zip(titles, sentence_groups)
        ]

        facts = row.get("supporting_facts") or {}
        fact_titles = list(facts.get("title") or [])
        fact_sent_ids = list(facts.get("sent_id") or [])
        supporting_facts = [
            {"title": str(title), "sent_id": int(sent_id)}
            for title, sent_id in zip(fact_titles, fact_sent_ids)
        ]

        return Sample(
            sample_id=f"hotpotqa-{row.get('id', row_index)}",
            dataset="hotpotqa",
            scenario="search_agent",
            question=str(row.get("question", "")),
            context=build_corpus(paragraphs),
            answer=str(row.get("answer", "")),
            answer_type=AnswerType.SEARCH_ANSWER,
            metadata={
                "paragraphs": paragraphs,
                "supporting_facts": supporting_facts,
                "id": row.get("id"),
                "type": row.get("type"),
                "level": row.get("level"),
                "row_index": row_index,
                "split": "validation",
                "num_paragraphs": len(paragraphs),
            },
        )


__all__ = ["HotpotQAAdapter", "build_corpus", "DEFAULT_LIMIT"]
