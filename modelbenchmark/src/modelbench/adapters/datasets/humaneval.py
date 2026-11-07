"""HumanEval (Python code generation) dataset adapter.

Source: ``openai/openai_humaneval`` on HuggingFace, ``test`` split (164 tasks).
The Sample's ``question`` is the official prompt (imports + signature +
docstring); the metadata carries everything the code evaluator needs to execute
unit tests in a sandbox: ``test`` (the ``check(candidate)`` harness),
``entry_point``, ``canonical_solution`` and the original ``task_id``.

``answer`` is intentionally ``None``: correctness is decided by executing the
hidden tests, not by matching a reference string.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...domain.models import AnswerType, Sample
from ._download import download_file
from .base import DatasetAdapter

_URL = (
    "https://hf-mirror.com/datasets/openai/openai_humaneval/resolve/main/"
    "openai_humaneval/test-00000-of-00001.parquet"
)
_RELATIVE_PATH = Path("openai_humaneval") / "test-00000-of-00001.parquet"


def _read_parquet_rows(path: Path) -> List[Dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment issue
        raise RuntimeError(
            "pyarrow is required to read parquet datasets; install it with "
            "`uv pip install --python .venv/bin/python pyarrow`"
        ) from exc
    return pq.read_table(path).to_pylist()


def _sample_id_for(task_id: str) -> str:
    """``HumanEval/0`` -> ``humaneval-000`` (zero padded for lexical ordering)."""

    match = re.search(r"(\d+)$", task_id or "")
    if match:
        return f"humaneval-{int(match.group(1)):03d}"
    return "humaneval-" + re.sub(r"[^A-Za-z0-9._-]+", "_", task_id or "unknown")


class HumanEvalAdapter(DatasetAdapter):
    """HumanEval function-level Python tasks (pass@1 via unit tests)."""

    name = "humaneval"
    scenario = "coding"
    source = "https://huggingface.co/datasets/openai/openai_humaneval (split=test)"
    license = "MIT"

    # ---- download -------------------------------------------------------

    def raw_file(self) -> Path:
        return self.raw_dir / _RELATIVE_PATH

    def ensure_downloaded(self) -> Path:
        download_file(
            _URL,
            self.raw_file(),
            raw_root=self.raw_dir,
            min_bytes=30_000,
            force=bool(self.params.get("force_download", False)),
        )
        return self.raw_dir

    # ---- load -----------------------------------------------------------

    def load(self) -> List[Sample]:
        path = self.raw_file()
        if not path.is_file():
            raise FileNotFoundError(
                f"HumanEval raw file not found: {path}; call ensure_downloaded() or "
                "`modelbenchmark download --dataset humaneval` first"
            )
        rows = _read_parquet_rows(path)

        task_filter = self.params.get("task_ids")
        wanted = {str(item) for item in task_filter} if task_filter else None
        limit: Optional[int] = self.params.get("limit")
        if limit is not None:
            limit = int(limit)

        samples: List[Sample] = []
        for row_index, row in enumerate(rows):
            if limit is not None and len(samples) >= limit:
                break
            task_id = str(row.get("task_id", ""))
            if wanted is not None and task_id not in wanted:
                continue
            prompt = str(row.get("prompt", ""))
            samples.append(
                Sample(
                    sample_id=_sample_id_for(task_id),
                    dataset="humaneval",
                    scenario="coding",
                    question=prompt,
                    answer=None,
                    answer_type=AnswerType.CODE,
                    metadata={
                        "task_id": task_id,
                        "prompt": prompt,
                        "test": str(row.get("test", "")),
                        "entry_point": str(row.get("entry_point", "")),
                        "canonical_solution": str(row.get("canonical_solution", "")),
                        "row_index": row_index,
                    },
                )
            )
        return samples


__all__ = ["HumanEvalAdapter"]
