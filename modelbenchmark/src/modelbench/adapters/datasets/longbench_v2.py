"""LongBench-v2 dataset adapter (required deliverable).

Source: https://huggingface.co/datasets/zai-org/LongBench-v2 (formerly THUDM)
Raw file: ``data/raw/longbench_v2/data.json`` (single JSON array, 503 items,
SHA-256 ``15d61c2...`` recorded in docs/DATASETS.md).

Format (official):
    {_id, domain, sub_domain, difficulty, length, question,
     choice_A..D, answer (A-D), context}

The context is very long (median ~417K chars ≈ 100K+ tokens), far beyond the
fixed 1K..64K context targets, so consumers must truncate.  The adapter exposes an
optional deterministic truncation helper that keeps head+tail text.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from ...domain.models import AnswerType, Sample
from .base import DatasetAdapter, sha256_file

#: Upstream identity of the exact file this adapter understands.
LONGBENCH_V2_URL = (
    "https://hf-mirror.com/datasets/zai-org/LongBench-v2/resolve/main/data.json"
)
LONGBENCH_V2_SHA256 = "15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2"
LONGBENCH_V2_BYTES = 465_490_535


class LongBenchV2Adapter(DatasetAdapter):
    name = "longbench_v2"
    scenario = "long_context"
    source = "zai-org/LongBench-v2 (THUDM LongBench v2, arXiv:2412.15204)"
    license = "See upstream repository (research use; cite LongBench v2 paper)"

    @property
    def _file(self) -> Path:
        return Path(self.raw_dir) / "data.json"

    def ensure_downloaded(self) -> Path:
        """Verify the local copy; downloading is handled by `modelbenchmark download --dataset longbench_v2`."""
        path = self._file
        if not path.is_file():
            raise FileNotFoundError(
                f"LongBench-v2 data not found at {path}. "
                "Run: modelbenchmark download --dataset longbench_v2"
            )
        if path.stat().st_size != LONGBENCH_V2_BYTES:
            raise ValueError(
                f"LongBench-v2 file has unexpected size ({path.stat().st_size} bytes, "
                f"expected {LONGBENCH_V2_BYTES}). Re-run the downloader with resume support."
            )
        return path.parent

    def load(self) -> List[Sample]:
        """Load samples without materializing the whole 465MB JSON array.

        Naively calling ``json.loads(read_text())`` peaks at ~3.9GB RSS for this
        dataset, which can push oMLX over its dynamic memory ceiling during a
        run.  We stream the array twice instead: first to select the deterministic
        sample-id subset, then to build only those samples.
        """
        self.ensure_downloaded()
        limit = self.params.get("limit")
        domains = self.params.get("domains")
        difficulties = self.params.get("difficulties")

        def passes(row: Dict[str, Any]) -> bool:
            if domains and row.get("domain") not in domains:
                return False
            if difficulties and row.get("difficulty") not in difficulties:
                return False
            return True

        selected_ids = [
            str(row["_id"])
            for row in _iter_json_array(self._file)
            if passes(row)
        ]
        selected_ids.sort()
        if limit is not None:
            selected_ids = selected_ids[: int(limit)]
        wanted = set(selected_ids)
        if not wanted:
            return []

        samples: List[Sample] = []
        for row in _iter_json_array(self._file):
            sample_id = str(row["_id"])
            if sample_id not in wanted or not passes(row):
                continue
            samples.append(_row_to_sample(row))
        samples.sort(key=lambda sample: sample.sample_id)
        return samples


def _iter_json_array(path: Path, chunk_size: int = 1 << 20) -> Iterator[Dict[str, Any]]:
    """Stream objects from a top-level JSON array using bounded memory."""
    decoder = json.JSONDecoder()
    with Path(path).open("r", encoding="utf-8") as fh:
        buffer = ""
        eof = False

        def fill() -> bool:
            nonlocal buffer, eof
            if eof:
                return False
            chunk = fh.read(chunk_size)
            if not chunk:
                eof = True
                return False
            buffer += chunk
            return True

        while "[" not in buffer:
            if not fill():
                raise ValueError(f"{path}: not a JSON array")
        buffer = buffer[buffer.index("[") + 1 :]

        while True:
            buffer = buffer.lstrip()
            while not buffer:
                if not fill():
                    raise ValueError(f"{path}: unterminated JSON array")
                buffer = buffer.lstrip()
            if buffer.startswith("]"):
                return
            while True:
                try:
                    obj, end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    if not fill():
                        raise ValueError(f"{path}: truncated JSON array")
            if not isinstance(obj, dict):
                raise ValueError(f"{path}: array items must be objects")
            yield obj
            buffer = buffer[end:].lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:]
            elif buffer.startswith("]"):
                return
            elif not buffer:
                if not fill():
                    return
            else:
                raise ValueError(f"{path}: malformed JSON array")


def _row_to_sample(row: Dict[str, Any]) -> Sample:
    context = row.get("context") or ""
    choices = [row.get(f"choice_{c}") for c in "ABCD"]
    return Sample(
        sample_id=str(row["_id"]),
        dataset=LongBenchV2Adapter.name,
        scenario=LongBenchV2Adapter.scenario,
        subtask=row.get("sub_domain"),
        context=context,
        question=row["question"],
        choices=choices if all(isinstance(c, str) for c in choices) else None,
        answer=str(row["answer"]).strip().upper(),
        answer_type=AnswerType.MC_LETTER,
        metadata={
            "domain": row.get("domain"),
            "sub_domain": row.get("sub_domain"),
            "difficulty": row.get("difficulty"),
            "length_category": row.get("length"),
        },
        context_chars=len(context),
        est_context_tokens=len(context) // 4,
    )


def truncate_context(text: str, max_chars: int, head_ratio: float = 0.7) -> str:
    """Keep head+tail of a long context (problem usually follows the context)."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = int(max_chars * head_ratio)
    tail = max_chars - head
    if head <= 0:
        return text[-tail:] if tail else ""
    if tail <= 0:
        return text[:head]
    return text[:head] + "\n\n[...内容已截断...]\n\n" + text[-tail:]
