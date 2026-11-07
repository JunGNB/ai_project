"""IFEval (instruction-following) dataset adapter.

Source: ``google/IFEval`` on HuggingFace, ``ifeval_input_data.jsonl`` (541
prompts).  Every prompt carries a list of verifiable instruction ids plus the
per-instruction kwargs the checker needs; the Sample keeps both verbatim in
``metadata`` and leaves ``answer`` empty (there is no reference answer — the
response is validated against the constraints).

Sampling is deliberately spread over *instruction types* rather than taking the
first N rows (which would over-represent a handful of frequent instruction ids):
one row per instruction id per round, deterministically, then re-sorted into
source order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ...domain.models import AnswerType, Sample
from ._download import download_file
from .base import DatasetAdapter, read_jsonl

_URL = "https://hf-mirror.com/datasets/google/IFEval/resolve/main/ifeval_input_data.jsonl"
_RELATIVE_PATH = Path("ifeval_input_data.jsonl")

DEFAULT_LIMIT = 30


class IFEvalAdapter(DatasetAdapter):
    """IFEval prompts with verifiable-instruction metadata."""

    name = "ifeval"
    scenario = "instruction_following"
    source = "https://huggingface.co/datasets/google/IFEval (ifeval_input_data.jsonl)"
    license = "Apache-2.0"

    # ---- download -------------------------------------------------------

    def raw_file(self) -> Path:
        return self.raw_dir / _RELATIVE_PATH

    def ensure_downloaded(self) -> Path:
        download_file(
            _URL,
            self.raw_file(),
            raw_root=self.raw_dir,
            min_bytes=50_000,
            force=bool(self.params.get("force_download", False)),
        )
        return self.raw_dir

    # ---- load -----------------------------------------------------------

    def load(self) -> List[Sample]:
        path = self.raw_file()
        if not path.is_file():
            raise FileNotFoundError(
                f"IFEval raw file not found: {path}; call ensure_downloaded() or "
                "`modelbenchmark download --dataset ifeval` first"
            )

        wanted = self.params.get("instructions")
        if isinstance(wanted, str):
            wanted_ids = {part.strip() for part in wanted.split(",") if part.strip()}
        elif wanted is None:
            wanted_ids = set()
        else:
            wanted_ids = {str(item) for item in wanted}

        records: List[Dict[str, Any]] = []
        for row_index, row in enumerate(read_jsonl(path)):
            instruction_ids = [str(item) for item in (row.get("instruction_id_list") or [])]
            if wanted_ids and not wanted_ids.intersection(instruction_ids):
                continue
            records.append(
                {
                    "row_index": row_index,
                    "row": row,
                    "instruction_ids": instruction_ids,
                }
            )

        limit = self.params.get("limit", DEFAULT_LIMIT)
        if limit is None:
            selected_positions = list(range(len(records)))
        else:
            limit = int(limit)
            if limit < 0:
                raise ValueError("limit must be >= 0")
            selected_positions = _sample_across_instruction_types(records, limit)

        return [self._record_to_sample(records[position]) for position in selected_positions]

    @staticmethod
    def _record_to_sample(record: Dict[str, Any]) -> Sample:
        row = record["row"]
        kwargs = row.get("kwargs") or []
        return Sample(
            sample_id=f"ifeval-{record['row_index']:04d}",
            dataset="ifeval",
            scenario="instruction_following",
            question=str(row.get("prompt", "")),
            answer=None,
            answer_type=AnswerType.CONSTRAINTS,
            metadata={
                "instruction_id_list": record["instruction_ids"],
                "kwargs": kwargs,
                "key": row.get("key"),
                "row_index": record["row_index"],
            },
        )


def _sample_across_instruction_types(records: List[Dict[str, Any]], limit: int) -> List[int]:
    """Pick ``limit`` records, round-robin over instruction ids.

    Returns selected positions in ``records`` (sorted by source order so the
    loader output is stable and easy to diff against the raw file).
    """

    if limit <= 0:
        return []
    if limit >= len(records):
        return list(range(len(records)))

    groups: Dict[str, List[int]] = {}
    for position, record in enumerate(records):
        for instruction_id in record["instruction_ids"]:
            groups.setdefault(instruction_id, []).append(position)

    ordered_ids = sorted(groups)
    cursor = {instruction_id: 0 for instruction_id in ordered_ids}
    selected: List[int] = []
    selected_set: set[int] = set()
    while len(selected) < limit:
        progressed = False
        for instruction_id in ordered_ids:
            positions = groups[instruction_id]
            index = cursor[instruction_id]
            while index < len(positions) and positions[index] in selected_set:
                index += 1
            cursor[instruction_id] = index
            if index >= len(positions):
                continue
            position = positions[index]
            cursor[instruction_id] = index + 1
            selected.append(position)
            selected_set.add(position)
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break
    selected.sort()
    return selected


__all__ = ["IFEvalAdapter", "DEFAULT_LIMIT"]
