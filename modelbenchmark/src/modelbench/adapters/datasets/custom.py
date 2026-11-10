"""Generic JSONL dataset adapter: bring your own samples.

This is the "任意测试样本" entry point.  Point a scenario at it with
``dataset: custom`` and ``params.dataset_params.path``; the file is a JSONL of
objects that are normalised into :class:`modelbench.domain.models.Sample`.

Recognised keys (all optional except the question):

=====================  ============================================
``sample_id`` / ``_id`` / ``id``   unique id (default: row index)
``question`` / ``prompt``          the question text
``context`` / ``passage``          long context (optional)
``choices`` (list)  or ``choice_A``..``choice_D`` (str)
``answer`` / ``gold`` / ``label``  gold answer (letter, number, code, ...)
``answer_type``                    mc | numeric | exact | code | constraints | tool | search
``tools`` (list)                   tool specs for agent scenarios
``metadata`` (object)              extra fields kept in the record
``scenario``                       scenario hint (informational)
=====================  ============================================

Example::

    python -m modelbench.interfaces.cli run --models minicpm5-1b --samples 5 \\
        --scenarios knowledge_qa \\
        --models minicpm5-1b
    # with src/modelbench/resources/config/scenarios.yaml overriding knowledge_qa:
    #   dataset: custom
    #   params: {dataset_params: {path: dataset/custom/my_questions.jsonl}}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...domain.models import AnswerType, Sample, ToolSpec
from .base import DatasetAdapter, read_jsonl, resolve_project_path

_SCENARIO_BY_ANSWER_TYPE = {
    "mc": AnswerType.MC_LETTER,
    "numeric": AnswerType.NUMERIC,
    "exact": AnswerType.EXACT,
    "code": AnswerType.CODE,
    "constraints": AnswerType.CONSTRAINTS,
    "tool": AnswerType.TOOL_CALL,
    "search": AnswerType.SEARCH_ANSWER,
}


class CustomJSONLAdapter(DatasetAdapter):
    name = "custom"
    scenario = "knowledge_qa"
    source = "user-provided JSONL"
    license = "user responsibility"

    def ensure_downloaded(self) -> Path:
        path = self._file()
        if not path.is_file():
            raise FileNotFoundError(
                f"custom dataset not found: {path} (set params.dataset_params.path)"
            )
        return self.raw_dir

    def _file(self) -> Path:
        value = self.params.get("path") or self.params.get("file")
        if not value:
            raise ValueError("custom adapter requires params.dataset_params.path")
        return resolve_project_path(value)

    def load(self) -> List[Sample]:
        path = self._file()
        if not path.is_file():
            raise FileNotFoundError(f"custom dataset not found: {path}")
        samples: List[Sample] = []
        for index, row in enumerate(read_jsonl(path)):
            samples.append(self._to_sample(row, index))
        if not samples:
            raise ValueError(f"custom dataset is empty: {path}")
        return samples

    # ------------------------------------------------------------------ #

    def _to_sample(self, row: Dict[str, Any], index: int) -> Sample:
        sample_id = str(
            row.get("sample_id") or row.get("_id") or row.get("id") or f"custom-{index:05d}"
        )
        question = row.get("question") or row.get("prompt") or ""
        if not question:
            raise ValueError(f"custom sample {sample_id} has no 'question'/'prompt'")
        context = row.get("context") or row.get("passage") or ""

        choices: List[str] = []
        if isinstance(row.get("choices"), list):
            choices = [str(c) for c in row["choices"]]
        else:
            for letter in "ABCDEFGH":
                value = row.get(f"choice_{letter}")
                if value is not None:
                    choices.append(str(value))
        if not choices:
            for key in ("choice_A", "choice_a", "A"):
                if key in row and isinstance(row[key], list):
                    choices = [str(c) for c in row[key]]
                    break

        answer = row.get("answer", row.get("gold", row.get("label", row.get("target", ""))))
        raw_type = str(row.get("answer_type") or self.params.get("answer_type") or "exact").lower()
        answer_type = _SCENARIO_BY_ANSWER_TYPE.get(raw_type, AnswerType.EXACT)
        if choices and raw_type in ("exact", "auto"):
            answer_type = AnswerType.MC_LETTER
        if isinstance(answer, bool):
            answer = "true" if answer else "false"

        tools = None
        if isinstance(row.get("tools"), list) and row["tools"]:
            tools = [ToolSpec(**tool) for tool in row["tools"]]

        metadata = dict(row.get("metadata") or {})
        for key in ("scenario", "domain", "category", "difficulty", "language"):
            if key in row and key not in metadata:
                metadata[key] = row[key]

        return Sample(
            sample_id=sample_id,
            dataset=self.name,
            scenario=str(row.get("scenario") or self.params.get("scenario_name") or self.scenario),
            subtask=row.get("subtask"),
            question=str(question),
            context=str(context),
            choices=choices or None,
            answer=str(answer),
            answer_type=answer_type,
            tools=tools,
            metadata=metadata,
        )
