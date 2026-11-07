"""MMMLU (multilingual MMLU) dataset adapter.

Source: ``openai/MMMLU`` on HuggingFace, ``test/mmlu_<LANG>.csv`` files.  The
project benchmarks six languages (ZH-CN, JA-JP, KO-KR, FR-FR, DE-DE, ES-LA) on
the same question structure as MMLU, so accuracy can be compared across
languages for the same task family.

The CSV files use the columns ``,Question,A,B,C,D,Answer,Subject`` (the first,
unnamed column is a row counter; ``Answer`` is already a letter).  The loader is
tolerant of column-name variations (BOM, surrounding whitespace, case) and still
accepts a numeric answer index.

Default load: 10 rows per language (60 samples).  Override with
``{"languages": ["ZH-CN", "JA-JP"], "per_language": 5}``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...domain.models import AnswerType, Sample
from ._download import DownloadError, download_file
from .base import DatasetAdapter

_URL_TEMPLATE = "https://hf-mirror.com/datasets/openai/MMMLU/resolve/main/test/mmlu_{lang}.csv"

#: Canonical language order (also the default load order).
LANGUAGES: tuple[str, ...] = ("ZH-CN", "JA-JP", "KO-KR", "FR-FR", "DE-DE", "ES-LA")

DEFAULT_PER_LANGUAGE = 10

_CHOICE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

#: Convenience aliases so callers can write ``["zh", "ja"]`` etc.
_LANGUAGE_ALIASES: Dict[str, str] = {
    "zh": "ZH-CN",
    "zh-cn": "ZH-CN",
    "zh_cn": "ZH-CN",
    "cn": "ZH-CN",
    "chinese": "ZH-CN",
    "ja": "JA-JP",
    "jp": "JA-JP",
    "ja-jp": "JA-JP",
    "japanese": "JA-JP",
    "ko": "KO-KR",
    "kr": "KO-KR",
    "ko-kr": "KO-KR",
    "korean": "KO-KR",
    "fr": "FR-FR",
    "fr-fr": "FR-FR",
    "french": "FR-FR",
    "de": "DE-DE",
    "de-de": "DE-DE",
    "german": "DE-DE",
    "es": "ES-LA",
    "es-la": "ES-LA",
    "es_latam": "ES-LA",
    "spanish": "ES-LA",
}


def normalize_language(value: str) -> str:
    """Map user-supplied language aliases onto canonical MMMLU codes."""

    text = str(value).strip()
    if text.upper() in LANGUAGES:
        return text.upper()
    alias = _LANGUAGE_ALIASES.get(text.lower())
    if alias is not None:
        return alias
    raise ValueError(
        f"unsupported MMMLU language {value!r}; use one of {list(LANGUAGES)} "
        "(aliases like 'zh', 'ja', 'fr' are accepted)"
    )


def _get_field(row: Dict[str, Any], *names: str) -> Optional[str]:
    """Case/whitespace-insensitive lookup in a CSV row dict."""

    normalized = {
        (key or "").strip().lstrip("\ufeff").lower(): value for key, value in row.items()
    }
    for name in names:
        value = normalized.get(name.lower())
        if value is not None:
            return value
    return None


class MMMLUAdapter(DatasetAdapter):
    """MMMLU multiple-choice questions in six languages."""

    name = "mmmlu"
    scenario = "multilingual"
    source = "https://huggingface.co/datasets/openai/MMMLU (test/mmlu_<LANG>.csv)"
    license = "MIT (see the openai/MMMLU dataset card)"

    # ---- download -------------------------------------------------------

    def raw_file(self, language: str) -> Path:
        return self.raw_dir / "test" / f"mmlu_{language}.csv"

    def languages(self) -> List[str]:
        wanted = self.params.get("languages")
        if wanted is None:
            return list(LANGUAGES)
        if isinstance(wanted, str):
            wanted = [part for part in wanted.split(",") if part.strip()]
        return [normalize_language(item) for item in wanted]

    def ensure_downloaded(self) -> Path:
        failures: List[str] = []
        for language in self.languages():
            try:
                download_file(
                    _URL_TEMPLATE.format(lang=language),
                    self.raw_file(language),
                    raw_root=self.raw_dir,
                    min_bytes=500_000,
                    force=bool(self.params.get("force_download", False)),
                )
            except DownloadError as exc:
                failures.append(f"{language}: {exc}")
        if failures:
            raise DownloadError(
                "MMMLU download failed for one or more languages:\n  " + "\n  ".join(failures)
            )
        return self.raw_dir

    # ---- load -----------------------------------------------------------

    def load(self) -> List[Sample]:
        per_language = self.params.get(
            "per_language", self.params.get("limit_per_language", DEFAULT_PER_LANGUAGE)
        )
        if per_language is not None:
            per_language = int(per_language)
            if per_language < 0:
                raise ValueError("per_language must be >= 0")

        samples: List[Sample] = []
        for language in self.languages():
            path = self.raw_file(language)
            if not path.is_file():
                raise FileNotFoundError(
                    f"MMMLU raw file not found: {path}; call ensure_downloaded() or "
                    "`modelbenchmark download --dataset mmmlu` first"
                )
            samples.extend(self._load_language(path, language, per_language))
        return samples

    @staticmethod
    def _load_language(path: Path, language: str, per_language: Optional[int]) -> List[Sample]:
        samples: List[Sample] = []
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            for row_index, row in enumerate(reader):
                if per_language is not None and len(samples) >= per_language:
                    break
                question = _get_field(row, "Question")
                if question is None:
                    raise ValueError(f"MMMLU file {path} has no 'Question' column")
                choices = [_get_field(row, letter) or "" for letter in "ABCD"]
                raw_answer = (_get_field(row, "Answer") or "").strip()
                answer = _normalize_answer(raw_answer, choices, path, row_index)
                samples.append(
                    Sample(
                        sample_id=f"mmmlu-{language}-{row_index:04d}",
                        dataset="mmmlu",
                        scenario="multilingual",
                        subtask=language,
                        question=question,
                        choices=choices,
                        answer=answer,
                        answer_type=AnswerType.MC_LETTER,
                        metadata={
                            "language": language,
                            "subject": (_get_field(row, "Subject") or None),
                            "row_index": row_index,
                            "split": "test",
                        },
                    )
                )
        return samples


def _normalize_answer(raw: str, choices: List[str], path: Path, row_index: int) -> str:
    if raw.upper() in _CHOICE_LETTERS[: len(choices)]:
        return raw.upper()
    # Some distributions store the integer choice index instead of a letter.
    try:
        index = int(raw)
    except (TypeError, ValueError):
        index = -1
    if 0 <= index < len(choices):
        return _CHOICE_LETTERS[index]
    raise ValueError(
        f"MMMLU file {path} row {row_index} has invalid Answer {raw!r} "
        f"(expected A-D or 0-{len(choices) - 1})"
    )


__all__ = ["MMMLUAdapter", "LANGUAGES", "DEFAULT_PER_LANGUAGE", "normalize_language"]
