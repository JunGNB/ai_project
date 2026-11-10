#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reference tokenizer for sample metadata (``input_tokens``).

The benchmark pins the MiniCPM5-1B tokenizer as its reference tokenizer (same
convention as ``tool/common/tokenizer.py``).  The file is fetched by
``tool/dataset/accuracy/fetch.py`` into ``dataset/raw/tokenizer/MiniCPM5-1B/`` and
only ``tokenizer.json`` is needed here (no chat template).

``input_tokens`` counts the *prompt content* tokens only; chat-template control
tokens (~10-30 per prompt) are excluded.  This is documented in the manifest's
``tokens_semantics`` field and is stable across hosts.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

from .fs import RAW_DIR

TOKENIZER_NAME = "MiniCPM5-1B"
TOKENIZER_PATH = RAW_DIR / "tokenizer" / TOKENIZER_NAME / "tokenizer.json"
FETCH_HINT = "run: python tool/dataset/accuracy/fetch.py --only tokenizer"


class ReferenceTokenizer:
    """Lazy wrapper around ``tokenizers.Tokenizer`` (imported on first use)."""

    def __init__(self, path: Optional[Union[str, Path]] = None):
        self.path = Path(path) if path else TOKENIZER_PATH
        self._tok = None

    def _load(self):
        if self._tok is not None:
            return self._tok
        if not self.path.is_file():
            raise FileNotFoundError(
                f"reference tokenizer not found: {self.path}\n{FETCH_HINT}"
            )
        try:
            from tokenizers import Tokenizer  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError(
                "the 'tokenizers' package is required to build sample metadata; "
                "run ./modelbench setup (or pip install -e '.[dev,datasets]')"
            ) from exc
        self._tok = Tokenizer.from_file(str(self.path))
        return self._tok

    def encode(self, text: str) -> List[int]:
        return self._load().encode(text).ids

    def count(self, text: str) -> int:
        return len(self.encode(text))


_DEFAULT: Optional[ReferenceTokenizer] = None


def default_tokenizer() -> ReferenceTokenizer:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = ReferenceTokenizer()
    return _DEFAULT
