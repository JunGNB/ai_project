#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reference tokenizer for sample metadata (``input_tokens``).

Both the performance and accuracy sides count content tokens with
**cl100k_base** -- the generic OpenAI tokenizer that is the industry de-facto
reference for token counting -- rather than any model-specific tokenizer, so
declared counts stay credible and model-agnostic (user decision 2026-11-10:
no side binds to a model tokenizer).  ``dataset/raw/tokenizer/cl100k_base/
tokenizer.json`` is derived deterministically from the official tiktoken blob
by ``tool/dataset/performance/derive_cl100k.py`` and fetched/verified by both
``tool/dataset/{performance,accuracy}/fetch.py`` (each side's ``raw_spec.json``
pins the same bytes/SHA-256 for the shared on-disk asset).

``ACCURACY_TOKENIZER_NAME`` / ``accuracy_tokenizer()`` keep the accuracy side's
import surface; both point at the same cl100k asset, so the shared
``ReferenceTokenizer`` class serves both without coupling their conventions.

``input_tokens`` counts the *prompt content* tokens only; chat-template
control tokens are excluded.  cl100k counts no BOS/special tokens, so the
declared value is exactly the content-token count.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

from .fs import RAW_DIR

TOKENIZER_NAME = "cl100k_base"
TOKENIZER_PATH = RAW_DIR / "tokenizer" / TOKENIZER_NAME / "tokenizer.json"
FETCH_HINT = "run: python tool/dataset/performance/fetch.py --only tokenizer"

#: The accuracy test set's reference tokenizer: the same generic cl100k asset
#: (shared path, each side's raw_spec pins it independently).
ACCURACY_TOKENIZER_NAME = TOKENIZER_NAME
ACCURACY_TOKENIZER_PATH = TOKENIZER_PATH
ACCURACY_FETCH_HINT = "run: python tool/dataset/accuracy/fetch.py --only tokenizer"


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
