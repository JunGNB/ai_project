"""Content-addressed store for the prompt payloads a run actually sent.

A benchmark sends the *same* prompt text to every model in a batch, and a
repeated sample re-sends it on every repeat.  Committing that text inside each
``records.jsonl`` line duplicates it once per model per repeat; measured on the
real ``output/`` tree the 240 unique prompts of one test set were stored ~32
times (~440 MB).  Instead every record carries ``request.prompt_sha256`` and the
bytes live here exactly once.

Payload shape (``put`` adds ``sha256``)::

    {"sha256": "<64 hex>", "chars": 12034,
     "messages": [{"role": "user", "content": "...", ...}],
     "tools": null | [...],
     "chat_template_kwargs": {...},
     "decode_guard_version": 1}

The digest is taken over the canonical JSON of everything except ``sha256``, so
the file is self-verifying and any change to a message, a tool definition or a
chat-template kwarg yields a different blob (no version bookkeeping needed).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from modelbench.adapters.storage import layout

#: Bumped when the decode-guard instruction text changes, so older runs stay
#: explainable even though the constant in code has moved on.
DECODE_GUARD_VERSION = 1

#: Keys covered by the digest, in fixed canonical order (``sort_keys`` handles
#: the encoding; this tuple documents the contract).
_DIGEST_KEYS = ("messages", "tools", "chat_template_kwargs", "decode_guard_version")


def _canonical_bytes(payload: Dict[str, Any]) -> bytes:
    body = {key: value for key, value in payload.items() if key != "sha256"}
    return json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def prompt_chars(messages: Any) -> int:
    """Character length of the concatenated message contents."""
    total = 0
    for message in messages or []:
        if isinstance(message, dict):
            total += len(message.get("content") or "")
        else:
            total += len(getattr(message, "content", None) or "")
    return total


class PromptBlobStore:
    """Read/write ``output/blobs/prompts`` by content hash."""

    def __init__(self, output_root: Path):
        self.output_root = Path(output_root)

    # -- encoding -------------------------------------------------------- #

    @staticmethod
    def canonical(payload: Dict[str, Any]) -> bytes:
        return _canonical_bytes(payload)

    @classmethod
    def digest(cls, payload: Dict[str, Any]) -> str:
        return hashlib.sha256(cls.canonical(payload)).hexdigest()

    @classmethod
    def build_payload(
        cls,
        messages: Any,
        tools: Any = None,
        chat_template_kwargs: Any = None,
        decode_guard_version: int = DECODE_GUARD_VERSION,
    ) -> Dict[str, Any]:
        """Normalize runtime pydantic/dict objects into a storable payload."""
        return {
            "chars": prompt_chars(messages),
            "messages": [_message_to_dict(message) for message in messages or []],
            "tools": _plain(tools),
            "chat_template_kwargs": _plain(chat_template_kwargs),
            "decode_guard_version": int(decode_guard_version),
        }

    # -- io -------------------------------------------------------------- #

    def path_for(self, sha: str) -> Path:
        return layout.blob_path(self.output_root, sha)

    def exists(self, sha: str) -> bool:
        return bool(sha) and self.path_for(sha).is_file()

    def put(self, payload: Dict[str, Any]) -> str:
        """Store ``payload`` and return its digest (idempotent, atomic)."""
        body = {key: value for key, value in payload.items() if key != "sha256"}
        sha = self.digest(body)
        path = self.path_for(sha)
        if path.is_file():
            return sha
        record = dict(body)
        record["sha256"] = sha
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{sha}.tmp-{os.getpid()}")
        tmp.write_text(
            json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, path)
        return sha

    def put_messages(
        self,
        messages: Any,
        tools: Any = None,
        chat_template_kwargs: Any = None,
        decode_guard_version: int = DECODE_GUARD_VERSION,
    ) -> str:
        return self.put(
            self.build_payload(
                messages,
                tools=tools,
                chat_template_kwargs=chat_template_kwargs,
                decode_guard_version=decode_guard_version,
            )
        )

    def get(self, sha: str) -> Optional[Dict[str, Any]]:
        if not self.exists(sha):
            return None
        try:
            return json.loads(self.path_for(sha).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def messages(self, sha: str) -> Optional[List[Dict[str, Any]]]:
        payload = self.get(sha)
        return None if payload is None else payload.get("messages")

    def iter_shas(self) -> Iterator[str]:
        root = layout.blobs_root(self.output_root)
        if not root.is_dir():
            return
        for path in sorted(root.glob("*/*.json")):
            yield path.stem

    def stats(self) -> Dict[str, int]:
        count = 0
        total = 0
        root = layout.blobs_root(self.output_root)
        if root.is_dir():
            for path in root.glob("*/*.json"):
                count += 1
                try:
                    total += path.stat().st_size
                except OSError:
                    pass
        return {"prompts": count, "bytes": total}

    def missing(self, shas: Any) -> List[str]:
        seen = set()
        absent: List[str] = []
        for sha in shas:
            if not sha or sha in seen:
                continue
            seen.add(sha)
            if not self.exists(sha):
                absent.append(sha)
        return absent


def _message_to_dict(message: Any) -> Dict[str, Any]:
    if isinstance(message, dict):
        data = dict(message)
    elif hasattr(message, "model_dump"):
        data = message.model_dump(mode="json")
    else:  # pragma: no cover - defensive
        data = {
            "role": getattr(message, "role", "user"),
            "content": getattr(message, "content", None),
        }
    for key in ("tool_call_id", "name", "tool_calls"):
        data.setdefault(key, None)
    return data


def _plain(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump(mode="json"))
    return value
