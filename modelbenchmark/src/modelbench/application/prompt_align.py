"""Exact prompt-token alignment at run time.

The committed performance test set is built *oversized* (~2x the bucket content)
with a neutral reference tokenizer.  The prompt a model finally receives is
decided by that model's own tokenizer, so before sending a strict sample the
runner normalises the prompt to the bucket target exactly:

1. measure the rendered prompt with the target model tokenizer
   (``POST /v1/tokenize``, i.e. chat template + tools included);
2. cut the sample body at token level using the model's local tokenizer file
   (``tokenizer.json``); the decode of the kept ids is stripped of trailing
   partial bytes;
3. re-measure and correct by the observed delta (1-3 rounds in practice);
4. pad the residual (0-3 tokens) with newlines, re-measuring after each step.

If no local tokenizer file can be resolved, the body is trimmed by character
level binary search against the server count instead (slower, same result).
Prompts that cannot be aligned raise :class:`PromptAlignError` -- a run must
never silently send a prompt whose token count differs from the bucket target,
and must never silently skip the sample for that reason either.

Only the sample body (the longest user message) is trimmed/padded; other
messages (system prompt, decode guard) are kept verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

from modelbench.domain.models import ChatMessage

__all__ = [
    "AlignInfo",
    "PromptAlignError",
    "align_messages",
    "resolve_local_tokenizer",
]

CountFn = Callable[[List[ChatMessage], Any], int]

#: hard stop for the id-level correction loop
_MAX_ID_ROUNDS = 8
#: hard stop for the text-level binary search (each step halves the range)
_MAX_SEARCH_STEPS = 24
#: hard stop for the residual padding loop
_MAX_PAD_ROUNDS = 48
#: filler candidates used to pad the last few tokens.  Whitespace-only fillers
#: can be swallowed by a chat template (a trailing newline sometimes adds no
#: token at all), so printable ASCII fallbacks are included as well.
_PAD_UNITS = (
    "\n",
    "\n ",
    " ",
    "\n\n",
    " x",
    " 0",
    " .",
    " -",
    " z",
    "\n x",
    "\n\n x",
    " 0 0",
    # single printable characters: the last resort when one token is missing and
    # every whitespace-only candidate is swallowed by the template
    "x",
    "0",
    ".",
    "-",
    "z",
)


class PromptAlignError(RuntimeError):
    """The prompt could not be normalised to the exact bucket target."""


@dataclass
class AlignInfo:
    """What the run-time normalisation did to one prompt."""

    target: int
    original_tokens: int
    final_tokens: int
    mode: str  # exact | token-trim | token-tile | text-trim | padded
    tokenizer_path: Optional[str] = None
    server_calls: int = 0
    body_tokens_before: int = 0
    body_tokens_after: int = 0
    trim_chars: int = 0
    pad_tokens: int = 0
    iterations: int = 0

    def as_dict(self) -> dict:
        return {
            "target": self.target,
            "original_tokens": self.original_tokens,
            "final_tokens": self.final_tokens,
            "mode": self.mode,
            "tokenizer": self.tokenizer_path,
            "server_calls": self.server_calls,
            "body_tokens_before": self.body_tokens_before,
            "body_tokens_after": self.body_tokens_after,
            "trim_chars": self.trim_chars,
            "pad_tokens": self.pad_tokens,
            "iterations": self.iterations,
        }


# --------------------------------------------------------------------------- #
# local tokenizer resolution
# --------------------------------------------------------------------------- #
def _candidate_names(model_id: str) -> List[str]:
    names = [model_id.strip()]
    # e.g. "qwen3.6-35B-A3B-oQ4" -> also try the alias-ish lower-case form
    names.append(model_id.strip().lower())
    return [n for n in names if n]


def resolve_local_tokenizer(
    model_id: str, roots: Optional[Sequence[Path]] = None
) -> Optional[Path]:
    """Find the model's ``tokenizer.json`` under the local oMLX model store.

    The oMLX store keeps models either directly under ``models/<id>`` or under
    ``models/<org>/<id>``; both layouts are covered by a bounded depth-3 walk.
    Returns ``None`` when no file matches -- the caller then falls back to
    server-side counting only.
    """
    if not model_id:
        return None
    if roots is None:
        home = Path.home()
        roots = (home / ".omlx" / "models",)
    wanted = {n.lower() for n in _candidate_names(model_id)}
    best: Optional[Path] = None
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        try:
            dirs = [p for p in root.iterdir() if p.is_dir()]
        except OSError:
            continue
        for path in dirs:
            candidates = [path]
            try:
                candidates.extend(p for p in path.iterdir() if p.is_dir())
            except OSError:
                continue
            for candidate in candidates:
                if candidate.name.lower() not in wanted:
                    continue
                tokenizer = candidate / "tokenizer.json"
                if tokenizer.is_file():
                    # prefer the shallowest match (directly under models/)
                    if best is None or len(candidate.parts) < len(best.parts):
                        best = tokenizer
        if best is not None:
            break
    return best


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _strip_partial(text: str) -> str:
    """Drop the U+FFFD the tokenizer emits when a byte-level token is cut."""
    return text.rstrip("\ufffd")


def _tile(ids: Sequence[int], need: int) -> List[int]:
    """Repeat ``ids`` (keeping the order) until at least ``need`` tokens exist."""
    if not ids:
        return []
    out: List[int] = []
    while len(out) < need:
        out.extend(ids)
    return out


def _body_index(messages: Sequence[ChatMessage]) -> int:
    best = -1
    best_len = -1
    for idx, message in enumerate(messages):
        if message.role != "user" or not message.content:
            continue
        if len(message.content) > best_len:
            best, best_len = idx, len(message.content)
    return best


def _copy_messages(messages: Sequence[ChatMessage]) -> List[ChatMessage]:
    return [message.model_copy(deep=True) for message in messages]


def _set_body(messages: List[ChatMessage], index: int, body: str) -> None:
    message = messages[index]
    messages[index] = message.model_copy(update={"content": body})


def align_messages(
    messages: Sequence[ChatMessage],
    target: int,
    count_fn: CountFn,
    tools: Any = None,
    tokenizer_path: Optional[Path] = None,
) -> "tuple[List[ChatMessage], AlignInfo]":
    """Return messages whose target-tokenizer prompt length equals ``target``.

    ``count_fn(messages, tools)`` must return the server's token count for the
    rendered request (chat template included).  Raises :class:`PromptAlignError`
    when the prompt cannot be brought to ``target`` exactly.
    """
    if target <= 0:
        raise PromptAlignError(f"invalid target {target}")
    msgs = _copy_messages(messages)
    total_calls = 0

    def count() -> int:
        nonlocal total_calls
        total_calls += 1
        return int(count_fn(msgs, tools))

    index = _body_index(msgs)
    if index < 0:
        raise PromptAlignError("no user message to align")

    original = count()
    info = AlignInfo(
        target=int(target),
        original_tokens=original,
        final_tokens=original,
        mode="exact",
        tokenizer_path=str(tokenizer_path) if tokenizer_path else None,
    )
    if original == int(target):
        info.server_calls = total_calls
        return msgs, info

    body = msgs[index].content or ""
    body_tokens_before = 0
    pad = ""

    tokenizer = None
    if tokenizer_path is not None:
        try:
            from tokenizers import Tokenizer  # local import: optional dependency

            tokenizer = Tokenizer.from_file(str(tokenizer_path))
        except Exception:  # noqa: BLE001 - fall back to text-level trimming
            tokenizer = None

    if tokenizer is not None:
        ids = tokenizer.encode(body, add_special_tokens=False).ids
        body_tokens_before = len(ids)
        need = len(ids) if original < int(target) else max(1, len(ids) - (original - int(target)))
        mode = "token-trim" if original > int(target) else "token-tile"
        final = original
        for iteration in range(1, _MAX_ID_ROUNDS + 1):
            material = _tile(ids, need) if len(ids) < need else list(ids)
            candidate = _strip_partial(tokenizer.decode(material[:need]))
            _set_body(msgs, index, candidate + pad)
            final = count()
            if final == int(target):
                info.iterations = iteration
                break
            delta = int(target) - final
            need = max(1, need + delta) if need + delta > 0 else 1
            if need > len(material):
                need = len(material) + delta
            # safety: never oscillate past the hard limits
            if iteration == _MAX_ID_ROUNDS:
                break
        info.mode = mode
        info.body_tokens_after = need
        info.trim_chars = max(0, len(body) - len(msgs[index].content or body))
    else:
        # Text-level fallback: binary search the longest character prefix whose
        # server count still fits, then pad the residual.
        def fits(prefix: str) -> bool:
            _set_body(msgs, index, prefix)
            return count() <= int(target)

        text = body
        if count() < int(target):
            # tile the body until the prompt covers the target, then trim back
            while count() < int(target) and len(text) < (len(body) + 1) * 64:
                text = text + "\n\n" + body
            info.mode = "token-tile"
        else:
            info.mode = "text-trim"
        lo, hi = 0, len(text)
        best = 0
        for _ in range(_MAX_SEARCH_STEPS):
            if lo >= hi:
                break
            mid = (lo + hi) // 2
            if fits(text[:mid]):
                best = mid
                lo = mid + 1
            else:
                hi = mid
        _set_body(msgs, index, _strip_partial(text[:best]))
        final = count()
        info.trim_chars = max(0, len(body) - len(text[:best]))

    # Residual padding: the id-level trim/tile above lands on the target plus or
    # minus a few tokens because BPE boundaries do not line up with characters.
    # Strip trailing whitespace first (appending a printable character after a
    # stripped newline can un-mask the whitespace and add several tokens at
    # once), then adjust by single-character edits: characters are appended,
    # prepended or inserted near the tail, and an edit is kept only when the
    # server count moves toward the target without overshooting.  Single edits
    # can move the count by two tokens, so more than one edit site is needed.
    content = (msgs[index].content or "").rstrip(" \n\t")
    _set_body(msgs, index, content)
    edit_chars = "x0.-z 0. a1\n\t"
    for _ in range(_MAX_PAD_ROUNDS):
        final = count()
        if final == int(target):
            break
        if final > int(target):
            content = content[:-1]
            _set_body(msgs, index, content)
            continue
        progressed = False
        tail = content[-1:]
        for ch in edit_chars:
            candidates = [content + ch, ch + content]
            if len(content) > 1:
                candidates.append(content[:-1] + ch + tail)
                candidates.append(content[: len(content) // 2] + ch + content[len(content) // 2 :])
            for candidate in candidates:
                _set_body(msgs, index, candidate)
                got = count()
                if got > int(target):
                    continue
                if got > final:
                    content = candidate
                    info.pad_tokens += got - final
                    progressed = True
                    break
            if progressed:
                break
        if not progressed:
            _set_body(msgs, index, content)
            break
    final = count()
    info.final_tokens = final
    info.server_calls = total_calls
    if final != int(target):
        raise PromptAlignError(
            f"cannot align prompt to {target} tokens (reached {final} after "
            f"{total_calls} tokenize calls, mode={info.mode})"
        )
    if info.mode in ("token-trim", "text-trim") and info.pad_tokens == 0:
        info.mode = "trim"
    return msgs, info
