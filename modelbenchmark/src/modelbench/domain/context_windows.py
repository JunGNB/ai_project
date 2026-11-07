"""Context-window filtering for performance runs.

The user selects one or more context sizes (in tokens, "K" = 1024) on the Web
console or CLI.  A sample is measured only when its *native prompt token count*
falls inside one of the selected windows:

* ``1K`` is special: everything from 0 up to 1.5K tokens is accepted
  (short samples would otherwise be impossible to measure);
* every other window ``N`` accepts ``N - 0.5K .. N + 0.5K`` tokens.

No window is selected by default, so every bucket/sample is tested.  The filter
only selects samples; it never changes the cell key.  Cells are always
``(model, scenario, context_length, dataset)`` where ``context_length`` comes
from the sample metadata (1K/8K/32K buckets) and is ``None`` for legacy samples.

Token counts for pre-built performance samples use the sample's declared
``metadata.tokens`` (reference tokenizer, matches the dataset bucket definition);
legacy samples fall back to the model tokenizer (``/v1/tokenize``).
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

#: One "K" of context, in tokens.
K = 1024
#: Half-window tolerance for non-1K windows.
HALF_K = 512


def window_bounds(window: int) -> Tuple[int, int]:
    """Inclusive ``(low, high)`` prompt-token bounds for one selected window."""
    value = int(window)
    if value <= 0:
        raise ValueError("context window must be a positive token count")
    if value == K:
        # 1K is special: short prompts must be measurable too.
        return 0, K + HALF_K
    return max(0, value - HALF_K), value + HALF_K


def in_context_windows(tokens: int, windows: Sequence[int]) -> bool:
    """True when ``tokens`` falls inside any selected window (empty = all)."""
    if not windows:
        return True
    value = int(tokens)
    return any(low <= value <= high for low, high in (window_bounds(w) for w in windows))


def filter_in_context_windows(
    items: Iterable[Tuple[object, int]], windows: Sequence[int]
) -> Tuple[List[object], List[Tuple[object, int]]]:
    """Split ``(item, tokens)`` pairs into (kept, dropped).  Empty windows = keep all."""
    if not windows:
        kept = [item for item, _ in items]
        return kept, []
    kept: List[object] = []
    dropped: List[Tuple[object, int]] = []
    for item, tokens in items:
        if in_context_windows(int(tokens), windows):
            kept.append(item)
        else:
            dropped.append((item, int(tokens)))
    return kept, dropped


def _k(value: int) -> str:
    if value == 0:
        return "0"
    return f"{value / K:.1f}".rstrip("0").rstrip(".") + "K"


def describe_windows(windows: Sequence[int]) -> str:
    """Human-readable ranges, e.g. ``1K:0–1.5K, 8K:7.5–8.5K``."""
    if not windows:
        return "全部（不筛选）"
    parts: List[str] = []
    for window in windows:
        low, high = window_bounds(window)
        parts.append(f"{_k(window)}:{_k(low)}–{_k(high)}")
    return ", ".join(parts)


def normalize_windows(windows: Iterable[object]) -> List[int]:
    """Deduplicate/sort selected windows (tokens), rejecting invalid values."""
    out: List[int] = []
    for raw in windows or []:
        value = int(raw)
        if value < 1:
            raise ValueError("context_windows entries must be >= 1 token")
        if value not in out:
            out.append(value)
    return sorted(out)
