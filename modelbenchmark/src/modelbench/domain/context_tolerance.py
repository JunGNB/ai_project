"""Bucket / actual-input-size tolerance policy (performance benchmark).

The performance test set is constructed against a **reference tokenizer**
(MiniCPM5-1B) so the 1K/8K/32K buckets are comparable in the dataset.  At run
time the target model's own tokenizer can count the same content differently,
and the chat template + decode guard add protocol tokens.  The product decision
(2026-11-10) is:

* every bucket uses the same single tolerance: ``target +/- 10%`` (symmetric);
* out-of-band samples are **still measured** (never skipped for tokenizer
  mismatch); ``records.jsonl`` records the actual count and the band flag;
* aggregates only use in-band samples (see ``domain.metrics``), and the report
  exposes both in-band and out-of-band counts;
* the model-window check uses the tokenized actual count, not the declared
  bucket value.  32K has no special target: 32,768 stays the bucket target.
"""

from __future__ import annotations

from typing import Optional, Tuple

#: Nominal bucket targets in content tokens (the dataset's declared value).
BUCKET_TARGETS = {"1k": 1024, "8k": 8192, "32k": 32768}
TOLERANCE_RATIO = 0.10

__all__ = [
    "BUCKET_TARGETS",
    "TOLERANCE_RATIO",
    "tolerance_bounds",
    "tolerance_for_bucket",
    "within_tolerance",
]


def tolerance_bounds(target: int) -> Tuple[int, int]:
    """Return ``(low, high)`` inclusive bounds for a nominal target.

    All buckets use the same symmetric ``target * 0.90 .. target * 1.10``
    bounds.  Bounds are integers and rounded outward at the edges.
    """
    target = int(target)
    low = int(target * (1.0 - TOLERANCE_RATIO))
    high = int(target * (1.0 + TOLERANCE_RATIO))
    return max(0, low), max(0, high)


def tolerance_for_bucket(
    bucket: Optional[str], declared_tokens: Optional[int] = None
) -> Optional[Tuple[int, int]]:
    """Resolve tolerance bounds from a bucket name (or a declared token count)."""
    key = str(bucket or "").strip().lower()
    if key in BUCKET_TARGETS:
        return tolerance_bounds(BUCKET_TARGETS[key])
    if declared_tokens is not None and int(declared_tokens) > 0:
        return tolerance_bounds(int(declared_tokens))
    return None


def within_tolerance(actual: int, bounds: Optional[Tuple[int, int]]) -> bool:
    if bounds is None:
        return True
    low, high = bounds
    return low <= int(actual) <= high
