"""Bucket / actual-input-size tolerance policy (performance benchmark).

The performance test set declares cl100k-calibrated budgets against a generic
**reference tokenizer** (cl100k_base) so the 1K/8K/32K buckets are comparable
in the dataset while the declared counts stay model-agnostic.  At run
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

A performance test set may opt into **strict input-token alignment**
(``strict_input_tokens: true``): the dataset guarantees that the target model's
tokenizer counts exactly the bucket target at run time, so the runner requires
``actual == target``.  The +/-10% helpers above stay untouched for every other
use (historical sets, reporting, documentation); strict mode uses
:func:`exact_bounds`, i.e. ``low == high == target`` (again no 32K special
case: the target stays 32,768).
"""

from __future__ import annotations

from typing import Optional, Tuple

#: Nominal bucket targets in content tokens (the dataset's declared value).
BUCKET_TARGETS = {"1k": 1024, "8k": 8192, "32k": 32768}
TOLERANCE_RATIO = 0.10

__all__ = [
    "BUCKET_TARGETS",
    "TOLERANCE_RATIO",
    "exact_bounds",
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


def exact_bounds(target: int) -> Tuple[int, int]:
    """Return the strict ``(target, target)`` bounds (no +/-10% band).

    Used by ``perf.strict_input_tokens`` test sets: the sample is only measured
    when the target tokenizer's count is exactly the bucket target, so the
    tolerance band degenerates to a single point.  ``target`` is never special
    cased: 32K stays 32,768.
    """
    target = int(target)
    return target, target


def tolerance_for_bucket(
    bucket: Optional[str],
    declared_tokens: Optional[int] = None,
    strict: bool = False,
) -> Optional[Tuple[int, int]]:
    """Resolve tolerance bounds from a bucket name (or a declared token count).

    ``strict=True`` returns :func:`exact_bounds` for the same resolved target
    (the strict input-token contract); the historical +/-10% behaviour is
    unchanged when ``strict`` is false.
    """
    key = str(bucket or "").strip().lower()
    target: Optional[int] = None
    if key in BUCKET_TARGETS:
        target = BUCKET_TARGETS[key]
    elif declared_tokens is not None and int(declared_tokens) > 0:
        target = int(declared_tokens)
    if target is None:
        return None
    return exact_bounds(target) if strict else tolerance_bounds(target)


def within_tolerance(actual: int, bounds: Optional[Tuple[int, int]]) -> bool:
    if bounds is None:
        return True
    low, high = bounds
    return low <= int(actual) <= high
