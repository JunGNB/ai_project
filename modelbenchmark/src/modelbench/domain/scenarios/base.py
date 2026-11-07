"""Scenario adapter contract.

A scenario turns one dataset :class:`Sample` into one concrete chat request
(prompt construction + context-length alignment) and scores the response with a
pure evaluator.  Scenarios are the only layer that knows both about datasets
and about evaluators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..models import ChatMessage, RawResponse, RequestSpec, Sample, Score

#: Counts the tokens of a rendered chat prompt exactly as the target model
#: would see it.  Supplied by the runner (oMLX ``/v1/tokenize``).
TokenCounter = Callable[[List[ChatMessage], Optional[List[Any]]], int]


class ScenarioAdapter(ABC):
    #: scenario id, e.g. ``"long_context"``
    name: str = ""
    #: dataset adapter name this scenario consumes
    dataset_name: str = ""
    #: agentic scenarios are driven by a tool loop in the runner
    agentic: bool = False
    #: False for performance-only probes (no accuracy is attributed)
    scoring: bool = True

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params: Dict[str, Any] = dict(params or {})

    # ---- required -------------------------------------------------------

    @abstractmethod
    def build_request(
        self,
        model: str,
        sample: Sample,
        target_context_tokens: Optional[int],
        tokenize: TokenCounter,
        *,
        max_tokens: int,
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ) -> RequestSpec:
        """Build the exact request for one (sample, context-length) cell."""

    @abstractmethod
    def score(self, sample: Sample, response: RawResponse) -> Score:
        """Score a response against the sample."""

    # ---- optional agent hooks (only used when ``agentic=True``) ---------

    def execute_tool(self, sample: Sample, call: Any) -> str:
        """Execute one tool call emitted by the model; return tool output text."""
        raise NotImplementedError

    def final_answer_text(self, sample: Sample, response: RawResponse) -> str:
        """Text that should be scored at the end of an agentic episode."""
        return response.text

    # ---- helpers --------------------------------------------------------

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dataset": self.dataset_name,
            "params": self.params,
        }


def deterministic_sample(samples: List[Sample], n: Optional[int]) -> List[Sample]:
    """Take a deterministic subset of samples (stable across runs).

    Samples are ordered by ``sample_id`` first, so dataset iteration order
    changes never alter which samples a run uses.
    """
    ordered = sorted(samples, key=lambda s: s.sample_id)
    if n is None or n >= len(ordered):
        return ordered
    return ordered[:n]


def largest_remainder_allocation(
    sizes: Dict[Any, int], n: int
) -> Dict[Any, int]:
    """Allocate ``n`` deterministically proportional to stratum ``sizes``.

    Floor allocation + largest remainder; ties break by ``str(key)`` so the
    result is identical across processes/runs.  Shared by
    :func:`stratified_sample` and dataset planning so the estimate always
    matches what the runner actually selects.
    """
    total = sum(int(value) for value in sizes.values())
    if n <= 0 or total <= 0:
        return {key: 0 for key in sizes}
    if n >= total:
        return {key: int(value) for key, value in sizes.items()}
    quotas = {key: n * int(value) / total for key, value in sizes.items()}
    allocation = {key: int(quota) for key, quota in quotas.items()}
    remaining = int(n) - sum(allocation.values())
    if remaining > 0:
        ranked = sorted(
            quotas,
            key=lambda key: (-(quotas[key] - allocation[key]), str(key)),
        )
        for key in ranked[:remaining]:
            allocation[key] += 1
    return allocation


def _strat_group_key(sample: Sample) -> Tuple[Optional[int], str]:
    metadata = sample.metadata or {}
    context_length = metadata.get("context_length")
    try:
        context_length = int(context_length) if context_length is not None else None
    except (TypeError, ValueError):
        context_length = None
    return context_length, str(metadata.get("lang") or "")


def stratified_sample(samples: List[Sample], n: Optional[int]) -> List[Sample]:
    """Take N samples proportional to (context_length, lang) strata.

    ``perf_jsonl`` sorts by ``sample_id`` where ``32k`` sorts before ``8k``, so
    a plain "first N" would always pick the largest bucket for scenarios that
    mix buckets.  Stratifying keeps a small ``--samples`` run representative and
    deterministic: floor allocation + largest-remainder distribution, then the
    first ``k`` samples (by ``sample_id``) inside each stratum.

    Legacy samples without a declared context bucket fall back to
    :func:`deterministic_sample`.
    """
    ordered = sorted(samples, key=lambda s: s.sample_id)
    if n is None or n >= len(ordered) or n <= 0:
        return ordered
    if not any((s.metadata or {}).get("prebuilt") for s in ordered):
        return ordered[:n]

    groups: Dict[Tuple[Optional[int], str], List[Sample]] = {}
    for sample in ordered:
        groups.setdefault(_strat_group_key(sample), []).append(sample)
    allocation = largest_remainder_allocation(
        {key: len(group) for key, group in groups.items()}, n
    )

    chosen: List[Sample] = []
    for key in sorted(groups, key=lambda k: (str(k[0]), k[1])):
        take = allocation.get(key, 0)
        if take <= 0:
            continue
        chosen.extend(groups[key][:take])
    return sorted(chosen, key=lambda s: s.sample_id)
