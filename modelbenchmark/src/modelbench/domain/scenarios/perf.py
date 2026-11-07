"""Throughput probe scenario (performance only, no accuracy).

Uses a uniform synthetic prompt and asks for a long answer, so
``completion_tokens >= 16`` and decode metrics (``gen_tps``/``tpot_ms``) are
meaningful.  Performance mode never scores, so this scenario simply
contributes one more dataset to the per-sample results.
"""

from __future__ import annotations

from typing import Optional

from ..models import RawResponse, RequestSpec, Sample, Score
from .common import SimpleScenario


class ThroughputScenario(SimpleScenario):
    name = "throughput"
    dataset_name = "synthetic"
    scoring = False  # tell the runner not to attribute accuracy to this cell

    def render_question(self, sample: Sample) -> str:
        return sample.question or ""

    def build_request(self, model, sample, target_context_tokens, tokenize, *, max_tokens, temperature=0.0, seed=None) -> RequestSpec:
        # A probe needs a long enough decode window to measure tok/s.
        return super().build_request(
            model,
            sample,
            target_context_tokens,
            tokenize,
            max_tokens=max(128, max_tokens),
            temperature=temperature,
            seed=seed,
        )

    def score(self, sample: Sample, response: RawResponse) -> Optional[Score]:  # type: ignore[override]
        return None
