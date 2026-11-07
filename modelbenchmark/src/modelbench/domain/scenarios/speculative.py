"""Speculative-decoding workload (single-arm, default model configuration).

The 60 pre-built samples come from Spec-Bench (`dataset/test/performance`,
`scenario=speculative_decoding`, `meta.arms=2`).  This project measures them
**once per sample with the model's default configuration**: no MTP/DFlash/
SpecPrefill is enabled and no speculative alpha/speedup is computed.  The
request is sent verbatim (prebuilt passthrough), so rendering only exists for
legacy/non-prebuilt samples.
"""

from __future__ import annotations

from ..models import RawResponse, Sample, Score
from .common import SimpleScenario


class SpeculativeDecodingScenario(SimpleScenario):
    name = "speculative_decoding"
    dataset_name = "perf_jsonl"
    scoring = False

    def render_question(self, sample: Sample) -> str:
        return sample.question or ""

    def evaluator(self, sample: Sample, response: RawResponse) -> Score:
        return Score(correct=False, score=0.0, detail={"note": "performance-only"})
