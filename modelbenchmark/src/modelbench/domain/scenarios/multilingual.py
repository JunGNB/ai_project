"""Multilingual knowledge QA scenario (MMMLU)."""

from __future__ import annotations

from ..evaluators.mc import score_mc
from ..models import RawResponse, Sample, Score
from .common import SimpleScenario, render_mc_question


class MultilingualScenario(SimpleScenario):
    name = "multilingual"
    dataset_name = "mmmlu"
    system_prompt = (
        "You are a multilingual expert. Answer the question in the language of "
        "the question, selecting exactly one option."
    )

    def render_question(self, sample: Sample) -> str:
        language = (sample.metadata or {}).get("language", "")
        instruction = (
            "请只回答正确选项的字母（A、B、C 或 D）。"
            if not language or language.upper().startswith("ZH")
            else "Answer with the single letter of the correct option (A/B/C/D) only."
        )
        return render_mc_question(sample, instruction)

    def evaluator(self, sample: Sample, response: RawResponse) -> Score:
        return score_mc(sample, response)


class ChineseLongContextScenario(MultilingualScenario):
    """Chinese-only long-document QA (LongBench v1 dureader) performance probe.

    The performance suite's former ``multilingual`` scenario only used the
    Chinese dureader corpus, so it is renamed to avoid claiming cross-lingual
    coverage.  The accuracy suite keeps its true 8-language ``multilingual``
    scenario.
    """

    name = "chinese_long_context"
    dataset_name = "longbench_v1:dureader"
    system_prompt = "请阅读以下中文材料并回答问题。"
