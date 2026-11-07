"""Knowledge QA scenario (MMLU / MMLU-style multiple choice)."""

from __future__ import annotations

from ..evaluators.mc import score_mc
from ..models import RawResponse, Sample, Score
from .common import SimpleScenario, render_mc_question


class KnowledgeQAScenario(SimpleScenario):
    name = "knowledge_qa"
    dataset_name = "mmlu"
    system_prompt = (
        "You are a knowledgeable expert. Answer multiple-choice questions "
        "accurately and concisely."
    )

    def render_question(self, sample: Sample) -> str:
        return render_mc_question(
            sample,
            "请只回答正确选项的字母（A、B、C 或 D），不要输出任何解释。",
            'Output format: a single letter (A/B/C/D).',
        )

    def evaluator(self, sample: Sample, response: RawResponse) -> Score:
        return score_mc(sample, response)
