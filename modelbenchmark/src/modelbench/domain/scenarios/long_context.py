"""Long-context scenario backed by LongBench-v2."""

from __future__ import annotations

from ..evaluators.mc import score_mc
from ..models import RawResponse, Sample, Score
from .common import SimpleScenario, render_mc_question


class LongContextScenario(SimpleScenario):
    name = "long_context"
    dataset_name = "longbench_v2"
    system_prompt = (
        "你是一个严谨的长文档分析助手。请只依据给出的材料作答，"
        "不要依赖材料之外的知识。"
    )

    def render_question(self, sample: Sample) -> str:
        return render_mc_question(
            sample,
            "请仔细阅读以上材料，选出唯一正确的选项。只输出选项字母（A/B/C/D）。",
            'Output format: a single letter (A/B/C/D).',
        )

    def evaluator(self, sample: Sample, response: RawResponse) -> Score:
        return score_mc(sample, response)
