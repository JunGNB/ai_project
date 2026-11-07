"""Math / logic reasoning scenario (GSM8K)."""

from __future__ import annotations

from ..evaluators.math import score_numeric
from ..models import RawResponse, Sample, Score
from .common import SimpleScenario


class MathLogicScenario(SimpleScenario):
    name = "math_logic"
    dataset_name = "gsm8k"
    system_prompt = "你是一个擅长逐步推理的数学助手。请给出清晰、可验证的推导。"

    def render_question(self, sample: Sample) -> str:
        return (
            f"{sample.question.strip()}\n\n"
            "请逐步推理，并在最后一行严格使用如下格式给出最终答案：\n"
            "#### <最终数值>"
        )

    def evaluator(self, sample: Sample, response: RawResponse) -> Score:
        return score_numeric(sample, response)
