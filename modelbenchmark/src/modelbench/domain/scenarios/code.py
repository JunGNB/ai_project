"""Coding scenario (HumanEval, execution-based scoring)."""

from __future__ import annotations

from ..evaluators.code import score_code
from ..models import RawResponse, Sample, Score
from .common import SimpleScenario


class CodingScenario(SimpleScenario):
    name = "coding"
    dataset_name = "humaneval"
    system_prompt = "You are a careful Python programmer. Output runnable code only."

    def render_question(self, sample: Sample) -> str:
        problem = (sample.metadata or {}).get("prompt") or sample.question
        return (
            "请补全下面的 Python 函数。要求：\n"
            "1. 只输出完整、可直接运行的 Python 代码（包含函数定义及必要的 import）；\n"
            "2. 不要输出测试代码、示例调用或任何解释文字。\n\n"
            f"```python\n{problem.rstrip()}\n```"
        )

    def evaluator(self, sample: Sample, response: RawResponse) -> Score:
        return score_code(sample, response)
