"""General agent scenario: function calling (BFCL v4)."""

from __future__ import annotations

from ..evaluators.agent import score_tool_calls
from ..models import RawResponse, Sample, Score
from .common import SimpleScenario


class GeneralAgentScenario(SimpleScenario):
    name = "general_agent"
    dataset_name = "bfcl"
    system_prompt = "You are a function-calling assistant. Use the provided tools when appropriate."

    def render_question(self, sample: Sample) -> str:
        return (
            f"{sample.question.strip()}\n\n"
            "请在需要时调用合适的工具完成用户请求；如果无需调用任何工具，"
            "直接给出简短回答。"
        )

    def evaluator(self, sample: Sample, response: RawResponse) -> Score:
        return score_tool_calls(sample, response)
