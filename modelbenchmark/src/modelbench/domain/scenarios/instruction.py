"""Instruction-following scenario (IFEval verifiable constraints)."""

from __future__ import annotations

from ..evaluators.instruction import score_instructions
from ..models import RawResponse, Sample, Score
from .common import SimpleScenario


class InstructionFollowingScenario(SimpleScenario):
    name = "instruction_following"
    dataset_name = "ifeval"
    system_prompt = "You are an assistant that follows every instruction exactly."

    def render_question(self, sample: Sample) -> str:
        return (
            f"{sample.question.strip()}\n\n"
            "请严格遵守上述所有指令，直接给出最终回答。"
        )

    def evaluator(self, sample: Sample, response: RawResponse) -> Score:
        return score_instructions(sample, response)
