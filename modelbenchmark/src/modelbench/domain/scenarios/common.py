"""Shared implementation for non-agentic scenarios."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..models import ChatMessage, RawResponse, RequestSpec, Sample, Score, ToolSpec
from .base import ScenarioAdapter, TokenCounter
from .padding import fit_prompt

DEFAULT_SYSTEM_PROMPT = "You are a helpful, careful assistant. Follow the user's instructions exactly."


class SimpleScenario(ScenarioAdapter):
    """A single-turn scenario: render question, align context, score once.

    Samples loaded from the pre-constructed performance test set carry a
    verbatim request (``sample.metadata['prebuilt']`` + ``messages``/``tools``/
    ``tokens``): for those the scenario returns a :class:`RequestSpec` without
    rendering or tokenizing anything.  The runner owns the model-window check
    and skips (never silently truncates) a prebuilt request that does not fit.
    """

    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    def render_question(self, sample: Sample) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def context_text(self, sample: Sample) -> str:
        return sample.context or ""

    def tools_for(self, sample: Sample) -> Optional[List[ToolSpec]]:
        return sample.tools

    def evaluator(self, sample: Sample, response: RawResponse) -> Score:  # pragma: no cover
        raise NotImplementedError

    # ------------------------------------------------------------------ #

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
        prebuilt = self._prebuilt_request(model, sample, max_tokens, temperature, seed)
        if prebuilt is not None:
            return prebuilt
        question = self.render_question(sample)
        tools = self.tools_for(sample)
        system_prompt = self.params.get("system_prompt", self.system_prompt)
        fitted = fit_prompt(
            model=model,
            tokenize=tokenize,
            target_tokens=target_context_tokens,
            system_prompt=system_prompt,
            context_text=self.context_text(sample),
            question_block=question,
            tools=tools,
            max_tokens=max_tokens,
            seed=sample.sample_id,
        )
        return RequestSpec(
            model=model,
            messages=fitted.messages,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            tools=tools,
            target_context_tokens=target_context_tokens,
            actual_prompt_tokens=fitted.actual_tokens,
            truncated=fitted.truncated,
            padded=fitted.padded,
            context_meta={
                "iterations": fitted.iterations,
                "context_chars": fitted.context_chars,
                "truncation_ratio": fitted.truncation_ratio,
            },
        )

    @staticmethod
    def _prebuilt_request(
        model: str,
        sample: Sample,
        max_tokens: int,
        temperature: float,
        seed: Optional[int],
    ) -> Optional[RequestSpec]:
        metadata = sample.metadata or {}
        if not metadata.get("prebuilt"):
            return None
        raw_messages = metadata.get("messages") or []
        if not raw_messages:
            return None
        messages = [
            ChatMessage(
                **{
                    key: value
                    for key, value in message.items()
                    if key in ChatMessage.model_fields
                }
            )
            for message in raw_messages
        ]
        raw_tools = metadata.get("tools") or []
        tools = [
            ToolSpec(
                **{
                    key: value
                    for key, value in tool.items()
                    if key in ToolSpec.model_fields
                }
            )
            for tool in raw_tools
        ] or None
        declared = metadata.get("tokens")
        return RequestSpec(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            tools=tools,
            target_context_tokens=None,
            actual_prompt_tokens=int(declared) if declared is not None else None,
            truncated=False,
            padded=False,
            context_meta={
                "prebuilt": True,
                "source": "perf_jsonl",
                "bucket": metadata.get("bucket"),
                "context_length": metadata.get("context_length"),
                "declared_tokens": int(declared) if declared is not None else None,
            },
        )

    def score(self, sample: Sample, response: RawResponse) -> Score:
        try:
            return self.evaluator(sample, response)
        except Exception as exc:  # never let scoring break a run
            return Score(correct=False, score=0.0, detail={"error": f"scorer failed: {exc}"})


def render_mc_question(sample: Sample, instruction: str, extra_instruction: str = "") -> str:
    """Render a multiple-choice question with A-D labeled choices."""
    lines: List[str] = []
    if sample.question:
        lines.append(sample.question.strip())
    choices = sample.choices or []
    for idx, choice in enumerate(choices[:4]):
        lines.append(f"{chr(65 + idx)}. {choice}")
    lines.append("")
    lines.append(instruction)
    if extra_instruction:
        lines.append(extra_instruction)
    return "\n".join(lines)


def valid_tools(tools: Optional[List[ToolSpec]]) -> Optional[List[ToolSpec]]:
    return tools or None
