"""Prompt fitting helpers: optional context-length alignment and truncation.

Performance mode passes ``target_tokens=None`` (native sample context, no
padding).  When a sample would overflow the model window the runner rebuilds
it with an explicit target, which truncates the long context head+tail so the
question still sits at the end.

Exactness is achieved by asking the model's own tokenizer (oMLX
``/v1/tokenize``) through the ``tokenize`` callback, iterating a couple of
times on the filler length.  Typically 2-4 tokenizer calls per request build.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..models import ChatMessage

#: Neutral, domain-independent filler (Chinese + English).  It is deliberately
#: boring: it should occupy context without helping or hurting the task.
FILLER_CORPUS = (
    "背景资料：这是一段与当前问题无关的一般性说明文字，用于占位和压力测试。"
    "在长期实践中，人们发现保持记录整洁、按部就班地推进工作、定期复核结论，"
    "能够显著减少返工。团队协作时，清晰的分工与及时的沟通同样重要。"
    "The quick brown fox jumps over the lazy dog. "
    "Documentation should be concise, accurate and easy to search. "
    "Measurements are only useful when the environment is recorded alongside them. "
)

FILLER_PREFIX = "以下为补充的背景资料，与随后的问题无关，可忽略：\n"


@dataclass
class FittedPrompt:
    messages: List[ChatMessage]
    actual_tokens: int
    target_tokens: Optional[int]
    truncated: bool = False
    padded: bool = False
    context_chars: int = 0
    truncation_ratio: Optional[float] = None
    iterations: int = 0
    detail: Dict[str, Any] = field(default_factory=dict)


def filler_text(n_chars: int, seed: str = "") -> str:
    """Deterministic filler of approximately ``n_chars`` characters."""
    if n_chars <= 0:
        return ""
    # Rotate the corpus start based on the seed so different samples in one
    # cell do not share a byte-identical prefix (which would otherwise be a
    # free prefix-cache hit for the server).
    offset = int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8], 16) % len(FILLER_CORPUS)
    rotated = FILLER_CORPUS[offset:] + FILLER_CORPUS[:offset]
    repeats = n_chars // len(rotated) + 1
    return (rotated * repeats)[:n_chars]


def fit_prompt(
    model: str,
    tokenize: Any,
    *,
    target_tokens: Optional[int],
    system_prompt: Optional[str],
    context_text: str,
    question_block: str,
    tools: Optional[List[Any]] = None,
    max_tokens: int = 256,
    tolerance_ratio: float = 0.02,
    max_iterations: int = 6,
    seed: str = "",
    context_before_question: bool = True,
) -> FittedPrompt:
    """Build chat messages whose prompt token count is ~``target_tokens``.

    ``context_text`` is the task's own context (may be empty); if it is too long
    for the target it is truncated head+tail.  If too short, neutral filler is
    inserted before the question.
    """
    system_messages = [ChatMessage(role="system", content=system_prompt)] if system_prompt else []

    def build(messages: List[ChatMessage]) -> List[ChatMessage]:
        return system_messages + messages

    base_overhead = len(system_messages)

    if not target_tokens:
        messages = list(system_messages)
        if context_text:
            messages.append(ChatMessage(role="user", content=context_text))
        messages.append(ChatMessage(role="user", content=question_block))
        tokens = tokenize(messages, tools)
        return FittedPrompt(
            messages=messages,
            actual_tokens=tokens,
            target_tokens=None,
            context_chars=len(context_text),
        )

    # --- estimate chars-per-token from neutral filler (stable, sizable) ----
    probe = filler_text(4000, seed="probe")
    probe_tokens = max(tokenize([ChatMessage(role="user", content=probe)], None), 1)
    cpt = max(len(probe) / probe_tokens, 1.0)

    # --- fixed overhead: system prompt + question (no context/filler) ------
    overhead = tokenize(build([ChatMessage(role="user", content=question_block)]), tools)
    token_budget = max(target_tokens - overhead, 0)
    char_budget = max(int(token_budget * cpt), 0)

    truncated = False
    padded = False
    original_context_len = len(context_text)
    tolerance = max(16, int(target_tokens * tolerance_ratio))
    best: Optional[FittedPrompt] = None
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        if context_text and len(context_text) > char_budget:
            kept = _truncate_head_tail(context_text, char_budget)
            truncated = True
        else:
            kept = context_text
        remaining = max(char_budget - len(kept), 0)
        filler = (FILLER_PREFIX + filler_text(remaining, seed=seed + ":" + str(iterations))) if remaining > 0 else ""
        padded = padded or bool(filler)
        joined = (filler + kept)
        user_content = (joined + "\n\n" if joined else "") + question_block
        messages = build([ChatMessage(role="user", content=user_content)])
        tokens = tokenize(messages, tools)
        best = FittedPrompt(
            messages=messages,
            actual_tokens=tokens,
            target_tokens=target_tokens,
            truncated=truncated,
            padded=padded,
            context_chars=len(kept),
            truncation_ratio=(
                round(1 - len(kept) / original_context_len, 4)
                if truncated and original_context_len
                else None
            ),
            iterations=iterations,
        )
        diff = target_tokens - tokens
        if abs(diff) <= tolerance:
            break
        measured_cpt = len(user_content) / max(tokens, 1)
        # If the content is entirely filler, chars/token is the right scale;
        # clamp to avoid pathological single-iteration jumps.
        step = diff * min(max(measured_cpt, 1.0), 8.0)
        char_budget = max(0, int(char_budget + step))

    assert best is not None
    return best


def _truncate_head_tail(text: str, max_chars: int, head_ratio: float = 0.7) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    head = int(max_chars * head_ratio)
    tail = max_chars - head
    marker = "\n\n[...中间内容因长度限制已截断...]\n\n"
    if head <= 0:
        return text[-tail:]
    if tail <= 0:
        return text[:head]
    return text[:head] + marker + text[-tail:]
