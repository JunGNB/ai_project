"""Run-time prompt alignment: exact trim/tile/pad semantics."""

from __future__ import annotations

import json
import pathlib

import pytest

from modelbench.application import prompt_align
from modelbench.application.prompt_align import (
    PromptAlignError,
    align_messages,
    resolve_local_tokenizer,
)
from modelbench.domain.models import ChatMessage


TEMPLATE_OVERHEAD = 5


def count_chars(messages, tools=None):
    """Fake tokenizer: one token per character of the rendered messages."""
    return TEMPLATE_OVERHEAD + sum(len(m.content or "") for m in messages)


def user(text):
    return ChatMessage(role="user", content=text)


def test_exact_prompt_is_returned_unchanged():
    msgs = [user("x" * 100)]
    out, info = align_messages(msgs, 100 + TEMPLATE_OVERHEAD, count_chars)
    assert info.mode == "exact"
    assert info.original_tokens == info.final_tokens == 100 + TEMPLATE_OVERHEAD
    assert out[0].content == "x" * 100
    assert info.server_calls == 1


def test_long_body_is_trimmed_to_target():
    msgs = [user("y" * 500)]
    out, info = align_messages(msgs, 200 + TEMPLATE_OVERHEAD, count_chars)
    assert info.final_tokens == 200 + TEMPLATE_OVERHEAD
    assert info.mode == "trim"
    assert len(out[0].content) == 200
    assert info.trim_chars == 300
    assert out[0].content == "y" * 200


def test_short_body_is_tiled_then_trimmed():
    msgs = [user("abcd" * 10)]
    out, info = align_messages(msgs, 100 + TEMPLATE_OVERHEAD, count_chars)
    assert info.final_tokens == 100 + TEMPLATE_OVERHEAD
    assert info.mode == "token-tile"
    assert len(out[0].content) == 100
    assert out[0].content.startswith("abcd" * 10)


def test_only_longest_user_message_is_aligned():
    msgs = [
        ChatMessage(role="system", content="S" * 20),
        user("short"),
        user("z" * 400),
    ]
    # every message counts, so only the longest user message is cut
    out, info = align_messages(msgs, 150 + TEMPLATE_OVERHEAD, count_chars)
    assert info.final_tokens == 150 + TEMPLATE_OVERHEAD
    assert out[0].content == "S" * 20
    assert out[1].content == "short"
    assert len(out[2].content) == 150 - 20 - 5


def test_tokenizer_backed_trim_uses_real_ids():
    tokenizers = pytest.importorskip("tokenizers")
    tok_path = pathlib.Path("dataset/raw/tokenizer/cl100k_base/tokenizer.json")
    if not tok_path.is_file():
        pytest.skip("cl100k tokenizer.json not built")
    tok = tokenizers.Tokenizer.from_file(str(tok_path))

    def count(messages, tools=None):
        return 5 + sum(len(tok.encode(m.content or "", add_special_tokens=False).ids) for m in messages)

    body = " ".join(["alpha beta gamma delta epsilon"] * 200)
    out, info = align_messages([user(body)], 300, count, tokenizer_path=tok_path)
    assert info.final_tokens == 300
    assert info.mode in {"token-trim", "trim"}
    assert len(out[0].content) < len(body)
    assert count(out) == 300


def test_invalid_target_is_rejected():
    with pytest.raises(PromptAlignError):
        align_messages([user("abc")], 0, count_chars)


def test_messages_without_user_are_rejected():
    with pytest.raises(PromptAlignError):
        align_messages([ChatMessage(role="system", content="sys")], 10, count_chars)


def test_resolve_local_tokenizer_matches_alias_case_insensitively(tmp_path, monkeypatch):
    model_dir = tmp_path / "models" / "openbmb" / "MiniCPM5-2B"
    model_dir.mkdir(parents=True)
    tok = model_dir / "tokenizer.json"
    tok.write_text(json.dumps({"version": "1.0", "model": {"type": "BPE", "vocab": {}, "merges": []}}))
    roots = (tmp_path / "models",)
    assert resolve_local_tokenizer("minicpm5-2b", roots=roots) == tok
    assert resolve_local_tokenizer("MiniCPM5-2B", roots=roots) == tok
    assert resolve_local_tokenizer("no-such-model", roots=roots) is None


def count_ignoring_trailing_whitespace(messages, tools=None):
    """Fake tokenizer whose chat template swallows trailing whitespace.

    Whitespace appended to the end of the body adds no token (the template
    normalises it away), so the residual padding must fall back to printable
    fillers to reach the target exactly.
    """
    body = "".join((m.content or "").rstrip(" \n\t") for m in messages)
    return TEMPLATE_OVERHEAD + len(body)


def test_padding_falls_back_to_printable_filler_when_template_swallows_whitespace():
    body = "x" * 100 + "\n"
    target = TEMPLATE_OVERHEAD + 101
    out, info = align_messages([user(body)], target, count_ignoring_trailing_whitespace)
    assert info.final_tokens == target
    assert out[0].content.rstrip("\n ").startswith("x" * 100)


def count_with_two_token_filler(messages, tools=None):
    """Fake tokenizer where "x" costs two tokens and other characters one.

    From one token below the target, appending "x" overshoots by one (the BPE
    boundary jump that used to abort alignment); the pad loop must try other
    filler characters / edit sites instead of giving up.
    """
    body = "".join((m.content or "") for m in messages)
    return TEMPLATE_OVERHEAD + len(body) + body.count("x")


def test_padding_skips_two_token_jump_and_finds_single_token_filler():
    body = "y" * 1018
    target = TEMPLATE_OVERHEAD + 1022  # 3 tokens via "x" (1+1), then 1 via plain char
    out, info = align_messages([user(body)], target, count_with_two_token_filler)
    assert info.final_tokens == target
    assert count_with_two_token_filler(out) == target
    assert out[0].content.startswith("y" * 1018)
