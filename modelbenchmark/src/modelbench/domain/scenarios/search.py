"""Search agent scenario: local deterministic retrieval + answer (HotpotQA).

The model is given two tools:

* ``search(query)``   — lexical (BM25-lite) retrieval over the sample's
  paragraph corpus, fully local and deterministic;
* ``finish(answer)``   — submit the final answer.

The runner drives the tool loop; this module builds the request and scores the
final answer (EM/F1, mirroring HotpotQA).
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from ..evaluators.search import score_search
from ..models import RawResponse, Sample, Score, ToolCall, ToolSpec
from .common import SimpleScenario

SEARCH_TOOL = ToolSpec(
    type="function",
    function={
        "name": "search",
        "description": "Search the local document corpus and return the most relevant passages.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query keywords"}
            },
            "required": ["query"],
        },
    },
)

FINISH_TOOL = ToolSpec(
    type="function",
    function={
        "name": "finish",
        "description": "Submit the final short answer to the user's question.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "Final answer"}
            },
            "required": ["answer"],
        },
    },
)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


class SearchAgentScenario(SimpleScenario):
    name = "search_agent"
    dataset_name = "hotpotqa"
    agentic = True
    system_prompt = (
        "You are a research assistant. Use the search tool to find facts, then "
        "call finish with a short, exact answer (entity or phrase)."
    )

    # ---- request ------------------------------------------------------ #

    def render_question(self, sample: Sample) -> str:
        return (
            f"问题：{sample.question.strip()}\n\n"
            "请先调用 search 检索本地资料（可多次），确认答案后调用 finish 提交简短答案。"
        )

    def tools_for(self, sample: Sample):
        # Agent tools are defined by the scenario, not the dataset.
        return [SEARCH_TOOL, FINISH_TOOL]

    # ---- tool loop hooks ---------------------------------------------- #

    def execute_tool(self, sample: Sample, call: ToolCall) -> str:
        if call.name == "search":
            query = _extract_arg(call.arguments, "query")
            return self._search(sample, query)
        if call.name == "finish":
            return "ok"
        return f"error: unknown tool '{call.name}'"

    def final_answer_text(self, sample: Sample, response: RawResponse) -> str:
        for call in response.tool_calls:
            if call.name == "finish":
                answer = _extract_arg(call.arguments, "answer")
                if answer:
                    return answer
        return response.text or ""

    # ---- retrieval ----------------------------------------------------- #

    def _search(self, sample: Sample, query: str, top_k: int = 3) -> str:
        paragraphs: List[List[str]] = list((sample.metadata or {}).get("paragraphs") or [])
        if not paragraphs:
            return "No local corpus available for this sample."
        q_terms = Counter(_tokenize(query))
        if not q_terms:
            return "Empty query."
        scored = []
        for title, text in paragraphs:
            doc = _tokenize(f"{title} {text}")
            doc_counts = Counter(doc)
            score = 0.0
            for term, qf in q_terms.items():
                tf = doc_counts.get(term, 0)
                if tf:
                    score += qf * (1.0 + math.log(tf))
            if score > 0:
                scored.append((score, title, text))
        scored.sort(key=lambda x: (-x[0], x[1]))
        if not scored:
            return "No relevant passages found."
        chunks = []
        for score, title, text in scored[:top_k]:
            chunks.append(f"[{title}] (score {score:.2f})\n{text[:900]}")
        return "\n\n".join(chunks)

    # ---- scoring ------------------------------------------------------- #

    def evaluator(self, sample: Sample, response: RawResponse) -> Score:
        answer = self.final_answer_text(sample, response)
        synthetic = RawResponse(text=answer)
        score = score_search(sample, synthetic)
        score.detail["tool_calls"] = [
            {"name": c.name, "arguments": c.arguments} for c in response.tool_calls
        ]
        return score


def _extract_arg(arguments: str, key: str) -> str:
    try:
        data = json.loads(arguments or "{}")
        value = data.get(key)
        if isinstance(value, str):
            return value
        if value is not None:
            return json.dumps(value, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(rf'"{key}"\s*:\s*"([^"]*)"', arguments or "")
    return match.group(1) if match else ""
