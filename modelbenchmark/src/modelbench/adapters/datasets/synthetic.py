"""Synthetic dataset for the throughput probe.

The accuracy scenarios intentionally expect short answers (a letter, a number,
a tool call), so their `completion_tokens` is far below the 16-token floor used
for a valid decode-rate measurement.  This dataset feeds the ``throughput``
scenario, which asks for a long deterministic answer so that `gen_tps`, `tpot`
and peak CPU can be measured on a uniform synthetic prompt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ...domain.models import AnswerType, Sample
from .base import DatasetAdapter

_TOPICS = (
    "为什么沿海地区昼夜温差比内陆小",
    "如何为家庭网络设计一套备份方案",
    "解释光合作用与呼吸作用的关系",
    "一次完整的软件发布流程包含哪些阶段",
    "城市化对地方气候的影响",
    "怎样系统地学习一门新的编程语言",
    "远程协作团队如何保持信息同步",
    "数据库索引的原理与代价",
)

_QUESTION = (
    "请围绕下面的话题写一篇结构清晰的说明文，要求：\n"
    "1. 分 3-5 个小节，每节写 2-3 段；\n"
    "2. 总长度不少于 400 字；\n"
    "3. 使用具体例子，不要只列提纲；\n"
    "4. 不要提前结束，尽量写满。\n\n"
    "话题：{topic}"
)


class SyntheticThroughputAdapter(DatasetAdapter):
    """Deterministic long-form prompts, generated locally (no download)."""

    name = "synthetic"
    scenario = "throughput"
    source = "generated locally (deterministic template)"
    license = "project-internal"

    def ensure_downloaded(self) -> Path:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        return self.raw_dir

    def load(self) -> List[Sample]:
        count = int(self.params.get("count", 4))
        samples: List[Sample] = []
        for idx in range(max(count, 1)):
            topic = _TOPICS[idx % len(_TOPICS)]
            samples.append(
                Sample(
                    sample_id=f"synthetic-{idx:04d}",
                    scenario="throughput",
                    dataset=self.name,
                    subtask="long_form",
                    question=_QUESTION.format(topic=topic),
                    context="",
                    choices=None,
                    answer="",
                    answer_type=AnswerType.EXACT,
                    metadata={"topic": topic, "language": "zh"},
                )
            )
        return samples
