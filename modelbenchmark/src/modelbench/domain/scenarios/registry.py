"""Scenario registry: scenario id -> adapter class."""

from __future__ import annotations

from typing import Dict, List, Type

from .agent import GeneralAgentScenario
from .base import ScenarioAdapter
from .code import CodingScenario
from .instruction import InstructionFollowingScenario
from .knowledge import KnowledgeQAScenario
from .long_context import LongContextScenario
from .math import MathLogicScenario
from .multilingual import ChineseLongContextScenario, MultilingualScenario
from .perf import ThroughputScenario
from .search import SearchAgentScenario
from .speculative import SpeculativeDecodingScenario

_REGISTRY: Dict[str, Type[ScenarioAdapter]] = {
    cls.name: cls
    for cls in (
        KnowledgeQAScenario,
        InstructionFollowingScenario,
        LongContextScenario,
        MathLogicScenario,
        CodingScenario,
        GeneralAgentScenario,
        SearchAgentScenario,
        MultilingualScenario,
        ChineseLongContextScenario,
        SpeculativeDecodingScenario,
        ThroughputScenario,
    )
}


def available_scenarios() -> List[str]:
    return list(_REGISTRY)


def scenario_class(name: str) -> Type[ScenarioAdapter]:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown scenario '{name}'; available: {', '.join(available_scenarios())}"
        ) from exc


def build_scenario(name: str, params: Dict | None = None) -> ScenarioAdapter:
    return scenario_class(name)(params or {})
