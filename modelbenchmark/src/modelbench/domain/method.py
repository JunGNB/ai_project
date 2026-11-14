"""Test methods supported by the benchmark.

``performance`` measures speed / memory only (``score`` stays null);
``accuracy`` sends the pre-built accuracy prompts and scores every response with
a deterministic judge.  The two methods share the measurement kernel but never
share an aggregate: a run is always exactly one method.

The enum is a ``str`` enum on purpose: ``run.json.method``, ``batch.json.method``
and CLI/Web payloads keep working with plain strings, while code compares
against ``RunMethod.ACCURACY`` instead of hard-coded literals.
"""

from __future__ import annotations

from enum import Enum


class RunMethod(str, Enum):
    PERFORMANCE = "performance"
    ACCURACY = "accuracy"

    @classmethod
    def parse(cls, value: "str | RunMethod | None") -> "RunMethod":
        """Coerce a config/CLI value; unknown values raise ``ValueError``."""
        if isinstance(value, RunMethod):
            return value
        if value is None or value == "":
            return cls.PERFORMANCE
        return cls(str(value).strip().lower())
