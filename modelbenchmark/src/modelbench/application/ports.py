"""Application backend contract.

The benchmark core is written against this interface so a new served
application (oMLX today; Ollama / LM Studio / vLLM later) only needs an
adapter that implements model lifecycle + inference + cache administration.

`src/app_test/omlx` is the first implementation.  Keeping this contract small
is intentional: anything application-specific belongs in the adapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class BackendError(RuntimeError):
    """Raised for transport, protocol, or lifecycle failures."""


@dataclass
class ModelInfo:
    id: str
    loaded: bool = False
    max_context_window: Optional[int] = None
    estimated_size: Optional[int] = None
    engine_type: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


class AppBackend(ABC):
    """Minimal interface used by the runner and web console."""

    name: str = "backend"

    @abstractmethod
    def probe(self) -> Dict[str, Any]:
        """Return health/auth/version facts without side effects."""

    @abstractmethod
    def list_models(self) -> List[ModelInfo]:
        """Discover available models and their loaded state."""

    @abstractmethod
    def model_info(self, model_id: str) -> ModelInfo:
        """Return one model's current state/config."""

    @abstractmethod
    def apply_model_settings(self, model_id: str, settings: Dict[str, Any], optional: bool = True) -> Dict[str, Any]:
        """Persist per-model load/runtime settings (application-specific)."""

    @abstractmethod
    def load_model(self, model_id: str) -> None:
        """Load a model, blocking until the backend accepts the load."""

    @abstractmethod
    def wait_until_loaded(self, model_id: str, timeout_s: float = 600) -> bool:
        ...

    @abstractmethod
    def unload_model(self, model_id: str) -> None:
        ...

    @abstractmethod
    def wait_until_unloaded(self, model_id: str, timeout_s: float = 120) -> bool:
        ...

    @abstractmethod
    def clear_caches(self) -> Dict[str, Any]:
        """Clear prefix/KV/SSD caches; required for cold measurements."""

    @abstractmethod
    def tokenize_messages(self, messages: List[Any], model: str, tools: Optional[List[Any]] = None) -> int:
        ...

    @abstractmethod
    def chat_stream(self, request: Any, cache_store: bool = False) -> Any:
        """Stream one chat completion; return a StreamResult-like object."""

    @abstractmethod
    def close(self) -> None:
        ...
