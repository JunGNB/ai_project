"""oMLX implementation of the application-backend contract."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from modelbench.application.ports import ModelInfo
from .client import OMLXClient, OMLXError, StreamResult


class OMLXBackend(OMLXClient):
    """Thin adapter: inference/client behaviour + model-settings lifecycle."""

    name = "omlx"

    def apply_model_settings(
        self, model_id: str, settings: Dict[str, Any], optional: bool = True
    ) -> Dict[str, Any]:
        if not settings:
            return {"applied": True, "settings": {}, "skipped": True}
        try:
            response = self.update_model_settings(model_id, settings)
        except OMLXError as exc:
            if optional:
                return {
                    "applied": False,
                    "optional": True,
                    "settings": dict(settings),
                    "error": str(exc),
                }
            raise
        return {"applied": True, "settings": dict(settings), "response": response}

    def list_model_infos(self) -> List[ModelInfo]:
        status = self.models_status()
        out: List[ModelInfo] = []
        for entry in status.get("models", []):
            out.append(
                ModelInfo(
                    id=str(entry.get("id")),
                    loaded=bool(entry.get("loaded")),
                    max_context_window=entry.get("max_context_window")
                    or entry.get("model_context_length"),
                    estimated_size=entry.get("estimated_size"),
                    engine_type=entry.get("engine_type"),
                    raw=entry,
                )
            )
        return out

    def model_info(self, model_id: str) -> ModelInfo:
        entry = self.model_config(model_id)
        return ModelInfo(
            id=str(entry.get("id")),
            loaded=bool(entry.get("loaded")),
            max_context_window=entry.get("max_context_window")
            or entry.get("model_context_length"),
            estimated_size=entry.get("estimated_size"),
            engine_type=entry.get("engine_type"),
            raw=entry,
        )

    def probe(self) -> Dict[str, Any]:  # type: ignore[override]
        return super().probe()

    def chat_stream(self, request: Any, cache_store: bool = False) -> StreamResult:  # type: ignore[override]
        return super().chat_stream(request, cache_store=cache_store)
