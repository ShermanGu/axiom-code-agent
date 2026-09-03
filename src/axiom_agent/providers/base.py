from __future__ import annotations

from abc import ABC, abstractmethod

from axiom_agent.types import ModelRequest, ModelResponse


class ModelProvider(ABC):
    """Boundary implemented by every model backend."""

    @abstractmethod
    async def complete(self, request: ModelRequest) -> ModelResponse:
        raise NotImplementedError

    async def close(self) -> None:
        return None

