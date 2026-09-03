from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from axiom_agent.config import ModelConfig
from axiom_agent.providers.base import ModelProvider
from axiom_agent.providers.demo import DemoProvider
from axiom_agent.providers.openai import OpenAIResponsesProvider


def create_provider(config: ModelConfig) -> ModelProvider:
    if config.provider == "openai":
        return OpenAIResponsesProvider(config)
    if config.provider == "demo":
        return DemoProvider()
    if ":" in config.provider:
        module_name, factory_name = config.provider.split(":", 1)
        module = importlib.import_module(module_name)
        factory: Callable[[ModelConfig], Any] = getattr(module, factory_name)
        provider = factory(config)
        if not isinstance(provider, ModelProvider):
            raise TypeError(f"Custom provider {config.provider!r} did not return ModelProvider")
        return provider
    raise ValueError(
        f"Unknown model provider {config.provider!r}. Use 'openai', 'demo', "
        "or a 'module:factory' plugin."
    )
