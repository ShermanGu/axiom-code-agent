from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from axiom_agent.config import ModelConfig
from axiom_agent.providers.base import ModelProvider
from axiom_agent.types import ModelRequest, ModelResponse, ToolCall


class OpenAIResponsesProvider(ModelProvider):
    """OpenAI Responses API adapter.

    The SDK import and client construction are lazy so Axiom's offline demo,
    memory commands, and unit tests work without credentials or optional
    packages being available.
    """

    def __init__(self, config: ModelConfig, client: Any | None = None) -> None:
        self.config = config
        self._client = client

    def _client_instance(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on installation
            raise RuntimeError(
                "The OpenAI provider requires the 'openai' package. "
                "Install Axiom with: pip install -e ."
            ) from exc

        api_key = os.getenv(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing {self.config.api_key_env}. Set it in your environment, "
                "or run 'axiom demo' for the offline end-to-end demo."
            )
        kwargs: dict[str, Any] = {"api_key": api_key}
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        self._client = OpenAI(**kwargs)
        return self._client

    async def complete(self, request: ModelRequest) -> ModelResponse:
        client = self._client_instance()
        kwargs: dict[str, Any] = {
            "model": self.config.name,
            "instructions": request.instructions,
            "input": request.input_items,
            "tools": request.tools,
            "store": False,
            "parallel_tool_calls": True,
            "max_output_tokens": request.max_output_tokens,
        }
        if self.config.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.config.reasoning_effort}

        response = await asyncio.to_thread(client.responses.create, **kwargs)
        output_items = list(response.output)
        calls: list[ToolCall] = []
        for item in response.output:
            if getattr(item, "type", None) != "function_call":
                continue
            raw_arguments = getattr(item, "arguments", "{}") or "{}"
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                arguments = {"_raw": raw_arguments}
            calls.append(
                ToolCall(
                    id=getattr(item, "call_id", getattr(item, "id", "call")),
                    name=item.name,
                    arguments=arguments,
                )
            )

        usage_obj = getattr(response, "usage", None)
        if usage_obj is None:
            usage: dict[str, Any] = {}
        elif hasattr(usage_obj, "model_dump"):
            usage = usage_obj.model_dump(exclude_none=True)
        else:
            usage = dict(usage_obj)
        return ModelResponse(
            text=getattr(response, "output_text", "") or "",
            tool_calls=calls,
            output_items=output_items,
            response_id=getattr(response, "id", None),
            usage=usage,
        )

