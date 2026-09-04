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
        self._client = _create_client(
            self.config,
            "The OpenAI provider requires the 'openai' package. ",
        )
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


class OpenAIChatProvider(ModelProvider):
    """Adapter for OpenAI-compatible Chat Completions providers.

    Axiom's core uses Responses-style items because they represent tool loops
    cleanly. This boundary translates those items and tool schemas for services
    such as Groq, Gemini, and OpenRouter without coupling the executor to any
    one vendor.
    """

    def __init__(self, config: ModelConfig, client: Any | None = None) -> None:
        self.config = config
        self._client = client

    def _client_instance(self) -> Any:
        if self._client is not None:
            return self._client
        self._client = _create_client(
            self.config,
            "This provider requires the 'openai' package. ",
        )
        return self._client

    async def complete(self, request: ModelRequest) -> ModelResponse:
        client = self._client_instance()
        messages = [{"role": "system", "content": request.instructions}]
        messages.extend(_chat_messages(request.input_items))
        kwargs: dict[str, Any] = {
            "model": self.config.name,
            "messages": messages,
            # max_tokens is the widest-supported spelling across nominally
            # OpenAI-compatible services. Keep this boundary on their shared
            # subset instead of sending vendor-specific reasoning parameters.
            "max_tokens": request.max_output_tokens,
        }
        if request.tools:
            kwargs["tools"] = [_chat_tool_schema(tool) for tool in request.tools]

        response = await asyncio.to_thread(client.chat.completions.create, **kwargs)
        choice = response.choices[0]
        message = choice.message
        text = getattr(message, "content", "") or ""
        raw_tool_calls = getattr(message, "tool_calls", None) or []
        calls: list[ToolCall] = []
        serialized_calls: list[dict[str, Any]] = []
        for raw_call in raw_tool_calls:
            call_id = str(_field(raw_call, "id", "call"))
            function = _field(raw_call, "function", {})
            name = str(_field(function, "name", ""))
            raw_arguments = _field(function, "arguments", "{}") or "{}"
            if isinstance(raw_arguments, dict):
                arguments = raw_arguments
                raw_arguments = json.dumps(raw_arguments, ensure_ascii=False)
            else:
                raw_arguments = str(raw_arguments)
                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    arguments = {"_raw": raw_arguments}
            calls.append(ToolCall(id=call_id, name=name, arguments=arguments))
            serialized = (
                raw_call.model_dump(exclude_none=True)
                if hasattr(raw_call, "model_dump")
                else {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": raw_arguments},
                }
            )
            serialized_calls.append(serialized)

        output_item: dict[str, Any] = {"role": "assistant", "content": text}
        if serialized_calls:
            output_item["tool_calls"] = serialized_calls
        usage_obj = getattr(response, "usage", None)
        if usage_obj is None:
            usage: dict[str, Any] = {}
        elif hasattr(usage_obj, "model_dump"):
            usage = usage_obj.model_dump(exclude_none=True)
        else:
            usage = dict(usage_obj)
        return ModelResponse(
            text=text,
            tool_calls=calls,
            output_items=[output_item],
            response_id=getattr(response, "id", None),
            usage=usage,
        )


def _create_client(config: ModelConfig, missing_package_message: str) -> Any:
    try:
        from openai import DefaultHttpx2Client, OpenAI
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise RuntimeError(
            missing_package_message + "Install Axiom with: pip install -e ."
        ) from exc

    api_key = os.getenv(config.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing {config.api_key_env}. Set it in your environment, "
            "or run 'axiom demo' for the offline end-to-end demo."
        )
    kwargs: dict[str, Any] = {"api_key": api_key}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if mounts := _no_proxy_mounts(config.no_proxy):
        kwargs["http_client"] = DefaultHttpx2Client(mounts=mounts)
    return OpenAI(**kwargs)


def _no_proxy_mounts(value: str) -> dict[str, None]:
    entries = (item.strip() for item in value.split(","))
    return {entry if "://" in entry else f"all://{entry}": None for entry in entries if entry}


def _chat_messages(items: list[Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for original in items:
        item = (
            original.model_dump(exclude_none=True) if hasattr(original, "model_dump") else original
        )
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(item.get("call_id", "call")),
                    "content": str(item.get("output", "")),
                }
            )
        elif item_type == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": str(item.get("call_id", item.get("id", "call"))),
                            "type": "function",
                            "function": {
                                "name": str(item.get("name", "")),
                                "arguments": str(item.get("arguments", "{}")),
                            },
                        }
                    ],
                }
            )
        elif item.get("role") in {"user", "assistant", "tool"}:
            message = {key: value for key, value in item.items() if key != "type"}
            messages.append(message)
    return messages


def _chat_tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    function = {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
    }
    return {"type": "function", "function": function}


def _field(value: Any, name: str, default: Any) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)
