from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from axiom_agent.events import EventBus
from axiom_agent.types import ToolCall, ToolResult

ApprovalCallback = Callable[[str, str], bool | Awaitable[bool]]


@dataclass(slots=True)
class ToolContext:
    workspace: Path
    events: EventBus
    approval_mode: str = "on-risk"
    approve: ApprovalCallback | None = None
    services: dict[str, Any] = field(default_factory=dict)

    async def request_approval(self, action: str, reason: str) -> bool:
        if self.approval_mode == "deny":
            return False
        if self.approval_mode in {"never", "auto"}:
            return True
        if self.approve is None:
            return False
        decision = self.approve(action, reason)
        return await decision if isinstance(decision, Awaitable) else decision


class Tool(ABC):
    name: str
    description: str
    parameters: dict[str, Any]
    parallel_safe: bool = False

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "strict": False,
        }

    @abstractmethod
    async def run(self, arguments: dict[str, Any], context: ToolContext) -> Any:
        raise NotImplementedError


class FunctionTool(Tool):
    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        function: Callable[[dict[str, Any], ToolContext], Any | Awaitable[Any]],
        *,
        parallel_safe: bool = False,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.function = function
        self.parallel_safe = parallel_safe

    async def run(self, arguments: dict[str, Any], context: ToolContext) -> Any:
        result = self.function(arguments, context)
        return await result if isinstance(result, Awaitable) else result


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool, *, replace: bool = False) -> None:
        if tool.name in self._tools and not replace:
            raise ValueError(f"Tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def register_many(self, tools: Iterable[Tool]) -> None:
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        tool = self.get(call.name)
        context.events.emit(
            "tool.started", call_id=call.id, name=call.name, arguments=call.arguments
        )
        if tool is None:
            result = ToolResult(call.id, call.name, f"Unknown tool: {call.name}", is_error=True)
            context.events.emit("tool.failed", call_id=call.id, name=call.name, error=result.output)
            return result
        try:
            value = await tool.run(call.arguments, context)
            if isinstance(value, str):
                output = value
            else:
                output = json.dumps(value, ensure_ascii=False, default=str)
            result = ToolResult(call.id, call.name, output)
            context.events.emit("tool.completed", call_id=call.id, name=call.name, output=output)
            return result
        except Exception as exc:  # tool failures are observations, not loop failures
            result = ToolResult(call.id, call.name, f"{type(exc).__name__}: {exc}", is_error=True)
            context.events.emit("tool.failed", call_id=call.id, name=call.name, error=result.output)
            return result

    async def execute_many(self, calls: list[ToolCall], context: ToolContext) -> list[ToolResult]:
        tools = [self.get(call.name) for call in calls]
        if calls and all(tool is not None and tool.parallel_safe for tool in tools):
            return list(await asyncio.gather(*(self.execute(call, context) for call in calls)))
        results: list[ToolResult] = []
        for call in calls:
            results.append(await self.execute(call, context))
        return results
