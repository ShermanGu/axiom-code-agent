from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from axiom_agent.app import AxiomApp
from axiom_agent.config import AxiomConfig
from axiom_agent.events import Event
from axiom_agent.execution.store import UnsafeResumeError
from axiom_agent.providers.base import ModelProvider
from axiom_agent.tools.base import FunctionTool, ToolContext
from axiom_agent.types import ModelRequest, ModelResponse, ToolCall


@dataclass(slots=True)
class RecoveryDemoResult:
    checks: dict[str, bool]
    conversation_ids: dict[str, str]

    @property
    def success(self) -> bool:
        return all(self.checks.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "checks": self.checks,
            "conversation_ids": self.conversation_ids,
        }


class _RecoveryProvider(ModelProvider):
    def __init__(
        self,
        tool_name: str,
        *,
        allow_planning: bool,
        interrupt_after_output: bool = False,
    ) -> None:
        self.tool_name = tool_name
        self.allow_planning = allow_planning
        self.interrupt_after_output = interrupt_after_output

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if "task planner" in request.instructions.casefold():
            if not self.allow_planning:
                raise AssertionError("The checkpointed plan was unexpectedly recreated")
            text = json.dumps(
                {
                    "strategy": "Exercise one deterministic recovery step.",
                    "steps": [
                        {
                            "id": "recover",
                            "title": "Recovery checkpoint",
                            "description": "Call the deterministic recovery demo tool.",
                            "depends_on": [],
                        }
                    ],
                }
            )
            return _text_response(text)

        if _contains_tool_output(request.input_items):
            if self.interrupt_after_output:
                raise asyncio.CancelledError
            return _text_response("Recovery demo step completed.")

        call = ToolCall(f"{self.tool_name}-call", self.tool_name, {})
        return ModelResponse(
            text="",
            tool_calls=[call],
            output_items=[
                {
                    "type": "function_call",
                    "call_id": call.id,
                    "name": call.name,
                    "arguments": "{}",
                }
            ],
            usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )


def _text_response(text: str) -> ModelResponse:
    return ModelResponse(
        text=text,
        output_items=[{"role": "assistant", "content": text}],
        usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    )


def _contains_tool_output(items: list[Any]) -> bool:
    return any(
        isinstance(item, dict) and item.get("type") == "function_call_output"
        for item in items
    )


def _counter_tool(name: str, counter: dict[str, int], *, interrupt: bool = False) -> FunctionTool:
    async def run(_arguments: dict[str, Any], _context: ToolContext) -> str:
        counter["calls"] += 1
        if interrupt:
            raise asyncio.CancelledError
        return f"{name} completed"

    return FunctionTool(
        name,
        "Deterministic offline tool used by the recovery acceptance demo.",
        {"type": "object", "properties": {}, "additionalProperties": False},
        run,
    )


def _capture_started_run(target: list[str]) -> Callable[[Event], None]:
    def capture(event: Event) -> None:
        if event.type == "agent.started":
            target.append(str(event.data["run_id"]))

    return capture


async def run_recovery_demo(config: AxiomConfig) -> RecoveryDemoResult:
    """Exercise recovery behavior through the real Agent and SQLite store."""
    config.mcp.servers.clear()
    config.agent.planning = True
    config.agent.max_step_retries = 0
    config.agent.auto_memory = False

    checkpoint_calls = {"calls": 0}
    checkpoint_app = AxiomApp(
        config,
        provider=_RecoveryProvider(
            "demo_checkpoint", allow_planning=True, interrupt_after_output=True
        ),
    )
    checkpoint_ids: list[str] = []
    checkpoint_app.events.subscribe(_capture_started_run(checkpoint_ids))
    checkpoint_app.tools.register(_counter_tool("demo_checkpoint", checkpoint_calls))
    async with checkpoint_app:
        try:
            await checkpoint_app.agent.run("Demonstrate recovery after a completed tool")
        except asyncio.CancelledError:
            pass
        checkpoint_run = checkpoint_app.execution.get_run(checkpoint_ids[-1])

    resume_app = AxiomApp(
        config, provider=_RecoveryProvider("demo_checkpoint", allow_planning=False)
    )
    resume_app.tools.register(_counter_tool("demo_checkpoint", checkpoint_calls))
    async with resume_app:
        checkpoint_result = await resume_app.agent.resume(checkpoint_run.id)
        checkpoint_detail = resume_app.execution.detail(checkpoint_run.id)

    uncertain_calls = {"calls": 0}
    uncertain_app = AxiomApp(
        config, provider=_RecoveryProvider("demo_uncertain", allow_planning=True)
    )
    uncertain_ids: list[str] = []
    uncertain_app.events.subscribe(_capture_started_run(uncertain_ids))
    uncertain_app.tools.register(
        _counter_tool("demo_uncertain", uncertain_calls, interrupt=True)
    )
    async with uncertain_app:
        try:
            await uncertain_app.agent.run("Demonstrate an uncertain tool outcome")
        except asyncio.CancelledError:
            pass
        uncertain_run = uncertain_app.execution.get_run(uncertain_ids[-1])

    blocked_app = AxiomApp(
        config, provider=_RecoveryProvider("demo_uncertain", allow_planning=False)
    )
    blocked = False
    async with blocked_app:
        try:
            await blocked_app.agent.resume(uncertain_run.id)
        except UnsafeResumeError:
            blocked = blocked_app.execution.get_run(uncertain_run.id).status == "blocked"

    retry_calls = {"calls": 0}
    retry_app = AxiomApp(
        config, provider=_RecoveryProvider("demo_uncertain", allow_planning=False)
    )
    retry_app.tools.register(_counter_tool("demo_uncertain", retry_calls))
    async with retry_app:
        retry_result = await retry_app.agent.resume(
            uncertain_run.id, retry_uncertain_tools=True
        )
        uncertain_detail = retry_app.execution.detail(uncertain_run.id)
        history_visible = all(
            bool(detail["turns"])
            and bool(detail["tool_calls"])
            and bool(detail["metrics"])
            for detail in (checkpoint_detail, uncertain_detail)
        )

    return RecoveryDemoResult(
        checks={
            "checkpoint_persisted": checkpoint_run.status == "interrupted",
            "completed_tool_not_replayed": (
                checkpoint_result.success
                and checkpoint_calls["calls"] == 1
                and len(checkpoint_detail["tool_calls"]) == 1
            ),
            "uncertain_tool_blocked": blocked and uncertain_calls["calls"] == 1,
            "explicit_retry_resumed": (
                retry_result.success
                and retry_calls["calls"] == 1
                and len(uncertain_detail["tool_calls"]) == 2
            ),
            "history_and_metrics_inspectable": history_visible,
        },
        conversation_ids={
            "checkpoint": checkpoint_run.conversation_id,
            "uncertain": uncertain_run.conversation_id,
        },
    )
