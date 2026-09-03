from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

JSON = dict[str, Any]


@dataclass(slots=True)
class ToolCall:
    """A provider-neutral request to invoke a tool."""

    id: str
    name: str
    arguments: JSON


@dataclass(slots=True)
class ModelRequest:
    """One model turn.

    ``input_items`` intentionally mirrors the Responses API item model. Other
    providers can translate it at their boundary, while the agent core remains
    unaware of SDK-specific response classes.
    """

    instructions: str
    input_items: list[Any]
    tools: list[JSON] = field(default_factory=list)
    max_output_tokens: int = 8192


@dataclass(slots=True)
class ModelResponse:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    output_items: list[Any] = field(default_factory=list)
    response_id: str | None = None
    usage: JSON = field(default_factory=dict)


@dataclass(slots=True)
class ToolResult:
    call_id: str
    name: str
    output: str
    is_error: bool = False
    metadata: JSON = field(default_factory=dict)


StepStatus = Literal["pending", "running", "completed", "failed", "skipped"]


@dataclass(slots=True)
class PlanStep:
    id: str
    title: str
    description: str
    depends_on: list[str] = field(default_factory=list)
    status: StepStatus = "pending"
    result: str = ""
    error: str = ""

    def as_dict(self) -> JSON:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "depends_on": self.depends_on,
            "status": self.status,
            "result": self.result,
            "error": self.error,
        }


@dataclass(slots=True)
class TaskPlan:
    goal: str
    steps: list[PlanStep]
    strategy: str = ""

    def as_dict(self) -> JSON:
        return {
            "goal": self.goal,
            "strategy": self.strategy,
            "steps": [step.as_dict() for step in self.steps],
        }

    def ready_steps(self) -> list[PlanStep]:
        completed = {step.id for step in self.steps if step.status == "completed"}
        return [
            step
            for step in self.steps
            if step.status == "pending" and set(step.depends_on).issubset(completed)
        ]

    @property
    def done(self) -> bool:
        return all(step.status in {"completed", "failed", "skipped"} for step in self.steps)

