from __future__ import annotations

import json
import re
from typing import Any

from axiom_agent.providers.base import ModelProvider
from axiom_agent.types import ModelRequest, PlanStep, TaskPlan

PLANNER_INSTRUCTIONS = """You are Axiom's task planner. Decompose the goal into the smallest
useful set of executable steps. Prefer one step for simple tasks and at most eight for complex
tasks. Steps must be concrete, independently verifiable, and ordered through depends_on.
Do not include meta-steps such as 'understand the request' or 'make a plan'.

Return only valid JSON with this exact shape:
{
  "strategy": "short explanation",
  "steps": [
    {"id": "short_id", "title": "...", "description": "...", "depends_on": []}
  ]
}
IDs must be unique ASCII identifiers. Every dependency must name an earlier step.
"""


class Planner:
    def __init__(self, provider: ModelProvider, *, max_output_tokens: int = 4096) -> None:
        self.provider = provider
        self.max_output_tokens = max_output_tokens

    async def create_plan(
        self,
        goal: str,
        *,
        context: str = "",
        enabled: bool = True,
    ) -> TaskPlan:
        if not enabled:
            return _fallback_plan(goal)
        prompt = f"Goal:\n{goal}"
        if context:
            prompt += f"\n\nRelevant context:\n{context[:12_000]}"
        response = await self.provider.complete(
            ModelRequest(
                instructions=PLANNER_INSTRUCTIONS,
                input_items=[{"role": "user", "content": prompt}],
                tools=[],
                max_output_tokens=min(self.max_output_tokens, 4096),
            )
        )
        try:
            payload = _parse_json(response.text)
            return _validate_plan(goal, payload)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            return _fallback_plan(goal)


def _parse_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL)
    if fence:
        stripped = fence.group(1)
    if not stripped.startswith("{"):
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Planner did not return JSON")
        stripped = stripped[start : end + 1]
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise TypeError("Plan must be a JSON object")
    return value


def _validate_plan(goal: str, payload: dict[str, Any]) -> TaskPlan:
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("Plan requires at least one step")
    if len(raw_steps) > 12:
        raw_steps = raw_steps[:12]
    steps: list[PlanStep] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_steps, 1):
        if not isinstance(raw, dict):
            raise TypeError("Every step must be an object")
        identifier = re.sub(r"[^a-zA-Z0-9_-]", "_", str(raw.get("id") or f"step_{index}"))[:40]
        if not identifier or identifier in seen:
            identifier = f"step_{index}"
        dependencies = [str(item) for item in raw.get("depends_on", [])]
        if any(item not in seen for item in dependencies):
            raise ValueError("Dependencies must refer to earlier steps")
        seen.add(identifier)
        steps.append(
            PlanStep(
                id=identifier,
                title=str(raw.get("title") or identifier)[:160],
                description=str(raw.get("description") or raw.get("title") or goal)[:2000],
                depends_on=dependencies,
            )
        )
    return TaskPlan(goal=goal, steps=steps, strategy=str(payload.get("strategy", ""))[:2000])


def _fallback_plan(goal: str) -> TaskPlan:
    return TaskPlan(
        goal=goal,
        strategy="Execute the requested task directly and verify the result.",
        steps=[PlanStep(id="execute", title="Execute task", description=goal)],
    )

