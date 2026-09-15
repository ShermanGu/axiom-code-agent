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

You receive a compact catalog of available tool capabilities for planning context only. You cannot
call tools during planning. Use the catalog to make tool-dependent prerequisites and side effects
visible in the plan. required_capabilities describes what a step needs to accomplish;
candidate_tools contains exact catalog names that may help the executor. Candidate tools are hints,
not mandatory calls, and may be empty when no listed tool is relevant.

Return only valid JSON with this exact shape:
{
  "strategy": "short explanation",
  "steps": [
    {
      "id": "short_id",
      "title": "...",
      "description": "...",
      "depends_on": [],
      "required_capabilities": ["read repository files"],
      "candidate_tools": ["fs_read"]
    }
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
        tool_catalog: list[dict[str, Any]] | None = None,
        enabled: bool = True,
    ) -> TaskPlan:
        if not enabled:
            return self.fallback_plan(goal)
        request = self.build_request(goal, context=context, tool_catalog=tool_catalog)
        response = await self.provider.complete(request)
        available_tools = (
            {str(item.get("name", "")) for item in tool_catalog} if tool_catalog else None
        )
        return self.parse_response(goal, response.text, available_tools=available_tools)

    def build_request(
        self,
        goal: str,
        *,
        context: str = "",
        tool_catalog: list[dict[str, Any]] | None = None,
    ) -> ModelRequest:
        prompt = f"Goal:\n{goal}"
        if context:
            prompt += f"\n\nRelevant context:\n{context[:12_000]}"
        if tool_catalog:
            prompt += (
                "\n\nAvailable tool capabilities:\n"
                f"{_render_tool_catalog(tool_catalog, max_chars=12_000)}"
            )
        return ModelRequest(
            instructions=PLANNER_INSTRUCTIONS,
            input_items=[{"role": "user", "content": prompt}],
            tools=[],
            max_output_tokens=min(self.max_output_tokens, 4096),
        )

    def parse_response(
        self,
        goal: str,
        text: str,
        *,
        available_tools: set[str] | None = None,
    ) -> TaskPlan:
        try:
            payload = _parse_json(text)
            return _validate_plan(goal, payload, available_tools=available_tools)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            return self.fallback_plan(goal)

    def fallback_plan(self, goal: str) -> TaskPlan:
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


def _validate_plan(
    goal: str,
    payload: dict[str, Any],
    *,
    available_tools: set[str] | None = None,
) -> TaskPlan:
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
        candidate_tools = _string_list(
            raw.get("candidate_tools", []), limit=12, item_limit=64
        )
        if available_tools is not None:
            candidate_tools = [item for item in candidate_tools if item in available_tools]
        steps.append(
            PlanStep(
                id=identifier,
                title=str(raw.get("title") or identifier)[:160],
                description=str(raw.get("description") or raw.get("title") or goal)[:2000],
                depends_on=dependencies,
                required_capabilities=_string_list(
                    raw.get("required_capabilities", []), limit=12, item_limit=160
                ),
                candidate_tools=candidate_tools,
            )
        )
    return TaskPlan(goal=goal, steps=steps, strategy=str(payload.get("strategy", ""))[:2000])


def _fallback_plan(goal: str) -> TaskPlan:
    return TaskPlan(
        goal=goal,
        strategy="Execute the requested task directly and verify the result.",
        steps=[PlanStep(id="execute", title="Execute task", description=goal)],
    )


def _string_list(value: Any, *, limit: int, item_limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip()[:item_limit] for item in value[:limit] if str(item).strip()]


def _render_tool_catalog(catalog: list[dict[str, Any]], *, max_chars: int) -> str:
    entries: list[str] = []
    used = 2
    for item in catalog:
        rendered = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        additional = len(rendered) + (1 if entries else 0)
        if entries and used + additional > max_chars:
            break
        entries.append(rendered)
        used += additional
    return f"[{','.join(entries)}]"
