from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from axiom_agent.config import AxiomConfig
from axiom_agent.events import EventBus
from axiom_agent.memory.store import MemoryRecord, SQLiteMemoryStore
from axiom_agent.planning.planner import Planner
from axiom_agent.prompts import FINALIZER_PROMPT, build_step_input, build_step_instructions
from axiom_agent.providers.base import ModelProvider
from axiom_agent.skills.loader import Skill, SkillRegistry
from axiom_agent.tools.base import ToolContext, ToolRegistry
from axiom_agent.types import ModelRequest, PlanStep, TaskPlan


@dataclass(slots=True)
class AgentResult:
    output: str
    conversation_id: str
    plan: TaskPlan
    success: bool
    usage: dict[str, int] = field(default_factory=dict)


class Agent:
    def __init__(
        self,
        *,
        config: AxiomConfig,
        provider: ModelProvider,
        tools: ToolRegistry,
        memory: SQLiteMemoryStore,
        skills: SkillRegistry,
        events: EventBus,
        tool_context: ToolContext,
    ) -> None:
        self.config = config
        self.provider = provider
        self.tools = tools
        self.memory = memory
        self.skills = skills
        self.events = events
        self.tool_context = tool_context
        self.planner = Planner(provider, max_output_tokens=config.model.max_output_tokens)

    async def run(self, goal: str, *, conversation_id: str | None = None) -> AgentResult:
        goal = goal.strip()
        if not goal:
            raise ValueError("Goal must not be empty")
        conversation_id = self.memory.create_conversation(goal, conversation_id)
        previous_messages = self.memory.recent_messages(
            conversation_id, self.config.memory.recent_messages
        )
        self.memory.add_message(conversation_id, "user", goal)
        self.tool_context.services["conversation_id"] = conversation_id
        self.events.emit("agent.started", conversation_id=conversation_id, goal=goal)

        memories = self.memory.search(goal, limit=self.config.memory.retrieval_limit)
        selected_skills = self.skills.select(
            goal,
            limit=self.config.skills.max_active,
            auto=self.config.skills.auto_select,
        )
        self.events.emit(
            "context.prepared",
            memories=[item.id for item in memories],
            skills=[item.name for item in selected_skills],
        )
        planner_context = _planner_context(previous_messages, memories, selected_skills)
        plan = await self.planner.create_plan(
            goal, context=planner_context, enabled=self.config.agent.planning
        )
        self.events.emit("plan.created", plan=plan.as_dict())

        usage: dict[str, int] = {}
        while not plan.done:
            ready = plan.ready_steps()
            if not ready:
                for step in plan.steps:
                    if step.status == "pending":
                        step.status = "skipped"
                        step.error = "A dependency did not complete"
                        self.events.emit("step.skipped", step_id=step.id, reason=step.error)
                break
            step = ready[0]
            await self._execute_with_retries(
                goal,
                plan,
                step,
                previous_messages,
                memories,
                selected_skills,
                usage,
            )

        output = await self._finalize(goal, plan, usage)
        success = all(step.status == "completed" for step in plan.steps)
        self.memory.add_message(
            conversation_id,
            "assistant",
            output,
            {"success": success, "plan": plan.as_dict()},
        )
        if self.config.agent.auto_memory:
            episode = f"Goal: {goal}\nOutcome: {output}"[:12_000]
            self.memory.remember(
                episode,
                kind="episode",
                tags=["task", "success" if success else "incomplete"],
                importance=0.6 if success else 0.7,
                source=conversation_id,
            )
        self.events.emit(
            "agent.completed",
            conversation_id=conversation_id,
            success=success,
            output=output,
            usage=usage,
        )
        return AgentResult(output, conversation_id, plan, success, usage)

    async def _execute_with_retries(
        self,
        goal: str,
        plan: TaskPlan,
        step: PlanStep,
        previous_messages: list[dict[str, Any]],
        memories: list[MemoryRecord],
        selected_skills: list[Skill],
        usage: dict[str, int],
    ) -> None:
        attempts = self.config.agent.max_step_retries + 1
        last_error = ""
        for attempt in range(1, attempts + 1):
            step.status = "running"
            self.events.emit(
                "step.started",
                step_id=step.id,
                title=step.title,
                attempt=attempt,
                attempts=attempts,
            )
            try:
                result = await self._execute_step(
                    goal,
                    plan,
                    step,
                    previous_messages,
                    memories,
                    selected_skills,
                    usage,
                )
                step.result = result
                step.status = "completed"
                self.events.emit("step.completed", step_id=step.id, result=result)
                return
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                self.events.emit("step.attempt_failed", step_id=step.id, error=last_error)
        step.status = "failed"
        step.error = last_error
        self.events.emit("step.failed", step_id=step.id, error=last_error)

    async def _execute_step(
        self,
        goal: str,
        plan: TaskPlan,
        step: PlanStep,
        previous_messages: list[dict[str, Any]],
        memories: list[MemoryRecord],
        selected_skills: list[Skill],
        usage: dict[str, int],
    ) -> str:
        completed = [item for item in plan.steps if item.status == "completed"]
        history: list[Any] = build_step_input(
            goal=goal,
            step=step,
            previous_messages=previous_messages,
            completed_steps=completed,
        )
        instructions = build_step_instructions(
            agent_name=self.config.agent.name,
            workspace=self.config.workspace.root,
            plan=plan,
            step=step,
            memories=memories,
            skill_registry=self.skills,
            active_skills=selected_skills,
        )
        for turn in range(1, self.config.agent.max_turns + 1):
            self.events.emit("model.started", step_id=step.id, turn=turn)
            response = await self.provider.complete(
                ModelRequest(
                    instructions=instructions,
                    input_items=history,
                    tools=self.tools.schemas(),
                    max_output_tokens=self.config.model.max_output_tokens,
                )
            )
            _merge_usage(usage, response.usage)
            self.events.emit(
                "model.completed",
                step_id=step.id,
                turn=turn,
                response_id=response.response_id,
                tool_calls=[call.name for call in response.tool_calls],
                text=response.text,
            )
            history.extend(response.output_items)
            if response.tool_calls:
                if not response.output_items:
                    history.extend(
                        {
                            "type": "function_call",
                            "call_id": call.id,
                            "name": call.name,
                            "arguments": json.dumps(call.arguments, ensure_ascii=False),
                        }
                        for call in response.tool_calls
                    )
                results = await self.tools.execute_many(response.tool_calls, self.tool_context)
                history.extend(
                    {
                        "type": "function_call_output",
                        "call_id": result.call_id,
                        "output": result.output,
                    }
                    for result in results
                )
                continue
            if response.text.strip():
                return response.text.strip()
            raise RuntimeError("Model returned neither tool calls nor a final response")
        raise RuntimeError(f"Step exceeded max_turns={self.config.agent.max_turns}")

    async def _finalize(self, goal: str, plan: TaskPlan, usage: dict[str, int]) -> str:
        completed = [step for step in plan.steps if step.status == "completed"]
        failed = [step for step in plan.steps if step.status != "completed"]
        if len(plan.steps) == 1 and completed and not failed:
            return completed[0].result
        execution = json.dumps(plan.as_dict(), ensure_ascii=False, indent=2)[:30_000]
        try:
            response = await self.provider.complete(
                ModelRequest(
                    instructions=FINALIZER_PROMPT,
                    input_items=[
                        {
                            "role": "user",
                            "content": f"Original goal:\n{goal}\n\nExecution results:\n{execution}",
                        }
                    ],
                    tools=[],
                    max_output_tokens=self.config.model.max_output_tokens,
                )
            )
            _merge_usage(usage, response.usage)
            if response.text.strip():
                return response.text.strip()
        except Exception as exc:
            self.events.emit("finalizer.failed", error=f"{type(exc).__name__}: {exc}")
        lines = [step.result for step in completed if step.result]
        lines.extend(f"{step.title}: {step.error or step.status}" for step in failed)
        return "\n\n".join(lines) or "The task produced no result."


def _planner_context(
    messages: list[dict[str, Any]], memories: list[MemoryRecord], skills: list[Skill]
) -> str:
    recent = "\n".join(
        f"{item.get('role')}: {str(item.get('content', ''))[:1000]}" for item in messages[-6:]
    )
    durable = "\n".join(f"[{item.kind}] {item.content[:1000]}" for item in memories)
    skill_names = ", ".join(item.name for item in skills) or "none"
    return (
        f"Recent conversation:\n{recent or '(none)'}\n\n"
        f"Memory:\n{durable or '(none)'}\n\nActive skills: {skill_names}"
    )


def _merge_usage(target: dict[str, int], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, int):
            target[key] = target.get(key, 0) + value
