from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from axiom_agent.config import AxiomConfig
from axiom_agent.events import EventBus
from axiom_agent.execution.store import (
    ExecutionStore,
    RunRecord,
    RunStatus,
    UnsafeResumeError,
)
from axiom_agent.memory.store import MemoryRecord, SQLiteMemoryStore
from axiom_agent.planning.planner import Planner
from axiom_agent.prompts import FINALIZER_PROMPT, build_step_input, build_step_instructions
from axiom_agent.providers.base import ModelProvider
from axiom_agent.skills.loader import Skill, SkillRegistry
from axiom_agent.tools.base import ToolContext, ToolRegistry
from axiom_agent.types import ModelRequest, ModelResponse, PlanStep, TaskPlan


@dataclass(slots=True)
class AgentResult:
    output: str
    conversation_id: str
    plan: TaskPlan
    success: bool
    usage: dict[str, int] = field(default_factory=dict)
    run_id: str = ""
    status: str = "completed"
    metrics: dict[str, dict[str, int]] = field(default_factory=dict)


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
        execution: ExecutionStore,
    ) -> None:
        self.config = config
        self.provider = provider
        self.tools = tools
        self.memory = memory
        self.skills = skills
        self.events = events
        self.tool_context = tool_context
        self.execution = execution
        self.planner = Planner(provider, max_output_tokens=config.model.max_output_tokens)
        self._active_run_id: str | None = None
        self._cancel_requested = False
        self.tool_context.services["execution_store"] = execution

    async def run(self, goal: str, *, conversation_id: str | None = None) -> AgentResult:
        if self._active_run_id is not None:
            raise RuntimeError(f"Agent is already running {self._active_run_id}")
        goal = goal.strip()
        if not goal:
            raise ValueError("Goal must not be empty")
        conversation_id = self.memory.create_conversation(goal, conversation_id)
        previous_messages = self.memory.recent_messages(
            conversation_id, self.config.memory.recent_messages
        )
        self.memory.add_message(conversation_id, "user", goal)
        run_id = self.execution.create_run(conversation_id, goal, {})
        self._begin_active(run_id, conversation_id)
        self.events.emit(
            "agent.started", run_id=run_id, conversation_id=conversation_id, goal=goal
        )
        try:
            memories = self.memory.search(goal, limit=self.config.memory.retrieval_limit)
            selected_skills = self.skills.select(
                goal,
                limit=self.config.skills.max_active,
                auto=self.config.skills.auto_select,
            )
            context = _context_snapshot(
                previous_messages, memories, selected_skills, config=self.config
            )
            self.execution.update_context(run_id, context)
            self.events.emit(
                "context.prepared",
                run_id=run_id,
                conversation_id=conversation_id,
                memories=[item.id for item in memories],
                skills=[item.name for item in selected_skills],
            )
            record = self.execution.get_run(run_id)
            return await self._execute_record(
                record,
                previous_messages,
                memories,
                selected_skills,
                resuming=False,
                retry_uncertain=False,
            )
        except asyncio.CancelledError:
            self._interrupt_active(run_id)
            raise
        except Exception as exc:
            self._fail_active(run_id, exc)
            raise
        finally:
            self._end_active()

    async def resume(
        self, run_id: str, *, retry_uncertain_tools: bool = False
    ) -> AgentResult:
        if self._active_run_id is not None:
            raise RuntimeError(f"Agent is already running {self._active_run_id}")
        try:
            record = self.execution.prepare_resume(
                run_id, retry_uncertain=retry_uncertain_tools
            )
        except UnsafeResumeError as exc:
            resolved = self.execution.resolve_run_id(run_id)
            self.events.emit("agent.blocked", run_id=resolved, error=str(exc))
            raise
        self._begin_active(record.id, record.conversation_id)
        self.events.emit(
            "agent.resumed",
            run_id=record.id,
            conversation_id=record.conversation_id,
            goal=record.goal,
            retry_uncertain_tools=retry_uncertain_tools,
        )
        try:
            previous_messages, memories, selected_skills = _restore_context(record.context)
            return await self._execute_record(
                record,
                previous_messages,
                memories,
                selected_skills,
                resuming=True,
                retry_uncertain=retry_uncertain_tools,
            )
        except asyncio.CancelledError:
            self._interrupt_active(record.id)
            raise
        except Exception as exc:
            self._fail_active(record.id, exc)
            raise
        finally:
            self._end_active()

    def request_cancel(self) -> None:
        """Mark an upcoming task cancellation as explicit rather than environmental."""

        self._cancel_requested = True

    async def _execute_record(
        self,
        record: RunRecord,
        previous_messages: list[dict[str, Any]],
        memories: list[MemoryRecord],
        selected_skills: list[Skill],
        *,
        resuming: bool,
        retry_uncertain: bool,
    ) -> AgentResult:
        plan = record.plan
        plan_id = record.plan_id
        step_record_ids = record.step_record_ids or {}
        if plan is None:
            plan = await self._create_or_recover_plan(
                record.id,
                record.goal,
                previous_messages,
                memories,
                selected_skills,
                resuming=resuming,
            )
            plan_id, step_record_ids = self.execution.save_plan(record.id, plan)
            self.events.emit(
                "plan.created",
                run_id=record.id,
                conversation_id=record.conversation_id,
                plan_id=plan_id,
                plan=plan.as_dict(),
            )
        assert plan_id is not None

        while not plan.done:
            ready = plan.ready_steps()
            if not ready:
                for step in plan.steps:
                    if step.status == "pending":
                        step.status = "skipped"
                        step.error = "A dependency did not complete"
                        self.execution.set_step_status(
                            step_record_ids[step.id], "skipped", error=step.error
                        )
                        self.events.emit(
                            "step.skipped",
                            run_id=record.id,
                            plan_id=plan_id,
                            step_id=step.id,
                            step_record_id=step_record_ids[step.id],
                            reason=step.error,
                        )
                break
            step = ready[0]
            await self._execute_with_retries(
                record,
                plan,
                plan_id,
                step,
                step_record_ids[step.id],
                previous_messages,
                memories,
                selected_skills,
                resume_step=resuming,
                retry_uncertain=retry_uncertain,
            )

        output = await self._finalize(record.id, record.goal, plan, resuming=resuming)
        success = all(step.status == "completed" for step in plan.steps)
        status: RunStatus = "completed" if success else "failed"
        self.memory.add_message(
            record.conversation_id,
            "assistant",
            output,
            {"run_id": record.id, "success": success, "plan": plan.as_dict()},
        )
        if self.config.agent.auto_memory:
            episode = f"Goal: {record.goal}\nOutcome: {output}"[:12_000]
            self.memory.remember(
                episode,
                kind="episode",
                tags=["task", "success" if success else "incomplete"],
                importance=0.6 if success else 0.7,
                source=record.conversation_id,
                memory_id=f"run-{record.id}",
            )
        self.execution.finish_run(record.id, status, output=output)
        usage = self.execution.usage(record.id)
        metrics = self.execution.metrics(record.id)
        self.events.emit(
            "agent.completed",
            run_id=record.id,
            conversation_id=record.conversation_id,
            status=status,
            success=success,
            output=output,
            usage=usage,
            metrics=metrics,
        )
        return AgentResult(
            output=output,
            conversation_id=record.conversation_id,
            plan=plan,
            success=success,
            usage=usage,
            run_id=record.id,
            status=status,
            metrics=metrics,
        )

    async def _create_or_recover_plan(
        self,
        run_id: str,
        goal: str,
        previous_messages: list[dict[str, Any]],
        memories: list[MemoryRecord],
        selected_skills: list[Skill],
        *,
        resuming: bool,
    ) -> TaskPlan:
        if not self.config.agent.planning:
            return self.planner.fallback_plan(goal)
        if resuming and (text := self.execution.latest_stage_text(run_id, "planner")):
            return self.planner.parse_response(goal, text)
        context = _planner_context(previous_messages, memories, selected_skills)
        request = self.planner.build_request(goal, context=context)
        response, _turn_id = await self._complete_model(
            run_id, "planner", request, step_id=None, attempt_id=None, turn=1
        )
        return self.planner.parse_response(goal, response.text)

    async def _execute_with_retries(
        self,
        record: RunRecord,
        plan: TaskPlan,
        plan_id: str,
        step: PlanStep,
        step_record_id: str,
        previous_messages: list[dict[str, Any]],
        memories: list[MemoryRecord],
        selected_skills: list[Skill],
        *,
        resume_step: bool,
        retry_uncertain: bool,
    ) -> None:
        starting_history: list[Any] | None = None
        if resume_step:
            state = self.execution.step_resume_state(
                step_record_id, retry_uncertain=retry_uncertain
            )
            if state.uncertain_tool and not retry_uncertain:
                raise UnsafeResumeError("The step contains an uncertain tool call")
            if state.completed_result is not None:
                step.result = state.completed_result
                step.status = "completed"
                self.execution.set_step_status(
                    step_record_id, "completed", result=step.result
                )
                self.events.emit(
                    "step.completed",
                    run_id=record.id,
                    plan_id=plan_id,
                    step_id=step.id,
                    step_record_id=step_record_id,
                    recovered=True,
                    result=step.result,
                )
                return
            starting_history = state.history

        attempts = self.config.agent.max_step_retries + 1
        last_error = ""
        for invocation_attempt in range(1, attempts + 1):
            attempt_id, attempt_number = self.execution.start_attempt(
                record.id, step_record_id
            )
            step.status = "running"
            self.events.emit(
                "step.started",
                run_id=record.id,
                plan_id=plan_id,
                step_id=step.id,
                step_record_id=step_record_id,
                title=step.title,
                attempt_id=attempt_id,
                attempt=attempt_number,
                attempts=attempts,
            )
            try:
                result = await self._execute_step(
                    record.id,
                    record.goal,
                    plan,
                    step,
                    step_record_id,
                    attempt_id,
                    previous_messages,
                    memories,
                    selected_skills,
                    starting_history=starting_history if invocation_attempt == 1 else None,
                )
                step.result = result
                step.status = "completed"
                self.execution.finish_attempt(attempt_id, "completed")
                self.execution.set_step_status(step_record_id, "completed", result=result)
                self.events.emit(
                    "step.completed",
                    run_id=record.id,
                    plan_id=plan_id,
                    step_id=step.id,
                    step_record_id=step_record_id,
                    attempt_id=attempt_id,
                    result=result,
                )
                return
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                self.execution.finish_attempt(attempt_id, "failed", last_error)
                self.events.emit(
                    "step.attempt_failed",
                    run_id=record.id,
                    plan_id=plan_id,
                    step_id=step.id,
                    step_record_id=step_record_id,
                    attempt_id=attempt_id,
                    error=last_error,
                )
        step.status = "failed"
        step.error = last_error
        self.execution.set_step_status(step_record_id, "failed", error=last_error)
        self.events.emit(
            "step.failed",
            run_id=record.id,
            plan_id=plan_id,
            step_id=step.id,
            step_record_id=step_record_id,
            error=last_error,
        )

    async def _execute_step(
        self,
        run_id: str,
        goal: str,
        plan: TaskPlan,
        step: PlanStep,
        step_record_id: str,
        attempt_id: str,
        previous_messages: list[dict[str, Any]],
        memories: list[MemoryRecord],
        selected_skills: list[Skill],
        *,
        starting_history: list[Any] | None,
    ) -> str:
        completed = [item for item in plan.steps if item.status == "completed"]
        history = (
            list(starting_history)
            if starting_history is not None
            else build_step_input(
                goal=goal,
                step=step,
                previous_messages=previous_messages,
                completed_steps=completed,
            )
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
        self.tool_context.services.update(
            {
                "run_id": run_id,
                "step_key": step.id,
                "step_record_id": step_record_id,
                "attempt_id": attempt_id,
            }
        )
        for turn in range(1, self.config.agent.max_turns + 1):
            response, turn_id = await self._complete_model(
                run_id,
                "executor",
                ModelRequest(
                    instructions=instructions,
                    input_items=history,
                    tools=self.tools.schemas(),
                    max_output_tokens=self.config.model.max_output_tokens,
                ),
                step_id=step_record_id,
                attempt_id=attempt_id,
                turn=turn,
                step_key=step.id,
            )
            self.tool_context.services["turn_id"] = turn_id
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

    async def _finalize(
        self, run_id: str, goal: str, plan: TaskPlan, *, resuming: bool
    ) -> str:
        completed = [step for step in plan.steps if step.status == "completed"]
        failed = [step for step in plan.steps if step.status != "completed"]
        if len(plan.steps) == 1 and completed and not failed:
            return completed[0].result
        if resuming and (text := self.execution.latest_stage_text(run_id, "finalizer")):
            return text
        execution = json.dumps(plan.as_dict(), ensure_ascii=False, indent=2)[:30_000]
        try:
            response, _turn_id = await self._complete_model(
                run_id,
                "finalizer",
                ModelRequest(
                    instructions=FINALIZER_PROMPT,
                    input_items=[
                        {
                            "role": "user",
                            "content": (
                                f"Original goal:\n{goal}\n\nExecution results:\n{execution}"
                            ),
                        }
                    ],
                    tools=[],
                    max_output_tokens=self.config.model.max_output_tokens,
                ),
                step_id=None,
                attempt_id=None,
                turn=1,
            )
            if response.text.strip():
                return response.text.strip()
        except Exception as exc:
            self.events.emit(
                "finalizer.failed", run_id=run_id, error=f"{type(exc).__name__}: {exc}"
            )
        lines = [step.result for step in completed if step.result]
        lines.extend(f"{step.title}: {step.error or step.status}" for step in failed)
        return "\n\n".join(lines) or "The task produced no result."

    async def _complete_model(
        self,
        run_id: str,
        stage: str,
        request: ModelRequest,
        *,
        step_id: str | None,
        attempt_id: str | None,
        turn: int,
        step_key: str | None = None,
    ) -> tuple[ModelResponse, str]:
        request_payload = {
            "instructions": request.instructions,
            "input_items": _portable(request.input_items),
            "tools": _portable(request.tools),
            "max_output_tokens": request.max_output_tokens,
        }
        turn_id, sequence = self.execution.start_turn(
            run_id,
            stage,
            request_payload,
            step_id=step_id,
            attempt_id=attempt_id,
        )
        self.events.emit(
            "model.started",
            run_id=run_id,
            stage=stage,
            step_id=step_key,
            step_record_id=step_id,
            attempt_id=attempt_id,
            turn_id=turn_id,
            turn=turn,
            sequence=sequence,
        )
        started = time.perf_counter()
        try:
            response = await self.provider.complete(request)
        except asyncio.CancelledError:
            status = "cancelled" if self._cancel_requested else "interrupted"
            duration_ms = round((time.perf_counter() - started) * 1000)
            self.execution.finish_turn(turn_id, status, duration_ms=duration_ms)
            self.events.emit(
                f"model.{status}", run_id=run_id, stage=stage, turn_id=turn_id
            )
            raise
        except Exception as exc:
            duration_ms = round((time.perf_counter() - started) * 1000)
            error = f"{type(exc).__name__}: {exc}"
            self.execution.finish_turn(
                turn_id, "failed", duration_ms=duration_ms, error=error
            )
            self.events.emit(
                "model.failed",
                run_id=run_id,
                stage=stage,
                step_id=step_key,
                step_record_id=step_id,
                attempt_id=attempt_id,
                turn_id=turn_id,
                error=error,
            )
            raise
        duration_ms = round((time.perf_counter() - started) * 1000)
        response_payload = {
            "text": response.text,
            "tool_calls": [
                {"id": call.id, "name": call.name, "arguments": call.arguments}
                for call in response.tool_calls
            ],
            "output_items": _portable(response.output_items),
        }
        self.execution.finish_turn(
            turn_id,
            "completed",
            response=response_payload,
            response_id=response.response_id,
            usage=response.usage,
            duration_ms=duration_ms,
        )
        self.events.emit(
            "model.completed",
            run_id=run_id,
            stage=stage,
            step_id=step_key,
            step_record_id=step_id,
            attempt_id=attempt_id,
            turn_id=turn_id,
            turn=turn,
            response_id=response.response_id,
            tool_calls=[call.name for call in response.tool_calls],
            text=response.text,
            usage=response.usage,
            duration_ms=duration_ms,
        )
        return response, turn_id

    def _begin_active(self, run_id: str, conversation_id: str) -> None:
        if self._active_run_id is not None:
            raise RuntimeError(f"Agent is already running {self._active_run_id}")
        self._active_run_id = run_id
        self._cancel_requested = False
        self.tool_context.services.update(
            {"run_id": run_id, "conversation_id": conversation_id}
        )

    def _interrupt_active(self, run_id: str) -> None:
        status: Literal["cancelled", "interrupted"] = (
            "cancelled" if self._cancel_requested else "interrupted"
        )
        message = "Task cancelled by the user" if self._cancel_requested else "Task interrupted"
        self.execution.interrupt_run(run_id, status, message)
        self.events.emit(f"agent.{status}", run_id=run_id, status=status, error=message)

    def _fail_active(self, run_id: str, exc: Exception) -> None:
        error = f"{type(exc).__name__}: {exc}"
        self.execution.finish_run(run_id, "failed", error=error)
        self.events.emit("agent.failed", run_id=run_id, status="failed", error=error)

    def _end_active(self) -> None:
        self._active_run_id = None
        self._cancel_requested = False
        for key in (
            "run_id",
            "conversation_id",
            "step_key",
            "step_record_id",
            "attempt_id",
            "turn_id",
        ):
            self.tool_context.services.pop(key, None)


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


def _context_snapshot(
    messages: list[dict[str, Any]],
    memories: list[MemoryRecord],
    skills: list[Skill],
    *,
    config: AxiomConfig,
) -> dict[str, Any]:
    return {
        "model": {"provider": config.model.provider, "name": config.model.name},
        "workspace": str(config.workspace.root),
        "previous_messages": _portable(messages),
        "memories": [asdict(item) for item in memories],
        "skills": [
            {
                "name": item.name,
                "description": item.description,
                "instructions": item.instructions,
                "path": str(item.path),
            }
            for item in skills
        ],
    }


def _restore_context(
    context: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[MemoryRecord], list[Skill]]:
    messages = [
        dict(item) for item in context.get("previous_messages", []) if isinstance(item, dict)
    ]
    memories = [
        MemoryRecord(**item) for item in context.get("memories", []) if isinstance(item, dict)
    ]
    skills = [
        Skill(
            name=str(item.get("name", "")),
            description=str(item.get("description", "")),
            instructions=str(item.get("instructions", "")),
            path=Path(str(item.get("path", "."))),
        )
        for item in context.get("skills", [])
        if isinstance(item, dict)
    ]
    return messages, memories, skills


def _portable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _portable(value.model_dump(exclude_none=True))
    if isinstance(value, dict):
        return {str(key): _portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)
