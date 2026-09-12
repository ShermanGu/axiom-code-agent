from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from axiom_agent.app import AxiomApp
from axiom_agent.config import AxiomConfig
from axiom_agent.events import Event
from axiom_agent.execution.store import ExecutionStore, SQLiteEventLogger, UnsafeResumeError
from axiom_agent.providers.base import ModelProvider
from axiom_agent.types import ModelRequest, ModelResponse, PlanStep, TaskPlan, ToolCall


def _response(text: str, tokens: int = 1) -> ModelResponse:
    return ModelResponse(
        text=text,
        output_items=[{"role": "assistant", "content": text}],
        response_id=f"response-{tokens}",
        usage={"input_tokens": tokens, "output_tokens": tokens, "total_tokens": tokens * 2},
    )


class _MetricsProvider(ModelProvider):
    async def complete(self, request: ModelRequest) -> ModelResponse:
        instructions = request.instructions.casefold()
        if "task planner" in instructions:
            return _response(
                json.dumps(
                    {
                        "strategy": "two stages",
                        "steps": [
                            {"id": "one", "title": "One", "description": "First"},
                            {
                                "id": "two",
                                "title": "Two",
                                "description": "Second",
                                "depends_on": ["one"],
                            },
                        ],
                    }
                ),
                1,
            )
        if "result synthesizer" in instructions:
            return _response("finalized", 4)
        current = "\n".join(str(item) for item in request.input_items)
        return _response("step two" if "Second" in current else "step one", 2)


class _InterruptedProvider(ModelProvider):
    async def complete(self, request: ModelRequest) -> ModelResponse:
        if "task planner" in request.instructions.casefold():
            return _response(
                json.dumps(
                    {
                        "strategy": "write once",
                        "steps": [
                            {
                                "id": "write",
                                "title": "Write",
                                "description": "Create one marker file",
                                "depends_on": [],
                            }
                        ],
                    }
                )
            )
        if any(
            isinstance(item, dict) and item.get("type") == "function_call_output"
            for item in request.input_items
        ):
            raise asyncio.CancelledError
        call = ToolCall(
            "write-once",
            "fs_write",
            {"path": "once.txt", "content": "created once\n", "mode": "create"},
        )
        return ModelResponse(
            text="",
            tool_calls=[call],
            output_items=[
                {
                    "type": "function_call",
                    "call_id": call.id,
                    "name": call.name,
                    "arguments": json.dumps(call.arguments),
                }
            ],
            usage={"total_tokens": 2},
        )


class _ResumeProvider(ModelProvider):
    def __init__(self) -> None:
        self.saw_tool_output = False

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if "task planner" in request.instructions.casefold():
            raise AssertionError("A checkpointed plan must not be recreated")
        self.saw_tool_output = any(
            isinstance(item, dict) and item.get("type") == "function_call_output"
            for item in request.input_items
        )
        if not self.saw_tool_output:
            raise AssertionError("Completed tool output was not restored")
        return _response("resumed without replay", 3)


class _CancellableProvider(ModelProvider):
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def complete(self, _request: ModelRequest) -> ModelResponse:
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class _FailingProvider(ModelProvider):
    async def complete(self, request: ModelRequest) -> ModelResponse:
        instructions = request.instructions.casefold()
        if "task planner" in instructions:
            return _response(
                json.dumps(
                    {
                        "steps": [
                            {"id": "fail", "title": "Fail", "description": "Fail"},
                            {
                                "id": "dependent",
                                "title": "Dependent",
                                "description": "Skip",
                                "depends_on": ["fail"],
                            },
                        ]
                    }
                )
            )
        if "result synthesizer" in instructions:
            return _response("failure summarized")
        raise RuntimeError("executor failed")


class ExecutionTests(unittest.TestCase):
    def test_run_persists_hierarchy_events_and_stage_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            config = AxiomConfig()
            config.workspace.root = workspace
            config.resolve_paths(workspace)

            async def scenario() -> None:
                async with AxiomApp(config, provider=_MetricsProvider()) as app:
                    result = await app.agent.run("Perform two stages")
                    self.assertTrue(result.success)
                    self.assertEqual(result.status, "completed")
                    self.assertEqual(result.metrics["planner"]["model_calls"], 1)
                    self.assertEqual(result.metrics["executor"]["model_calls"], 2)
                    self.assertEqual(result.metrics["finalizer"]["model_calls"], 1)
                    self.assertEqual(result.metrics["total"]["total_tokens"], 18)
                    detail = app.execution.detail(result.run_id)
                    self.assertEqual(len(detail["step_record_ids"]), 2)
                    self.assertEqual(len(set(detail["step_record_ids"].values())), 2)
                    self.assertEqual(len(detail["attempts"]), 2)
                    self.assertEqual(len(detail["turns"]), 4)
                    events = app.execution.event_rows(result.run_id)
                    self.assertIn("agent.completed", [event["type"] for event in events])

            asyncio.run(scenario())

    def test_resume_continues_after_completed_tool_without_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            config = AxiomConfig()
            config.workspace.root = workspace
            config.resolve_paths(workspace)

            async def scenario() -> None:
                async with AxiomApp(config, provider=_InterruptedProvider()) as app:
                    with self.assertRaises(asyncio.CancelledError):
                        await app.agent.run("Create a marker")
                    interrupted = app.execution.list_runs(1)[0]
                    self.assertEqual(interrupted.status, "interrupted")
                    run_id = interrupted.id

                resumed_provider = _ResumeProvider()
                async with AxiomApp(config, provider=resumed_provider) as app:
                    result = await app.agent.resume(run_id)
                    self.assertTrue(result.success)
                    self.assertTrue(resumed_provider.saw_tool_output)
                    detail = app.execution.detail(run_id)
                    self.assertEqual(len(detail["tool_calls"]), 1)
                    self.assertEqual(
                        [attempt["status"] for attempt in detail["attempts"]],
                        ["interrupted", "completed"],
                    )
                self.assertEqual(
                    (workspace / "once.txt").read_text(encoding="utf-8"), "created once\n"
                )

            asyncio.run(scenario())

    def test_uncertain_tool_call_blocks_automatic_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ExecutionStore(Path(directory) / "runs.db")
            try:
                run_id = store.create_run("conversation", "goal", {})
                _plan_id, step_ids = store.save_plan(
                    run_id,
                    TaskPlan("goal", [PlanStep("write", "Write", "Write a file")]),
                )
                step_id = step_ids["write"]
                attempt_id, _number = store.start_attempt(run_id, step_id)
                turn_id, _sequence = store.start_turn(
                    run_id,
                    "executor",
                    {"input_items": [{"role": "user", "content": "write"}]},
                    step_id=step_id,
                    attempt_id=attempt_id,
                )
                store.start_tool_call(
                    run_id=run_id,
                    step_id=step_id,
                    attempt_id=attempt_id,
                    turn_id=turn_id,
                    provider_call_id="unknown-outcome",
                    name="fs_write",
                    arguments={"path": "file.txt"},
                )
                store.interrupt_run(run_id, "interrupted", "process stopped")

                with self.assertRaises(UnsafeResumeError):
                    store.prepare_resume(run_id)
                self.assertEqual(store.get_run(run_id).status, "blocked")
                resumed = store.prepare_resume(run_id, retry_uncertain=True)
                self.assertEqual(resumed.status, "running")
                state = store.step_resume_state(step_id, retry_uncertain=True)
                self.assertTrue(state.uncertain_tool)
                self.assertEqual(state.history, [{"role": "user", "content": "write"}])
            finally:
                store.close()

    def test_sqlite_event_log_redacts_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ExecutionStore(Path(directory) / "runs.db")
            try:
                run_id = store.create_run("conversation", "goal", {})
                SQLiteEventLogger(store)(
                    Event("tool.started", {"run_id": run_id, "password": "private"})
                )
                rendered = json.dumps(store.event_rows(run_id))
                self.assertNotIn("private", rendered)
                self.assertIn("REDACTED", rendered)
            finally:
                store.close()

    def test_explicit_cancellation_is_distinct_from_interruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            config = AxiomConfig()
            config.workspace.root = workspace
            config.resolve_paths(workspace)
            provider = _CancellableProvider()

            async def scenario() -> None:
                async with AxiomApp(config, provider=provider) as app:
                    task = asyncio.create_task(app.agent.run("Wait"))
                    await provider.started.wait()
                    app.agent.request_cancel()
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                    self.assertEqual(app.execution.list_runs(1)[0].status, "cancelled")

            asyncio.run(scenario())

    def test_failed_step_and_dependency_skip_are_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            config = AxiomConfig()
            config.workspace.root = workspace
            config.agent.max_step_retries = 0
            config.resolve_paths(workspace)

            async def scenario() -> None:
                async with AxiomApp(config, provider=_FailingProvider()) as app:
                    result = await app.agent.run("Fail then skip")
                    self.assertFalse(result.success)
                    self.assertEqual(result.status, "failed")
                    detail = app.execution.detail(result.run_id)
                    statuses = [step["status"] for step in detail["plan"]["steps"]]
                    self.assertEqual(statuses, ["failed", "skipped"])
                    self.assertEqual(detail["attempts"][0]["status"], "failed")

            asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
