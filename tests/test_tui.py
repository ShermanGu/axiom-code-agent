from __future__ import annotations

import asyncio
import inspect
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from textual.widgets import Button, Static

from axiom_agent.cli import build_parser
from axiom_agent.config import AxiomConfig
from axiom_agent.events import Event, EventBus
from axiom_agent.execution.store import ExecutionStore
from axiom_agent.tools.base import ApprovalCallback
from axiom_agent.tui import ApprovalScreen, AxiomTUI, PromptArea, RunHistoryScreen


async def _wait_for(
    pilot: Any,
    condition: Callable[[], bool],
    description: str,
    *,
    timeout_seconds: float = 5.0,
) -> None:
    try:
        async with asyncio.timeout(timeout_seconds):
            while not condition():
                await pilot.pause(0.05)
    except TimeoutError as exc:
        raise AssertionError(f"Timed out waiting for {description}") from exc


class _FakeAgent:
    def __init__(self, events: EventBus, *, require_approval: bool = False) -> None:
        self.events = events
        self.require_approval = require_approval
        self.approve: ApprovalCallback | None = None
        self.goals: list[tuple[str, str | None]] = []
        self.resumes: list[tuple[str, bool]] = []
        self.approved: bool | None = None

    async def run(self, goal: str, *, conversation_id: str | None = None) -> Any:
        self.goals.append((goal, conversation_id))
        self.events.emit("plan.created", plan={"steps": [{"title": "Inspect"}]})
        self.events.emit("step.started", step_id="step-1", title="Inspect")
        if self.require_approval and self.approve is not None:
            decision = self.approve("Remove-Item example.txt", "The command can delete data")
            resolved = await decision if inspect.isawaitable(decision) else decision
            self.approved = bool(resolved)
        self.events.emit("tool.started", name="fs_read")
        self.events.emit("tool.completed", name="fs_read")
        self.events.emit("step.completed", step_id="step-1")
        return SimpleNamespace(
            output=f"Completed: {goal}",
            conversation_id="conversation-1",
            success=True,
            usage={"input_tokens": 12},
        )

    async def resume(self, run_id: str, *, retry_uncertain_tools: bool = False) -> Any:
        self.resumes.append((run_id, retry_uncertain_tools))
        self.events.emit("agent.resumed", run_id=run_id)
        return SimpleNamespace(
            output=f"Recovered: {run_id[:12]}",
            conversation_id="conversation-recovered",
            success=True,
            usage={"input_tokens": 4},
        )


class _FakeBackend:
    def __init__(
        self,
        *,
        require_approval: bool = False,
        execution: ExecutionStore | None = None,
    ) -> None:
        self.events = EventBus()
        self.agent = _FakeAgent(self.events, require_approval=require_approval)
        self.execution = execution
        self.closed = False

    async def start(self) -> _FakeBackend:
        self.events.emit("mcp.connected", server="test", tools=2)
        return self

    async def close(self) -> None:
        self.closed = True


class TUITests(unittest.TestCase):
    def test_cli_parser_accepts_tui_options(self) -> None:
        arguments = build_parser().parse_args(["tui", "--no-plan", "--yes"])
        self.assertEqual(arguments.command, "tui")
        self.assertTrue(arguments.no_plan)
        self.assertTrue(arguments.yes)

    def test_prompt_runs_in_persistent_conversation(self) -> None:
        async def scenario() -> None:
            backend = _FakeBackend()

            def factory(_config: AxiomConfig, approve: ApprovalCallback) -> _FakeBackend:
                backend.agent.approve = approve
                return backend

            app = AxiomTUI(AxiomConfig(), backend_factory=factory)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                prompt = app.query_one("#prompt", PromptArea)
                self.assertFalse(prompt.disabled)
                prompt.load_text("inspect this workspace")
                prompt.action_submit()
                await _wait_for(
                    pilot,
                    lambda: not app.busy and len(list(app.query(".assistant"))) == 1,
                    "the completed assistant response",
                )
                self.assertEqual(backend.agent.goals, [("inspect this workspace", None)])
                self.assertEqual(app.conversation_id, "conversation-1")
                self.assertEqual(len(list(app.query(".assistant"))), 1)
                self.assertFalse(app.busy)
            self.assertTrue(backend.closed)

        asyncio.run(scenario())

    def test_approval_modal_returns_user_choice(self) -> None:
        async def scenario() -> None:
            backend = _FakeBackend(require_approval=True)

            def factory(_config: AxiomConfig, approve: ApprovalCallback) -> _FakeBackend:
                backend.agent.approve = approve
                return backend

            app = AxiomTUI(AxiomConfig(), backend_factory=factory)
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                prompt = app.query_one("#prompt", PromptArea)
                prompt.load_text("delete the example")
                prompt.action_submit()
                await _wait_for(
                    pilot,
                    lambda: isinstance(app.screen, ApprovalScreen)
                    and len(list(app.screen.query("#allow"))) == 1,
                    "the approval modal to mount",
                )
                self.assertIsInstance(app.screen, ApprovalScreen)
                allow = app.screen.query_one("#allow", Button)
                allow.post_message(Button.Pressed(allow))
                await _wait_for(
                    pilot,
                    lambda: backend.agent.approved is True and not app.busy,
                    "the approved task to finish",
                )
                self.assertTrue(backend.agent.approved)
                self.assertFalse(app.busy)

        asyncio.run(scenario())

    def test_run_history_shows_details_exports_and_resumes(self) -> None:
        async def scenario() -> None:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                store = ExecutionStore(root / ".axiom" / "runs.db")
                run_id = store.create_run("conversation-old", "Recover an interrupted task", {})
                store.record_event(Event("agent.started", {"run_id": run_id}))
                store.interrupt_run(run_id, "interrupted", "Task interrupted")
                backend = _FakeBackend(execution=store)
                config = AxiomConfig()
                config.workspace.root = root

                def factory(_config: AxiomConfig, approve: ApprovalCallback) -> _FakeBackend:
                    backend.agent.approve = approve
                    return backend

                app = AxiomTUI(config, backend_factory=factory)
                async with app.run_test(size=(140, 44)) as pilot:
                    await _wait_for(pilot, lambda: app.ready, "TUI startup")
                    await pilot.click("#runs")
                    await _wait_for(
                        pilot,
                        lambda: isinstance(app.screen, RunHistoryScreen),
                        "run history screen",
                    )
                    screen = app.screen
                    self.assertIsInstance(screen, RunHistoryScreen)
                    detail = screen.query_one("#run-detail", Static)
                    self.assertIn("Recover an interrupted task", str(detail.content))
                    self.assertFalse(screen.query_one("#run-resume", Button).disabled)

                    await pilot.click("#run-export")
                    output = root / ".axiom" / "exports" / f"{run_id}-events.jsonl"
                    self.assertTrue(output.is_file())
                    self.assertIn("agent.started", output.read_text(encoding="utf-8"))

                    await pilot.click("#run-resume")
                    await _wait_for(
                        pilot,
                        lambda: backend.agent.resumes == [(run_id, False)] and not app.busy,
                        "run recovery",
                    )
                    self.assertEqual(app.conversation_id, "conversation-recovered")
                    await pilot.press("ctrl+r")
                    await _wait_for(
                        pilot,
                        lambda: isinstance(app.screen, RunHistoryScreen),
                        "Ctrl+R run history shortcut",
                    )
                    await pilot.press("escape")
                store.close()

        asyncio.run(scenario())

    def test_blocked_run_requires_confirmation_before_retry(self) -> None:
        async def scenario() -> None:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                store = ExecutionStore(root / ".axiom" / "runs.db")
                run_id = store.create_run("conversation-blocked", "Retry uncertain tool", {})
                store.finish_run(run_id, "blocked", error="Tool outcome is unknown")
                backend = _FakeBackend(execution=store)
                config = AxiomConfig()
                config.workspace.root = root

                def factory(_config: AxiomConfig, approve: ApprovalCallback) -> _FakeBackend:
                    backend.agent.approve = approve
                    return backend

                app = AxiomTUI(config, auto_approve=True, backend_factory=factory)
                async with app.run_test(size=(140, 44)) as pilot:
                    await _wait_for(pilot, lambda: app.ready, "TUI startup")
                    await pilot.click("#runs")
                    await _wait_for(
                        pilot,
                        lambda: isinstance(app.screen, RunHistoryScreen),
                        "run history screen",
                    )
                    screen = app.screen
                    self.assertIsInstance(screen, RunHistoryScreen)
                    self.assertTrue(screen.query_one("#run-resume", Button).disabled)
                    self.assertFalse(screen.query_one("#run-retry", Button).disabled)
                    await pilot.click("#run-retry")
                    await _wait_for(
                        pilot,
                        lambda: isinstance(app.screen, ApprovalScreen),
                        "uncertain retry confirmation",
                    )
                    self.assertEqual(backend.agent.resumes, [])
                    await pilot.click("#allow")
                    await _wait_for(
                        pilot,
                        lambda: backend.agent.resumes == [(run_id, True)] and not app.busy,
                        "confirmed blocked-run retry",
                    )
                store.close()

        asyncio.run(scenario())

    def test_slash_commands_offer_direct_export_and_resume(self) -> None:
        async def scenario() -> None:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                store = ExecutionStore(root / ".axiom" / "runs.db")
                run_id = store.create_run("conversation-command", "Resume by command", {})
                store.record_event(Event("agent.started", {"run_id": run_id}))
                store.interrupt_run(run_id, "interrupted", "Task interrupted")
                backend = _FakeBackend(execution=store)
                config = AxiomConfig()
                config.workspace.root = root

                def factory(_config: AxiomConfig, approve: ApprovalCallback) -> _FakeBackend:
                    backend.agent.approve = approve
                    return backend

                app = AxiomTUI(config, backend_factory=factory)
                async with app.run_test(size=(120, 40)) as pilot:
                    await _wait_for(pilot, lambda: app.ready, "TUI startup")
                    prompt = app.query_one("#prompt", PromptArea)
                    prompt.load_text(f"/show {run_id[:12]}")
                    prompt.action_submit()
                    await _wait_for(
                        pilot,
                        lambda: isinstance(app.screen, RunHistoryScreen),
                        "slash-command run details",
                    )
                    await pilot.press("escape")

                    prompt.load_text(f"/export {run_id[:12]}")
                    prompt.action_submit()
                    output = root / ".axiom" / "exports" / f"{run_id}-events.jsonl"
                    await _wait_for(pilot, output.is_file, "slash-command event export")

                    prompt.load_text(f"/resume {run_id[:12]}")
                    prompt.action_submit()
                    await _wait_for(
                        pilot,
                        lambda: backend.agent.resumes == [(run_id, False)] and not app.busy,
                        "slash-command recovery",
                    )
                store.close()

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
