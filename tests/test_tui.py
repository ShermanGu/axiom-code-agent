from __future__ import annotations

import asyncio
import inspect
import unittest
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from axiom_agent.cli import build_parser
from axiom_agent.config import AxiomConfig
from axiom_agent.events import EventBus
from axiom_agent.tools.base import ApprovalCallback
from axiom_agent.tui import ApprovalScreen, AxiomTUI, PromptArea


async def _wait_for(
    pilot: Any,
    condition: Callable[[], bool],
    description: str,
    *,
    timeout: float = 5.0,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError(f"Timed out waiting for {description}")
        await pilot.pause(0.05)


class _FakeAgent:
    def __init__(self, events: EventBus, *, require_approval: bool = False) -> None:
        self.events = events
        self.require_approval = require_approval
        self.approve: ApprovalCallback | None = None
        self.goals: list[tuple[str, str | None]] = []
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


class _FakeBackend:
    def __init__(self, *, require_approval: bool = False) -> None:
        self.events = EventBus()
        self.agent = _FakeAgent(self.events, require_approval=require_approval)
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
                self.assertTrue(await pilot.click("#allow"))
                await _wait_for(
                    pilot,
                    lambda: backend.agent.approved is True and not app.busy,
                    "the approved task to finish",
                )
                self.assertTrue(backend.agent.approved)
                self.assertFalse(app.busy)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
