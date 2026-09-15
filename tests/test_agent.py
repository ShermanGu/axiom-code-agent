from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from axiom_agent.app import AxiomApp
from axiom_agent.config import load_config
from axiom_agent.providers.base import ModelProvider
from axiom_agent.providers.demo import DemoProvider
from axiom_agent.tools.base import FunctionTool
from axiom_agent.types import ModelRequest, ModelResponse


class _CapabilityAwareProvider(ModelProvider):
    def __init__(self) -> None:
        self.planner_request: ModelRequest | None = None

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if "task planner" in request.instructions.casefold():
            self.planner_request = request
            return ModelResponse(
                text=json.dumps(
                    {
                        "strategy": "Use available mailbox data.",
                        "steps": [
                            {
                                "id": "mail",
                                "title": "Read mail",
                                "description": "Read today's messages.",
                                "depends_on": [],
                                "required_capabilities": ["read mailbox data"],
                                "candidate_tools": ["mcp__mail__search_email"],
                            }
                        ],
                    }
                ),
                output_items=[],
            )
        return ModelResponse(
            text="Mail summarized.",
            output_items=[{"role": "assistant", "content": "Mail summarized."}],
        )


class AgentEndToEndTests(unittest.TestCase):
    def test_planner_receives_registered_tool_capabilities_without_callable_tools(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            config = load_config(workspace=workspace)
            provider = _CapabilityAwareProvider()

            async def scenario() -> None:
                async with AxiomApp(config, provider=provider) as app:
                    app.tools.register(
                        FunctionTool(
                            "mcp__mail__search_email",
                            "Search mailbox messages",
                            {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                            },
                            lambda _arguments, _context: "unused",
                        )
                    )
                    result = await app.agent.run("Summarize today's mail")

                self.assertIsNotNone(provider.planner_request)
                assert provider.planner_request is not None
                self.assertEqual(provider.planner_request.tools, [])
                self.assertIn(
                    "mcp__mail__search_email",
                    str(provider.planner_request.input_items),
                )
                self.assertEqual(
                    result.plan.steps[0].candidate_tools,
                    ["mcp__mail__search_email"],
                )

            asyncio.run(scenario())

    def test_offline_plan_tool_memory_loop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "hello.txt").write_text("hello", encoding="utf-8")
            config = load_config(workspace=workspace)
            config.model.provider = "demo"

            async def scenario() -> None:
                async with AxiomApp(config, provider=DemoProvider()) as app:
                    event_types: list[str] = []
                    app.events.subscribe(lambda event: event_types.append(event.type))
                    result = await app.agent.run("Inspect workspace")
                    self.assertTrue(result.success)
                    self.assertEqual(result.plan.steps[0].status, "completed")
                    self.assertIn("tool.started", event_types)
                    self.assertIn("agent.completed", event_types)
                    episodes = app.memory.list_memories(kind="episode")
                    self.assertEqual(len(episodes), 1)
                    self.assertIn("Inspect workspace", episodes[0].content)

            asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
