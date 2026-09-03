from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from axiom_agent.app import AxiomApp
from axiom_agent.config import load_config
from axiom_agent.providers.demo import DemoProvider


class AgentEndToEndTests(unittest.TestCase):
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
