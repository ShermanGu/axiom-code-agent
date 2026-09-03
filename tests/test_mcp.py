from __future__ import annotations

import asyncio
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from axiom_agent.config import MCPConfig, MCPServerConfig
from axiom_agent.events import EventBus
from axiom_agent.mcp.client import MCPManager
from axiom_agent.tools.base import ToolContext, ToolRegistry
from axiom_agent.types import ToolCall


class _Transport:
    pass


class _Client:
    closed = False

    def __init__(self, transport):
        self.transport = transport

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        type(self).closed = True

    async def list_tools(self):
        tool = SimpleNamespace(
            name="echo",
            description="Echo text",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        )
        return SimpleNamespace(tools=[tool])

    async def call_tool(self, name, arguments):
        content = SimpleNamespace(
            model_dump=lambda **_kwargs: {"type": "text", "text": arguments["text"]}
        )
        return SimpleNamespace(content=[content], structured_content=None, is_error=False)


class _StdioServerParameters:
    def __init__(self, command, args, env):
        self.command = command
        self.args = args
        self.env = env


class MCPManagerTests(unittest.TestCase):
    def test_v2_client_discovery_registration_and_call(self) -> None:
        mcp_module = types.ModuleType("mcp")
        mcp_module.__path__ = []
        mcp_module.Client = _Client
        mcp_module.StdioServerParameters = _StdioServerParameters
        client_module = types.ModuleType("mcp.client")
        client_module.__path__ = []
        stdio_module = types.ModuleType("mcp.client.stdio")
        stdio_module.stdio_client = lambda _parameters: _Transport()
        modules = {
            "mcp": mcp_module,
            "mcp.client": client_module,
            "mcp.client.stdio": stdio_module,
        }

        async def scenario() -> None:
            events = EventBus()
            seen: list[str] = []
            events.subscribe(lambda event: seen.append(event.type))
            manager = MCPManager(
                MCPConfig(
                    servers=[
                        MCPServerConfig(name="demo", command="fake-server", transport="stdio")
                    ]
                ),
                events,
            )
            registry = ToolRegistry()
            await manager.connect(registry)
            self.assertIn("mcp__demo__echo", registry.names())
            result = await registry.execute(
                ToolCall("call_1", "mcp__demo__echo", {"text": "hello"}),
                ToolContext(Path.cwd(), events),
            )
            self.assertFalse(result.is_error)
            self.assertIn("hello", result.output)
            self.assertIn("mcp.connected", seen)
            await manager.close()
            self.assertTrue(_Client.closed)

        with patch.dict(sys.modules, modules):
            asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
