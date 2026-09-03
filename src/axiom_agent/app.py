from __future__ import annotations

from axiom_agent.agent import Agent
from axiom_agent.config import AxiomConfig
from axiom_agent.events import EventBus, JsonlEventLogger
from axiom_agent.mcp.client import MCPManager
from axiom_agent.memory.store import SQLiteMemoryStore
from axiom_agent.memory.tools import memory_tools
from axiom_agent.providers.base import ModelProvider
from axiom_agent.providers.factory import create_provider
from axiom_agent.skills.loader import SkillRegistry
from axiom_agent.tools.base import ApprovalCallback, ToolContext, ToolRegistry
from axiom_agent.tools.filesystem import filesystem_tools
from axiom_agent.tools.shell import ShellTool


class AxiomApp:
    def __init__(
        self,
        config: AxiomConfig,
        *,
        provider: ModelProvider | None = None,
        approve: ApprovalCallback | None = None,
    ) -> None:
        self.config = config
        if not config.workspace.root.is_dir():
            raise FileNotFoundError(f"Workspace does not exist: {config.workspace.root}")
        self.events = EventBus()
        self.events.subscribe(JsonlEventLogger(config.workspace.root / ".axiom/events.jsonl"))
        self.memory = SQLiteMemoryStore(config.memory.path)
        self.skills = SkillRegistry.discover(config.skills.paths)
        self.provider = provider or create_provider(config.model)
        self.tools = ToolRegistry()
        self.tools.register_many(filesystem_tools())
        self.tools.register(ShellTool(config.workspace))
        self.tools.register_many(memory_tools(self.memory))
        self.tools.register(self.skills.tool())
        self.tool_context = ToolContext(
            workspace=config.workspace.root,
            events=self.events,
            approval_mode=config.workspace.approval,
            approve=approve,
        )
        self.mcp = MCPManager(config.mcp, self.events)
        self.agent = Agent(
            config=config,
            provider=self.provider,
            tools=self.tools,
            memory=self.memory,
            skills=self.skills,
            events=self.events,
            tool_context=self.tool_context,
        )
        self._started = False
        self._closed = False

    async def start(self) -> AxiomApp:
        if self._closed:
            raise RuntimeError("AxiomApp cannot be restarted after close")
        if not self._started:
            try:
                await self.mcp.connect(self.tools)
            except Exception:
                await self.provider.close()
                self.memory.close()
                self._closed = True
                raise
            self._started = True
        return self

    async def close(self) -> None:
        if self._closed:
            return
        await self.mcp.close()
        await self.provider.close()
        self.memory.close()
        self._started = False
        self._closed = True

    async def __aenter__(self) -> AxiomApp:
        return await self.start()

    async def __aexit__(self, *_args: object) -> None:
        await self.close()
