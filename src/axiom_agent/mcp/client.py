from __future__ import annotations

import re
from contextlib import AsyncExitStack
from typing import Any

from axiom_agent.config import MCPConfig, MCPServerConfig
from axiom_agent.events import EventBus
from axiom_agent.tools.base import Tool, ToolContext, ToolRegistry


class MCPTool(Tool):
    parallel_safe = False

    def __init__(
        self,
        *,
        exposed_name: str,
        remote_name: str,
        description: str,
        parameters: dict[str, Any],
        session: Any,
        server_name: str,
    ) -> None:
        self.name = exposed_name
        self.remote_name = remote_name
        self.description = description
        self.parameters = parameters
        self.session = session
        self.server_name = server_name

    async def run(self, arguments: dict[str, Any], _context: ToolContext) -> dict[str, Any]:
        result = await self.session.call_tool(self.remote_name, arguments=arguments)
        content: list[Any] = []
        for item in getattr(result, "content", []) or []:
            if hasattr(item, "model_dump"):
                content.append(item.model_dump(exclude_none=True))
            elif hasattr(item, "text"):
                content.append({"type": "text", "text": item.text})
            else:
                content.append(str(item))
        structured = getattr(result, "structuredContent", None)
        if structured is None:
            structured = getattr(result, "structured_content", None)
        return {
            "server": self.server_name,
            "tool": self.remote_name,
            "is_error": bool(getattr(result, "isError", getattr(result, "is_error", False))),
            "content": content,
            "structured_content": structured,
        }


class MCPManager:
    """Connects configured MCP servers and exposes their tools through one registry."""

    def __init__(self, config: MCPConfig, events: EventBus) -> None:
        self.config = config
        self.events = events
        self._stack = AsyncExitStack()
        self._sessions: dict[str, Any] = {}
        self._connected = False

    async def connect(self, registry: ToolRegistry) -> None:
        if self._connected:
            return
        self._connected = True
        for server in self.config.servers:
            try:
                session = await self._connect_server(server)
                response = await session.list_tools()
                count = 0
                for remote_tool in response.tools:
                    remote_name = str(remote_tool.name)
                    exposed_name = _tool_name(server, remote_name)
                    schema = getattr(remote_tool, "inputSchema", None)
                    if schema is None:
                        schema = getattr(remote_tool, "input_schema", None)
                    if hasattr(schema, "model_dump"):
                        schema = schema.model_dump(exclude_none=True)
                    registry.register(
                        MCPTool(
                            exposed_name=exposed_name,
                            remote_name=remote_name,
                            description=(
                                f"MCP server {server.name}: "
                                f"{getattr(remote_tool, 'description', '') or remote_name}"
                            ),
                            parameters=schema or {"type": "object", "properties": {}},
                            session=session,
                            server_name=server.name,
                        )
                    )
                    count += 1
                self.events.emit("mcp.connected", server=server.name, tools=count)
            except Exception as exc:
                self.events.emit(
                    "mcp.failed", server=server.name, error=f"{type(exc).__name__}: {exc}"
                )
                await self.close()
                raise RuntimeError(f"Could not connect MCP server {server.name!r}: {exc}") from exc

    async def _connect_server(self, server: MCPServerConfig) -> Any:
        try:
            from mcp import Client, StdioServerParameters
        except ImportError as exc:  # pragma: no cover - depends on installation
            raise RuntimeError("MCP support requires the 'mcp' Python package") from exc

        if server.transport == "stdio":
            if not server.command:
                raise ValueError("stdio MCP server requires command")
            from mcp.client.stdio import stdio_client

            parameters = StdioServerParameters(
                command=server.command,
                args=server.args,
                env=server.env or None,
            )
            transport = stdio_client(parameters)
        elif server.transport == "streamable_http":
            if not server.url:
                raise ValueError("streamable_http MCP server requires url")
            from mcp.client.streamable_http import streamable_http_client

            transport = await self._http_transport(
                streamable_http_client, server.url, server.headers
            )
        elif server.transport == "sse":
            if not server.url:
                raise ValueError("sse MCP server requires url")
            from mcp.client.sse import sse_client

            transport = sse_client(server.url, headers=server.headers or None)
        else:
            raise ValueError(f"Unsupported MCP transport: {server.transport}")

        client = await self._stack.enter_async_context(Client(transport))
        self._sessions[server.name] = client
        return client

    async def _http_transport(
        self, factory: Any, url: str, headers: dict[str, str]
    ) -> Any:
        if not headers:
            return factory(url)
        try:
            import httpx2
        except ImportError as exc:  # pragma: no cover - installed with MCP v2
            raise RuntimeError("Authenticated MCP HTTP transports require 'httpx2'") from exc
        http_client = await self._stack.enter_async_context(httpx2.AsyncClient(headers=headers))
        return factory(url, http_client=http_client)

    async def close(self) -> None:
        if self._connected:
            await self._stack.aclose()
            self._sessions.clear()
            self._connected = False

    def status(self) -> list[dict[str, Any]]:
        return [
            {
                "name": server.name,
                "transport": server.transport,
                "endpoint": server.command or server.url,
                "connected": server.name in self._sessions,
            }
            for server in self.config.servers
        ]


def _tool_name(server: MCPServerConfig, remote_name: str) -> str:
    raw = f"mcp__{server.name}__{remote_name}" if server.tool_prefix else remote_name
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", raw)
    if len(sanitized) <= 64:
        return sanitized
    suffix = str(abs(hash(raw)))[:8]
    return f"{sanitized[:55]}_{suffix}"
