from __future__ import annotations

from typing import Any

from axiom_agent.memory.store import SQLiteMemoryStore
from axiom_agent.tools.base import FunctionTool, Tool, ToolContext


def memory_tools(store: SQLiteMemoryStore) -> list[Tool]:
    async def search(arguments: dict[str, Any], _context: ToolContext) -> dict[str, Any]:
        records = store.search(
            str(arguments["query"]),
            limit=max(1, min(int(arguments.get("limit", 8)), 30)),
            kinds=list(arguments.get("kinds", [])) or None,
        )
        return {
            "memories": [
                {
                    "id": item.id,
                    "kind": item.kind,
                    "content": item.content,
                    "tags": item.tags,
                    "importance": item.importance,
                    "score": round(item.score, 4),
                }
                for item in records
            ]
        }

    async def remember(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        record = store.remember(
            str(arguments["content"]),
            kind=str(arguments.get("kind", "fact")),
            tags=[str(tag) for tag in arguments.get("tags", [])],
            importance=float(arguments.get("importance", 0.5)),
            source=str(context.services.get("conversation_id", "agent")),
        )
        return {"id": record.id, "kind": record.kind, "stored": True}

    base = {"type": "object", "additionalProperties": False}
    return [
        FunctionTool(
            "memory_search",
            "Search durable memories from earlier tasks, including facts, decisions, "
            "and preferences.",
            {
                **base,
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 30},
                    "kinds": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["query"],
            },
            search,
            parallel_safe=True,
        ),
        FunctionTool(
            "memory_remember",
            "Store a durable fact, user preference, constraint, or decision for future tasks.",
            {
                **base,
                "properties": {
                    "content": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": ["fact", "preference", "decision", "procedure", "episode"],
                    },
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "importance": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["content"],
            },
            remember,
        ),
    ]
