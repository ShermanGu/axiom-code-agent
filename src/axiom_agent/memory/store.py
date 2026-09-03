from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

TOKEN_PATTERN = re.compile(r"[\w-]+", re.UNICODE)
HAN_PATTERN = re.compile(r"[\u3400-\u9fff]+")


@dataclass(slots=True)
class MemoryRecord:
    id: str
    kind: str
    content: str
    tags: list[str]
    importance: float
    created_at: str
    updated_at: str
    access_count: int = 0
    score: float = 0.0
    source: str = ""

    @classmethod
    def from_row(cls, row: sqlite3.Row, *, score: float = 0.0) -> MemoryRecord:
        return cls(
            id=row["id"],
            kind=row["kind"],
            content=row["content"],
            tags=json.loads(row["tags"] or "[]"),
            importance=float(row["importance"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            access_count=int(row["access_count"]),
            source=row["source"] or "",
            score=score,
        )


class SQLiteMemoryStore:
    """Auditable local memory: conversations, durable memories, and events.

    Retrieval is deliberately dependency-free. It combines token/character
    overlap, importance, recency, and prior access. A vector database can be
    swapped in later without changing the agent-facing API.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._initialize()

    def _initialize(self) -> None:
        with self._lock, self._connection:
            self._connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA foreign_keys=ON;
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    goal TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                    ON messages(conversation_id, id DESC);
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tags TEXT NOT NULL DEFAULT '[]',
                    importance REAL NOT NULL DEFAULT 0.5,
                    source TEXT NOT NULL DEFAULT '',
                    access_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);
                CREATE INDEX IF NOT EXISTS idx_memories_updated ON memories(updated_at DESC);
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT,
                    type TEXT NOT NULL,
                    data TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                """
            )

    def create_conversation(self, goal: str = "", conversation_id: str | None = None) -> str:
        identifier = conversation_id or uuid4().hex
        now = _now()
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO conversations(id, goal, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    goal = CASE
                        WHEN excluded.goal = '' THEN conversations.goal
                        ELSE excluded.goal
                    END,
                    updated_at = excluded.updated_at
                """,
                (identifier, goal, now, now),
            )
        return identifier

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.create_conversation(conversation_id=conversation_id)
        now = _now()
        with self._lock, self._connection:
            self._connection.execute(
                """INSERT INTO messages(conversation_id, role, content, metadata, created_at)
                VALUES (?, ?, ?, ?, ?)""",
                (
                    conversation_id,
                    role,
                    content,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    now,
                ),
            )
            self._connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id)
            )

    def recent_messages(self, conversation_id: str, limit: int = 12) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """SELECT role, content, metadata, created_at FROM messages
                WHERE conversation_id = ? ORDER BY id DESC LIMIT ?""",
                (conversation_id, limit),
            ).fetchall()
        return [
            {
                "role": row["role"],
                "content": row["content"],
                "metadata": json.loads(row["metadata"] or "{}"),
                "created_at": row["created_at"],
            }
            for row in reversed(rows)
        ]

    def remember(
        self,
        content: str,
        *,
        kind: str = "fact",
        tags: list[str] | None = None,
        importance: float = 0.5,
        source: str = "",
        memory_id: str | None = None,
    ) -> MemoryRecord:
        identifier = memory_id or uuid4().hex
        now = _now()
        importance = max(0.0, min(float(importance), 1.0))
        serialized_tags = json.dumps(tags or [], ensure_ascii=False)
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO memories(
                    id, kind, content, tags, importance,
                    source, access_count, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    kind = excluded.kind,
                    content = excluded.content,
                    tags = excluded.tags,
                    importance = excluded.importance,
                    source = excluded.source,
                    updated_at = excluded.updated_at
                """,
                (identifier, kind, content, serialized_tags, importance, source, now, now),
            )
            row = self._connection.execute(
                "SELECT * FROM memories WHERE id = ?", (identifier,)
            ).fetchone()
        assert row is not None
        return MemoryRecord.from_row(row)

    def search(
        self,
        query: str,
        *,
        limit: int = 8,
        kinds: list[str] | None = None,
    ) -> list[MemoryRecord]:
        parameters: list[Any] = []
        where = ""
        if kinds:
            placeholders = ",".join("?" for _ in kinds)
            where = f"WHERE kind IN ({placeholders})"
            parameters.extend(kinds)
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM memories {where} ORDER BY updated_at DESC LIMIT 2000", parameters
            ).fetchall()

        query_features = _features(query)
        now = datetime.now(UTC)
        ranked: list[MemoryRecord] = []
        for row in rows:
            haystack = f"{row['content']} {' '.join(json.loads(row['tags'] or '[]'))}"
            features = _features(haystack)
            overlap = _jaccard(query_features, features)
            exact = 1.0 if query.casefold() in haystack.casefold() and query else 0.0
            updated = datetime.fromisoformat(row["updated_at"])
            age_days = max(0.0, (now - updated).total_seconds() / 86400)
            recency = math.exp(-age_days / 45.0)
            importance = float(row["importance"])
            access = min(math.log1p(int(row["access_count"])) / 5.0, 1.0)
            score = (
                overlap * 0.58
                + exact * 0.17
                + importance * 0.13
                + recency * 0.09
                + access * 0.03
            )
            if query_features and overlap == 0 and exact == 0:
                score *= 0.2
            ranked.append(MemoryRecord.from_row(row, score=score))
        ranked.sort(key=lambda item: (item.score, item.updated_at), reverse=True)
        selected = ranked[: max(0, limit)]
        if selected:
            with self._lock, self._connection:
                self._connection.executemany(
                    "UPDATE memories SET access_count = access_count + 1 WHERE id = ?",
                    [(item.id,) for item in selected],
                )
        return selected

    def list_memories(self, limit: int = 50, kind: str | None = None) -> list[MemoryRecord]:
        if kind:
            sql = "SELECT * FROM memories WHERE kind = ? ORDER BY updated_at DESC LIMIT ?"
            parameters: tuple[Any, ...] = (kind, limit)
        else:
            sql = "SELECT * FROM memories ORDER BY updated_at DESC LIMIT ?"
            parameters = (limit,)
        with self._lock:
            rows = self._connection.execute(sql, parameters).fetchall()
        return [MemoryRecord.from_row(row) for row in rows]

    def forget(self, memory_id: str) -> bool:
        with self._lock, self._connection:
            cursor = self._connection.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        return cursor.rowcount > 0

    def record_event(
        self,
        event_type: str,
        data: dict[str, Any],
        conversation_id: str | None = None,
    ) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO events(conversation_id, type, data, created_at) VALUES (?, ?, ?, ?)",
                (
                    conversation_id,
                    event_type,
                    json.dumps(data, ensure_ascii=False, default=str),
                    _now(),
                ),
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _features(text: str) -> set[str]:
    normalized = text.casefold()
    features = {token for token in TOKEN_PATTERN.findall(normalized) if len(token) > 1}
    for chunk in HAN_PATTERN.findall(normalized):
        features.update(chunk[index : index + 2] for index in range(max(1, len(chunk) - 1)))
        features.update(chunk)
    return features


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
