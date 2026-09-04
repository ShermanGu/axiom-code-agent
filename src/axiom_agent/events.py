from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Event:
    type: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


EventHandler = Callable[[Event], None]


class EventBus:
    """Tiny synchronous event bus used for UI, logs, and observability adapters."""

    def __init__(self) -> None:
        self._handlers: list[EventHandler] = []

    def subscribe(self, handler: EventHandler) -> None:
        self._handlers.append(handler)

    def emit(self, event_type: str, **data: Any) -> Event:
        event = Event(event_type, data)
        for handler in tuple(self._handlers):
            handler(event)
        return event


class JsonlEventLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def __call__(self, event: Event) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = _sanitize(asdict(event))
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


SECRET_KEY = re.compile(r"api.?key|authorization|password|secret|access.?token", re.IGNORECASE)
SECRET_TEXT = re.compile(
    r"(?i)(bearer\s+)[a-z0-9._~+/-]+=*|"
    r"((?:api[_-]?key|password|secret|access[_-]?token)\s*[:=]\s*)[^\s,;]+|"
    r"\bsk-[a-z0-9_-]{12,}\b"
)


def _sanitize(value: Any, *, key: str = "") -> Any:
    if SECRET_KEY.search(key):
        return "***REDACTED***"
    if isinstance(value, dict):
        return {
            str(item_key): _sanitize(item, key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize(item) for item in value]
    if isinstance(value, str):
        redacted = SECRET_TEXT.sub(
            lambda match: (match.group(1) or match.group(2) or "") + "***REDACTED***",
            value,
        )
        if len(redacted) > 20_000:
            return redacted[:10_000] + "\n... event field truncated ...\n" + redacted[-10_000:]
        return redacted
    return value
