from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from axiom_agent.events import Event, JsonlEventLogger


class EventLoggerTests(unittest.TestCase):
    def test_common_secrets_are_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            logger = JsonlEventLogger(path)
            logger(
                Event(
                    "tool.started",
                    {
                        "authorization": "Bearer private-token",
                        "command": "API_KEY=super-secret curl example.com",
                    },
                )
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            rendered = json.dumps(payload)
            self.assertNotIn("private-token", rendered)
            self.assertNotIn("super-secret", rendered)
            self.assertIn("REDACTED", rendered)


if __name__ == "__main__":
    unittest.main()
