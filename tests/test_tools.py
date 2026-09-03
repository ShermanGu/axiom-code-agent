from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from axiom_agent.events import EventBus
from axiom_agent.tools.base import ToolContext
from axiom_agent.tools.filesystem import fs_read, fs_replace, fs_write, resolve_workspace_path


class FilesystemToolTests(unittest.TestCase):
    def test_path_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            with self.assertRaises(PermissionError):
                resolve_workspace_path(workspace, "../outside.txt")

    def test_write_read_and_exact_replace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            context = ToolContext(workspace=workspace, events=EventBus())

            async def scenario() -> None:
                await fs_write(
                    {"path": "src/example.py", "content": "value = 1\n", "mode": "create"},
                    context,
                )
                read = await fs_read({"path": "src/example.py"}, context)
                self.assertIn("value = 1", read["content"])
                await fs_replace(
                    {"path": "src/example.py", "old": "value = 1", "new": "value = 2"},
                    context,
                )
                self.assertEqual((workspace / "src/example.py").read_text(), "value = 2\n")
                with self.assertRaises(ValueError):
                    await fs_replace(
                        {"path": "src/example.py", "old": "missing", "new": "x"}, context
                    )

            asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()

