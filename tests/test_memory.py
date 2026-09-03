from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from axiom_agent.memory.store import SQLiteMemoryStore


class MemoryTests(unittest.TestCase):
    def test_conversation_and_hybrid_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteMemoryStore(Path(directory) / "memory.db")
            conversation = store.create_conversation("build agent")
            store.add_message(conversation, "user", "Please use Python")
            store.add_message(conversation, "assistant", "Understood")
            messages = store.recent_messages(conversation)
            self.assertEqual([item["role"] for item in messages], ["user", "assistant"])

            python_memory = store.remember(
                "用户偏好使用 Python 构建代码代理",
                kind="preference",
                tags=["python", "代码代理"],
                importance=0.9,
            )
            store.remember("The deployment region is Frankfurt", kind="fact", importance=0.4)
            results = store.search("Python 代码代理", limit=1)
            self.assertEqual(results[0].id, python_memory.id)
            self.assertGreater(results[0].score, 0)
            store.close()

    def test_forget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteMemoryStore(Path(directory) / "memory.db")
            item = store.remember("temporary")
            self.assertTrue(store.forget(item.id))
            self.assertFalse(store.forget(item.id))
            store.close()


if __name__ == "__main__":
    unittest.main()

