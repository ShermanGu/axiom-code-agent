from __future__ import annotations

import asyncio
import json
import unittest
from types import SimpleNamespace

from axiom_agent.config import ModelConfig
from axiom_agent.providers.openai import OpenAIResponsesProvider
from axiom_agent.types import ModelRequest


class _Responses:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        call = SimpleNamespace(
            type="function_call",
            call_id="call_1",
            name="fs_read",
            arguments=json.dumps({"path": "README.md"}),
        )
        usage = SimpleNamespace(model_dump=lambda **_kwargs: {"input_tokens": 12})
        return SimpleNamespace(
            id="resp_1", output=[call], output_text="", usage=usage
        )


class OpenAIProviderTests(unittest.TestCase):
    def test_responses_function_call_is_translated(self) -> None:
        responses = _Responses()
        client = SimpleNamespace(responses=responses)
        provider = OpenAIResponsesProvider(ModelConfig(), client=client)

        async def scenario() -> None:
            result = await provider.complete(
                ModelRequest(
                    instructions="test",
                    input_items=[{"role": "user", "content": "read"}],
                    tools=[{"type": "function", "name": "fs_read"}],
                )
            )
            self.assertEqual(result.response_id, "resp_1")
            self.assertEqual(result.tool_calls[0].name, "fs_read")
            self.assertEqual(result.tool_calls[0].arguments, {"path": "README.md"})
            self.assertEqual(result.usage["input_tokens"], 12)
            self.assertFalse(responses.kwargs["store"])
            self.assertTrue(responses.kwargs["parallel_tool_calls"])

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()

