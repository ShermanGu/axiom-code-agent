from __future__ import annotations

import asyncio
import json
import unittest
from types import SimpleNamespace

from axiom_agent.config import ModelConfig
from axiom_agent.providers.openai import OpenAIChatProvider, OpenAIResponsesProvider
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

    def test_chat_function_call_and_tool_output_are_translated(self) -> None:
        captured: dict[str, object] = {}

        def create(**kwargs):
            captured.update(kwargs)
            function = SimpleNamespace(name="fs_read", arguments='{"path":"README.md"}')
            call = SimpleNamespace(id="chat_call_1", function=function)
            message = SimpleNamespace(content="", tool_calls=[call])
            usage = SimpleNamespace(model_dump=lambda **_kwargs: {"prompt_tokens": 10})
            return SimpleNamespace(
                id="chat_1",
                choices=[SimpleNamespace(message=message)],
                usage=usage,
            )

        completions = SimpleNamespace(create=create)
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        config = ModelConfig(provider="groq", name="qwen/qwen3.6-27b")
        provider = OpenAIChatProvider(config, client=client)

        async def scenario() -> None:
            result = await provider.complete(
                ModelRequest(
                    instructions="test",
                    input_items=[
                        {"role": "user", "content": "read"},
                        {
                            "type": "function_call_output",
                            "call_id": "previous_call",
                            "output": "done",
                        },
                    ],
                    tools=[
                        {
                            "type": "function",
                            "name": "fs_read",
                            "description": "Read a file",
                            "parameters": {"type": "object"},
                            "strict": False,
                        }
                    ],
                )
            )
            self.assertEqual(result.tool_calls[0].arguments, {"path": "README.md"})
            self.assertEqual(result.output_items[0]["tool_calls"][0]["id"], "chat_call_1")
            messages = captured["messages"]
            self.assertEqual(messages[0]["role"], "system")
            self.assertEqual(messages[-1]["role"], "tool")
            self.assertEqual(messages[-1]["tool_call_id"], "previous_call")
            tool = captured["tools"][0]
            self.assertEqual(tool["function"]["name"], "fs_read")
            self.assertEqual(captured["max_tokens"], 8192)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
