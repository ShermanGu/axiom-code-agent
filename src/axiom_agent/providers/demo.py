from __future__ import annotations

import json
from uuid import uuid4

from axiom_agent.providers.base import ModelProvider
from axiom_agent.types import ModelRequest, ModelResponse, ToolCall


class DemoProvider(ModelProvider):
    """Deterministic provider that exercises planning, tools, and memory offline."""

    async def complete(self, request: ModelRequest) -> ModelResponse:
        joined = "\n".join(str(item) for item in request.input_items)
        if "Return only valid JSON" in request.instructions:
            text = json.dumps(
                {
                    "strategy": "Inspect the workspace and report what the agent can see.",
                    "steps": [
                        {
                            "id": "inspect",
                            "title": "Inspect workspace",
                            "description": "List the files in the current workspace.",
                            "depends_on": [],
                        }
                    ],
                }
            )
            return ModelResponse(text=text, output_items=[{"role": "assistant", "content": text}])

        if "function_call_output" not in joined:
            call = ToolCall(id=f"demo_{uuid4().hex[:8]}", name="fs_list", arguments={"path": "."})
            return ModelResponse(
                text="",
                tool_calls=[call],
                output_items=[
                    {
                        "type": "function_call",
                        "call_id": call.id,
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    }
                ],
            )

        text = "Offline demo completed: I inspected the workspace through Axiom's tool loop."
        return ModelResponse(text=text, output_items=[{"role": "assistant", "content": text}])

