# Axiom architecture

## Design principles

1. **One explicit state machine.** Plans, step status, turns, retries, tool observations, and final
   synthesis are visible data rather than hidden framework behavior.
2. **Ports at volatile boundaries.** Models, tools, MCP, memory, planning, approvals, and UI events
   can evolve independently.
3. **Progressive context disclosure.** A goal gets a small set of retrieved memories and selected
   skills. The model can load another skill or search memory during execution.
4. **Local-first auditability.** Conversation state lives in SQLite; lifecycle events live in JSONL;
   OpenAI responses default to `store=false`.
5. **Errors become observations.** An individual tool error is returned to the model. A failed step
   is retried, then represented in the plan so dependent work can be skipped honestly.

## Runtime lifecycle

`AxiomApp` is the composition root. It creates one provider, memory store, skill registry, tool
registry, MCP manager, approval context, event bus, planner, and agent.

For each `Agent.run(goal)` call:

1. Create or reopen a conversation and persist the user goal.
2. Retrieve recent conversation messages and ranked durable memories.
3. Select matching skills, favoring explicit `$skill-name` mentions.
4. Ask the planner for JSON. Validate IDs and dependencies; fall back to one safe step on malformed
   output.
5. Pick a ready step, construct bounded context, and call the provider with the unified tool schemas.
6. Execute returned tools. Parallelize only when every requested tool is marked `parallel_safe`.
7. Append native function-call outputs to the model input and continue until the step returns text.
8. Retry failures, mark terminal status, and skip steps whose dependencies did not complete.
9. Return a one-step result directly or synthesize multiple results without tools.
10. Persist the assistant answer, a durable episode, and lifecycle events.

## Core contracts

- `ModelProvider.complete(ModelRequest) -> ModelResponse`
- `Tool.run(arguments, ToolContext) -> JSON-compatible value`
- `TaskPlan` / `PlanStep` for scheduler interchange
- `SQLiteMemoryStore` conversation and durable-memory methods
- `EventBus.emit(type, data)` for observers

The internal model input-item representation follows the Responses API because it preserves native
function-call continuity. Non-OpenAI adapters translate these items at their boundary.

## Memory layers

- **Working memory:** current model input items, tool calls, and tool outputs.
- **Short-term memory:** recent user/assistant messages per conversation in SQLite.
- **Long-term memory:** facts, preferences, decisions, procedures, and task episodes.
- **Operational memory:** JSONL events for debugging, evaluation, replay tooling, and cost tracking.

Durable retrieval currently ranks up to 2,000 recent candidates locally. A future vector adapter can
retain the same `search` return type and blend semantic distance into the score.

## Extension examples

### Custom provider

```python
from axiom_agent.providers.base import ModelProvider

class MyProvider(ModelProvider):
    async def complete(self, request):
        ...

def create(config):
    return MyProvider(config)
```

Then set `provider = "my_package.provider:create"`.

### Custom event observer

```python
app.events.subscribe(lambda event: send_to_telemetry(event.type, event.data))
```

### Alternate schedulers

The plan already carries dependency edges. A scheduler can take all `ready_steps()` and execute
independent analysis steps concurrently, while serializing tools that mutate the shared workspace.

