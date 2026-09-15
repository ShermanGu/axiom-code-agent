# Axiom

> [!WARNING]
> **Work in progress (WIP).** Axiom is under active development. Interfaces, configuration, and
> behavior may change between minor releases. Use it on non-critical workspaces first, keep changes
> under version control, and review tool approvals carefully.

Axiom is a modular Python code agent that can plan a task, operate on a workspace, call local or
remote MCP tools, load task-specific skills, and remember useful context across sessions.

It is intentionally built around a small state machine instead of a large agent framework. The
model provider, planner, tools, MCP transport, skill router, memory store, approval policy, and event
observers are separate boundaries, so each can be replaced without rewriting the execution loop.

## What works now

- OpenAI Responses API plus Groq, Gemini, OpenRouter, and generic Chat Completions adapters
- Capability-aware task decomposition with dependency-aware step execution and retries
- Workspace-scoped file listing, reading, search, creation, and exact replacement
- Bounded shell execution with network and destructive-command policy checks
- MCP client support for stdio, Streamable HTTP, and SSE servers
- `SKILL.md` discovery, automatic routing, explicit `$skill-name` activation, and lazy loading
- SQLite short-term conversation history, durable memories, hybrid local retrieval, and task episodes
- Durable SQLite checkpoints for task executions, plans, steps, attempts, model turns, and tool calls
- Safe interrupted-task recovery, explicit terminal states, and per-stage usage/latency metrics
- SQLite plus JSONL lifecycle events for model calls, tools, plans, steps, MCP, and outcomes
- Interactive CLI and TUI, offline demo, diagnostics, memory inspection, and isolated tests

## Quick start

Requires Python 3.11 or newer.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
axiom init .
$env:OPENAI_API_KEY = "your-key"
axiom doctor
axiom run "inspect this repository, run its tests, and explain the highest-risk issue"
```

No API key is needed for the full offline smoke test:

```powershell
axiom demo
```

Inside Codex Desktop on Windows, the bundled Python runtime can launch Axiom without installing
anything first:

```powershell
powershell -ExecutionPolicy Bypass -File .\axiom-local.ps1 demo
```

On macOS/Linux, activate with `source .venv/bin/activate` and export the key with
`export OPENAI_API_KEY=...`.

## CLI

```text
axiom init [path]                  create .axiom/config.toml
axiom run "goal"                  execute one task
axiom chat                        keep a persistent conversation
axiom chat --conversation ID      continue an existing conversation
axiom tui                         open the full-screen terminal interface
axiom demo                        run planning -> tool -> memory offline
axiom demo recovery               verify durable recovery behavior offline
axiom eval                        run deterministic offline regression scenarios
axiom doctor                      inspect the local setup
axiom skills [list]               list discovered skills concisely
axiom skills search "query"       search skill names and descriptions
axiom skills show SKILL_NAME      show one skill's source and instructions
axiom mcp                         list configured MCP servers
axiom memory list                 inspect durable memory
axiom memory search "query"       test memory retrieval
axiom memory forget MEMORY_ID     delete one memory
axiom conversations               list durable conversation history
axiom conversations show ID       inspect a conversation's tasks and metrics
axiom conversations export ID     export one conversation's events as JSONL
axiom resume ID                    resume its latest interrupted or failed task
```

Use `--no-plan` for a direct single-step run. `--yes` approves commands that the configured policy
would otherwise ask about; it does not override the network-disable setting or workspace path guard.

### Terminal interface

Run the interactive interface in PowerShell, Windows Terminal, or another modern terminal:

```powershell
axiom tui
```

Use `Enter` or `Ctrl+Enter` to send, `Ctrl+J` to insert a new line, `Escape` to stop the active task,
`Ctrl+N` for a new conversation, `Ctrl+R` to open conversation history, `Ctrl+L` to clear the
visible transcript, and `Ctrl+Q` to exit. Planning, steps, MCP connections, and tool calls appear in
the activity pane.
Commands requiring approval open a modal confirmation; `axiom tui --yes` automatically approves
ordinary policy-gated commands. Starting a new conversation clears the transcript and activity pane.

The **Conversations** dialog groups the current workspace's durable history by conversation ID. It
shows every task in the selected conversation plus lifecycle and Planner/Executor/Finalizer metrics.
**Continue conversation** restores its transcript and context; when its latest task is incomplete,
**Resume task** recovers that task from its checkpoint. You can also explicitly retry a
blocked task or export the conversation's sanitized events. The same actions have direct commands:

```text
/conversations                 open conversation history
/show CONVERSATION_ID          open one conversation's details
/continue CONVERSATION_ID      restore it and continue with a new prompt
/resume CONVERSATION_ID        recover its latest incomplete task
/retry CONVERSATION_ID         confirm and retry an uncertain tool outcome
/export CONVERSATION_ID        export events under .axiom/exports/
```

Uncertain-tool retry always opens a dedicated side-effect warning, even with `axiom tui --yes`.

The TUI updates task state and tool activity in real time. Model text is displayed when each model
request completes; token-by-token streaming is not yet implemented.

### Durable conversations and task recovery

Every conversation receives a stable conversation ID. Each submitted prompt creates a separate,
internal task execution so its plan, metrics, and recovery state remain isolated. Users only need
the conversation ID; use the first 12 characters shown by `axiom conversations` anywhere one is
accepted:

```powershell
axiom conversations
axiom conversations show 4f2a09c71d6e
axiom chat --conversation 4f2a09c71d6e
axiom resume 4f2a09c71d6e
axiom conversations export 4f2a09c71d6e --output conversation-events.jsonl
```

Run the complete P0-2 acceptance demo without a model key, MCP server, or carefully timed manual
interruption:

```powershell
axiom demo recovery
```

It verifies that checkpoints persist, completed tools are not replayed, uncertain tool outcomes
block automatic recovery, explicit retry succeeds, and the resulting history and metrics remain
inspectable. It also leaves both demo conversations in the selected workspace, so you can
immediately run `axiom tui` and open **Conversations** (or press `Ctrl+R`) to inspect and export them.

Completed steps and recorded tool outputs are restored rather than replayed. If Axiom stopped while
a tool was running, its outcome may be unknown; the conversation's latest task becomes `blocked`
instead of silently repeating the operation. After inspecting the workspace and
`axiom conversations show`, replay that step only when acceptable:

```powershell
axiom resume 4f2a09c71d6e --retry-uncertain-tools
```

Tasks and steps distinguish `failed`, `cancelled`, `interrupted`, `skipped`, and `blocked`. Planner,
Executor, and Finalizer model calls, token usage, failures, and latency are reported separately by
`axiom conversations show` and `--json` output.

## Configuration

Copy [`axiom.example.toml`](axiom.example.toml) to `axiom.toml`, or run `axiom init` to create
`.axiom/config.toml`. Axiom searches upward from the current directory for either location.

The default model is `gpt-5.6-terra`; change it without editing the file by setting `AXIOM_MODEL`.
The OpenAI adapter uses the Responses API with custom function tools and `store=false`. The adapter
replays response output items and function outputs so the core does not require server-side
conversation storage. See the official [Responses API reference](https://developers.openai.com/api/reference/cli/resources/responses/methods/create).

When a model endpoint must bypass system proxy settings, set a comma-separated list of URLs or
hostnames in the model configuration:

```toml
[model]
provider = "openai-chat"
name = "your-model"
base_url = "https://llm.internal.example/v1"
api_key_env = "LLM_API_KEY"
# api_key = "your-key"  # Optional local fallback when LLM_API_KEY is unset.
no_proxy = "llm.internal.example"
```

Use `provider = "openai"` for endpoints that implement the Responses API, and `openai-chat` for
OpenAI-compatible Chat Completions endpoints. `base_url` can also be overridden with
`AXIOM_BASE_URL`. Environment-variable credentials take precedence over `api_key`. A plaintext key
should only be stored in the generated `.axiom/config.toml`, which `axiom init` adds to `.gitignore`;
never commit a key in `axiom.toml` or another tracked file.

`no_proxy` is merged into both `NO_PROXY` and `no_proxy` when configuration is loaded, while the
model client also receives explicit direct-routing rules. Existing environment exclusions are
preserved. Full endpoint URLs are reduced to their host (and optional port) for the environment.

### Free-tier model providers

Axiom has presets for three OpenAI-compatible services. Select one with `AXIOM_PROVIDER`; its
endpoint, recommended model, and credential-variable name are filled in automatically.

| Provider | Default model | Credential variable | Notes |
| --- | --- | --- | --- |
| `groq` | `qwen/qwen3.6-27b` | `GROQ_API_KEY` | Recommended for the first run; local tool use and parallel calls |
| `gemini` | `gemini-3.1-flash-lite` | `GEMINI_API_KEY` | Free input/output tier; free-tier content may improve Google products |
| `openrouter` | `openrouter/free` | `OPENROUTER_API_KEY` | Automatically routes to a currently free compatible model |

Example using Groq:

```powershell
$env:GROQ_API_KEY = "your-key"
$env:AXIOM_PROVIDER = "groq"
.\axiom-local.ps1 run "inspect this repository and explain its architecture"
```

The optional overrides are `AXIOM_MODEL`, `AXIOM_BASE_URL`, `AXIOM_API_KEY_ENV`, and
`AXIOM_MAX_OUTPUT_TOKENS`. Provider keys belong in environment variables, never in tracked TOML.

### MCP

```toml
[[mcp.servers]]
name = "project-data"
transport = "stdio"
command = "python"
args = ["-m", "my_mcp_server"]
tool_prefix = true

[[mcp.servers]]
name = "remote-tools"
transport = "streamable_http"
url = "https://example.com/mcp"
headers = { Authorization = "Bearer ${MCP_TOKEN}" }
```

MCP tools appear to the model as `mcp__SERVER__TOOL`, preventing collisions between servers. Values
inside `env` and `headers` support `$NAME`/`${NAME}` environment expansion, so secrets do not need to
be committed to TOML.

For a standalone stdio server executable, set `command` to its absolute path. `args` is optional;
omitting it and writing `args = []` are equivalent. Axiom keeps the server's diagnostic stream on a
real console handle so stdio MCP startup also works from the Textual TUI on Windows.

The adapter targets the current stable MCP Python SDK v2 and uses its high-level `Client`. Streamable
HTTP is the production transport; SSE remains available for older servers. See the official
[MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk).

### Skills

A skill is a directory containing `SKILL.md`:

```markdown
---
name: migration-review
description: Review database migrations for rollback and data-loss risk.
---

# Migration review

Inspect both upgrade and downgrade paths...
```

Put skills under `.axiom/skills`, add more directories to `skills.paths`, or explicitly invoke one
in a goal with `$migration-review`. Axiom injects only selected skills; the full catalog remains
available through `skill_activate`.

Use `axiom skills` for a concise catalog, `axiom skills search "review"` to filter it, and
`axiom skills show migration-review` to inspect one package's source path and full instructions.

## Architecture

```text
User goal
   |
   +--> recent conversation + durable-memory retrieval + skill routing
   |
Planner --> dependency-aware TaskPlan
   |
Executor (step/turn/retry state machine)
   |        ^
   v        |
Model adapter <--> unified ToolRegistry
                    |-- workspace file tools
                    |-- policy-gated shell
                    |-- memory tools
                    |-- skill loader
                    `-- MCP server tools
   |
Final synthesis --> conversation history + durable episode + SQLite/JSONL events
```

The Planner receives a compact catalog containing tool names, descriptions, and argument names, but
no callable tools. It records required capabilities and optional candidate-tool hints in each step;
the Executor receives the complete schemas and remains responsible for every actual tool call.

Read [`docs/architecture.md`](docs/architecture.md) for lifecycle and extension points, and
[`docs/security.md`](docs/security.md) before granting an agent access to sensitive repositories.

## Replaceable components

- **Model:** implement `ModelProvider.complete`, or set `model.provider` to `module:factory`.
- **Tool:** subclass `Tool`, provide a JSON schema, and register it in `AxiomApp`.
- **Memory:** preserve the `SQLiteMemoryStore` method contract behind the `Agent` constructor.
- **Execution:** `ExecutionStore` owns durable checkpoints, event history, and model accounting.
- **Planning:** replace `Planner` while returning the same `TaskPlan` data type.
- **Interface:** subscribe to `EventBus` for a TUI, web UI, OpenTelemetry, or eval harness.
- **Scheduler:** the current executor runs ready steps sequentially. Independent read-only steps can
  later be dispatched in parallel without changing the plan format.

## Roadmap

Axiom will continue to improve in small, reviewable releases. Near-term priorities are:

- token-by-token model streaming in the TUI
- stronger sandbox and permission backends
- parallel execution for independent, read-only plan steps
- richer evals, trace inspection, and transactional file changes
- packaging and cross-platform installation polish

Issues and focused pull requests are welcome; see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Current boundaries

Axiom v0.5 is a strong local foundation, not an OS sandbox. File tools enforce a resolved workspace
boundary, but a command deliberately given to the shell runs with the current user's permissions.
High-risk command matching is defense in depth, not a security boundary. Run untrusted agents in a
container or disposable VM and keep `allow_network = false` unless the task requires it.

The default long-term retrieval is local and dependency-free; it combines token/Chinese-character
overlap, exact matches, recency, importance, and access frequency. For very large memory collections,
replace it with an embedding and vector-index adapter.

## Development

```powershell
python -m unittest discover -s tests -v
python -m compileall -q src
python -m ruff check src tests
python -m mypy src/axiom_agent
```

The offline end-to-end test covers planner output, an actual tool call, tool-result feedback, final
completion, event emission, conversation persistence, and durable episode creation.

Maintainers can also run `axiom eval` from any directory. The packaged core suite exercises eight
deterministic runtime scenarios and writes a JSON report under `.axiom/evals/`. See
[`evals/README.md`](evals/README.md) and the tracked [`roadmap`](docs/roadmap.md).
