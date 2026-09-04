# Axiom

Axiom is a modular Python code agent that can plan a task, operate on a workspace, call local or
remote MCP tools, load task-specific skills, and remember useful context across sessions.

It is intentionally built around a small state machine instead of a large agent framework. The
model provider, planner, tools, MCP transport, skill router, memory store, approval policy, and event
observers are separate boundaries, so each can be replaced without rewriting the execution loop.

## What works now

- OpenAI Responses API plus Groq, Gemini, OpenRouter, and generic Chat Completions adapters
- Structured task decomposition with dependency-aware step execution and retries
- Workspace-scoped file listing, reading, search, creation, and exact replacement
- Bounded shell execution with network and destructive-command policy checks
- MCP client support for stdio, Streamable HTTP, and SSE servers
- `SKILL.md` discovery, automatic routing, explicit `$skill-name` activation, and lazy loading
- SQLite short-term conversation history, durable memories, hybrid local retrieval, and task episodes
- JSONL lifecycle events for model calls, tool calls, plans, steps, MCP, and final outcomes
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
axiom chat                        keep a persistent conversation thread
axiom tui                         open the full-screen terminal interface
axiom demo                        run planning -> tool -> memory offline
axiom doctor                      inspect the local setup
axiom skills                      list discovered SKILL.md packages
axiom mcp                         list configured MCP servers
axiom memory list                 inspect durable memory
axiom memory search "query"       test memory retrieval
axiom memory forget MEMORY_ID     delete one memory
```

Use `--no-plan` for a direct single-step run. `--yes` approves commands that the configured policy
would otherwise ask about; it does not override the network-disable setting or workspace path guard.

### Terminal interface

Run the interactive interface in PowerShell, Windows Terminal, or another modern terminal:

```powershell
axiom tui
```

Use `Ctrl+Enter` to send a multiline prompt, `Escape` to stop the active task, `Ctrl+N` for a new
conversation thread, `Ctrl+L` to clear the visible transcript, and `Ctrl+Q` to exit. Planning,
steps, MCP connections, and tool calls appear in the activity pane. Commands requiring approval
open a modal confirmation; `axiom tui --yes` automatically approves policy-gated commands.

The first TUI release updates task state and tool activity in real time. Model text is displayed
when each model request completes; token-by-token streaming is not yet implemented.

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
Final synthesis --> conversation history + durable episode + JSONL events
```

Read [`docs/architecture.md`](docs/architecture.md) for lifecycle and extension points, and
[`docs/security.md`](docs/security.md) before granting an agent access to sensitive repositories.

## Replaceable components

- **Model:** implement `ModelProvider.complete`, or set `model.provider` to `module:factory`.
- **Tool:** subclass `Tool`, provide a JSON schema, and register it in `AxiomApp`.
- **Memory:** preserve the `SQLiteMemoryStore` method contract behind the `Agent` constructor.
- **Planning:** replace `Planner` while returning the same `TaskPlan` data type.
- **Interface:** subscribe to `EventBus` for a TUI, web UI, OpenTelemetry, or eval harness.
- **Scheduler:** the current executor runs ready steps sequentially. Independent read-only steps can
  later be dispatched in parallel without changing the plan format.

## Current boundaries

Axiom v0.1 is a strong local foundation, not an OS sandbox. File tools enforce a resolved workspace
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
ruff check src tests
```

The offline end-to-end test covers planner output, an actual tool call, tool-result feedback, final
completion, event emission, conversation persistence, and durable episode creation.
