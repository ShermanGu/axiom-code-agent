# Security model

Axiom separates application-level safeguards from actual isolation.

## Enforced by Axiom

- File tools resolve symlinks and reject paths outside the configured workspace.
- Exact replacement aborts unless the expected number of matches is present.
- Reads and command output are bounded to prevent accidental context floods.
- Commands have timeouts.
- Known network commands are blocked when `workspace.allow_network = false`.
- Known destructive commands require policy approval.
- MCP tools are namespaced per server.
- Tool failures are recorded and returned to the model rather than hidden.
- API credentials are read from environment variables and are not written to memory by Axiom.
- JSONL logging redacts common credential keys and token patterns and bounds large text fields.

## Not enforced by Axiom

The shell tool is not an OS sandbox. A shell command inherits the current user's filesystem and
process permissions, and command classification cannot recognize every dangerous program or script.
Likewise, an MCP server is a separate trust boundary with its own credentials, data retention, and
side effects.

For untrusted goals, repositories, skills, or MCP servers:

1. Run Axiom inside a container, disposable VM, or restricted operating-system account.
2. Mount only the intended workspace and never mount credential directories.
3. Keep networking disabled or restrict egress at the OS/container layer.
4. Require human approval for tools that mutate external systems.
5. Review skill instructions and MCP server provenance before enabling them.
6. Back up or commit important work before autonomous runs.

The event log still contains task text, file-edit arguments, model output, and tool output. Treat the
`.axiom` directory as sensitive operational data even with credential-pattern redaction enabled.

## Approval modes

- `on-risk` (default): ask interactively for commands matching high-risk patterns.
- `always`: request approval for every shell command.
- `deny`: deny every policy-gated command.
- `never` / `auto`: approve policy-gated commands. Use only inside a real external sandbox.

`axiom run --yes` supplies approval for policy-gated commands during that CLI run. It does not turn
on network access and does not disable file path checks.
