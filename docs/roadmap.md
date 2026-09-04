# Axiom roadmap

This roadmap tracks the path from the current modular single-agent runtime to a comprehensive
coding-agent platform. Priority levels are dependency order, not promises that each level fits in
one release. Every release gets a separate tracker under `docs/releases/`; copy `TEMPLATE.md` when
opening a new milestone and link it from this document.

## Current milestone

- [`v0.4.0`](releases/v0.4.0.md) — P0-1 deterministic evaluation baseline (ready for review)

## P0 — Reliable single-agent kernel

Exit condition: Axiom can inspect, edit, test, recover, and report evidence on small and medium
repositories without silently losing state or overwriting unrelated user work.

### P0-1 Evaluation baseline

- [x] Define a versioned, deterministic offline suite format.
- [x] Cover inspection, edits, tool-error recovery, parallel-safe reads, multi-step synthesis, and
  dependency failure.
- [x] Record pass/fail, duration, model calls, tool calls, token estimates, changed files, and
  assertion failures.
- [x] Add `axiom eval` and a machine-readable report.
- [x] Add exact, allowed, and forbidden file-change assertions.
- [x] Add command exit-code and verification-evidence assertions.
- [ ] Add live-provider benchmark suites with explicit cost controls.
- [x] Run the core suite in cross-platform CI and retain comparable baseline summaries.

### P0-2 Durable execution and recovery

- [ ] Persist runs, plans, steps, attempts, turns, and tool calls with stable IDs.
- [ ] Checkpoint every state transition and resume an interrupted run safely.
- [ ] Distinguish failed, cancelled, interrupted, skipped, and blocked states.
- [ ] Connect EventBus to the SQLite event store while retaining JSONL export.
- [ ] Account for Planner, Executor, and Finalizer model usage and latency.

### P0-3 Transactional code changes

- [ ] Add a unified-diff `fs_patch` tool with stale-content preconditions.
- [ ] Use atomic writes and produce a structured diff for every mutation.
- [ ] Separate pre-existing user changes from agent-owned changes.
- [ ] Add per-step checkpoints, rollback, rename, and delete semantics.
- [ ] Add approval policy for writes and deletes.

### P0-4 Unified safety policy

- [ ] Describe tool capabilities as read, write, delete, network, process, and secret access.
- [ ] Apply one policy engine to built-in tools, shell commands, and MCP tools.
- [ ] Add network allowlists, process-tree cancellation, environment filtering, and robust Windows
  symlink/junction boundary checks.
- [ ] Define a sandbox backend interface for later container and VM implementations.

### P0-5 Context and prompt budgets

- [ ] Introduce a token-aware ContextBuilder with budgets per context source.
- [ ] Summarize oversized tool output while preserving references to raw evidence.
- [ ] Retrieve memory per step and prevent plan/finalizer truncation from dropping critical state.
- [ ] Add prompt snapshots and instruction-priority tests.

### P0-6 Verification and retry semantics

- [ ] Add explicit acceptance criteria and verification phases to plan steps.
- [ ] Distinguish tool, model, validation, and runtime failures.
- [ ] Make retries side-effect aware and deduplicate tool calls.
- [ ] Replan when observations invalidate the original plan.
- [ ] Enforce time, cost, turn, and tool-call budgets.

## P1 — Repository intelligence and memory

Exit condition: one Agent can handle cross-file tasks in a medium or large repository using precise
code context rather than repeated blind traversal.

- [ ] Build ignore-aware repository maps and incremental symbol indexes.
- [ ] Add AST/LSP definitions, references, diagnostics, dependency graphs, and hybrid code search.
- [ ] Add patch-oriented refactoring tools and change-impact context selection.
- [ ] Add short-term summarization and scoped user/workspace/repository/branch memories.
- [ ] Combine lexical and embedding retrieval with thresholds, provenance, deduplication, expiry,
  consolidation, and per-step queries.
- [ ] Add replanning, step acceptance criteria, and simple-task planner bypass.
- [ ] Add versioned Skills with dependencies, trust, resources, semantic routing, and activation that
  can persist across steps.
- [ ] Add streaming, provider fallback, model capability declarations, and cost-aware model routing.

## P2 — Multi-agent and long-running work

Exit condition: isolated agents can investigate, implement, test, and review in parallel, then merge
their work safely under a shared budget.

- [ ] Execute independent DAG steps concurrently with resource and file locks.
- [ ] Add Explorer, Implementer, Tester, Reviewer, and Researcher roles.
- [ ] Exchange structured artifacts rather than complete private histories.
- [ ] Isolate mutating agents in Git worktrees and merge reviewed patches or commits.
- [ ] Add conflict detection, cancellation propagation, priorities, and shared budgets.
- [ ] Make MCP startup partially tolerant, permission-aware, cancellable, and versioned.
- [ ] Support issue-to-branch-to-test-to-PR workflows and human approval at plan, step, and diff
  boundaries.

## P3 — Coding-agent platform

Exit condition: Axiom supports personal, team, and enterprise development workflows with isolation,
governance, extensibility, and continuous evaluation.

- [ ] Add local container, VM, and remote sandbox backends with resource quotas and ephemeral secrets.
- [ ] Add IDE integration, streaming task views, inline diff review, diagnostics, and task queues.
- [ ] Add trace replay, versioned prompts/tools/skills, benchmark automation, A/B evaluation, and
  failure clustering.
- [ ] Define signed plugin contracts for providers, tools, memory, Skills, sandboxes, and observers.
- [ ] Add team and organization configuration, RBAC, audit policy, knowledge scopes, and cost quotas.
