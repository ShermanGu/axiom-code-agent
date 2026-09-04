# Axiom evaluations

Evaluation suites are deterministic, offline scenarios that exercise the real `AxiomApp`, planner,
agent loop, tools, memory, finalizer, and event bus. They are regression tests for the runtime, not
a claim about the quality of any particular hosted model.

Run the core suite from the repository root:

```powershell
$env:PYTHONPATH = "src"
python -m axiom_agent.cli eval
```

The command writes a machine-readable report to `.axiom/evals/core-latest.json`. Use `--json` for
JSON on stdout, `--suite` to select another version-1 suite, and `--output` to choose the report
path.

Each case defines an isolated workspace, a deterministic model script, and observable expectations.
Expectations can constrain exact, allowed, and forbidden file changes and can validate structured
Shell exit codes and output evidence. The report records pass/fail status, duration, model calls,
tool calls, token estimates, changed files, verification evidence, and assertion failures.

Committed summaries under `baselines/` provide release comparison points. CI runs the suite on
Windows and Linux and uploads the complete JSON report for each platform. Live-model quality and
cost-controlled benchmarks can be added without changing the version-1 report envelope. Evaluation
suites are executable developer input and must only be loaded from trusted sources.
