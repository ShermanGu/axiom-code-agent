# Contributing to Axiom

Axiom is a work in progress. Focused bug reports, design discussions, tests, documentation fixes,
and small pull requests are welcome.

## Development setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Before opening a pull request, run:

```powershell
python -m pytest -q
python -m compileall -q src
python -m ruff check src tests
python -m mypy src/axiom_agent
```

Keep changes scoped, add tests for behavior changes, and update the README or architecture notes
when a user-facing contract changes. Never include API keys, tokens, private prompts, proprietary
code, or sensitive event logs in issues, tests, commits, or pull requests.

Every milestone and user-facing pull request must include an Interface Impact Review. Check whether
the change affects the CLI, TUI, configuration, events/APIs, and documentation; implement and test
all affected supported surfaces in the same milestone. If one surface is intentionally deferred,
name it with a reason and target milestone instead of silently leaving the interfaces inconsistent.

For security-sensitive reports, follow [`SECURITY.md`](SECURITY.md) instead of opening a public
issue with exploit details.
