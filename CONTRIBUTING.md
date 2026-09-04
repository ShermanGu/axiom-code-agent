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
ruff check src tests
mypy src/axiom_agent
```

Keep changes scoped, add tests for behavior changes, and update the README or architecture notes
when a user-facing contract changes. Never include API keys, tokens, private prompts, proprietary
code, or sensitive event logs in issues, tests, commits, or pull requests.

For security-sensitive reports, follow [`SECURITY.md`](SECURITY.md) instead of opening a public
issue with exploit details.
