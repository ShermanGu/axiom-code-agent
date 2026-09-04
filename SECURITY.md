# Security policy

## Project status

Axiom is pre-1.0 work-in-progress software. Security fixes are applied to the latest `main` branch;
older releases are not currently maintained as separate support lines.

## Reporting a vulnerability

Do not publish credentials, exploit details, or sensitive logs in a public issue. Use GitHub's
private vulnerability reporting when it is available for this repository. If it is unavailable,
open a minimal issue asking the maintainer for a private contact channel without disclosing the
vulnerability itself.

Include the affected version or commit, operating system, configuration, impact, and the smallest
safe reproduction you can provide. Reports involving shell isolation, workspace escape, approval
bypass, credential exposure, MCP trust boundaries, or event-log redaction are especially useful.

For the runtime threat model and safe deployment guidance, read [`docs/security.md`](docs/security.md).
