---
name: code-review
description: Review code changes for correctness, security, maintainability, and missing tests.
---

# Code review

When reviewing code:

1. Inspect the actual diff and the surrounding implementation before reaching a conclusion.
2. Prioritize correctness, data loss, security boundaries, concurrency, and compatibility.
3. Confirm every finding with a concrete execution path; do not report speculative style concerns.
4. Attach each actionable finding to the tightest possible file and line range.
5. If no actionable findings remain, say so and mention any testing gaps.
