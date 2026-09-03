from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from axiom_agent.skills.loader import SkillRegistry


class SkillTests(unittest.TestCase):
    def test_discovery_explicit_and_automatic_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            skill_dir = Path(directory) / "review"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                """---
name: secure-review
description: Review Python code for security vulnerabilities
---
Inspect trust boundaries and prove each finding.
""",
                encoding="utf-8",
            )
            registry = SkillRegistry.discover([Path(directory)])
            self.assertEqual(len(registry.all()), 1)
            explicit = registry.select("Use $secure-review", auto=False)
            self.assertEqual(explicit[0].name, "secure-review")
            self.assertEqual(
                registry.select("Review Python security", auto=True)[0].name, "secure-review"
            )
            self.assertIn("Inspect trust boundaries", registry.render(registry.all()))


if __name__ == "__main__":
    unittest.main()
