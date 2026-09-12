from __future__ import annotations

import argparse
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from axiom_agent.cli import _skills, build_parser
from axiom_agent.config import AxiomConfig


class SkillCommandTests(unittest.TestCase):
    def test_parser_accepts_run_history_and_resume_commands(self) -> None:
        parser = build_parser()
        listed = parser.parse_args(["runs"])
        self.assertEqual(listed.runs_command, "list")
        shown = parser.parse_args(["runs", "show", "abc123", "--json"])
        self.assertEqual((shown.runs_command, shown.run_id), ("show", "abc123"))
        resumed = parser.parse_args(["resume", "abc123", "--retry-uncertain-tools"])
        self.assertEqual(resumed.run_id, "abc123")
        self.assertTrue(resumed.retry_uncertain_tools)

    def test_parser_supports_legacy_list_search_and_show(self) -> None:
        parser = build_parser()
        self.assertEqual(parser.parse_args(["skills"]).skills_command, "list")
        self.assertEqual(parser.parse_args(["skills", "list"]).skills_command, "list")
        self.assertEqual(
            parser.parse_args(["skills", "search", "python"]).skills_query, ["python"]
        )
        self.assertEqual(
            parser.parse_args(["skills", "show", "secure-review"]).skills_query,
            ["secure-review"],
        )

    def test_skill_commands_keep_lists_concise_and_show_details_on_demand(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_skill(
                root,
                "secure-review",
                "Review Python code for security vulnerabilities",
                "Inspect trust boundaries and prove each finding.",
            )
            self._write_skill(
                root,
                "spreadsheet",
                "Analyze tabular data",
                "Check formulas and formatting.",
            )
            config = AxiomConfig()
            config.skills.paths = [root]

            listed = self._run(config, ["skills"])
            self.assertEqual(listed[0], 0)
            self.assertIn("secure-review: Review Python code", listed[1])
            self.assertNotIn("Inspect trust boundaries", listed[1])
            self.assertNotIn(str(root), listed[1])

            searched = self._run(config, ["skills", "search", "python", "security"])
            self.assertEqual(searched[0], 0)
            self.assertIn("secure-review", searched[1])
            self.assertNotIn("spreadsheet", searched[1])

            shown = self._run(config, ["skills", "show", "SECURE-REVIEW"])
            self.assertEqual(shown[0], 0)
            self.assertIn("Source:", shown[1])
            self.assertIn("Inspect trust boundaries", shown[1])

    @staticmethod
    def _write_skill(
        root: Path, name: str, description: str, instructions: str
    ) -> None:
        directory = root / name
        directory.mkdir()
        (directory / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {description}\n---\n{instructions}\n",
            encoding="utf-8",
        )

    @staticmethod
    def _run(config: AxiomConfig, argv: list[str]) -> tuple[int, str, str]:
        arguments: argparse.Namespace = build_parser().parse_args(argv)
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = _skills(config, arguments)
        return code, stdout.getvalue(), stderr.getvalue()


if __name__ == "__main__":
    unittest.main()
