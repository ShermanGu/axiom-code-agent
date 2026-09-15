from __future__ import annotations

import argparse
import asyncio
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from axiom_agent.cli import _conversations, _demo, _skills, build_parser
from axiom_agent.config import AxiomConfig
from axiom_agent.execution.store import ExecutionStore


class CommandTests(unittest.TestCase):
    def test_parser_accepts_conversation_history_and_resume_commands(self) -> None:
        parser = build_parser()
        listed = parser.parse_args(["conversations"])
        self.assertEqual(listed.conversations_command, "list")
        shown = parser.parse_args(["conversations", "show", "abc123", "--json"])
        self.assertEqual(
            (shown.conversations_command, shown.conversation_id), ("show", "abc123")
        )
        resumed = parser.parse_args(["resume", "abc123", "--retry-uncertain-tools"])
        self.assertEqual(resumed.conversation_id, "abc123")
        self.assertTrue(resumed.retry_uncertain_tools)
        chat = parser.parse_args(["chat", "--conversation", "abc123"])
        self.assertEqual(chat.conversation, "abc123")

        recovery = parser.parse_args(["demo", "recovery", "--json"])
        self.assertEqual(recovery.scenario, "recovery")
        self.assertTrue(recovery.json_output)

    def test_recovery_demo_is_a_one_command_offline_acceptance_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            arguments = build_parser().parse_args(
                ["demo", "recovery", "--workspace", directory, "--json"]
            )
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = asyncio.run(_demo(arguments))
            payload = json.loads(stdout.getvalue())

            self.assertEqual(code, 0)
            self.assertTrue(payload["success"])
            self.assertTrue(all(payload["checks"].values()))
            self.assertEqual(
                set(payload["conversation_ids"]), {"checkpoint", "uncertain"}
            )
            self.assertTrue((Path(directory) / ".axiom" / "runs.db").is_file())

    def test_conversation_command_groups_internal_task_executions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.db"
            store = ExecutionStore(path)
            first = store.create_run("conversation-one", "First task", {})
            store.finish_run(first, "completed", output="done")
            second = store.create_run("conversation-one", "Second task", {})
            store.finish_run(second, "completed", output="done")
            store.close()

            config = AxiomConfig()
            config.execution.path = path
            arguments = build_parser().parse_args(["conversations"])
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = _conversations(config, arguments)

            self.assertEqual(code, 0)
            self.assertEqual(stdout.getvalue().count("conversation"), 1)
            self.assertIn("tasks=2", stdout.getvalue())
            self.assertNotIn(first[:12], stdout.getvalue())
            self.assertNotIn(second[:12], stdout.getvalue())

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
