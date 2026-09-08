from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from importlib.resources import files
from pathlib import Path

from axiom_agent.cli import build_parser
from axiom_agent.evals import run_eval_suite


class EvaluationTests(unittest.TestCase):
    def test_cli_parser_accepts_eval_options(self) -> None:
        arguments = build_parser().parse_args(
            ["eval", "--suite", "custom.json", "--output", "report.json", "--json"]
        )

        self.assertEqual(arguments.command, "eval")
        self.assertEqual(arguments.suite, "custom.json")
        self.assertEqual(arguments.output, "report.json")
        self.assertTrue(arguments.json_output)

    def test_core_offline_suite_passes(self) -> None:
        root = Path(__file__).resolve().parents[1]
        suite = root / "evals/suites/core.json"
        result = asyncio.run(run_eval_suite(suite))

        self.assertEqual(result.total, 8)
        self.assertTrue(result.success, result.as_dict())
        self.assertGreater(result.as_dict()["metrics"]["model_calls"], 0)
        self.assertGreater(result.as_dict()["metrics"]["tool_calls"], 0)
        packaged = files("axiom_agent").joinpath("data/core-eval.json").read_text(encoding="utf-8")
        self.assertEqual(json.loads(packaged), json.loads(suite.read_text(encoding="utf-8")))

    def test_packaged_suite_is_isolated_from_repository_config(self) -> None:
        previous_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "axiom.toml").write_text(
                '[[mcp.servers]]\nname = "must-not-load"\ntransport = "invalid"\n',
                encoding="utf-8",
            )
            try:
                os.chdir(workspace)
                result = asyncio.run(run_eval_suite())
            finally:
                os.chdir(previous_directory)

        self.assertEqual(result.total, 8)
        self.assertTrue(result.success, result.as_dict())

    def test_change_policy_assertions_reject_unexpected_mutation(self) -> None:
        suite_payload = {
            "schema_version": 1,
            "name": "change-policy",
            "cases": [
                {
                    "id": "unexpected-change",
                    "goal": "Write the unexpected file.",
                    "initial_files": {"allowed.txt": "original\n"},
                    "plan": {
                        "strategy": "Exercise change assertions.",
                        "steps": [
                            {
                                "id": "write",
                                "title": "Write",
                                "description": "Write a file.",
                                "depends_on": [],
                            }
                        ],
                    },
                    "responses": [
                        {
                            "tool_calls": [
                                {
                                    "name": "fs_write",
                                    "arguments": {
                                        "path": "forbidden.txt",
                                        "content": "changed\n",
                                        "mode": "create",
                                    },
                                }
                            ]
                        },
                        {"text": "Wrote the file."},
                    ],
                    "expected": {
                        "success": True,
                        "allowed_changes": ["allowed.txt"],
                        "forbidden_changes": ["forbidden.txt"],
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            suite = Path(directory) / "suite.json"
            suite.write_text(json.dumps(suite_payload), encoding="utf-8")
            result = asyncio.run(run_eval_suite(suite))

        self.assertFalse(result.success)
        failures = "\n".join(result.cases[0].failures)
        self.assertIn("outside allowed patterns", failures)
        self.assertIn("Forbidden files changed", failures)


if __name__ == "__main__":
    unittest.main()
