from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from axiom_agent.config import load_config


class ConfigTests(unittest.TestCase):
    def test_groq_environment_preset_is_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "axiom.toml"
            config_path.write_text(
                '[model]\nname = "gpt-5.6-terra"\nmax_output_tokens = 8192\n',
                encoding="utf-8",
            )
            with patch.dict("os.environ", {"AXIOM_PROVIDER": "groq"}, clear=True):
                config = load_config(config_path, workspace=directory)
            self.assertEqual(config.model.provider, "groq")
            self.assertEqual(config.model.name, "qwen/qwen3.6-27b")
            self.assertEqual(config.model.api_key_env, "GROQ_API_KEY")
            self.assertEqual(config.model.base_url, "https://api.groq.com/openai/v1")
            self.assertEqual(config.model.max_output_tokens, 4096)

    def test_config_paths_are_relative_to_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            config_dir = workspace / ".axiom"
            config_dir.mkdir()
            config_path = config_dir / "config.toml"
            config_path.write_text(
                """
[workspace]
root = "."
[memory]
path = ".axiom/custom.db"
[skills]
paths = ["skills", ".axiom/skills"]
""",
                encoding="utf-8",
            )
            config = load_config(config_path)
            self.assertEqual(config.workspace.root, workspace.resolve())
            self.assertEqual(config.memory.path, (workspace / ".axiom/custom.db").resolve())
            self.assertEqual(config.skills.paths[0], (workspace / "skills").resolve())

    def test_workspace_override_is_a_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(workspace=directory)
            self.assertEqual(config.workspace.root, Path(directory).resolve())

    def test_mcp_environment_values_expand(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "axiom.toml"
            config_path.write_text(
                """
[[mcp.servers]]
name = "remote"
transport = "streamable_http"
url = "https://example.com/mcp"
headers = { Authorization = "Bearer ${TEST_MCP_TOKEN}" }
""",
                encoding="utf-8",
            )
            with patch.dict("os.environ", {"TEST_MCP_TOKEN": "test-value"}):
                config = load_config(config_path)
            self.assertEqual(config.mcp.servers[0].headers["Authorization"], "Bearer test-value")


if __name__ == "__main__":
    unittest.main()
