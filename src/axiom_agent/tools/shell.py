from __future__ import annotations

import asyncio
import os
import re
import shutil
from pathlib import Path
from typing import Any

from axiom_agent.config import WorkspaceConfig
from axiom_agent.tools.base import Tool, ToolContext

NETWORK_PATTERN = re.compile(
    r"\b(curl|wget|Invoke-WebRequest|Invoke-RestMethod|pip\s+install|uv\s+add|"
    r"npm\s+(install|add)|pnpm\s+(install|add)|git\s+(clone|fetch|pull|push))\b",
    re.IGNORECASE,
)
HIGH_RISK_PATTERN = re.compile(
    r"\b(rm\s+-[^\n]*r|Remove-Item\b[^\n]*-Recurse|rmdir\s+/s|del\s+/[sq]|"
    r"format\b|diskpart\b|shutdown\b|reboot\b|git\s+reset\s+--hard|"
    r"git\s+clean\s+-[^\n]*f|git\s+push\b[^\n]*--force)\b",
    re.IGNORECASE,
)


class ShellTool(Tool):
    name = "shell_run"
    description = (
        "Run a command in the workspace with a timeout and bounded output. "
        "Network and high-risk commands are policy-gated."
    )
    parameters = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "command": {"type": "string", "description": "Command to run."},
            "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 900},
        },
        "required": ["command"],
    }
    parallel_safe = False

    def __init__(self, config: WorkspaceConfig) -> None:
        self.config = config

    async def run(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        command = str(arguments["command"]).strip()
        if not command:
            raise ValueError("command must not be empty")
        timeout = max(
            1,
            min(int(arguments.get("timeout_seconds", self.config.command_timeout_seconds)), 900),
        )
        uses_network = bool(NETWORK_PATTERN.search(command))
        high_risk = bool(HIGH_RISK_PATTERN.search(command))
        if uses_network and not self.config.allow_network:
            raise PermissionError("Network commands are disabled by workspace.allow_network")
        if high_risk or context.approval_mode == "always":
            reason = (
                "The command can delete or overwrite data"
                if high_risk
                else "The approval policy requires confirmation for shell commands"
            )
            allowed = await context.request_approval(command, reason)
            if not allowed:
                raise PermissionError("Command was not approved")

        executable, prefix = _shell_command()
        environment = os.environ.copy()
        environment.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
        process = await asyncio.create_subprocess_exec(
            executable,
            *prefix,
            command,
            cwd=str(context.workspace),
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            output_bytes, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except TimeoutError:
            process.kill()
            await process.communicate()
            raise TimeoutError(f"Command exceeded {timeout} seconds") from None
        output = output_bytes.decode("utf-8", errors="replace")
        limit = self.config.max_command_output_chars
        truncated = len(output) > limit
        if truncated:
            head = output[: limit // 2]
            tail = output[-limit // 2 :]
            output = head + "\n... output truncated ...\n" + tail
        return {
            "exit_code": process.returncode,
            "output": output,
            "truncated": truncated,
            "cwd": str(Path(context.workspace)),
        }


def _shell_command() -> tuple[str, list[str]]:
    if os.name == "nt":
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        if not powershell:
            raise RuntimeError("PowerShell is required on Windows")
        return powershell, ["-NoLogo", "-NoProfile", "-NonInteractive", "-Command"]
    shell = shutil.which("bash") or "/bin/sh"
    return shell, ["-lc"]
