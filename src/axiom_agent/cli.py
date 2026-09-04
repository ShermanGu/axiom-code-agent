from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from axiom_agent.app import AxiomApp
from axiom_agent.config import AxiomConfig, load_config
from axiom_agent.events import Event
from axiom_agent.memory.store import SQLiteMemoryStore
from axiom_agent.providers.demo import DemoProvider
from axiom_agent.skills.loader import SkillRegistry

DEFAULT_CONFIG = """[agent]
name = "Axiom"
max_turns = 20
max_step_retries = 1
planning = true
auto_memory = true

[model]
provider = "openai"
name = "gpt-5.6-terra"
reasoning_effort = "medium"
max_output_tokens = 8192
no_proxy = ""

[workspace]
root = "."
approval = "on-risk"
command_timeout_seconds = 120
max_command_output_chars = 30000
allow_network = false

[memory]
path = ".axiom/memory.db"
recent_messages = 12
retrieval_limit = 8

[skills]
paths = [".axiom/skills"]
auto_select = true
max_active = 3

# [[mcp.servers]]
# name = "example"
# transport = "stdio"
# command = "your-mcp-server"
# args = []
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="axiom", description="A modular code agent with MCP, skills, planning, and memory."
    )
    parser.add_argument("--version", action="version", version="Axiom 0.2.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize Axiom in a workspace")
    init_parser.add_argument("path", nargs="?", default=".")
    init_parser.add_argument("--force", action="store_true")

    run_parser = subparsers.add_parser("run", help="Run one task")
    run_parser.add_argument("goal", nargs="+")
    _common_run_arguments(run_parser)

    chat_parser = subparsers.add_parser("chat", help="Start a persistent interactive session")
    _common_run_arguments(chat_parser)

    demo_parser = subparsers.add_parser("demo", help="Run an offline end-to-end demo")
    demo_parser.add_argument("--workspace", default=".")
    demo_parser.add_argument("--json", action="store_true", dest="json_output")

    doctor_parser = subparsers.add_parser("doctor", help="Check the local setup")
    doctor_parser.add_argument("--config")
    doctor_parser.add_argument("--workspace")

    memory_parser = subparsers.add_parser("memory", help="Inspect durable memory")
    memory_sub = memory_parser.add_subparsers(dest="memory_command", required=True)
    memory_list = memory_sub.add_parser("list")
    memory_list.add_argument("--limit", type=int, default=20)
    memory_list.add_argument("--kind")
    memory_search = memory_sub.add_parser("search")
    memory_search.add_argument("query", nargs="+")
    memory_search.add_argument("--limit", type=int, default=8)
    memory_forget = memory_sub.add_parser("forget")
    memory_forget.add_argument("id")
    for child in (memory_list, memory_search, memory_forget):
        child.add_argument("--config")
        child.add_argument("--workspace")

    skills_parser = subparsers.add_parser("skills", help="List discovered skills")
    skills_parser.add_argument("--config")
    skills_parser.add_argument("--workspace")

    mcp_parser = subparsers.add_parser("mcp", help="List configured MCP servers")
    mcp_parser.add_argument("--config")
    mcp_parser.add_argument("--workspace")
    return parser


def _common_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config")
    parser.add_argument("--workspace")
    parser.add_argument("--no-plan", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Approve policy-gated commands")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")


def main(argv: list[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    try:
        code = asyncio.run(dispatch(arguments))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        code = 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        code = 1
    raise SystemExit(code)


async def dispatch(arguments: argparse.Namespace) -> int:
    if arguments.command == "init":
        return _init_workspace(Path(arguments.path), arguments.force)
    if arguments.command == "demo":
        return await _demo(arguments)

    config = load_config(
        getattr(arguments, "config", None), workspace=getattr(arguments, "workspace", None)
    )
    if arguments.command == "doctor":
        return _doctor(config)
    if arguments.command == "memory":
        return _memory(config, arguments)
    if arguments.command == "skills":
        return _skills(config)
    if arguments.command == "mcp":
        return _mcp(config)
    if getattr(arguments, "no_plan", False):
        config.agent.planning = False
    approve = _approval_callback(bool(getattr(arguments, "yes", False)))
    async with AxiomApp(config, approve=approve) as app:
        if not getattr(arguments, "quiet", False):
            app.events.subscribe(_console_observer)
        if arguments.command == "run":
            result = await app.agent.run(" ".join(arguments.goal))
            _print_result(result, arguments.json_output)
            return 0 if result.success else 2
        return await _chat(app, arguments.json_output)


def _init_workspace(path: Path, force: bool) -> int:
    workspace = path.resolve()
    config_dir = workspace / ".axiom"
    config_path = config_dir / "config.toml"
    if config_path.exists() and not force:
        raise FileExistsError(f"{config_path} already exists; pass --force to replace it")
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "skills").mkdir(exist_ok=True)
    config_path.write_text(DEFAULT_CONFIG, encoding="utf-8", newline="")
    gitignore = workspace / ".gitignore"
    ignore_lines = [".axiom/memory.db*", ".axiom/events.jsonl"]
    existing = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
    additions = [line for line in ignore_lines if line not in existing.splitlines()]
    if additions:
        prefix = "" if not existing or existing.endswith("\n") else "\n"
        with gitignore.open("a", encoding="utf-8", newline="") as handle:
            handle.write(prefix + "\n".join(additions) + "\n")
    print(f"Initialized Axiom at {config_path}")
    print(
        "Next: set a provider key and run Axiom. "
        "For a free first run, set GROQ_API_KEY and AXIOM_PROVIDER=groq."
    )
    return 0


async def _demo(arguments: argparse.Namespace) -> int:
    config = load_config(workspace=arguments.workspace)
    config.model.provider = "demo"
    config.agent.planning = True
    async with AxiomApp(config, provider=DemoProvider()) as app:
        app.events.subscribe(_console_observer)
        result = await app.agent.run("Inspect this workspace in the offline demo")
        _print_result(result, arguments.json_output)
    return 0 if result.success else 2


async def _chat(app: AxiomApp, json_output: bool) -> int:
    conversation_id: str | None = None
    print("Axiom chat. Type /exit to leave, /new to start a new memory thread.")
    while True:
        try:
            goal = await asyncio.to_thread(input, "you> ")
        except EOFError:
            break
        if goal.strip() in {"/exit", "/quit"}:
            break
        if goal.strip() == "/new":
            conversation_id = None
            print("Started a new conversation.")
            continue
        if not goal.strip():
            continue
        result = await app.agent.run(goal, conversation_id=conversation_id)
        conversation_id = result.conversation_id
        _print_result(result, json_output)
    return 0


def _doctor(config: AxiomConfig) -> int:
    openai_installed = bool(importlib.util.find_spec("openai"))
    mcp_installed = bool(importlib.util.find_spec("mcp"))
    api_key_set = bool(os.getenv(config.model.api_key_env))
    shell = shutil.which("pwsh") or shutil.which("powershell") or shutil.which("bash")
    checks = [
        ("Python", sys.version.split()[0], sys.version_info >= (3, 11)),
        ("Workspace", str(config.workspace.root), config.workspace.root.is_dir()),
        ("Config", str(config.config_path or "defaults"), True),
        ("OpenAI SDK", "installed" if openai_installed else "missing", openai_installed),
        ("MCP SDK", "installed" if mcp_installed else "missing", mcp_installed),
        (config.model.api_key_env, "set" if api_key_set else "missing", api_key_set),
        ("Shell", shell or "missing", bool(shell)),
    ]
    for name, detail, healthy in checks:
        print(f"[{'ok' if healthy else '--'}] {name}: {detail}")
    required = [item for item in checks[:2] if not item[2]]
    return 1 if required else 0


def _memory(config: AxiomConfig, arguments: argparse.Namespace) -> int:
    store = SQLiteMemoryStore(config.memory.path)
    try:
        if arguments.memory_command == "list":
            records = store.list_memories(max(1, arguments.limit), arguments.kind)
        elif arguments.memory_command == "search":
            records = store.search(" ".join(arguments.query), limit=max(1, arguments.limit))
        else:
            deleted = store.forget(arguments.id)
            print("Forgot memory." if deleted else "Memory not found.")
            return 0 if deleted else 1
        for item in records:
            score = f" score={item.score:.3f}" if item.score else ""
            print(f"{item.id}  {item.kind}  importance={item.importance:.2f}{score}")
            print(f"  {item.content[:500].replace(chr(10), ' ')}")
    finally:
        store.close()
    return 0


def _skills(config: AxiomConfig) -> int:
    skills = SkillRegistry.discover(config.skills.paths)
    for skill in skills.all():
        print(f"{skill.name}\n  {skill.description}\n  {skill.path}")
    return 0


def _mcp(config: AxiomConfig) -> int:
    if not config.mcp.servers:
        print("No MCP servers configured.")
        return 0
    for server in config.mcp.servers:
        print(f"{server.name}  {server.transport}  {server.command or server.url}")
    return 0


def _approval_callback(auto_approve: bool) -> Any:
    if auto_approve:
        return lambda _action, _reason: True

    def approve(action: str, reason: str) -> bool:
        print(f"\nApproval needed: {reason}\n  {action}", file=sys.stderr)
        answer = input("Allow once? [y/N] ")
        return answer.strip().casefold() in {"y", "yes"}

    return approve


def _console_observer(event: Event) -> None:
    if event.type == "plan.created":
        steps = event.data["plan"]["steps"]
        titles = " -> ".join(item["title"] for item in steps)
        print(f"[plan] {len(steps)} step(s): {titles}", file=sys.stderr)
    elif event.type == "step.started":
        print(f"[step] {event.data['title']}", file=sys.stderr)
    elif event.type == "tool.started":
        print(f"[tool] {event.data['name']}", file=sys.stderr)
    elif event.type in {"step.failed", "mcp.failed"}:
        print(f"[error] {event.data.get('error')}", file=sys.stderr)


def _print_result(result: Any, json_output: bool) -> None:
    if json_output:
        print(
            json.dumps(
                {
                    "output": result.output,
                    "success": result.success,
                    "conversation_id": result.conversation_id,
                    "plan": result.plan.as_dict(),
                    "usage": result.usage,
                },
                ensure_ascii=False,
            )
        )
    else:
        print(f"\n{result.output}")


if __name__ == "__main__":
    main()
