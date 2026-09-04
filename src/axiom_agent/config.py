from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

MODEL_PROVIDER_PRESETS: dict[str, dict[str, Any]] = {
    "groq": {
        "name": "qwen/qwen3.6-27b",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "max_output_tokens": 4096,
    },
    "gemini": {
        "name": "gemini-3.1-flash-lite",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "max_output_tokens": 4096,
    },
    "openrouter": {
        "name": "openrouter/free",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "max_output_tokens": 4096,
    },
}


@dataclass(slots=True)
class AgentConfig:
    name: str = "Axiom"
    max_turns: int = 20
    max_step_retries: int = 1
    planning: bool = True
    auto_memory: bool = True


@dataclass(slots=True)
class ModelConfig:
    provider: str = "openai"
    name: str = "gpt-5.6-terra"
    reasoning_effort: str = "medium"
    max_output_tokens: int = 8192
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    api_key: str | None = None
    no_proxy: str = ""


@dataclass(slots=True)
class WorkspaceConfig:
    root: Path = Path(".")
    approval: str = "on-risk"
    command_timeout_seconds: int = 120
    max_command_output_chars: int = 30_000
    allow_network: bool = False


@dataclass(slots=True)
class MemoryConfig:
    path: Path = Path(".axiom/memory.db")
    recent_messages: int = 12
    retrieval_limit: int = 8


@dataclass(slots=True)
class SkillsConfig:
    paths: list[Path] = field(default_factory=lambda: [Path(".axiom/skills")])
    auto_select: bool = True
    max_active: int = 3


@dataclass(slots=True)
class MCPServerConfig:
    name: str
    transport: str = "stdio"
    command: str | None = None
    args: list[str] = field(default_factory=list)
    url: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    tool_prefix: bool = True


@dataclass(slots=True)
class MCPConfig:
    servers: list[MCPServerConfig] = field(default_factory=list)


@dataclass(slots=True)
class AxiomConfig:
    agent: AgentConfig = field(default_factory=AgentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    skills: SkillsConfig = field(default_factory=SkillsConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    config_path: Path | None = None

    def resolve_paths(self, base: Path | None = None) -> None:
        anchor = (base or Path.cwd()).resolve()
        root = self.workspace.root
        self.workspace.root = (
            (anchor / root).resolve() if not root.is_absolute() else root.resolve()
        )
        memory = self.memory.path
        self.memory.path = (
            (self.workspace.root / memory).resolve()
            if not memory.is_absolute()
            else memory.resolve()
        )
        self.skills.paths = [
            (self.workspace.root / path).resolve() if not path.is_absolute() else path.resolve()
            for path in self.skills.paths
        ]


def _section(cls: type[Any], values: dict[str, Any] | None) -> Any:
    values = values or {}
    allowed = cls.__dataclass_fields__.keys()
    return cls(**{key: value for key, value in values.items() if key in allowed})


def find_config(start: Path | None = None) -> Path | None:
    current = (start or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        for name in ("axiom.toml", ".axiom/config.toml"):
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


def load_config(
    path: str | Path | None = None, *, workspace: str | Path | None = None
) -> AxiomConfig:
    config_path = Path(path).resolve() if path else find_config()
    raw: dict[str, Any] = {}
    if config_path:
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)

    workspace_values = dict(raw.get("workspace", {}))
    if workspace is not None:
        workspace_values["root"] = str(workspace)
    if "root" in workspace_values:
        workspace_values["root"] = Path(workspace_values["root"])

    model_values = dict(raw.get("model", {}))
    if model_provider := os.getenv("AXIOM_PROVIDER"):
        model_values["provider"] = model_provider
        # A provider selected at runtime should not inherit another provider's
        # endpoint, credential variable, or model unless explicitly overridden.
        for key, environment_name in (
            ("name", "AXIOM_MODEL"),
            ("base_url", "AXIOM_BASE_URL"),
            ("api_key_env", "AXIOM_API_KEY_ENV"),
            ("max_output_tokens", "AXIOM_MAX_OUTPUT_TOKENS"),
        ):
            if not os.getenv(environment_name):
                model_values.pop(key, None)
        model_values.pop("api_key", None)
    if model_name := os.getenv("AXIOM_MODEL"):
        model_values["name"] = model_name
    if base_url := os.getenv("AXIOM_BASE_URL"):
        model_values["base_url"] = base_url
    if api_key_env := os.getenv("AXIOM_API_KEY_ENV"):
        model_values["api_key_env"] = api_key_env
    if max_output_tokens := os.getenv("AXIOM_MAX_OUTPUT_TOKENS"):
        model_values["max_output_tokens"] = int(max_output_tokens)
    provider_name = str(model_values.get("provider", "openai"))
    preset = MODEL_PROVIDER_PRESETS.get(provider_name, {})
    model_values = {**preset, **model_values}

    memory_values = dict(raw.get("memory", {}))
    if "path" in memory_values:
        memory_values["path"] = Path(memory_values["path"])
    skills_values = dict(raw.get("skills", {}))
    if "paths" in skills_values:
        skills_values["paths"] = [Path(item) for item in skills_values["paths"]]

    raw_servers = raw.get("mcp", {}).get("servers", [])
    mcp_servers = [_section(MCPServerConfig, item) for item in raw_servers]
    for server in mcp_servers:
        server.env = {str(key): os.path.expandvars(str(value)) for key, value in server.env.items()}
        server.headers = {
            str(key): os.path.expandvars(str(value)) for key, value in server.headers.items()
        }
    config = AxiomConfig(
        agent=_section(AgentConfig, raw.get("agent")),
        model=_section(ModelConfig, model_values),
        workspace=_section(WorkspaceConfig, workspace_values),
        memory=_section(MemoryConfig, memory_values),
        skills=_section(SkillsConfig, skills_values),
        mcp=MCPConfig(servers=mcp_servers),
        config_path=config_path,
    )
    base = config_path.parent if config_path and config_path.name == "axiom.toml" else None
    if config_path and config_path.name == "config.toml" and config_path.parent.name == ".axiom":
        base = config_path.parent.parent
    config.resolve_paths(base)
    _validate(config)
    apply_no_proxy_environment(config.model.no_proxy)
    return config


def apply_no_proxy_environment(value: str) -> None:
    configured = [_no_proxy_host(entry) for entry in _split_no_proxy(value)]
    if not configured:
        return
    existing = _split_no_proxy(os.getenv("NO_PROXY", "") + "," + os.getenv("no_proxy", ""))
    merged = ",".join(dict.fromkeys([*existing, *configured]))
    os.environ["NO_PROXY"] = merged
    os.environ["no_proxy"] = merged


def _split_no_proxy(value: str) -> list[str]:
    return [entry.strip() for entry in value.split(",") if entry.strip()]


def _no_proxy_host(entry: str) -> str:
    return urlsplit(entry).netloc if "://" in entry else entry


def _validate(config: AxiomConfig) -> None:
    if config.agent.max_turns < 1:
        raise ValueError("agent.max_turns must be at least 1")
    if config.agent.max_step_retries < 0:
        raise ValueError("agent.max_step_retries must not be negative")
    if config.workspace.approval not in {"on-risk", "always", "deny", "never", "auto"}:
        raise ValueError("workspace.approval must be on-risk, always, deny, never, or auto")
    if config.workspace.command_timeout_seconds < 1:
        raise ValueError("workspace.command_timeout_seconds must be at least 1")
    if config.model.api_key is not None and not isinstance(config.model.api_key, str):
        raise ValueError("model.api_key must be a string")
    if not isinstance(config.model.no_proxy, str):
        raise ValueError("model.no_proxy must be a comma-separated string")
    names: set[str] = set()
    for server in config.mcp.servers:
        if server.name in names:
            raise ValueError(f"Duplicate MCP server name: {server.name}")
        names.add(server.name)
        if server.transport not in {"stdio", "streamable_http", "sse"}:
            raise ValueError(f"Unsupported MCP transport: {server.transport}")
        if server.transport == "stdio" and not server.command:
            raise ValueError(f"MCP server {server.name!r} requires command")
        if server.transport != "stdio" and not server.url:
            raise ValueError(f"MCP server {server.name!r} requires url")
