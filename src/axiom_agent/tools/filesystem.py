from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from axiom_agent.tools.base import FunctionTool, Tool, ToolContext

IGNORED_PARTS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache"}


def resolve_workspace_path(workspace: Path, value: str | None) -> Path:
    root = workspace.resolve()
    raw = Path(value or ".")
    candidate = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise PermissionError(f"Path is outside the workspace: {value}") from exc
    return candidate


def _relative(path: Path, workspace: Path) -> str:
    text = str(path.relative_to(workspace.resolve()))
    return text.replace("\\", "/") or "."


async def fs_list(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    base = resolve_workspace_path(context.workspace, str(arguments.get("path", ".")))
    depth = max(0, min(int(arguments.get("depth", 2)), 6))
    max_entries = max(1, min(int(arguments.get("max_entries", 500)), 2000))
    if not base.exists():
        raise FileNotFoundError(_relative(base, context.workspace))
    if base.is_file():
        return {"entries": [_relative(base, context.workspace)], "truncated": False}

    entries: list[str] = []
    for path in sorted(base.rglob("*")):
        relative_to_base = path.relative_to(base)
        if len(relative_to_base.parts) > depth or any(part in IGNORED_PARTS for part in path.parts):
            continue
        suffix = "/" if path.is_dir() else ""
        entries.append(_relative(path, context.workspace) + suffix)
        if len(entries) >= max_entries:
            return {"entries": entries, "truncated": True}
    return {"entries": entries, "truncated": False}


async def fs_read(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    path = resolve_workspace_path(context.workspace, str(arguments["path"]))
    if not path.is_file():
        raise FileNotFoundError(_relative(path, context.workspace))
    start = max(1, int(arguments.get("start_line", 1)))
    end_raw = arguments.get("end_line")
    end = int(end_raw) if end_raw is not None else start + 399
    end = max(start, min(end, start + 1999))
    max_chars = max(100, min(int(arguments.get("max_chars", 30_000)), 100_000))
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    selected = lines[start - 1 : end]
    rendered = "\n".join(f"{index:>5} | {line}" for index, line in enumerate(selected, start))
    truncated = len(rendered) > max_chars or end < len(lines)
    return {
        "path": _relative(path, context.workspace),
        "content": rendered[:max_chars],
        "start_line": start,
        "end_line": min(end, len(lines)),
        "total_lines": len(lines),
        "truncated": truncated,
    }


async def fs_search(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    base = resolve_workspace_path(context.workspace, str(arguments.get("path", ".")))
    pattern = str(arguments["query"])
    is_regex = bool(arguments.get("regex", False))
    case_sensitive = bool(arguments.get("case_sensitive", False))
    max_results = max(1, min(int(arguments.get("max_results", 100)), 1000))
    flags = 0 if case_sensitive else re.IGNORECASE
    expression = re.compile(pattern if is_regex else re.escape(pattern), flags)
    matches: list[dict[str, Any]] = []
    candidates = [base] if base.is_file() else base.rglob("*")
    for path in candidates:
        if not path.is_file() or any(part in IGNORED_PARTS for part in path.parts):
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line_number, line in enumerate(content.splitlines(), 1):
            if expression.search(line):
                matches.append(
                    {
                        "path": _relative(path, context.workspace),
                        "line": line_number,
                        "text": line[:500],
                    }
                )
                if len(matches) >= max_results:
                    return {"matches": matches, "truncated": True}
    return {"matches": matches, "truncated": False}


async def fs_write(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    path = resolve_workspace_path(context.workspace, str(arguments["path"]))
    content = str(arguments.get("content", ""))
    mode = str(arguments.get("mode", "overwrite"))
    if mode not in {"overwrite", "append", "create"}:
        raise ValueError("mode must be overwrite, append, or create")
    if mode == "create" and path.exists():
        raise FileExistsError(_relative(path, context.workspace))
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "append":
        with path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(content)
    else:
        path.write_text(content, encoding="utf-8", newline="")
    return {"path": _relative(path, context.workspace), "bytes": len(content.encode("utf-8"))}


async def fs_replace(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    path = resolve_workspace_path(context.workspace, str(arguments["path"]))
    old = str(arguments["old"])
    new = str(arguments.get("new", ""))
    expected = int(arguments.get("expected_replacements", 1))
    if not old:
        raise ValueError("old must not be empty")
    content = path.read_text(encoding="utf-8")
    count = content.count(old)
    if count != expected:
        raise ValueError(f"Expected {expected} occurrence(s), found {count}; file was not changed")
    path.write_text(content.replace(old, new, expected), encoding="utf-8", newline="")
    return {"path": _relative(path, context.workspace), "replacements": expected}


def filesystem_tools() -> list[Tool]:
    object_schema = {"type": "object", "additionalProperties": False}
    return [
        FunctionTool(
            "fs_list",
            "List files and directories inside the workspace. Returns relative paths.",
            {
                **object_schema,
                "properties": {
                    "path": {"type": "string", "description": "Workspace-relative path."},
                    "depth": {"type": "integer", "minimum": 0, "maximum": 6},
                    "max_entries": {"type": "integer", "minimum": 1, "maximum": 2000},
                },
            },
            fs_list,
            parallel_safe=True,
        ),
        FunctionTool(
            "fs_read",
            "Read a UTF-8 text file with line numbers. Paths must stay inside the workspace.",
            {
                **object_schema,
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                    "max_chars": {"type": "integer", "minimum": 100},
                },
                "required": ["path"],
            },
            fs_read,
            parallel_safe=True,
        ),
        FunctionTool(
            "fs_search",
            "Search text files inside the workspace and return matching lines.",
            {
                **object_schema,
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": "string"},
                    "regex": {"type": "boolean"},
                    "case_sensitive": {"type": "boolean"},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 1000},
                },
                "required": ["query"],
            },
            fs_search,
            parallel_safe=True,
        ),
        FunctionTool(
            "fs_write",
            "Create, overwrite, or append a UTF-8 file inside the workspace.",
            {
                **object_schema,
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "mode": {"type": "string", "enum": ["overwrite", "append", "create"]},
                },
                "required": ["path", "content"],
            },
            fs_write,
        ),
        FunctionTool(
            "fs_replace",
            "Replace an exact, unique text block in a workspace file. "
            "Fails safely on count mismatch.",
            {
                **object_schema,
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                    "expected_replacements": {"type": "integer", "minimum": 1},
                },
                "required": ["path", "old", "new"],
            },
            fs_replace,
        ),
    ]
