from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from axiom_agent.tools.base import FunctionTool, Tool, ToolContext

WORD_PATTERN = re.compile(r"[\w-]+|[\u3400-\u9fff]", re.UNICODE)


@dataclass(slots=True)
class Skill:
    name: str
    description: str
    instructions: str
    path: Path

    def catalog_line(self) -> str:
        return f"- {self.name}: {self.description}"


class SkillRegistry:
    """Discovers SKILL.md capability packs and selects them with progressive disclosure."""

    def __init__(self, skills: list[Skill] | None = None) -> None:
        self._skills = {skill.name: skill for skill in skills or []}

    @classmethod
    def discover(cls, paths: list[Path]) -> SkillRegistry:
        discovered: dict[str, Skill] = {}
        for root in paths:
            candidates: list[Path]
            if root.is_file() and root.name.casefold() == "skill.md":
                candidates = [root]
            elif root.is_dir():
                candidates = sorted(root.rglob("SKILL.md"))
            else:
                continue
            for skill_path in candidates:
                skill = _load_skill(skill_path)
                discovered.setdefault(skill.name, skill)
        return cls(list(discovered.values()))

    def all(self) -> list[Skill]:
        return sorted(self._skills.values(), key=lambda item: item.name)

    def get(self, name: str) -> Skill | None:
        direct = self._skills.get(name)
        if direct:
            return direct
        lowered = name.casefold()
        return next(
            (skill for key, skill in self._skills.items() if key.casefold() == lowered), None
        )

    def select(self, goal: str, *, limit: int = 3, auto: bool = True) -> list[Skill]:
        explicit_names = {
            match.group(1).casefold()
            for match in re.finditer(r"\$([\w.-]+)", goal, flags=re.UNICODE)
        }
        selected: list[tuple[float, Skill]] = []
        goal_features = _features(goal)
        goal_folded = goal.casefold()
        for skill in self._skills.values():
            explicit = skill.name.casefold() in explicit_names
            named = skill.name.casefold() in goal_folded
            if explicit:
                score = 100.0
            elif named:
                score = 10.0
            elif auto:
                skill_features = _features(f"{skill.name} {skill.description}")
                shared = goal_features & skill_features
                score = len(shared) / max(1, len(goal_features))
            else:
                score = 0.0
            if explicit or named or score >= 0.08:
                selected.append((score, skill))
        selected.sort(key=lambda item: (item[0], item[1].name), reverse=True)
        return [skill for _, skill in selected[: max(0, limit)]]

    def catalog(self) -> str:
        if not self._skills:
            return "(no skills discovered)"
        return "\n".join(skill.catalog_line() for skill in self.all())

    def render(self, skills: list[Skill]) -> str:
        if not skills:
            return ""
        blocks = []
        for skill in skills:
            blocks.append(
                f"<skill name=\"{skill.name}\" source=\"{skill.path}\">\n"
                f"{skill.instructions}\n</skill>"
            )
        return "\n\n".join(blocks)

    def tool(self) -> Tool:
        async def activate(arguments: dict[str, Any], _context: ToolContext) -> dict[str, Any]:
            name = str(arguments["name"])
            skill = self.get(name)
            if skill is None:
                return {"found": False, "available": [item.name for item in self.all()]}
            return {
                "found": True,
                "name": skill.name,
                "source": str(skill.path),
                "instructions": skill.instructions,
            }

        return FunctionTool(
            "skill_activate",
            "Load the full instructions for one available skill when the current task needs it.",
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            activate,
            parallel_safe=True,
        )


def _load_skill(path: Path) -> Skill:
    content = path.read_text(encoding="utf-8", errors="replace")
    if len(content) > 200_000:
        raise ValueError(f"Skill is too large (>200k characters): {path}")
    metadata, body = _frontmatter(content)
    name = metadata.get("name") or path.parent.name
    description = metadata.get("description") or _first_paragraph(body) or f"Skill {name}"
    return Skill(str(name).strip(), str(description).strip(), body.strip(), path.resolve())


def _frontmatter(content: str) -> tuple[dict[str, str], str]:
    if not content.startswith("---"):
        return {}, content
    lines = content.splitlines()
    try:
        closing = lines.index("---", 1)
    except ValueError:
        return {}, content
    metadata: dict[str, str] = {}
    for line in lines[1:closing]:
        key, separator, value = line.partition(":")
        if separator:
            metadata[key.strip()] = value.strip().strip('"\'')
    return metadata, "\n".join(lines[closing + 1 :])


def _first_paragraph(body: str) -> str:
    for paragraph in re.split(r"\n\s*\n", body):
        text = " ".join(line.strip(" #\t") for line in paragraph.splitlines()).strip()
        if text:
            return text[:300]
    return ""


def _features(text: str) -> set[str]:
    return {token.casefold() for token in WORD_PATTERN.findall(text) if token.strip()}
