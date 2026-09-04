from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from hashlib import sha256
from pathlib import Path
from typing import Any

from axiom_agent.app import AxiomApp
from axiom_agent.config import load_config
from axiom_agent.providers.base import ModelProvider
from axiom_agent.types import ModelRequest, ModelResponse, ToolCall


@dataclass(slots=True)
class EvalCaseResult:
    case_id: str
    passed: bool
    duration_ms: int
    agent_success: bool
    model_calls: int
    tool_calls: int
    files_changed: list[str]
    verification_evidence: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.case_id,
            "passed": self.passed,
            "duration_ms": self.duration_ms,
            "agent_success": self.agent_success,
            "model_calls": self.model_calls,
            "tool_calls": self.tool_calls,
            "files_changed": self.files_changed,
            "verification_evidence": self.verification_evidence,
            "usage": self.usage,
            "failures": self.failures,
        }


@dataclass(slots=True)
class EvalSuiteResult:
    suite: str
    cases: list[EvalCaseResult]
    duration_ms: int

    @property
    def passed(self) -> int:
        return sum(case.passed for case in self.cases)

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def success(self) -> bool:
        return self.total > 0 and self.passed == self.total

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "suite": self.suite,
            "passed": self.passed,
            "total": self.total,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "metrics": {
                "model_calls": sum(case.model_calls for case in self.cases),
                "tool_calls": sum(case.tool_calls for case in self.cases),
                "input_tokens": sum(case.usage.get("input_tokens", 0) for case in self.cases),
                "output_tokens": sum(
                    case.usage.get("output_tokens", 0) for case in self.cases
                ),
                "total_tokens": sum(case.usage.get("total_tokens", 0) for case in self.cases),
            },
            "cases": [case.as_dict() for case in self.cases],
        }


class ScriptedEvalProvider(ModelProvider):
    """Deterministic provider used to exercise the real agent runtime offline."""

    def __init__(self, case: dict[str, Any]) -> None:
        self.plan = dict(case["plan"])
        self.responses = list(case.get("responses", []))
        self.finalizer_text = str(case.get("finalizer", "Evaluation finalizer completed."))
        self.model_calls = 0
        self.tool_calls = 0
        self.usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.model_calls += 1
        usage = {
            "input_tokens": max(1, sum(len(str(item)) for item in request.input_items) // 20),
            "output_tokens": 5,
        }
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        for key, value in usage.items():
            self.usage[key] += value

        if "task planner" in request.instructions.casefold():
            text = json.dumps(self.plan, ensure_ascii=False)
            return ModelResponse(
                text=text,
                output_items=[{"role": "assistant", "content": text}],
                usage=usage,
            )
        if "result synthesizer" in request.instructions.casefold():
            return ModelResponse(
                text=self.finalizer_text,
                output_items=[{"role": "assistant", "content": self.finalizer_text}],
                usage=usage,
            )
        if not self.responses:
            raise RuntimeError("Evaluation script ran out of execution responses")

        scripted = self.responses.pop(0)
        text = str(scripted.get("text", ""))
        calls: list[ToolCall] = []
        output_items: list[dict[str, Any]] = []
        for index, raw_call in enumerate(scripted.get("tool_calls", []), 1):
            call_id = str(raw_call.get("id") or f"eval_{self.model_calls}_{index}")
            call = ToolCall(
                id=call_id,
                name=str(raw_call["name"]),
                arguments=dict(raw_call.get("arguments", {})),
            )
            calls.append(call)
            output_items.append(
                {
                    "type": "function_call",
                    "call_id": call.id,
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                }
            )
        self.tool_calls += len(calls)
        if text:
            output_items.append({"role": "assistant", "content": text})
        return ModelResponse(text=text, tool_calls=calls, output_items=output_items, usage=usage)


async def run_eval_suite(
    suite_path: Path,
    *,
    report_path: Path | None = None,
) -> EvalSuiteResult:
    payload = json.loads(suite_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Evaluation suite schema_version must be 1")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Evaluation suite must contain at least one case")

    suite_started = time.perf_counter()
    cases = [await _run_case(case) for case in raw_cases]
    result = EvalSuiteResult(
        suite=str(payload.get("name") or suite_path.stem),
        cases=cases,
        duration_ms=round((time.perf_counter() - suite_started) * 1000),
    )
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(result.as_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return result


async def _run_case(case: dict[str, Any]) -> EvalCaseResult:
    case_id = str(case.get("id", "unnamed"))
    started = time.perf_counter()
    failures: list[str] = []
    provider = ScriptedEvalProvider(case)
    agent_success = False
    event_types: list[str] = []
    verification_evidence: list[dict[str, Any]] = []
    before: dict[str, str] = {}
    after: dict[str, str] = {}

    with tempfile.TemporaryDirectory(prefix=f"axiom-eval-{case_id}-") as directory:
        workspace = Path(directory)
        _write_files(workspace, dict(case.get("initial_files", {})))
        before = _snapshot(workspace)
        config = load_config(workspace=workspace)
        options = dict(case.get("options", {}))
        config.model.provider = "demo"
        config.agent.planning = bool(options.get("planning", True))
        config.agent.max_turns = int(options.get("max_turns", 20))
        config.agent.max_step_retries = int(options.get("max_step_retries", 1))
        config.agent.auto_memory = bool(options.get("auto_memory", True))

        result = None
        try:
            async with AxiomApp(config, provider=provider) as app:
                app.events.subscribe(
                    lambda event: _observe_event(event, event_types, verification_evidence)
                )
                result = await app.agent.run(str(case["goal"]))
                agent_success = result.success
        except Exception as exc:
            failures.append(f"Agent raised {type(exc).__name__}: {exc}")
        after = _snapshot(workspace)
        changed = _changed_paths(before, after)

        if result is not None:
            _check_expectations(
                case,
                result=result,
                workspace=workspace,
                event_types=event_types,
                changed_paths=changed,
                verification_evidence=verification_evidence,
                provider=provider,
                failures=failures,
            )

    return EvalCaseResult(
        case_id=case_id,
        passed=not failures,
        duration_ms=round((time.perf_counter() - started) * 1000),
        agent_success=agent_success,
        model_calls=provider.model_calls,
        tool_calls=provider.tool_calls,
        files_changed=changed,
        verification_evidence=verification_evidence,
        usage=provider.usage,
        failures=failures,
    )


def _check_expectations(
    case: dict[str, Any],
    *,
    result: Any,
    workspace: Path,
    event_types: list[str],
    changed_paths: list[str],
    verification_evidence: list[dict[str, Any]],
    provider: ScriptedEvalProvider,
    failures: list[str],
) -> None:
    expected = dict(case.get("expected", {}))
    if "success" in expected and result.success != bool(expected["success"]):
        failures.append(f"Expected success={expected['success']}, got {result.success}")
    for text in expected.get("output_contains", []):
        if str(text) not in result.output:
            failures.append(f"Output did not contain {text!r}")
    actual_statuses = {step.id: step.status for step in result.plan.steps}
    for step_id, status in dict(expected.get("step_statuses", {})).items():
        if actual_statuses.get(step_id) != status:
            failures.append(
                f"Expected step {step_id!r} status {status!r}, "
                f"got {actual_statuses.get(step_id)!r}"
            )
    for path, content in dict(expected.get("files", {})).items():
        target = _safe_path(workspace, str(path))
        actual = target.read_text(encoding="utf-8") if target.is_file() else None
        if actual != content:
            failures.append(f"Expected file {path!r} to equal {content!r}, got {actual!r}")
    if "changed_files" in expected:
        wanted = sorted(str(path) for path in expected["changed_files"])
        if changed_paths != wanted:
            failures.append(f"Expected changed files {wanted!r}, got {changed_paths!r}")
    allowed = [str(pattern) for pattern in expected.get("allowed_changes", [])]
    if allowed:
        unexpected = [path for path in changed_paths if not _matches_any(path, allowed)]
        if unexpected:
            failures.append(
                f"Changed files were outside allowed patterns {allowed!r}: {unexpected!r}"
            )
    forbidden = [str(pattern) for pattern in expected.get("forbidden_changes", [])]
    violations = [path for path in changed_paths if _matches_any(path, forbidden)]
    if violations:
        failures.append(f"Forbidden files changed for patterns {forbidden!r}: {violations!r}")
    for event_type in expected.get("events", []):
        if str(event_type) not in event_types:
            failures.append(f"Expected event {event_type!r} was not emitted")
    if "model_calls" in expected and provider.model_calls != int(expected["model_calls"]):
        failures.append(
            f"Expected {expected['model_calls']} model calls, got {provider.model_calls}"
        )
    if "tool_calls" in expected and provider.tool_calls != int(expected["tool_calls"]):
        failures.append(f"Expected {expected['tool_calls']} tool calls, got {provider.tool_calls}")
    _check_shell_results(expected, verification_evidence, failures)


def _observe_event(
    event: Any,
    event_types: list[str],
    verification_evidence: list[dict[str, Any]],
) -> None:
    event_types.append(event.type)
    if event.type != "tool.completed" or event.data.get("name") != "shell_run":
        return
    try:
        result = json.loads(str(event.data.get("output", "{}")))
    except json.JSONDecodeError:
        return
    verification_evidence.append(
        {
            "tool": "shell_run",
            "exit_code": result.get("exit_code"),
            "output": str(result.get("output", ""))[:2000],
            "truncated": bool(result.get("truncated", False)),
        }
    )


def _check_shell_results(
    expected: dict[str, Any],
    verification_evidence: list[dict[str, Any]],
    failures: list[str],
) -> None:
    for index, wanted in enumerate(expected.get("shell_results", []), 1):
        exit_code = int(wanted["exit_code"])
        candidates = [
            item for item in verification_evidence if item.get("exit_code") == exit_code
        ]
        if not candidates:
            failures.append(
                f"Shell result {index} expected exit_code={exit_code}, but none matched"
            )
            continue
        for text in wanted.get("output_contains", []):
            if not any(str(text) in str(item.get("output", "")) for item in candidates):
                failures.append(
                    f"Shell result {index} with exit_code={exit_code} "
                    f"did not contain {text!r}"
                )


def _matches_any(path: str, patterns: list[str]) -> bool:
    return any(fnmatchcase(path, pattern) for pattern in patterns)


def _write_files(workspace: Path, files: dict[str, Any]) -> None:
    for path, content in files.items():
        target = _safe_path(workspace, str(path))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(content), encoding="utf-8", newline="")


def _safe_path(workspace: Path, value: str) -> Path:
    target = (workspace / value).resolve()
    try:
        target.relative_to(workspace.resolve())
    except ValueError as exc:
        raise ValueError(f"Evaluation path escapes workspace: {value}") from exc
    return target


def _snapshot(workspace: Path) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for path in workspace.rglob("*"):
        if not path.is_file() or ".axiom" in path.relative_to(workspace).parts:
            continue
        relative = str(path.relative_to(workspace)).replace("\\", "/")
        snapshot[relative] = sha256(path.read_bytes()).hexdigest()
    return snapshot


def _changed_paths(before: dict[str, str], after: dict[str, str]) -> list[str]:
    return sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))
