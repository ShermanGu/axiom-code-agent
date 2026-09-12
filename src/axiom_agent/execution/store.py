from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

from axiom_agent.events import Event, sanitize
from axiom_agent.types import PlanStep, StepStatus, TaskPlan

RunStatus = Literal[
    "pending", "running", "completed", "failed", "cancelled", "interrupted", "blocked"
]
TERMINAL_STATUSES = {"completed", "failed", "cancelled", "interrupted", "blocked", "skipped"}


class UnsafeResumeError(RuntimeError):
    """Raised when resuming could repeat a tool whose outcome is unknown."""


@dataclass(slots=True)
class RunRecord:
    id: str
    conversation_id: str
    goal: str
    status: str
    output: str
    error: str
    usage: dict[str, int]
    metrics: dict[str, dict[str, int]]
    context: dict[str, Any]
    created_at: str
    updated_at: str
    started_at: str
    ended_at: str | None
    plan_id: str | None = None
    plan: TaskPlan | None = None
    step_record_ids: dict[str, str] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "goal": self.goal,
            "status": self.status,
            "output": self.output,
            "error": self.error,
            "usage": self.usage,
            "metrics": self.metrics,
            "context": self.context,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "plan_id": self.plan_id,
            "plan": self.plan.as_dict() if self.plan else None,
            "step_record_ids": self.step_record_ids or {},
        }


@dataclass(slots=True)
class StepResumeState:
    history: list[Any] | None = None
    completed_result: str | None = None
    uncertain_tool: bool = False


class ExecutionStore:
    """SQLite checkpoints for runs, plans, steps, attempts, turns, tools, and events."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._initialize()

    def _initialize(self) -> None:
        with self._lock, self._connection:
            self._connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA foreign_keys=ON;
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    status TEXT NOT NULL,
                    output TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    context TEXT NOT NULL DEFAULT '{}',
                    usage TEXT NOT NULL DEFAULT '{}',
                    metrics TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_runs_updated ON runs(updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status, updated_at DESC);

                CREATE TABLE IF NOT EXISTS plans (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL UNIQUE REFERENCES runs(id) ON DELETE CASCADE,
                    strategy TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS steps (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                    plan_id TEXT NOT NULL REFERENCES plans(id) ON DELETE CASCADE,
                    step_key TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    depends_on TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL,
                    result TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    ended_at TEXT,
                    UNIQUE(run_id, step_key)
                );
                CREATE INDEX IF NOT EXISTS idx_steps_run ON steps(run_id, sequence);

                CREATE TABLE IF NOT EXISTS attempts (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                    step_id TEXT NOT NULL REFERENCES steps(id) ON DELETE CASCADE,
                    number INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    UNIQUE(step_id, number)
                );
                CREATE INDEX IF NOT EXISTS idx_attempts_step ON attempts(step_id, number);

                CREATE TABLE IF NOT EXISTS turns (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                    step_id TEXT REFERENCES steps(id) ON DELETE CASCADE,
                    attempt_id TEXT REFERENCES attempts(id) ON DELETE CASCADE,
                    stage TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    request TEXT NOT NULL DEFAULT '{}',
                    response TEXT NOT NULL DEFAULT '{}',
                    response_id TEXT,
                    usage TEXT NOT NULL DEFAULT '{}',
                    duration_ms INTEGER NOT NULL DEFAULT 0,
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_turns_run ON turns(run_id, sequence);
                CREATE INDEX IF NOT EXISTS idx_turns_step ON turns(step_id, sequence);

                CREATE TABLE IF NOT EXISTS tool_calls (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                    step_id TEXT NOT NULL REFERENCES steps(id) ON DELETE CASCADE,
                    attempt_id TEXT NOT NULL REFERENCES attempts(id) ON DELETE CASCADE,
                    turn_id TEXT NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
                    provider_call_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    arguments TEXT NOT NULL DEFAULT '{}',
                    output TEXT NOT NULL DEFAULT '',
                    metadata TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL,
                    is_error INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_tool_calls_turn ON tool_calls(turn_id, sequence);
                CREATE INDEX IF NOT EXISTS idx_tool_calls_run ON tool_calls(run_id, created_at);

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    conversation_id TEXT,
                    type TEXT NOT NULL,
                    data TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_execution_events_run ON events(run_id, id);
                """
            )

    def create_run(
        self, conversation_id: str, goal: str, context: dict[str, Any]
    ) -> str:
        run_id = uuid4().hex
        now = _now()
        with self._lock, self._connection:
            self._connection.execute(
                """INSERT INTO runs(
                    id, conversation_id, goal, status, context,
                    created_at, updated_at, started_at
                ) VALUES (?, ?, ?, 'running', ?, ?, ?, ?)""",
                (run_id, conversation_id, goal, _json(context), now, now, now),
            )
        return run_id

    def save_plan(self, run_id: str, plan: TaskPlan) -> tuple[str, dict[str, str]]:
        plan_id = uuid4().hex
        now = _now()
        step_ids = {step.id: uuid4().hex for step in plan.steps}
        with self._lock, self._connection:
            self._connection.execute(
                """INSERT INTO plans(id, run_id, strategy, status, created_at, updated_at)
                VALUES (?, ?, ?, 'running', ?, ?)""",
                (plan_id, run_id, plan.strategy, now, now),
            )
            self._connection.executemany(
                """INSERT INTO steps(
                    id, run_id, plan_id, step_key, sequence, title, description,
                    depends_on, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        step_ids[step.id],
                        run_id,
                        plan_id,
                        step.id,
                        sequence,
                        step.title,
                        step.description,
                        _json(step.depends_on),
                        step.status,
                        now,
                        now,
                    )
                    for sequence, step in enumerate(plan.steps, 1)
                ],
            )
        return plan_id, step_ids

    def update_context(self, run_id: str, context: dict[str, Any]) -> None:
        now = _now()
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE runs SET context = ?, updated_at = ? WHERE id = ?",
                (_json(context), now, run_id),
            )

    def list_runs(self, limit: int = 20, status: str | None = None) -> list[RunRecord]:
        if status:
            query = "SELECT * FROM runs WHERE status = ? ORDER BY updated_at DESC LIMIT ?"
            parameters: tuple[Any, ...] = (status, limit)
        else:
            query = "SELECT * FROM runs ORDER BY updated_at DESC LIMIT ?"
            parameters = (limit,)
        with self._lock:
            rows = self._connection.execute(query, parameters).fetchall()
        return [self._run_from_row(row) for row in rows]

    def get_run(self, run_id_or_prefix: str, *, with_plan: bool = True) -> RunRecord:
        run_id = self.resolve_run_id(run_id_or_prefix)
        with self._lock:
            row = self._connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            assert row is not None
            record = self._run_from_row(row)
            if not with_plan:
                return record
            plan_row = self._connection.execute(
                "SELECT * FROM plans WHERE run_id = ?", (run_id,)
            ).fetchone()
            if plan_row is None:
                return record
            step_rows = self._connection.execute(
                "SELECT * FROM steps WHERE run_id = ? ORDER BY sequence", (run_id,)
            ).fetchall()
        record.plan_id = str(plan_row["id"])
        record.plan = TaskPlan(
            goal=record.goal,
            strategy=str(plan_row["strategy"]),
            steps=[
                PlanStep(
                    id=str(step["step_key"]),
                    title=str(step["title"]),
                    description=str(step["description"]),
                    depends_on=_json_load(step["depends_on"], []),
                    status=cast(StepStatus, str(step["status"])),
                    result=str(step["result"]),
                    error=str(step["error"]),
                )
                for step in step_rows
            ],
        )
        record.step_record_ids = {
            str(step["step_key"]): str(step["id"]) for step in step_rows
        }
        return record

    def resolve_run_id(self, run_id_or_prefix: str) -> str:
        prefix = run_id_or_prefix.strip()
        if not prefix:
            raise ValueError("Run ID must not be empty")
        with self._lock:
            rows = self._connection.execute(
                "SELECT id FROM runs WHERE id LIKE ? ORDER BY created_at DESC LIMIT 2",
                (f"{prefix}%",),
            ).fetchall()
        if not rows:
            raise KeyError(f"Run not found: {prefix}")
        if len(rows) > 1:
            raise ValueError(f"Run ID prefix is ambiguous: {prefix}")
        return str(rows[0]["id"])

    def prepare_resume(self, run_id_or_prefix: str, *, retry_uncertain: bool = False) -> RunRecord:
        record = self.get_run(run_id_or_prefix)
        if record.status == "completed":
            raise ValueError(f"Run {record.id} is already completed")
        blocked_message: str | None = None
        with self._lock, self._connection:
            uncertain = self._connection.execute(
                """SELECT COUNT(*) AS count FROM tool_calls
                WHERE run_id = ? AND status IN ('running', 'interrupted', 'cancelled')""",
                (record.id,),
            ).fetchone()
            assert uncertain is not None
            uncertain_count = int(uncertain["count"])
            now = _now()
            self._connection.execute(
                """UPDATE tool_calls SET status = 'interrupted', updated_at = ?, ended_at = ?
                WHERE run_id = ? AND status = 'running'""",
                (now, now, record.id),
            )
            self._connection.execute(
                """UPDATE turns SET status = 'interrupted', updated_at = ?, ended_at = ?
                WHERE run_id = ? AND status = 'running'""",
                (now, now, record.id),
            )
            self._connection.execute(
                """UPDATE attempts SET status = 'interrupted', updated_at = ?, ended_at = ?
                WHERE run_id = ? AND status = 'running'""",
                (now, now, record.id),
            )
            if uncertain_count and not retry_uncertain:
                blocked_message = (
                    "An interrupted tool call has no recorded outcome; automatic replay could "
                    "repeat side effects. Inspect the run, then explicitly pass "
                    "--retry-uncertain-tools if replay is acceptable."
                )
                self._connection.execute(
                    """UPDATE steps SET status = 'blocked', error = ?, updated_at = ?, ended_at = ?
                    WHERE run_id = ? AND status != 'completed'""",
                    (blocked_message, now, now, record.id),
                )
                self._connection.execute(
                    """UPDATE plans SET status = 'blocked', updated_at = ? WHERE run_id = ?""",
                    (now, record.id),
                )
                self._connection.execute(
                    """UPDATE runs SET status = 'blocked', error = ?, updated_at = ?, ended_at = ?
                    WHERE id = ?""",
                    (blocked_message, now, now, record.id),
                )
            else:
                self._connection.execute(
                    """UPDATE steps SET status = 'pending', result = '', error = '',
                    updated_at = ?, started_at = NULL, ended_at = NULL
                    WHERE run_id = ? AND status != 'completed'""",
                    (now, record.id),
                )
                self._connection.execute(
                    """UPDATE plans SET status = 'running', updated_at = ? WHERE run_id = ?""",
                    (now, record.id),
                )
                self._connection.execute(
                    """UPDATE runs SET status = 'running', error = '', output = '',
                    updated_at = ?, ended_at = NULL WHERE id = ?""",
                    (now, record.id),
                )
        if blocked_message is not None:
            raise UnsafeResumeError(blocked_message)
        return self.get_run(record.id)

    def start_attempt(self, run_id: str, step_id: str) -> tuple[str, int]:
        now = _now()
        attempt_id = uuid4().hex
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT COALESCE(MAX(number), 0) + 1 AS number FROM attempts WHERE step_id = ?",
                (step_id,),
            ).fetchone()
            assert row is not None
            number = int(row["number"])
            self._connection.execute(
                """INSERT INTO attempts(
                    id, run_id, step_id, number, status, created_at, updated_at, started_at
                ) VALUES (?, ?, ?, ?, 'running', ?, ?, ?)""",
                (attempt_id, run_id, step_id, number, now, now, now),
            )
            self._connection.execute(
                """UPDATE steps SET status = 'running', updated_at = ?,
                started_at = COALESCE(started_at, ?) WHERE id = ?""",
                (now, now, step_id),
            )
        return attempt_id, number

    def finish_attempt(self, attempt_id: str, status: str, error: str = "") -> None:
        now = _now()
        with self._lock, self._connection:
            self._connection.execute(
                """UPDATE attempts SET status = ?, error = ?, updated_at = ?, ended_at = ?
                WHERE id = ?""",
                (status, error, now, now, attempt_id),
            )

    def set_step_status(
        self, step_id: str, status: str, *, result: str = "", error: str = ""
    ) -> None:
        now = _now()
        ended_at = now if status in TERMINAL_STATUSES else None
        with self._lock, self._connection:
            self._connection.execute(
                """UPDATE steps SET status = ?, result = ?, error = ?, updated_at = ?,
                ended_at = ? WHERE id = ?""",
                (status, result, error, now, ended_at, step_id),
            )

    def start_turn(
        self,
        run_id: str,
        stage: str,
        request: dict[str, Any],
        *,
        step_id: str | None = None,
        attempt_id: str | None = None,
    ) -> tuple[str, int]:
        turn_id = uuid4().hex
        now = _now()
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 AS sequence FROM turns WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            assert row is not None
            sequence = int(row["sequence"])
            self._connection.execute(
                """INSERT INTO turns(
                    id, run_id, step_id, attempt_id, stage, sequence, status, request,
                    created_at, updated_at, started_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'running', ?, ?, ?, ?)""",
                (
                    turn_id,
                    run_id,
                    step_id,
                    attempt_id,
                    stage,
                    sequence,
                    _json(request),
                    now,
                    now,
                    now,
                ),
            )
        return turn_id, sequence

    def finish_turn(
        self,
        turn_id: str,
        status: str,
        *,
        response: dict[str, Any] | None = None,
        response_id: str | None = None,
        usage: dict[str, Any] | None = None,
        duration_ms: int = 0,
        error: str = "",
    ) -> None:
        now = _now()
        with self._lock, self._connection:
            self._connection.execute(
                """UPDATE turns SET status = ?, response = ?, response_id = ?, usage = ?,
                duration_ms = ?, error = ?, updated_at = ?, ended_at = ? WHERE id = ?""",
                (
                    status,
                    _json(response or {}),
                    response_id,
                    _json(usage or {}),
                    duration_ms,
                    error,
                    now,
                    now,
                    turn_id,
                ),
            )
            run_row = self._connection.execute(
                "SELECT run_id FROM turns WHERE id = ?", (turn_id,)
            ).fetchone()
            if run_row:
                self._update_accounting(str(run_row["run_id"]), now)

    def start_tool_call(
        self,
        *,
        run_id: str,
        step_id: str,
        attempt_id: str,
        turn_id: str,
        provider_call_id: str,
        name: str,
        arguments: dict[str, Any],
    ) -> str:
        tool_call_id = uuid4().hex
        now = _now()
        with self._lock, self._connection:
            row = self._connection.execute(
                """SELECT COALESCE(MAX(sequence), 0) + 1 AS sequence
                FROM tool_calls WHERE turn_id = ?""",
                (turn_id,),
            ).fetchone()
            assert row is not None
            self._connection.execute(
                """INSERT INTO tool_calls(
                    id, run_id, step_id, attempt_id, turn_id, provider_call_id,
                    sequence, name, arguments, status, created_at, updated_at, started_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?, ?, ?)""",
                (
                    tool_call_id,
                    run_id,
                    step_id,
                    attempt_id,
                    turn_id,
                    provider_call_id,
                    int(row["sequence"]),
                    name,
                    _json(arguments),
                    now,
                    now,
                    now,
                ),
            )
        return tool_call_id

    def finish_tool_call(
        self,
        tool_call_id: str,
        *,
        output: str,
        metadata: dict[str, Any],
        is_error: bool,
    ) -> None:
        now = _now()
        status = "failed" if is_error else "completed"
        with self._lock, self._connection:
            self._connection.execute(
                """UPDATE tool_calls SET status = ?, output = ?, metadata = ?, is_error = ?,
                updated_at = ?, ended_at = ? WHERE id = ?""",
                (status, _text(output), _json(metadata), int(is_error), now, now, tool_call_id),
            )

    def step_resume_state(
        self, step_id: str, *, retry_uncertain: bool = False
    ) -> StepResumeState:
        with self._lock:
            turn = self._connection.execute(
                """SELECT * FROM turns WHERE step_id = ? AND stage = 'executor'
                ORDER BY sequence DESC LIMIT 1""",
                (step_id,),
            ).fetchone()
            if turn is None:
                return StepResumeState()
            request = _json_load(turn["request"], {})
            history = list(request.get("input_items", []))
            calls = self._connection.execute(
                "SELECT * FROM tool_calls WHERE turn_id = ? ORDER BY sequence", (turn["id"],)
            ).fetchall()
        uncertain = any(
            call["status"] in {"running", "interrupted", "cancelled"} for call in calls
        )
        if uncertain:
            return StepResumeState(
                history=history,
                uncertain_tool=True,
            ) if retry_uncertain else StepResumeState(uncertain_tool=True)
        if turn["status"] != "completed":
            return StepResumeState(history=history)
        response = _json_load(turn["response"], {})
        text = str(response.get("text", ""))
        if not calls and text.strip():
            return StepResumeState(history=history, completed_result=text.strip())
        output_items = list(response.get("output_items", []))
        history.extend(output_items)
        if calls and not output_items:
            history.extend(
                {
                    "type": "function_call",
                    "call_id": str(call["provider_call_id"]),
                    "name": str(call["name"]),
                    "arguments": str(call["arguments"]),
                }
                for call in calls
            )
        history.extend(
            {
                "type": "function_call_output",
                "call_id": str(call["provider_call_id"]),
                "output": str(call["output"]),
            }
            for call in calls
        )
        return StepResumeState(history=history)

    def latest_stage_text(self, run_id: str, stage: str) -> str | None:
        with self._lock:
            row = self._connection.execute(
                """SELECT response FROM turns
                WHERE run_id = ? AND stage = ? AND status = 'completed'
                ORDER BY sequence DESC LIMIT 1""",
                (run_id, stage),
            ).fetchone()
        if row is None:
            return None
        text = str(_json_load(row["response"], {}).get("text", "")).strip()
        return text or None

    def finish_run(
        self, run_id: str, status: RunStatus, *, output: str = "", error: str = ""
    ) -> None:
        now = _now()
        with self._lock, self._connection:
            self._update_accounting(run_id, now)
            self._connection.execute(
                """UPDATE runs SET status = ?, output = ?, error = ?, updated_at = ?,
                ended_at = ? WHERE id = ?""",
                (status, _text(output), _text(error), now, now, run_id),
            )
            self._connection.execute(
                "UPDATE plans SET status = ?, updated_at = ? WHERE run_id = ?",
                (status, now, run_id),
            )

    def interrupt_run(
        self,
        run_id: str,
        status: Literal["cancelled", "interrupted"],
        error: str,
    ) -> None:
        now = _now()
        with self._lock, self._connection:
            for table in ("tool_calls", "turns", "attempts", "steps"):
                self._connection.execute(
                    f"""UPDATE {table} SET status = ?, updated_at = ?, ended_at = ?
                    WHERE run_id = ? AND status = 'running'""",
                    (status, now, now, run_id),
                )
            self._update_accounting(run_id, now)
            self._connection.execute(
                """UPDATE plans SET status = ?, updated_at = ? WHERE run_id = ?""",
                (status, now, run_id),
            )
            self._connection.execute(
                """UPDATE runs SET status = ?, error = ?, updated_at = ?, ended_at = ?
                WHERE id = ?""",
                (status, error, now, now, run_id),
            )

    def metrics(self, run_id: str) -> dict[str, dict[str, int]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT stage, status, usage, duration_ms FROM turns WHERE run_id = ?",
                (run_id,),
            ).fetchall()
        stages = {name: _empty_metrics() for name in ("planner", "executor", "finalizer")}
        for row in rows:
            stage = str(row["stage"])
            target = stages.setdefault(stage, _empty_metrics())
            target["model_calls"] += 1
            target["duration_ms"] += int(row["duration_ms"])
            if row["status"] == "failed":
                target["failed_calls"] += 1
            for key, value in _json_load(row["usage"], {}).items():
                if isinstance(value, int):
                    target[key] = target.get(key, 0) + value
        total = _empty_metrics()
        for values in stages.values():
            for key, value in values.items():
                total[key] = total.get(key, 0) + value
        return {**stages, "total": total}

    def usage(self, run_id: str) -> dict[str, int]:
        total = self.metrics(run_id)["total"]
        return {
            key: value
            for key, value in total.items()
            if key not in {"model_calls", "failed_calls", "duration_ms"} and value
        }

    def record_event(self, event: Event) -> None:
        data = sanitize(event.data)
        run_id = data.get("run_id") if isinstance(data, dict) else None
        conversation_id = data.get("conversation_id") if isinstance(data, dict) else None
        with self._lock, self._connection:
            self._connection.execute(
                """INSERT INTO events(run_id, conversation_id, type, data, created_at)
                VALUES (?, ?, ?, ?, ?)""",
                (run_id, conversation_id, event.type, _json(data), event.timestamp),
            )

    def event_rows(self, run_id_or_prefix: str) -> list[dict[str, Any]]:
        run_id = self.resolve_run_id(run_id_or_prefix)
        with self._lock:
            rows = self._connection.execute(
                "SELECT type, data, created_at FROM events WHERE run_id = ? ORDER BY id",
                (run_id,),
            ).fetchall()
        return [
            {
                "type": str(row["type"]),
                "data": _json_load(row["data"], {}),
                "timestamp": str(row["created_at"]),
            }
            for row in rows
        ]

    def detail(self, run_id_or_prefix: str) -> dict[str, Any]:
        record = self.get_run(run_id_or_prefix)
        with self._lock:
            attempts = self._connection.execute(
                """SELECT a.*, s.step_key FROM attempts a JOIN steps s ON s.id = a.step_id
                WHERE a.run_id = ? ORDER BY a.created_at""",
                (record.id,),
            ).fetchall()
            turns = self._connection.execute(
                "SELECT * FROM turns WHERE run_id = ? ORDER BY sequence", (record.id,)
            ).fetchall()
            tools = self._connection.execute(
                "SELECT * FROM tool_calls WHERE run_id = ? ORDER BY created_at, sequence",
                (record.id,),
            ).fetchall()
        payload = record.as_dict()
        payload["attempts"] = [
            {
                "id": row["id"],
                "step": row["step_key"],
                "number": row["number"],
                "status": row["status"],
                "error": row["error"],
                "started_at": row["started_at"],
                "ended_at": row["ended_at"],
            }
            for row in attempts
        ]
        payload["turns"] = [
            {
                "id": row["id"],
                "step_id": row["step_id"],
                "attempt_id": row["attempt_id"],
                "stage": row["stage"],
                "sequence": row["sequence"],
                "status": row["status"],
                "response_id": row["response_id"],
                "usage": _json_load(row["usage"], {}),
                "duration_ms": row["duration_ms"],
                "error": row["error"],
            }
            for row in turns
        ]
        payload["tool_calls"] = [
            {
                "id": row["id"],
                "turn_id": row["turn_id"],
                "provider_call_id": row["provider_call_id"],
                "name": row["name"],
                "status": row["status"],
                "is_error": bool(row["is_error"]),
                "started_at": row["started_at"],
                "ended_at": row["ended_at"],
            }
            for row in tools
        ]
        return payload

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _run_from_row(self, row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            id=str(row["id"]),
            conversation_id=str(row["conversation_id"]),
            goal=str(row["goal"]),
            status=str(row["status"]),
            output=str(row["output"]),
            error=str(row["error"]),
            usage=_json_load(row["usage"], {}),
            metrics=_json_load(row["metrics"], {}),
            context=_json_load(row["context"], {}),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            started_at=str(row["started_at"]),
            ended_at=str(row["ended_at"]) if row["ended_at"] else None,
        )

    def _update_accounting(self, run_id: str, now: str) -> None:
        metrics = self.metrics(run_id)
        usage = {
            key: value
            for key, value in metrics["total"].items()
            if key not in {"model_calls", "failed_calls", "duration_ms"} and value
        }
        self._connection.execute(
            "UPDATE runs SET usage = ?, metrics = ?, updated_at = ? WHERE id = ?",
            (_json(usage), _json(metrics), now, run_id),
        )


class SQLiteEventLogger:
    def __init__(self, store: ExecutionStore) -> None:
        self.store = store

    def __call__(self, event: Event) -> None:
        self.store.record_event(event)


def _empty_metrics() -> dict[str, int]:
    return {"model_calls": 0, "failed_calls": 0, "duration_ms": 0}


def _json(value: Any) -> str:
    return json.dumps(sanitize(value), ensure_ascii=False, default=str)


def _json_load(value: str, default: Any) -> Any:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _text(value: str) -> str:
    sanitized = sanitize(value)
    return str(sanitized)


def _now() -> str:
    return datetime.now(UTC).isoformat()
