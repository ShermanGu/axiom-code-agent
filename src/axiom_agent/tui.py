from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Label,
    LoadingIndicator,
    Markdown,
    RichLog,
    Static,
    TextArea,
)
from textual.worker import Worker

from axiom_agent.app import AxiomApp
from axiom_agent.config import AxiomConfig
from axiom_agent.events import Event
from axiom_agent.execution.store import ExecutionStore, RunRecord, UnsafeResumeError
from axiom_agent.tools.base import ApprovalCallback

BackendFactory = Callable[[AxiomConfig, ApprovalCallback], Any]
RUN_TRANSCRIPT_LIMIT = 200


class PromptArea(TextArea):
    BINDINGS = [
        Binding("enter,ctrl+enter", "submit", "Send", show=False, priority=True),
        Binding("ctrl+j", "newline", "New line", show=False, priority=True),
    ]

    class Submitted(Message):
        def __init__(self, value: str) -> None:
            self.value = value
            super().__init__()

    def action_submit(self) -> None:
        self.post_message(self.Submitted(self.text))

    def action_newline(self) -> None:
        start, end = self.selection
        self.replace("\n", start, end, maintain_selection_offset=False)


class ApprovalScreen(ModalScreen[bool]):
    BINDINGS = [Binding("escape", "deny", "Deny")]

    def __init__(self, action: str, reason: str) -> None:
        self.action_text = action
        self.reason = reason
        super().__init__()

    def compose(self) -> ComposeResult:
        with Vertical(id="approval-dialog"):
            yield Label("Approval required", id="approval-title")
            yield Static(Text(self.reason), id="approval-reason")
            yield Static(Text(self.action_text), id="approval-action")
            with Horizontal(id="approval-buttons"):
                yield Button("Deny", id="deny", variant="error")
                yield Button("Allow once", id="allow", variant="success")

    @on(Button.Pressed)
    def choose(self, event: Button.Pressed) -> None:
        if event.button.id in {"allow", "deny"}:
            self.dismiss(event.button.id == "allow")

    def action_deny(self) -> None:
        self.dismiss(False)


@dataclass(frozen=True, slots=True)
class RunAction:
    kind: Literal["resume", "retry"]
    run_id: str


class RunHistoryScreen(ModalScreen[RunAction | None]):
    BINDINGS = [Binding("escape", "close", "Close")]

    def __init__(
        self,
        store: ExecutionStore,
        export_run: Callable[[str], Path],
        *,
        selected_run_id: str | None = None,
    ) -> None:
        self.store = store
        self.export_run = export_run
        self.selected_run_id = selected_run_id
        self.records: dict[str, RunRecord] = {}
        self.initialized = False
        super().__init__()

    def compose(self) -> ComposeResult:
        with Vertical(id="runs-dialog"):
            yield Label("RUN HISTORY", id="runs-title")
            with Horizontal(id="runs-content"):
                yield DataTable(id="run-table", cursor_type="row")
                with VerticalScroll(id="run-detail-scroll"):
                    yield Static("Select a run to inspect it.", id="run-detail")
            yield Static("", id="runs-message")
            with Horizontal(id="runs-buttons"):
                yield Button("Refresh", id="run-refresh", disabled=True)
                yield Button("Export", id="run-export", disabled=True)
                yield Button("Retry blocked", id="run-retry", variant="warning", disabled=True)
                yield Button("Resume run", id="run-resume", variant="primary", disabled=True)
                yield Button("Close", id="run-close")

    def on_mount(self) -> None:
        # On some Windows/Python combinations the screen receives Mount before every
        # composed descendant is queryable. Defer child access until the first refresh.
        self.call_after_refresh(self._initialize)

    def _initialize(self) -> None:
        table = self.query_one("#run-table", DataTable)
        table.add_columns("Run ID", "Status", "Updated", "Goal")
        self.initialized = True
        self.query_one("#run-refresh", Button).disabled = False
        self._refresh()

    @on(DataTable.RowHighlighted)
    def select_run(self, event: DataTable.RowHighlighted) -> None:
        run_id = str(event.row_key.value)
        if run_id in self.records:
            self._show_detail(run_id)

    @on(Button.Pressed)
    def handle_button(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "run-close":
            self.dismiss(None)
        elif button_id == "run-refresh":
            self._refresh()
        elif button_id == "run-export":
            self._export_selected()
        elif button_id in {"run-resume", "run-retry"} and self.selected_run_id:
            kind: Literal["resume", "retry"] = (
                "retry" if button_id == "run-retry" else "resume"
            )
            self.dismiss(RunAction(kind, self.selected_run_id))

    def action_close(self) -> None:
        self.dismiss(None)

    def _refresh(self) -> None:
        if not self.initialized:
            return
        table = self.query_one("#run-table", DataTable)
        records = self.store.list_runs(limit=100)
        self.records = {record.id: record for record in records}
        table.clear()
        selected_index = 0
        requested_id = self.selected_run_id
        if requested_id:
            try:
                requested_id = self.store.resolve_run_id(requested_id)
            except (KeyError, ValueError):
                requested_id = None
        for index, record in enumerate(records):
            goal = " ".join(record.goal.split())
            table.add_row(
                record.id[:12],
                record.status,
                _short_time(record.updated_at),
                _truncate(goal, 52),
                key=record.id,
            )
            if record.id == requested_id:
                selected_index = index
        self.query_one("#runs-message", Static).update(
            f"{len(records)} run(s) • newest first"
            if records
            else "No runs recorded in this workspace."
        )
        if records:
            table.move_cursor(row=selected_index, animate=False)
            self._show_detail(records[selected_index].id)
        else:
            self.selected_run_id = None
            self.query_one("#run-detail", Static).update("No run selected.")
            self._set_action_buttons(None)

    def _show_detail(self, run_id: str) -> None:
        detail = self.store.detail(run_id)
        self.selected_run_id = str(detail["id"])
        self.query_one("#run-detail", Static).update(Text(_format_run_detail(detail)))
        self._set_action_buttons(str(detail["status"]))

    def _set_action_buttons(self, status: str | None) -> None:
        self.query_one("#run-export", Button).disabled = status is None
        resume = self.query_one("#run-resume", Button)
        resume.label = "Continue run" if status == "completed" else "Resume run"
        resume.disabled = status in {None, "blocked"}
        self.query_one("#run-retry", Button).disabled = status != "blocked"

    def _export_selected(self) -> None:
        if not self.selected_run_id:
            return
        try:
            output = self.export_run(self.selected_run_id)
        except Exception as exc:
            self.query_one("#runs-message", Static).update(
                Text(f"Export failed: {type(exc).__name__}: {exc}", style="bold red")
            )
            return
        self.query_one("#runs-message", Static).update(f"Exported to {output}")
        self.notify(f"Exported {output.name}")


def _create_backend(config: AxiomConfig, approve: ApprovalCallback) -> AxiomApp:
    return AxiomApp(config, approve=approve)


class AxiomTUI(App[int]):
    TITLE = "Axiom"
    SUB_TITLE = "Code agent"
    HORIZONTAL_BREAKPOINTS = [(0, "-narrow"), (100, "-wide")]
    BINDINGS = [
        Binding("ctrl+n", "new_run", "New run"),
        Binding("ctrl+r", "show_runs", "Runs"),
        Binding("ctrl+l", "clear_chat", "Clear"),
        Binding("escape", "cancel_task", "Stop"),
        Binding("ctrl+q", "quit", "Quit"),
    ]
    CSS = """
    Screen { background: #0b0f14; color: #d8dee9; }
    Header { background: #111923; color: #e6edf3; }
    Footer { background: #111923; color: #9fb0c3; }
    #body { height: 1fr; }
    #conversation { width: 1fr; padding: 0 1; scrollbar-gutter: stable; }
    #activity-pane {
        width: 38;
        padding: 1;
        background: #0f1620;
        border-left: solid #263445;
    }
    Screen.-narrow #activity-pane { display: none; }
    #activity-title { height: 1; color: #7dd3fc; text-style: bold; }
    #activity { height: 1fr; margin-top: 1; background: #0f1620; }
    .message {
        height: auto;
        min-height: 3;
        margin: 1 2;
        padding: 1 2;
        background: #111923;
        border: round #334155;
    }
    .user { margin-left: 10; background: #13233a; border: round #2563a6; }
    .assistant { margin-right: 5; border: round #2f766d; }
    .system { min-height: 1; color: #93a4b7; border: none; background: transparent; }
    .error { color: #ffb4ab; border: round #a63d40; }
    #status-row { height: 3; padding: 0 2; background: #0f1620; }
    #busy-indicator { width: 5; height: 1; margin-top: 1; color: #7dd3fc; }
    #status { width: 1fr; height: 1; margin-top: 1; color: #9fb0c3; }
    #composer { height: 9; padding: 0 1 1 1; background: #0f1620; }
    #prompt { height: 5; border: round #334155; background: #111923; }
    #prompt:focus { border: round #38bdf8; }
    #composer-actions { height: 3; align: right middle; }
    #composer-actions Button { width: 14; margin-left: 1; }
    ApprovalScreen, RunHistoryScreen { align: center middle; background: #000000 65%; }
    #approval-dialog {
        width: 78;
        max-width: 92%;
        height: auto;
        padding: 1 2;
        background: #111923;
        border: round #f59e0b;
    }
    #approval-title { height: 2; color: #fbbf24; text-style: bold; }
    #approval-reason { height: auto; margin-bottom: 1; }
    #approval-action {
        height: auto;
        max-height: 12;
        padding: 1;
        background: #0b0f14;
        border: solid #334155;
        overflow-y: auto;
    }
    #approval-buttons { height: 3; align: right middle; margin-top: 1; }
    #approval-buttons Button { width: 16; margin-left: 1; }
    #runs-dialog {
        width: 112;
        max-width: 96%;
        height: 90%;
        padding: 1 2;
        background: #111923;
        border: round #38bdf8;
    }
    #runs-title { height: 2; color: #7dd3fc; text-style: bold; }
    #runs-content { height: 1fr; }
    #run-table { width: 58%; border: solid #334155; }
    #run-detail-scroll {
        width: 42%;
        padding: 0 1;
        margin-left: 1;
        background: #0b0f14;
        border: solid #334155;
    }
    #run-detail { height: auto; padding: 1; }
    #runs-message { height: 2; padding-top: 1; color: #9fb0c3; }
    #runs-buttons { height: 3; align: right middle; }
    #runs-buttons Button { width: 16; margin-left: 1; }
    Screen.-narrow #runs-dialog { width: 96%; height: 94%; }
    Screen.-narrow #runs-content { layout: vertical; }
    Screen.-narrow #run-table { width: 1fr; height: 55%; }
    Screen.-narrow #run-detail-scroll { width: 1fr; height: 45%; margin-left: 0; }
    """

    def __init__(
        self,
        config: AxiomConfig,
        *,
        auto_approve: bool = False,
        backend_factory: BackendFactory = _create_backend,
    ) -> None:
        self.axiom_config = config
        self.auto_approve = auto_approve
        self.backend_factory = backend_factory
        self.backend: Any | None = None
        self.conversation_id: str | None = None
        self.agent_worker: Worker[None] | None = None
        self.ready = False
        self.busy = False
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with VerticalScroll(id="conversation"):
                welcome = Static(
                    Text("Axiom is starting. Enter or Ctrl+Enter sends; Ctrl+J adds a new line."),
                    id="welcome",
                    classes="message system",
                )
                welcome.border_title = "WELCOME"
                yield welcome
            with Vertical(id="activity-pane"):
                yield Label("ACTIVITY", id="activity-title")
                yield RichLog(id="activity", wrap=True, markup=False)
        with Horizontal(id="status-row"):
            yield LoadingIndicator(id="busy-indicator")
            yield Static("Starting…", id="status")
        with Vertical(id="composer"):
            yield PromptArea(
                id="prompt",
                placeholder="Ask Axiom to inspect, edit, or test this workspace…",
                disabled=True,
            )
            with Horizontal(id="composer-actions"):
                yield Button("Runs", id="runs")
                yield Button("New run", id="new")
                yield Button("Stop", id="stop", variant="error", disabled=True)
                yield Button("Send", id="send", variant="primary", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        workspace = self.axiom_config.workspace.root
        self.sub_title = f"{self.axiom_config.model.name} • {workspace}"
        self.run_worker(
            self._start_backend(),
            name="startup",
            group="startup",
            exit_on_error=False,
        )

    async def _start_backend(self) -> None:
        try:
            self.backend = self.backend_factory(self.axiom_config, self._approve)
            self.backend.events.subscribe(self._on_agent_event)
            await self.backend.start()
        except Exception as exc:
            await self._append_error(f"Startup failed: {type(exc).__name__}: {exc}")
            self._set_busy(False, "Startup failed")
            self._set_status("Startup failed", "bold red")
            return
        self.ready = True
        self._set_busy(False, "Ready")
        self.query_one("#welcome", Static).update(
            "Ready. Enter or Ctrl+Enter sends; Ctrl+J adds a new line. Type /help for commands."
        )
        model = f"{self.axiom_config.model.provider}:{self.axiom_config.model.name}"
        self._activity("READY", model)
        self.query_one("#prompt", PromptArea).focus()

    async def on_unmount(self) -> None:
        if self.backend is not None:
            await self.backend.close()

    @on(PromptArea.Submitted)
    async def submit_from_keyboard(self, event: PromptArea.Submitted) -> None:
        await self._submit(event.value)

    @on(Button.Pressed)
    async def handle_button(self, event: Button.Pressed) -> None:
        if event.button.id == "send":
            await self._submit(self.query_one("#prompt", PromptArea).text)
        elif event.button.id == "stop":
            self.action_cancel_task()
        elif event.button.id == "new":
            await self.action_new_run()
        elif event.button.id == "runs":
            self.action_show_runs()

    async def _submit(self, raw: str) -> None:
        text = raw.strip()
        if not text:
            return
        if not self.ready:
            self.notify("Axiom is still starting", severity="warning")
            return
        if self.busy:
            self.notify("A task is already running", severity="warning")
            return
        prompt = self.query_one("#prompt", PromptArea)
        prompt.clear()
        if text.startswith("/") and "\n" not in text:
            await self._slash_command(text)
            return
        await self._append_user(text)
        self._set_busy(True, "Planning…")
        self.agent_worker = self.run_worker(
            self._run_goal(text),
            name="agent",
            group="agent",
            exit_on_error=False,
        )

    async def _slash_command(self, command: str) -> None:
        name, _, value = command.strip().partition(" ")
        name = name.casefold()
        value = value.strip()
        if name in {"/exit", "/quit"}:
            await self.action_quit()
        elif name == "/new":
            await self.action_new_run()
        elif name == "/clear":
            await self.action_clear_chat()
        elif name in {"/runs", "/show"}:
            self.action_show_runs(value or None)
        elif name in {"/resume", "/retry"}:
            if not value:
                await self._append_error(f"Usage: {name} RUN_ID")
            else:
                self._schedule_resume(value, retry_uncertain=name == "/retry")
        elif name == "/export":
            if not value:
                await self._append_error("Usage: /export RUN_ID")
            else:
                try:
                    output = self._export_run(value)
                except Exception as exc:
                    await self._append_error(f"Export failed: {type(exc).__name__}: {exc}")
                else:
                    await self._append_system(f"Exported run events to {output}")
        elif name == "/help":
            await self._append_system(
                "Commands: /runs, /show RUN_ID, /resume RUN_ID, /retry RUN_ID, "
                "/export RUN_ID, /new, /clear, /help, /exit"
            )
        else:
            await self._append_error(f"Unknown command: {name}")

    async def _run_goal(self, goal: str) -> None:
        status = "Ready"
        try:
            if self.backend is None:
                raise RuntimeError("Axiom backend is not ready")
            result = await self.backend.agent.run(goal, conversation_id=self.conversation_id)
            self.conversation_id = result.conversation_id
            await self._append_assistant(result.output)
            usage = _format_usage(result.usage)
            outcome = "Completed" if result.success else "Incomplete"
            self._activity("DONE", f"{outcome}{f' • {usage}' if usage else ''}")
            status = f"{outcome}{f' • {usage}' if usage else ''}"
        except asyncio.CancelledError:
            status = "Task stopped"
            raise
        except Exception as exc:
            await self._append_error(f"{type(exc).__name__}: {exc}")
            self._activity("ERROR", f"{type(exc).__name__}: {exc}", "bold red")
            status = "Task failed"
        finally:
            self.agent_worker = None
            self._set_busy(False, status)

    async def _resume_run(self, run_id: str, *, retry_uncertain_tools: bool) -> None:
        status = "Ready"
        try:
            if self.backend is None:
                raise RuntimeError("Axiom backend is not ready")
            result = await self.backend.agent.resume(
                run_id, retry_uncertain_tools=retry_uncertain_tools
            )
            self.conversation_id = result.conversation_id
            await self._append_assistant(result.output)
            usage = _format_usage(result.usage)
            outcome = "Recovered" if result.success else "Recovery incomplete"
            self._activity("DONE", f"{outcome}{f' • {usage}' if usage else ''}")
            status = f"{outcome}{f' • {usage}' if usage else ''}"
        except asyncio.CancelledError:
            status = "Recovery stopped"
            raise
        except UnsafeResumeError as exc:
            await self._append_error(str(exc))
            self._activity("BLOCKED", str(exc), "bold yellow")
            status = "Recovery blocked"
        except Exception as exc:
            await self._append_error(f"{type(exc).__name__}: {exc}")
            self._activity("ERROR", f"{type(exc).__name__}: {exc}", "bold red")
            status = "Recovery failed"
        finally:
            self.agent_worker = None
            self._set_busy(False, status)

    def action_show_runs(self, selected_run_id: str | None = None) -> None:
        if not self.ready or self.backend is None:
            self.notify("Axiom is still starting", severity="warning")
            return
        if self.busy:
            self.notify("Stop the active task before opening run history", severity="warning")
            return
        self.run_worker(
            self._show_runs(selected_run_id),
            name="run-history",
            group="run-history",
            exclusive=True,
            exit_on_error=False,
        )

    async def _show_runs(self, selected_run_id: str | None) -> None:
        if self.backend is None:
            return
        screen = RunHistoryScreen(
            self.backend.execution,
            self._export_run,
            selected_run_id=selected_run_id,
        )
        action = await self.push_screen_wait(screen)
        if action is not None:
            await self._request_resume(
                action.run_id, retry_uncertain=action.kind == "retry"
            )

    def _schedule_resume(self, run_id: str, *, retry_uncertain: bool) -> None:
        if not self.ready or self.backend is None:
            self.notify("Axiom is still starting", severity="warning")
            return
        if self.busy:
            self.notify("A task is already running", severity="warning")
            return
        self.run_worker(
            self._request_resume(run_id, retry_uncertain=retry_uncertain),
            name="resume-request",
            group="run-history",
            exclusive=True,
            exit_on_error=False,
        )

    async def _request_resume(self, run_id: str, *, retry_uncertain: bool) -> None:
        if self.backend is None:
            return
        try:
            record = self.backend.execution.get_run(run_id)
        except (KeyError, ValueError) as exc:
            await self._append_error(str(exc))
            return
        if retry_uncertain and record.status != "blocked":
            await self._append_error(
                f"Run {record.id[:12]} is {record.status}, not blocked; use /resume RUN_ID."
            )
            return
        if record.status == "completed":
            await self._load_run_context(
                record,
                notice=(
                    f"Continued run {record.id[:12]}. Its saved context is active; "
                    "send a prompt to continue."
                ),
            )
            self._activity("RUN", f"{record.id[:12]} • context restored", "bold cyan")
            return
        if record.status == "blocked" and not retry_uncertain:
            await self._append_error(
                f"Run {record.id[:12]} is blocked because a tool outcome is uncertain. "
                "Open Runs and choose Retry blocked, or use /retry RUN_ID."
            )
            return
        if retry_uncertain:
            approved = await self._confirm_uncertain_retry(
                f"Retry uncertain tools for run {record.id[:12]}",
                "The interrupted tool may already have changed files or external systems. "
                "Retrying can repeat those side effects. Inspect the run and workspace first.",
            )
            if not approved:
                await self._append_system("Blocked-run retry was cancelled.")
                return
        await self._load_run_context(
            record,
            notice=(
                f"Resuming run {record.id[:12]}"
                f"{' with uncertain-tool retry' if retry_uncertain else ''}."
            ),
        )
        self._set_busy(True, "Recovering…")
        self.agent_worker = self.run_worker(
            self._resume_run(record.id, retry_uncertain_tools=retry_uncertain),
            name="agent-resume",
            group="agent",
            exit_on_error=False,
        )

    def _export_run(self, run_id: str) -> Path:
        if self.backend is None:
            raise RuntimeError("Axiom backend is not ready")
        resolved = self.backend.execution.resolve_run_id(run_id)
        rows = self.backend.execution.event_rows(resolved)
        output = (
            self.axiom_config.workspace.root
            / ".axiom"
            / "exports"
            / f"{resolved}-events.jsonl"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
            newline="",
        )
        return output

    async def _confirm_uncertain_retry(self, action: str, reason: str) -> bool:
        # Unlike ordinary policy approvals, uncertain replay is never bypassed by --yes.
        screen = cast(Screen[object], ApprovalScreen(action, reason))
        return bool(await self.push_screen_wait(screen))

    async def _load_run_context(self, record: RunRecord, *, notice: str) -> None:
        messages: list[dict[str, Any]] = []
        memory = getattr(self.backend, "memory", None)
        if memory is not None and hasattr(memory, "recent_messages"):
            messages = list(
                memory.recent_messages(record.conversation_id, RUN_TRANSCRIPT_LIMIT)
            )
        await self._clear_run_view()
        for message in messages:
            role = str(message.get("role", "system"))
            content = str(message.get("content", ""))
            if role == "user":
                await self._append_user(content)
            elif role == "assistant":
                await self._append_assistant(content)
            else:
                await self._append_system(f"{role.upper()}: {content}")
        self.conversation_id = record.conversation_id
        await self._append_system(notice)
        self._set_status(f"Run {record.id[:12]} selected")
        self.query_one("#prompt", PromptArea).focus()

    def _on_agent_event(self, event: Event) -> None:
        data = event.data
        if event.type in {"agent.started", "agent.resumed"}:
            label = "RESUME" if event.type == "agent.resumed" else "RUN"
            self._activity(label, str(data.get("run_id", ""))[:12], "bold cyan")
        elif event.type == "plan.created":
            steps = data.get("plan", {}).get("steps", [])
            titles = " → ".join(str(step.get("title", "step")) for step in steps)
            self._activity("PLAN", titles or "Direct execution", "bold cyan")
        elif event.type == "step.started":
            title = str(data.get("title", "step"))
            self._activity("STEP", title, "bold blue")
            self._set_status(f"Working • {title}")
        elif event.type == "step.completed":
            self._activity("OK", str(data.get("step_id", "step")), "green")
        elif event.type == "model.started":
            self._set_status(f"Thinking • turn {data.get('turn', '?')}")
        elif event.type == "tool.started":
            name = str(data.get("name", "tool"))
            self._activity("TOOL", name, "bold magenta")
            self._set_status(f"Running tool • {name}")
        elif event.type == "tool.completed":
            self._activity("OK", str(data.get("name", "tool")), "green")
        elif event.type in {"tool.failed", "step.failed", "mcp.failed"}:
            self._activity("ERROR", str(data.get("error", event.type)), "bold red")
        elif event.type == "agent.blocked":
            self._activity("BLOCKED", str(data.get("error", event.type)), "bold yellow")
        elif event.type == "mcp.connected":
            self._activity(
                "MCP", f"{data.get('server', 'server')} • {data.get('tools', 0)} tools", "cyan"
            )

    def _activity(self, label: str, message: str, style: str = "bold") -> None:
        line = Text()
        line.append(f"{label:<7}", style=style)
        line.append(message[:500])
        self.query_one("#activity", RichLog).write(line)

    def _set_busy(self, busy: bool, status: str) -> None:
        self.busy = busy
        self.query_one("#busy-indicator", LoadingIndicator).display = busy
        self.query_one("#prompt", PromptArea).disabled = busy or not self.ready
        self.query_one("#send", Button).disabled = busy or not self.ready
        self.query_one("#new", Button).disabled = busy or not self.ready
        self.query_one("#runs", Button).disabled = busy or not self.ready
        self.query_one("#stop", Button).disabled = not busy
        self._set_status(status)

    def _set_status(self, status: str, style: str = "") -> None:
        self.query_one("#status", Static).update(Text(status, style=style))

    async def _append_user(self, text: str) -> None:
        widget = Static(Text(text), classes="message user")
        widget.border_title = "YOU"
        await self._append(widget)

    async def _append_assistant(self, text: str) -> None:
        widget = Markdown(text or "_(No response)_", classes="message assistant")
        widget.border_title = "AXIOM"
        await self._append(widget)

    async def _append_system(self, text: str) -> None:
        widget = Static(Text(text), classes="message system")
        widget.border_title = "SYSTEM"
        await self._append(widget)

    async def _append_error(self, text: str) -> None:
        widget = Static(Text(text), classes="message error")
        widget.border_title = "ERROR"
        await self._append(widget)

    async def _append(self, widget: Static | Markdown) -> None:
        conversation = self.query_one("#conversation", VerticalScroll)
        await conversation.mount(widget)
        conversation.scroll_end(animate=False, force=True)

    async def _approve(self, action: str, reason: str) -> bool:
        if self.auto_approve:
            return True
        screen = cast(Screen[object], ApprovalScreen(action, reason))
        return bool(await self.push_screen_wait(screen))

    async def action_new_run(self) -> None:
        if self.busy:
            self.notify("Stop the active task before starting a new run", severity="warning")
            return
        self.conversation_id = None
        await self._clear_run_view()
        await self._append_system("Started a new run.")
        self._set_status("Ready • new run")
        self.query_one("#prompt", PromptArea).focus()

    async def _clear_run_view(self) -> None:
        await self.query_one("#conversation", VerticalScroll).remove_children()
        self.query_one("#activity", RichLog).clear()
        self.query_one("#prompt", PromptArea).clear()

    async def action_clear_chat(self) -> None:
        if self.busy:
            self.notify("Stop the active task before clearing the chat", severity="warning")
            return
        await self.query_one("#conversation", VerticalScroll).remove_children()
        await self._append_system("Chat display cleared. Saved run context is unchanged.")

    def action_cancel_task(self) -> None:
        if self.agent_worker is not None:
            if self.backend is not None and hasattr(self.backend.agent, "request_cancel"):
                self.backend.agent.request_cancel()
            self.agent_worker.cancel()
            self._set_status("Stopping…")

    async def action_quit(self) -> None:
        if self.agent_worker is not None:
            if self.backend is not None and hasattr(self.backend.agent, "request_cancel"):
                self.backend.agent.request_cancel()
            self.agent_worker.cancel()
        self.exit(0)


def _short_time(value: str) -> str:
    return value.replace("T", " ")[:19]


def _truncate(value: str, limit: int) -> str:
    return value if len(value) <= limit else f"{value[: limit - 1]}…"


def _format_run_detail(detail: dict[str, Any]) -> str:
    lines = [
        f"Run      {str(detail['id'])[:12]}",
        f"Status   {detail['status']}",
        f"Started  {_short_time(str(detail['started_at']))}",
        f"Updated  {_short_time(str(detail['updated_at']))}",
    ]
    model = detail.get("context", {}).get("model", {})
    if model:
        lines.append(f"Model    {model.get('provider')}:{model.get('name')}")
    lines.extend(("", "Goal", str(detail["goal"])))
    if detail.get("error"):
        lines.extend(("", "Error", str(detail["error"])))

    lines.extend(("", "Steps"))
    steps = (detail.get("plan") or {}).get("steps", [])
    if steps:
        for step in steps:
            lines.append(
                f"[{step.get('status', '?')}] {step.get('id', 'step')} — "
                f"{step.get('title', '')}"
            )
    else:
        lines.append("(No plan recorded)")

    lines.extend(("", "Model metrics"))
    metrics = detail.get("metrics", {})
    for stage in ("planner", "executor", "finalizer", "total"):
        values = metrics.get(stage, {})
        lines.append(
            f"{stage:<9} calls={values.get('model_calls', 0)} "
            f"failed={values.get('failed_calls', 0)} "
            f"tokens={values.get('total_tokens', 0)} "
            f"latency={values.get('duration_ms', 0)}ms"
        )

    tools = detail.get("tool_calls", [])
    lines.extend(("", f"Tool calls ({len(tools)})"))
    if tools:
        for tool in tools:
            lines.append(f"[{tool.get('status', '?')}] {tool.get('name', 'tool')}")
    else:
        lines.append("(None)")

    attempts = detail.get("attempts", [])
    lines.extend(("", f"Attempts ({len(attempts)})"))
    for attempt in attempts:
        lines.append(
            f"[{attempt.get('status', '?')}] {attempt.get('step', 'step')} "
            f"#{attempt.get('number', '?')}"
        )
    return "\n".join(lines)


def _format_usage(usage: dict[str, int]) -> str:
    return " • ".join(f"{key}={value}" for key, value in sorted(usage.items()) if value)


async def run_tui(
    config: AxiomConfig,
    *,
    auto_approve: bool = False,
    backend_factory: BackendFactory = _create_backend,
) -> int:
    result = await AxiomTUI(
        config,
        auto_approve=auto_approve,
        backend_factory=backend_factory,
    ).run_async()
    return result or 0
