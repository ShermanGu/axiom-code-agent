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
from axiom_agent.execution.store import ConversationRecord, ExecutionStore, UnsafeResumeError
from axiom_agent.tools.base import ApprovalCallback

BackendFactory = Callable[[AxiomConfig, ApprovalCallback], Any]
CONVERSATION_TRANSCRIPT_LIMIT = 200


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
class ConversationAction:
    kind: Literal["open", "resume", "retry"]
    conversation_id: str


class ConversationHistoryScreen(ModalScreen[ConversationAction | None]):
    BINDINGS = [Binding("escape", "close", "Close")]

    def __init__(
        self,
        store: ExecutionStore,
        export_conversation: Callable[[str], Path],
        *,
        selected_conversation_id: str | None = None,
    ) -> None:
        self.store = store
        self.export_conversation = export_conversation
        self.selected_conversation_id = selected_conversation_id
        self.records: dict[str, ConversationRecord] = {}
        self.initialized = False
        super().__init__()

    def compose(self) -> ComposeResult:
        with Vertical(id="conversations-dialog"):
            yield Label("CONVERSATIONS", id="conversations-title")
            with Horizontal(id="conversations-content"):
                yield DataTable(id="conversation-table", cursor_type="row")
                with VerticalScroll(id="conversation-detail-scroll"):
                    yield Static(
                        "Select a conversation to inspect it.", id="conversation-detail"
                    )
            yield Static("", id="conversations-message")
            with Horizontal(id="conversations-buttons"):
                yield Button("Export", id="conversation-export", disabled=True)
                yield Button(
                    "Retry blocked", id="conversation-retry", variant="warning", disabled=True
                )
                yield Button(
                    "Resume task",
                    id="conversation-resume",
                    variant="warning",
                    disabled=True,
                )
                yield Button(
                    "Continue",
                    id="conversation-open",
                    variant="primary",
                    disabled=True,
                )
                yield Button("Close", id="conversation-close")

    def on_mount(self) -> None:
        # On some Windows/Python combinations the screen receives Mount before every
        # composed descendant is queryable. Defer child access until the first refresh.
        self.call_after_refresh(self._initialize)

    def _initialize(self) -> None:
        table = self.query_one("#conversation-table", DataTable)
        table.add_columns("Conversation ID", "Status", "Tasks", "Updated", "Title")
        self.initialized = True
        self._refresh()

    @on(DataTable.RowHighlighted)
    def select_conversation(self, event: DataTable.RowHighlighted) -> None:
        conversation_id = str(event.row_key.value)
        if conversation_id in self.records:
            self._show_detail(conversation_id)

    @on(Button.Pressed)
    def handle_button(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "conversation-close":
            self.dismiss(None)
        elif button_id == "conversation-export":
            self._export_selected()
        elif (
            button_id
            in {"conversation-open", "conversation-resume", "conversation-retry"}
            and self.selected_conversation_id
        ):
            kinds: dict[str, Literal["open", "resume", "retry"]] = {
                "conversation-open": "open",
                "conversation-resume": "resume",
                "conversation-retry": "retry",
            }
            kind = kinds[button_id]
            self.dismiss(ConversationAction(kind, self.selected_conversation_id))

    def action_close(self) -> None:
        self.dismiss(None)

    def _refresh(self) -> None:
        if not self.initialized:
            return
        table = self.query_one("#conversation-table", DataTable)
        records = self.store.list_conversations(limit=100)
        self.records = {record.id: record for record in records}
        table.clear()
        selected_index = 0
        requested_id = self.selected_conversation_id
        if requested_id:
            try:
                requested_id = self.store.resolve_conversation_id(requested_id)
            except (KeyError, ValueError):
                requested_id = None
        for index, record in enumerate(records):
            title = " ".join(record.title.split())
            table.add_row(
                record.id[:12],
                record.status,
                str(record.execution_count),
                _short_time(record.updated_at),
                _truncate(title, 44),
                key=record.id,
            )
            if record.id == requested_id:
                selected_index = index
        self.query_one("#conversations-message", Static).update(
            f"{len(records)} conversation(s) • newest first"
            if records
            else "No conversations recorded in this workspace."
        )
        if records:
            table.move_cursor(row=selected_index, animate=False)
            self._show_detail(records[selected_index].id)
        else:
            self.selected_conversation_id = None
            self.query_one("#conversation-detail", Static).update(
                "No conversation selected."
            )
            self._set_action_buttons(None)

    def _show_detail(self, conversation_id: str) -> None:
        detail = self.store.conversation_detail(conversation_id)
        self.selected_conversation_id = str(detail["id"])
        self.query_one("#conversation-detail", Static).update(
            Text(_format_conversation_detail(detail))
        )
        self._set_action_buttons(str(detail["status"]))

    def _set_action_buttons(self, status: str | None) -> None:
        self.query_one("#conversation-export", Button).disabled = status is None
        self.query_one("#conversation-open", Button).disabled = status is None
        self.query_one("#conversation-resume", Button).disabled = status in {
            None,
            "completed",
            "blocked",
        }
        self.query_one("#conversation-retry", Button).disabled = status != "blocked"

    def _export_selected(self) -> None:
        if not self.selected_conversation_id:
            return
        try:
            output = self.export_conversation(self.selected_conversation_id)
        except Exception as exc:
            self.query_one("#conversations-message", Static).update(
                Text(f"Export failed: {type(exc).__name__}: {exc}", style="bold red")
            )
            return
        self.query_one("#conversations-message", Static).update(f"Exported to {output}")
        self.notify(f"Exported {output.name}")


def _create_backend(config: AxiomConfig, approve: ApprovalCallback) -> AxiomApp:
    return AxiomApp(config, approve=approve)


class AxiomTUI(App[int]):
    TITLE = "Axiom"
    SUB_TITLE = "Code agent"
    HORIZONTAL_BREAKPOINTS = [(0, "-narrow"), (100, "-wide")]
    BINDINGS = [
        Binding("ctrl+n", "new_conversation", "New conversation"),
        Binding("ctrl+r", "show_conversations", "Conversations"),
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
    ApprovalScreen, ConversationHistoryScreen { align: center middle; background: #000000 65%; }
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
    #conversations-dialog {
        width: 112;
        max-width: 96%;
        height: 90%;
        padding: 1 2;
        background: #111923;
        border: round #38bdf8;
    }
    #conversations-title { height: 2; color: #7dd3fc; text-style: bold; }
    #conversations-content { height: 1fr; }
    #conversation-table { width: 62%; border: solid #334155; }
    #conversation-detail-scroll {
        width: 42%;
        padding: 0 1;
        margin-left: 1;
        background: #0b0f14;
        border: solid #334155;
    }
    #conversation-detail { height: auto; padding: 1; }
    #conversations-message { height: 2; padding-top: 1; color: #9fb0c3; }
    #conversations-buttons { height: 3; align: right middle; }
    #conversations-buttons Button { width: 16; margin-left: 1; }
    Screen.-narrow #conversations-dialog { width: 96%; height: 94%; }
    Screen.-narrow #conversations-content { layout: vertical; }
    Screen.-narrow #conversation-table { width: 1fr; height: 55%; }
    Screen.-narrow #conversation-detail-scroll {
        width: 1fr;
        height: 45%;
        margin-left: 0;
    }
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
                yield Button("Conversations", id="conversations")
                yield Button("New conversation", id="new")
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
            await self.action_new_conversation()
        elif event.button.id == "conversations":
            self.action_show_conversations()

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
            await self.action_new_conversation()
        elif name == "/clear":
            await self.action_clear_chat()
        elif name in {"/conversations", "/show"}:
            self.action_show_conversations(value or None)
        elif name == "/continue":
            if not value:
                await self._append_error(f"Usage: {name} CONVERSATION_ID")
            else:
                await self._open_conversation(value)
        elif name in {"/resume", "/retry"}:
            if not value:
                await self._append_error(f"Usage: {name} CONVERSATION_ID")
            else:
                self._schedule_resume(value, retry_uncertain=name == "/retry")
        elif name == "/export":
            if not value:
                await self._append_error("Usage: /export CONVERSATION_ID")
            else:
                try:
                    output = self._export_conversation(value)
                except Exception as exc:
                    await self._append_error(f"Export failed: {type(exc).__name__}: {exc}")
                else:
                    await self._append_system(f"Exported conversation events to {output}")
        elif name == "/help":
            await self._append_system(
                "Commands: /conversations, /show CONVERSATION_ID, "
                "/continue CONVERSATION_ID, /resume CONVERSATION_ID, "
                "/retry CONVERSATION_ID, "
                "/export CONVERSATION_ID, /new, /clear, /help, /exit"
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
            status = (
                f"{outcome} • conversation {result.conversation_id[:12]}"
                f"{f' • {usage}' if usage else ''}"
            )
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

    async def _resume_conversation(
        self, conversation_id: str, *, retry_uncertain_tools: bool
    ) -> None:
        status = "Ready"
        try:
            if self.backend is None:
                raise RuntimeError("Axiom backend is not ready")
            result = await self.backend.agent.resume_conversation(
                conversation_id, retry_uncertain_tools=retry_uncertain_tools
            )
            self.conversation_id = result.conversation_id
            await self._append_assistant(result.output)
            usage = _format_usage(result.usage)
            outcome = "Recovered" if result.success else "Recovery incomplete"
            self._activity("DONE", f"{outcome}{f' • {usage}' if usage else ''}")
            status = (
                f"{outcome} • conversation {result.conversation_id[:12]}"
                f"{f' • {usage}' if usage else ''}"
            )
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

    def action_show_conversations(
        self, selected_conversation_id: str | None = None
    ) -> None:
        if not self.ready or self.backend is None:
            self.notify("Axiom is still starting", severity="warning")
            return
        if self.busy:
            self.notify(
                "Stop the active task before opening conversations", severity="warning"
            )
            return
        self.run_worker(
            self._show_conversations(selected_conversation_id),
            name="conversation-history",
            group="conversation-history",
            exclusive=True,
            exit_on_error=False,
        )

    async def _show_conversations(self, selected_conversation_id: str | None) -> None:
        if self.backend is None:
            return
        screen = ConversationHistoryScreen(
            self.backend.execution,
            self._export_conversation,
            selected_conversation_id=selected_conversation_id,
        )
        action = await self.push_screen_wait(screen)
        if action is not None:
            if action.kind == "open":
                await self._open_conversation(action.conversation_id)
            else:
                await self._request_resume(
                    action.conversation_id, retry_uncertain=action.kind == "retry"
                )

    def _schedule_resume(
        self, conversation_id: str, *, retry_uncertain: bool
    ) -> None:
        if not self.ready or self.backend is None:
            self.notify("Axiom is still starting", severity="warning")
            return
        if self.busy:
            self.notify("A task is already running", severity="warning")
            return
        self.run_worker(
            self._request_resume(conversation_id, retry_uncertain=retry_uncertain),
            name="conversation-request",
            group="conversation-history",
            exclusive=True,
            exit_on_error=False,
        )

    async def _open_conversation(self, conversation_id: str) -> None:
        if self.backend is None:
            return
        try:
            conversation = self.backend.execution.get_conversation(conversation_id)
        except (KeyError, ValueError) as exc:
            await self._append_error(str(exc))
            return
        if conversation.status == "completed":
            notice = (
                f"Continued conversation {conversation.id[:12]}. Its saved context is active; "
                "send a prompt to continue."
            )
        else:
            notice = (
                f"Opened conversation {conversation.id[:12]}. Its latest task is "
                f"{conversation.status} and was not resumed; choose Resume task to recover it, "
                "or send a new prompt to move on."
            )
        await self._load_conversation_context(
            conversation.id,
            notice=notice,
        )
        self._activity("CHAT", f"{conversation.id[:12]} • context restored", "bold cyan")

    async def _request_resume(
        self, conversation_id: str, *, retry_uncertain: bool
    ) -> None:
        if self.backend is None:
            return
        try:
            conversation = self.backend.execution.get_conversation(conversation_id)
            record = self.backend.execution.latest_run_for_conversation(conversation.id)
        except (KeyError, ValueError) as exc:
            await self._append_error(str(exc))
            return
        if retry_uncertain and record.status != "blocked":
            await self._append_error(
                f"Conversation {conversation.id[:12]} is {record.status}, not blocked; "
                "use /resume CONVERSATION_ID to recover its latest task."
            )
            return
        if record.status == "completed":
            await self._append_error(
                f"Conversation {conversation.id[:12]} has no incomplete task. "
                "Use /continue CONVERSATION_ID to continue chatting."
            )
            return
        if record.status == "blocked" and not retry_uncertain:
            await self._append_error(
                f"Conversation {conversation.id[:12]} is blocked because a tool outcome is "
                "uncertain. Open Conversations and choose Retry blocked, or use "
                "/retry CONVERSATION_ID."
            )
            return
        if retry_uncertain:
            approved = await self._confirm_uncertain_retry(
                f"Retry uncertain tools for conversation {conversation.id[:12]}",
                "The interrupted tool may already have changed files or external systems. "
                "Retrying can repeat those side effects. Inspect the conversation and workspace "
                "first.",
            )
            if not approved:
                await self._append_system("Blocked-conversation retry was cancelled.")
                return
        await self._load_conversation_context(
            conversation.id,
            notice=(
                f"Resuming conversation {conversation.id[:12]}"
                f"{' with uncertain-tool retry' if retry_uncertain else ''}."
            ),
        )
        self._set_busy(True, "Recovering…")
        self.agent_worker = self.run_worker(
            self._resume_conversation(
                conversation.id, retry_uncertain_tools=retry_uncertain
            ),
            name="conversation-resume",
            group="agent",
            exit_on_error=False,
        )

    def _export_conversation(self, conversation_id: str) -> Path:
        if self.backend is None:
            raise RuntimeError("Axiom backend is not ready")
        resolved = self.backend.execution.resolve_conversation_id(conversation_id)
        rows = self.backend.execution.conversation_event_rows(resolved)
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

    async def _load_conversation_context(
        self, conversation_id: str, *, notice: str
    ) -> None:
        messages: list[dict[str, Any]] = []
        memory = getattr(self.backend, "memory", None)
        if memory is not None and hasattr(memory, "recent_messages"):
            messages = list(
                memory.recent_messages(
                    conversation_id, CONVERSATION_TRANSCRIPT_LIMIT
                )
            )
        await self._clear_conversation_view()
        for message in messages:
            role = str(message.get("role", "system"))
            content = str(message.get("content", ""))
            if role == "user":
                await self._append_user(content)
            elif role == "assistant":
                await self._append_assistant(content)
            else:
                await self._append_system(f"{role.upper()}: {content}")
        self.conversation_id = conversation_id
        await self._append_system(notice)
        self._set_status(f"Conversation {conversation_id[:12]} selected")
        self.query_one("#prompt", PromptArea).focus()

    def _on_agent_event(self, event: Event) -> None:
        data = event.data
        if event.type in {"agent.started", "agent.resumed"}:
            label = "RESUME" if event.type == "agent.resumed" else "TASK"
            conversation_id = str(data.get("conversation_id", self.conversation_id or ""))
            message = (
                f"conversation {conversation_id[:12]}"
                if conversation_id
                else "execution started"
            )
            self._activity(label, message, "bold cyan")
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
        self.query_one("#conversations", Button).disabled = busy or not self.ready
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

    async def action_new_conversation(self) -> None:
        if self.busy:
            self.notify(
                "Stop the active task before starting a new conversation",
                severity="warning",
            )
            return
        self.conversation_id = None
        await self._clear_conversation_view()
        await self._append_system("Started a new conversation.")
        self._set_status("Ready • new conversation")
        self.query_one("#prompt", PromptArea).focus()

    async def _clear_conversation_view(self) -> None:
        await self.query_one("#conversation", VerticalScroll).remove_children()
        self.query_one("#activity", RichLog).clear()
        self.query_one("#prompt", PromptArea).clear()

    async def action_clear_chat(self) -> None:
        if self.busy:
            self.notify("Stop the active task before clearing the chat", severity="warning")
            return
        await self.query_one("#conversation", VerticalScroll).remove_children()
        await self._append_system(
            "Chat display cleared. Saved conversation context is unchanged."
        )

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


def _format_conversation_detail(detail: dict[str, Any]) -> str:
    lines = [
        f"Conversation  {detail['id']}",
        f"Status        {detail['status']}",
        f"Tasks         {detail['execution_count']}",
        f"Started       {_short_time(str(detail['created_at']))}",
        f"Updated       {_short_time(str(detail['updated_at']))}",
    ]
    lines.extend(("", "Title", str(detail["title"])))

    lines.extend(("", "Conversation metrics"))
    metrics = detail.get("metrics", {})
    for stage in ("planner", "executor", "finalizer", "total"):
        values = metrics.get(stage, {})
        lines.append(
            f"{stage:<9} calls={values.get('model_calls', 0)} "
            f"failed={values.get('failed_calls', 0)} "
            f"tokens={values.get('total_tokens', 0)} "
            f"latency={values.get('duration_ms', 0)}ms"
        )

    executions = detail.get("executions", [])
    lines.extend(("", f"Tasks ({len(executions)})"))
    for execution in executions:
        sequence = execution.get("sequence", "?")
        status = execution.get("status", "?")
        lines.append(f"#{sequence} [{status}] {execution.get('goal', '')}")
        steps = (execution.get("plan") or {}).get("steps", [])
        tools = execution.get("tool_calls", [])
        turns = execution.get("turns", [])
        lines.append(
            f"    steps={len(steps)} model_calls={len(turns)} tool_calls={len(tools)}"
        )
        if execution.get("error"):
            lines.append(f"    error: {execution['error']}")
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
