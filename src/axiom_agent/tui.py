from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, cast

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
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
from axiom_agent.tools.base import ApprovalCallback

BackendFactory = Callable[[AxiomConfig, ApprovalCallback], Any]


class PromptArea(TextArea):
    BINDINGS = [Binding("ctrl+enter", "submit", "Send", show=False)]

    class Submitted(Message):
        def __init__(self, value: str) -> None:
            self.value = value
            super().__init__()

    def action_submit(self) -> None:
        self.post_message(self.Submitted(self.text))


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


def _create_backend(config: AxiomConfig, approve: ApprovalCallback) -> AxiomApp:
    return AxiomApp(config, approve=approve)


class AxiomTUI(App[int]):
    TITLE = "Axiom"
    SUB_TITLE = "Code agent"
    HORIZONTAL_BREAKPOINTS = [(0, "-narrow"), (100, "-wide")]
    BINDINGS = [
        Binding("ctrl+n", "new_thread", "New thread"),
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
    ApprovalScreen { align: center middle; background: #000000 65%; }
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
                    Text("Axiom is starting. Ctrl+Enter sends; Enter adds a new line."),
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
                yield Button("New thread", id="new")
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
            "Ready. Ctrl+Enter sends; Enter adds a new line. Type /help for commands."
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
            await self.action_new_thread()

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
            await self._slash_command(text.casefold())
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
        if command in {"/exit", "/quit"}:
            await self.action_quit()
        elif command == "/new":
            await self.action_new_thread()
        elif command == "/clear":
            await self.action_clear_chat()
        elif command == "/help":
            await self._append_system("Commands: /new, /clear, /help, /exit")
        else:
            await self._append_error(f"Unknown command: {command}")

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

    def _on_agent_event(self, event: Event) -> None:
        data = event.data
        if event.type == "plan.created":
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

    async def action_new_thread(self) -> None:
        if self.busy:
            self.notify("Stop the active task before starting a new thread", severity="warning")
            return
        self.conversation_id = None
        await self._append_system("Started a new conversation thread.")

    async def action_clear_chat(self) -> None:
        if self.busy:
            self.notify("Stop the active task before clearing the chat", severity="warning")
            return
        await self.query_one("#conversation", VerticalScroll).remove_children()
        await self._append_system("Chat display cleared. Conversation memory is unchanged.")

    def action_cancel_task(self) -> None:
        if self.agent_worker is not None:
            self.agent_worker.cancel()
            self._set_status("Stopping…")

    async def action_quit(self) -> None:
        if self.agent_worker is not None:
            self.agent_worker.cancel()
        self.exit(0)


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
