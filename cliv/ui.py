"""UI boundary for the agent.

The core agent should not know about Rich, spinners, or ``input()``.
``ConsoleUI`` is the terminal implementation; tests can pass ``NullUI``
or a tiny fake that records approval requests.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, Optional, Protocol

from rich.console import Console
from rich.markup import escape

from cliv.diff_view import render_preview
from cliv.models import SessionStats


class UI(Protocol):
    def status(self, message: str, spinner: str = "dots"):
        ...

    def print(self, message: str = "", **kwargs: Any) -> None:
        ...

    def print_preview(self, preview: str) -> None:
        ...

    def confirm(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        preview: Optional[str] = None,
    ) -> bool:
        ...


class NullUI:
    """No-op UI used by tests and non-interactive callers."""

    def __init__(self, approve: bool = False):
        self.approve = approve
        self.approvals: list[tuple[str, Dict[str, Any], Optional[str]]] = []

    def status(self, message: str, spinner: str = "dots"):
        return nullcontext()

    def print(self, message: str = "", **kwargs: Any) -> None:
        pass

    def print_preview(self, preview: str) -> None:
        pass

    def confirm(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        preview: Optional[str] = None,
    ) -> bool:
        self.approvals.append((tool_name, tool_input, preview))
        return self.approve


class ConsoleUI:
    """Rich terminal implementation."""

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()

    def status(self, message: str, spinner: str = "dots"):
        return self.console.status(message, spinner=spinner)

    def print(self, message: str = "", **kwargs: Any) -> None:
        self.console.print(message, **kwargs)

    def print_preview(self, preview: str) -> None:
        render_preview(preview, self.console)

    def confirm(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        preview: Optional[str] = None,
    ) -> bool:
        import json

        args_text = json.dumps(tool_input, indent=2)
        if len(args_text) > 1500:
            args_text = args_text[:1500] + "\n... [args truncated]"
        self.console.print(
            f"\n[bold yellow]Approval Required[/bold yellow]"
            f"\nTool: [cyan]{escape(tool_name)}[/cyan]"
            f"\nArgs: [dim]{escape(args_text)}[/dim]"
        )
        if preview:
            self.console.print("\n[bold]Proposed change:[/bold]")
            render_preview(preview, self.console)
        try:
            self.console.print("[bold yellow]Proceed? [Y/N]: [/bold yellow]", end="")
            response = input().strip().lower()
            return response in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False


def render_cost_dashboard(
    console: Console,
    stats: SessionStats,
    *,
    model_name: Optional[str] = None,
    mode: Optional[str] = None,
) -> None:
    """Print a real-time token/cost/speed dashboard line.

    Kept to a single compact line so it stays readable after every turn
    without flooding the terminal.
    """

    parts: list[str] = []
    if mode:
        parts.append(f"Mode: [bold]{mode.upper()}[/bold]")
    if model_name:
        parts.append(f"Model: [bold]{model_name}[/bold]")
    parts.append(
        f"Tokens: [cyan]{stats.total_tokens:,}[/cyan] "
        f"(in [green]{stats.input_tokens:,}[/green], "
        f"out [green]{stats.output_tokens:,}[/green])"
    )
    parts.append(f"Cost: [bold magenta]${stats.cost_usd:.6f}[/bold magenta]")
    console.print(" · ".join(parts))

