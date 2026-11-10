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
