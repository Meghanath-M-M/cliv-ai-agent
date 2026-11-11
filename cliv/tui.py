"""Terminal User Interface (TUI) components for interactive model selection and navigation.

Provides keyboard navigation (arrows, j/k, Enter, Esc/q) using rich.live.Live
modeled after the graph explorer.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def get_key() -> str:
    """Read a single keypress without requiring Enter (cross-platform)."""
    if not sys.stdin.isatty():
        return ""

    if os.name == "nt":
        try:
            import msvcrt
            ch = msvcrt.getwch()
            # Handle arrow keys (special keys return two chars)
            if ch in ("\x00", "\xe0"):
                ch = msvcrt.getwch()
                return {"H": "\x1b[a", "P": "\x1b[b", "K": "\x1b[d", "M": "\x1b[c"}.get(ch, ch)
            if ch == "\r":
                return "\n"
            return ch.lower()
        except Exception:
            return ""
    else:
        try:
            import termios
            import tty
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                ch = sys.stdin.read(1)
                if ch == "\x1b":
                    import select
                    r, _, _ = select.select([sys.stdin], [], [], 0.05)
                    if r:
                        ch += sys.stdin.read(2)
                if ch == "\r":
                    return "\n"
                return ch.lower()
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            return ""


class ModelSelector:
    """Interactive TUI selector for choosing active LLM models."""

    def __init__(
        self,
        health_data: Dict[str, Any],
        current_model: str,
        current_mode: str,
        console: Optional[Console] = None,
    ):
        self.console = console or Console()
        self.current_model = current_model
        self.current_mode = current_mode
        self.models = self._build_model_list(health_data)
        self.selected_index = 0

        # Set default selection to active model
        for idx, m in enumerate(self.models):
            if m["is_active"]:
                self.selected_index = idx
                break

    def _build_model_list(self, health: Dict[str, Any]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []

        ollama_info = health.get("ollama", {})
        groq_info = health.get("groq", {})

        # 1. Ollama models
        if ollama_info.get("models"):
            for m in ollama_info["models"]:
                is_active = (m["name"] == self.current_model and self.current_mode == "offline")
                items.append({
                    "id": m["name"],
                    "display_name": m["name"],
                    "provider": "Ollama",
                    "provider_type": "local",
                    "supports_tools": bool(m.get("supports_tools", False)),
                    "tool_status": "✓ Direct tool calling" if m.get("supports_tools") else "Intent fallback",
                    "spec": f"{m.get('parameter_size') or '?'} ({m.get('size_gb', 0)} GB)",
                    "description": m.get("desc") or "Local open-weights model running via Ollama daemon",
                    "is_active": is_active,
                    "usable_for_chat": True,
                    "family": m.get("family", "Local LLM"),
                })

        # 2. Groq models
        if groq_info.get("models"):
            for m in groq_info["models"]:
                is_active = (m["id"] == self.current_model and self.current_mode == "online")
                usable = bool(m.get("usable_for_chat", True))
                items.append({
                    "id": m["id"],
                    "display_name": m["id"],
                    "provider": "Groq",
                    "provider_type": "cloud",
                    "supports_tools": usable,
                    "tool_status": "✓ Direct tool calling" if usable else f"No ({m.get('category', 'specialized')})",
                    "spec": m.get("owned_by", "Groq Cloud LPU"),
                    "description": m.get("desc") or "High-speed cloud model running on Groq LPU inference",
                    "is_active": is_active,
                    "usable_for_chat": usable,
                    "family": m.get("category", "Conversational Coding LLM"),
                })

        return items

    def _render_table(self) -> Table:
        table = Table(
            title="Available Language Models",
            show_header=True,
            header_style="bold cyan",
            box=None,
            padding=(0, 2),
        )
        table.add_column(" ", width=2, justify="center")
        table.add_column("Provider", width=10, style="bold")
        table.add_column("Model Name", min_width=32)
        table.add_column("Tool Calling", width=18)
        table.add_column("Size / Spec", width=18, style="dim")
        table.add_column("Status", width=12)

        for idx, m in enumerate(self.models):
            is_hovered = (idx == self.selected_index)
            cursor = "►" if is_hovered else " "

            # Style by active / hovered
            if m["is_active"]:
                status_str = "[bold green]● Active[/bold green]"
            elif not m.get("usable_for_chat", True):
                status_str = "[dim]Non-chat[/dim]"
            else:
                status_str = "[green]✓ Ready[/green]"

            tools_str = "[green]✓ Tools[/green]" if m["supports_tools"] else "[yellow]Intent only[/yellow]"
            if not m.get("usable_for_chat", True):
                tools_str = "[dim]No tools[/dim]"

            prov_color = "cyan" if m["provider"] == "Groq" else "magenta"
            prov_text = f"[{prov_color}]{m['provider']}[/{prov_color}]"

            model_style = "bold white on navy_blue" if is_hovered else ("bold" if m["is_active"] else "")

            table.add_row(
                f"[bold yellow]{cursor}[/bold yellow]" if is_hovered else " ",
                prov_text,
                Text(m["display_name"], style=model_style),
                tools_str,
                m["spec"],
                status_str,
                style="on grey19" if is_hovered else None,
            )

        return table

    def _render_details(self, model: Dict[str, Any]) -> Panel:
        lines: List[str] = []

        active_badge = " [bold green]● CURRENTLY ACTIVE[/bold green]" if model["is_active"] else ""
        lines.append(f"[bold cyan]Model:[/bold cyan] {model['id']}{active_badge}")
        lines.append(f"[bold cyan]Provider:[/bold cyan] {model['provider']} ({model['provider_type'].title()})")
        lines.append(f"[bold cyan]Architecture / Family:[/bold cyan] {model.get('family', 'General LLM')}")
        lines.append(f"[bold cyan]Tool Calling Capability:[/bold cyan] {model['tool_status']}")
        lines.append(f"[bold cyan]Hardware / Spec:[/bold cyan] {model['spec']}")
        lines.append(f"[bold cyan]Description:[/bold cyan] {model['description']}")

        if not model.get("usable_for_chat", True):
            lines.append("\n[bold yellow]⚠️ Note:[/bold yellow] This is a specialized model and cannot be used for chat or coding.")
        elif model["supports_tools"]:
            lines.append("\n[bold green]✓ Recommended:[/bold green] Supports native direct tool calling for editing, testing, and git operations.")
        else:
            lines.append("\n[bold yellow]ℹ️ Fallback Mode:[/bold yellow] Model uses heuristic intent routing rather than native JSON tool calls.")

        help_footer = (
            "\n[dim]Navigation: [bold]↑ / ↓[/bold] or [bold]j / k[/bold]  ·  "
            "[bold]Enter[/bold] select & switch  ·  [bold]q / Esc[/bold] cancel[/dim]"
        )
        lines.append(help_footer)

        return Panel("\n".join(lines), title="Model Details", border_style="cyan")

    def _render_view(self) -> Group:
        if not self.models:
            return Group(
                Panel(
                    "[bold yellow]No models detected.[/bold yellow]\n\n"
                    "- Start Ollama ([bold]ollama serve[/bold]) and pull a model ([bold]ollama pull llama3.1:8b[/bold])\n"
                    "- Or configure Groq API key: [bold]cliv config api-key <key>[/bold]",
                    title="Models",
                    border_style="yellow",
                )
            )

        current_item = self.models[self.selected_index]
        return Group(self._render_table(), self._render_details(current_item))

    def run(self) -> Optional[Dict[str, Any]]:
        """Run the interactive selector loop. Returns selected model dict or None."""
        if not self.models:
            self.console.print("[yellow]No models available to select.[/yellow]")
            return None

        # Fallback for non-interactive terminals (pipes, CI, tests)
        if not sys.stdin.isatty():
            return None

        from rich.live import Live

        with Live(
            self._render_view(),
            console=self.console,
            refresh_per_second=12,
            screen=True,
        ) as live:
            while True:
                live.update(self._render_view())

                try:
                    key = get_key()
                except (EOFError, KeyboardInterrupt):
                    return None

                if not key:
                    continue

                if key in ("q", "quit", "\x1b"):  # Quit / Escape
                    return None

                elif key in ("j", "\x1b[b"):  # Down arrow or j
                    self.selected_index = min(self.selected_index + 1, len(self.models) - 1)

                elif key in ("k", "\x1b[a"):  # Up arrow or k
                    self.selected_index = max(self.selected_index - 1, 0)

                elif key == "\n":  # Enter
                    chosen = self.models[self.selected_index]
                    if not chosen.get("usable_for_chat", True):
                        continue
                    return chosen
