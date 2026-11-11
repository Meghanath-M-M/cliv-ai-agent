"""Visual Codebase Explorer: terminal-based graph explorer using Rich.

Renders components as nodes with dependency edges, plus an interactive
explore mode and Mermaid/Graphviz export.
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from rich.console import Group

from cliv.intelligence.report import Component, ProjectIntelligence
from cliv.intelligence.dependency_graph import DependencyGraph
from cliv.intelligence.code_parser import ModuleRecord


def _get_key() -> str:
    """Read a single keypress without requiring Enter (cross-platform)."""
    if os.name == "nt":
        try:
            import msvcrt
            ch = msvcrt.getwch()
            # Handle arrow keys (special keys return two chars)
            if ch in ("\x00", "\xe0"):  # Special key prefix
                ch = msvcrt.getwch()
                return {"H": "\x1b[a", "P": "\x1b[b", "K": "\x1b[d", "M": "\x1b[c"}.get(ch, ch)
            return ch.lower()
        except Exception:
            # Fallback to regular input
            return input().strip().lower()
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
                    ch += sys.stdin.read(2)
                return ch.lower()
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            # Fallback to regular input
            return input().strip().lower()


@dataclass
class GraphNode:
    """A node in the visual graph."""

    id: str
    label: str
    role: str
    file_count: int
    symbol_count: int
    internal_edges: int
    is_selected: bool = False
    is_focused: bool = False


@dataclass
class GraphEdge:
    """An edge in the visual graph."""

    source: str
    target: str


class GraphExplorer:
    """Interactive terminal-based graph explorer for Project DNA."""

    ROLE_COLORS = {
        "Frontend": "cyan",
        "Backend": "green",
        "AI Integration": "magenta",
        "Database": "blue",
        "Authentication": "red",
        "Security": "red",
        "Tooling": "yellow",
        "Testing": "yellow",
        "Application": "white",
    }

    def __init__(
        self,
        intelligence: ProjectIntelligence,
        console: Optional[Console] = None,
    ):
        self.intelligence = intelligence
        self.console = console or Console()
        self.nodes: List[GraphNode] = []
        self.edges: List[GraphEdge] = []
        self._build_graph()
        self._selected_index = 0
        self._show_details = True
        self._filter_role: Optional[str] = None

    def _build_graph(self) -> None:
        """Build internal graph representation from ProjectIntelligence."""
        components = self.intelligence.components
        graph = self.intelligence.graph

        for component in components:
            node = GraphNode(
                id=component.key,
                label=component.name,
                role=component.role,
                file_count=len(component.files),
                symbol_count=component.symbols,
                internal_edges=component.internal_edges,
            )
            self.nodes.append(node)

        for source, targets in graph.edges.items():
            for target in targets:
                if source != target:
                    self.edges.append(GraphEdge(source=source, target=target))

    def _get_filtered_nodes(self) -> List[GraphNode]:
        """Get nodes filtered by role if a filter is active."""
        if self._filter_role is None:
            return self.nodes
        return [n for n in self.nodes if n.role == self._filter_role]

    def _get_node_by_id(self, node_id: str) -> Optional[GraphNode]:
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def _get_dependencies(self, node_id: str) -> Tuple[List[str], List[str]]:
        """Get incoming and outgoing dependencies for a node."""
        incoming = []
        outgoing = []
        for edge in self.edges:
            if edge.source == node_id:
                outgoing.append(edge.target)
            elif edge.target == node_id:
                incoming.append(edge.source)
        return incoming, outgoing

    def _render_node(self, node: GraphNode) -> Text:
        """Render a single node with styling."""
        color = self.ROLE_COLORS.get(node.role, "white")
        style = f"bold {color}"
        if node.is_selected:
            style = f"bold reverse {color}"
        elif node.is_focused:
            style = f"bold underline {color}"

        parts = [
            f"[{style}]{node.label}[/{style}]",
            f"[dim]({node.role})[/dim]",
            f"[dim]{node.file_count} files[/dim]",
            f"[dim]{node.symbol_count} symbols[/dim]",
        ]
        return Text(" ").join(Text.from_markup(p) for p in parts)

    def _render_details(self, node: GraphNode) -> Panel:
        """Render detailed information panel for a node."""
        incoming, outgoing = self._get_dependencies(node.id)

        color = self.ROLE_COLORS.get(node.role, "white")
        lines = [
            f"[bold]{node.label}[/bold]",
            f"Role: [cyan]{node.role}[/cyan]",
            f"Files: {node.file_count}",
            f"Symbols: {node.symbol_count}",
            f"Internal edges: {node.internal_edges}",
            "",
            f"[bold]Dependencies ([green]{len(outgoing)}[/green] outgoing, [red]{len(incoming)}[/red] incoming):[/bold]",
        ]

        if outgoing:
            lines.append("  [green]→[/green] " + ", ".join(sorted(outgoing)[:10]))
            if len(outgoing) > 10:
                lines.append(f"  [dim]... and {len(outgoing) - 10} more[/dim]")
        if incoming:
            lines.append("  [red]←[/red] " + ", ".join(sorted(incoming)[:10]))
            if len(incoming) > 10:
                lines.append(f"  [dim]... and {len(incoming) - 10} more[/dim]")

        if not outgoing and not incoming:
            lines.append("  [dim](no internal dependencies)[/dim]")

        lines.extend(["", "[bold]Files:[/bold]"])
        for file in self._get_node_files(node.id)[:15]:
            lines.append(f"  {file}")
        remaining = len(self._get_node_files(node.id)) - 15
        if remaining > 0:
            lines.append(f"  [dim]... and {remaining} more[/dim]")

        return Panel("\n".join(lines), title="Component Details", border_style=color)

    def _get_node_files(self, node_id: str) -> List[str]:
        for component in self.intelligence.components:
            if component.key == node_id:
                return component.files
        return []

    def _render_graph_view(self) -> Group:
        """Render the main graph view as a table."""
        filtered_nodes = self._get_filtered_nodes()

        if not filtered_nodes:
            return Group(Text("No components match the current filter.", style="dim"))

        table = Table(show_header=True, header_style="bold", box=None)
        table.add_column("#", width=3, justify="right")
        table.add_column("Component", min_width=30)
        table.add_column("Role", width=18)
        table.add_column("Files", width=6, justify="right")
        table.add_column("Symbols", width=8, justify="right")
        table.add_column("Edges", width=6, justify="right")

        for idx, node in enumerate(filtered_nodes):
            is_selected = (idx == self._selected_index)
            node.is_selected = is_selected
            node.is_focused = False

            color = self.ROLE_COLORS.get(node.role, "white")
            row_style = color if not is_selected else f"reverse {color}"

            table.add_row(
                str(idx + 1),
                node.label,
                f"[{color}]{node.role}[/{color}]",
                str(node.file_count),
                str(node.symbol_count),
                str(node.internal_edges),
                style=row_style,
            )

        if self._show_details and filtered_nodes:
            selected_node = filtered_nodes[self._selected_index]
            return Group(table, self._render_details(selected_node))

        return Group(table)

    def _render_help(self) -> Panel:
        """Render the help panel."""
        help_text = """
[bold]Navigation:[/bold]
  ↑/↓ or j/k    Navigate components
  Enter         Focus selected component
  Esc           Clear focus

[bold]Filtering:[/bold]
  f             Cycle role filter
  F             Clear filter

[bold]Export:[/bold]
  m             Export Mermaid diagram
  g             Export Graphviz DOT
  s             Save current view to file

[bold]Other:[/bold]
  h/?           Toggle this help
  q             Quit
        """.strip()
        return Panel(help_text, title="Help", border_style="dim")

    def run_interactive(self) -> None:
        """Run the interactive explorer."""
        from rich.live import Live

        with Live(
            self._render_graph_view(),
            console=self.console,
            refresh_per_second=10,
            screen=True,
        ) as live:
            while True:
                live.update(self._render_graph_view())

                try:
                    key = _get_key()
                except (EOFError, KeyboardInterrupt):
                    break

                if not key:
                    continue

                filtered_nodes = self._get_filtered_nodes()
                if not filtered_nodes:
                    continue

                if key in ("q", "quit"):
                    break
                elif key in ("h", "?"):
                    self.console.print(self._render_help())
                    self.console.input("\nPress Enter to continue...")
                elif key in ("j", "\x1b[b"):  # down arrow
                    self._selected_index = min(
                        self._selected_index + 1, len(filtered_nodes) - 1
                    )
                elif key in ("k", "\x1b[a"):  # up arrow
                    self._selected_index = max(self._selected_index - 1, 0)
                elif key in ("\r", "\n"):  # Enter
                    node = filtered_nodes[self._selected_index]
                    node.is_focused = True
                    self.console.print(self._render_details(node))
                    self.console.input("\nPress Enter to continue...")
                elif key == "f":
                    roles = sorted(set(n.role for n in self.nodes))
                    if self._filter_role is None:
                        self._filter_role = roles[0] if roles else None
                    else:
                        try:
                            idx = roles.index(self._filter_role)
                            self._filter_role = roles[(idx + 1) % len(roles)]
                        except ValueError:
                            self._filter_role = roles[0] if roles else None
                    self._selected_index = 0
                elif key == "F":
                    self._filter_role = None
                    self._selected_index = 0
                elif key == "m":
                    self.export_mermaid()
                    self.console.print("[green]Mermaid diagram exported![/green]")
                    self.console.input("\nPress Enter to continue...")
                elif key == "g":
                    self.export_graphviz()
                    self.console.print("[green]Graphviz DOT exported![/green]")
                    self.console.input("\nPress Enter to continue...")
                elif key == "s":
                    self.save_view()
                    self.console.print("[green]View saved![/green]")
                    self.console.input("\nPress Enter to continue...")

    def export_mermaid(self, path: Optional[str] = None) -> str:
        """Export the dependency graph as a Mermaid diagram."""
        lines = [
            "graph TD",
            "    %% Project DNA Dependency Graph",
            "    %% Generated by cliv Visual Codebase Explorer",
            "",
        ]

        for node in self.nodes:
            color = self.ROLE_COLORS.get(node.role, "white")
            node_id = node.id.replace("/", "_").replace(".", "_").replace("-", "_")
            label = node.label.replace('"', '\\"')
            lines.append(f'    {node_id}["{label}"]')
            lines.append(f'    class {node_id} {color.lower()}')
            lines.append("")

        for edge in self.edges:
            source_id = edge.source.replace("/", "_").replace(".", "_").replace("-", "_")
            target_id = edge.target.replace("/", "_").replace(".", "_").replace("-", "_")
            lines.append(f"    {source_id} --> {target_id}")

        lines.extend([
            "",
            "    classDef cyan fill:#06b6d4,color:#fff;",
            "    classDef green fill:#22c55e,color:#fff;",
            "    classDef magenta fill:#d946ef,color:#fff;",
            "    classDef blue fill:#3b82f6,color:#fff;",
            "    classDef red fill:#ef4444,color:#fff;",
            "    classDef yellow fill:#eab308,color:#000;",
            "    classDef white fill:#6b7280,color:#fff;",
        ])

        mermaid = "\n".join(lines)

        if path:
            with open(path, "w") as f:
                f.write(mermaid)

        return mermaid

    def export_graphviz(self, path: Optional[str] = None) -> str:
        """Export the dependency graph as Graphviz DOT."""
        lines = [
            'digraph "ProjectDNA" {',
            '    rankdir=LR;',
            '    node [fontname="Arial", fontsize=10];',
            '    edge [fontname="Arial", fontsize=8];',
            "",
        ]

        for node in self.nodes:
            color = self.ROLE_COLORS.get(node.role, "white")
            node_id = node.id.replace("/", "_").replace(".", "_").replace("-", "_")
            label = node.label.replace('"', '\\"')
            tooltip = f"Role: {node.role}\\nFiles: {node.file_count}\\nSymbols: {node.symbol_count}"
            lines.append(
                f'    {node_id} [label="{label}", tooltip="{tooltip}", '
                f'style=filled, fillcolor={color.lower()}, fontcolor=white];'
            )

        lines.append("")

        for edge in self.edges:
            source_id = edge.source.replace("/", "_").replace(".", "_").replace("-", "_")
            target_id = edge.target.replace("/", "_").replace(".", "_").replace("-", "_")
            lines.append(f"    {source_id} -> {target_id};")

        lines.append("}")

        dot = "\n".join(lines)

        if path:
            with open(path, "w") as f:
                f.write(dot)

        return dot

    def save_view(self, path: Optional[str] = None) -> str:
        """Save the current text view to a file."""
        from io import StringIO

        console = Console(file=StringIO(), force_terminal=True, width=120)
        console.print(self._render_graph_view())
        content = console.file.getvalue()

        if path is None:
            path = "graph_view.txt"

        with open(path, "w") as f:
            f.write(content)

        return content


def run_graph_explorer(
    root: Optional[str] = None,
    *,
    interactive: bool = True,
    export_mermaid: Optional[str] = None,
    export_graphviz: Optional[str] = None,
    console: Optional[Console] = None,
) -> int:
    """Run the Visual Codebase Explorer.

    Args:
        root: Project root directory (defaults to current project)
        interactive: Run in interactive mode
        export_mermaid: Path to export Mermaid diagram
        export_graphviz: Path to export Graphviz DOT
        console: Rich console instance

    Returns:
        Exit code (0 for success)
    """
    from cliv.intelligence import build_project_intelligence
    from cliv.security import project_root
    from pathlib import Path

    c = console or Console()

    try:
        project_path = Path(root) if root else project_root()
        c.print(f"[dim]Analyzing project at {project_path}...[/dim]")
        intelligence = build_project_intelligence(project_path)
    except Exception as exc:
        c.print(f"[bold red]Failed to analyze project:[/bold red] {exc}")
        return 1

    explorer = GraphExplorer(intelligence, console=c)

    if export_mermaid:
        explorer.export_mermaid(export_mermaid)
        c.print(f"[green]Mermaid diagram saved to {export_mermaid}[/green]")
        return 0

    if export_graphviz:
        explorer.export_graphviz(export_graphviz)
        c.print(f"[green]Graphviz DOT saved to {export_graphviz}[/green]")
        return 0

    if interactive:
        c.print("[bold cyan]Visual Codebase Explorer[/bold cyan]")
        c.print("Press [bold]?[/bold] for help, [bold]q[/bold] to quit\n")
        explorer.run_interactive()
    else:
        c.print(explorer._render_graph_view())

    return 0