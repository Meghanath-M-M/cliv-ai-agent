"""Slash-command router for the interactive REPL.

Inside the cliv chat session, any input that starts with ``/`` is handled
here — instantly, without consuming LLM tokens, without spawning a new
process, and with full access to the live session state (agent, snapshot
store, timeline, etc.).

Adding a new slash-command:
    1. Write a handler function: ``def handle_<name>(ctx, args) -> bool``
       - ``ctx`` is a :class:`SlashContext` carrying all live state.
       - ``args`` is the string *after* the slash-command word (may be empty).
       - Return ``True`` to continue the REPL; ``False`` to exit.
    2. Register it in ``SLASH_COMMANDS`` at the bottom of this file.

All handlers are intentionally synchronous and import-free at module level
so the router adds zero startup cost.
"""

from __future__ import annotations

import json
import re
import types
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from cliv.agent import AIAgent


# ---------------------------------------------------------------------------
# Context object passed to every handler
# ---------------------------------------------------------------------------

@dataclass
class SlashContext:
    """Live session state surfaced to slash-command handlers."""

    agent: "AIAgent"
    console: Console
    # Allow handlers to request REPL exit (return False from dispatch)
    _exit: bool = field(default=False, init=False, repr=False)

    # Convenience passthrough properties
    @property
    def snapshots(self):
        return self.agent.snapshots

    @property
    def timeline(self):
        return self.agent.timeline

    @property
    def auto_approve(self) -> bool:
        return self.agent.auto_approve


# ---------------------------------------------------------------------------
# Handler type alias
# ---------------------------------------------------------------------------

Handler = Callable[[SlashContext, str], bool]


# ---------------------------------------------------------------------------
# /help
# ---------------------------------------------------------------------------

def handle_help(ctx: SlashContext, args: str) -> bool:
    table = Table(
        title="cliv slash-commands",
        show_header=True,
        header_style="bold cyan",
        box=None,
        padding=(0, 2),
    )
    table.add_column("Command", style="bold green", no_wrap=True)
    table.add_column("Description")
    table.add_column("Example", style="dim")

    rows = [
        ("/help",             "Show this help",                      "/help"),
        ("/undo [id]",        "Restore last (or specific) snapshot", "/undo  or  /undo 20260101T120000Z-abc"),
        ("/snapshots [n]",    "List recent safety snapshots",        "/snapshots  or  /snapshots 10"),
        ("/diff",             "Show git working-tree diff",          "/diff"),
        ("/timeline [n]",     "Show last N timeline events",         "/timeline 20"),
        ("/rag [query|index]", "Semantic code search & index",        "/rag query auth  or  /rag index"),
        ("/dna",              "Project DNA architecture report",      "/dna"),
        ("/graph",            "ASCII dependency graph",              "/graph"),
        ("/clear",            "Clear conversation history",          "/clear"),
        ("/history [n]",      "Show recent conversation turns",      "/history 5"),
        ("/model [name]",     "Show or switch the active model",     "/model llama3.1:8b"),
        ("/permissions [p]",  "Show or set tool permissions",        "/permissions read,write"),
        ("/branch [name]",    "Show or create a git branch",         "/branch fix/login-bug"),
        ("/commit [msg]",     "Commit files touched this session",   "/commit 'fix login'"),
        ("/cost",             "Show session & project token spend",  "/cost"),
        ("/tools",            "List, inspect, or scaffold tool plugins", "/tools  or  /tools info <name>"),
        ("/memory",           "Manage persistent preferences & conventions", "/memory  or  /memory add 'use pytest'"),
        ("/export [path]",    "Export session to Markdown or JSON",  "/export session.md"),
        ("/import <path|id>", "Import a past session into context",  "/import session.json"),
        ("/sessions [n]",     "List archived conversation sessions", "/sessions 10"),
        ("/exit",             "Exit the session (also: quit)",       "/exit"),
    ]
    for cmd, desc, example in rows:
        table.add_row(cmd, desc, example)

    ctx.console.print()
    ctx.console.print(table)
    ctx.console.print()
    return True


# ---------------------------------------------------------------------------
# /undo [snapshot_id]
# ---------------------------------------------------------------------------

def handle_undo(ctx: SlashContext, args: str) -> bool:
    snapshot_id: Optional[str] = args.strip() or None

    if ctx.snapshots is None:
        ctx.console.print(
            "[bold yellow]Safety snapshots are not enabled in this session.[/bold yellow]"
        )
        return True

    # Delegate to the agent's unified undo handler (also updates history)
    response = ctx.agent._finish_undo(snapshot_id)
    ctx.console.print(Markdown(response))
    return True


# ---------------------------------------------------------------------------
# /snapshots [limit]
# ---------------------------------------------------------------------------

def handle_snapshots(ctx: SlashContext, args: str) -> bool:
    if ctx.snapshots is None:
        ctx.console.print(
            "[bold yellow]Safety snapshots are not enabled in this session.[/bold yellow]"
        )
        return True

    try:
        limit = int(args.strip()) if args.strip() else 15
    except ValueError:
        limit = 15

    from cliv.snapshots import render_snapshots
    items = ctx.snapshots.recent(limit=limit)
    ctx.console.print(render_snapshots(items), markup=False)
    return True


# ---------------------------------------------------------------------------
# /diff
# ---------------------------------------------------------------------------

def handle_diff(ctx: SlashContext, args: str) -> bool:
    from cliv.tools.git_diff import GitDiffTool
    from cliv.diff_view import render_preview

    result = GitDiffTool().execute()
    if result.startswith("No changes"):
        ctx.console.print(f"[dim]{result}[/dim]")
    else:
        # Render with the same syntax-highlighted diff view used for approvals.
        ctx.console.print()
        render_preview(result, ctx.console)
    return True


# ---------------------------------------------------------------------------
# /timeline [limit]
# ---------------------------------------------------------------------------

def handle_timeline(ctx: SlashContext, args: str) -> bool:
    if ctx.timeline is None:
        ctx.console.print(
            "[bold yellow]Timeline is not enabled in this session.[/bold yellow]"
        )
        return True

    try:
        limit = int(args.strip()) if args.strip() else 25
    except ValueError:
        limit = 25

    try:
        from cliv.timeline import render_timeline
        events = ctx.timeline.recent(limit=limit)
        ctx.console.print(render_timeline(events), markup=False)
    except Exception as exc:  # noqa: BLE001
        ctx.console.print(f"[bold red]Timeline error:[/bold red] {exc}")
    return True


# ---------------------------------------------------------------------------
# /dna
# ---------------------------------------------------------------------------

def handle_dna(ctx: SlashContext, args: str) -> bool:
    with ctx.console.status("[dim]Building Project DNA...[/dim]", spinner="dots"):
        result = ctx.agent._execute_tool("analyse_project", {"path": "."})
    ctx.console.print(result, markup=False)
    return True


# ---------------------------------------------------------------------------
# /graph
# ---------------------------------------------------------------------------

def handle_graph(ctx: SlashContext, args: str) -> bool:
    try:
        from cliv.intelligence import run_graph_explorer
        ctx.console.print(
            "[dim]Opening graph explorer (press q to return to chat)...[/dim]"
        )
        run_graph_explorer(
            interactive=True,
            export_mermaid=None,
            export_graphviz=None,
            console=ctx.console,
        )
    except Exception as exc:  # noqa: BLE001
        ctx.console.print(f"[bold red]Graph error:[/bold red] {exc}")
    return True


# ---------------------------------------------------------------------------
# /clear
# ---------------------------------------------------------------------------

def handle_clear(ctx: SlashContext, args: str) -> bool:
    ctx.agent.clear_history()
    ctx.console.print("[bold green]Memory cleared! Starting fresh.[/bold green]")
    return True


# ---------------------------------------------------------------------------
# /history [limit]
# ---------------------------------------------------------------------------

def handle_history(ctx: SlashContext, args: str) -> bool:
    try:
        limit = int(args.strip()) if args.strip() else 10
    except ValueError:
        limit = 10

    messages = ctx.agent.messages[-limit * 2:]  # rough: each turn ≈ 2 messages
    if not messages:
        ctx.console.print("[dim]No conversation history yet.[/dim]")
        return True

    ctx.console.print()
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls") or []

        if role == "user":
            ctx.console.print(Text(f"  You: {content[:200]}", style="bold cyan"))
        elif role == "assistant":
            if tool_calls:
                names = ", ".join(tc.get("function", {}).get("name", "?") for tc in tool_calls)
                ctx.console.print(Text(f"  cliv [tools: {names}]", style="bold yellow"))
            elif content:
                ctx.console.print(
                    Text(f"  cliv: {content[:200]}", style="bold green")
                )
        elif role == "tool":
            name = msg.get("name", "?")
            snippet = str(content)[:80].replace("\n", " ")
            ctx.console.print(Text(f"    ↳ {name}: {snippet}", style="dim"))
    ctx.console.print()
    return True


# ---------------------------------------------------------------------------
# /model [name]
# ---------------------------------------------------------------------------

def handle_model(ctx: SlashContext, args: str) -> bool:
    name = args.strip()
    backend = ctx.agent.backend

    if not name:
        # Show current model
        current = getattr(backend, "model", None) or ctx.agent.local_model
        mode = ctx.agent.mode
        ctx.console.print(
            f"Current model: [bold]{current}[/bold]  "
            f"Mode: [bold]{mode.upper()}[/bold]"
        )
        return True

    # Switch to named Ollama model (offline mode only for now)
    try:
        from cliv.llm import OllamaBackend
        from cliv.models import SessionStats

        if not OllamaBackend.is_available():
            ctx.console.print(
                "[bold yellow]Cannot switch model: Ollama is not running.[/bold yellow]"
            )
            return True

        new_backend = OllamaBackend(name, stats=ctx.agent.stats)
        ctx.agent.backend = new_backend
        ctx.agent.local_model = name
        ctx.agent.mode = "offline"
        ctx.console.print(
            f"[bold green]Switched to model:[/bold green] [bold]{name}[/bold]"
        )
    except Exception as exc:  # noqa: BLE001
        ctx.console.print(f"[bold red]Model switch failed:[/bold red] {exc}")
    return True


# ---------------------------------------------------------------------------
# /permissions [list]
# ---------------------------------------------------------------------------

def handle_permissions(ctx: SlashContext, args: str) -> bool:
    from cliv.guardian import ALL_PERMISSIONS, SecurityGuardian, SecurityPolicy

    guardian = ctx.agent.guardian
    if not args.strip():
        # Show current
        ctx.console.print(
            f"Current permissions: [bold]{guardian.policy.render()}[/bold]"
        )
        ctx.console.print(
            f"All available: {', '.join(sorted(ALL_PERMISSIONS))}"
        )
        return True

    requested = {t.lower() for t in re.split(r"[,\s]+", args.strip()) if t}
    unknown = requested - ALL_PERMISSIONS
    if unknown:
        ctx.console.print(
            f"[bold red]Unknown permission(s):[/bold red] {', '.join(sorted(unknown))}. "
            f"Choose from: {', '.join(sorted(ALL_PERMISSIONS))}"
        )
        return True

    ctx.agent.guardian = SecurityGuardian(SecurityPolicy.parse(args.strip()))
    ctx.console.print(
        f"[bold green]Permissions updated:[/bold green] "
        f"{ctx.agent.guardian.policy.render()}"
    )
    return True


# ---------------------------------------------------------------------------
# /branch [name]
# ---------------------------------------------------------------------------

def handle_branch(ctx: SlashContext, args: str) -> bool:
    name = args.strip()
    try:
        from cliv.git import GitRepo, is_git_repo
        from cliv.security import project_root

        if not is_git_repo(cwd=str(project_root())):
            ctx.console.print(
                "[bold yellow]Not inside a git repository.[/bold yellow]"
            )
            return True

        repo = GitRepo()
        if not name:
            current = repo.current_branch()
            ctx.console.print(f"Current branch: [bold cyan]{current}[/bold cyan]")
        else:
            repo.create_branch(name)
            ctx.agent.git_branch = name
            ctx.console.print(
                f"[bold green]Switched to branch:[/bold green] [bold]{name}[/bold]"
            )
    except Exception as exc:  # noqa: BLE001
        ctx.console.print(f"[bold red]Branch error:[/bold red] {exc}")
    return True


# ---------------------------------------------------------------------------
# /commit [message]
# ---------------------------------------------------------------------------

def handle_commit(ctx: SlashContext, args: str) -> bool:
    msg = args.strip().strip("'\"") or None
    try:
        from cliv.git import GitRepo, commit, is_git_repo
        from cliv.security import project_root

        if not is_git_repo(cwd=str(project_root())):
            ctx.console.print(
                "[bold yellow]Not inside a git repository.[/bold yellow]"
            )
            return True

        touched = list(ctx.agent.touched_files)
        if not touched:
            ctx.console.print(
                "[bold yellow]No files were modified this session.[/bold yellow]"
            )
            return True

        message = msg or f"cliv: session changes ({len(touched)} file(s))"
        commit_id = commit(message, touched, cwd=str(project_root()))
        if commit_id:
            ctx.console.print(
                f"[bold green]Committed {len(touched)} file(s):[/bold green] "
                f"[dim]{commit_id[:8]}[/dim] {message}"
            )
            ctx.agent.touched_files.clear()
        else:
            ctx.console.print(
                "[bold yellow]Nothing to commit (working tree clean).[/bold yellow]"
            )
    except Exception as exc:  # noqa: BLE001
        ctx.console.print(f"[bold red]Commit error:[/bold red] {exc}")
    return True


# ---------------------------------------------------------------------------
# /rag [query <terms> | index [--force] | stats]
# ---------------------------------------------------------------------------

def handle_rag(ctx: SlashContext, args: str) -> bool:
    subcmd, _, rest = args.strip().partition(" ")
    subcmd = subcmd.lower()
    rest = rest.strip()

    if not subcmd or subcmd in {"help", "-h", "--help"}:
        ctx.console.print(
            "\n[bold cyan]RAG Semantic Memory[/bold cyan]\n"
            "  /rag query <search terms>   – Semantic search across project codebase\n"
            "  /rag index [--force]        – Build or update the semantic index\n"
            "  /rag stats                  – Show index chunk, file, and dimension stats\n"
        )
        return True

    from cliv.intelligence.rag_indexer import create_rag_indexer

    try:
        if subcmd in {"query", "search"}:
            if not rest:
                ctx.console.print("[yellow]Usage: /rag query <search terms>[/yellow]")
                return True
            with ctx.console.status(f"[dim]Searching for '{rest}'...[/dim]", spinner="dots"):
                indexer = create_rag_indexer(prefer_offline=(ctx.agent.mode == "offline"))
                results = indexer.search(rest, k=5)
            if not results:
                ctx.console.print(f"[yellow]No semantic matches found for '{rest}'.[/yellow]")
                return True
            ctx.console.print(f"\n[bold]RAG Results for:[/bold] '{rest}'\n")
            for i, (chunk, score) in enumerate(results, 1):
                ctx.console.print(f"  [{i}] [bold cyan]{chunk.file_path}[/bold cyan] (score: {score:.3f})")
                if chunk.symbol_name:
                    ctx.console.print(f"      Symbol: {chunk.symbol_name} ({chunk.symbol_kind})")
                ctx.console.print(f"      Lines {chunk.start_line}-{chunk.end_line}:")
                for line in chunk.content.splitlines()[:6]:
                    ctx.console.print(f"      [dim]{line}[/dim]")
                if len(chunk.content.splitlines()) > 6:
                    ctx.console.print("      [dim]...[/dim]")
                ctx.console.print()
            return True

        elif subcmd in {"index", "build", "rebuild"}:
            force = "--force" in rest or "-f" in rest
            with ctx.console.status("[dim]Indexing codebase into semantic memory...[/dim]", spinner="dots"):
                indexer = create_rag_indexer(prefer_offline=(ctx.agent.mode == "offline"))
                stats = indexer.index(force_full=force)
            ctx.console.print(
                f"[bold green]RAG index updated:[/bold green] "
                f"{stats.chunks_added} chunks added, {stats.files_scanned} files scanned "
                f"({stats.duration_seconds:.2f}s)"
            )
            if stats.errors:
                for err in stats.errors:
                    ctx.console.print(f"[yellow]Warning:[/yellow] {err}")
            return True

        elif subcmd == "stats":
            indexer = create_rag_indexer(prefer_offline=(ctx.agent.mode == "offline"))
            stats = indexer.get_stats()
            ctx.console.print("\n[bold]RAG Index Statistics[/bold]")
            ctx.console.print(f"  Total chunks:     {stats.get('total_chunks', 0)}")
            ctx.console.print(f"  Unique files:     {stats.get('unique_files', 0)}")
            ctx.console.print(f"  Vector dimension: {stats.get('dimension', 0)}")
            ctx.console.print(f"  Index size:       {stats.get('index_size', '0 B')}\n")
            return True

        else:
            # Query shortcut: `/rag authentication`
            query = f"{subcmd} {rest}".strip()
            with ctx.console.status(f"[dim]Searching for '{query}'...[/dim]", spinner="dots"):
                indexer = create_rag_indexer(prefer_offline=(ctx.agent.mode == "offline"))
                results = indexer.search(query, k=5)
            if not results:
                ctx.console.print(f"[yellow]No semantic matches found for '{query}'.[/yellow]")
                return True
            ctx.console.print(f"\n[bold]RAG Results for:[/bold] '{query}'\n")
            for i, (chunk, score) in enumerate(results, 1):
                ctx.console.print(f"  [{i}] [bold cyan]{chunk.file_path}[/bold cyan] (score: {score:.3f})")
                if chunk.symbol_name:
                    ctx.console.print(f"      Symbol: {chunk.symbol_name} ({chunk.symbol_kind})")
                ctx.console.print(f"      Lines {chunk.start_line}-{chunk.end_line}:")
                for line in chunk.content.splitlines()[:6]:
                    ctx.console.print(f"      [dim]{line}[/dim]")
                if len(chunk.content.splitlines()) > 6:
                    ctx.console.print("      [dim]...[/dim]")
                ctx.console.print()
            return True

    except Exception as exc:  # noqa: BLE001
        ctx.console.print(f"[bold red]RAG operation failed:[/bold red] {exc}")
        return True


# ---------------------------------------------------------------------------
# /cost
# ---------------------------------------------------------------------------

def handle_cost(ctx: SlashContext, args: str) -> bool:
    from cliv.security import project_root

    stats = ctx.agent.stats
    backend = getattr(ctx.agent, "backend", None)
    model_name = getattr(backend, "model", None) or ctx.agent.local_model
    mode = ctx.agent.mode

    # Ensure current stats are persisted
    if hasattr(ctx.agent, "_persist_cost"):
        ctx.agent._persist_cost()

    ctx.console.print()

    # Active Session Table
    sess_table = Table(
        title="Active Session Cost",
        show_header=True,
        header_style="bold cyan",
        box=None,
        padding=(0, 2),
    )
    sess_table.add_column("Metric", style="bold")
    sess_table.add_column("Value")
    sess_table.add_row("Session ID", f"[dim]{ctx.agent.session_id}[/dim]")
    sess_table.add_row("Mode", f"[bold]{mode.upper()}[/bold]")
    sess_table.add_row("Model", f"[bold cyan]{model_name}[/bold cyan]")
    sess_table.add_row("Input Tokens", f"{stats.input_tokens:,}")
    sess_table.add_row("Output Tokens", f"{stats.output_tokens:,}")
    sess_table.add_row("Total Tokens", f"[bold]{stats.total_tokens:,}[/bold]")
    sess_table.add_row("Session Cost", f"[bold magenta]${stats.cost_usd:.6f}[/bold magenta]")

    ctx.console.print(sess_table)
    ctx.console.print()

    # Project Aggregate Table
    timeline = ctx.timeline
    if timeline is not None and hasattr(timeline, "get_project_cost"):
        proj_cost = timeline.get_project_cost(project=project_root())
        proj_table = Table(
            title=f"Project Historical Spend ({proj_cost['project_root']})",
            show_header=True,
            header_style="bold cyan",
            box=None,
            padding=(0, 2),
        )
        proj_table.add_column("Metric", style="bold")
        proj_table.add_column("Value")
        proj_table.add_row("Total Recorded Sessions", str(proj_cost["total_sessions"]))
        proj_table.add_row("Cumulative Input Tokens", f"{proj_cost['input_tokens']:,}")
        proj_table.add_row("Cumulative Output Tokens", f"{proj_cost['output_tokens']:,}")
        proj_table.add_row("Cumulative Total Tokens", f"[bold]{proj_cost['total_tokens']:,}[/bold]")
        proj_table.add_row("Cumulative Spend", f"[bold green]${proj_cost['total_cost_usd']:.6f}[/bold green]")
        ctx.console.print(proj_table)
        ctx.console.print()

        if proj_cost.get("models"):
            m_table = Table(
                title="Spend by Model",
                show_header=True,
                header_style="bold cyan",
                box=None,
                padding=(0, 2),
            )
            m_table.add_column("Model", style="bold")
            m_table.add_column("Tokens", justify="right")
            m_table.add_column("Cost", justify="right", style="bold magenta")
            for m in proj_cost["models"]:
                m_table.add_row(m["model"], f"{m['tokens']:,}", f"${m['cost']:.6f}")
            ctx.console.print(m_table)
            ctx.console.print()
    else:
        ctx.console.print("[dim]Timeline store not active; project totals unavailable.[/dim]\n")

    return True


# ---------------------------------------------------------------------------
# /export [path] [--format md|json]
# ---------------------------------------------------------------------------

def handle_export(ctx: SlashContext, args: str) -> bool:
    from pathlib import Path
    from cliv.session_io import export_session_json, export_session_markdown

    tokens = args.strip().split()
    fmt = "md"
    custom_path = None

    for token in tokens:
        if token in ("--json", "-j"):
            fmt = "json"
        elif token in ("--md", "-m", "--markdown"):
            fmt = "md"
        elif not token.startswith("-"):
            custom_path = token

    session_id = ctx.agent.session_id
    if custom_path:
        out_path = Path(custom_path)
        if out_path.suffix.lower() == ".json":
            fmt = "json"
        elif out_path.suffix.lower() == ".md":
            fmt = "md"
    else:
        ext = ".json" if fmt == "json" else ".md"
        out_path = Path.cwd() / f"cliv-session-{session_id[:8]}{ext}"

    try:
        session_data = ctx.agent.to_session_data()
        if fmt == "json":
            saved_path = export_session_json(session_data, out_path)
        else:
            saved_path = export_session_markdown(session_data, out_path)

        size_kb = saved_path.stat().st_size / 1024
        ctx.console.print(
            f"[bold green]Session exported:[/bold green] [bold cyan]{saved_path}[/bold cyan] "
            f"({len(session_data.messages)} messages, {size_kb:.1f} KB)"
        )
    except Exception as exc:  # noqa: BLE001
        ctx.console.print(f"[bold red]Export failed:[/bold red] {exc}")

    return True


# ---------------------------------------------------------------------------
# /import <path_or_session_id>
# ---------------------------------------------------------------------------

def handle_import(ctx: SlashContext, args: str) -> bool:
    from cliv.session_io import import_session

    target = args.strip().strip("'\"")
    if not target:
        ctx.console.print("[yellow]Usage: /import <path-to-json-file | session_id>[/yellow]")
        return True

    try:
        session_data = import_session(target)
        ctx.agent.load_session(session_data)
        ctx.console.print(
            f"[bold green]Session imported successfully:[/bold green] "
            f"[cyan]{session_data.session_id}[/cyan]\n"
            f"  Messages loaded: {len(session_data.messages)}\n"
            f"  Original model:  {session_data.model} ({session_data.mode})\n"
            f"  Date:            {session_data.created_at[:19].replace('T', ' ')}"
        )
    except Exception as exc:  # noqa: BLE001
        ctx.console.print(f"[bold red]Import failed:[/bold red] {exc}")

    return True


# ---------------------------------------------------------------------------
# /sessions [limit]
# ---------------------------------------------------------------------------

def handle_sessions(ctx: SlashContext, args: str) -> bool:
    from cliv.session_io import list_archived_sessions
    from cliv.security import project_root

    try:
        limit = int(args.strip()) if args.strip() else 15
    except ValueError:
        limit = 15

    sessions = list_archived_sessions(limit=limit, project=project_root())
    if not sessions:
        ctx.console.print("[dim]No archived sessions found for this project.[/dim]")
        return True

    table = Table(
        title=f"Archived Sessions ({project_root()})",
        show_header=True,
        header_style="bold cyan",
        box=None,
        padding=(0, 2),
    )
    table.add_column("Session ID", style="cyan", no_wrap=True)
    table.add_column("Created", style="dim")
    table.add_column("Model")
    table.add_column("Turns", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Cost", justify="right", style="bold magenta")

    for s in sessions:
        sid_display = s["id"] if len(s["id"]) <= 16 else f"{s['id'][:13]}..."
        table.add_row(
            sid_display,
            s["created"],
            s["model"],
            str(s["turns"]),
            f"{s['tokens']:,}",
            f"${s['cost']:.6f}",
        )

    ctx.console.print()
    ctx.console.print(table)
    ctx.console.print("\n[dim]Tip: Resume any session with /import <session_id>[/dim]\n")
    return True


# ---------------------------------------------------------------------------
# /tools [list|info|new|reload]
# ---------------------------------------------------------------------------

def handle_tools(ctx: SlashContext, args: str) -> bool:
    """List loaded tools, inspect schemas, or scaffold custom plugins."""
    parts = args.strip().split(None, 1)
    subcmd = parts[0].lower() if parts else "list"
    rest = parts[1].strip() if len(parts) > 1 else ""

    if subcmd in {"list", "ls"} or not subcmd:
        tools = getattr(ctx.agent, "tools", {})
        table = Table(
            title="Active Tools & Plugins",
            show_header=True,
            header_style="bold cyan",
            box=None,
            padding=(0, 2),
        )
        table.add_column("Tool Name", style="bold cyan", no_wrap=True)
        table.add_column("Source")
        table.add_column("Risk")
        table.add_column("Permissions", style="dim")
        table.add_column("Description")

        counts = {"builtin": 0, "user": 0, "project": 0}
        for name, tool in sorted(tools.items()):
            source = getattr(tool, "plugin_source", "builtin")
            counts[source] = counts.get(source, 0) + 1
            source_styled = {
                "builtin": "[dim]builtin[/dim]",
                "user": "[green]user[/green]",
                "project": "[yellow bold]project[/yellow bold]",
            }.get(source, source)

            risk = getattr(tool, "risk_level", "high")
            risk_styled = {
                "low": "[green]low[/green]",
                "medium": "[yellow]medium[/yellow]",
                "high": "[red]high[/red]",
                "critical": "[bold red]critical[/bold red]",
            }.get(risk, risk)

            perms = ",".join(sorted(getattr(tool, "permissions", []))) or "none"
            desc = getattr(tool, "description", "")
            if len(desc) > 50:
                desc = desc[:47] + "..."

            table.add_row(name, source_styled, risk_styled, perms, desc)

        ctx.console.print()
        ctx.console.print(table)
        ctx.console.print(
            f"\n[dim]{len(tools)} tools active ({counts.get('builtin', 0)} builtin, "
            f"{counts.get('user', 0)} user, {counts.get('project', 0)} project).[/dim]"
        )
        ctx.console.print("[dim]Tip: /tools info <name> | /tools new <name> | /tools reload[/dim]\n")
        return True

    if subcmd == "info":
        if not rest:
            ctx.console.print("[yellow]Usage: /tools info <tool_name>[/yellow]")
            return True
        tools = getattr(ctx.agent, "tools", {})
        tool = tools.get(rest)
        if not tool:
            ctx.console.print(f"[yellow]Tool '{rest}' not found.[/yellow] Type [bold]/tools[/bold] to list loaded tools.")
            return True

        ctx.console.print(f"\n[bold cyan]Tool:[/bold cyan] [bold]{tool.name}[/bold]")
        ctx.console.print(f"  Source:      {getattr(tool, 'plugin_source', 'builtin')}")
        if getattr(tool, "plugin_path", None):
            ctx.console.print(f"  Path:        {tool.plugin_path}")
        ctx.console.print(f"  Risk Level:  {getattr(tool, 'risk_level', 'high')}")
        ctx.console.print(f"  Permissions: {','.join(sorted(getattr(tool, 'permissions', []))) or 'none'}")
        ctx.console.print(f"  Description: {getattr(tool, 'description', '')}")
        ctx.console.print("\n[bold]Input Schema:[/bold]")
        ctx.console.print(json.dumps(getattr(tool, "input_schema", {}), indent=2))
        ctx.console.print()
        return True

    if subcmd == "new":
        if not rest:
            ctx.console.print("[yellow]Usage: /tools new <tool_name> [--global][/yellow]")
            return True
        is_global = "--global" in rest
        name = rest.replace("--global", "").strip()
        from cliv.config import PLUGINS_DIR
        from cliv.security import project_root
        from cliv.tools.loader import create_plugin_template

        target_dir = PLUGINS_DIR if is_global else (project_root() / ".cliv" / "plugins")
        path = create_plugin_template(name, target_dir)
        ctx.console.print(f"\n[bold green]Created tool plugin scaffold:[/bold green] {path}")
        ctx.console.print("[dim]Edit this file to implement your tool logic, then run /tools reload.[/dim]\n")
        return True

    if subcmd in {"reload", "refresh"}:
        if hasattr(ctx.agent, "reload_tools"):
            reloaded = ctx.agent.reload_tools()
        else:
            from cliv.tools.loader import load_tools
            reloaded = load_tools()
            ctx.agent.tools = reloaded

        ctx.console.print(f"\n[bold green]Reloaded tools:[/bold green] {len(reloaded)} tools currently loaded.\n")
        return True

    ctx.console.print(f"[yellow]Unknown /tools subcommand: '{subcmd}'.[/yellow] Use list, info, new, or reload.")
    return True


# ---------------------------------------------------------------------------
# /memory [list|add|remove|clear]
# ---------------------------------------------------------------------------

def handle_memory(ctx: SlashContext, args: str) -> bool:
    """Manage agent long-term memory, conventions, and user preferences."""
    parts = args.strip().split(None, 1)
    subcmd = parts[0].lower() if parts else "list"
    rest = parts[1].strip() if len(parts) > 1 else ""

    from cliv.memory import MemoryStore

    memory_store: MemoryStore = getattr(ctx.agent, "memory", None) or MemoryStore()

    if subcmd in {"list", "ls"} or not subcmd:
        scope_filter = rest.lower() if rest in {"global", "project"} else None
        entries = memory_store.list_entries(scope=scope_filter)
        cliv_md = memory_store.get_cliv_md_content()

        if not entries and not cliv_md:
            ctx.console.print("[dim]No long-term memories or CLIV.md guidelines found.[/dim]")
            ctx.console.print("[dim]Tip: Use /memory add <preference> to teach the agent conventions.[/dim]\n")
            return True

        table = Table(
            title="Agent Long-Term Memory & Conventions",
            show_header=True,
            header_style="bold cyan",
            box=None,
            padding=(0, 2),
        )
        table.add_column("Scope", style="bold")
        table.add_column("ID", style="cyan")
        table.add_column("Category", style="dim")
        table.add_column("Content")
        table.add_column("Date", style="dim")

        for e in entries:
            scope_styled = "[cyan]global[/cyan]" if e.scope == "global" else "[yellow]project[/yellow]"
            cat_styled = f"[{e.category}]" if e.category else ""
            table.add_row(
                scope_styled,
                e.id,
                cat_styled,
                e.content,
                e.created_at[:10],
            )

        ctx.console.print()
        ctx.console.print(table)

        if cliv_md:
            ctx.console.print(f"\n[bold]CLIV.md Guidelines:[/bold] ({len(cliv_md)} characters loaded from project root)")
            for line in cliv_md.splitlines()[:5]:
                ctx.console.print(f"  [dim]{line}[/dim]")
            if len(cliv_md.splitlines()) > 5:
                ctx.console.print("  [dim]...[/dim]")

        ctx.console.print(
            f"\n[dim]{len(entries)} items remembered. "
            "Tip: /memory add <text> [--global] | /memory remove <id> | /memory clear[/dim]\n"
        )
        return True

    if subcmd == "add":
        if not rest:
            ctx.console.print("[yellow]Usage: /memory add <preference or rule> [--global] [--category <cat>][/yellow]")
            return True

        is_global = "--global" in rest
        text = rest.replace("--global", "").strip()

        category = "general"
        if "--category" in text:
            m = re.search(r"--category\s+([a-zA-Z0-9_-]+)", text)
            if m:
                category = m.group(1)
                text = text.replace(m.group(0), "").strip()

        if not text:
            ctx.console.print("[yellow]Error: Memory content cannot be empty.[/yellow]")
            return True

        scope = "global" if is_global else "project"
        entry = memory_store.add(content=text, scope=scope, category=category)
        scope_styled = "[cyan]global[/cyan]" if scope == "global" else "[yellow]project[/yellow]"
        ctx.console.print(f"\n[bold green]Saved {scope_styled} memory (id: [bold]{entry.id}[/bold]):[/bold green]")
        ctx.console.print(f"  \"{entry.content}\"\n")
        return True

    if subcmd in {"remove", "rm", "delete"}:
        if not rest:
            ctx.console.print("[yellow]Usage: /memory remove <id_or_keyword>[/yellow]")
            return True

        removed = memory_store.remove(rest)
        if removed:
            ctx.console.print(f"\n[bold green]Removed memory matching:[/bold green] '{rest}'\n")
        else:
            ctx.console.print(f"[yellow]No memory found matching:[/yellow] '{rest}'")
        return True

    if subcmd == "clear":
        scope = rest.lower() if rest in {"global", "project"} else None
        count = memory_store.clear(scope=scope)
        target = f"{scope} " if scope else ""
        ctx.console.print(f"\n[bold green]Cleared {count} {target}memories.[/bold green]\n")
        return True

    ctx.console.print(f"[yellow]Unknown /memory subcommand: '{subcmd}'.[/yellow] Use list, add, remove, or clear.")
    return True


# ---------------------------------------------------------------------------
# /exit  /quit
# ---------------------------------------------------------------------------

def handle_exit(ctx: SlashContext, args: str) -> bool:
    ctx.console.print("[bold yellow]Session terminated.[/bold yellow]")
    return False  # signal the REPL to exit


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SLASH_COMMANDS: Dict[str, Tuple[Handler, str]] = {
    "/help":        (handle_help,        "Show available slash-commands"),
    "/undo":        (handle_undo,        "Restore last (or specific) snapshot"),
    "/snapshots":   (handle_snapshots,   "List recent safety snapshots"),
    "/diff":        (handle_diff,        "Show git working-tree diff"),
    "/timeline":    (handle_timeline,    "Show recent timeline events"),
    "/rag":         (handle_rag,         "Semantic memory search & index management"),
    "/dna":         (handle_dna,         "Project DNA architecture report"),
    "/graph":       (handle_graph,       "Visual codebase dependency graph"),
    "/clear":       (handle_clear,       "Clear conversation history"),
    "/history":     (handle_history,     "Show recent conversation turns"),
    "/model":       (handle_model,       "Show or switch the active model"),
    "/permissions": (handle_permissions, "Show or set tool permissions"),
    "/branch":      (handle_branch,      "Show or create a git branch"),
    "/commit":      (handle_commit,      "Commit files touched this session"),
    "/cost":        (handle_cost,        "Show session and project token usage & spend"),
    "/tools":       (handle_tools,       "List, inspect, scaffold, or reload tool plugins"),
    "/plugins":     (handle_tools,       "Alias for /tools"),
    "/memory":      (handle_memory,      "Manage persistent user preferences and project conventions"),
    "/memories":    (handle_memory,      "Alias for /memory"),
    "/export":      (handle_export,      "Export session to Markdown or JSON"),
    "/import":      (handle_import,      "Import a past session into active context"),
    "/sessions":    (handle_sessions,    "List archived conversation sessions"),
    "/exit":        (handle_exit,        "Exit the session"),
    "/quit":        (handle_exit,        "Exit the session"),
}


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def dispatch(raw_input: str, ctx: SlashContext) -> Optional[bool]:
    """Route a slash-command and return the REPL continuation flag.

    Returns:
        ``True``  – command handled, continue the REPL.
        ``False`` – command handled, exit the REPL.
        ``None``  – not a slash-command; caller should handle normally.
    """
    stripped = raw_input.strip()
    if not stripped.startswith("/"):
        return None

    # Split into command word and remaining args
    parts = stripped.split(None, 1)
    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    if cmd not in SLASH_COMMANDS:
        ctx.console.print(
            f"[bold red]Unknown slash-command:[/bold red] {cmd}  "
            f"(type [bold]/help[/bold] to see all commands)"
        )
        return True  # handled (with an error), continue REPL

    handler, _ = SLASH_COMMANDS[cmd]
    try:
        return handler(ctx, args)
    except Exception as exc:  # noqa: BLE001
        ctx.console.print(f"[bold red]{cmd} failed:[/bold red] {exc}")
        return True
