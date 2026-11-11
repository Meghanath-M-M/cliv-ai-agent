"""
Terminal UI for cliv.
Handles startup flow, routing, real output, and session management.
"""

import sys
import argparse
import re
import socket
from getpass import getpass
from cliv.agent import AIAgent
from cliv.config import get_api_key, save_api_key
from cliv.guardian import ALL_PERMISSIONS, SecurityGuardian, SecurityPolicy
from cliv.ui import ConsoleUI, render_cost_dashboard
from cliv.slash import SlashContext, dispatch as slash_dispatch

from rich.console import Console
from rich.markdown import Markdown

console = Console()


def check_internet(host="8.8.8.8", port=53, timeout=2):
    """Safely checks if the device is connected to the internet."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect((host, port))
        return True
    except OSError:
        return False


def process_prompt(agent, user_input):
    """Handles the rendering of the agent's response."""
    console.print("\n[bold cyan]cliv:[/bold cyan]")

    response = agent.chat(user_input)

    if response:
        console.print(Markdown(response))

    # --- Live cost dashboard ---
    backend = getattr(agent, "backend", None)
    model_name = getattr(backend, "model", None) if backend is not None else None
    render_cost_dashboard(
        console,
        agent.stats,
        model_name=model_name,
        mode=agent.mode,
    )
    print()


def build_agent(args, security_guardian=None, timeline_store=None, snapshot_store=None):
    """Build agent from parsed args."""
    api_key = args.api_key or get_api_key()
    if api_key == "" or args.offline:
        api_key = None

    if not api_key and not getattr(args, "prompt", None) and not args.offline:
        console.print("[dim]No Groq API key found.[/dim]")
        console.print(
            "[dim]For Lightning Fast execution, get a free key: https://console.groq.com/keys[/dim]"
        )
        console.print(
            "[dim]Or press [Enter] to run entirely OFFLINE using your local hardware (requires Ollama).[/dim]"
        )

        input_key = getpass("Enter API Key (or press Enter for Offline Mode): ").strip()

        if input_key:
            api_key = input_key
            save = (
                input("Would you like to save this key for future use? (y/n): ")
                .strip()
                .lower()
            )
            if save == "y":
                save_api_key(api_key)
                console.print("[green]API key saved.[/green]")
        else:
            console.print(
                "\n[dim]Starting in OFFLINE mode. (Ensure Ollama is running)[/dim]"
            )
            api_key = None

    # --- Automatic Offline Fallback ---
    if api_key:
        with console.status(
            "[dim]Checking network connection...[/dim]", spinner="dots"
        ):
            is_online = check_internet()

        if not is_online:
            console.print(
                "\n[bold yellow][SYSTEM WARNING][/bold yellow] No internet connection detected!"
            )
            console.print(
                "[bold yellow]Automatically falling back to local OFFLINE mode.[/bold yellow]\n"
            )
            api_key = None

    with console.status("[dim]Configuring and validating model...[/dim]", spinner="dots"):
        agent = AIAgent(
            api_key=api_key,
            verbose=args.verbose,
            auto_approve=args.yes,
            dry_run=args.dry_run,
            force_offline=args.offline,
            use_offline_tools=not args.legacy_offline,
            ui=ConsoleUI(console),
            timeline_store=timeline_store,
            snapshot_store=snapshot_store,
            guardian=security_guardian,
            auto_commit=args.auto_commit,
            git_branch=args.branch,
        )

    # Print mode banner
    mode_color = "green" if agent.mode == "online" else "yellow"

    logo = """[bold cyan]
     ██████╗ ██╗      ██╗ ██╗   ██╗
    ██╔════╝ ██║      ██║ ██║   ██║
    ██║      ██║      ██║ ██║   ██║
    ██║      ██║      ██║ ╚██╗ ██╔╝
    ╚██████╗ ███████╗ ██║  ╚████╔╝
     ╚═════╝ ╚══════╝ ╚═╝   ╚═══╝ 
    [/bold cyan]"""
    console.print(logo)

    console.print(f"Mode: [bold {mode_color}]{agent.mode.upper()}[/bold {mode_color}]")
    backend = getattr(agent, "backend", None)
    model_name = getattr(backend, "model", None)
    if model_name:
        console.print(f"Model: [bold]{model_name}[/bold]")
    for notice in getattr(agent, "model_notices", []):
        console.print(f"[bold yellow]Model notice:[/bold yellow] {notice}")
    if args.yes:
        console.print("[dim]Auto-approve: ENABLED (use with caution)[/dim]")
    if args.dry_run:
        console.print("[dim]Dry-run: ENABLED (no changes will be applied)[/dim]")
    if args.legacy_offline:
        console.print("[dim]Offline routing: legacy regex fallback[/dim]")
    guardian = getattr(agent, "guardian", None)
    if guardian is not None:
        console.print(
            f"[dim]Permissions: {guardian.policy.render()}[/dim]"
        )
    render_cost_dashboard(
        console,
        agent.stats,
        model_name=model_name,
        mode=agent.mode,
    )
    console.print("Terminal-native intelligence for your local codebase.")
    console.print(
        "Type [bold]/help[/bold] for slash-commands, "
        "[bold]exit[/bold] or [bold]quit[/bold] to end the session.\n"
    )

    return agent


def run_interactive(agent, prompt=None):
    """Run interactive REPL or single-shot."""
    if prompt:
        console.print(f"[dim]Executing command:[/dim] {prompt}")
        process_prompt(agent, prompt)
        return

    ctx = SlashContext(agent=agent, console=console)

    while True:
        try:
            user_input = input("> ").strip()

            if not user_input:
                continue

            # Check slash commands first (/undo, /diff, /timeline, /help, etc.)
            slash_result = slash_dispatch(user_input, ctx)
            if slash_result is not None:
                if not slash_result:
                    break
                continue

            # Legacy word commands
            if user_input.lower() in ["exit", "quit"]:
                console.print("[bold yellow]Session terminated.[/bold yellow]")
                break

            if user_input.lower() == "clear":
                agent.clear_history()
                console.print(
                    "[bold green]Memory cleared! Starting fresh.[/bold green]"
                )
                continue

            # Route natural-language "undo" to slash-command handler
            if user_input.lower() in {"undo", "undo last", "undo that"}:
                slash_dispatch("/undo", ctx)
                continue

            process_prompt(agent, user_input)

        except KeyboardInterrupt:
            console.print("\n\n[bold yellow]Session terminated by user.[/bold yellow]")
            break
        except Exception as e:
            console.print(f"\n[bold red]Error: {str(e)}[/bold red]\n")


# ============================================================
# Subcommand handlers
# ============================================================

def cmd_run(args):
    """Interactive chat or single-shot."""
    security_guardian = None
    if args.permissions is not None:
        requested = {
            token.lower()
            for token in re.split(r"[,\s]+", args.permissions)
            if token
        }
        unknown = requested - ALL_PERMISSIONS
        if unknown:
            console.print(
                "[bold red]Error: unknown permission(s): "
                f"{', '.join(sorted(unknown))}. "
                "Choose from read,write,execute,network.[/bold red]"
            )
            sys.exit(1)
        security_guardian = SecurityGuardian(
            SecurityPolicy.parse(args.permissions)
        )

    timeline_store = None
    try:
        from cliv.timeline import TimelineStore
        timeline_store = TimelineStore()
    except Exception as exc:
        console.print(f"[dim]Timeline disabled: {exc}[/dim]")

    snapshot_store = None
    try:
        from cliv.snapshots import SnapshotStore
        snapshot_store = SnapshotStore()
    except Exception as exc:
        console.print(f"[bold yellow]Safety snapshots disabled:[/bold yellow] {exc}")

    agent = build_agent(args, security_guardian, timeline_store, snapshot_store)
    if getattr(args, "resume", None):
        try:
            from cliv.session_io import import_session
            session_data = import_session(args.resume)
            agent.load_session(session_data)
            console.print(
                f"[bold green]Resumed session {session_data.session_id}:[/bold green] "
                f"{len(session_data.messages)} messages loaded"
            )
        except Exception as exc:
            console.print(f"[bold red]Failed to resume session '{args.resume}':[/bold red] {exc}")
            sys.exit(1)
    run_interactive(agent, args.prompt)


def cmd_dna(args):
    """Project DNA report."""
    from cliv.intelligence import analyse_project
    from cliv.security import project_root

    try:
        report = analyse_project(project_root())
    except Exception as exc:
        console.print(f"[bold red]Project DNA failed:[/bold red] {exc}")
        sys.exit(1)
    console.print(report, markup=False)


def cmd_graph(args):
    """Visual codebase explorer."""
    from cliv.intelligence import run_graph_explorer

    try:
        sys.exit(
            run_graph_explorer(
                interactive=args.explore,
                export_mermaid=args.mermaid,
                export_graphviz=args.graphviz,
                console=console,
            )
        )
    except Exception as exc:
        console.print(f"[bold red]Graph Explorer failed:[/bold red] {exc}")
        sys.exit(1)


def cmd_rag(args):
    """RAG semantic memory."""
    from cliv.intelligence import create_rag_indexer

    try:
        indexer = create_rag_indexer()
        if args.rag_command == "index":
            with console.status("[dim]Building RAG index...[/dim]", spinner="dots"):
                stats = indexer.index(force_full=args.force)
            console.print(f"[green]RAG index built:[/green] {stats.chunks_added} chunks, {stats.files_changed} files changed, {stats.duration_seconds:.1f}s")
            if stats.errors:
                for err in stats.errors:
                    console.print(f"[yellow]Warning:[/yellow] {err}")
        elif args.rag_command == "query":
            with console.status("[dim]Searching...[/dim]", spinner="dots"):
                results = indexer.search(args.query, k=args.k)
            if results:
                console.print(f"[bold]RAG Results for:[/bold] {args.query}")
                for i, (chunk, score) in enumerate(results, 1):
                    console.print(f"  [{i}] {chunk.file_path} (score: {score:.3f})")
                    if chunk.symbol_name:
                        console.print(f"      Symbol: {chunk.symbol_name} ({chunk.symbol_kind})")
                    for line in chunk.content.splitlines()[:5]:
                        console.print(f"      {line}")
                    if len(chunk.content.splitlines()) > 5:
                        console.print("      ...")
            else:
                console.print("[yellow]No results found. Build index first with 'cliv rag index'[/yellow]")
        elif args.rag_command == "stats":
            stats = indexer.get_stats()
            console.print("[bold]RAG Index Statistics[/bold]")
            console.print(f"  Total chunks: {stats['total_chunks']}")
            console.print(f"  Unique files: {stats['unique_files']}")
            console.print(f"  Vector dimension: {stats['dimension']}")
            console.print(f"  Index size: {stats['index_size']}")
    except Exception as exc:
        console.print(f"[bold red]RAG operation failed:[/bold red] {exc}")
        sys.exit(1)


def cmd_timeline(args):
    """Development timeline."""
    from cliv.timeline import TimelineStore, render_timeline

    try:
        store = TimelineStore()
        limit = getattr(args, 'limit', 25)
        events = store.recent(limit=limit)
        console.print(render_timeline(events), markup=False)
    except Exception as exc:
        console.print(f"[bold red]Timeline failed:[/bold red] {exc}")
        sys.exit(1)


def cmd_snapshots(args):
    """Safety snapshots."""
    from cliv.snapshots import SnapshotStore, render_snapshot, render_snapshots
    from cliv.timeline import TimelineStore

    command = getattr(args, 'snapshots_command', 'list')
    
    if command == "list":
        try:
            store = SnapshotStore()
            limit = getattr(args, 'limit', 25)
            console.print(
                render_snapshots(store.recent(limit=limit)), markup=False
            )
        except Exception as exc:
            console.print(f"[bold red]Snapshots failed:[/bold red] {exc}")
            sys.exit(1)

    elif command == "undo":
        try:
            store = SnapshotStore()
            selector = str(getattr(args, 'snapshot_id', 'last') or "last")
            if selector in {"last", "latest"}:
                snapshot = store.latest_applied()
            else:
                snapshot = store.get(selector)
            if snapshot is None:
                console.print("[bold yellow]No applied snapshots to undo.[/bold yellow]")
                return
            console.print(render_snapshot(snapshot), markup=False)
            if snapshot.status != "applied":
                console.print(
                    f"[bold yellow]Snapshot is {snapshot.status}; nothing to undo.[/bold yellow]"
                )
                return
            if not getattr(args, 'yes', False):
                try:
                    answer = input("Restore these files? [y/N]: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    console.print("\n[bold yellow]Undo cancelled.[/bold yellow]")
                    return
                if answer not in {"y", "yes"}:
                    console.print("[bold yellow]Undo cancelled.[/bold yellow]")
                    return
            result = store.rollback(snapshot.id, force=getattr(args, 'force', False))
            if result.success:
                console.print(f"[bold green]{result.message}[/bold green]")
            else:
                console.print(f"[bold red]{result.message}[/bold red]")
                for conflict in result.conflicts:
                    console.print(f"[dim]  - {conflict}[/dim]")
            try:
                TimelineStore().record(
                    "undo",
                    f"Undo {snapshot.id}: {'restored' if result.success else 'failed'}",
                    {
                        "snapshot_id": snapshot.id,
                        "success": result.success,
                        "files": [entry.path for entry in snapshot.files],
                        "conflicts": list(result.conflicts),
                    },
                )
            except Exception:
                pass
            if not result.success:
                sys.exit(1)
        except Exception as exc:
            console.print(f"[bold red]Undo failed:[/bold red] {exc}")
            sys.exit(1)


def cmd_git(args):
    """Git integration."""
    from cliv.git import GitRepo

    repo = GitRepo()
    if args.git_command == "status":
        console.print(repo.status_short())
    elif args.git_command == "branch":
        if args.name:
            repo.create_branch(args.name)
            console.print(f"[green]Created and switched to branch:[/green] {args.name}")
        else:
            console.print(f"Current branch: [cyan]{repo.current_branch()}[/cyan]")
    elif args.git_command == "commit":
        repo.add_and_commit(args.message, args.files)
        console.print("[green]Committed[/green]")


def cmd_config(args):
    """Configuration management."""
    from cliv.config import get_config, set_config, get_api_key, save_api_key

    if args.config_command == "get":
        if args.key:
            val = get_config(args.key)
            console.print(val if val is not None else "[dim]Not set[/dim]")
        else:
            console.print("[yellow]Specify a key[/yellow]")
    elif args.config_command == "set":
        if args.key and args.value is not None:
            set_config(args.key, args.value)
            console.print(f"[green]Set {args.key}[/green]")
        else:
            console.print("[yellow]Specify key and value[/yellow]")
    elif args.config_command == "list":
        # Show known config keys
        console.print("Config keys: api_key, groq_model, groq_fallback_models, ...")
    elif args.config_command == "api-key":
        if args.value:
            save_api_key(args.value)
            console.print("[green]API key saved[/green]")
        else:
            key = getpass("Enter API Key: ").strip()
            if key:
                save_api_key(key)
                console.print("[green]API key saved[/green]")


def cmd_session(args):
    """Session management."""
    from cliv.history import HistoryStore
    from cliv.session_io import export_session_json, export_session_markdown, import_session

    store = HistoryStore()
    if args.session_command == "list":
        sessions = store.list_sessions(limit=getattr(args, "limit", 10))
        if not sessions:
            console.print("[dim]No sessions found.[/dim]")
            return
        for s in sessions:
            console.print(f"  {s['id']}  {s['created']}  {s['turns']} turns  ({s['model']})")
    elif args.session_command == "clear":
        store.clear()
        console.print("[green]History cleared[/green]")
    elif args.session_command == "export":
        target = args.session_id
        fmt = getattr(args, "format", "md")
        out = getattr(args, "output", None) or f"{target}.{fmt}"
        try:
            session = import_session(target)
            if fmt == "json":
                saved = export_session_json(session, out)
            else:
                saved = export_session_markdown(session, out)
            console.print(f"[bold green]Exported session to:[/bold green] {saved}")
        except Exception as exc:
            console.print(f"[bold red]Export failed:[/bold red] {exc}")
            sys.exit(1)
    elif args.session_command == "import":
        target = args.path
        try:
            session = import_session(target)
            store.save(session.messages)
            console.print(
                f"[bold green]Imported session {session.session_id}:[/bold green] "
                f"{len(session.messages)} messages set as active conversation history"
            )
        except Exception as exc:
            console.print(f"[bold red]Import failed:[/bold red] {exc}")
            sys.exit(1)


def cmd_models(args):
    """List available models."""
    from cliv.llm.router import list_groq_models

    try:
        models = list_groq_models()
        for m in models[:20]:
            console.print(f"  {m['id']}  ({m.get('owned_by', 'unknown')})")
    except Exception as exc:
        console.print(f"[bold red]Models failed:[/bold red] {exc}")


def cmd_cost(args):
    """Cost & token usage for current project."""
    from cliv.timeline import TimelineStore
    from cliv.security import project_root

    try:
        store = TimelineStore()
        cost_data = store.get_project_cost(project=project_root())
        console.print(f"\n[bold cyan]Project Cost Summary:[/bold cyan] {cost_data['project_root']}")
        console.print(f"  Total Sessions:   {cost_data['total_sessions']}")
        console.print(f"  Input Tokens:     {cost_data['input_tokens']:,}")
        console.print(f"  Output Tokens:    {cost_data['output_tokens']:,}")
        console.print(f"  Total Tokens:     {cost_data['total_tokens']:,}")
        console.print(f"  Cumulative Spend: [bold green]${cost_data['total_cost_usd']:.6f}[/bold green]")
        if cost_data.get("models"):
            console.print("\n[bold]Spend by Model:[/bold]")
            for m in cost_data["models"]:
                console.print(f"  - {m['model']}: {m['tokens']:,} tokens (${m['cost']:.6f})")
        console.print()
    except Exception as exc:
        console.print(f"[bold red]Failed to retrieve project cost:[/bold red] {exc}")
        sys.exit(1)


def cmd_tools(args):
    """Custom tool plugins management."""
    import json
    from rich.table import Table
    from cliv.tools.loader import load_tools, create_plugin_template
    from cliv.security import project_root
    from cliv.config import PLUGINS_DIR

    tools_command = getattr(args, "tools_command", None) or "list"

    if tools_command == "list":
        tools = load_tools(project=project_root())
        table = Table(
            title=f"Installed Tools & Plugins ({project_root()})",
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

        for name, tool in sorted(tools.items()):
            source = getattr(tool, "plugin_source", "builtin")
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

        console.print()
        console.print(table)
        console.print(f"\n[dim]{len(tools)} tools available. Run 'cliv tools new <name>' to create a plugin.[/dim]\n")

    elif tools_command == "new":
        name = getattr(args, "name", "")
        if not name:
            console.print("[bold red]Error:[/bold red] Tool name is required.")
            sys.exit(1)
        is_global = getattr(args, "is_global", False)
        target_dir = PLUGINS_DIR if is_global else (project_root() / ".cliv" / "plugins")
        path = create_plugin_template(name, target_dir)
        console.print(f"\n[bold green]Created tool plugin scaffold:[/bold green] {path}")
        console.print("[dim]Edit the file to customize arguments and behavior.[/dim]\n")

    elif tools_command == "info":
        name = getattr(args, "name", "")
        if not name:
            console.print("[bold red]Error:[/bold red] Tool name is required.")
            sys.exit(1)
        tools = load_tools(project=project_root())
        tool = tools.get(name)
        if not tool:
            console.print(f"[bold red]Tool not found:[/bold red] {name}")
            sys.exit(1)

        console.print(f"\n[bold cyan]Tool:[/bold cyan] [bold]{tool.name}[/bold]")
        console.print(f"  Source:      {getattr(tool, 'plugin_source', 'builtin')}")
        if getattr(tool, "plugin_path", None):
            console.print(f"  Path:        {tool.plugin_path}")
        console.print(f"  Risk Level:  {getattr(tool, 'risk_level', 'high')}")
        console.print(f"  Permissions: {','.join(sorted(getattr(tool, 'permissions', []))) or 'none'}")
        console.print(f"  Description: {getattr(tool, 'description', '')}")
        console.print("\n[bold]Input Schema:[/bold]")
        console.print(json.dumps(getattr(tool, "input_schema", {}), indent=2))
        console.print()


def cmd_memory(args):
    """Manage long-term agent memory and project conventions."""
    from rich.table import Table
    from cliv.memory import MemoryStore

    memory_command = getattr(args, "memory_command", None) or "list"
    store = MemoryStore()

    if memory_command == "list":
        scope = getattr(args, "scope", None)
        entries = store.list_entries(scope=scope)
        cliv_md = store.get_cliv_md_content()

        if not entries and not cliv_md:
            console.print("[dim]No remembered preferences or CLIV.md guidelines found.[/dim]")
            console.print("[dim]Run 'cliv memory add <text>' to save a convention.[/dim]")
            return

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

        console.print()
        console.print(table)
        if cliv_md:
            console.print(f"\n[bold]CLIV.md Guidelines:[/bold] ({len(cliv_md)} chars loaded from project root)")
        console.print()

    elif memory_command == "add":
        content = getattr(args, "content", "")
        if not content:
            console.print("[bold red]Error:[/bold red] Memory content is required.")
            sys.exit(1)
        is_global = getattr(args, "is_global", False)
        category = getattr(args, "category", "general") or "general"
        scope = "global" if is_global else "project"
        entry = store.add(content=content, scope=scope, category=category)
        console.print(f"\n[bold green]Saved {scope} memory (id: {entry.id}):[/bold green]")
        console.print(f"  \"{entry.content}\"\n")

    elif memory_command in {"remove", "rm", "delete"}:
        target = getattr(args, "target", "")
        if not target:
            console.print("[bold red]Error:[/bold red] Target ID or keyword is required.")
            sys.exit(1)
        removed = store.remove(target)
        if removed:
            console.print(f"\n[bold green]Removed memory matching:[/bold green] '{target}'\n")
        else:
            console.print(f"[yellow]No memory found matching:[/yellow] '{target}'")

    elif memory_command == "clear":
        scope = getattr(args, "scope", None)
        count = store.clear(scope=scope)
        target = f"{scope} " if scope else ""
        console.print(f"\n[bold green]Cleared {count} {target}memories.[/bold green]\n")


def cmd_upgrade(args):
    """Self-upgrade."""
    console.print("[dim]Upgrade not yet implemented[/dim]")


# ============================================================
# Main entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        prog="cliv",
        description="> cliv_ : Terminal-native intelligence for your local codebase.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )

    # Global flags
    parser.add_argument(
        "-h", "--help", action="store_true", help="Show this help message and exit"
    )
    parser.add_argument(
        "--api-key", help="Groq API key (overrides saved config or env var)"
    )
    parser.add_argument(
        "--offline", action="store_true", help="Force offline mode (use Ollama)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--yes", "-y", action="store_true", help="Auto-approve all tool operations"
    )
    parser.add_argument(
        "--dry-run", "-n", action="store_true", help="Preview without applying changes"
    )
    parser.add_argument(
        "--legacy-offline",
        action="store_true",
        help="Use regex intent routing instead of Ollama tool calling",
    )
    parser.add_argument(
        "--permissions",
        metavar="LIST",
        help="Comma-separated permissions: read,write,execute,network (default: all)",
    )
    parser.add_argument(
        "--branch",
        metavar="NAME",
        help="Create/switch to a git branch for this session",
    )
    parser.add_argument(
        "--auto-commit",
        action="store_true",
        help="Auto-commit files cliv modifies during the session",
    )
    parser.add_argument(
        "--resume",
        metavar="SESSION",
        help="Resume a past session by ID or file path",
    )

    # Legacy flags (for backward compatibility)
    parser.add_argument(
        "--project-dna", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--timeline", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--snapshots", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--undo", nargs="?", const="last", metavar="SNAPSHOT_ID", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--force", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--graph-explorer", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--export-mermaid", metavar="PATH", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--export-graphviz", metavar="PATH", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--rag-index", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--rag-query", metavar="QUERY", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--rag-stats", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--rag-force-rebuild", action="store_true", help=argparse.SUPPRESS
    )

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # run (default)
    p_run = subparsers.add_parser("run", help="Interactive chat or single-shot (default)", add_help=False)
    p_run.add_argument("-h", "--help", action="store_true")
    p_run.add_argument("prompt", nargs="*", help="Prompt for single-shot mode")
    p_run.set_defaults(func=cmd_run)

    # dna
    p_dna = subparsers.add_parser("dna", help="Project DNA architecture report", add_help=False)
    p_dna.add_argument("-h", "--help", action="store_true")
    p_dna.set_defaults(func=cmd_dna)

    # graph
    p_graph = subparsers.add_parser("graph", help="Visual codebase explorer", add_help=False)
    p_graph.add_argument("-h", "--help", action="store_true")
    p_graph.add_argument("--explore", action="store_true", help="Interactive mode (default)")
    p_graph.add_argument("--mermaid", metavar="PATH", help="Export Mermaid diagram")
    p_graph.add_argument("--graphviz", metavar="PATH", help="Export Graphviz DOT")
    p_graph.set_defaults(func=cmd_graph, explore=True)

    # rag
    p_rag = subparsers.add_parser("rag", help="RAG semantic memory", add_help=False)
    p_rag.add_argument("-h", "--help", action="store_true")
    rag_sub = p_rag.add_subparsers(dest="rag_command", required=True)
    p_rag_index = rag_sub.add_parser("index", help="Build/update index", add_help=False)
    p_rag_index.add_argument("-h", "--help", action="store_true")
    p_rag_index.add_argument("--force", action="store_true", help="Full rebuild")
    p_rag_query = rag_sub.add_parser("query", help="Search index", add_help=False)
    p_rag_query.add_argument("-h", "--help", action="store_true")
    p_rag_query.add_argument("query", help="Search query")
    p_rag_query.add_argument("-k", type=int, default=10, help="Max results")
    p_rag_stats = rag_sub.add_parser("stats", help="Index statistics", add_help=False)
    p_rag_stats.add_argument("-h", "--help", action="store_true")
    p_rag.set_defaults(func=cmd_rag)

    # timeline
    p_timeline = subparsers.add_parser("timeline", help="Development timeline", add_help=False)
    p_timeline.add_argument("-h", "--help", action="store_true")
    p_timeline.add_argument("-n", "--limit", type=int, default=25, help="Max events")
    p_timeline.set_defaults(func=cmd_timeline)

    # snapshots
    p_snap = subparsers.add_parser("snapshots", help="Safety snapshots", add_help=False)
    p_snap.add_argument("-h", "--help", action="store_true")
    snap_sub = p_snap.add_subparsers(dest="snapshots_command", required=False)
    p_snap_list = snap_sub.add_parser("list", help="List snapshots", add_help=False)
    p_snap_list.add_argument("-h", "--help", action="store_true")
    p_snap_list.add_argument("-n", "--limit", type=int, default=25)
    p_snap_undo = snap_sub.add_parser("undo", help="Restore snapshot", add_help=False)
    p_snap_undo.add_argument("-h", "--help", action="store_true")
    p_snap_undo.add_argument("snapshot_id", nargs="?", default="last")
    p_snap_undo.add_argument("--force", action="store_true", help="Overwrite conflicts")
    p_snap_undo.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")
    p_snap.set_defaults(func=cmd_snapshots)

    # git
    p_git = subparsers.add_parser("git", help="Git integration", add_help=False)
    p_git.add_argument("-h", "--help", action="store_true")
    git_sub = p_git.add_subparsers(dest="git_command", required=True)
    p_git_status = git_sub.add_parser("status", help="Show status", add_help=False)
    p_git_status.add_argument("-h", "--help", action="store_true")
    p_git_branch = git_sub.add_parser("branch", help="Show or create branch", add_help=False)
    p_git_branch.add_argument("-h", "--help", action="store_true")
    p_git_branch.add_argument("name", nargs="?", help="New branch name")
    p_git_commit = git_sub.add_parser("commit", help="Commit changes", add_help=False)
    p_git_commit.add_argument("-h", "--help", action="store_true")
    p_git_commit.add_argument("message", help="Commit message")
    p_git_commit.add_argument("files", nargs="*", help="Files to stage")
    p_git.set_defaults(func=cmd_git)

    # config
    p_config = subparsers.add_parser("config", help="Configuration management", add_help=False)
    p_config.add_argument("-h", "--help", action="store_true")
    config_sub = p_config.add_subparsers(dest="config_command", required=True)
    p_config_get = config_sub.add_parser("get", help="Get config value", add_help=False)
    p_config_get.add_argument("-h", "--help", action="store_true")
    p_config_get.add_argument("key", help="Config key")
    p_config_set = config_sub.add_parser("set", help="Set config value", add_help=False)
    p_config_set.add_argument("-h", "--help", action="store_true")
    p_config_set.add_argument("key", help="Config key")
    p_config_set.add_argument("value", help="Config value")
    p_config_list = config_sub.add_parser("list", help="List config keys", add_help=False)
    p_config_list.add_argument("-h", "--help", action="store_true")
    p_config_apikey = config_sub.add_parser("api-key", help="Set API key interactively", add_help=False)
    p_config_apikey.add_argument("-h", "--help", action="store_true")
    p_config_apikey.add_argument("value", nargs="?", help="API key (optional)")
    p_config.set_defaults(func=cmd_config)

    # session
    p_session = subparsers.add_parser("session", help="Session management", add_help=False)
    p_session.add_argument("-h", "--help", action="store_true")
    session_sub = p_session.add_subparsers(dest="session_command", required=True)
    p_session_list = session_sub.add_parser("list", help="List sessions", add_help=False)
    p_session_list.add_argument("-h", "--help", action="store_true")
    p_session_list.add_argument("-n", "--limit", type=int, default=10)
    p_session_clear = session_sub.add_parser("clear", help="Clear history", add_help=False)
    p_session_clear.add_argument("-h", "--help", action="store_true")
    p_session_export = session_sub.add_parser("export", help="Export session to file", add_help=False)
    p_session_export.add_argument("-h", "--help", action="store_true")
    p_session_export.add_argument("session_id", help="Session ID or archive file")
    p_session_export.add_argument("-o", "--output", help="Output file path")
    p_session_export.add_argument("--format", choices=["md", "json"], default="md", help="Export format")
    p_session_import = session_sub.add_parser("import", help="Import session from file", add_help=False)
    p_session_import.add_argument("-h", "--help", action="store_true")
    p_session_import.add_argument("path", help="Path to session JSON file or session ID")
    p_session.set_defaults(func=cmd_session)

    # models
    p_models = subparsers.add_parser("models", help="List available models", add_help=False)
    p_models.add_argument("-h", "--help", action="store_true")
    p_models.set_defaults(func=cmd_models)

    # cost
    p_cost = subparsers.add_parser("cost", help="Show token usage and cost metrics for this project", add_help=False)
    p_cost.add_argument("-h", "--help", action="store_true")
    p_cost.set_defaults(func=cmd_cost)

    # tools / plugins
    p_tools = subparsers.add_parser("tools", aliases=["plugins"], help="Custom tool plugins", add_help=False)
    p_tools.add_argument("-h", "--help", action="store_true")
    tools_sub = p_tools.add_subparsers(dest="tools_command", required=False)
    p_tools_list = tools_sub.add_parser("list", help="List loaded tools", add_help=False)
    p_tools_list.add_argument("-h", "--help", action="store_true")
    p_tools_new = tools_sub.add_parser("new", help="Scaffold a new tool plugin", add_help=False)
    p_tools_new.add_argument("-h", "--help", action="store_true")
    p_tools_new.add_argument("name", help="Tool name")
    p_tools_new.add_argument("--global", dest="is_global", action="store_true", help="Scaffold in user-global plugins directory")
    p_tools_info = tools_sub.add_parser("info", help="Inspect a tool's schema and details", add_help=False)
    p_tools_info.add_argument("-h", "--help", action="store_true")
    p_tools_info.add_argument("name", help="Tool name")
    p_tools.set_defaults(func=cmd_tools)

    # memory
    p_memory = subparsers.add_parser("memory", help="Agent long-term memory & conventions", add_help=False)
    p_memory.add_argument("-h", "--help", action="store_true")
    mem_sub = p_memory.add_subparsers(dest="memory_command", required=False)
    p_mem_list = mem_sub.add_parser("list", help="List remembered conventions", add_help=False)
    p_mem_list.add_argument("-h", "--help", action="store_true")
    p_mem_list.add_argument("--scope", choices=["global", "project"], help="Filter by scope")
    p_mem_add = mem_sub.add_parser("add", help="Add a new memory or convention", add_help=False)
    p_mem_add.add_argument("-h", "--help", action="store_true")
    p_mem_add.add_argument("content", help="The preference or convention to remember")
    p_mem_add.add_argument("--global", dest="is_global", action="store_true", help="Store in global memory across projects")
    p_mem_add.add_argument("--category", default="general", help="Category (preference, convention, architecture, tooling)")
    p_mem_rm = mem_sub.add_parser("remove", aliases=["rm", "delete"], help="Remove a remembered item", add_help=False)
    p_mem_rm.add_argument("-h", "--help", action="store_true")
    p_mem_rm.add_argument("target", help="Memory ID or text substring to delete")
    p_mem_clear = mem_sub.add_parser("clear", help="Clear remembered items", add_help=False)
    p_mem_clear.add_argument("-h", "--help", action="store_true")
    p_mem_clear.add_argument("--scope", choices=["global", "project"], help="Scope to clear")
    p_memory.set_defaults(func=cmd_memory)

    # upgrade
    p_upgrade = subparsers.add_parser("upgrade", help="Upgrade cliv", add_help=False)
    p_upgrade.add_argument("-h", "--help", action="store_true")
    p_upgrade.set_defaults(func=cmd_upgrade)

    # Check sys.argv directly for legacy flags BEFORE argparse
    argv = sys.argv[1:]
    
    # Check for legacy flags that should bypass subcommand parsing
    # These are the OLD flat flags that were replaced by subcommands
    legacy_flags = {"--project-dna", "--timeline", "--snapshots", "--undo", "--graph-explorer", 
                    "--export-mermaid", "--export-graphviz", "--rag-index", "--rag-query", "--rag-stats",
                    "--rag-force-rebuild", "--force"}
    has_legacy = any(arg in legacy_flags or arg.startswith(tuple(f"{f}=" for f in legacy_flags if "=" not in f)) for arg in argv)
    
    # Also check if first non-flag arg is a legacy positional command (undo, snapshots)
    first_non_flag = None
    for arg in argv:
        if not arg.startswith("-"):
            first_non_flag = arg
            break
    if first_non_flag in ("undo", "snapshots"):
        has_legacy = True
    
    # Check for help first
    if "-h" in argv or "--help" in argv:
        parser.print_help()
        print("\nCommands:")
        print("  run          Interactive chat or single-shot (default)")
        print("  dna          Project DNA architecture report")
        print("  graph        Visual codebase explorer")
        print("  rag          RAG semantic memory (index, query, stats)")
        print("  timeline     Development timeline")
        print("  snapshots    Safety snapshots (list, undo)")
        print("  git          Git integration (status, branch, commit)")
        print("  config       Configuration management (get, set, list, api-key)")
        print("  session      Session management (list, clear)")
        print("  models       List available models")
        print("  cost         Token usage and spend metrics for this project")
        print("  upgrade      Upgrade cliv")
        print("\nGlobal options:")
        print("  --api-key, --offline, --verbose, --yes, --dry-run, --legacy-offline")
        print("  --permissions, --branch, --auto-commit")
        sys.exit(0)

    # Handle legacy flags directly (they don't need subcommand parsing)
    if has_legacy:
        # Handle legacy positional commands (undo, snapshots) before parsing
        # These were positional commands in the old flat interface
        legacy_argv = list(argv)
        first_non_flag_idx = None
        for i, arg in enumerate(legacy_argv):
            if not arg.startswith("-"):
                first_non_flag_idx = i
                break
        
        if first_non_flag_idx is not None and legacy_argv[first_non_flag_idx] in ("undo", "snapshots"):
            # Map legacy positional to new subcommand structure
            if legacy_argv[first_non_flag_idx] == "undo":
                legacy_argv[first_non_flag_idx] = "snapshots"
                legacy_argv.insert(first_non_flag_idx + 1, "undo")
            elif legacy_argv[first_non_flag_idx] == "snapshots":
                legacy_argv[first_non_flag_idx] = "snapshots"
                if first_non_flag_idx + 1 >= len(legacy_argv) or legacy_argv[first_non_flag_idx + 1].startswith("-"):
                    legacy_argv.insert(first_non_flag_idx + 1, "list")
        
        # Parse with legacy flags included
        args = parser.parse_args(legacy_argv)
        
        # Validate conflicts
        if args.yes and args.dry_run:
            console.print("[bold red]Error: --yes and --dry-run are mutually exclusive.[/bold red]")
            sys.exit(1)

        if args.project_dna:
            cmd_dna(args)
            return
        if args.timeline:
            cmd_timeline(args)
            return
        if args.snapshots:
            args.snapshots_command = "list"
            args.limit = getattr(args, 'limit', 25)
            cmd_snapshots(args)
            return
        if args.undo is not None:
            args.snapshots_command = "undo"
            args.snapshot_id = args.undo
            args.force = getattr(args, 'force', False)
            args.yes = getattr(args, 'yes', False)
            cmd_snapshots(args)
            return
        if args.graph_explorer or args.export_mermaid or args.export_graphviz:
            args.explore = args.graph_explorer
            args.mermaid = args.export_mermaid
            args.graphviz = args.export_graphviz
            cmd_graph(args)
            return
        if args.rag_index or args.rag_query or args.rag_stats:
            if args.rag_index:
                args.rag_command = "index"
                args.force = args.rag_force_rebuild
            elif args.rag_query:
                args.rag_command = "query"
                args.query = args.rag_query
                args.k = 10
            elif args.rag_stats:
                args.rag_command = "stats"
            cmd_rag(args)
            return

        # Handle mapped subcommands (from legacy positional commands like "undo", "snapshots")
        if getattr(args, 'command', None) == "snapshots":
            if getattr(args, 'snapshots_command', None) == "undo":
                args.snapshot_id = getattr(args, 'snapshot_id', 'last')
                args.force = getattr(args, 'force', False)
                args.yes = getattr(args, 'yes', False)
                cmd_snapshots(args)
                return
            elif getattr(args, 'snapshots_command', None) == "list":
                args.limit = getattr(args, 'limit', 25)
                cmd_snapshots(args)
                return
            # Default to list if no subcommand
            args.snapshots_command = "list"
            args.limit = getattr(args, 'limit', 25)
            cmd_snapshots(args)
            return

    # No legacy flags - use subcommand parsing
    # First, parse global flags using parse_known_args
    global_parser = argparse.ArgumentParser(add_help=False)
    global_parser.add_argument("--api-key", help=argparse.SUPPRESS)
    global_parser.add_argument("--offline", action="store_true", help=argparse.SUPPRESS)
    global_parser.add_argument("--verbose", "-v", action="store_true", help=argparse.SUPPRESS)
    global_parser.add_argument("--yes", "-y", action="store_true", help=argparse.SUPPRESS)
    global_parser.add_argument("--dry-run", "-n", action="store_true", help=argparse.SUPPRESS)
    global_parser.add_argument("--legacy-offline", action="store_true", help=argparse.SUPPRESS)
    global_parser.add_argument("--permissions", metavar="LIST", help=argparse.SUPPRESS)
    global_parser.add_argument("--branch", metavar="NAME", help=argparse.SUPPRESS)
    global_parser.add_argument("--auto-commit", action="store_true", help=argparse.SUPPRESS)
    global_parser.add_argument("--resume", metavar="SESSION", help=argparse.SUPPRESS)
    global_parser.add_argument("-h", "--help", action="store_true", help=argparse.SUPPRESS)
    
    # Parse global flags first
    global_args, remaining_argv = global_parser.parse_known_args(argv)
    
    # Check if there's a positional prompt (not a flag) in remaining
    first_non_flag = None
    for arg in remaining_argv:
        if not arg.startswith("-"):
            first_non_flag = arg
            break
    
    # If first non-flag is a known subcommand, use it; otherwise default to run
    if first_non_flag and first_non_flag not in subparsers.choices:
        # It's a prompt, not a subcommand - default to run
        remaining_argv = ["run", *remaining_argv]
    elif not first_non_flag and remaining_argv and remaining_argv[0].startswith("-"):
        # Only flags, no command - default to run
        remaining_argv = ["run", *remaining_argv]
    elif remaining_argv and remaining_argv[0] in ("undo", "snapshots"):
        # Legacy aliases
        if remaining_argv[0] == "undo":
            remaining_argv = ["snapshots", "undo", *remaining_argv[1:]]
        elif remaining_argv[0] == "snapshots":
            remaining_argv = ["snapshots", "list", *remaining_argv[1:]]
    elif not remaining_argv or remaining_argv[0] not in subparsers.choices:
        # No args or unknown - default to run
        remaining_argv = ["run", *remaining_argv]

    # Now parse with the main parser (which includes global flags + subcommands)
    # We need to reconstruct argv with global flags first
    reconstructed_argv = []
    if global_args.api_key:
        reconstructed_argv.extend(["--api-key", global_args.api_key])
    if global_args.offline:
        reconstructed_argv.append("--offline")
    if global_args.verbose:
        reconstructed_argv.append("--verbose")
    if global_args.yes:
        reconstructed_argv.append("--yes")
    if global_args.dry_run:
        reconstructed_argv.append("--dry-run")
    if global_args.legacy_offline:
        reconstructed_argv.append("--legacy-offline")
    if global_args.permissions:
        reconstructed_argv.extend(["--permissions", global_args.permissions])
    if global_args.branch:
        reconstructed_argv.extend(["--branch", global_args.branch])
    if global_args.auto_commit:
        reconstructed_argv.append("--auto-commit")
    if global_args.resume:
        reconstructed_argv.extend(["--resume", global_args.resume])
    if global_args.help:
        reconstructed_argv.append("--help")
    reconstructed_argv.extend(remaining_argv)

    args = parser.parse_args(reconstructed_argv)

    # Join prompt list into string for run command
    if args.command == "run" and isinstance(args.prompt, list):
        args.prompt = " ".join(args.prompt) if args.prompt else None

    # Check subcommand help
    if getattr(args, "help", False):
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                if args.command in action.choices:
                    action.choices[args.command].print_help()
                    sys.exit(0)

    # Validate conflicts
    if args.yes and args.dry_run:
        console.print("[bold red]Error: --yes and --dry-run are mutually exclusive.[/bold red]")
        sys.exit(1)

    # Call handler
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()