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
from cliv.workspace import WorkspaceRegistry

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


def process_prompt(agent, user_input, is_json=False, is_quiet=False):
    """Handles the rendering of the agent's response."""
    if is_json:
        try:
            response = agent.chat(user_input)
            from cliv.output import format_agent_run_json, print_json

            print_json(format_agent_run_json(agent, user_input, response))
            return response
        except Exception as exc:
            from cliv.output import format_agent_run_json, print_json

            print_json(format_agent_run_json(agent, user_input, "", error=str(exc)))
            return None

    if is_quiet:
        response = agent.chat(user_input)
        if response:
            print(response)
        return response

    backend = getattr(agent, "backend", None)
    model_name = (
        getattr(backend, "model", None)
        if backend is not None
        else getattr(agent, "local_model", "unknown")
    )
    mode_tag = (
        f"[green]{agent.mode.upper()}[/green]"
        if agent.mode == "online"
        else f"[yellow]{agent.mode.upper()}[/yellow]"
    )

    console.print(
        f"\n[bold cyan]◆ cliv[/bold cyan] [dim]({model_name} · {mode_tag})[/dim]"
    )

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
    return response


def build_agent(args, security_guardian=None, timeline_store=None, snapshot_store=None):
    """Build agent from parsed args."""
    api_key = getattr(args, "api_key", None) or get_api_key()
    if api_key == "" or getattr(args, "offline", False):
        api_key = None

    quiet_mode = getattr(args, "quiet", False) or getattr(args, "json", False)

    if (
        not api_key
        and not getattr(args, "prompt", None)
        and not getattr(args, "offline", False)
        and not quiet_mode
    ):
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
            console.print("\n[bold yellow]Would you like to save this key for future use? (y/n): [/bold yellow]", end="")
            sys.stdout.flush()
            save = input().strip().lower()
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
        if not quiet_mode:
            with console.status(
                "[dim]Checking network connection...[/dim]", spinner="dots"
            ):
                is_online = check_internet()
        else:
            is_online = check_internet()

        if not is_online:
            if not quiet_mode:
                console.print(
                    "\n[bold yellow][SYSTEM WARNING][/bold yellow] No internet connection detected!"
                )
                console.print(
                    "[bold yellow]Automatically falling back to local OFFLINE mode.[/bold yellow]\n"
                )
            api_key = None

    if not quiet_mode:
        with console.status(
            "[dim]Configuring and validating model...[/dim]", spinner="dots"
        ):
            agent = AIAgent(
                api_key=api_key,
                verbose=getattr(args, "verbose", False),
                auto_approve=getattr(args, "yes", False),
                dry_run=getattr(args, "dry_run", False),
                force_offline=getattr(args, "offline", False),
                use_offline_tools=not getattr(args, "legacy_offline", False),
                ui=ConsoleUI(console),
                timeline_store=timeline_store,
                snapshot_store=snapshot_store,
                guardian=security_guardian,
                auto_commit=getattr(args, "auto_commit", False),
                git_branch=getattr(args, "branch", None),
            )
    else:
        agent = AIAgent(
            api_key=api_key,
            verbose=getattr(args, "verbose", False),
            auto_approve=getattr(args, "yes", False),
            dry_run=getattr(args, "dry_run", False),
            force_offline=getattr(args, "offline", False),
            use_offline_tools=not getattr(args, "legacy_offline", False),
            ui=ConsoleUI(console),
            timeline_store=timeline_store,
            snapshot_store=snapshot_store,
            guardian=security_guardian,
            auto_commit=getattr(args, "auto_commit", False),
            git_branch=getattr(args, "branch", None),
        )

    if quiet_mode:
        return agent

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
    if backend is None:
        console.print("[bold red]Model: None (No LLM backend available)[/bold red]")
        console.print(
            "[yellow]Notice:[/yellow] Neither Groq API key nor local Ollama backend is active.\n"
            "  • Online: run [bold]cliv config api-key[/bold] (free key at console.groq.com)\n"
            "  • Offline: run [bold]ollama serve[/bold] and [bold]ollama pull llama3.1:8b[/bold]\n"
            "  • Commands like [bold]/dna[/bold], [bold]/graph[/bold], [bold]/timeline[/bold] remain fully functional."
        )
    elif model_name:
        tier = (
            "direct tool calling"
            if getattr(agent, "use_offline_tools", True)
            else "intent fallback"
        )
        if agent.mode == "offline":
            console.print(f"Model: [bold]{model_name}[/bold] [dim]({tier})[/dim]")
        else:
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
        console.print(f"[dim]Permissions: {guardian.policy.render()}[/dim]")
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


def run_interactive(agent, prompt=None, is_json=False, is_quiet=False):
    """Run interactive REPL or single-shot."""
    if prompt:
        prompt_str = " ".join(prompt) if isinstance(prompt, list) else str(prompt)
        if is_json:
            process_prompt(agent, prompt_str, is_json=True)
            return
        if is_quiet:
            process_prompt(agent, prompt_str, is_quiet=True)
            return
        console.print(f"[dim]Executing command:[/dim] {prompt_str}")
        process_prompt(agent, prompt_str)
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

            is_agent_json = getattr(agent, "json_mode", False)
            is_agent_quiet = getattr(agent, "quiet_mode", False)
            process_prompt(
                agent, user_input, is_json=is_agent_json, is_quiet=is_agent_quiet
            )

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
            token.lower() for token in re.split(r"[,\s]+", args.permissions) if token
        }
        unknown = requested - ALL_PERMISSIONS
        if unknown:
            console.print(
                "[bold red]Error: unknown permission(s): "
                f"{', '.join(sorted(unknown))}. "
                "Choose from read,write,execute,network.[/bold red]"
            )
            sys.exit(1)
        security_guardian = SecurityGuardian(SecurityPolicy.parse(args.permissions))

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
            console.print(
                f"[bold red]Failed to resume session '{args.resume}':[/bold red] {exc}"
            )
            sys.exit(1)
    is_json = getattr(args, "json", False)
    is_quiet = getattr(args, "quiet", False)
    run_interactive(agent, args.prompt, is_json=is_json, is_quiet=is_quiet)


def cmd_dna(args):
    """Project DNA report."""
    from cliv.intelligence import analyse_project
    from cliv.intelligence.report import build_project_intelligence
    from cliv.security import project_root
    from cliv.output import print_json

    root = project_root()
    if getattr(args, "json", False):
        intel = build_project_intelligence(root)
        print_json(intel.to_dict())
        return

    try:
        report = analyse_project(root)
    except Exception as exc:
        console.print(f"[bold red]Project DNA failed:[/bold red] {exc}")
        sys.exit(1)
    console.print(report, markup=False)


def cmd_onboard(args):
    """Generate developer and agent onboarding report."""
    from cliv.onboarding import generate_onboarding_report
    from cliv.output import print_json

    path = getattr(args, "path", ".") or "."
    try:
        report = generate_onboarding_report(path)

        if getattr(args, "json", False):
            print_json(report.to_dict())
            return

        if getattr(args, "save", False) or getattr(args, "output", None):
            saved = report.save(dest=getattr(args, "output", None))
            console.print(
                f"\n[bold green]✓ Onboarding report saved to:[/bold green] {saved}\n"
            )
            return

        console.print()
        console.print(Markdown(report.render_markdown()))
        console.print(
            "\n[dim]To save this report to ONBOARDING.md, run: [bold]cliv onboard --save[/bold] "
            "or [bold]cliv onboard -o <path>[/bold][/dim]\n"
        )
    except Exception as exc:
        console.print(
            f"[bold red]Onboarding report generation failed:[/bold red] {exc}"
        )
        sys.exit(1)


def cmd_review(args):
    """Automated code review & CI check."""
    from cliv.review import generate_github_workflow, run_code_review
    from cliv.output import print_json

    if getattr(args, "init_ci", False):
        wf = generate_github_workflow()
        console.print(
            f"\n[bold green]✓ GitHub Actions workflow initialized:[/bold green] {wf}\n"
        )
        return

    path = getattr(args, "path", ".") or "."
    base_ref = getattr(args, "base", None)
    format_mode = getattr(args, "format", "standard") or "standard"
    execute_tests = getattr(args, "run_tests", False)

    try:
        report = run_code_review(
            root=path,
            base_ref=base_ref,
            execute_tests=execute_tests,
        )

        if getattr(args, "json", False):
            print_json(report.to_dict())
        else:
            rendered = report.render_markdown(format_mode=format_mode)
            if format_mode == "github":
                print(rendered)
            else:
                console.print()
                console.print(Markdown(rendered))
                console.print()

        if not report.passed and not getattr(args, "no_fail", False):
            sys.exit(1)

    except Exception as exc:
        console.print(f"[bold red]Code review failed:[/bold red] {exc}")
        sys.exit(1)


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
            console.print(
                f"[green]RAG index built:[/green] {stats.chunks_added} chunks, {stats.files_changed} files changed, {stats.duration_seconds:.1f}s"
            )
            if stats.errors:
                for err in stats.errors:
                    console.print(f"[yellow]Warning:[/yellow] {err}")
        elif args.rag_command == "query":
            with console.status("[dim]Searching...[/dim]", spinner="dots"):
                results = indexer.search(args.query, k=args.k)
            if getattr(args, "json", False):
                from cliv.output import print_json

                print_json(
                    [
                        {
                            "file_path": c.file_path,
                            "score": score,
                            "symbol_name": c.symbol_name,
                            "symbol_kind": c.symbol_kind,
                            "content": c.content,
                        }
                        for c, score in results
                    ]
                )
                return
            if results:
                console.print(f"[bold]RAG Results for:[/bold] {args.query}")
                for i, (chunk, score) in enumerate(results, 1):
                    console.print(f"  [{i}] {chunk.file_path} (score: {score:.3f})")
                    if chunk.symbol_name:
                        console.print(
                            f"      Symbol: {chunk.symbol_name} ({chunk.symbol_kind})"
                        )
                    for line in chunk.content.splitlines()[:5]:
                        console.print(f"      {line}")
                    if len(chunk.content.splitlines()) > 5:
                        console.print("      ...")
            else:
                console.print(
                    "[yellow]No results found. Build index first with 'cliv rag index'[/yellow]"
                )
        elif args.rag_command == "stats":
            stats = indexer.get_stats()
            if getattr(args, "json", False):
                from cliv.output import print_json

                print_json(stats)
                return
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
    from cliv.output import print_json

    try:
        store = TimelineStore()
        limit = getattr(args, "limit", 25)
        events = store.recent(limit=limit)
        if getattr(args, "json", False):
            print_json(
                [
                    {
                        "id": getattr(e, "id", None),
                        "event_type": getattr(e, "event_type", ""),
                        "summary": getattr(e, "summary", ""),
                        "timestamp": getattr(e, "timestamp", ""),
                        "metadata": getattr(e, "metadata", {}),
                    }
                    for e in events
                ]
            )
            return
        console.print(render_timeline(events), markup=False)
    except Exception as exc:
        console.print(f"[bold red]Timeline failed:[/bold red] {exc}")
        sys.exit(1)


def cmd_snapshots(args):
    """Safety snapshots."""
    from cliv.snapshots import SnapshotStore, render_snapshot, render_snapshots
    from cliv.timeline import TimelineStore
    from cliv.output import print_json

    command = getattr(args, "snapshots_command", "list")

    if command == "list":
        try:
            store = SnapshotStore()
            limit = getattr(args, "limit", 25)
            snapshots = store.recent(limit=limit)
            if getattr(args, "json", False):
                print_json(
                    [
                        {
                            "id": getattr(s, "id", ""),
                            "timestamp": getattr(
                                s, "timestamp", getattr(s, "created_at", "")
                            ),
                            "files": [
                                getattr(f, "path", str(f))
                                for f in getattr(s, "files", [])
                            ],
                        }
                        for s in snapshots
                    ]
                )
                return
            console.print(render_snapshots(snapshots), markup=False)
        except Exception as exc:
            console.print(f"[bold red]Snapshots failed:[/bold red] {exc}")
            sys.exit(1)

    elif command == "undo":
        try:
            store = SnapshotStore()
            selector = str(getattr(args, "snapshot_id", "last") or "last")
            if selector in {"last", "latest"}:
                snapshot = store.latest_applied()
            else:
                snapshot = store.get(selector)
            if snapshot is None:
                console.print(
                    "[bold yellow]No applied snapshots to undo.[/bold yellow]"
                )
                return
            console.print(render_snapshot(snapshot), markup=False)
            if snapshot.status != "applied":
                console.print(
                    f"[bold yellow]Snapshot is {snapshot.status}; nothing to undo.[/bold yellow]"
                )
                return
            if not getattr(args, "yes", False):
                try:
                    console.print("\n[bold yellow]Restore these files? \\[y/N]: [/bold yellow]", end="")
                    sys.stdout.flush()
                    answer = input().strip().lower()
                except (EOFError, KeyboardInterrupt):
                    console.print("\n[bold yellow]Undo cancelled.[/bold yellow]")
                    return
                if answer not in {"y", "yes"}:
                    console.print("[bold yellow]Undo cancelled.[/bold yellow]")
                    return
            result = store.rollback(snapshot.id, force=getattr(args, "force", False))
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
    from cliv.session_io import (
        export_session_json,
        export_session_markdown,
        import_session,
    )

    store = HistoryStore()
    if args.session_command == "list":
        sessions = store.list_sessions(limit=getattr(args, "limit", 10))
        if getattr(args, "json", False):
            from cliv.output import print_json

            print_json(sessions)
            return
        if not sessions:
            console.print("[dim]No sessions found.[/dim]")
            return
        for s in sessions:
            console.print(
                f"  {s['id']}  {s['created']}  {s['turns']} turns  ({s['model']})"
            )
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
    """List available online and offline models and their capabilities."""
    from rich.table import Table
    from cliv.llm.discovery import check_model_health, RECOMMENDED_OFFLINE_MODELS
    from cliv.output import print_json

    health = check_model_health()
    if getattr(args, "json", False):
        print_json(health)
        return

    groq_info = health["groq"]
    ollama_info = health["ollama"]

    tbl = Table(title="Available AI Models", box=None, padding=(0, 2))
    tbl.add_column("Provider", style="bold cyan", no_wrap=True)
    tbl.add_column("Model Name / ID", style="bold", no_wrap=True)
    tbl.add_column("Tool Support", no_wrap=True)
    tbl.add_column("Size / Spec", style="dim")
    tbl.add_column("Status")

    has_any = False

    # 1. Ollama local models
    if ollama_info["models"]:
        for m in ollama_info["models"]:
            has_any = True
            tools_str = (
                "[green]✓ Tools[/green]"
                if m.get("supports_tools")
                else "[yellow]Intent only[/yellow]"
            )
            spec = f"{m.get('parameter_size') or '?'} ({m.get('size_gb', 0)} GB)"
            status = "[bold green]✓ Downloaded[/bold green]"
            tbl.add_row("Ollama (Local)", m["name"], tools_str, spec, status)

    # 2. Groq cloud models
    if groq_info["models"]:
        for m in groq_info["models"][:25]:
            has_any = True
            if m.get("usable_for_chat"):
                tools_str = "[green]✓ Tools[/green]"
            else:
                tools_str = f"[dim]No ({m.get('category', 'specialized')})[/dim]"
            status = (
                "[green]✓ Available[/green]"
                if m.get("active")
                else "[dim]Inactive[/dim]"
            )
            desc = m.get("desc") or m.get("owned_by", "groq")
            tbl.add_row("Groq (Cloud)", m["id"], tools_str, desc, status)

    console.print()
    if has_any:
        console.print(tbl)
    else:
        console.print("[yellow]No models currently ready to use.[/yellow]")
    console.print()

    # Diagnostics & recommendations
    if not ollama_info["running"]:
        console.print(
            "[bold yellow]Ollama (Local):[/bold yellow] Daemon not reachable.\n"
            "  Start Ollama: [bold]ollama serve[/bold]\n"
            "  Recommended models: [cyan]ollama pull llama3.1:8b[/cyan] or [cyan]ollama pull qwen2.5-coder:7b[/cyan]\n"
        )
    elif not ollama_info["models"]:
        console.print(
            "[bold yellow]Ollama is running, but no models downloaded.[/bold yellow]\n"
            "  Recommended: [bold cyan]ollama pull llama3.1:8b[/bold cyan] (tool calling supported)\n"
            "  Lightweight: [bold cyan]ollama pull qwen2.5-coder:3b[/bold cyan]\n"
        )

    if not groq_info["has_key"]:
        console.print(
            "[dim]Groq (Cloud): No API key configured. Run 'cliv config api-key' to enable fast cloud models.[/dim]\n"
        )
    elif groq_info.get("error"):
        console.print(f"[dim]Groq status: {groq_info['error']}[/dim]\n")


def cmd_cost(args):
    """Cost & token usage for current project."""
    from cliv.timeline import TimelineStore
    from cliv.security import project_root
    from cliv.output import print_json

    try:
        store = TimelineStore()
        cost_data = store.get_project_cost(project=project_root())
        if getattr(args, "json", False):
            print_json(cost_data)
            return

        console.print(
            f"\n[bold cyan]Project Cost Summary:[/bold cyan] {cost_data['project_root']}"
        )
        console.print(f"  Total Sessions:   {cost_data['total_sessions']}")
        console.print(f"  Input Tokens:     {cost_data['input_tokens']:,}")
        console.print(f"  Output Tokens:    {cost_data['output_tokens']:,}")
        console.print(f"  Total Tokens:     {cost_data['total_tokens']:,}")
        console.print(
            f"  Cumulative Spend: [bold green]${cost_data['total_cost_usd']:.6f}[/bold green]"
        )
        if cost_data.get("models"):
            console.print("\n[bold]Spend by Model:[/bold]")
            for m in cost_data["models"]:
                console.print(
                    f"  - {m['model']}: {m['tokens']:,} tokens (${m['cost']:.6f})"
                )
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
        if getattr(args, "json", False):
            from cliv.output import print_json

            print_json(
                [
                    {
                        "name": t.name,
                        "description": t.description,
                        "risk_level": getattr(t, "risk_level", "high"),
                        "permissions": sorted(list(getattr(t, "permissions", []))),
                        "plugin_source": getattr(t, "plugin_source", "builtin"),
                    }
                    for t in sorted(tools.values(), key=lambda x: x.name)
                ]
            )
            return
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
        console.print(
            f"\n[dim]{len(tools)} tools available. Run 'cliv tools new <name>' to create a plugin.[/dim]\n"
        )

    elif tools_command == "new":
        name = getattr(args, "name", "")
        if not name:
            console.print("[bold red]Error:[/bold red] Tool name is required.")
            sys.exit(1)
        is_global = getattr(args, "is_global", False)
        target_dir = (
            PLUGINS_DIR if is_global else (project_root() / ".cliv" / "plugins")
        )
        path = create_plugin_template(name, target_dir)
        console.print(
            f"\n[bold green]Created tool plugin scaffold:[/bold green] {path}"
        )
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
        console.print(
            f"  Permissions: {','.join(sorted(getattr(tool, 'permissions', []))) or 'none'}"
        )
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
        if getattr(args, "json", False):
            from cliv.output import print_json

            print_json([e.to_dict() for e in entries])
            return
        cliv_md = store.get_cliv_md_content()

        if not entries and not cliv_md:
            console.print(
                "[dim]No remembered preferences or CLIV.md guidelines found.[/dim]"
            )
            console.print(
                "[dim]Run 'cliv memory add <text>' to save a convention.[/dim]"
            )
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
            scope_styled = (
                "[cyan]global[/cyan]"
                if e.scope == "global"
                else "[yellow]project[/yellow]"
            )
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
            console.print(
                f"\n[bold]CLIV.md Guidelines:[/bold] ({len(cliv_md)} chars loaded from project root)"
            )
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
        console.print(
            f"\n[bold green]Saved {scope} memory (id: {entry.id}):[/bold green]"
        )
        console.print(f'  "{entry.content}"\n')

    elif memory_command in {"remove", "rm", "delete"}:
        target = getattr(args, "target", "")
        if not target:
            console.print(
                "[bold red]Error:[/bold red] Target ID or keyword is required."
            )
            sys.exit(1)
        removed = store.remove(target)
        if removed:
            console.print(
                f"\n[bold green]Removed memory matching:[/bold green] '{target}'\n"
            )
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


def cmd_workspace(args):
    """Manage multi-repo workspaces."""
    from rich.table import Table

    registry = WorkspaceRegistry()
    subcmd = getattr(args, "workspace_command", None) or "list"

    if subcmd == "list":
        entries = registry.list()
        if getattr(args, "json", False):
            from cliv.output import print_json

            print_json([e.to_dict() for e in entries])
            return
        if not entries:
            console.print(
                "\n[dim]No workspaces registered.[/dim]\n"
                "Add one with: [bold]cliv workspace add <path>[/bold]\n"
            )
            return
        active = registry.active_entry()
        tbl = Table(title="Registered Workspaces", box=None, padding=(0, 2))
        tbl.add_column("Status", style="green", no_wrap=True)
        tbl.add_column("Name", style="bold cyan", no_wrap=True)
        tbl.add_column("Path", style="dim")
        tbl.add_column("Description")
        for e in entries:
            marker = " [ACTIVE]" if (active and e.name == active.name) else ""
            status = "✓" if e.exists() else "[red]✗ MISSING[/red]"
            tbl.add_row(status, e.name + marker, e.path, e.description or "")
        console.print()
        console.print(tbl)
        console.print()

    elif subcmd == "add":
        path = getattr(args, "path", "").strip()
        name = getattr(args, "name", None)
        desc = getattr(args, "desc", "") or ""
        tags = getattr(args, "tag", []) or []
        try:
            entry, is_new = registry.add(path, name=name, description=desc, tags=tags)
            verb = "Registered" if is_new else "Updated"
            console.print(
                f"\n[bold green]{verb} workspace:[/bold green] "
                f"[bold cyan]{entry.name}[/bold cyan] → {entry.path}\n"
            )
        except ValueError as exc:
            console.print(f"[bold red]Error:[/bold red] {exc}")

    elif subcmd in ("remove", "rm"):
        name = getattr(args, "name", "").strip()
        removed = registry.remove(name)
        if removed:
            console.print(
                f"\n[bold green]Removed workspace:[/bold green] [cyan]{name}[/cyan]\n"
            )
        else:
            console.print(f"[yellow]No workspace named '{name}'.[/yellow]")

    elif subcmd == "info":
        name = getattr(args, "name", "").strip()
        entry = registry.get(name)
        if getattr(args, "json", False):
            from cliv.output import print_json

            print_json(entry.to_dict() if entry else None)
            return
        if entry is None:
            console.print(f"[yellow]No workspace named '{name}'.[/yellow]")
            return
        status = "✓ exists" if entry.exists() else "✗ MISSING"
        console.print(f"\n[bold cyan]{entry.name}[/bold cyan]")
        console.print(f"  Path:        {entry.path} ({status})")
        console.print(f"  Description: {entry.description or '(none)'}")
        console.print(f"  Tags:        {', '.join(entry.tags) or '(none)'}")
        console.print(f"  Added:       {entry.added_at}")
        console.print(f"  Last active: {entry.last_active}")
        console.print()

    else:
        console.print(f"[yellow]Unknown workspace subcommand: '{subcmd}'.[/yellow]")


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
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output structured JSON instead of human-readable text",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress banner, notices, spinners, and diagnostics (scripting mode)",
    )

    # Legacy flags (for backward compatibility)
    parser.add_argument("--project-dna", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--timeline", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--snapshots", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--undo", nargs="?", const="last", metavar="SNAPSHOT_ID", help=argparse.SUPPRESS
    )
    parser.add_argument("--force", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--graph-explorer", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--export-mermaid", metavar="PATH", help=argparse.SUPPRESS)
    parser.add_argument("--export-graphviz", metavar="PATH", help=argparse.SUPPRESS)
    parser.add_argument("--rag-index", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--rag-query", metavar="QUERY", help=argparse.SUPPRESS)
    parser.add_argument("--rag-stats", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--rag-force-rebuild", action="store_true", help=argparse.SUPPRESS
    )

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # run (default)
    p_run = subparsers.add_parser(
        "run", help="Interactive chat or single-shot (default)", add_help=False
    )
    p_run.add_argument("-h", "--help", action="store_true")
    p_run.add_argument(
        "--json", action="store_true", help="Output machine-readable JSON"
    )
    p_run.add_argument(
        "-q", "--quiet", action="store_true", help="Minimal quiet output"
    )
    p_run.add_argument("prompt", nargs="*", help="Prompt for single-shot mode")
    p_run.set_defaults(func=cmd_run)

    # dna
    p_dna = subparsers.add_parser(
        "dna", help="Project DNA architecture report", add_help=False
    )
    p_dna.add_argument("-h", "--help", action="store_true")
    p_dna.add_argument(
        "--json", action="store_true", help="Output machine-readable JSON"
    )
    p_dna.set_defaults(func=cmd_dna)

    # onboard
    p_onboard = subparsers.add_parser(
        "onboard", help="Generate developer and agent onboarding report", add_help=False
    )
    p_onboard.add_argument("-h", "--help", action="store_true")
    p_onboard.add_argument(
        "path", nargs="?", default=".", help="Project path (defaults to current dir)"
    )
    p_onboard.add_argument(
        "-o", "--output", metavar="PATH", help="Save report to specified markdown file"
    )
    p_onboard.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="Save report to ONBOARDING.md in project root",
    )
    p_onboard.add_argument(
        "--json", action="store_true", help="Output machine-readable JSON dataset"
    )
    p_onboard.set_defaults(func=cmd_onboard)

    # review / check
    p_review = subparsers.add_parser(
        "review",
        aliases=["check"],
        help="Automated code review & CI check",
        add_help=False,
    )
    p_review.add_argument("-h", "--help", action="store_true")
    p_review.add_argument("path", nargs="?", default=".", help="Project path to review")
    p_review.add_argument(
        "-b",
        "--base",
        metavar="REF",
        help="Git base ref to compare against (e.g. main, origin/main, HEAD~1)",
    )
    p_review.add_argument(
        "--format",
        choices=["standard", "github", "markdown"],
        default="standard",
        help="Markdown formatting style",
    )
    p_review.add_argument(
        "--json", action="store_true", help="Output machine-readable JSON dataset"
    )
    p_review.add_argument(
        "--run-tests", action="store_true", help="Execute test suite as part of review"
    )
    p_review.add_argument(
        "--no-fail", action="store_true", help="Exit with 0 even if errors are found"
    )
    p_review.add_argument(
        "--init-ci",
        action="store_true",
        help="Scaffold GitHub Actions CI review workflow",
    )
    p_review.set_defaults(func=cmd_review)

    # graph
    p_graph = subparsers.add_parser(
        "graph", help="Visual codebase explorer", add_help=False
    )
    p_graph.add_argument("-h", "--help", action="store_true")
    p_graph.add_argument(
        "--explore", action="store_true", help="Interactive mode (default)"
    )
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
    p_rag_query.add_argument(
        "--json", action="store_true", help="Output machine-readable JSON"
    )
    p_rag_stats = rag_sub.add_parser("stats", help="Index statistics", add_help=False)
    p_rag_stats.add_argument("-h", "--help", action="store_true")
    p_rag_stats.add_argument(
        "--json", action="store_true", help="Output machine-readable JSON"
    )
    p_rag.set_defaults(func=cmd_rag)

    # timeline
    p_timeline = subparsers.add_parser(
        "timeline", help="Development timeline", add_help=False
    )
    p_timeline.add_argument("-h", "--help", action="store_true")
    p_timeline.add_argument("-n", "--limit", type=int, default=25, help="Max events")
    p_timeline.add_argument(
        "--json", action="store_true", help="Output machine-readable JSON"
    )
    p_timeline.set_defaults(func=cmd_timeline)

    # snapshots
    p_snap = subparsers.add_parser("snapshots", help="Safety snapshots", add_help=False)
    p_snap.add_argument("-h", "--help", action="store_true")
    snap_sub = p_snap.add_subparsers(dest="snapshots_command", required=False)
    p_snap_list = snap_sub.add_parser("list", help="List snapshots", add_help=False)
    p_snap_list.add_argument("-h", "--help", action="store_true")
    p_snap_list.add_argument("-n", "--limit", type=int, default=25)
    p_snap_list.add_argument(
        "--json", action="store_true", help="Output machine-readable JSON"
    )
    p_snap_undo = snap_sub.add_parser("undo", help="Restore snapshot", add_help=False)
    p_snap_undo.add_argument("-h", "--help", action="store_true")
    p_snap_undo.add_argument("snapshot_id", nargs="?", default="last")
    p_snap_undo.add_argument("--force", action="store_true", help="Overwrite conflicts")
    p_snap_undo.add_argument(
        "--yes", "-y", action="store_true", help="Skip confirmation"
    )
    p_snap.set_defaults(func=cmd_snapshots)

    # git
    p_git = subparsers.add_parser("git", help="Git integration", add_help=False)
    p_git.add_argument("-h", "--help", action="store_true")
    git_sub = p_git.add_subparsers(dest="git_command", required=True)
    p_git_status = git_sub.add_parser("status", help="Show status", add_help=False)
    p_git_status.add_argument("-h", "--help", action="store_true")
    p_git_branch = git_sub.add_parser(
        "branch", help="Show or create branch", add_help=False
    )
    p_git_branch.add_argument("-h", "--help", action="store_true")
    p_git_branch.add_argument("name", nargs="?", help="New branch name")
    p_git_commit = git_sub.add_parser("commit", help="Commit changes", add_help=False)
    p_git_commit.add_argument("-h", "--help", action="store_true")
    p_git_commit.add_argument("message", help="Commit message")
    p_git_commit.add_argument("files", nargs="*", help="Files to stage")
    p_git.set_defaults(func=cmd_git)

    # config
    p_config = subparsers.add_parser(
        "config", help="Configuration management", add_help=False
    )
    p_config.add_argument("-h", "--help", action="store_true")
    config_sub = p_config.add_subparsers(dest="config_command", required=True)
    p_config_get = config_sub.add_parser("get", help="Get config value", add_help=False)
    p_config_get.add_argument("-h", "--help", action="store_true")
    p_config_get.add_argument("key", help="Config key")
    p_config_set = config_sub.add_parser("set", help="Set config value", add_help=False)
    p_config_set.add_argument("-h", "--help", action="store_true")
    p_config_set.add_argument("key", help="Config key")
    p_config_set.add_argument("value", help="Config value")
    p_config_list = config_sub.add_parser(
        "list", help="List config keys", add_help=False
    )
    p_config_list.add_argument("-h", "--help", action="store_true")
    p_config_apikey = config_sub.add_parser(
        "api-key", help="Set API key interactively", add_help=False
    )
    p_config_apikey.add_argument("-h", "--help", action="store_true")
    p_config_apikey.add_argument("value", nargs="?", help="API key (optional)")
    p_config.set_defaults(func=cmd_config)

    # session
    p_session = subparsers.add_parser(
        "session", help="Session management", add_help=False
    )
    p_session.add_argument("-h", "--help", action="store_true")
    session_sub = p_session.add_subparsers(dest="session_command", required=True)
    p_session_list = session_sub.add_parser(
        "list", help="List sessions", add_help=False
    )
    p_session_list.add_argument("-h", "--help", action="store_true")
    p_session_list.add_argument("-n", "--limit", type=int, default=10)
    p_session_list.add_argument(
        "--json", action="store_true", help="Output machine-readable JSON"
    )
    p_session_clear = session_sub.add_parser(
        "clear", help="Clear history", add_help=False
    )
    p_session_clear.add_argument("-h", "--help", action="store_true")
    p_session_export = session_sub.add_parser(
        "export", help="Export session to file", add_help=False
    )
    p_session_export.add_argument("-h", "--help", action="store_true")
    p_session_export.add_argument("session_id", help="Session ID or archive file")
    p_session_export.add_argument("-o", "--output", help="Output file path")
    p_session_export.add_argument(
        "--format", choices=["md", "json"], default="md", help="Export format"
    )
    p_session_import = session_sub.add_parser(
        "import", help="Import session from file", add_help=False
    )
    p_session_import.add_argument("-h", "--help", action="store_true")
    p_session_import.add_argument(
        "path", help="Path to session JSON file or session ID"
    )
    p_session.set_defaults(func=cmd_session)

    # models
    p_models = subparsers.add_parser(
        "models", help="List available models", add_help=False
    )
    p_models.add_argument("-h", "--help", action="store_true")
    p_models.add_argument(
        "--json", action="store_true", help="Output machine-readable JSON"
    )
    p_models.set_defaults(func=cmd_models)

    # cost
    p_cost = subparsers.add_parser(
        "cost",
        help="Show token usage and cost metrics for this project",
        add_help=False,
    )
    p_cost.add_argument("-h", "--help", action="store_true")
    p_cost.add_argument(
        "--json", action="store_true", help="Output machine-readable JSON"
    )
    p_cost.set_defaults(func=cmd_cost)

    # tools / plugins
    p_tools = subparsers.add_parser(
        "tools", aliases=["plugins"], help="Custom tool plugins", add_help=False
    )
    p_tools.add_argument("-h", "--help", action="store_true")
    tools_sub = p_tools.add_subparsers(dest="tools_command", required=False)
    p_tools_list = tools_sub.add_parser(
        "list", help="List loaded tools", add_help=False
    )
    p_tools_list.add_argument("-h", "--help", action="store_true")
    p_tools_list.add_argument(
        "--json", action="store_true", help="Output machine-readable JSON"
    )
    p_tools_new = tools_sub.add_parser(
        "new", help="Scaffold a new tool plugin", add_help=False
    )
    p_tools_new.add_argument("-h", "--help", action="store_true")
    p_tools_new.add_argument("name", help="Tool name")
    p_tools_new.add_argument(
        "--global",
        dest="is_global",
        action="store_true",
        help="Scaffold in user-global plugins directory",
    )
    p_tools_info = tools_sub.add_parser(
        "info", help="Inspect a tool's schema and details", add_help=False
    )
    p_tools_info.add_argument("-h", "--help", action="store_true")
    p_tools_info.add_argument("name", help="Tool name")
    p_tools.set_defaults(func=cmd_tools)

    # memory
    p_memory = subparsers.add_parser(
        "memory", help="Agent long-term memory & conventions", add_help=False
    )
    p_memory.add_argument("-h", "--help", action="store_true")
    mem_sub = p_memory.add_subparsers(dest="memory_command", required=False)
    p_mem_list = mem_sub.add_parser(
        "list", help="List remembered conventions", add_help=False
    )
    p_mem_list.add_argument("-h", "--help", action="store_true")
    p_mem_list.add_argument(
        "--scope", choices=["global", "project"], help="Filter by scope"
    )
    p_mem_list.add_argument(
        "--json", action="store_true", help="Output machine-readable JSON"
    )
    p_mem_add = mem_sub.add_parser(
        "add", help="Add a new memory or convention", add_help=False
    )
    p_mem_add.add_argument("-h", "--help", action="store_true")
    p_mem_add.add_argument("content", help="The preference or convention to remember")
    p_mem_add.add_argument(
        "--global",
        dest="is_global",
        action="store_true",
        help="Store in global memory across projects",
    )
    p_mem_add.add_argument(
        "--category",
        default="general",
        help="Category (preference, convention, architecture, tooling)",
    )
    p_mem_rm = mem_sub.add_parser(
        "remove",
        aliases=["rm", "delete"],
        help="Remove a remembered item",
        add_help=False,
    )
    p_mem_rm.add_argument("-h", "--help", action="store_true")
    p_mem_rm.add_argument("target", help="Memory ID or text substring to delete")
    p_mem_clear = mem_sub.add_parser(
        "clear", help="Clear remembered items", add_help=False
    )
    p_mem_clear.add_argument("-h", "--help", action="store_true")
    p_mem_clear.add_argument(
        "--scope", choices=["global", "project"], help="Scope to clear"
    )
    p_memory.set_defaults(func=cmd_memory)

    # upgrade
    p_upgrade = subparsers.add_parser("upgrade", help="Upgrade cliv", add_help=False)
    p_upgrade.add_argument("-h", "--help", action="store_true")
    p_upgrade.set_defaults(func=cmd_upgrade)

    # workspace
    p_ws = subparsers.add_parser(
        "workspace",
        aliases=["ws"],
        help="Multi-repo workspace management",
        add_help=False,
    )
    p_ws.add_argument("-h", "--help", action="store_true")
    ws_sub = p_ws.add_subparsers(dest="workspace_command", required=False)
    p_ws_list = ws_sub.add_parser(
        "list", help="List registered workspaces", add_help=False
    )
    p_ws_list.add_argument("-h", "--help", action="store_true")
    p_ws_list.add_argument(
        "--json", action="store_true", help="Output machine-readable JSON"
    )
    p_ws_add = ws_sub.add_parser("add", help="Register a new workspace", add_help=False)
    p_ws_add.add_argument("-h", "--help", action="store_true")
    p_ws_add.add_argument("path", help="Path to the workspace root directory")
    p_ws_add.add_argument(
        "name",
        nargs="?",
        default=None,
        help="Optional workspace name (default: directory name)",
    )
    p_ws_add.add_argument(
        "--desc", metavar="TEXT", default="", help="Short description"
    )
    p_ws_add.add_argument(
        "--tag", metavar="TAG", action="append", default=[], help="Tag (repeatable)"
    )
    p_ws_rm = ws_sub.add_parser(
        "remove", aliases=["rm"], help="Unregister a workspace", add_help=False
    )
    p_ws_rm.add_argument("-h", "--help", action="store_true")
    p_ws_rm.add_argument("name", help="Workspace name to remove")
    p_ws_info = ws_sub.add_parser(
        "info", help="Show details about a workspace", add_help=False
    )
    p_ws_info.add_argument("-h", "--help", action="store_true")
    p_ws_info.add_argument(
        "--json", action="store_true", help="Output machine-readable JSON"
    )
    p_ws_info.add_argument("name", help="Workspace name")
    p_ws.set_defaults(func=cmd_workspace, workspace_command="list")

    # Check sys.argv directly for legacy flags BEFORE argparse
    argv = sys.argv[1:]

    # Check for legacy flags that should bypass subcommand parsing
    # These are the OLD flat flags that were replaced by subcommands
    legacy_flags = {
        "--project-dna",
        "--timeline",
        "--snapshots",
        "--undo",
        "--graph-explorer",
        "--export-mermaid",
        "--export-graphviz",
        "--rag-index",
        "--rag-query",
        "--rag-stats",
        "--rag-force-rebuild",
        "--force",
    }
    has_legacy = any(
        arg in legacy_flags
        or arg.startswith(tuple(f"{f}=" for f in legacy_flags if "=" not in f))
        for arg in argv
    )

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
        print(
            "  workspace    Multi-repo workspace management (list, add, remove, info)"
        )
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

        if first_non_flag_idx is not None and legacy_argv[first_non_flag_idx] in (
            "undo",
            "snapshots",
        ):
            # Map legacy positional to new subcommand structure
            if legacy_argv[first_non_flag_idx] == "undo":
                legacy_argv[first_non_flag_idx] = "snapshots"
                legacy_argv.insert(first_non_flag_idx + 1, "undo")
            elif legacy_argv[first_non_flag_idx] == "snapshots":
                legacy_argv[first_non_flag_idx] = "snapshots"
                if first_non_flag_idx + 1 >= len(legacy_argv) or legacy_argv[
                    first_non_flag_idx + 1
                ].startswith("-"):
                    legacy_argv.insert(first_non_flag_idx + 1, "list")

        # Parse with legacy flags included
        args = parser.parse_args(legacy_argv)

        # Validate conflicts
        if args.yes and args.dry_run:
            console.print(
                "[bold red]Error: --yes and --dry-run are mutually exclusive.[/bold red]"
            )
            sys.exit(1)

        if args.project_dna:
            cmd_dna(args)
            return
        if args.timeline:
            cmd_timeline(args)
            return
        if args.snapshots:
            args.snapshots_command = "list"
            args.limit = getattr(args, "limit", 25)
            cmd_snapshots(args)
            return
        if args.undo is not None:
            args.snapshots_command = "undo"
            args.snapshot_id = args.undo
            args.force = getattr(args, "force", False)
            args.yes = getattr(args, "yes", False)
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
        if getattr(args, "command", None) == "snapshots":
            if getattr(args, "snapshots_command", None) == "undo":
                args.snapshot_id = getattr(args, "snapshot_id", "last")
                args.force = getattr(args, "force", False)
                args.yes = getattr(args, "yes", False)
                cmd_snapshots(args)
                return
            elif getattr(args, "snapshots_command", None) == "list":
                args.limit = getattr(args, "limit", 25)
                cmd_snapshots(args)
                return
            # Default to list if no subcommand
            args.snapshots_command = "list"
            args.limit = getattr(args, "limit", 25)
            cmd_snapshots(args)
            return

    # No legacy flags - use subcommand parsing
    # First, parse global flags using parse_known_args
    global_parser = argparse.ArgumentParser(add_help=False)
    global_parser.add_argument("--api-key", help=argparse.SUPPRESS)
    global_parser.add_argument("--offline", action="store_true", help=argparse.SUPPRESS)
    global_parser.add_argument(
        "--verbose", "-v", action="store_true", help=argparse.SUPPRESS
    )
    global_parser.add_argument(
        "--yes", "-y", action="store_true", help=argparse.SUPPRESS
    )
    global_parser.add_argument(
        "--dry-run", "-n", action="store_true", help=argparse.SUPPRESS
    )
    global_parser.add_argument(
        "--legacy-offline", action="store_true", help=argparse.SUPPRESS
    )
    global_parser.add_argument("--permissions", metavar="LIST", help=argparse.SUPPRESS)
    global_parser.add_argument("--branch", metavar="NAME", help=argparse.SUPPRESS)
    global_parser.add_argument(
        "--auto-commit", action="store_true", help=argparse.SUPPRESS
    )
    global_parser.add_argument("--resume", metavar="SESSION", help=argparse.SUPPRESS)
    global_parser.add_argument(
        "-h", "--help", action="store_true", help=argparse.SUPPRESS
    )

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
        console.print(
            "[bold red]Error: --yes and --dry-run are mutually exclusive.[/bold red]"
        )
        sys.exit(1)

    # Call handler
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
