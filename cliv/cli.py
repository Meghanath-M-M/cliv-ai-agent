"""
Terminal UI for cliv.
Handles startup flow, routing, real output, and session management.
"""

import sys
import argparse
import socket
from getpass import getpass
from cliv.agent import AIAgent
from cliv.config import get_api_key, save_api_key

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

    # --- Telemetry Display ---
    if agent.mode == "online" and agent.stats.total_tokens > 0:
        console.print(
            f"\n[dim italic]"
            f"Session: {agent.stats.total_tokens:,} tokens "
            f"(in: {agent.stats.input_tokens:,}, out: {agent.stats.output_tokens:,}) "
            f"· ~${agent.stats.cost_usd:.6f}"
            f"[/dim italic]"
        )
    print()


def main():
    parser = argparse.ArgumentParser(
        description="> cliv_ : Terminal-native intelligence for your local codebase."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Optional prompt to execute a single command directly.",
    )
    parser.add_argument(
        "--api-key", help="Groq API key (overrides saved config or env var)"
    )
    parser.add_argument(
        "--offline", action="store_true", help="Force offline mode (use Ollama)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging to console"
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Auto-approve all tool operations without prompting",
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Preview tool operations without applying changes",
    )
    args = parser.parse_args()

    # Validate conflicting flags
    if args.yes and args.dry_run:
        console.print(
            "[bold red]Error: --yes and --dry-run are mutually exclusive.[/bold red]"
        )
        sys.exit(1)

    api_key = args.api_key or get_api_key()
    if api_key == "" or args.offline:
        api_key = None

    if not api_key and not args.prompt:
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

    # Initialize the agent with flags
    agent = AIAgent(
        api_key=api_key,
        verbose=args.verbose,
        auto_approve=args.yes,
        dry_run=args.dry_run,
    )

    # Print mode banner
    mode_color = "green" if agent.mode == "online" else "yellow"

    logo = """[bold cyan]
     ██████╗ ██╗     ██╗██╗   ██╗
    ██╔════╝ ██║     ██║██║   ██║
    ██║      ██║     ██║██║   ██║
    ██║      ██║     ██║╚██╗ ██╔╝
    ╚██████╗ ███████╗██║ ╚████╔╝
     ╚═════╝ ╚══════╝╚═╝  ╚═══╝ 
    [/bold cyan]"""
    console.print(logo)

    console.print(f"Mode: [bold {mode_color}]{agent.mode.upper()}[/bold {mode_color}]")
    if args.yes:
        console.print("[dim]Auto-approve: ENABLED (use with caution)[/dim]")
    if args.dry_run:
        console.print("[dim]Dry-run: ENABLED (no changes will be applied)[/dim]")
    console.print("Terminal-native intelligence for your local codebase.")
    console.print("Type 'exit', 'quit', or 'clear' to manage the session.\n")

    # ---------------------------------------------------------
    # ROUTING: Single-shot Command vs. Interactive Loop
    # ---------------------------------------------------------

    # 1. Single-shot execution
    if args.prompt:
        console.print(f"[dim]Executing command:[/dim] {args.prompt}")
        process_prompt(agent, args.prompt)
        return

    # 2. Interactive REPL Mode
    while True:
        try:
            user_input = input("> ").strip()

            if user_input.lower() in ["exit", "quit"]:
                console.print("[bold yellow]Session terminated.[/bold yellow]")
                break

            if user_input.lower() == "clear":
                agent.clear_history()
                console.print(
                    "[bold green]Memory cleared! Starting fresh.[/bold green]"
                )
                continue

            if not user_input:
                continue

            process_prompt(agent, user_input)

        except KeyboardInterrupt:
            console.print("\n\n[bold yellow]Session terminated by user.[/bold yellow]")
            break
        except Exception as e:
            console.print(f"\n[bold red]Error: {str(e)}[/bold red]\n")


if __name__ == "__main__":
    main()
