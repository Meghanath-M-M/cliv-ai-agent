import sys
import time
import argparse
import socket
from getpass import getpass
from cliv.agent import AIAgent
from cliv.config import get_api_key, save_api_key
from rich.live import Live

# --- PHASE 2: Rich UI Imports ---
from rich.console import Console
from rich.markdown import Markdown

console = Console()


def check_internet(host="8.8.8.8", port=53, timeout=2):
    """Safely checks if the device is connected to the internet."""
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except OSError:
        return False


def process_prompt(agent, user_input):
    """Handles the streaming and rendering of the agent's response."""
    console.print("\n[bold cyan]cliv:[/bold cyan]")

    response = agent.chat(user_input)

    # --- PHASE 3: Animated Markdown Streaming ---
    with Live(Markdown(""), refresh_per_second=24, console=console) as live:
        accumulated_text = ""
        # Chunking by 3 chars simulates the cadence of actual network tokens
        chunk_size = 3
        for i in range(0, len(response), chunk_size):
            accumulated_text += response[i : i + chunk_size]
            live.update(Markdown(accumulated_text))
            time.sleep(0.01)  # Adjust this to make it type faster or slower

    # --- Telemetry Display ---
    if agent.mode == "online" and agent.session_tokens > 0:
        console.print(
            f"\n[dim italic]Session: {agent.session_tokens:,} tokens · ~${agent.session_cost:.5f}[/dim italic]"
        )
    print()


def main():
    parser = argparse.ArgumentParser(
        description="> cliv_ : Terminal-native intelligence for your local codebase."
    )
    # Added positional argument for single-shot commands (e.g., cliv "fix app.py")
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
    args = parser.parse_args()

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
            # Overriding the key to None forces the AIAgent to use Ollama
            api_key = None

    # Initialize the agent
    agent = AIAgent(api_key=api_key)

    # ---------------------------------------------------------
    # ROUTING LOGIC: Single-shot Command vs. Interactive Loop
    # ---------------------------------------------------------

    # 1. Single-shot execution (e.g., user typed: cliv "run tests")
    if args.prompt:
        console.print(f"[dim]Executing command:[/dim] {args.prompt}")
        process_prompt(agent, args.prompt)
        return

    # 2. Interactive REPL Mode (e.g., user just typed: cliv)
    console.print("\n[bold cyan]> cliv_[/bold cyan]")
    console.print("==================")
    console.print(f"Mode: [bold green]{agent.mode.upper()}[/bold green]")
    console.print("Terminal-native intelligence for your local codebase.")
    console.print("Type 'exit', 'quit', or 'clear' to manage the session.\n")

    while True:
        try:
            # Swapped "You: " for a sleeker terminal prompt
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
