import time
import argparse
import socket
from getpass import getpass
from ai_cli.agent import AIAgent
from ai_cli.config import get_api_key, save_api_key
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


def main():
    parser = argparse.ArgumentParser(
        description="AI Code Assistant - A conversational AI agent with file editing capabilities"
    )
    parser.add_argument(
        "--api-key", help="Groq API key (overrides saved config or env var)"
    )
    args = parser.parse_args()

    api_key = args.api_key or get_api_key()

    if not api_key:
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

    # ... previous logic where you get the api_key ...

    # --- NEW: Automatic Offline Fallback ---
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

    # ... rest of your UI startup code ...

    # UI Polish: Colored Welcome Screen
    console.print("\n[bold blue]🤖 AI Code Assistant[/bold blue]")
    console.print("==================")
    console.print(f"Mode: [bold green]{agent.mode.upper()}[/bold green]")
    console.print(
        "A conversational AI agent that can safely read, list, and edit files."
    )
    console.print("Type 'exit' or 'quit' to end the conversation.\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ["exit", "quit"]:
                console.print("[bold yellow]Goodbye![/bold yellow]")
                break
            if user_input.lower() == "clear":
                agent.clear_history()
                console.print(
                    "[bold green]Memory cleared! Starting fresh.[/bold green]"
                )
                continue

            if not user_input:
                continue

            console.print("\n[bold cyan]Assistant:[/bold cyan]")

            response = agent.chat(user_input)

            # --- PHASE 3: Animated Markdown Streaming ---
            # We use rich.live to constantly redraw the markdown box as text is "typed"
            with Live(Markdown(""), refresh_per_second=24, console=console) as live:
                accumulated_text = ""
                # Chunking by 3 chars simulates the cadence of actual network tokens
                chunk_size = 3
                for i in range(0, len(response), chunk_size):
                    accumulated_text += response[i : i + chunk_size]
                    live.update(Markdown(accumulated_text))
                    time.sleep(0.01)  # Adjust this to make it type faster or slower

            # --- NEW: Display the Telemetry ---
            if agent.mode == "online" and agent.session_tokens > 0:
                console.print(
                    f"\n[dim italic]Session: {agent.session_tokens:,} tokens · ~${agent.session_cost:.5f}[/dim italic]"
                )
            print()

        except KeyboardInterrupt:
            console.print("\n\n[bold yellow]Goodbye![/bold yellow]")
            break
        except Exception as e:
            console.print(f"\n[bold red]Error: {str(e)}[/bold red]\n")


if __name__ == "__main__":
    main()
