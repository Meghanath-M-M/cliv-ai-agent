import sys
import argparse
from getpass import getpass
from ai_cli.agent import AIAgent
from ai_cli.config import get_api_key, save_api_key

# --- PHASE 2: Rich UI Imports ---
from rich.console import Console
from rich.markdown import Markdown

console = Console()


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

    agent = AIAgent(api_key=api_key)

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

            # --- PHASE 2: Render the response as formatted Markdown ---
            console.print(Markdown(response))
            print()

        except KeyboardInterrupt:
            console.print("\n\n[bold yellow]Goodbye![/bold yellow]")
            break
        except Exception as e:
            console.print(f"\n[bold red]Error: {str(e)}[/bold red]\n")


if __name__ == "__main__":
    main()
