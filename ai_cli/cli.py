import sys
import argparse
from getpass import getpass
from ai_cli.agent import AIAgent
from ai_cli.config import get_api_key, save_api_key

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
        print("No Groq API key found.")
        print("For Lightning Fast execution, get a free key: https://console.groq.com/keys")
        print("Or press [Enter] to run entirely OFFLINE using your local hardware (requires Ollama).")
        
        input_key = getpass("Enter API Key (or press Enter for Offline Mode): ").strip()
        
        if input_key:
            api_key = input_key
            save = input("Would you like to save this key for future use? (y/n): ").strip().lower()
            if save == 'y':
                save_api_key(api_key)
                print("API key saved.")
        else:
            print("\nStarting in OFFLINE mode. (Ensure Ollama is running)")
            api_key = None

    agent = AIAgent(api_key=api_key)

    print("\nAI Code Assistant")
    print("================")
    print(f"Mode: {agent.mode.upper()}")
    print("A conversational AI agent that can read, list, and edit files.")
    print("Type 'exit' or 'quit' to end the conversation.\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            if not user_input:
                continue

            print("\nAssistant: ", end="", flush=True)
            response = agent.chat(user_input)
            print(response)
            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print()

if __name__ == "__main__":
    main()