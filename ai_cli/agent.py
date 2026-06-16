import os
import logging
import json
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel
from dotenv import load_dotenv

# --- PHASE 2: Rich UI Imports ---
from rich.console import Console
from rich.markdown import Markdown

console = Console()

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    import ollama
except ImportError:
    ollama = None

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler("agent.log")],
)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ==========================================
# PHASE 3: Object-Oriented Tool Architecture
# ==========================================


class BaseTool:
    """Base interface for all AI Tools"""

    name: str
    description: str
    input_schema: Dict[str, Any]

    def execute(self, **kwargs) -> str:
        raise NotImplementedError("Tools must implement the execute method")


class ReadFileTool(BaseTool):
    def __init__(self):
        self.name = "read_file"
        self.description = "Read the contents of a file at the specified path"
        self.input_schema = {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to read",
                }
            },
            "required": ["path"],
        }

    def execute(self, path: str, **kwargs) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            return f"File contents of {path}:\n{content}"
        except FileNotFoundError:
            return f"File not found: {path}"
        except Exception as e:
            return f"Error reading file: {str(e)}"


class ListFilesTool(BaseTool):
    def __init__(self):
        self.name = "list_files"
        self.description = "List all files and directories in the specified path"
        self.input_schema = {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The directory path to list"}
            },
            "required": [],
        }

    def execute(self, path: str = ".", **kwargs) -> str:
        try:
            if not os.path.exists(path):
                return f"Path not found: {path}"
            items = []
            for item in sorted(os.listdir(path)):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    items.append(f"[DIR]  {item}/")
                else:
                    items.append(f"[FILE] {item}")
            if not items:
                return f"Empty directory: {path}"
            return f"Contents of {path}:\n" + "\n".join(items)
        except Exception as e:
            return f"Error listing files: {str(e)}"


class EditFileTool(BaseTool):
    def __init__(self):
        self.name = "edit_file"
        self.description = "Edit a file by replacing old_text with new_text. Creates the file if it doesn't exist."
        self.input_schema = {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to edit",
                },
                "old_text": {
                    "type": "string",
                    "description": "The text to search for and replace (leave empty to create)",
                },
                "new_text": {
                    "type": "string",
                    "description": "The text to replace old_text with",
                },
            },
            "required": ["path", "new_text"],
        }

    def execute(self, path: str, new_text: str, old_text: str = "", **kwargs) -> str:
        # SAFETY RAIL: Require manual confirmation before modifying disk
        console.print(
            f"\n[bold yellow][SYSTEM WARNING][/bold yellow] The AI wants to modify the file: '{path}'"
        )
        confirm = input("Allow this change? [y/N]: ").strip().lower()
        if confirm != "y":
            return f"Operation blocked: User denied permission to edit {path}."

        try:
            if os.path.exists(path) and old_text:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                if old_text not in content:
                    return f"Text not found in file: {old_text}"
                content = content.replace(old_text, new_text)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                return f"Successfully edited {path}"
            else:
                dir_name = os.path.dirname(path)
                if dir_name:
                    os.makedirs(dir_name, exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(new_text)
                return f"Successfully created {path}"
        except Exception as e:
            return f"Error editing file: {str(e)}"


# ==========================================
# MAIN AGENT CLASS
# ==========================================


class AIAgent:
    def __init__(self, api_key: str = None, local_model: str = "qwen2.5-coder:3b"):
        self.api_key = api_key
        self.local_model = local_model

        # --- PHASE 3: Conversation Persistence ---
        self.history_file = Path.home() / ".config" / "ai_cli" / "history.json"
        self.messages: List[Dict[str, Any]] = self._load_history()

        # Initialize OOP Tools
        self.tools: Dict[str, BaseTool] = {
            "read_file": ReadFileTool(),
            "list_files": ListFilesTool(),
            "edit_file": EditFileTool(),
        }

        if self.api_key and Groq:
            self.mode = "online"
            self.client = Groq(api_key=self.api_key)
            self.model_name = "meta-llama/llama-4-scout-17b-16e-instruct"
        else:
            self.mode = "offline"
            if not ollama:
                logging.warning("Ollama library not found. Offline mode may fail.")

    def _load_history(self) -> List[Dict[str, Any]]:
        """Loads previous conversation from disk."""
        if self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Failed to load history: {e}")
        return []

    def _save_history(self):
        """Saves current conversation to disk."""
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.messages, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save history: {e}")

    def clear_history(self):
        """Wipes the conversation history from RAM and Disk."""
        self.messages = []
        if self.history_file.exists():
            try:
                os.remove(self.history_file)
            except Exception as e:
                logging.error(f"Failed to delete history file: {e}")

    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        logging.info(f"Executing tool: {tool_name} with input: {tool_input}")

        tool = self.tools.get(tool_name)
        if not tool:
            return f"Unknown tool: {tool_name}"

        try:
            # OOP Magic: We unpack the JSON args directly into the class method
            return tool.execute(**tool_input)
        except Exception as e:
            logging.error(f"Error executing {tool_name}: {str(e)}")
            return f"Error executing {tool_name}: {str(e)}"

    def chat(self, user_input: str) -> str:
        logging.info(f"User input: {user_input}")
        self.messages.append({"role": "user", "content": user_input})
        self._save_history()

        tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in self.tools.values()
        ]

        system_msg = {
            "role": "system",
            "content": (
                f"You are {self.local_model}, a helpful coding assistant operating in a terminal environment.\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. When the user says a simple greeting like 'hi' or 'hello', you MUST respond ONLY with plain English text.\n"
                "   Example User: 'hi'\n"
                "   Example Assistant: 'Hello! I am your AI code assistant. How can I help you with your files today?'\n"
                "2. NEVER output raw JSON in your text responses.\n"
                "3. To interact with files, use the provided tools silently in the background."
            ),
        }

        while True:
            try:
                normalized_tool_calls = []
                content = None

                # --- PHASE 2: UI/UX Loading Spinner ---
                with console.status("[bold cyan]Agent is thinking...", spinner="dots"):
                    if self.mode == "online":
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            max_tokens=4096,
                            messages=[system_msg] + self.messages,
                            tools=tool_schemas,
                            tool_choice="auto",
                        )
                        message = response.choices[0].message
                        content = message.content

                        if message.tool_calls:
                            for tc in message.tool_calls:
                                normalized_tool_calls.append(
                                    {
                                        "id": tc.id,
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    }
                                )
                    else:
                        response = ollama.chat(
                            model=self.local_model,
                            messages=[system_msg] + self.messages,
                            tools=tool_schemas,
                        )

                        if hasattr(response, "message") or isinstance(response, dict):
                            msg_obj = (
                                response.message
                                if hasattr(response, "message")
                                else response.get("message", {})
                            )

                            if hasattr(msg_obj, "content"):
                                content = msg_obj.content
                                raw_tool_calls = msg_obj.tool_calls or []
                            else:
                                content = msg_obj.get("content")
                                raw_tool_calls = msg_obj.get("tool_calls") or []
                        else:
                            content = ""
                            raw_tool_calls = []

                        for i, tc in enumerate(raw_tool_calls):
                            if hasattr(tc, "function"):
                                tc_name = tc.function.name
                                args = tc.function.arguments
                            else:
                                tc_name = tc["function"]["name"]
                                args = tc["function"]["arguments"]

                            args_str = (
                                json.dumps(args) if isinstance(args, dict) else args
                            )
                            normalized_tool_calls.append(
                                {
                                    "id": f"call_{i}",
                                    "name": tc_name,
                                    "arguments": args_str,
                                }
                            )

                # Record assistant response
                assistant_message = {"role": "assistant"}
                if content is not None:
                    assistant_message["content"] = content

                if normalized_tool_calls:
                    assistant_message["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                            },
                        }
                        for tc in normalized_tool_calls
                    ]

                self.messages.append(assistant_message)
                self._save_history()

                # Process tools
                if normalized_tool_calls:
                    for tc in normalized_tool_calls:
                        try:
                            function_args = json.loads(tc["arguments"])
                        except json.JSONDecodeError:
                            function_args = {}

                        result = self._execute_tool(tc["name"], function_args)
                        logging.info(f"Tool result: {result[:500]}...")

                        self.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "name": tc["name"],
                                "content": result,
                            }
                        )
                        self._save_history()
                else:
                    # Defensive Parsing Shield
                    if content:
                        clean_content = content.strip()
                        if clean_content.startswith("```json"):
                            clean_content = clean_content[7:]
                        if clean_content.startswith("```"):
                            clean_content = clean_content[3:]
                        if clean_content.endswith("```"):
                            clean_content = clean_content[:-3]

                        clean_content = clean_content.strip()

                        if clean_content.startswith("{") and clean_content.endswith(
                            "}"
                        ):
                            try:
                                leaked_json = json.loads(clean_content)
                                if "name" in leaked_json and "arguments" in leaked_json:
                                    tool_name = leaked_json["name"]
                                    tool_args = leaked_json["arguments"]

                                    console.print(
                                        f"\n[dim italic][SYSTEM: Intercepted raw text tool call for '{tool_name}'][/dim italic]"
                                    )
                                    result = self._execute_tool(tool_name, tool_args)

                                    self.messages.append(
                                        {
                                            "role": "tool",
                                            "tool_call_id": "manual_fallback",
                                            "name": tool_name,
                                            "content": result,
                                        }
                                    )
                                    self._save_history()
                                    return f"Action complete: {result}"

                                elif "message" in leaked_json.get("arguments", {}):
                                    return leaked_json["arguments"]["message"]

                            except Exception:
                                pass

                    return content if content else ""

            except Exception as e:
                return f"Error [{self.mode.upper()} Mode]: {str(e)}"
