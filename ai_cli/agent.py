import importlib
import inspect
import os
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from ai_cli.tools.base import BaseTool

# --- PHASE 2: Rich UI Imports ---
from rich.console import Console

console = Console()

try:
    from groq import Groq
except ImportError:
    Groq = None  # type: ignore

try:
    import ollama
except ImportError:
    ollama = None  # type: ignore

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler("agent.log")],
)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


class AIAgent:
    def __init__(self, api_key: Optional[str] = None, local_model: str = "qwen2.5-coder:3b"):
        self.api_key = api_key
        self.local_model = local_model

        self.session_tokens = 0
        self.session_cost = 0.0

        self.history_file = Path.home() / ".config" / "ai_cli" / "history.json"
        self.messages: List[Dict[str, Any]] = self._load_history()

        self.tools: Dict[str, BaseTool] = self._load_tools()

        if self.api_key and Groq is not None:
            self.mode = "online"
            self.client = Groq(api_key=self.api_key)
            self.model_name = "meta-llama/llama-4-scout-17b-16e-instruct"
        else:
            self.mode = "offline"
            if not ollama:
                logging.warning("Ollama library not found. Offline mode may fail.")

    def _load_tools(self) -> Dict[str, BaseTool]:
        """Dynamically loads all tools from the ai_cli/tools directory."""
        tools = {}
        tools_dir = Path(__file__).parent / "tools"

        for file in tools_dir.glob("*.py"):
            if file.name.startswith("__") or file.name == "base.py":
                continue

            module_name = f"ai_cli.tools.{file.stem}"
            module = importlib.import_module(module_name)

            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, BaseTool) and obj.__module__ == module_name:
                    instance = obj()
                    tools[instance.name] = instance
                    logging.info(f"Dynamically loaded tool: {instance.name}")

        return tools

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

                with console.status("[bold cyan]Agent is thinking...", spinner="dots"):
                    if self.mode == "online":
                        response = self.client.chat.completions.create(
                            model=self.model_name,  # type: ignore
                            max_tokens=4096,
                            messages=[system_msg] + self.messages,  # type: ignore
                            tools=tool_schemas,
                            tool_choice="auto",
                        )
                        message = response.choices[0].message
                        content = message.content

                        if hasattr(response, "usage") and response.usage:
                            tokens = response.usage.total_tokens
                            self.session_tokens += tokens
                            self.session_cost = (self.session_tokens / 1_000_000) * 0.05

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
                            messages=[system_msg] + self.messages,  # type: ignore
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

                assistant_message: Dict[str, Any] = {"role": "assistant"}
                if content is not None:
                    assistant_message["content"] = content

                if normalized_tool_calls:
                    assistant_message["tool_calls"] = []
                    for tc in normalized_tool_calls:
                        # Groq expects a string, Ollama expects a dictionary
                        if self.mode == "offline":
                            try:
                                formatted_args = json.loads(tc["arguments"])
                            except json.JSONDecodeError:
                                formatted_args = {}
                        else:
                            formatted_args = tc["arguments"]

                        assistant_message["tool_calls"].append(
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": formatted_args,
                                },
                            }
                        )

                self.messages.append(assistant_message)
                self._save_history()

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
