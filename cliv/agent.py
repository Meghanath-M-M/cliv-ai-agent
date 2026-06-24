import importlib
import inspect
import os
import socket
import logging
import json
import copy
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from cliv.tools.base import BaseTool

# --- PHASE 2: Rich UI Imports ---
from rich.console import Console

console = Console()

try:
    from groq import Groq
except ImportError:
    Groq = None  # type: ignore

try:
    from ollama import Client

    def _make_ollama_client(_):
        return Client()

    def _ollama_chat(client, model, messages, tools=None):
        return client.chat(model=model, messages=messages, tools=tools or [])

    def _ollama_message(obj):
        return obj.message

    def _has_tool_calls(msg):
        return bool(getattr(msg, "tool_calls", None))
except Exception:
    try:
        import ollama

        def _make_ollama_client(_):
            return ollama

        def _ollama_chat(client, model, messages, tools=None):
            return client.chat(model=model, messages=messages, tools=tools or [])

        def _ollama_message(obj):
            return obj.get("message", obj)

        def _has_tool_calls(msg):
            return bool(getattr(msg, "tool_calls", None) or msg.get("tool_calls"))
    except ImportError:
        _make_ollama_client = None
        _ollama_chat = None
        _ollama_message = None

        def _has_tool_calls(_):
            return False  # noqa: E301


load_dotenv()

log_dir = Path.home() / ".config" / "cliv"
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.FileHandler(log_dir / "agent.log")],
)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


class AIAgent:
    def _check_internet(self, host="8.8.8.8", port=53, timeout=2) -> bool:
        try:
            socket.setdefaulttimeout(timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
            return True
        except OSError:
            return False

    def __init__(
        self, api_key: Optional[str] = None, local_model: str = "qwen2.5-coder:3b"
    ):
        self.api_key = api_key
        self.local_model = local_model

        self.session_tokens = 0
        self.session_cost = 0.0

        self.history_file = Path.home() / ".config" / "cliv" / "history.json"
        self.messages: List[Dict[str, Any]] = self._load_history()

        self.tools: Dict[str, BaseTool] = self._load_tools()

        if self.api_key and Groq is not None and self._check_internet():
            self.mode = "online"
            self.client = Groq(api_key=self.api_key)
            self.model_name = "meta-llama/llama-4-scout-17b-16e-instruct"
        else:
            self.mode = "offline"
            if self.api_key and not self._check_internet():
                logging.warning(
                    "No internet connection detected. Switching to offline mode."
                )
            if _make_ollama_client is None:
                logging.warning("ollama is unavailable. Offline mode may fail.")

    def _load_tools(self) -> Dict[str, BaseTool]:
        """Dynamically loads all tools from the cliv/tools directory."""
        tools = {}
        tools_dir = Path(__file__).parent / "tools"

        for file in tools_dir.glob("*.py"):
            if file.name.startswith("__") or file.name == "base.py":
                continue

            module_name = f"cliv.tools.{file.stem}"
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
                    messages = json.load(f)
                for msg in messages:
                    if "tool_calls" in msg:
                        for tc in msg["tool_calls"]:
                            if "function" in tc and "arguments" in tc["function"]:
                                args = tc["function"]["arguments"]
                                if isinstance(args, str):
                                    try:
                                        tc["function"]["arguments"] = json.loads(args)
                                    except json.JSONDecodeError:
                                        tc["function"]["arguments"] = {}
                return messages
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

    def _auto_check_mode(self):
        if self.mode == "online" and not self._check_internet():
            self.mode = "offline"
            self.client = None
            if _make_ollama_client is None:
                logging.warning(
                    "Connection lost and ollama unavailable; offline mode may fail."
                )

    def chat(self, user_input: str) -> str:
        logging.info(f"User input: {user_input}")
        self.messages.append({"role": "user", "content": user_input})
        self._save_history()

        self._auto_check_mode()

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
                f"You are {self.local_model}, an autonomous coding agent operating in a terminal environment.\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. When the user says a simple greeting, respond with plain English text.\n"
                "2. THE DIAGNOSTIC PROTOCOL: If asked to check, fix, or modify a file, you are STRICTLY FORBIDDEN from answering immediately or guessing. You MUST follow this exact sequence:\n"
                "   - Step A: Use the `read_file` tool to thoroughly inspect the actual contents of the file.\n"
                "   - Step A-fallback: If `read_file` reports the file was not found, immediately use the `list_files` tool on the current directory before giving up, then suggest the closest matching filename you find.\n"
                "   - Step B: Use the `edit_file` tool to fix any syntax typos, logical errors, or bugs you find during your inspection.\n"
                "3. You MUST use the `edit_file` tool to make changes. NEVER print code blocks in your chat response. The user can see the files on their own disk.\n"
                "4. NEVER output raw JSON, tool names, or argument dictionaries in your text responses. If you intend to call a tool, use the actual tool-calling mechanism — never describe a tool call as plain text.\n"
                "5. After using a tool, always summarize the result in your own clear, conversational words — never paste a tool's raw output (like a bare list or array) directly back to the user. Explain what you found or changed in a few natural sentences. Keep it concise, but prioritize clarity over brevity — don't sacrifice a helpful explanation just to hit a sentence count. Never echo full code blocks."
            ),
        }

        while True:
            try:
                normalized_tool_calls = []
                content = None

                # --- API Compatibility Layer ---
                api_messages = copy.deepcopy(self.messages)
                for msg in api_messages:
                    if "tool_calls" in msg:
                        for tc in msg["tool_calls"]:
                            if "function" in tc and "arguments" in tc["function"]:
                                args = tc["function"]["arguments"]
                                if self.mode == "online":
                                    if isinstance(args, dict):
                                        tc["function"]["arguments"] = json.dumps(args)
                                elif self.mode == "offline":
                                    if isinstance(args, str):
                                        try:
                                            tc["function"]["arguments"] = json.loads(
                                                args
                                            )
                                        except json.JSONDecodeError:
                                            tc["function"]["arguments"] = {}

                with console.status("[bold cyan]Agent is thinking...", spinner="dots"):
                    if self.mode == "online":
                        try:
                            response = self.client.chat.completions.create(
                                model=self.model_name,  # type: ignore
                                max_tokens=4096,
                                messages=[system_msg] + api_messages,  # type: ignore
                                tools=tool_schemas,
                                tool_choice="auto",
                            )
                            message = response.choices[0].message
                            content = message.content

                            if hasattr(response, "usage") and response.usage:
                                tokens = response.usage.total_tokens
                                self.session_tokens += tokens
                                self.session_cost = (
                                    self.session_tokens / 1_000_000
                                ) * 0.05

                            if message.tool_calls:
                                for tc in message.tool_calls:
                                    args = tc.function.arguments
                                    if isinstance(args, str):
                                        try:
                                            args = json.loads(args)
                                        except json.JSONDecodeError:
                                            args = {}
                                    normalized_tool_calls.append(
                                        {
                                            "id": tc.id,
                                            "name": tc.function.name,
                                            "arguments": args,
                                        }
                                    )
                        except Exception as e:
                            if (
                                "503" in str(e)
                                or "capacity" in str(e).lower()
                                or "500" in str(e)
                            ):
                                logging.warning(
                                    f"Cloud API unavailable ({e}). Triggering local hardware failover."
                                )
                                console.print(
                                    "\n[dim italic][SYSTEM: Cloud provider over capacity. Seamlessly rerouting to local hardware...][/dim italic]"
                                )
                                self.mode = "offline"
                                self.client = None
                                continue
                            else:
                                raise e

                    else:
                        if _ollama_chat is None:
                            raise RuntimeError(
                                "Offline mode requires 'ollama'. Install it and start the Ollama service."
                            )

                        client = _make_ollama_client(None)
                        response = _ollama_chat(
                            client,
                            model=self.local_model,
                            messages=[system_msg] + api_messages,
                            tools=tool_schemas,
                        )
                        msg_obj = _ollama_message(response)

                        if hasattr(msg_obj, "content"):
                            content = getattr(msg_obj, "content", None)
                            raw_tool_calls = getattr(msg_obj, "tool_calls", None) or []
                        elif isinstance(msg_obj, dict):
                            content = msg_obj.get("content")
                            raw_tool_calls = msg_obj.get("tool_calls", []) or []
                        else:
                            content = None
                            raw_tool_calls = []

                        if content is None:
                            content = (
                                getattr(response, "content", None)
                                or getattr(response, "text", None)
                                or ""
                            )

                        for i, tc in enumerate(raw_tool_calls):
                            if hasattr(tc, "function"):
                                tc_name = tc.function.name
                                args = tc.function.arguments
                            else:
                                tc_name = tc["function"]["name"]
                                args = tc["function"]["arguments"]

                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except json.JSONDecodeError:
                                    args = {}

                            normalized_tool_calls.append(
                                {
                                    "id": f"call_{i}",
                                    "name": tc_name,
                                    "arguments": args,
                                }
                            )

                assistant_message: Dict[str, Any] = {"role": "assistant"}
                if content is not None:
                    assistant_message["content"] = content

                if normalized_tool_calls:
                    assistant_message["tool_calls"] = []
                    for tc in normalized_tool_calls:
                        formatted_args = tc["arguments"]
                        if self.mode == "online" and isinstance(formatted_args, dict):
                            formatted_args = json.dumps(formatted_args)

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
                        function_args = tc["arguments"]
                        if isinstance(function_args, str):
                            try:
                                function_args = json.loads(function_args)
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

                        # --- UPGRADED Regex JSON Shield ---
                        # Widened to also catch malformed leaks where the tool name
                        # isn't quoted, e.g. {"name": list_files, "arguments": {...}}
                        json_match = re.search(
                            r'\{.*"name"\s*:\s*"?[\w\.]+"?.*\}',
                            clean_content,
                            re.DOTALL,
                        )

                        if json_match:
                            raw_match = json_match.group(0)
                            leaked_json = None

                            try:
                                leaked_json = json.loads(raw_match)
                            except json.JSONDecodeError:
                                # Attempt a repair pass: quote bare identifiers used as
                                # the "name" value (e.g. list_files -> "list_files")
                                repaired = re.sub(
                                    r'("name"\s*:\s*)([\w\.]+)(?!")',
                                    r'\1"\2"',
                                    raw_match,
                                )
                                try:
                                    leaked_json = json.loads(repaired)
                                except json.JSONDecodeError:
                                    leaked_json = None

                            if leaked_json is not None:
                                if "name" in leaked_json and (
                                    "arguments" in leaked_json
                                    or "parameters" in leaked_json
                                ):
                                    tool_name = leaked_json["name"]
                                    tool_args = leaked_json.get(
                                        "arguments", leaked_json.get("parameters")
                                    )

                                    if tool_name not in self.tools:
                                        # Looked like a tool call but isn't a real tool —
                                        # don't execute it, and don't show the raw JSON either.
                                        logging.warning(
                                            f"Leaked text resembled a tool call but '{tool_name}' is unknown."
                                        )
                                        return (
                                            "I had trouble formatting that response — "
                                            "could you rephrase your request?"
                                        )

                                    logging.info(
                                        f"Intercepted raw text tool call for '{tool_name}'"
                                    )
                                    result = self._execute_tool(tool_name, tool_args)

                                    # --- THE FIX: Prevent Infinite Loop (Context Hack) ---
                                    if (
                                        len(self.messages) > 0
                                        and self.messages[-1]["role"] == "assistant"
                                    ):
                                        self.messages[-1]["tool_calls"] = [
                                            {
                                                "id": "call_manual_fallback",
                                                "type": "function",
                                                "function": {
                                                    "name": tool_name,
                                                    "arguments": tool_args,
                                                },
                                            }
                                        ]

                                    self.messages.append(
                                        {
                                            "role": "tool",
                                            "tool_call_id": "call_manual_fallback",
                                            "name": tool_name,
                                            "content": result,
                                        }
                                    )
                                    self._save_history()

                                    # --- THE FIX: Natural Language Recovery ---
                                    continue

                                elif "message" in leaked_json.get("arguments", {}):
                                    return leaked_json["arguments"]["message"]
                            else:
                                # The text looked like a tool call (matched the regex)
                                # but could not be parsed or repaired. Never show raw
                                # JSON-shaped text to the user — fail safely instead.
                                logging.warning(
                                    f"Unrecoverable JSON-like leak in model output: {raw_match[:200]}"
                                )
                                return (
                                    "I had trouble formatting that response — "
                                    "could you rephrase your request?"
                                )

                    return content if content else ""

            except Exception as e:
                return f"Error [{self.mode.upper()} Mode]: {str(e)}"
