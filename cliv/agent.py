"""
Core AI Agent for cliv.
Handles LLM calls, tool execution, memory, and response sanitization.
"""

import importlib
import inspect
import os
import socket
import logging
import json
import copy
import re
import ast
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from dotenv import load_dotenv

from cliv.config import CONFIG_DIR, LOG_FILE, HISTORY_FILE
from cliv.tools.base import BaseTool

# --- Rich UI ---
from rich.console import Console

console = Console()

# --- Optional LLM Clients ---
try:
    from groq import Groq
except ImportError:
    Groq = None  # type: ignore

try:
    import ollama

    _OLLAMA_AVAILABLE = True
except ImportError:
    _OLLAMA_AVAILABLE = False

load_dotenv()

# --- Logging Setup ---
LOG_DIR = CONFIG_DIR
LOG_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging(verbose: bool = False):
    """Configure logging with optional console output."""
    handlers = [logging.FileHandler(LOG_FILE)]
    if verbose:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=handlers,
    )
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


# --- Model Pricing (Groq, per 1M tokens) ---
MODEL_PRICING = {
    "meta-llama/llama-4-scout-17b-16e-instruct": {"input": 0.11, "output": 0.34},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "qwen/qwen3-32b": {"input": 0.29, "output": 0.59},
}


@dataclass
class SessionStats:
    """Tracks session-level token usage and cost."""

    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def add_usage(self, input_tokens: int, output_tokens: int, model: str):
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += input_tokens + output_tokens
        pricing = MODEL_PRICING.get(model, {"input": 0.11, "output": 0.34})
        self.cost_usd += (
            input_tokens / 1_000_000 * pricing["input"]
            + output_tokens / 1_000_000 * pricing["output"]
        )


@dataclass
class ToolCall:
    """Normalized representation of a tool call."""

    id: str
    name: str
    arguments: Dict[str, Any]


class AgentError(Exception):
    """Custom exception for agent-level errors."""

    pass


class AIAgent:
    """
    Autonomous coding agent with hybrid online/offline execution.

    Safety guards:
      - Max iteration limit prevents infinite loops.
      - Tool approval required for write operations.
      - Defensive shield validates tool names before execution.
      - Offline mode uses direct intent handlers for small models.
    """

    MAX_ITERATIONS = 10
    MAX_HISTORY_MESSAGES = 50  # Prevent context bloat

    def __init__(
        self,
        api_key: Optional[str] = None,
        local_model: str = "qwen2.5-coder:3b",
        verbose: bool = False,
    ):
        self.api_key = api_key
        self.local_model = local_model
        self.verbose = verbose
        setup_logging(verbose)

        self.stats = SessionStats()
        self.messages: List[Dict[str, Any]] = self._load_history()
        self.tools: Dict[str, BaseTool] = self._load_tools()

        # Track which tools require user approval
        self._write_tools = {
            "edit_file",
            "write_file",
            "remove_file",
        }  # extend as needed

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
            if not _OLLAMA_AVAILABLE:
                logging.warning("ollama is unavailable. Offline mode may fail.")

    # ------------------------------------------------------------------
    # Network & Mode
    # ------------------------------------------------------------------
    @staticmethod
    def _check_internet(
        host: str = "8.8.8.8", port: int = 53, timeout: int = 2
    ) -> bool:
        try:
            socket.setdefaulttimeout(timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
            return True
        except OSError:
            return False

    def _auto_check_mode(self):
        """Failover from online to offline if connection drops."""
        if self.mode == "online" and not self._check_internet():
            self.mode = "offline"
            self.client = None
            if not _OLLAMA_AVAILABLE:
                logging.warning(
                    "Connection lost and ollama unavailable; offline mode may fail."
                )

    # ------------------------------------------------------------------
    # Tool Loading
    # ------------------------------------------------------------------
    def _load_tools(self) -> Dict[str, BaseTool]:
        """Dynamically loads all tools from the cliv/tools directory."""
        tools: Dict[str, BaseTool] = {}
        tools_dir = Path(__file__).parent / "tools"

        if not tools_dir.exists():
            logging.warning(f"Tools directory not found: {tools_dir}")
            return tools

        for file in tools_dir.glob("*.py"):
            if file.name.startswith("__") or file.name == "base.py":
                continue

            module_name = f"cliv.tools.{file.stem}"
            try:
                module = importlib.import_module(module_name)
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, BaseTool) and obj.__module__ == module_name:
                        instance = obj()
                        tools[instance.name] = instance
                        logging.info(f"Dynamically loaded tool: {instance.name}")
            except Exception as e:
                logging.error(f"Failed to load tool module {module_name}: {e}")

        return tools

    # ------------------------------------------------------------------
    # History Management
    # ------------------------------------------------------------------
    def _load_history(self) -> List[Dict[str, Any]]:
        """Loads previous conversation from disk, normalizing tool call args."""
        if not HISTORY_FILE.exists():
            return []

        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                messages = json.load(f)

            # Normalize tool call arguments to dicts
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
        """Saves current conversation to disk, capping history size."""
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            # Cap history to prevent unbounded growth
            trimmed = self.messages[-self.MAX_HISTORY_MESSAGES :]
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(trimmed, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save history: {e}")

    def clear_history(self):
        """Wipes the conversation history from RAM and disk."""
        self.messages = []
        if HISTORY_FILE.exists():
            try:
                os.remove(HISTORY_FILE)
            except Exception as e:
                logging.error(f"Failed to delete history file: {e}")

    # ------------------------------------------------------------------
    # Tool Execution
    # ------------------------------------------------------------------
    def _needs_approval(self, tool_name: str) -> bool:
        """Returns True if the tool modifies the filesystem."""
        return tool_name in self._write_tools

    def _request_approval(self, tool_name: str, tool_input: Dict[str, Any]) -> bool:
        """Prompts the user for approval before executing a write tool."""
        console.print(
            f"\n[bold yellow]⚠️  Approval Required[/bold yellow]"
            f"\nTool: [cyan]{tool_name}[/cyan]"
            f"\nArgs: [dim]{json.dumps(tool_input, indent=2)}[/dim]"
        )
        try:
            response = input("Proceed? [y/N]: ").strip().lower()
            return response in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False

    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Executes a tool with optional user approval for writes."""
        logging.info(f"Executing tool: {tool_name} with input: {tool_input}")

        tool = self.tools.get(tool_name)
        if not tool:
            return f"Unknown tool: {tool_name}"

        # Human-in-the-loop for write operations
        if self._needs_approval(tool_name):
            if not self._request_approval(tool_name, tool_input):
                logging.info(f"User denied approval for {tool_name}")
                return f"Tool '{tool_name}' was cancelled by user."

        try:
            return tool.execute(**tool_input)
        except Exception as e:
            logging.error(f"Error executing {tool_name}: {e}")
            return f"Error executing {tool_name}: {e}"

    # ------------------------------------------------------------------
    # Offline Direct Intent Handlers
    # ------------------------------------------------------------------
    def _handle_direct_tool_intent(self, user_input: str) -> Optional[str]:
        """
        For offline mode: bypass the LLM for simple, unambiguous tool intents.
        Small models (3B) often fail to emit proper tool calls. We detect
        creation/read/fix/remove intents locally, use the LLM for reasoning, then
        execute tools ourselves.
        Returns the final response string, or None to fall through to normal flow.
        """
        if self.mode != "offline" or not _OLLAMA_AVAILABLE:
            return None

        lowered = user_input.lower().strip()

        # --- FILE DIAGNOSTIC / FIX ---
        fix_match = re.search(
            r"(?:find|fix|debug|check|correct|repair)(?:\s+(?:the|any|all|some))?\s+(?:error|errors|bug|bugs|issue|issues|problem|problems)?\s+(?:in|with)?\s+['\"]?([^\s'\"]+\.(?:py|js|ts|html|css|md|txt|json|yaml|yml|sh|rs|go))['\"]?",
            lowered,
        )
        if not fix_match:
            fix_match = re.search(
                r"(?:find|fix|debug|check|correct|repair)(?:\s+(?:the|any|all|some))?\s+(?:error|errors|bug|bugs|issue|issues|problem|problems)?\s+(?:in|with)?\s+['\"]?([^\s'\"]+)['\"]?",
                lowered,
            )
        if fix_match:
            filename = fix_match.group(1)
            return self._offline_diagnose_file(user_input, filename)

        # --- FILE REMOVAL ---
        remove_match = re.search(
            r"(?:remove|delete|rm|destroy)\s+(?:the\s+)?(?:file\s+)?['\"]?([^\s'\"]+\.(?:py|js|ts|html|css|md|txt|json|yaml|yml|sh|rs|go))['\"]?",
            lowered,
        )
        if not remove_match:
            remove_match = re.search(
                r"(?:remove|delete|rm|destroy)\s+(?:the\s+)?(?:file\s+)?['\"]?([^\s'\"]+)['\"]?",
                lowered,
            )
        if remove_match:
            filename = remove_match.group(1)
            return self._execute_tool("remove_file", {"path": filename})

        # --- FILE CREATION ---
        create_match = re.search(
            r"(?:create|make|write|generate)\s+(?:a\s+)?(?:new\s+)?(?:file\s+)?(?:called\s+)?(?:named\s+)?['\"]?([^\s'\"]+\.(?:py|js|ts|html|css|md|txt|json|yaml|yml|sh|rs|go))['\"]?",
            lowered,
        )
        if not create_match:
            create_match = re.search(
                r"(?:create|make|write|generate)\s+(?:a\s+)?(?:new\s+)?(?:file\s+)?(?:called\s+)?(?:named\s+)?['\"]?([^\s'\"]+)['\"]?",
                lowered,
            )
        if create_match:
            return self._offline_create_file(user_input, create_match.group(1))

        # --- FILE READING ---
        read_match = re.search(
            r"(?:show|read|display|open|view|cat)\s+(?:me\s+)?(?:the\s+)?(?:file\s+)?['\"]?([^\s'\"]+)['\"]?",
            lowered,
        )
        if read_match:
            return self._offline_read_file(read_match.group(1))

        # --- DIRECTORY LISTING ---
        if re.search(
            r"(?:list|show|ls|what's|what is)\s+(?:in\s+)?(?:the\s+)?(?:current\s+)?(?:directory|folder|dir)",
            lowered,
        ) or re.search(r"what files are (?:here|in this folder)", lowered):
            return self._execute_tool("list_files", {"path": "."})

        return None

    def _offline_create_file(self, user_input: str, filename: str) -> str:
        """Offline handler: generate content and create file."""
        content_prompt = (
            f"Generate the complete content for a file named '{filename}' "
            f"based on this request: '{user_input}'. "
            f"Output ONLY the raw file content. No markdown fences, no explanations, no preamble."
        )
        try:
            temp_response = ollama.chat(
                model=self.local_model,
                messages=[{"role": "user", "content": content_prompt}],
            )
            file_content = temp_response.message.content or ""
            file_content = self._strip_markdown_fences(file_content)

            result = self._execute_tool(
                "edit_file", {"path": filename, "new_string": file_content}
            )

            summary_prompt = (
                f"Summarize in 1-2 sentences what you just created in '{filename}'. "
                f"The file starts with: {file_content[:200]!r}"
            )
            summary_response = ollama.chat(
                model=self.local_model,
                messages=[{"role": "user", "content": summary_prompt}],
            )
            return summary_response.message.content or f"Created {filename}."
        except Exception as e:
            logging.error(f"Offline create handler failed: {e}")
            return f"Error creating file: {e}"

    def _check_syntax_python(self, code: str) -> Optional[str]:
        """
        Uses Python's ast module to check for syntax errors.
        Returns the error message if invalid, None if valid.
        """
        try:
            ast.parse(code)
            return None
        except SyntaxError as e:
            return f"Syntax error on line {e.lineno}: {e.msg} — {e.text.strip() if e.text else ''}"

    def _offline_diagnose_file(self, user_input: str, filename: str) -> str:
        """
        Offline handler: read file, check syntax programmatically, ask LLM to fix if broken.
        Forces the read -> check -> fix sequence. Does NOT trust the LLM to self-diagnose.
        """
        # Step 1: Read the file
        file_content = self._execute_tool("read_file", {"path": filename})
        if file_content.startswith("Error:"):
            listing = self._execute_tool("list_files", {"path": "."})
            return f"Could not find '{filename}'. Here's what's in the current directory:\n\n{listing}"

        # Step 2: ACTUAL SYNTAX CHECK (don't trust the 3B model)
        syntax_error = None
        if filename.endswith(".py"):
            syntax_error = self._check_syntax_python(file_content)

        if not syntax_error:
            # No syntax errors found by actual parser
            return f"I checked `{filename}` with Python's syntax checker. No issues found — the file is valid."

        # Step 3: We have a real syntax error. Ask LLM to fix it.
        fix_prompt = (
            f"The file '{filename}' has a syntax error. Fix it and output ONLY the corrected file content.\n\n"
            f"Error: {syntax_error}\n\n"
            f"Broken code:\n```python\n{file_content}\n```\n\n"
            f"Output ONLY the fixed code. No explanations, no markdown fences:"
        )

        try:
            fix_response = ollama.chat(
                model=self.local_model,
                messages=[{"role": "user", "content": fix_prompt}],
            )
            fixed_code = fix_response.message.content or ""
            fixed_code = self._strip_markdown_fences(fixed_code)

            # Verify the fix actually works
            fixed_error = self._check_syntax_python(fixed_code)
            if fixed_error:
                return (
                    f"I found a syntax error in `{filename}`: {syntax_error}\n\n"
                    f"I tried to fix it but the correction still has issues: {fixed_error}\n\n"
                    f"Here's what I generated:\n```python\n{fixed_code}\n```"
                )

            # Step 4: Apply the fix via tool (triggers approval)
            result = self._execute_tool(
                "edit_file",
                {
                    "path": filename,
                    "old_string": file_content,
                    "new_string": fixed_code,
                },
            )

            return (
                f"Found and fixed a syntax error in `{filename}`:\n"
                f"  → {syntax_error}\n\n"
                f"The file has been patched. {result}"
            )

        except Exception as e:
            logging.error(f"Offline diagnose handler failed: {e}")
            return f"Error diagnosing file: {e}"

    def _offline_read_file(self, filename: str) -> str:
        """Offline handler: read and summarize file."""
        file_content = self._execute_tool("read_file", {"path": filename})
        if file_content.startswith("Error:"):
            listing = self._execute_tool("list_files", {"path": "."})
            return f"Could not find '{filename}'. Here's what's in the current directory:\n\n{listing}"

        summary_prompt = (
            f"The user asked about '{filename}'. Here's the content (first 800 chars):\n"
            f"{file_content[:800]}\n\n"
            f"Summarize what this file does in 1-2 sentences."
        )
        try:
            summary_response = ollama.chat(
                model=self.local_model,
                messages=[{"role": "user", "content": summary_prompt}],
            )
            return summary_response.message.content or file_content
        except Exception:
            return file_content

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Removes markdown code fences from model output."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        return text.strip()

    # ------------------------------------------------------------------
    # Intent Detection (Creation Reminder)
    # ------------------------------------------------------------------
    def _maybe_inject_creation_reminder(self, user_input: str) -> str:
        """
        Detects file-creation intent and appends a tool-use reminder.
        This helps small offline models map natural language to tool calls.
        """
        creation_keywords = [
            "create",
            "make",
            "write",
            "generate",
            "build",
            "new file",
            "new script",
            "save a",
            "save the",
        ]
        lowered = user_input.lower()
        if any(kw in lowered for kw in creation_keywords):
            return (
                f"{user_input}\n\n"
                "[SYSTEM REMINDER: The user wants a file created on disk. "
                "Use the edit_file tool with new_string. Do NOT output code in your response.]"
            )
        return user_input

    # ------------------------------------------------------------------
    # LLM Abstraction Layer
    # ------------------------------------------------------------------
    def _build_system_message(self) -> Dict[str, str]:
        """Constructs the system prompt with current tool inventory."""
        tool_list = "\n".join(
            f"  - {t.name}: {t.description}" for t in self.tools.values()
        )
        return {
            "role": "system",
            "content": (
                f"You are {self.local_model}, an autonomous coding agent operating in a terminal environment.\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. When the user says a simple greeting, respond with plain English text.\n"
                "2. THE DIAGNOSTIC PROTOCOL: If asked to check, fix, or modify a file, you are STRICTLY FORBIDDEN from answering immediately or guessing. You MUST follow this exact sequence:\n"
                "   - Step A: Use the `read_file` tool to thoroughly inspect the actual contents of the file.\n"
                "   - Step A-fallback: If `read_file` reports the file was not found, immediately use the `list_files` tool on the current directory before giving up, then suggest the closest matching filename you find.\n"
                "   - Step B: Use the `edit_file` tool to fix any syntax typos, logical errors, or bugs you find during your inspection.\n"
                "3. FILE CREATION PROTOCOL: If the user asks you to create, make, write, generate, or build a file, script, or code, you MUST use the `edit_file` tool with the `new_string` parameter containing the full content. You are FORBIDDEN from outputting the code in your text response. The user cannot copy-paste from the terminal — you must write it to disk for them.\n"
                "4. FILE DELETION PROTOCOL: If the user asks you to remove, delete, or destroy a file, you MUST use the `remove_file` tool. NEVER use `edit_file` with empty content to simulate deletion.\n"
                "5. You MUST use the `edit_file` tool to make changes or create files. NEVER print code blocks in your chat response. The user can see the files on their own disk.\n"
                "6. NEVER output raw JSON, tool names, or argument dictionaries in your text responses. If you intend to call a tool, use the actual tool-calling mechanism — never describe a tool call as plain text.\n"
                "7. After using a tool, always summarize the result in your own clear, conversational words — never paste a tool's raw output (like a bare list or array) directly back to the user. Explain what you found or changed in a few natural sentences. Keep it concise, but prioritize clarity over brevity — don't sacrifice a helpful explanation just to hit a sentence count. Never echo full code blocks.\n"
                f"\nAvailable tools:\n{tool_list}"
            ),
        }

    def _build_tool_schemas(self) -> List[Dict[str, Any]]:
        """Returns OpenAI-compatible tool schemas."""
        return [
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

    def _normalize_messages_for_api(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Normalizes message arguments for the target API.
        Online (Groq): args must be JSON strings.
        Offline (Ollama): args must be dicts.
        """
        normalized = copy.deepcopy(messages)
        for msg in normalized:
            if "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    if "function" in tc and "arguments" in tc["function"]:
                        args = tc["function"]["arguments"]
                        if self.mode == "online":
                            if isinstance(args, dict):
                                tc["function"]["arguments"] = json.dumps(args)
                        else:
                            if isinstance(args, str):
                                try:
                                    tc["function"]["arguments"] = json.loads(args)
                                except json.JSONDecodeError:
                                    tc["function"]["arguments"] = {}
        return normalized

    def _call_llm_online(
        self, messages: List[Dict[str, Any]], tool_schemas: List[Dict[str, Any]]
    ) -> Tuple[Optional[str], List[ToolCall]]:
        """Calls Groq API and returns (content, tool_calls)."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens=4096,
            messages=messages,
            tools=tool_schemas,
            tool_choice="auto",
        )
        message = response.choices[0].message
        content = message.content or ""

        # Track usage
        if hasattr(response, "usage") and response.usage:
            self.stats.add_usage(
                getattr(response.usage, "prompt_tokens", 0),
                getattr(response.usage, "completion_tokens", 0),
                self.model_name,
            )

        tool_calls: List[ToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                tool_calls.append(
                    ToolCall(id=tc.id, name=tc.function.name, arguments=args)
                )

        return content, tool_calls

    def _call_llm_offline(
        self, messages: List[Dict[str, Any]], tool_schemas: List[Dict[str, Any]]
    ) -> Tuple[Optional[str], List[ToolCall]]:
        """Calls Ollama API and returns (content, tool_calls)."""
        if not _OLLAMA_AVAILABLE:
            raise AgentError(
                "Offline mode requires 'ollama'. Install it and start the Ollama service."
            )

        response = ollama.chat(
            model=self.local_model,
            messages=messages,
            tools=tool_schemas,
        )
        msg_obj = response.message

        content = getattr(msg_obj, "content", None) or ""
        raw_tool_calls = getattr(msg_obj, "tool_calls", None) or []

        tool_calls: List[ToolCall] = []
        for i, tc in enumerate(raw_tool_calls):
            if hasattr(tc, "function"):
                tc_name = tc.function.name
                args = tc.function.arguments
            else:
                tc_name = tc.get("function", {}).get("name", "")
                args = tc.get("function", {}).get("arguments", {})

            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            tool_calls.append(ToolCall(id=f"call_{i}", name=tc_name, arguments=args))

        return content, tool_calls

    def _call_llm(
        self, messages: List[Dict[str, Any]], tool_schemas: List[Dict[str, Any]]
    ) -> Tuple[Optional[str], List[ToolCall]]:
        """Routes to online or offline LLM based on current mode."""
        if self.mode == "online":
            try:
                return self._call_llm_online(messages, tool_schemas)
            except Exception as e:
                err_str = str(e)
                if any(code in err_str for code in ("503", "500", "capacity")):
                    logging.warning(
                        f"Cloud API unavailable ({e}). Triggering local hardware failover."
                    )
                    console.print(
                        "\n[dim italic][SYSTEM: Cloud provider over capacity. Seamlessly rerouting to local hardware...][/dim italic]"
                    )
                    self.mode = "offline"
                    self.client = None
                    raise AgentError("FAILOVER")
                raise
        else:
            return self._call_llm_offline(messages, tool_schemas)

    # ------------------------------------------------------------------
    # Defensive Shield (Hardened)
    # ------------------------------------------------------------------
    def _sanitize_response(
        self, content: str
    ) -> Tuple[Optional[str], Optional[ToolCall]]:
        """
        Intercepts raw JSON tool-call leaks in model text output.
        Returns (cleaned_content, intercepted_tool_call) or (content, None).
        """
        if not content:
            return content, None

        clean_content = content.strip()
        # Strip markdown fences
        for prefix in ("```json", "```"):
            if clean_content.startswith(prefix):
                clean_content = clean_content[len(prefix) :]
        if clean_content.endswith("```"):
            clean_content = clean_content[:-3]
        clean_content = clean_content.strip()

        # Look for JSON-shaped tool call leaks
        json_match = re.search(
            r'\{.*"name"\s*:\s*"?[\w\.]+"?.*\}',
            clean_content,
            re.DOTALL,
        )
        if not json_match:
            return content, None

        raw_match = json_match.group(0)
        leaked_json = None

        try:
            leaked_json = json.loads(raw_match)
        except json.JSONDecodeError:
            # Repair pass: quote bare identifiers
            repaired = re.sub(
                r'("name"\s*:\s*)([\w\.]+)(?!")',
                r'\1"\2"',
                raw_match,
            )
            try:
                leaked_json = json.loads(repaired)
            except json.JSONDecodeError:
                leaked_json = None

        if leaked_json is None:
            logging.warning(f"Unrecoverable JSON-like leak: {raw_match[:200]}")
            return (
                "I had trouble formatting that response — could you rephrase your request?",
                None,
            )

        # Validate it looks like a tool call
        if "name" not in leaked_json:
            return content, None

        tool_name = leaked_json["name"]
        tool_args = leaked_json.get("arguments") or leaked_json.get("parameters") or {}

        # SECURITY: Only accept known tools
        if tool_name not in self.tools:
            logging.warning(
                f"Leaked text resembled tool call but '{tool_name}' is unknown."
            )
            return (
                "I had trouble formatting that response — could you rephrase your request?",
                None,
            )

        # SECURITY: Block write tools from shield (they need explicit approval)
        if tool_name in self._write_tools:
            logging.warning(
                f"Shield intercepted write tool '{tool_name}' — requiring explicit tool call."
            )
            return (
                f"I was about to use `{tool_name}`, but write operations require an explicit tool call. "
                "Please rephrase your request so I can call it properly.",
                None,
            )

        logging.info(f"Shield intercepted tool call for '{tool_name}'")
        return None, ToolCall(
            id="call_shield_fallback", name=tool_name, arguments=tool_args
        )

    # ------------------------------------------------------------------
    # Main Chat Loop
    # ------------------------------------------------------------------
    def chat(self, user_input: str) -> str:
        """
        Main conversation loop. Handles tool calls with iteration limits
        and human-in-the-loop approval for write operations.
        """
        logging.info(f"User input: {user_input}")

        # Inject creation reminder for small models that miss implicit intent
        processed_input = self._maybe_inject_creation_reminder(user_input)
        self.messages.append({"role": "user", "content": processed_input})
        self._save_history()

        self._auto_check_mode()

        # OFFLINE SHORTCUT: Handle simple intents directly
        # Small models (3B) often fail to emit proper tool calls via Ollama's API
        direct_result = self._handle_direct_tool_intent(user_input)
        if direct_result:
            return direct_result

        system_msg = self._build_system_message()
        tool_schemas = self._build_tool_schemas()

        iteration = 0
        while iteration < self.MAX_ITERATIONS:
            iteration += 1
            logging.debug(f"Agent loop iteration {iteration}/{self.MAX_ITERATIONS}")

            try:
                # Normalize messages for the target API
                api_messages = self._normalize_messages_for_api(self.messages)
                full_messages = [system_msg] + api_messages

                with console.status("[bold cyan]Agent is thinking...", spinner="dots"):
                    content, tool_calls = self._call_llm(full_messages, tool_schemas)

            except AgentError as e:
                if str(e) == "FAILOVER":
                    continue  # Retry in offline mode
                return f"Error [{self.mode.upper()} Mode]: {e}"
            except Exception as e:
                return f"Error [{self.mode.upper()} Mode]: {e}"

            # Build assistant message
            assistant_message: Dict[str, Any] = {"role": "assistant"}
            if content is not None:
                assistant_message["content"] = content

            if tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments
                            if self.mode == "offline"
                            else json.dumps(tc.arguments),
                        },
                    }
                    for tc in tool_calls
                ]

            self.messages.append(assistant_message)
            self._save_history()

            if tool_calls:
                # Execute tools and append results
                for tc in tool_calls:
                    result = self._execute_tool(tc.name, tc.arguments)
                    logging.info(f"Tool result: {result[:500]}...")

                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tc.name,
                            "content": result,
                        }
                    )
                    self._save_history()
                # Loop back for LLM to process results
                continue

            # No tool calls — check for leaked JSON in text
            if content:
                cleaned, intercepted = self._sanitize_response(content)
                if intercepted:
                    # Execute the intercepted tool call
                    result = self._execute_tool(intercepted.name, intercepted.arguments)

                    # Append as proper tool call to prevent infinite loop
                    if (
                        len(self.messages) > 0
                        and self.messages[-1]["role"] == "assistant"
                    ):
                        self.messages[-1].setdefault("tool_calls", []).append(
                            {
                                "id": intercepted.id,
                                "type": "function",
                                "function": {
                                    "name": intercepted.name,
                                    "arguments": intercepted.arguments,
                                },
                            }
                        )

                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": intercepted.id,
                            "name": intercepted.name,
                            "content": result,
                        }
                    )
                    self._save_history()
                    continue  # Let LLM summarize the result

                if cleaned is not None:
                    return cleaned

            return content if content else ""

        # Max iterations reached
        logging.warning(f"Max iterations ({self.MAX_ITERATIONS}) reached.")
        return (
            "I've reached the maximum number of tool calls for this turn. "
            "Please break your request into smaller steps or rephrase."
        )
