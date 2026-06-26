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
import time
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
    """Configure logging with optional console output and rotation."""
    handlers = []

    try:
        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(
            LOG_FILE, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
    except ImportError:
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    handlers.append(file_handler)

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
    """

    MAX_ITERATIONS = 10
    MAX_HISTORY_MESSAGES = 50

    def __init__(
        self,
        api_key: Optional[str] = None,
        local_model: str = "llama3.1:8b",
        verbose: bool = False,
        auto_approve: bool = False,
        dry_run: bool = False,
    ):
        self.api_key = api_key
        self.local_model = local_model
        self.verbose = verbose
        self.auto_approve = auto_approve
        self.dry_run = dry_run
        setup_logging(verbose)

        self.stats = SessionStats()
        self.messages: List[Dict[str, Any]] = self._load_history()
        self.tools: Dict[str, BaseTool] = self._load_tools()

        self._write_tools = {"edit_file", "write_file", "remove_file"}

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
        if self.mode == "online" and not self._check_internet():
            self.mode = "offline"
            self.client = None
            if not _OLLAMA_AVAILABLE:
                logging.warning(
                    "Connection lost and ollama unavailable; offline mode may fail."
                )

    def _load_tools(self) -> Dict[str, BaseTool]:
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

    def _load_history(self) -> List[Dict[str, Any]]:
        if not HISTORY_FILE.exists():
            return []
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
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
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            trimmed = self.messages[-self.MAX_HISTORY_MESSAGES :]
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(trimmed, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save history: {e}")

    def clear_history(self):
        self.messages = []
        if HISTORY_FILE.exists():
            try:
                os.remove(HISTORY_FILE)
            except Exception as e:
                logging.error(f"Failed to delete history file: {e}")

    def _needs_approval(self, tool_name: str) -> bool:
        return tool_name in self._write_tools

    def _request_approval(self, tool_name: str, tool_input: Dict[str, Any]) -> bool:
        if self.auto_approve:
            console.print(f"[dim]Auto-approved {tool_name} (--yes mode)[/dim]")
            return True
        if self.dry_run:
            console.print(
                f"\n[bold blue]DRY RUN[/bold blue]"
                f"\nTool: [cyan]{tool_name}[/cyan]"
                f"\nArgs: [dim]{json.dumps(tool_input, indent=2)}[/dim]"
                f"\n[dim]No changes applied (dry-run mode).[/dim]"
            )
            return True

        console.print(
            f"\n[bold yellow]Approval Required[/bold yellow]"
            f"\nTool: [cyan]{tool_name}[/cyan]"
            f"\nArgs: [dim]{json.dumps(tool_input, indent=2)}[/dim]"
        )
        try:
            console.print("[bold yellow]Proceed? [Y/N]: [/bold yellow]", end="")
            response = input().strip().lower()

            return response in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False

    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        logging.info(f"Executing tool: {tool_name} with input: {tool_input}")
        if not isinstance(tool_input, dict):
            return f"Error: Malformed tool call for '{tool_name}' — arguments must be an object."
        tool = self.tools.get(tool_name)
        if not tool:
            return f"Unknown tool: {tool_name}"
        required = tool.input_schema.get("required", [])
        missing = [arg for arg in required if arg not in tool_input]
        if missing:
            return f"Error: Missing required arguments for '{tool_name}': {missing}"
        if self._needs_approval(tool_name):
            if not self._request_approval(tool_name, tool_input):
                logging.info(f"User denied approval for {tool_name}")
                return f"Tool '{tool_name}' was cancelled by user."
        if self.dry_run:
            return f"[DRY RUN] Would execute {tool_name} with args: {json.dumps(tool_input)}"
        try:
            return tool.execute(**tool_input)
        except Exception as e:
            logging.error(f"Error executing {tool_name}: {e}")
            return f"Error executing {tool_name}: {e}"

    # ==================================================================
    # INTENT CLASSIFICATION (NEW - ROBUST)
    # ==================================================================
    def _classify_intent(self, user_input: str) -> Tuple[str, Optional[str]]:
        """
        Classifies user intent and extracts filename if present.
        Returns (intent_category, filename_or_None)

        Categories: 'conversation', 'file_create', 'file_read', 'file_fix',
                    'file_remove', 'file_rename', 'list_dir', 'unknown'
        """
        lowered = user_input.lower().strip()

        # Extract any filename with extension from the input FIRST
        file_exts = r"(?:py|js|ts|html|css|md|txt|json|yaml|yml|sh|rs|go|java|cpp|c|h)"
        file_match = re.search(
            r"['\"]?([\w\-\./]+\.(?:" + file_exts + r"))['\"]?", lowered
        )
        extracted_file = file_match.group(1) if file_match else None

        # --- PURE SOCIAL (always conversation) ---
        social_patterns = [
            r"^\s*(hi+|hello|hey|greetings|howdy|hola|yo|sup)\b",
            r"^\s*(thanks?|thank\s+(?:you|u)|thx|ty)\b",
            r"^\s*(bye|goodbye|see\s+ya|cya|later|peace)\b",
            r"^\s*(ok|okay|cool|nice|great|awesome|good\s+job|well\s+done)\b",
        ]
        for pattern in social_patterns:
            if re.search(pattern, lowered):
                return ("conversation", None)

        # --- CONVERSATION (only if no file mentioned) ---
        if not extracted_file:
            conv_patterns = [
                r"^\s*(who\s+(?:are|r)\s+(?:you|u)|what\s+(?:are|r)\s+(?:you|u)|what\s+is\s+(?:your|ur)\s+name)",
                r"^\s*(what\s+can\s+(?:you|u)\s+do|what\s+(?:are|r)\s+(?:your|ur)\s+capabilities|how\s+can\s+(?:you|u)\s+help)",
                r"^\s*(how\s+(?:are|r)\s+(?:you|u)|how's\s+it\s+going|whats\s+up|what's\s+up)",
                r"^\s*(can\s+(?:you|u)\s+help|could\s+(?:you|u)\s+help)\s*$",
            ]
            for pattern in conv_patterns:
                if re.search(pattern, lowered):
                    return ("conversation", None)

        # --- LIST DIRECTORY ---
        if (
            re.search(
                r"(?:list|show|ls|display)\s+(?:all\s+)?(?:the\s+)?(?:files?|contents?|items?)",
                lowered,
            )
            or re.search(
                r"what\s+(?:files?|.*)\s+(?:are|is)\s+(?:here|in\s+this|present)",
                lowered,
            )
            or re.search(
                r"what\s*(?:'s|s|is)\s+in\s+(?:the\s+)?(?:this\s+)?(?:current\s+)?(?:directory|folder|dir)",
                lowered,
            )
        ):
            return ("list_dir", None)

        # --- FILE RENAME / MOVE ---
        rename_match = re.search(
            r"(?:rename|mv|move)\s+(?:the\s+)?(?:file\s+)?['\"]?([\w\-\.]+\.\w+)['\"]?\s+(?:to|as|->)?\s*['\"]?([\w\-\.]+\.\w+)['\"]?",
            lowered,
        )
        if rename_match:
            return ("file_rename", f"{rename_match.group(1)}|{rename_match.group(2)}")

        # --- FILE CREATION ---
        create_patterns = [
            r"(?:create|make|write|generate|build)\s+(?:a\s+)?(?:new\s+)?(?:file\s+)?(?:called|named)?\s*['\"]?([\w\-\.]+\.\w+)['\"]?",
            r"(?:create|make|write|generate|build)\s+['\"]?([\w\-\.]+\.\w+)['\"]?",
        ]
        for pattern in create_patterns:
            m = re.search(pattern, lowered)
            if m:
                fname = m.group(1)
                conversational = [
                    "make sense",
                    "make sure",
                    "make it",
                    "make this",
                    "write down",
                    "write back",
                    "generate ideas",
                    "build on",
                ]
                if not any(c in lowered for c in conversational):
                    return ("file_create", fname)

        # --- FILE FIX ---
        fix_patterns = [
            r"(?:fix|debug|correct|repair|patch)(?:\s+\w+){0,4}\s+(?:the\s+)?(?:file\s+)?['\"]?([\w\-\.]+\.\w+)['\"]?",
            r"(?:fix|debug|correct|repair)(?:\s+(?:the|any|all|some|this|that|a|an))?(?:\s+(?:error|errors|bug|bugs|issue|issues|problem|problems))?(?:\s+(?:in|with|on|of|for))\s+['\"]?([\w\-\.]+\.\w+)['\"]?",
        ]
        for pattern in fix_patterns:
            m = re.search(pattern, lowered)
            if m:
                return ("file_fix", m.group(1))

        # --- FILE READ ---
        read_patterns = [
            r"(?:show|read|display|open|view|cat|print|get|see)(?:\s+(?:me|us|the|this|that|my))?(?:\s+(?:content|contents|file|code|script))?(?:\s+of)?\s+['\"]?([\w\-\.]+\.\w+)['\"]?",
        ]
        for pattern in read_patterns:
            m = re.search(pattern, lowered)
            if m:
                return ("file_read", m.group(1))

        # --- FILE REMOVE ---
        remove_patterns = [
            r"(?:remove|delete|rm|destroy)\s+(?:the\s+)?(?:file\s+)?['\"]?([\w\-\.]+\.\w+)['\"]?",
        ]
        for pattern in remove_patterns:
            m = re.search(pattern, lowered)
            if m:
                return ("file_remove", m.group(1))

        # If we extracted a file but couldn't determine intent, guess based on keywords
        if extracted_file:
            if any(
                k in lowered
                for k in ["fix", "debug", "correct", "repair", "error", "bug"]
            ):
                return ("file_fix", extracted_file)
            if any(
                k in lowered
                for k in ["show", "read", "display", "view", "see", "open", "get"]
            ):
                return ("file_read", extracted_file)
            if any(k in lowered for k in ["remove", "delete", "rm", "destroy"]):
                return ("file_remove", extracted_file)
            if any(
                k in lowered for k in ["create", "make", "write", "generate", "build"]
            ):
                return ("file_create", extracted_file)
            # Default: if a file is mentioned but no keywords matched, treat as read
            return ("file_read", extracted_file)

        return ("unknown", None)

    # ==================================================================
    # OFFLINE MODE: Direct tool execution (NO Ollama tool calling)
    # ==================================================================
    def _offline_chat_simple(
        self, user_input: str, history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        OFFLINE MODE: Does NOT use Ollama's tool calling API.
        Instead, we classify intent in code, execute tools directly,
        and use the LLM only for content generation and summarization.
        This avoids all tool-call hallucination issues with small models.
        """
        intent, filename = self._classify_intent(user_input)
        logging.info(f"Offline intent: {intent}, file: {filename}")

        # --- CONVERSATION ---
        if intent == "conversation":
            return self._ollama_generate(user_input, history=history)

        # --- LIST DIRECTORY ---
        if intent == "list_dir":
            with console.status("[dim]Listing files...", spinner="dots"):
                listing = self._execute_tool("list_files", {"path": "."})
            return self._ollama_generate(
                f"The user asked what's in the current directory.\n\n"
                f"Here is the directory listing:\n{listing}\n\n"
                f"Summarize this in 1-2 sentences. Mention the number of files.",
                history=history,
            )

        # --- FILE READ ---
        if intent == "file_read":
            if not filename:
                return "I need to know which file to read. Please specify a filename."
            resolved = self._resolve_file(filename)
            if resolved:
                filename = resolved
            with console.status(f"[dim]Reading {filename}...", spinner="dots"):
                content = self._execute_tool("read_file", {"path": filename})
            if content.startswith("Error:"):
                with console.status("[dim]Listing files...", spinner="dots"):
                    listing = self._execute_tool("list_files", {"path": "."})
                return (
                    f"Could not find '{filename}'. Here's what's in the current directory:\n\n"
                    f"{listing}"
                )
            return self._ollama_generate(
                f"The user asked about '{filename}'. Here's the content (first 1000 chars):\n"
                f"{content[:1000]}\n\n"
                f"Summarize what this file does in 1-2 sentences. If it's code, explain its purpose.",
                history=history,
            )

        # --- FILE CREATE ---
        if intent == "file_create":
            if not filename:
                return "I need to know what file to create. Please specify a filename."
            with console.status("[dim]Generating file content...", spinner="dots"):
                content = self._ollama_generate(
                    f"Generate the complete content for a file named '{filename}' "
                    f"based on this request: '{user_input}'. "
                    f"Output ONLY the raw file content. No markdown fences, no explanations.",
                    history=history,
                )
            content = self._strip_markdown_fences(content)
            # NO spinner — approval prompt needs clean terminal
            result = self._execute_tool(
                "edit_file", {"path": filename, "new_string": content}
            )
            summary = self._ollama_generate(
                f"You just created '{filename}'. It starts with:\n{content[:200]}\n\n"
                f"Summarize in 1 sentence what you created.",
                history=history,
            )
            return f"{summary}\n\n{result}"

        # --- FILE FIX ---
        if intent == "file_fix":
            if not filename:
                return "I need to know which file to fix. Please specify a filename."
            resolved = self._resolve_file(filename)
            if resolved:
                filename = resolved
            with console.status(f"[dim]Reading {filename}...", spinner="dots"):
                content = self._execute_tool("read_file", {"path": filename})
            if content.startswith("Error:"):
                with console.status("[dim]Listing files...", spinner="dots"):
                    listing = self._execute_tool("list_files", {"path": "."})
                return (
                    f"Could not find '{filename}'. Here's what's in the current directory:\n\n"
                    f"{listing}"
                )

            # Check syntax first (authoritative)
            syntax_error = None
            if filename.endswith(".py"):
                syntax_error = self._check_syntax_python(content)

            if syntax_error:
                with console.status("[dim]Generating syntax fix...", spinner="dots"):
                    fixed = self._ollama_generate(
                        f"The file '{filename}' has a syntax error:\n{syntax_error}\n\n"
                        f"Code:\n```python\n{content}\n```\n\n"
                        f"Output ONLY the corrected code. No explanations, no markdown fences:",
                        history=history,
                    )
                fixed = self._strip_markdown_fences(fixed)
                if self._check_syntax_python(fixed):
                    return (
                        f"Found a syntax error in `{filename}`: {syntax_error}\n\n"
                        f"I generated a fix but it still has errors. Please review manually."
                    )
                # NO spinner — approval needs clean terminal
                result = self._execute_tool(
                    "edit_file",
                    {"path": filename, "old_string": content, "new_string": fixed},
                )
                summary = self._ollama_generate(
                    f"You fixed a syntax error in '{filename}'. The error was: {syntax_error}\n"
                    f"Summarize the fix in 1 sentence.",
                    history=history,
                )
                return f"{summary}\n\n{result}"

            # No syntax error - check for logic bugs
            with console.status("[dim]Checking for logic bugs...", spinner="dots"):
                fixed = self._ollama_generate(
                    f"You are a senior software engineer.\n\n"
                    f"User request: {user_input}\n"
                    f"File: {filename}\n"
                    f"Code:\n```python\n{content}\n```\n\n"
                    f"If there are bugs (logic errors, wrong operators, missing imports, etc.), "
                    f"return ONLY the corrected code. If no bugs, return EXACTLY: NO_BUG_FOUND",
                    history=history,
                )
            fixed = self._strip_markdown_fences(fixed)

            # Robust NO_BUG_FOUND detection (handles punctuation, whitespace, case)
            if re.match(r"^\s*NO_BUG_FOUND[\.!]?\s*$", fixed.strip(), re.IGNORECASE):
                return (
                    f"I inspected `{filename}` for syntax and logic issues. "
                    f"No obvious problems were found."
                )

            if fixed.strip() == content.strip():
                return (
                    f"I inspected `{filename}` for syntax and logic issues. "
                    f"No obvious problems were found."
                )

            # Retry once if the generated fix has syntax errors
            syntax_err = self._check_syntax_python(fixed)
            if filename.endswith(".py") and syntax_err:
                with console.status("[dim]Retrying fix...", spinner="dots"):
                    fixed = self._ollama_generate(
                        f"The previous fix for '{filename}' has a syntax error: {syntax_err}\n\n"
                        f"Original code:\n```python\n{content}\n```\n\n"
                        f"Return ONLY the corrected code. No markdown fences, no explanations:",
                        history=history,
                    )
                fixed = self._strip_markdown_fences(fixed)
                syntax_err = self._check_syntax_python(fixed)

            if filename.endswith(".py") and syntax_err:
                return (
                    f"I found potential issues in `{filename}` but the generated fix has syntax errors: {syntax_err}\n"
                    f"Please review manually."
                )

            # NO spinner — approval needs clean terminal
            result = self._execute_tool(
                "edit_file",
                {"path": filename, "old_string": content, "new_string": fixed},
            )
            summary = self._ollama_generate(
                f"You fixed bugs in '{filename}'. The file was modified.\n"
                f"Summarize what you fixed in 1-2 sentences.",
                history=history,
            )
            return f"{summary}\n\n{result}"

        # --- FILE REMOVE ---
        if intent == "file_remove":
            if not filename:
                return "I need to know which file to remove. Please specify a filename."
            resolved = self._resolve_file(filename)
            if resolved:
                filename = resolved
            # NO spinner — approval needs clean terminal
            result = self._execute_tool("remove_file", {"path": filename})
            if "cancelled" in result.lower() or "error" in result.lower():
                return result
            return f"Removed `{filename}`.\n\n{result}"

        # --- FILE RENAME ---
        if intent == "file_rename":
            if not filename or "|" not in filename:
                return "I need to know which file to rename and what to call it. Example: 'rename old.py to new.py'"
            old_name, new_name = filename.split("|", 1)
            resolved_old = self._resolve_file(old_name)
            if resolved_old:
                old_name = resolved_old
            # NO spinner — approval may be needed for edit_file and remove_file
            file_content = self._execute_tool("read_file", {"path": old_name})
            if file_content.startswith("Error:"):
                listing = self._execute_tool("list_files", {"path": "."})
                return f"Could not find '{old_name}'. Here's what's in the current directory:\n\n{listing}"
            create_result = self._execute_tool(
                "edit_file", {"path": new_name, "new_string": file_content}
            )
            if "error" in create_result.lower():
                return create_result
            remove_result = self._execute_tool("remove_file", {"path": old_name})
            if "cancelled" in remove_result.lower():
                self._execute_tool("remove_file", {"path": new_name})
                return f"Rename cancelled. {remove_result}"
            create_msg = create_result.replace("Successfully wrote", "Created").replace(
                "Error:", "Failed to create:"
            )
            remove_msg = remove_result.replace(
                "Successfully removed", "Removed"
            ).replace("Error:", "Failed to remove:")
            return (
                f"Renamed `{old_name}` to `{new_name}`.\n\n{create_msg}\n{remove_msg}"
            )

        # --- UNKNOWN: Try to answer with LLM, but don't use tools ---
        return self._ollama_generate(
            f"The user said: '{user_input}'\n\n"
            "You are a helpful coding assistant. If they're asking about files, "
            "tell them to specify a filename. Otherwise, answer their question naturally.",
            history=history,
        )

    def _ollama_generate(
        self, prompt: str, history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Simple wrapper for Ollama text generation. No tool calling.

        If history is provided, it is used as-is. The prompt replaces the last
        user message rather than being appended, preventing duplicate turns.
        """
        if not _OLLAMA_AVAILABLE:
            return "Error: Ollama is not available."

        messages = []
        if history and len(history) > 0:
            for msg in history[-10:]:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})
            if messages and messages[-1]["role"] == "user":
                messages[-1]["content"] = prompt
            else:
                messages.append({"role": "user", "content": prompt})
        else:
            messages.append({"role": "user", "content": prompt})

        max_retries = 3
        for attempt in range(max_retries):
            try:
                with console.status(
                    "[bold cyan]Agent is thinking...",
                    spinner="dots",
                ):
                    response = ollama.chat(
                        model=self.local_model,
                        messages=messages,
                    )
                return response.message.content or ""
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 1 * (attempt + 1)
                    logging.warning(
                        f"Ollama call failed (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {wait}s: {e}"
                    )
                    time.sleep(wait)
                else:
                    logging.error(
                        f"Ollama generation failed after {max_retries} attempts: {e}"
                    )
                    return f"Error: {e}"

    def _check_syntax_python(self, code: str) -> Optional[str]:
        try:
            ast.parse(code)
            return None
        except SyntaxError as e:
            return f"Syntax error on line {e.lineno}: {e.msg}"

    def _resolve_file(self, filename: str) -> Optional[str]:
        try:
            listing = self._execute_tool("list_files", {"path": "."})
            for line in listing.splitlines():
                candidate = line.strip()
                if candidate.startswith("[DIR]"):
                    continue
                name = candidate.removeprefix("[FILE]").strip()
                if "(" in name:
                    name = name[: name.rindex("(")].strip()
                if name.lower() == filename.lower():
                    return name
                if name.lower().endswith(filename.lower()):
                    return name
                if filename.lower() in name.lower():
                    return name
                name_stem = Path(name).stem
                filename_stem = Path(filename).stem
                if (
                    name_stem
                    and filename_stem
                    and name_stem.lower() == filename_stem.lower()
                ):
                    return name
        except Exception as e:
            logging.error(f"File resolution failed: {e}")
        return None

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            return "\n".join(lines).strip()

        code_block_match = re.search(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).strip()

        return text

    # ==================================================================
    # ONLINE MODE: Full tool calling via Groq
    # ==================================================================
    def _build_system_message(self) -> Dict[str, str]:
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
                "3. FILE EDITING PROTOCOL: If the user asks you to edit, modify, update, change, refactor, or add to an existing file, you MUST use the `edit_file` tool with both `old_string` (the exact text to replace) and `new_string` (the corrected text).\n"
                "4. FILE CREATION PROTOCOL: If the user asks you to create, make, write, generate, or build a new file, script, or code, you MUST use the `edit_file` tool with the `new_string` parameter containing the full content. You are FORBIDDEN from outputting the code in your text response. The user cannot copy-paste from the terminal — you must write it to disk for them.\n"
                "5. FILE DELETION PROTOCOL: If the user asks you to remove, delete, or destroy a file, you MUST use the `remove_file` tool. NEVER use `edit_file` with empty content to simulate deletion.\n"
                "6. You MUST use the `edit_file` tool to make changes or create files. NEVER print code blocks in your chat response. The user can see the files on their own disk.\n"
                "7. NEVER output raw JSON, tool names, or argument dictionaries in your text responses. If you intend to call a tool, use the actual tool-calling mechanism — never describe a tool call as plain text.\n"
                "8. After using a tool, always summarize the result in your own clear, conversational words — never paste a tool's raw output (like a bare list or array) directly back to the user. Explain what you found or changed in a few natural sentences. Keep it concise, but prioritize clarity over brevity — don't sacrifice a helpful explanation just to hit a sentence count. Never echo full code blocks.\n"
                f"\nAvailable tools:\n{tool_list}"
            ),
        }

    def _build_tool_schemas(self) -> List[Dict[str, Any]]:
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

    def _sanitize_response(
        self, content: str
    ) -> Tuple[Optional[str], Optional[ToolCall]]:
        if not content:
            return content, None

        clean_content = content.strip()
        for prefix in ("```json", "```"):
            if clean_content.startswith(prefix):
                clean_content = clean_content[len(prefix) :]
        if clean_content.endswith("```"):
            clean_content = clean_content[:-3]
        clean_content = clean_content.strip()

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

        if "name" not in leaked_json:
            return content, None

        tool_name = leaked_json["name"]
        tool_args = leaked_json.get("arguments") or leaked_json.get("parameters") or {}

        if tool_name not in self.tools:
            logging.warning(
                f"Leaked text resembled tool call but '{tool_name}' is unknown."
            )
            return (
                "I had trouble formatting that response — could you rephrase your request?",
                None,
            )

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

    # ==================================================================
    # MAIN CHAT LOOP
    # ==================================================================
    def chat(self, user_input: str) -> str:
        """
        Main conversation loop.

        ONLINE MODE: Uses Groq API with proper tool calling.
        OFFLINE MODE: Uses intent classification + direct tool execution.
                      Does NOT use Ollama's tool calling API (too unreliable).
        """
        logging.info(f"User input: {user_input}")

        self.messages.append({"role": "user", "content": user_input})
        self._save_history()

        self._auto_check_mode()

        # OFFLINE MODE: Use simple intent-based flow, NOT Ollama tool calling
        if self.mode == "offline":
            response = self._offline_chat_simple(user_input, history=self.messages)
            self.messages.append({"role": "assistant", "content": response})
            self._save_history()
            return response

        # ONLINE MODE: Full agent loop with Groq tool calling
        system_msg = self._build_system_message()
        tool_schemas = self._build_tool_schemas()

        iteration = 0
        while iteration < self.MAX_ITERATIONS:
            iteration += 1
            logging.debug(f"Agent loop iteration {iteration}/{self.MAX_ITERATIONS}")

            try:
                api_messages = self._normalize_messages_for_api(self.messages)
                full_messages = [system_msg] + api_messages

                with console.status("[bold cyan]Agent is thinking...", spinner="dots"):
                    content, tool_calls = self._call_llm_online(
                        full_messages, tool_schemas
                    )

            except Exception as e:
                return f"Error [ONLINE Mode]: {e}"

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
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in tool_calls
                ]

            self.messages.append(assistant_message)
            self._save_history()

            if tool_calls:
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
                continue

            if content:
                cleaned, intercepted = self._sanitize_response(content)
                if intercepted:
                    result = self._execute_tool(intercepted.name, intercepted.arguments)

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
                    continue

                if cleaned is not None:
                    return cleaned

            return content if content else ""

        logging.warning(f"Max iterations ({self.MAX_ITERATIONS}) reached.")
        return (
            "I've reached the maximum number of tool calls for this turn. "
            "Please break your request into smaller steps or rephrase."
        )
