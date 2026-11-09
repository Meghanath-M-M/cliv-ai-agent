"""Core AI agent for cliv.

This module owns the agent loop and tool approval.  Provider-specific
code lives in ``cliv.llm``, persistence in ``cliv.history``, offline
fallback routing in ``cliv.offline``, and terminal I/O in ``cliv.ui``.
"""

from __future__ import annotations

import json
import logging
import socket
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from cliv.config import CONFIG_DIR, HISTORY_FILE, LOG_FILE
from cliv.history import HistoryStore, trim_messages
from cliv.intent import classify_intent
from cliv.llm import (
    BackendError,
    LLMBackend,
    OllamaBackend,
    ToolCallingUnsupported,
    build_backend,
)
from cliv.models import AgentError, SessionStats, ToolCall
from cliv.offline import OfflineAssistant
from cliv.sanitizer import sanitize_response
from cliv.text import strip_markdown_fences
from cliv.tools.base import BaseTool
from cliv.tools.loader import load_tools
from cliv.ui import ConsoleUI, UI

# Re-exported for backwards compatibility with older imports/tests.
__all__ = [
    "AIAgent",
    "AgentError",
    "SessionStats",
    "ToolCall",
    "setup_logging",
]

load_dotenv()

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
    except ImportError:  # pragma: no cover - stdlib always provides it
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


class OfflineFallback(Exception):
    """Internal signal: tool calling failed, use intent routing instead."""


class AIAgent:
    """Autonomous coding agent with hybrid online/offline execution."""

    MAX_ITERATIONS = 10
    MAX_HISTORY_MESSAGES = 200
    WRITE_TOOLS = {
        "edit_file",
        "multi_edit",
        "remove_file",
        "run_command",
        "run_tests",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        local_model: str = "llama3.1:8b",
        verbose: bool = False,
        auto_approve: bool = False,
        dry_run: bool = False,
        *,
        force_offline: bool = False,
        ui: Optional[UI] = None,
        history_store: Optional[HistoryStore] = None,
        backend: Optional[LLMBackend] = None,
        mode: Optional[str] = None,
        use_offline_tools: bool = True,
        internet_check=None,
    ):
        self.api_key = api_key
        self.local_model = local_model
        self.verbose = verbose
        self.auto_approve = auto_approve
        self.dry_run = dry_run
        self.force_offline = force_offline
        self.use_offline_tools = use_offline_tools
        self.ui: UI = ui or ConsoleUI()
        setup_logging(verbose)

        self.stats = SessionStats()
        self._internet_check = internet_check or self._check_internet
        self.history = history_store or HistoryStore(
            HISTORY_FILE, max_messages=self.MAX_HISTORY_MESSAGES
        )
        self.messages: List[Dict[str, Any]] = self.history.load()
        self.tools: Dict[str, BaseTool] = self._load_tools()
        self._write_tools = set(self.WRITE_TOOLS)
        self.model_notices: List[str] = []

        if backend is not None:
            self.backend: Optional[LLMBackend] = backend
            self.mode = mode or (
                "online" if getattr(backend, "name", "") == "groq" else "offline"
            )
        else:
            self.backend, self.mode = build_backend(
                api_key=api_key,
                local_model=local_model,
                stats=self.stats,
                force_offline=force_offline,
                internet_check=self._internet_check,
                tool_schemas=self._build_tool_schemas(),
                notices=self.model_notices,
            )

        self.offline = OfflineAssistant(self)

    @staticmethod
    def _check_internet(
        host: str = "8.8.8.8", port: int = 53, timeout: int = 2
    ) -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    def _auto_check_mode(self) -> None:
        if self.mode != "online" or self.force_offline:
            return
        if self._internet_check():
            return

        logging.warning("Connection lost. Switching to offline mode.")
        if OllamaBackend.is_available():
            self.backend = OllamaBackend(self.local_model, stats=self.stats)
            self.ui.print(
                "[bold yellow]Network lost:[/bold yellow] switched to local "
                f"Ollama model '{self.local_model}'."
            )
        else:
            self.backend = None
            self.ui.print(
                "[bold yellow]Network lost:[/bold yellow] Ollama is not "
                "installed, so no backend remains."
            )
        self.mode = "offline"

    def _load_tools(self) -> Dict[str, BaseTool]:
        return load_tools()

    def _load_history(self) -> List[Dict[str, Any]]:
        return self.history.load()

    def _save_history(self) -> None:
        self.history.save(self.messages)
        # Keep the in-memory list bounded too; the file is trimmed by the
        # store, but a long REPL session should not grow forever.
        if len(self.messages) > self.history.max_messages:
            self.messages = trim_messages(
                self.messages,
                max_tokens=None,
                max_messages=self.history.max_messages,
            )

    def clear_history(self) -> None:
        self.messages = []
        self.history.clear()

    def _needs_approval(self, tool_name: str) -> bool:
        tool = self.tools.get(tool_name)
        return bool(tool and tool.requires_approval)

    def _request_approval(self, tool_name: str, tool_input: Dict[str, Any]) -> bool:
        if self.auto_approve:
            self.ui.print(f"[dim]Auto-approved {tool_name} (--yes mode)[/dim]")
            return True

        tool = self.tools.get(tool_name)
        preview = tool.preview(tool_input) if tool else None

        if self.dry_run:
            preview_line = f"\nPreview:\n{preview}" if preview else ""
            self.ui.print(
                f"\n[bold blue]DRY RUN[/bold blue]"
                f"\nTool: [cyan]{tool_name}[/cyan]"
                f"\nArgs: [dim]{json.dumps(tool_input, indent=2)}[/dim]"
                f"{preview_line}"
                f"\n[dim]No changes applied (dry-run mode).[/dim]"
            )
            return True

        return self.ui.confirm(tool_name, tool_input, preview)

    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        logging.info("Executing tool: %s with input: %s", tool_name, tool_input)
        if not isinstance(tool_input, dict):
            return (
                f"Error: Malformed tool call for '{tool_name}' — "
                "arguments must be an object."
            )
        tool = self.tools.get(tool_name)
        if not tool:
            return f"Unknown tool: {tool_name}"

        required = tool.input_schema.get("required", [])
        missing = [arg for arg in required if arg not in tool_input]
        if missing:
            return f"Error: Missing required arguments for '{tool_name}': {missing}"

        if self._needs_approval(tool_name):
            if not self._request_approval(tool_name, tool_input):
                logging.info("User denied approval for %s", tool_name)
                return f"Tool '{tool_name}' was cancelled by user."
        if self.dry_run:
            return (
                f"[DRY RUN] Would execute {tool_name} with args: "
                f"{json.dumps(tool_input)}"
            )
        try:
            return tool.execute(**tool_input)
        except Exception as exc:  # noqa: BLE001 - tool failures become model context
            logging.error("Error executing %s: %s", tool_name, exc)
            return f"Error executing {tool_name}: {exc}"

    def _classify_intent(self, user_input: str) -> Tuple[str, Optional[str]]:
        """Backwards-compatible wrapper around ``cliv.intent``."""

        return classify_intent(user_input)

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        return strip_markdown_fences(text)

    def _check_syntax_python(self, code: str) -> Optional[str]:
        import ast

        try:
            ast.parse(code)
            return None
        except SyntaxError as exc:
            detail = f"Syntax error on line {exc.lineno}: {exc.msg}"
            if exc.text:
                detail += f" — {exc.text.strip()}"
            return detail

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
        except Exception as exc:  # noqa: BLE001
            logging.error("File resolution failed: %s", exc)
        return None

    # ------------------------------------------------------------------
    # Generation helpers (offline summaries and fallback)
    # ------------------------------------------------------------------
    def _generation_messages(
        self, prompt: str, history: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if history:
            for message in history[-10:]:
                role = message.get("role", "")
                content = message.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})
        if messages and messages[-1]["role"] == "user":
            messages[-1]["content"] = prompt
        else:
            messages.append({"role": "user", "content": prompt})
        return messages

    def _generate(
        self, prompt: str, history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        if self.backend is None:
            return "Error: No LLM backend is available. Install cliv[offline]."
        try:
            return self.backend.generate(
                self._generation_messages(prompt, history=history)
            )
        except BackendError as exc:
            logging.error("Generation failed: %s", exc)
            return f"Error: {exc}"

    def _ollama_generate(
        self, prompt: str, history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Backwards-compatible alias."""

        return self._generate(prompt, history=history)

    # ------------------------------------------------------------------
    # Online/unified tool loop
    # ------------------------------------------------------------------
    def _build_system_message(self) -> Dict[str, str]:
        tool_list = "\n".join(
            f"  - {tool.name}: {tool.description}" for tool in self.tools.values()
        )
        model_name = self.backend.model if self.backend else self.local_model
        return {
            "role": "system",
            "content": (
                f"You are {model_name}, an autonomous coding agent operating in a terminal environment.\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. When the user says a simple greeting, respond with plain English text.\n"
                "2. THE DIAGNOSTIC PROTOCOL: If asked to check, fix, or modify a file, you are STRICTLY FORBIDDEN from answering immediately or guessing. You MUST follow this exact sequence:\n"
                "   - Step A: Use the `read_file` tool to thoroughly inspect the actual contents of the file.\n"
                "   - Step A-fallback: If `read_file` reports the file was not found, immediately use the `list_files` tool on the current directory before giving up, then suggest the closest matching filename you find.\n"
                "   - Step B: Use the `edit_file` or `multi_edit` tool to fix any syntax typos, logical errors, or bugs you find during your inspection.\n"
                "3. FILE EDITING PROTOCOL: If the user asks you to edit, modify, update, change, refactor, or add to an existing file, you MUST use the `edit_file` or `multi_edit` tool. Prefer `multi_edit` when changing several files in one request so the user sees one combined diff.\n"
                "4. FILE CREATION PROTOCOL: If the user asks you to create, make, write, generate, or build a new file, script, or code, you MUST use the `edit_file` tool with the `new_string` parameter containing the full content. You are FORBIDDEN from outputting the code in your text response.\n"
                "5. FILE DELETION PROTOCOL: If the user asks you to remove, delete, or destroy a file, you MUST use the `remove_file` tool. NEVER use `edit_file` with empty content to simulate deletion.\n"
                "6. You MUST use tools to make changes or create files. NEVER print code blocks in your chat response. The user can see the files on their own disk.\n"
                "7. NEVER output raw JSON, tool names, or argument dictionaries in your text responses. If you intend to call a tool, use the actual tool-calling mechanism — never describe a tool call as plain text.\n"
                "8. After using a tool, always summarize the result in your own clear, conversational words — never paste a tool's raw output directly back to the user.\n"
                "9. All file tools are confined to the project directory. Do not try to read `.env`, private keys, or files outside the project; those accesses are blocked.\n"
                "10. Use `run_tests` or `run_command` only when the user asks you to run code. Both require explicit approval.\n"
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

    def _sanitize_response(
        self, content: str
    ) -> Tuple[Optional[str], Optional[ToolCall]]:
        return sanitize_response(
            content,
            available_tools=self.tools.keys(),
            write_tools=self._write_tools,
        )

    def _run_tool_loop(self) -> str:
        if self.backend is None:
            return "Error: No LLM backend is available. Install cliv[online] or cliv[offline]."

        system_msg = self._build_system_message()
        tool_schemas = self._build_tool_schemas()
        iteration = 0

        while iteration < self.MAX_ITERATIONS:
            iteration += 1
            logging.debug(
                "Agent loop iteration %s/%s", iteration, self.MAX_ITERATIONS
            )

            context = self.history.for_context(self.messages)
            full_messages = [system_msg] + context

            try:
                content, tool_calls = self.backend.chat(full_messages, tool_schemas)
            except ToolCallingUnsupported:
                raise
            except BackendError as exc:
                if self.mode == "offline":
                    raise OfflineFallback(str(exc)) from exc
                return f"Error [ONLINE Mode]: {exc}"
            except Exception as exc:  # noqa: BLE001
                logging.exception("Unexpected LLM failure")
                if self.mode == "offline":
                    raise OfflineFallback(str(exc)) from exc
                return f"Error [ONLINE Mode]: {exc}"

            if not content and not tool_calls:
                if self.mode == "offline":
                    raise OfflineFallback("model returned an empty response")
                return "I didn't receive a response from the model. Please try again."

            assistant_message: Dict[str, Any] = {"role": "assistant"}
            if content is not None:
                assistant_message["content"] = content

            if tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": tool_call.arguments,
                        },
                    }
                    for tool_call in tool_calls
                ]

            self.messages.append(assistant_message)
            self._save_history()

            if tool_calls:
                for tool_call in tool_calls:
                    result = self._execute_tool(
                        tool_call.name, tool_call.arguments
                    )
                    logging.info("Tool result: %s...", result[:500])
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.name,
                            "content": result,
                        }
                    )
                    self._save_history()
                continue

            if content:
                cleaned, intercepted = self._sanitize_response(content)
                if intercepted:
                    result = self._execute_tool(
                        intercepted.name, intercepted.arguments
                    )
                    if (
                        self.messages
                        and self.messages[-1].get("role") == "assistant"
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

        logging.warning("Max iterations (%s) reached.", self.MAX_ITERATIONS)
        return (
            "I've reached the maximum number of tool calls for this turn. "
            "Please break your request into smaller steps or rephrase."
        )

    def _finish_offline(self, user_input: str) -> str:
        # Intent routing still needs a model for generated content.  Without
        # one, a create request would otherwise write the error text to disk.
        if self.backend is None:
            return (
                "Error: No LLM backend is available. "
                "Install `cliv[online]` or `cliv[offline]`."
            )
        response = self.offline.chat(user_input, history=self.messages)
        self.messages.append({"role": "assistant", "content": response})
        self._save_history()
        return response

    def chat(self, user_input: str) -> str:
        """Run one conversation turn.

        Online and capable offline models use the real tool loop.
        Offline models without tool support fall back to intent routing.
        """

        logging.info("User input: %s", user_input)
        self.messages.append({"role": "user", "content": user_input})
        self._save_history()
        self._auto_check_mode()

        if self.mode == "offline":
            if not self.use_offline_tools or self.backend is None:
                return self._finish_offline(user_input)
            try:
                return self._run_tool_loop()
            except ToolCallingUnsupported as exc:
                logging.warning("Falling back to intent routing: %s", exc)
                return self._finish_offline(user_input)
            except OfflineFallback as exc:
                logging.warning("Falling back to intent routing: %s", exc)
                return self._finish_offline(user_input)

        try:
            return self._run_tool_loop()
        except ToolCallingUnsupported as exc:
            return f"Error [ONLINE Mode]: {exc}"
