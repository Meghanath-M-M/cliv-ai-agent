"""Intent-based offline assistant.

This is the fallback used when the local model cannot call tools.  It is
not an autonomous loop: code classifies the request, executes a
predictable tool sequence, and uses the LLM only for generation and
summaries.  The agent prefers real Ollama tool calling when available.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from cliv.intent import classify_intent
from cliv.text import strip_markdown_fences

if TYPE_CHECKING:  # avoid a runtime import cycle
    from cliv.agent import AIAgent


class OfflineAssistant:
    def __init__(self, agent: "AIAgent"):
        self.agent = agent

    def chat(
        self, user_input: str, history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        intent, target = classify_intent(user_input)
        logging.info("Offline intent: %s, target: %s", intent, target)

        if intent == "conversation":
            return self._generate(user_input, history=history)

        if intent == "project_analysis":
            with self.agent.ui.status(
                "[dim]Building Project DNA...", spinner="dots"
            ):
                report = self.agent._execute_tool(
                    "analyse_project", {"path": "."}
                )
            return report

        if intent == "list_dir":
            with self.agent.ui.status("[dim]Listing files...", spinner="dots"):
                listing = self.agent._execute_tool("list_files", {"path": "."})
            return self._generate(
                "The user asked what's in the current directory.\n\n"
                f"Here is the directory listing:\n{listing}\n\n"
                "Summarize this in 1-2 sentences. Mention the number of files.",
                history=history,
            )

        if intent == "file_read":
            if not target:
                return "I need to know which file to read. Please specify a filename."
            target = self.agent._resolve_file(target) or target
            with self.agent.ui.status(
                f"[dim]Reading {target}...", spinner="dots"
            ):
                content = self.agent._execute_tool("read_file", {"path": target})
            if content.startswith("Error:"):
                with self.agent.ui.status(
                    "[dim]Listing files...", spinner="dots"
                ):
                    listing = self.agent._execute_tool("list_files", {"path": "."})
                return (
                    f"Could not find '{target}'. Here's what's in the current directory:\n\n"
                    f"{listing}"
                )
            return self._generate(
                f"The user asked about '{target}'. Here's the content (first 4000 chars):\n"
                f"{content[:4000]}\n\n"
                "Summarize what this file does in 1-2 sentences. If it's code, explain its purpose.",
                history=history,
            )

        if intent == "file_create":
            if not target:
                return "I need to know what file to create. Please specify a filename."
            with self.agent.ui.status(
                "[dim]Generating file content...", spinner="dots"
            ):
                content = self._generate(
                    f"Generate the complete content for a file named '{target}' "
                    f"based on this request: '{user_input}'. "
                    "Output ONLY the raw file content. No markdown fences, no explanations.",
                    history=history,
                )
            content = strip_markdown_fences(content)
            result = self.agent._execute_tool(
                "edit_file", {"path": target, "new_string": content}
            )
            summary = self._generate(
                f"You just created '{target}'. It starts with:\n{content[:200]}\n\n"
                "Summarize in 1 sentence what you created.",
                history=history,
            )
            return f"{summary}\n\n{result}"

        if intent == "file_fix":
            if not target:
                return "I need to know which file to fix. Please specify a filename."
            target = self.agent._resolve_file(target) or target
            with self.agent.ui.status(
                f"[dim]Reading {target}...", spinner="dots"
            ):
                content = self.agent._execute_tool("read_file", {"path": target})
            if content.startswith("Error:"):
                with self.agent.ui.status(
                    "[dim]Listing files...", spinner="dots"
                ):
                    listing = self.agent._execute_tool("list_files", {"path": "."})
                return (
                    f"Could not find '{target}'. Here's what's in the current directory:\n\n"
                    f"{listing}"
                )

            syntax_error = None
            if target.endswith(".py"):
                syntax_error = self.agent._check_syntax_python(content)

            if syntax_error:
                with self.agent.ui.status(
                    "[dim]Generating syntax fix...", spinner="dots"
                ):
                    fixed = self._generate(
                        f"The file '{target}' has a syntax error:\n{syntax_error}\n\n"
                        f"Code:\n```python\n{content}\n```\n\n"
                        "Output ONLY the corrected code. No explanations, no markdown fences:",
                        history=history,
                    )
                fixed = strip_markdown_fences(fixed)
                if self.agent._check_syntax_python(fixed):
                    return (
                        f"Found a syntax error in `{target}`: {syntax_error}\n\n"
                        "I generated a fix but it still has errors. Please review manually."
                    )
                result = self.agent._execute_tool(
                    "edit_file",
                    {"path": target, "old_string": content, "new_string": fixed},
                )
                summary = self._generate(
                    f"You fixed a syntax error in '{target}'. The error was: {syntax_error}\n"
                    "Summarize the fix in 1 sentence.",
                    history=history,
                )
                return f"{summary}\n\n{result}"

            with self.agent.ui.status(
                "[dim]Checking for logic bugs...", spinner="dots"
            ):
                fixed = self._generate(
                    "You are a senior software engineer.\n\n"
                    f"User request: {user_input}\n"
                    f"File: {target}\n"
                    f"Code:\n```python\n{content}\n```\n\n"
                    "If there are bugs (logic errors, wrong operators, missing imports, etc.), "
                    "return ONLY the corrected code. If no bugs, return EXACTLY: NO_BUG_FOUND",
                    history=history,
                )
            fixed = strip_markdown_fences(fixed)

            if re.match(r"^\s*NO_BUG_FOUND[\.!]?\s*$", fixed.strip(), re.IGNORECASE):
                return (
                    f"I inspected `{target}` for syntax and logic issues. "
                    "No obvious problems were found."
                )

            if fixed.strip() == content.strip():
                return (
                    f"I inspected `{target}` for syntax and logic issues. "
                    "No obvious problems were found."
                )

            syntax_err = self.agent._check_syntax_python(fixed)
            if target.endswith(".py") and syntax_err:
                with self.agent.ui.status(
                    "[dim]Retrying fix...", spinner="dots"
                ):
                    fixed = self._generate(
                        f"The previous fix for '{target}' has a syntax error: {syntax_err}\n\n"
                        f"Original code:\n```python\n{content}\n```\n\n"
                        "Return ONLY the corrected code. No markdown fences, no explanations:",
                        history=history,
                    )
                fixed = strip_markdown_fences(fixed)
                syntax_err = self.agent._check_syntax_python(fixed)

            if target.endswith(".py") and syntax_err:
                return (
                    f"I found potential issues in `{target}` but the generated fix has syntax errors: {syntax_err}\n"
                    "Please review manually."
                )

            result = self.agent._execute_tool(
                "edit_file",
                {"path": target, "old_string": content, "new_string": fixed},
            )
            summary = self._generate(
                f"You fixed bugs in '{target}'. The file was modified.\n"
                "Summarize what you fixed in 1-2 sentences.",
                history=history,
            )
            return f"{summary}\n\n{result}"

        if intent == "file_remove":
            if not target:
                return "I need to know which file to remove. Please specify a filename."
            target = self.agent._resolve_file(target) or target
            result = self.agent._execute_tool("remove_file", {"path": target})
            if "cancelled" in result.lower() or "error" in result.lower():
                return result
            return f"Removed `{target}`.\n\n{result}"

        if intent == "file_rename":
            if not target or "|" not in target:
                return (
                    "I need to know which file to rename and what to call it. "
                    "Example: 'rename old.py to new.py'"
                )
            old_name, new_name = target.split("|", 1)
            old_name = self.agent._resolve_file(old_name) or old_name
            file_content = self.agent._execute_tool(
                "read_file", {"path": old_name}
            )
            if file_content.startswith("Error:"):
                listing = self.agent._execute_tool("list_files", {"path": "."})
                return (
                    f"Could not find '{old_name}'. Here's what's in the current directory:\n\n"
                    f"{listing}"
                )
            create_result = self.agent._execute_tool(
                "edit_file", {"path": new_name, "new_string": file_content}
            )
            if "error" in create_result.lower():
                return create_result
            remove_result = self.agent._execute_tool(
                "remove_file", {"path": old_name}
            )
            if "cancelled" in remove_result.lower():
                self.agent._execute_tool("remove_file", {"path": new_name})
                return f"Rename cancelled. {remove_result}"
            create_msg = create_result.replace(
                "Successfully wrote", "Created"
            ).replace("Error:", "Failed to create:")
            remove_msg = remove_result.replace(
                "Successfully removed", "Removed"
            ).replace("Error:", "Failed to remove:")
            return (
                f"Renamed `{old_name}` to `{new_name}`.\n\n{create_msg}\n{remove_msg}"
            )

        return self._generate(
            f"The user said: '{user_input}'\n\n"
            "You are a helpful coding assistant. If they're asking about files, "
            "tell them to specify a filename. Otherwise, answer their question naturally.",
            history=history,
        )

    def _generate(
        self, prompt: str, history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        return self.agent._generate(prompt, history=history)
