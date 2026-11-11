"""Conversation history persistence and context-window trimming.

Instead of sending the last arbitrary 50 messages to the model, messages
are grouped into complete turns (a user message plus everything the
assistant did in response).  The newest turns are kept until the
approximate token budget is exhausted.  This preserves tool-call /
tool-result pairs, which APIs reject when split.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from cliv.config import HISTORY_FILE

# Rough estimate: one token is ~4 characters for English/code.
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


def estimate_message_tokens(message: Dict[str, Any]) -> int:
    total = estimate_tokens(str(message.get("content") or ""))
    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        total += estimate_tokens(str(function.get("name") or ""))
        total += estimate_tokens(str(function.get("arguments") or ""))
    return total


def _normalise_content(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure every message's ``content`` field is a string or None.

    Older history files (or a bug in the interactive undo path) could persist
    ``content`` as a list (e.g. ``["undo"]``).  Ollama / Pydantic rejects
    list-typed content, so we coerce it here before anything else touches the
    messages.
    """

    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            # Flatten: join any string items, ignore non-strings.
            message["content"] = " ".join(
                str(item) for item in content if item is not None
            )
        elif content is not None and not isinstance(content, str):
            message["content"] = str(content)
    return messages


def _normalise_tool_arguments(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Make persisted tool arguments a dict again (older files used JSON strings)."""

    for message in messages:
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function")
            if not function or "arguments" not in function:
                continue
            args = function["arguments"]
            if isinstance(args, str):
                try:
                    function["arguments"] = json.loads(args)
                except json.JSONDecodeError:
                    function["arguments"] = {}
    return messages


def _split_turns(messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Group messages so a tool exchange is never cut in half."""

    turns: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "user" and current:
            turns.append(current)
            current = [message]
        else:
            current.append(message)
    if current:
        turns.append(current)

    # Drop leading orphan messages (for example a truncated assistant turn).
    while turns and turns[0][0].get("role") != "user":
        turns.pop(0)
    return turns


def trim_messages(
    messages: List[Dict[str, Any]],
    *,
    max_tokens: Optional[int] = 12_000,
    max_messages: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return a context-safe slice of *messages*.

    ``max_tokens=None`` disables the budget.  ``max_messages`` is a hard
    upper bound applied before turn grouping.
    """

    if not messages:
        return []

    if max_messages is not None and max_messages > 0:
        messages = messages[-max_messages:]

    if max_tokens is None:
        return list(messages)

    turns = _split_turns(messages)
    kept: List[List[Dict[str, Any]]] = []
    total = 0

    for turn in reversed(turns):
        cost = sum(estimate_message_tokens(message) for message in turn)
        # Always keep at least the newest turn, even if it is oversized.
        if kept and total + cost > max_tokens:
            break
        kept.append(turn)
        total += cost

    kept.reverse()
    flat = [message for turn in kept for message in turn]
    while flat and flat[0].get("role") != "user":
        flat.pop(0)
    return flat


class HistoryStore:
    """Load, save, trim and clear the on-disk conversation."""

    def __init__(
        self,
        path: Path = HISTORY_FILE,
        *,
        max_messages: int = 200,
        max_context_tokens: int = 12_000,
    ):
        self.path = Path(path)
        self.max_messages = max_messages
        self.max_context_tokens = max_context_tokens

    def load(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                messages = json.load(handle)
            if not isinstance(messages, list):
                return []
            messages = _normalise_content(messages)
            return _normalise_tool_arguments(messages)
        except Exception as exc:  # noqa: BLE001 - corrupt history must not crash startup
            logging.error("Failed to load history: %s", exc)
            return []

    def save(self, messages: List[Dict[str, Any]]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Normalise before trimming so we never persist malformed content.
            sanitised = _normalise_content(list(messages))
            trimmed = trim_messages(
                sanitised,
                max_tokens=None,
                max_messages=self.max_messages,
            )
            # Write to a temporary file first so a crash cannot corrupt the
            # previous history file.
            temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
            with open(temp_path, "w", encoding="utf-8") as handle:
                json.dump(trimmed, handle, indent=2)
            os.replace(temp_path, self.path)
        except Exception as exc:  # noqa: BLE001
            logging.error("Failed to save history: %s", exc)

    def for_context(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return trim_messages(
            messages,
            max_tokens=self.max_context_tokens,
            max_messages=self.max_messages,
        )

    def clear(self) -> None:
        try:
            if self.path.exists():
                self.path.unlink()
        except Exception as exc:  # noqa: BLE001
            logging.error("Failed to delete history file: %s", exc)

    def list_sessions(
        self, limit: int = 50, project: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """List archived sessions sorted by most recent first."""
        from cliv.session_io import list_archived_sessions
        return list_archived_sessions(limit=limit, project=project)
