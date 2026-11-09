"""Backend-neutral LLM interface.

Both Groq and Ollama return the same ``(content, tool_calls)`` shape so
the agent loop does not need provider-specific branches.
"""

from __future__ import annotations

import copy
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from cliv.models import SessionStats, ToolCall


class BackendError(Exception):
    """A recoverable provider error."""


class BackendUnavailable(BackendError):
    """The optional provider package is not installed."""


class ToolCallingUnsupported(BackendError):
    """The selected local model cannot use the tool-calling API."""


def normalise_messages(
    messages: List[Dict[str, Any]], *, stringify_arguments: bool
) -> List[Dict[str, Any]]:
    """Deep-copy messages and adapt tool arguments to a provider format.

    Groq expects ``function.arguments`` to be a JSON string.  Ollama
    expects a mapping.  Internally cliv always stores a dict.
    """

    normalised = copy.deepcopy(messages)
    for message in normalised:
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function")
            if not function or "arguments" not in function:
                continue
            arguments = function["arguments"]
            if stringify_arguments:
                if isinstance(arguments, dict):
                    function["arguments"] = json.dumps(arguments)
            elif isinstance(arguments, str):
                try:
                    function["arguments"] = json.loads(arguments)
                except json.JSONDecodeError:
                    function["arguments"] = {}
    return normalised


class LLMBackend(ABC):
    """Common interface implemented by every provider."""

    name = "base"
    supports_tools = True

    def __init__(
        self,
        model: str,
        *,
        stats: Optional[SessionStats] = None,
    ):
        self.model = model
        self.stats = stats or SessionStats()

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Whether the optional SDK needed by this backend is installed."""

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, List[ToolCall]]:
        """Call the model with tool definitions."""

    @abstractmethod
    def generate(self, messages: List[Dict[str, Any]]) -> str:
        """Call the model without tools (used for offline summaries)."""
