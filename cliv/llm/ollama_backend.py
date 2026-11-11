"""Ollama backend.

Ollama models that support tools use the same agent loop as Groq.
Older/smaller models raise ``ToolCallingUnsupported`` and the agent
falls back to the intent-based offline assistant.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from cliv.llm.base import (
    BackendError,
    BackendUnavailable,
    LLMBackend,
    ToolCallingUnsupported,
    normalise_messages,
)
from cliv.models import SessionStats, ToolCall


def _get_attr(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


class OllamaBackend(LLMBackend):
    name = "ollama"
    supports_tools = True

    def __init__(
        self,
        model: str = "llama3.1:8b",
        *,
        stats: Optional[SessionStats] = None,
        host: Optional[str] = None,
    ):
        super().__init__(model, stats=stats)
        try:
            import ollama
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise BackendUnavailable(
                "ollama is not installed. Run `uv sync --extra offline`."
            ) from exc
        self._ollama = ollama
        self.host = host
        if host:
            self._client = ollama.Client(host=host)
        else:
            self._client = ollama

    @classmethod
    def is_available(cls) -> bool:
        try:
            import ollama  # noqa: F401
        except ImportError:
            return False
        return True

    def _extract_message(self, response: Any) -> Tuple[str, List[ToolCall]]:
        message = _get_attr(response, "message", None)
        if message is None:
            return "", []

        content = _get_attr(message, "content", "") or ""
        raw_tool_calls = _get_attr(message, "tool_calls", None) or []
        tool_calls: List[ToolCall] = []

        for raw in raw_tool_calls:
            function = _get_attr(raw, "function", None)
            if function is None:
                continue
            name = _get_attr(function, "name", "") or ""
            arguments = _get_attr(function, "arguments", {}) or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            call_id = _get_attr(raw, "id", None) or f"call_ollama_{uuid.uuid4().hex[:8]}"
            if name:
                tool_calls.append(
                    ToolCall(id=call_id, name=name, arguments=arguments)
                )
        return content, tool_calls

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, List[ToolCall]]:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": normalise_messages(messages, stringify_arguments=False),
        }
        if tools:
            kwargs["tools"] = tools

        try:
            response = self._client.chat(**kwargs)
            self._record_usage(response)
        except Exception as exc:  # noqa: BLE001 - provider errors vary
            message = str(exc).lower()
            if "tool" in message and (
                "support" in message or "not support" in message or "unsupported" in message
            ):
                raise ToolCallingUnsupported(
                    f"Model '{self.model}' does not support tool calling."
                ) from exc
            logging.error("Ollama call failed: %s", exc)
            raise BackendError(f"Ollama call failed: {exc}") from exc

        return self._extract_message(response)

    def _record_usage(self, response: Any) -> None:
        if not self.stats:
            return
        prompt_tokens = _get_attr(response, "prompt_eval_count", 0) or 0
        completion_tokens = _get_attr(response, "eval_count", 0) or 0
        if prompt_tokens or completion_tokens:
            self.stats.add_usage(prompt_tokens, completion_tokens, self.model)

    def generate(self, messages: List[Dict[str, Any]]) -> str:
        try:
            response = self._client.chat(
                model=self.model,
                messages=normalise_messages(messages, stringify_arguments=False),
            )
            self._record_usage(response)
        except Exception as exc:  # noqa: BLE001
            logging.error("Ollama generation failed: %s", exc)
            raise BackendError(f"Ollama call failed: {exc}") from exc
        content, _ = self._extract_message(response)
        return content
