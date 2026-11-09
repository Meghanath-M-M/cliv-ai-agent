"""Groq backend with real function calling and token accounting."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from cliv.config import DEFAULT_GROQ_MODEL
from cliv.llm.base import (
    BackendError,
    BackendUnavailable,
    LLMBackend,
    normalise_messages,
)
from cliv.models import SessionStats, ToolCall


def list_groq_model_ids(client: Any) -> Set[str]:
    """Return the model IDs exposed by Groq's Models API."""

    response = client.models.list()
    if isinstance(response, dict):
        data = response.get("data", [])
    else:
        data = getattr(response, "data", response)
    model_ids: Set[str] = set()
    for item in data or []:
        model_id = (
            item.get("id") if isinstance(item, dict) else getattr(item, "id", None)
        )
        if model_id:
            model_ids.add(str(model_id))
    return model_ids


class GroqBackend(LLMBackend):
    name = "groq"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_GROQ_MODEL,
        *,
        stats: Optional[SessionStats] = None,
    ):
        super().__init__(model, stats=stats)
        try:
            from groq import Groq
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise BackendUnavailable(
                "groq is not installed. Run `uv sync --extra online`."
            ) from exc
        self.client = Groq(api_key=api_key)

    @classmethod
    def is_available(cls) -> bool:
        try:
            import groq  # noqa: F401
        except ImportError:
            return False
        return True

    def list_model_ids(self) -> Set[str]:
        """Return the model IDs currently offered to this API key."""

        try:
            return list_groq_model_ids(self.client)
        except Exception as exc:  # noqa: BLE001 - provider errors vary
            raise BackendError(f"Could not list Groq models: {exc}") from exc

    def probe_tool_calling(self, tool_schemas: List[Dict[str, Any]]) -> None:
        """Verify this model accepts CLIV's real tool schemas.

        The request is deliberately tiny and its response is discarded; we
        only care that the provider accepts the same tool payload used by
        the agent loop.  A model being listed does not imply tool support.
        """

        if not tool_schemas:
            return
        try:
            self.client.chat.completions.create(
                model=self.model,
                max_tokens=16,
                messages=[
                    {
                        "role": "user",
                        "content": "Reply with the single word: ready.",
                    }
                ],
                tools=tool_schemas,
                tool_choice="auto",
            )
        except Exception as exc:  # noqa: BLE001 - provider errors vary
            raise BackendError(
                f"Model '{self.model}' rejected CLIV's tool schemas: {exc}"
            ) from exc

    def _record_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if not usage:
            return
        self.stats.add_usage(
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
            self.model,
        )

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, List[ToolCall]]:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": normalise_messages(messages, stringify_arguments=True),
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - provider errors vary
            logging.error("Groq call failed: %s", exc)
            raise BackendError(f"Groq call failed: {exc}") from exc

        self._record_usage(response)
        message = response.choices[0].message
        content = message.content or ""

        tool_calls: List[ToolCall] = []
        for tool_call in message.tool_calls or []:
            arguments = tool_call.function.arguments
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            tool_calls.append(
                ToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=arguments if isinstance(arguments, dict) else {},
                )
            )
        return content, tool_calls

    def generate(self, messages: List[Dict[str, Any]]) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=4096,
                messages=normalise_messages(messages, stringify_arguments=True),
            )
        except Exception as exc:  # noqa: BLE001
            logging.error("Groq generation failed: %s", exc)
            raise BackendError(f"Groq call failed: {exc}") from exc
        self._record_usage(response)
        return response.choices[0].message.content or ""
