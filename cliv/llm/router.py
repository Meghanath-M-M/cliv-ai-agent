"""Choose and validate the best available backend at startup.

Groq model IDs are retired regularly.  The router therefore:

1. checks the configured primary model against Groq's Models API;
2. falls back through the configured model list;
3. probes each candidate with CLIV's actual tool schemas;
4. reports every switch through logging and an optional notice list; and
5. tries Ollama when no Groq candidate passes.

Ollama is allowed to use models without tool support because the agent
already has a deterministic intent-routing fallback for those models.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from cliv.config import get_groq_fallback_models, get_groq_model
from cliv.llm.base import BackendError, LLMBackend
from cliv.llm.groq_backend import GroqBackend
from cliv.llm.ollama_backend import OllamaBackend
from cliv.models import SessionStats


def _notify(notices: Optional[List[str]], message: str) -> None:
    """Log a model switch and record it for the UI."""

    logging.warning(message)
    if notices is not None:
        notices.append(message)


def _unique_models(models: Sequence[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for model in models:
        name = (model or "").strip()
        if name and name not in seen:
            seen.add(name)
            result.append(name)
    return result


def _select_groq(
    api_key: str,
    primary: str,
    fallbacks: Sequence[str],
    stats: SessionStats,
    tool_schemas: Optional[List[Dict[str, Any]]],
    notices: Optional[List[str]],
) -> Optional[GroqBackend]:
    candidates = _unique_models([primary, *fallbacks])
    try:
        backend = GroqBackend(api_key, model=primary, stats=stats)
    except BackendError as exc:
        _notify(notices, f"Groq is unavailable: {exc}")
        return None

    try:
        available = backend.list_model_ids()
    except BackendError as exc:
        _notify(notices, f"Could not verify Groq model availability: {exc}")
        return None

    for model_name in candidates:
        if model_name not in available:
            _notify(notices, f"Groq model '{model_name}' is unavailable.")
            continue

        backend.model = model_name
        if tool_schemas:
            try:
                backend.probe_tool_calling(tool_schemas)
            except BackendError as exc:
                _notify(
                    notices,
                    f"Groq model '{model_name}' cannot use CLIV's tools: {exc}",
                )
                continue

        if model_name != primary:
            _notify(
                notices,
                f"Switched to Groq fallback model '{model_name}' "
                f"(requested '{primary}').",
            )
        return backend

    _notify(
        notices,
        "No configured Groq model passed the availability and tool checks.",
    )
    return None


def build_backend(
    *,
    api_key: Optional[str],
    local_model: str,
    stats: SessionStats,
    force_offline: bool = False,
    internet_check: Optional[Callable[[], bool]] = None,
    tool_schemas: Optional[List[Dict[str, Any]]] = None,
    model: Optional[str] = None,
    fallback_models: Optional[Sequence[str]] = None,
    notices: Optional[List[str]] = None,
) -> Tuple[Optional[LLMBackend], str]:
    """Return ``(backend, mode)``.

    Online is preferred only when a key, the Groq SDK, the network, and a
    validated tool-capable model are all present.  Otherwise the local
    backend is used if installed.
    """

    internet_check = internet_check or (lambda: False)
    groq_requested = bool(api_key) and not force_offline

    if groq_requested:
        if not GroqBackend.is_available():
            _notify(
                notices,
                "Groq support is not installed; falling back to Ollama.",
            )
        elif not internet_check():
            _notify(
                notices,
                "No internet connection detected; falling back to Ollama.",
            )
        else:
            primary = model or get_groq_model()
            fallbacks = (
                list(fallback_models)
                if fallback_models is not None
                else get_groq_fallback_models()
            )
            selected = _select_groq(
                api_key,
                primary,
                fallbacks,
                stats,
                tool_schemas,
                notices,
            )
            if selected is not None:
                return selected, "online"

    if OllamaBackend.is_available():
        if groq_requested:
            _notify(
                notices,
                f"Using local Ollama model '{local_model}' because Groq "
                "is unavailable.",
            )
        return OllamaBackend(local_model, stats=stats), "offline"

    if groq_requested:
        _notify(
            notices,
            "No usable LLM backend. Install the `online` or `offline` extra.",
        )
    else:
        logging.warning(
            "No usable LLM backend. Install `cliv[online]` or `cliv[offline]`."
        )
    return None, "offline"
