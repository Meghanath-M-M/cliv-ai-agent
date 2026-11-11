"""Model discovery and health inspection for Groq and Ollama backends."""

from __future__ import annotations

import json
import logging
import socket
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Set

from cliv.config import DEFAULT_GROQ_MODEL, get_api_key
from cliv.llm.base import BackendError
from cliv.llm.groq_backend import GroqBackend, list_groq_model_ids
from cliv.llm.ollama_backend import OllamaBackend


# Recommended models for offline coding
RECOMMENDED_OFFLINE_MODELS = [
    {"name": "llama3.1:8b", "desc": "Meta Llama 3.1 8B (tool-calling supported, balanced)", "tools": True},
    {"name": "qwen2.5-coder:7b", "desc": "Qwen 2.5 Coder 7B (state-of-the-art local code reasoning)", "tools": True},
    {"name": "qwen2.5-coder:3b", "desc": "Qwen 2.5 Coder 3B (lightweight, fast, tool-capable)", "tools": True},
    {"name": "phi4-mini:latest", "desc": "Microsoft Phi-4 Mini (compact, tool-capable)", "tools": True},
    {"name": "deepseek-coder:6.7b", "desc": "DeepSeek Coder 6.7B (strong code generation, intent-fallback)", "tools": False},
]

# Recommended models for online Groq
RECOMMENDED_GROQ_MODELS = [
    {"name": "llama-3.3-70b-versatile", "desc": "Llama 3.3 70B (fast, large context, tool-calling)"},
    {"name": "llama-3.1-8b-instant", "desc": "Llama 3.1 8B Instant (low latency, high throughput)"},
    {"name": "mixtral-8x7b-32768", "desc": "Mixtral 8x7B (moe architecture, large context)"},
]


def is_ollama_running(host: Optional[str] = None, timeout: float = 0.5) -> bool:
    """Check if the Ollama HTTP server is reachable."""
    raw_host = host or "127.0.0.1:11434"
    if "://" in raw_host:
        raw_host = raw_host.split("://", 1)[1]
    host_ip = raw_host.split(":", 1)[0]
    port = int(raw_host.split(":", 1)[1]) if ":" in raw_host else 11434

    try:
        with socket.create_connection((host_ip, port), timeout=timeout):
            return True
    except OSError:
        return False


def list_ollama_models(host: Optional[str] = None, timeout: float = 2.0) -> List[Dict[str, Any]]:
    """List locally installed Ollama models with metadata and tool capabilities."""
    if not is_ollama_running(host=host, timeout=1.0):
        return []

    url = "http://127.0.0.1:11434/api/tags"
    if host:
        base = host if host.startswith("http") else f"http://{host}"
        url = f"{base.rstrip('/')}/api/tags"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "cliv"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logging.debug("Could not query Ollama /api/tags: %s", exc)
        return []

    raw_models = data.get("models", [])
    results: List[Dict[str, Any]] = []

    for item in raw_models:
        name = item.get("name", "")
        if not name:
            continue
        size = item.get("size", 0)
        size_gb = round(size / (1024**3), 2) if size else 0.0
        details = item.get("details", {}) or {}
        param_size = details.get("parameter_size", "")
        quant = details.get("quantization_level", "")
        caps = item.get("capabilities", []) or []
        supports_tools = "tools" in caps

        results.append({
            "name": name,
            "size_gb": size_gb,
            "parameter_size": param_size,
            "quantization": quant,
            "supports_tools": supports_tools,
            "family": details.get("family", ""),
            "modified_at": item.get("modified_at", ""),
        })

    return results


def classify_groq_model(model_id: str) -> Dict[str, Any]:
    """Classify a Groq model into category and tool support."""
    m = model_id.lower()
    if "whisper" in m:
        return {
            "category": "audio",
            "desc": "Audio Speech-to-Text",
            "supports_tools": False,
            "usable_for_chat": False,
        }
    if "guard" in m or "moderation" in m or "prompt-guard" in m:
        return {
            "category": "safety_guard",
            "desc": "Safety / Jailbreak Classifier",
            "supports_tools": False,
            "usable_for_chat": False,
        }
    if "embed" in m:
        return {
            "category": "embedding",
            "desc": "Text Embedding",
            "supports_tools": False,
            "usable_for_chat": False,
        }
    return {
        "category": "chat",
        "desc": "Conversational Code / Chat LLM",
        "supports_tools": True,
        "usable_for_chat": True,
    }


def list_groq_models(api_key: Optional[str] = None, client: Any = None) -> List[Dict[str, Any]]:
    """List available Groq models for the configured API key."""
    key = api_key or get_api_key()
    if not key and client is None:
        return []

    try:
        if client is None:
            backend = GroqBackend(key)
            client = backend.client
        resp = client.models.list()
        data = resp.get("data", []) if isinstance(resp, dict) else getattr(resp, "data", resp)
        models: List[Dict[str, Any]] = []
        for item in data or []:
            m_id = item.get("id") if isinstance(item, dict) else getattr(item, "id", None)
            owned_by = item.get("owned_by") if isinstance(item, dict) else getattr(item, "owned_by", "groq")
            active = item.get("active", True) if isinstance(item, dict) else getattr(item, "active", True)
            if m_id:
                info = classify_groq_model(str(m_id))
                models.append({
                    "id": str(m_id),
                    "owned_by": str(owned_by or "groq"),
                    "active": bool(active),
                    "category": info["category"],
                    "desc": info["desc"],
                    "supports_tools": info["supports_tools"],
                    "usable_for_chat": info["usable_for_chat"],
                })
        return models
    except Exception as exc:
        logging.debug("Could not list Groq models: %s", exc)
        return []


def check_model_health(
    api_key: Optional[str] = None,
    current_model: Optional[str] = None,
    current_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Compile comprehensive health and availability diagnostic report."""
    key = api_key or get_api_key()
    groq_installed = GroqBackend.is_available()
    ollama_installed = OllamaBackend.is_available()
    ollama_running = is_ollama_running()

    groq_models = []
    groq_error = None
    if groq_installed and key:
        try:
            groq_models = list_groq_models(key)
        except Exception as exc:
            groq_error = str(exc)
    elif not key:
        groq_error = "No API key configured (set with: cliv config api-key)"
    elif not groq_installed:
        groq_error = "groq package not installed (run: pip install groq)"

    ollama_models = []
    ollama_error = None
    if ollama_running:
        ollama_models = list_ollama_models()
    elif ollama_installed:
        ollama_error = "Ollama server is not running (start with: ollama serve)"
    else:
        ollama_error = "ollama package not installed (run: pip install ollama)"

    return {
        "current_model": current_model or DEFAULT_GROQ_MODEL,
        "current_mode": current_mode or "unknown",
        "groq": {
            "installed": groq_installed,
            "has_key": bool(key),
            "models": groq_models,
            "error": groq_error,
        },
        "ollama": {
            "installed": ollama_installed,
            "running": ollama_running,
            "models": ollama_models,
            "error": ollama_error,
        },
        "recommendations": {
            "offline": RECOMMENDED_OFFLINE_MODELS,
            "online": RECOMMENDED_GROQ_MODELS,
        },
    }
