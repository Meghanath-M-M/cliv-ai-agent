"""Structured output and machine-readable formatters for cliv (CI/CD and scripting)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def print_json(data: Any, indent: int = 2) -> None:
    """Print an object as indented, UTF-8 JSON to stdout."""
    print(json.dumps(data, indent=indent, ensure_ascii=False, default=str))


def format_agent_run_json(
    agent: Any,
    prompt: str,
    response: str,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Format single-shot agent interaction as structured JSON."""
    backend = getattr(agent, "backend", None)
    model = getattr(backend, "model", getattr(agent, "local_model", "unknown"))
    stats = getattr(agent, "stats", None)

    cost_usd = getattr(stats, "cost_usd", getattr(stats, "total_cost", 0.0)) if stats else 0.0
    prompt_tokens = getattr(stats, "input_tokens", getattr(stats, "prompt_tokens", 0)) if stats else 0
    completion_tokens = getattr(stats, "output_tokens", getattr(stats, "completion_tokens", 0)) if stats else 0
    total_tokens = getattr(stats, "total_tokens", prompt_tokens + completion_tokens) if stats else 0

    return {
        "status": "error" if error else "success",
        "prompt": prompt,
        "response": response,
        "error": error,
        "session_id": getattr(agent, "session_id", None),
        "mode": getattr(agent, "mode", "unknown"),
        "model": model,
        "touched_files": sorted(list(getattr(agent, "touched_files", []))),
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": cost_usd,
        },
    }
