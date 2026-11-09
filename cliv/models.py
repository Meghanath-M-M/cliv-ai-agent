"""Shared data models for cliv."""

from dataclasses import dataclass
from typing import Any, Dict


# Groq pricing, USD per 1M tokens. Update as pricing changes.
MODEL_PRICING = {
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60},
    "openai/gpt-oss-20b": {"input": 0.075, "output": 0.30},
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
