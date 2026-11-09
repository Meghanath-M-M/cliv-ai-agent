"""Defensive parsing shield.

Small local models sometimes print a raw JSON tool call instead of using
the provider's function-calling API.  The shield can recover read-only
tool calls, but deliberately refuses to recover write operations: those
must be real tool calls so the approval gate always sees them.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Iterable, Optional, Tuple

from cliv.models import ToolCall


def sanitize_response(
    content: Optional[str],
    *,
    available_tools: Iterable[str],
    write_tools: Iterable[str],
) -> Tuple[Optional[str], Optional[ToolCall]]:
    """Return ``(cleaned_text, recovered_tool_call)``.

    ``cleaned_text`` is ``None`` when a tool call was recovered.  A
    malformed but JSON-like response is turned into a safe message rather
    than being shown to the user.
    """

    if not content:
        return content, None

    available = set(available_tools)
    write_only = set(write_tools)

    clean_content = content.strip()
    for prefix in ("```json", "```"):
        if clean_content.startswith(prefix):
            clean_content = clean_content[len(prefix) :]
    if clean_content.endswith("```"):
        clean_content = clean_content[:-3]
    clean_content = clean_content.strip()

    json_match = re.search(
        r'\{.*"name"\s*:\s*"?[\w\.]+"?.*\}',
        clean_content,
        re.DOTALL,
    )
    if not json_match:
        return content, None

    raw_match = json_match.group(0)
    leaked_json: Optional[Dict[str, Any]] = None

    try:
        leaked_json = json.loads(raw_match)
    except json.JSONDecodeError:
        repaired = re.sub(
            r'("name"\s*:\s*)([\w\.]+)(?!")',
            r'\1"\2"',
            raw_match,
        )
        try:
            leaked_json = json.loads(repaired)
        except json.JSONDecodeError:
            leaked_json = None

    if leaked_json is None:
        logging.warning("Unrecoverable JSON-like leak: %s", raw_match[:200])
        return (
            "I had trouble formatting that response — could you rephrase your request?",
            None,
        )

    if "name" not in leaked_json:
        return content, None

    tool_name = leaked_json["name"]
    tool_args = leaked_json.get("arguments") or leaked_json.get("parameters") or {}

    if tool_name not in available:
        logging.warning(
            "Leaked text resembled tool call but '%s' is unknown.", tool_name
        )
        return (
            "I had trouble formatting that response — could you rephrase your request?",
            None,
        )

    if tool_name in write_only:
        logging.warning(
            "Shield intercepted write tool '%s' — requiring explicit tool call.",
            tool_name,
        )
        return (
            f"I was about to use `{tool_name}`, but write operations require an explicit tool call. "
            "Please rephrase your request so I can call it properly.",
            None,
        )

    if not isinstance(tool_args, dict):
        tool_args = {}

    logging.info("Shield intercepted tool call for '%s'", tool_name)
    return None, ToolCall(
        id="call_shield_fallback", name=tool_name, arguments=tool_args
    )
