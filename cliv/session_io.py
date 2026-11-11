"""Session serialization, export, import, and archiving.

Provides data portability for cliv conversations:
- Export to Markdown for team sharing, pull requests, and documentation.
- Export to JSON for backup, debugging, and cross-machine portability.
- Import from JSON to resume past conversations or inspect external sessions.
- Automatic session archiving under ~/.config/cliv/sessions/.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from cliv.config import SESSIONS_DIR
from cliv.history import _normalise_content, _normalise_tool_arguments
from cliv.security import project_root


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class SessionData:
    """Full snapshot of an agent conversation session."""

    session_id: str
    project_root: str
    created_at: str
    updated_at: str
    model: str
    mode: str
    stats: Dict[str, Any]
    messages: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionData":
        messages = data.get("messages", [])
        if not isinstance(messages, list):
            messages = []
        messages = _normalise_content(messages)
        messages = _normalise_tool_arguments(messages)

        return cls(
            session_id=str(data.get("session_id", "")),
            project_root=str(data.get("project_root", "")),
            created_at=str(data.get("created_at", _now())),
            updated_at=str(data.get("updated_at", _now())),
            model=str(data.get("model", "unknown")),
            mode=str(data.get("mode", "online")),
            stats=dict(data.get("stats", {})),
            messages=messages,
            metadata=dict(data.get("metadata", {})),
        )


# ---------------------------------------------------------------------------
# Archiving
# ---------------------------------------------------------------------------

def archive_session(session: SessionData, sessions_dir: Optional[Path] = None) -> Path:
    """Save session to disk under the sessions archive directory."""
    target_dir = sessions_dir or SESSIONS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / f"{session.session_id}.json"

    temp_path = file_path.with_suffix(".json.tmp")
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(session.to_dict(), handle, indent=2, ensure_ascii=False)
    os.replace(temp_path, file_path)
    return file_path


def list_archived_sessions(
    limit: int = 50,
    project: Optional[Path] = None,
    sessions_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """List archived sessions sorted by most recent first."""
    target_dir = sessions_dir or SESSIONS_DIR
    if not target_dir.exists():
        return []

    target_project = str((project or project_root()).resolve()) if project else None

    results: List[Dict[str, Any]] = []
    for path in target_dir.glob("*.json"):
        if path.name.endswith(".tmp"):
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                continue

            session_proj = payload.get("project_root", "")
            if target_project and session_proj and session_proj != target_project:
                continue

            messages = payload.get("messages", [])
            user_turns = sum(1 for m in messages if m.get("role") == "user")
            stats = payload.get("stats", {})

            results.append(
                {
                    "id": payload.get("session_id", path.stem),
                    "created": payload.get("created_at", "")[:19].replace("T", " "),
                    "model": payload.get("model", "unknown"),
                    "mode": payload.get("mode", "unknown"),
                    "turns": user_turns,
                    "total_messages": len(messages),
                    "tokens": stats.get("total_tokens", 0),
                    "cost": stats.get("cost_usd", 0.0),
                    "project": session_proj,
                    "file_path": str(path),
                }
            )
        except Exception:
            continue

    results.sort(key=lambda s: s["created"], reverse=True)
    return results[:limit]


# ---------------------------------------------------------------------------
# Exporting
# ---------------------------------------------------------------------------

def export_session_json(session: SessionData, output_path: Union[str, Path]) -> Path:
    """Export session data to a JSON file."""
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(session.to_dict(), handle, indent=2, ensure_ascii=False)
    return path


def export_session_markdown(session: SessionData, output_path: Union[str, Path]) -> Path:
    """Export session conversation to a clean, formatted Markdown document."""
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = [
        f"# cliv Session Transcript: {session.session_id[:12]}",
        "",
        "| Attribute | Value |",
        "|---|---|",
        f"| **Session ID** | `{session.session_id}` |",
        f"| **Project** | `{session.project_root}` |",
        f"| **Created** | {session.created_at.replace('T', ' ')[:19]} UTC |",
        f"| **Updated** | {session.updated_at.replace('T', ' ')[:19]} UTC |",
        f"| **Model** | `{session.model}` ({session.mode.upper()}) |",
    ]

    stats = session.stats
    if stats:
        lines.extend(
            [
                f"| **Total Tokens** | {stats.get('total_tokens', 0):,} (in: {stats.get('input_tokens', 0):,}, out: {stats.get('output_tokens', 0):,}) |",
                f"| **Session Cost** | ${stats.get('cost_usd', 0.0):.6f} |",
            ]
        )

    lines.extend(["", "---", "", "## Conversation", ""])

    for msg in session.messages:
        role = msg.get("role", "")
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls") or []

        if role == "user":
            lines.append("### 👤 User")
            lines.append("")
            lines.append(content)
            lines.append("")

        elif role == "assistant":
            lines.append("### 🤖 cliv")
            lines.append("")
            if content:
                lines.append(content)
                lines.append("")

            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function") or {}
                    name = fn.get("name", "tool")
                    args = fn.get("arguments", {})
                    args_str = json.dumps(args, indent=2) if isinstance(args, dict) else str(args)
                    lines.append(f"> **🛠️ Tool Call:** `{name}`")
                    lines.append("```json")
                    lines.append(args_str)
                    lines.append("```")
                    lines.append("")

        elif role == "tool":
            name = msg.get("name", "tool")
            tool_content = str(content)
            # Truncate large tool output for readable markdown
            display_lines = tool_content.splitlines()
            if len(display_lines) > 30:
                short_content = "\n".join(display_lines[:30]) + f"\n... [{len(display_lines) - 30} lines truncated]"
            else:
                short_content = tool_content

            lines.append("<details>")
            lines.append(f"<summary><b>Result from <code>{name}</code></b></summary>")
            lines.append("")
            lines.append("```")
            lines.append(short_content)
            lines.append("```")
            lines.append("</details>")
            lines.append("")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    return path


# ---------------------------------------------------------------------------
# Importing
# ---------------------------------------------------------------------------

def import_session(source: Union[str, Path], sessions_dir: Optional[Path] = None) -> SessionData:
    """Import session data from a JSON file or an archived session ID.

    Raises:
        FileNotFoundError: If the file or session ID is not found.
        ValueError: If the file is not valid session JSON.
    """
    path = Path(source)

    if not path.exists():
        # Check if source is a session ID in sessions_dir
        target_dir = sessions_dir or SESSIONS_DIR
        candidate = target_dir / f"{source}.json"
        if candidate.exists():
            path = candidate
        else:
            raise FileNotFoundError(f"Session file or ID not found: '{source}'")

    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in session file '{path}': {exc}") from exc

    if isinstance(raw, list):
        # Raw message list export fallback
        messages = _normalise_content(raw)
        messages = _normalise_tool_arguments(messages)
        return SessionData(
            session_id=path.stem,
            project_root=str(project_root()),
            created_at=_now(),
            updated_at=_now(),
            model="unknown",
            mode="offline",
            stats={},
            messages=messages,
        )

    if not isinstance(raw, dict):
        raise ValueError(f"Unexpected JSON format in '{path}' (expected object or message array)")

    return SessionData.from_dict(raw)
