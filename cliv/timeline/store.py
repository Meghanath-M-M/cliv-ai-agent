"""Append-only development timeline backed by SQLite.

The timeline is local metadata: it records what happened (requests, approvals,
tool results, test outcomes, and self-heal runs) without storing full file
contents.  It is intentionally append-only at the database level.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from cliv.config import TIMELINE_FILE
from cliv.security import project_root

MAX_SUMMARY_CHARS = 400
MAX_DETAILS_CHARS = 8_000


@dataclass(frozen=True)
class TimelineEvent:
    id: int
    created_at: str
    project_root: str
    session_id: str
    event_type: str
    summary: str
    details: Dict[str, Any]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class TimelineStore:
    """Small SQLite event log scoped by project root."""

    def __init__(self, path: Optional[Path] = None):
        configured = os.environ.get("CLIV_TIMELINE_DB")
        if path is not None:
            self.path = Path(path).expanduser()
        elif configured:
            self.path = Path(configured).expanduser()
        else:
            self.path = TIMELINE_FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialise()
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialise(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    project_root TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    details TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_events_project_id
                    ON events(project_root, id DESC);
                CREATE TRIGGER IF NOT EXISTS timeline_no_update
                    BEFORE UPDATE ON events
                BEGIN
                    SELECT RAISE(ABORT, 'timeline is append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS timeline_no_delete
                    BEFORE DELETE ON events
                BEGIN
                    SELECT RAISE(ABORT, 'timeline is append-only');
                END;
                """
            )

    def record(
        self,
        event_type: str,
        summary: str,
        details: Optional[Dict[str, Any]] = None,
        *,
        project: Optional[Path] = None,
        session_id: str = "",
    ) -> bool:
        """Append one event.  Failures are non-fatal for the caller."""

        try:
            root = (project or project_root()).resolve()
        except (OSError, RuntimeError):
            root = Path.cwd().resolve()
        summary = " ".join(str(summary).split())[:MAX_SUMMARY_CHARS]
        event_type = str(event_type).strip()[:64] or "event"

        raw_details = json.dumps(
            details or {}, ensure_ascii=False, default=str
        )
        if len(raw_details) > MAX_DETAILS_CHARS:
            raw_details = json.dumps(
                {"truncated": True, "preview": raw_details[:MAX_DETAILS_CHARS]}
            )

        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO events
                        (created_at, project_root, session_id, event_type,
                         summary, details)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _now(),
                        str(root),
                        session_id[:64],
                        event_type,
                        summary,
                        raw_details,
                    ),
                )
            return True
        except (OSError, sqlite3.Error):
            return False

    def recent(
        self,
        limit: int = 20,
        *,
        project: Optional[Path] = None,
        event_type: Optional[str] = None,
    ) -> List[TimelineEvent]:
        limit = max(1, min(int(limit), 200))
        try:
            root = (project or project_root()).resolve()
        except (OSError, RuntimeError):
            root = Path.cwd().resolve()
        query = (
            "SELECT id, created_at, project_root, session_id, event_type, "
            "summary, details FROM events WHERE project_root = ?"
        )
        params: List[Any] = [str(root)]
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()

        events: List[TimelineEvent] = []
        for row in rows:
            try:
                details = json.loads(row["details"])
            except (TypeError, json.JSONDecodeError):
                details = {}
            events.append(
                TimelineEvent(
                    id=row["id"],
                    created_at=row["created_at"],
                    project_root=row["project_root"],
                    session_id=row["session_id"],
                    event_type=row["event_type"],
                    summary=row["summary"],
                    details=details if isinstance(details, dict) else {},
                )
            )
        return events


def render_timeline(
    events: List[TimelineEvent], *, project: Optional[Path] = None
) -> str:
    lines = ["CLIV DEVELOPMENT TIMELINE", ""]
    try:
        root = (project or project_root()).resolve()
    except (OSError, RuntimeError):
        root = Path.cwd().resolve()
    lines.append(f"Project: {root}")
    if not events:
        lines.append("No events recorded yet.")
        return "\n".join(lines)

    lines.append("Recent events:")
    for event in events:
        timestamp = event.created_at.replace("T", " ")[:19]
        summary = " ".join(event.summary.split())
        lines.append(
            f"{timestamp}Z  {event.event_type:<14} {summary}"
        )
    return "\n".join(lines)
