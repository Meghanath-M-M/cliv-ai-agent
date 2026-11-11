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
from cliv.guardian import redact_data, redact_secrets
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
                CREATE TABLE IF NOT EXISTS session_usage (
                    session_id TEXT PRIMARY KEY,
                    project_root TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    model TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    cost_usd REAL NOT NULL DEFAULT 0.0
                );
                CREATE INDEX IF NOT EXISTS idx_usage_project
                    ON session_usage(project_root);
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

        return self.record_event(
            event_type,
            summary,
            details,
            project=project,
            session_id=session_id,
        ) is not None

    def record_event(
        self,
        event_type: str,
        summary: str,
        details: Optional[Dict[str, Any]] = None,
        *,
        project: Optional[Path] = None,
        session_id: str = "",
    ) -> Optional[int]:
        """Append one event and return its row id, or ``None`` on failure."""

        try:
            root = (project or project_root()).resolve()
        except (OSError, RuntimeError):
            root = Path.cwd().resolve()
        summary = redact_secrets(" ".join(str(summary).split())).text[
            :MAX_SUMMARY_CHARS
        ]
        event_type = str(event_type).strip()[:64] or "event"

        raw_details = json.dumps(
            redact_data(details or {}), ensure_ascii=False, default=str
        )
        if len(raw_details) > MAX_DETAILS_CHARS:
            raw_details = json.dumps(
                {"truncated": True, "preview": raw_details[:MAX_DETAILS_CHARS]}
            )

        try:
            with self._connect() as connection:
                cursor = connection.execute(
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
                event_id = cursor.lastrowid
            return int(event_id) if event_id is not None else None
        except (OSError, sqlite3.Error):
            return None

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

    def record_usage(
        self,
        session_id: str,
        *,
        project: Optional[Path] = None,
        model: str = "",
        mode: str = "online",
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> bool:
        """Update or insert session token usage and cost."""
        if not session_id:
            return False
        try:
            root = (project or project_root()).resolve()
        except (OSError, RuntimeError):
            root = Path.cwd().resolve()
        now = _now()
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO session_usage
                        (session_id, project_root, created_at, updated_at,
                         model, mode, input_tokens, output_tokens, total_tokens, cost_usd)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        updated_at = excluded.updated_at,
                        model = CASE WHEN excluded.model != '' THEN excluded.model ELSE session_usage.model END,
                        mode = excluded.mode,
                        input_tokens = excluded.input_tokens,
                        output_tokens = excluded.output_tokens,
                        total_tokens = excluded.total_tokens,
                        cost_usd = excluded.cost_usd
                    """,
                    (
                        session_id,
                        str(root),
                        now,
                        now,
                        model,
                        mode,
                        int(input_tokens),
                        int(output_tokens),
                        int(total_tokens),
                        float(cost_usd),
                    ),
                )
            return True
        except Exception as exc:  # noqa: BLE001 - non-fatal
            import logging
            logging.warning("Failed to record usage in timeline: %s", exc)
            return False

    def get_project_cost(self, project: Optional[Path] = None) -> Dict[str, Any]:
        """Aggregate token usage and cost across all sessions for a project."""
        try:
            root = (project or project_root()).resolve()
        except (OSError, RuntimeError):
            root = Path.cwd().resolve()
        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT
                        COUNT(*) as total_sessions,
                        COALESCE(SUM(input_tokens), 0) as total_input_tokens,
                        COALESCE(SUM(output_tokens), 0) as total_output_tokens,
                        COALESCE(SUM(total_tokens), 0) as total_tokens,
                        COALESCE(SUM(cost_usd), 0.0) as total_cost_usd
                    FROM session_usage
                    WHERE project_root = ?
                    """,
                    (str(root),),
                ).fetchone()

                model_rows = connection.execute(
                    """
                    SELECT model, SUM(total_tokens) as tokens, SUM(cost_usd) as cost
                    FROM session_usage
                    WHERE project_root = ? AND model != ''
                    GROUP BY model
                    ORDER BY cost DESC
                    """,
                    (str(root),),
                ).fetchall()

                return {
                    "project_root": str(root),
                    "total_sessions": int(row["total_sessions"]),
                    "input_tokens": int(row["total_input_tokens"]),
                    "output_tokens": int(row["total_output_tokens"]),
                    "total_tokens": int(row["total_tokens"]),
                    "total_cost_usd": float(row["total_cost_usd"]),
                    "models": [
                        {
                            "model": m["model"],
                            "tokens": int(m["tokens"]),
                            "cost": float(m["cost"]),
                        }
                        for m in model_rows
                    ],
                }
        except Exception as exc:  # noqa: BLE001
            import logging
            logging.warning("Failed to get project cost: %s", exc)
            return {
                "project_root": str(root),
                "total_sessions": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "models": [],
            }

    def get_session_cost(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve token and cost stats for a single session."""
        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT * FROM session_usage WHERE session_id = ?
                    """,
                    (session_id,),
                ).fetchone()
                if not row:
                    return None
                return dict(row)
        except Exception as exc:  # noqa: BLE001
            import logging
            logging.warning("Failed to get session cost: %s", exc)
            return None


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
