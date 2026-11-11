"""Tests for session export, import, archiving, and resume."""

import json
from pathlib import Path
import pytest

from cliv.session_io import (
    SessionData,
    archive_session,
    list_archived_sessions,
    export_session_json,
    export_session_markdown,
    import_session,
)


def _make_sample_session() -> SessionData:
    return SessionData(
        session_id="test-session-42",
        project_root="/path/to/my_proj",
        created_at="2026-11-10T20:00:00",
        updated_at="2026-11-10T20:05:00",
        model="openai/gpt-oss-120b",
        mode="online",
        stats={
            "input_tokens": 1200,
            "output_tokens": 400,
            "total_tokens": 1600,
            "cost_usd": 0.00042,
        },
        messages=[
            {"role": "user", "content": "Fix the bug in auth.py"},
            {
                "role": "assistant",
                "content": "I will inspect auth.py.",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": {"path": "auth.py"},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "read_file",
                "content": "def check(): return False\n",
            },
            {
                "role": "assistant",
                "content": "Bug fixed! Returning True.",
            },
        ],
        metadata={"git_branch": "fix/auth"},
    )


def test_export_and_import_json_roundtrip(tmp_path):
    session = _make_sample_session()
    out_file = tmp_path / "exported.json"

    saved = export_session_json(session, out_file)
    assert saved.exists()

    imported = import_session(out_file)
    assert imported.session_id == session.session_id
    assert imported.model == session.model
    assert imported.stats["total_tokens"] == 1600
    assert len(imported.messages) == 4
    assert imported.messages[0]["content"] == "Fix the bug in auth.py"
    assert imported.messages[1]["tool_calls"][0]["function"]["name"] == "read_file"


def test_export_markdown_transcript(tmp_path):
    session = _make_sample_session()
    out_file = tmp_path / "transcript.md"

    saved = export_session_markdown(session, out_file)
    assert saved.exists()

    content = out_file.read_text(encoding="utf-8")
    assert "# cliv Session Transcript: test-session" in content
    assert "test-session-42" in content
    assert "openai/gpt-oss-120b" in content
    assert "$0.000420" in content
    assert "### 👤 User" in content
    assert "Fix the bug in auth.py" in content
    assert "### 🤖 cliv" in content
    assert "Tool Call:** `read_file`" in content
    assert "Result from <code>read_file</code>" in content
    assert "Bug fixed! Returning True." in content


def test_archive_and_list_sessions(tmp_path):
    sessions_dir = tmp_path / "sessions"
    session1 = _make_sample_session()
    session2 = SessionData(
        session_id="session-second",
        project_root="/path/to/my_proj",
        created_at="2026-11-10T21:00:00",
        updated_at="2026-11-10T21:02:00",
        model="llama-3.1-8b-instant",
        mode="offline",
        stats={"total_tokens": 500, "cost_usd": 0.0},
        messages=[{"role": "user", "content": "hello"}],
    )

    p1 = archive_session(session1, sessions_dir=sessions_dir)
    p2 = archive_session(session2, sessions_dir=sessions_dir)

    assert p1.exists()
    assert p2.exists()

    listed = list_archived_sessions(sessions_dir=sessions_dir)
    assert len(listed) == 2
    # Sorted by most recent first
    assert listed[0]["id"] == "session-second"
    assert listed[1]["id"] == "test-session-42"

    # Import by session ID from archive dir
    imported_by_id = import_session("test-session-42", sessions_dir=sessions_dir)
    assert imported_by_id.session_id == "test-session-42"


def test_import_invalid_path_raises_error():
    with pytest.raises(FileNotFoundError):
        import_session("nonexistent-file-path.json")


def test_import_invalid_json_raises_error(tmp_path):
    corrupt = tmp_path / "bad.json"
    corrupt.write_text("{not valid json")
    with pytest.raises(ValueError):
        import_session(corrupt)
