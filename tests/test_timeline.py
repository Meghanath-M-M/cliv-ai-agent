"""Tests for the append-only development timeline."""

import sqlite3

import pytest

from cliv.agent import AIAgent
from cliv.history import HistoryStore
from cliv.llm import LLMBackend
from cliv.models import SessionStats
from cliv.timeline import TimelineStore, render_timeline
from cliv.tools.read_timeline import ReadTimelineTool
from cliv.ui import NullUI


class StubBackend(LLMBackend):
    name = "stub"
    model = "stub-model"

    def __init__(self):
        super().__init__("stub-model", stats=SessionStats())
        self.generated = []

    @classmethod
    def is_available(cls):
        return True

    def chat(self, messages, tools=None):
        return ("acknowledged", [])

    def generate(self, messages):
        self.generated.append(messages[-1]["content"])
        return "generated"


def test_timeline_is_project_scoped_and_rendered(tmp_path):
    root_a = tmp_path / "alpha"
    root_b = tmp_path / "beta"
    root_a.mkdir()
    root_b.mkdir()
    store = TimelineStore(tmp_path / "timeline.db")

    assert store.record(
        "request",
        "Fix all failing tests",
        {"intent": "self_heal"},
        project=root_a,
        session_id="session-a",
    )
    assert store.record(
        "tool_result",
        "run_tests: 1 passed",
        {"tool": "run_tests", "success": True},
        project=root_a,
        session_id="session-a",
    )
    assert store.record(
        "request", "Read notes.txt", project=root_b, session_id="session-b"
    )

    events = store.recent(project=root_a)
    assert [event.event_type for event in events] == ["tool_result", "request"]
    assert events[0].details["success"] is True
    assert events[1].details["intent"] == "self_heal"
    assert all(event.project_root == str(root_a.resolve()) for event in events)
    assert store.recent(project=root_b)[0].summary == "Read notes.txt"

    rendered = render_timeline(events, project=root_a)
    assert "CLIV DEVELOPMENT TIMELINE" in rendered
    assert "Fix all failing tests" in rendered
    assert "run_tests: 1 passed" in rendered


def test_timeline_database_is_append_only(tmp_path):
    store = TimelineStore(tmp_path / "timeline.db")
    store.record("request", "original", project=tmp_path)

    with sqlite3.connect(store.path) as connection:
        with pytest.raises(sqlite3.DatabaseError):
            connection.execute("UPDATE events SET summary = 'changed'")
        with pytest.raises(sqlite3.DatabaseError):
            connection.execute("DELETE FROM events")

    assert store.recent(project=tmp_path)[0].summary == "original"


def test_agent_records_session_request_approval_and_tool_result(project_dir):
    store = TimelineStore(project_dir / "timeline.db")
    agent = AIAgent(
        api_key=None,
        ui=NullUI(approve=True),
        history_store=HistoryStore(project_dir / "history.json"),
        backend=StubBackend(),
        mode="offline",
        use_offline_tools=True,
        auto_approve=True,
        timeline_store=store,
        internet_check=lambda: True,
    )

    assert agent.chat("hello") == "acknowledged"
    result = agent._execute_tool(
        "edit_file", {"path": "notes.txt", "new_string": "timeline test\n"}
    )
    assert "Successfully wrote" in result

    events = store.recent(project=project_dir)
    event_types = [event.event_type for event in events]
    assert "session_start" in event_types
    assert "request" in event_types
    assert "approval" in event_types
    assert "tool_result" in event_types
    approval = next(event for event in events if event.event_type == "approval")
    assert approval.details["decision"] == "auto"


def test_read_timeline_tool_returns_recent_events(project_dir):
    store = TimelineStore()
    store.record(
        "request",
        "Analyse this project",
        {"intent": "project_analysis"},
        project=project_dir,
    )

    result = ReadTimelineTool().execute(limit=5)

    assert "CLIV DEVELOPMENT TIMELINE" in result
    assert "Analyse this project" in result
