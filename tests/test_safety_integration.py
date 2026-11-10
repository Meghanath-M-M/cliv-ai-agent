"""Integration tests for snapshots, approvals, and the timeline."""

import pytest

from cliv.agent import AIAgent
from cliv.history import HistoryStore
from cliv.llm import LLMBackend
from cliv.models import SessionStats
from cliv.snapshots import SnapshotError, SnapshotStore
from cliv.timeline import TimelineStore
from cliv.ui import NullUI


class StubBackend(LLMBackend):
    name = "stub"
    model = "stub-model"

    def __init__(self):
        super().__init__("stub-model", stats=SessionStats())

    @classmethod
    def is_available(cls):
        return True

    def chat(self, messages, tools=None):
        return ("acknowledged", [])

    def generate(self, messages):
        return "generated"


def make_agent(project_dir, *, approve=True, dry_run=False, snapshot_store=None):
    return AIAgent(
        api_key=None,
        ui=NullUI(approve=approve),
        history_store=HistoryStore(project_dir / "history.json"),
        backend=StubBackend(),
        mode="online",
        dry_run=dry_run,
        timeline_store=TimelineStore(project_dir / "timeline.db"),
        snapshot_store=snapshot_store or SnapshotStore(project_dir / "snapshots"),
        internet_check=lambda: True,
    )


def test_edit_creates_snapshot_and_undo_restores(project_dir):
    target = project_dir / "app.py"
    target.write_text("old\n")
    store = SnapshotStore(project_dir / "snapshots")
    agent = make_agent(project_dir, snapshot_store=store)

    result = agent._execute_tool(
        "edit_file", {"path": "app.py", "new_string": "new\n"}
    )

    assert "Successfully wrote" in result
    assert target.read_text() == "new\n"
    snapshot = store.latest_applied()
    assert snapshot is not None
    assert snapshot.tool == "edit_file"
    assert snapshot.files[0].path == "app.py"

    rollback = store.rollback(snapshot.id)

    assert rollback.success is True
    assert target.read_text() == "old\n"


def test_denied_edit_does_not_create_snapshot(project_dir):
    target = project_dir / "app.py"
    target.write_text("old\n")
    store = SnapshotStore(project_dir / "snapshots")
    agent = make_agent(project_dir, approve=False, snapshot_store=store)

    result = agent._execute_tool(
        "edit_file", {"path": "app.py", "new_string": "new\n"}
    )

    assert "cancelled" in result
    assert target.read_text() == "old\n"
    assert store.recent() == []


def test_dry_run_does_not_create_snapshot(project_dir):
    target = project_dir / "app.py"
    target.write_text("old\n")
    store = SnapshotStore(project_dir / "snapshots")
    agent = make_agent(project_dir, dry_run=True, snapshot_store=store)

    result = agent._execute_tool(
        "edit_file", {"path": "app.py", "new_string": "new\n"}
    )

    assert "DRY RUN" in result
    assert target.read_text() == "old\n"
    assert store.recent() == []


def test_snapshot_failure_blocks_the_write(project_dir):
    target = project_dir / "app.py"
    target.write_text("old\n")

    class FailingStore:
        def capture(self, *args, **kwargs):
            raise SnapshotError("disk full")

    agent = make_agent(project_dir, snapshot_store=FailingStore())

    result = agent._execute_tool(
        "edit_file", {"path": "app.py", "new_string": "new\n"}
    )

    assert "safety snapshot" in result
    assert "disk full" in result
    assert target.read_text() == "old\n"


def test_multi_edit_snapshots_every_file(project_dir):
    first = project_dir / "a.py"
    second = project_dir / "b.py"
    first.write_text("a\n")
    second.write_text("b\n")
    store = SnapshotStore(project_dir / "snapshots")
    agent = make_agent(project_dir, snapshot_store=store)

    result = agent._execute_tool(
        "multi_edit",
        {
            "edits": [
                {"path": "a.py", "old_string": "a", "new_string": "A"},
                {"path": "b.py", "old_string": "b", "new_string": "B"},
            ]
        },
    )

    assert "Successfully applied 2" in result
    snapshot = store.latest_applied()
    assert snapshot is not None
    assert {entry.path for entry in snapshot.files} == {"a.py", "b.py"}

    rollback = store.rollback(snapshot.id)

    assert rollback.success is True
    assert first.read_text() == "a\n"
    assert second.read_text() == "b\n"


def test_timeline_records_snapshot_and_snapshot_id(project_dir):
    target = project_dir / "app.py"
    target.write_text("old\n")
    store = SnapshotStore(project_dir / "snapshots")
    timeline = TimelineStore(project_dir / "timeline.db")
    agent = AIAgent(
        api_key=None,
        ui=NullUI(approve=True),
        history_store=HistoryStore(project_dir / "history.json"),
        backend=StubBackend(),
        mode="online",
        timeline_store=timeline,
        snapshot_store=store,
        internet_check=lambda: True,
    )

    agent._execute_tool("edit_file", {"path": "app.py", "new_string": "new\n"})

    events = timeline.recent(project=project_dir)
    snapshot_event = next(event for event in events if event.event_type == "snapshot")
    result_event = next(event for event in events if event.event_type == "tool_result")
    snapshot = store.latest_applied()

    assert snapshot is not None
    assert snapshot_event.details["snapshot_id"] == snapshot.id
    assert result_event.details["snapshot_id"] == snapshot.id
