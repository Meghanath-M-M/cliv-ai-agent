"""Tests for safety snapshots and time-travel undo."""

from pathlib import Path

import pytest

from cliv.snapshots import (
    SnapshotError,
    SnapshotStore,
    render_snapshot,
    render_snapshots,
)


def make_store(tmp_path) -> SnapshotStore:
    return SnapshotStore(tmp_path / "snapshots")


def test_capture_and_rollback_existing_file(project_dir):
    target = project_dir / "app.py"
    target.write_text("old\n")
    store = make_store(project_dir)

    snapshot = store.capture(
        [target], tool="edit_file", summary="Before edit_file", project=project_dir
    )
    target.write_text("new\n")
    applied = store.mark_applied(snapshot.id, project=project_dir)

    assert applied.status == "applied"
    assert applied.files[0].before_hash != applied.files[0].after_hash

    result = store.rollback(snapshot.id, project=project_dir)

    assert result.success is True
    assert target.read_text() == "old\n"
    assert result.snapshot is not None
    assert result.snapshot.status == "rolled_back"


def test_undo_removes_a_new_file(project_dir):
    target = project_dir / "new.py"
    store = make_store(project_dir)

    snapshot = store.capture([target], tool="edit_file", project=project_dir)
    target.write_text("created\n")
    store.mark_applied(snapshot.id, project=project_dir)

    result = store.rollback(snapshot.id, project=project_dir)

    assert result.success is True
    assert not target.exists()


def test_undo_restores_a_deleted_file(project_dir):
    target = project_dir / "delete_me.txt"
    target.write_text("keep me\n")
    store = make_store(project_dir)

    snapshot = store.capture([target], tool="remove_file", project=project_dir)
    target.unlink()
    store.mark_applied(snapshot.id, project=project_dir)

    result = store.rollback(snapshot.id, project=project_dir)

    assert result.success is True
    assert target.read_text() == "keep me\n"


def test_undo_refuses_to_clobber_later_edits(project_dir):
    target = project_dir / "app.py"
    target.write_text("original\n")
    store = make_store(project_dir)

    snapshot = store.capture([target], tool="edit_file", project=project_dir)
    target.write_text("first edit\n")
    store.mark_applied(snapshot.id, project=project_dir)

    target.write_text("later edit\n")
    result = store.rollback(snapshot.id, project=project_dir)

    assert result.success is False
    assert "changed" in result.message
    assert result.conflicts
    assert target.read_text() == "later edit\n"


def test_force_rollback_overrides_conflict(project_dir):
    target = project_dir / "app.py"
    target.write_text("original\n")
    store = make_store(project_dir)

    snapshot = store.capture([target], tool="edit_file", project=project_dir)
    target.write_text("first edit\n")
    store.mark_applied(snapshot.id, project=project_dir)
    target.write_text("later edit\n")

    result = store.rollback(snapshot.id, project=project_dir, force=True)

    assert result.success is True
    assert target.read_text() == "original\n"


def test_snapshots_are_project_scoped(project_dir, tmp_path):
    other = tmp_path / "other-project"
    other.mkdir()
    (other / "other.txt").write_text("other\n")
    store = make_store(project_dir)
    first = store.capture(
        [project_dir / "a.txt"], tool="edit_file", project=project_dir
    )
    second = store.capture([other / "other.txt"], tool="edit_file", project=other)

    first_ids = [item.id for item in store.recent(project=project_dir)]
    second_ids = [item.id for item in store.recent(project=other)]

    assert first.id in first_ids
    assert second.id in second_ids
    assert first.id not in second_ids


def test_missing_and_invalid_snapshot_ids(project_dir):
    store = make_store(project_dir)

    with pytest.raises(SnapshotError):
        store.get("missing")
    with pytest.raises(SnapshotError):
        store.get("../../escape")


def test_render_snapshot_and_list(project_dir):
    target = project_dir / "app.py"
    target.write_text("old\n")
    store = make_store(project_dir)
    snapshot = store.capture([target], tool="edit_file", project=project_dir)

    rendered = render_snapshot(snapshot)
    listing = render_snapshots(store.recent(project=project_dir), project=project_dir)

    assert "app.py" in rendered
    assert "captured" in rendered
    assert "CLIV SAFETY SNAPSHOTS" in listing
    assert snapshot.id in listing


def test_agent_finish_undo_with_confirmation(project_dir):
    from unittest.mock import patch, MagicMock
    from cliv.agent import AIAgent

    target = project_dir / "file_to_undo.py"
    target.write_text("initial\n")
    store = make_store(project_dir)

    snapshot = store.capture([target], tool="edit_file", project=project_dir)
    target.write_text("modified\n")
    store.mark_applied(snapshot.id, project=project_dir)

    ui = MagicMock()
    agent = AIAgent(api_key=None, snapshot_store=store, ui=ui, auto_approve=False)

    with patch("builtins.input", return_value="y"):
        response = agent._finish_undo(snapshot.id)

    assert "✅" in response or "Restored" in response
    assert target.read_text() == "initial\n"
    assert ui.print.called
