"""CLI tests for listing and undoing safety snapshots."""

import sys

from cliv import cli
from cliv.snapshots import SnapshotStore


def test_cli_lists_and_undoes_last_snapshot(project_dir, monkeypatch, capsys):
    target = project_dir / "app.py"
    target.write_text("old\n")
    store = SnapshotStore()
    snapshot = store.capture([target], tool="edit_file", project=project_dir)
    target.write_text("new\n")
    store.mark_applied(snapshot.id, project=project_dir)

    monkeypatch.setattr(sys, "argv", ["cliv", "snapshots"])
    cli.main()
    listing = capsys.readouterr().out
    assert "CLIV SAFETY SNAPSHOTS" in listing
    assert snapshot.id in listing

    monkeypatch.setattr(sys, "argv", ["cliv", "undo", "--yes"])
    cli.main()
    output = capsys.readouterr().out
    assert "Restored 1 file(s)" in output
    assert target.read_text() == "old\n"
    assert store.get(snapshot.id).status == "rolled_back"


def test_cli_undo_with_no_snapshot_is_friendly(project_dir, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["cliv", "undo", "--yes"])

    cli.main()

    assert "No applied snapshots to undo" in capsys.readouterr().out


def test_cli_force_undo_overwrites_later_edit(project_dir, monkeypatch, capsys):
    target = project_dir / "app.py"
    target.write_text("original\n")
    store = SnapshotStore()
    snapshot = store.capture([target], tool="edit_file", project=project_dir)
    target.write_text("first edit\n")
    store.mark_applied(snapshot.id, project=project_dir)
    target.write_text("later edit\n")

    monkeypatch.setattr(sys, "argv", ["cliv", "undo", "--yes", "--force"])
    cli.main()

    assert "Restored 1 file(s)" in capsys.readouterr().out
    assert target.read_text() == "original\n"
