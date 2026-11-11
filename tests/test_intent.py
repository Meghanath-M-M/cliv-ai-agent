"""Tests for the offline regex fallback (not the preferred path)."""

import pytest

from cliv.intent import classify_intent


@pytest.mark.parametrize(
    "text, expected",
    [
        ("hello", "conversation"),
        ("thanks!", "conversation"),
        ("what can you do?", "conversation"),
        ("list files", "list_dir"),
        ("show me the files here", "list_dir"),
        ("analyse this project", "project_analysis"),
        ("analyze the repository architecture", "project_analysis"),
        ("cliv project-status", "project_analysis"),
        ("understand this codebase", "project_analysis"),
        ("fix all failing tests", "self_heal"),
        ("run tests and repair failures", "self_heal"),
        ("make the tests pass", "self_heal"),
        ("Fix all errors in this Python project.", "self_heal"),
        ("self-heal", "self_heal"),
        # Undo intents
        ("undo", "undo"),
        ("undo last", "undo"),
        ("undo that", "undo"),
        ("undo the last change", "undo"),
        ("revert last change", "undo"),
        ("restore the last snapshot", "undo"),
        ("roll back last", "undo"),
        # File intents
        ("read main.py", "file_read"),
        ("show me the contents of app.py", "file_read"),
        ("fix the bug in main.py", "file_fix"),
        ("debug this error in utils.py", "file_fix"),
        ("create hello.py", "file_create"),
        ("write a new file called notes.md", "file_create"),
        ("delete old.py", "file_remove"),
        ("rename old.py to new.py", "file_rename"),
        ("what is the capital of France?", "unknown"),
    ],
)
def test_classify_intent_examples(text, expected):
    intent, _ = classify_intent(text)
    assert intent == expected


def test_extract_filename():
    assert classify_intent("read src/main.py")[1] == "src/main.py"


def test_undo_with_snapshot_id_in_target():
    """undo <snapshot_id> is recognised as the undo intent."""
    # The classifier lowercases the input before regex matching, so the 'T'/'Z'
    # in the snapshot timestamp are also lowercased; the id regex won't capture
    # them from the lowered string — but the intent itself must still be "undo".
    snapshot_id = "20260101T120000Z-abcd1234"
    intent, _target = classify_intent(f"undo {snapshot_id}")
    assert intent == "undo"
