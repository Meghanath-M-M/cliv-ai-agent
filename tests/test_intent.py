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
