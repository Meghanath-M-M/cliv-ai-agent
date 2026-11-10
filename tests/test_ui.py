"""Tests for the terminal approval boundary."""

from io import StringIO

from rich.console import Console

from cliv.diffs import preview_edit
from cliv.ui import ConsoleUI


def test_console_ui_renders_code_with_brackets_without_markup_errors(
    project_dir, monkeypatch
):
    buffer = StringIO()
    console = Console(
        file=buffer, width=80, no_color=True, force_terminal=False
    )
    monkeypatch.setattr("builtins.input", lambda: "n")
    preview = preview_edit(
        project_dir / "app.py",
        "value = [1]\n",
        "value = [2]\n",
    )

    approved = ConsoleUI(console).confirm(
        "edit_file",
        {"path": "app.py", "new_string": "value = [2]\n"},
        preview,
    )

    output = buffer.getvalue()
    assert approved is False
    assert "Approval Required" in output
    assert "value = [1]" in output
    assert "value = [2]" in output
