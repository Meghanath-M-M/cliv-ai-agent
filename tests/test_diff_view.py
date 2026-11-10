"""Tests for risk-aware diff previews and terminal rendering."""

from io import StringIO

from rich.console import Console

from cliv.diff_view import render_preview
from cliv.diffs import preview_edit


OLD = "def add(a, b):\n    return a - b\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"
NEW = "def add(a, b):\n    return a + b\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"


def make_console(width: int = 160) -> tuple[Console, StringIO]:
    buffer = StringIO()
    console = Console(
        file=buffer,
        width=width,
        no_color=True,
        force_terminal=False,
    )
    return console, buffer


def test_preview_contains_risk_header_and_unified_diff(project_dir):
    preview = preview_edit(project_dir / "maths.py", OLD, NEW)

    assert preview.startswith("RISK: MEDIUM")
    assert "Why: modifies source code" in preview
    assert "--- a/" in preview
    assert "-    return a - b" in preview
    assert "+    return a + b" in preview


def test_wide_terminal_renders_side_by_side(project_dir):
    preview = preview_edit(project_dir / "maths.py", OLD, NEW)
    console, buffer = make_console(160)

    render_preview(preview, console)

    output = buffer.getvalue()
    assert "RISK: MEDIUM" in output
    assert "maths.py" in output
    assert "│" in output
    assert "return a - b" in output
    assert "return a + b" in output


def test_narrow_terminal_uses_unified_view(project_dir):
    preview = preview_edit(project_dir / "maths.py", OLD, NEW)
    console, buffer = make_console(80)

    render_preview(preview, console)

    output = buffer.getvalue()
    assert "RISK: MEDIUM" in output
    assert "--- a/" in output
    assert "+++ b/" in output
    assert "│" not in output


def test_non_diff_preview_is_rendered_as_plain_text():
    console, buffer = make_console(80)

    render_preview("Will run:\n  python -m pytest", console)

    output = buffer.getvalue()
    assert "Will run:" in output
    assert "python -m pytest" in output
