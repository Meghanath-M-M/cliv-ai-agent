"""Risk-aware, syntax-highlighted approval previews.

Plain unified diff text remains the transport format between tools and the
UI.  This module only handles presentation, so tests and non-interactive
callers keep receiving stable text.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence, Tuple

from rich.console import Console
from rich.text import Text

try:  # pragma: no cover - exercised when Pygments is unavailable
    from pygments import lex
    from pygments.lexers import get_lexer_for_filename
    from pygments.token import Token
except Exception:  # noqa: BLE001
    lex = None  # type: ignore[assignment]
    get_lexer_for_filename = None  # type: ignore[assignment]
    Token = None  # type: ignore[assignment]


RISK_LINE = re.compile(r"(?:RISK:|\[risk\])\s*(LOW|MEDIUM|HIGH|CRITICAL)", re.I)
HUNK_HEADER = re.compile(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")

RISK_STYLES = {
    "low": "bold green",
    "medium": "bold yellow",
    "high": "bold red",
    "critical": "bold white on red",
}


def _risk_style(level: str) -> str:
    return RISK_STYLES.get(level.lower(), "bold yellow")


def _risk_level(line: str) -> Optional[str]:
    match = RISK_LINE.search(line)
    return match.group(1).lower() if match else None


def _lexer_for(filename: str):
    if get_lexer_for_filename is None:
        return None
    try:
        return get_lexer_for_filename(filename, stripnl=False)
    except Exception:  # noqa: BLE001
        return None


def _style_for_token(token) -> str:
    if Token is None:
        return ""
    if token in Token.Comment:
        return "italic dim"
    if token in Token.Keyword:
        return "bold magenta"
    if token in Token.Name.Function:
        return "bold cyan"
    if token in Token.Name.Class:
        return "bold yellow"
    if token in Token.Name.Builtin:
        return "bright_blue"
    if token in Token.Name.Decorator:
        return "yellow"
    if token in Token.String:
        return "green"
    if token in Token.Number:
        return "bright_magenta"
    if token in Token.Operator:
        return "cyan"
    return ""


def _highlight(text: str, filename: str, *, base_style: str = "") -> Text:
    result = Text(style=base_style)
    if not text:
        return result
    lexer = _lexer_for(filename)
    if lexer is None or lex is None:
        result.append(text)
        return result
    try:
        for token, value in lex(text, lexer):
            value = value.replace("\n", "")
            if value:
                result.append(value, style=_style_for_token(token))
    except Exception:  # noqa: BLE001 - highlighting must never block approval
        result = Text(text, style=base_style)
    return result


def _clip(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def _split_sections(lines: Sequence[str]) -> Tuple[List[str], List[List[str]]]:
    starts = [
        index
        for index in range(len(lines) - 1)
        if lines[index].startswith("--- ") and lines[index + 1].startswith("+++ ")
    ]
    if not starts:
        return list(lines), []
    meta = list(lines[: starts[0]])
    sections: List[List[str]] = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        sections.append(list(lines[start:end]))
    return meta, sections


def _render_meta(lines: Iterable[str], console: Console) -> None:
    for line in lines:
        level = _risk_level(line)
        if level is not None:
            console.print(Text(line, style=_risk_style(level)))
        elif line.startswith("Why:"):
            console.print(Text(line, style="bold"))
        elif line.startswith("  - "):
            console.print(Text(line, style="dim"))
        elif line:
            console.print(Text(line))
        else:
            console.print()


def _filename(section: Sequence[str]) -> str:
    for line in section:
        if line.startswith("+++ b/"):
            return line[6:]
    return "file"


def _render_unified(sections: Sequence[Sequence[str]], console: Console) -> None:
    for section in sections:
        filename = _filename(section)
        for line in section:
            if line.startswith("--- ") or line.startswith("+++ "):
                console.print(Text(line, style="bold"))
            elif line.startswith("@@"):
                console.print(Text(line, style="cyan"))
            elif line.startswith("\\ No newline"):
                console.print(Text(line, style="dim"))
            elif line.startswith("+") and not line.startswith("+++"):
                text = Text("+", style="bold green")
                text.append_text(_highlight(line[1:], filename, base_style="green"))
                console.print(text)
            elif line.startswith("-") and not line.startswith("---"):
                text = Text("-", style="bold red")
                text.append_text(_highlight(line[1:], filename, base_style="red"))
                console.print(text)
            elif line.startswith(" "):
                text = Text(" ", style="dim")
                text.append_text(_highlight(line[1:], filename, base_style="dim"))
                console.print(text)
            else:
                console.print(Text(line))
        console.print()


def _render_side_row(
    console: Console,
    old_number: Optional[int],
    old_text: str,
    new_number: Optional[int],
    new_text: str,
    *,
    filename: str,
    old_marker: str = "-",
    new_marker: str = "+",
) -> None:
    side_width = max(20, (console.width - 3) // 2)
    number_width = 5  # four digits plus one space
    marker_width = 2  # marker plus one space
    code_width = max(1, side_width - number_width - marker_width)

    def side(number: Optional[int], text: str, marker: str, style: str) -> Text:
        if number is None:
            return Text(" " * side_width, style="dim")
        rendered = Text()
        rendered.append(f"{number:>4} ", style="dim")
        rendered.append(marker, style=f"bold {style}")
        rendered.append(" ")
        rendered.append_text(
            _highlight(_clip(text, code_width), filename, base_style=style)
        )
        padding = side_width - rendered.cell_len
        if padding > 0:
            rendered.append(" " * padding)
        return rendered

    left = side(old_number, old_text, old_marker, "red")
    right = side(new_number, new_text, new_marker, "green")
    console.print(Text.assemble(left, Text(" │ ", style="dim"), right))


def _render_side_by_side(sections: Sequence[Sequence[str]], console: Console) -> None:
    for section in sections:
        filename = _filename(section)
        console.print(Text(f" {filename}", style="bold"))
        old_number = 1
        new_number = 1
        removed: List[Tuple[int, str]] = []
        added: List[Tuple[int, str]] = []

        def flush() -> None:
            nonlocal removed, added
            for index in range(max(len(removed), len(added))):
                left = removed[index] if index < len(removed) else (None, "")
                right = added[index] if index < len(added) else (None, "")
                _render_side_row(
                    console,
                    left[0],
                    left[1],
                    right[0],
                    right[1],
                    filename=filename,
                )
            removed, added = [], []

        for line in section[2:]:
            if line.startswith("@@"):
                flush()
                match = HUNK_HEADER.match(line)
                if match:
                    old_number = int(match.group(1))
                    new_number = int(match.group(2))
                console.print(Text(line, style="cyan"))
            elif line.startswith("\\ No newline"):
                flush()
                console.print(Text(line, style="dim"))
            elif line.startswith("-") and not line.startswith("---"):
                removed.append((old_number, line[1:]))
                old_number += 1
            elif line.startswith("+") and not line.startswith("+++"):
                added.append((new_number, line[1:]))
                new_number += 1
            elif line.startswith(" "):
                flush()
                _render_side_row(
                    console,
                    old_number,
                    line[1:],
                    new_number,
                    line[1:],
                    filename=filename,
                    old_marker=" ",
                    new_marker=" ",
                )
                old_number += 1
                new_number += 1
            else:
                flush()
                console.print(Text(line))
        flush()
        console.print()


def render_preview(preview: str, console: Console) -> None:
    """Render a plain preview string with risk and diff styling.

    A side-by-side view is used only when the terminal is wide enough; the
    unified view remains the fallback for narrow terminals and CI logs.
    """

    lines = preview.splitlines()
    meta, sections = _split_sections(lines)
    _render_meta(meta, console)
    if not sections:
        return
    if console.width >= 120:
        _render_side_by_side(sections, console)
    else:
        _render_unified(sections, console)
