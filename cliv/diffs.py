"""Small helpers for showing safe, readable previews before writes."""

from __future__ import annotations

import difflib
from pathlib import Path


def unified_diff(old: str, new: str, label: str, context: int = 3) -> str:
    """Return a unified diff, or a short note when nothing would change."""

    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = "".join(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{label}",
            tofile=f"b/{label}",
            n=context,
        )
    )
    if not diff:
        return f"No changes for {label}."
    return diff


def truncate(text: str, limit: int = 6000) -> str:
    """Keep approval previews from flooding the terminal."""

    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [diff truncated] ..."


def preview_edit(path: Path, old: str, new: str) -> str:
    return truncate(unified_diff(old, new, str(path)))
