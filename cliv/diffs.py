"""Small helpers for showing safe, readable previews before writes."""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from cliv.risk import RiskAssessment, assess_changes
from cliv.security import project_root


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


def preview_changes(
    changes: Sequence[Tuple[Path, str, str]],
    *,
    root: Optional[Path] = None,
    notes: Iterable[str] = (),
) -> str:
    """Build a risk-aware combined preview for one or more edits."""

    change_list: List[Tuple[Path, str, str]] = list(changes)
    note_list = list(notes)
    if not change_list:
        return "\n".join(note_list) if note_list else "No changes would be made."

    assessment: RiskAssessment = assess_changes(change_list, root=root)
    lines: List[str] = [assessment.render(), ""]
    lines.extend(note_list)
    if note_list:
        lines.append("")

    base = (root or project_root()).resolve()
    for path, old, new in change_list:
        try:
            label = str(path.resolve().relative_to(base)).replace("\\", "/")
        except (OSError, ValueError):
            label = str(path)
        lines.append(unified_diff(old, new, label))
    return truncate("\n".join(lines))


def preview_edit(
    path: Path,
    old: str,
    new: str,
    *,
    root: Optional[Path] = None,
) -> str:
    return preview_changes([(path, old, new)], root=root)
