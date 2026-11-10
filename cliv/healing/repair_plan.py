"""Validate the model's JSON repair plan before any file is changed."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

from cliv.security import resolve_project_path
from cliv.text import strip_markdown_fences

MAX_EDITS = 5
MAX_REPAIR_CHARS = 200_000


@dataclass(frozen=True)
class RepairEdit:
    path: str
    old_string: str
    new_string: str


def _extract_json(text: str) -> Any:
    cleaned = strip_markdown_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    if start == -1:
        raise ValueError("response did not contain a JSON object")
    try:
        value, _ = json.JSONDecoder().raw_decode(cleaned[start:])
        return value
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc


def _raw_edits(payload: Any) -> List[Any]:
    if not isinstance(payload, dict):
        raise ValueError("repair plan must be a JSON object")

    edits = payload.get("edits")
    if edits is None:
        files = payload.get("files")
        if isinstance(files, list):
            return [
                {
                    "path": item.get("path", ""),
                    "new_string": item.get("content", ""),
                }
                for item in files
                if isinstance(item, dict)
            ]
        raise ValueError("repair plan is missing an 'edits' array")

    if not isinstance(edits, list) or not edits:
        raise ValueError("'edits' must be a non-empty array")
    return edits


def parse_repair_plan(
    text: str, root: Path
) -> Tuple[Optional[List[RepairEdit]], Optional[str]]:
    """Return ``(edits, error)`` for a model-generated repair plan."""

    try:
        payload = _extract_json(text)
        raw_edits = _raw_edits(payload)
    except ValueError as exc:
        return None, str(exc)

    if len(raw_edits) > MAX_EDITS:
        return None, f"repair plan exceeds the {MAX_EDITS}-edit limit"

    edits: List[RepairEdit] = []
    total_chars = 0

    for index, item in enumerate(raw_edits, start=1):
        if not isinstance(item, dict):
            return None, f"edit #{index} must be an object"

        path = item.get("path")
        if not isinstance(path, str) or not path.strip():
            return None, f"edit #{index} is missing a valid path"

        new_string = item.get("new_string", item.get("content"))
        if not isinstance(new_string, str):
            return None, f"edit #{index} is missing a valid new_string"

        old_string = item.get("old_string", "")
        if old_string is None:
            old_string = ""
        if not isinstance(old_string, str):
            return None, f"edit #{index} has an invalid old_string"

        total_chars += len(new_string)
        if total_chars > MAX_REPAIR_CHARS:
            return None, "repair plan is too large"

        resolved, error = resolve_project_path(
            path, for_write=True, root=root
        )
        if error:
            return None, error
        assert resolved is not None
        if resolved.exists() and not resolved.is_file():
            return None, f"edit path is not a file: {path}"

        relative = resolved.relative_to(root.resolve()).as_posix()
        if old_string and old_string == new_string:
            continue
        if not old_string and resolved.is_file():
            try:
                current = resolved.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                current = None
            if current == new_string:
                continue

        edits.append(
            RepairEdit(
                path=relative,
                old_string=old_string,
                new_string=new_string,
            )
        )

    if not edits:
        return None, "repair plan contains no effective changes"
    return edits, None
