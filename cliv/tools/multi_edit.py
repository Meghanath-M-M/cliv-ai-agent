"""Apply multiple file edits after showing one combined diff."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cliv.diffs import preview_changes
from cliv.security import resolve_project_path
from cliv.tools.base import BaseTool


class MultiEditTool(BaseTool):
    name = "multi_edit"
    description = (
        "Apply several file edits atomically. Shows a combined diff before approval. "
        "Each edit needs a path and new_string; old_string makes it a patch instead "
        "of a full overwrite."
    )
    requires_approval = True
    input_schema = {
        "type": "object",
        "properties": {
            "edits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old_string": {"type": "string"},
                        "new_string": {"type": "string"},
                    },
                    "required": ["path", "new_string"],
                },
                "description": "Edits to apply; paths are relative to the project root",
            }
        },
        "required": ["edits"],
    }

    def _prepare(
        self, edits: Any
    ) -> Tuple[Optional[Dict[Path, Tuple[bool, str, str]]], Optional[str]]:
        if not isinstance(edits, list) or not edits:
            return None, "Error: edits must be a non-empty list."

        # path -> (existed, original content, current content)
        state: Dict[Path, Tuple[bool, str, str]] = {}
        for index, edit in enumerate(edits):
            if not isinstance(edit, dict):
                return None, f"Error: edit #{index + 1} must be an object."
            path_value = str(edit.get("path", ""))
            if not path_value:
                return None, f"Error: edit #{index + 1} is missing a path."
            if "new_string" not in edit:
                return None, f"Error: edit #{index + 1} is missing new_string."

            resolved, error = resolve_project_path(path_value, for_write=True)
            if error:
                return None, error
            assert resolved is not None

            if resolved not in state:
                existed = resolved.exists()
                if existed:
                    try:
                        original = resolved.read_text(encoding="utf-8")
                    except Exception as exc:  # noqa: BLE001
                        return None, f"Error reading {path_value}: {exc}"
                else:
                    original = ""
                state[resolved] = (existed, original, original)

            existed, original, current = state[resolved]
            old_string = str(edit.get("old_string", "") or "")
            new_string = str(edit.get("new_string", ""))

            if old_string:
                if not existed:
                    return None, f"Error: File not found for patching: {path_value}"
                if old_string not in current:
                    return (
                        None,
                        f"Error: old_string not found in {path_value}. Nothing applied.",
                    )
                current = current.replace(old_string, new_string, 1)
            else:
                current = new_string

            state[resolved] = (existed, original, current)

        return state, None

    def affected_paths(self, tool_input: Dict[str, Any]) -> List[Path]:
        edits = tool_input.get("edits")
        if not isinstance(edits, list):
            return []
        paths: List[Path] = []
        for edit in edits:
            if not isinstance(edit, dict):
                continue
            resolved, error = resolve_project_path(
                str(edit.get("path", "")), for_write=True
            )
            if resolved is not None and not error and resolved not in paths:
                paths.append(resolved)
        return paths

    def preview(self, tool_input: Dict[str, Any]) -> Optional[str]:
        state, error = self._prepare(tool_input.get("edits"))
        if error:
            return error
        assert state is not None

        changes: List[Tuple[Path, str, str]] = []
        notes: List[str] = []
        for path, (existed, original, current) in state.items():
            if original == current:
                if not existed:
                    notes.append(f"Create empty file: {path}")
                continue
            changes.append((path, original, current))
        return preview_changes(changes, notes=notes)

    def execute(self, edits: Any = None, **kwargs) -> str:
        state, error = self._prepare(edits)
        if error:
            return error
        assert state is not None

        written: List[Tuple[Path, bool, str]] = []
        try:
            for path, (existed, original, current) in state.items():
                if original == current and existed:
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(current, encoding="utf-8")
                written.append((path, existed, original))
            return f"Successfully applied {len(written)} edit(s)."
        except Exception as exc:  # noqa: BLE001
            # Best-effort rollback so a multi-file edit is not half-applied.
            for path, existed, original in reversed(written):
                try:
                    if existed:
                        path.write_text(original, encoding="utf-8")
                    elif path.exists():
                        path.unlink()
                except Exception:  # noqa: BLE001
                    pass
            return f"Error applying edits (rolled back): {exc}"
