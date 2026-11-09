"""Tool to write or patch files on disk."""

from pathlib import Path
from typing import Any, Dict, Optional

from cliv.diffs import preview_edit
from cliv.security import resolve_project_path
from cliv.tools.base import BaseTool


class EditFileTool(BaseTool):
    name = "edit_file"
    description = (
        "Write new content to a file, or replace a specific string within a file. "
        "If 'old_string' is provided, only that substring is replaced with 'new_string'. "
        "If 'old_string' is omitted, the entire file is overwritten with 'new_string'."
    )
    requires_approval = True
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file, relative to the project root",
            },
            "old_string": {
                "type": "string",
                "description": "The existing text to replace (omit to overwrite entire file)",
            },
            "new_string": {
                "type": "string",
                "description": "The new text to write or insert",
            },
        },
        "required": ["path", "new_string"],
    }

    def _target(self, path: str):
        return resolve_project_path(path, for_write=True)

    def preview(self, tool_input: Dict[str, Any]) -> Optional[str]:
        path = str(tool_input.get("path", ""))
        new_string = str(tool_input.get("new_string", ""))
        old_string = str(tool_input.get("old_string", ""))
        resolved, error = self._target(path)
        if error:
            return error

        assert resolved is not None
        if old_string:
            if not resolved.exists():
                return f"Error: File not found for patching: {path}"
            try:
                content = resolved.read_text(encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                return f"Error reading file: {exc}"
            if old_string not in content:
                return "Error: old_string not found in file. File unchanged."
            updated = content.replace(old_string, new_string, 1)
        else:
            if resolved.exists() and resolved.is_file():
                try:
                    content = resolved.read_text(encoding="utf-8")
                except Exception:  # noqa: BLE001 - binary preview is best effort
                    content = ""
            else:
                content = ""
            updated = new_string

        return preview_edit(resolved, content, updated)

    def execute(
        self, path: str, new_string: str, old_string: str = "", **kwargs
    ) -> str:
        file_path, error = self._target(path)
        if error:
            return error
        assert file_path is not None

        try:
            if old_string:
                if not file_path.exists():
                    return f"Error: File not found for patching: {path}"
                content = file_path.read_text(encoding="utf-8")
                if old_string not in content:
                    return "Error: old_string not found in file. File unchanged."
                new_content = content.replace(old_string, new_string, 1)
                file_path.write_text(new_content, encoding="utf-8")
                return f"Successfully patched {path}"

            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(new_string, encoding="utf-8")
            return f"Successfully wrote {path}"
        except Exception as exc:  # noqa: BLE001
            return f"Error editing file: {exc}"
