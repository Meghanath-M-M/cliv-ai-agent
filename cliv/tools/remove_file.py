"""Tool to remove/delete files on disk."""

from typing import Any, Dict, Optional

from cliv.security import resolve_project_path
from cliv.tools.base import BaseTool


class RemoveFileTool(BaseTool):
    name = "remove_file"
    description = "Delete/remove a file from disk. Requires approval."
    requires_approval = True
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to delete, relative to the project root",
            }
        },
        "required": ["path"],
    }

    def preview(self, tool_input: Dict[str, Any]) -> Optional[str]:
        path = str(tool_input.get("path", ""))
        resolved, error = resolve_project_path(path, for_write=True)
        if error:
            return error
        assert resolved is not None
        if not resolved.exists():
            return f"Error: File not found: {path}"
        return f"Will permanently delete: {resolved}"

    def execute(self, path: str, **kwargs) -> str:
        file_path, error = resolve_project_path(path, for_write=True)
        if error:
            return error
        assert file_path is not None

        try:
            if not file_path.exists():
                return f"Error: File not found: {path}"
            if not file_path.is_file():
                return f"Error: Path is not a file: {path}"
            file_path.unlink()
            return f"Successfully removed {path}"
        except Exception as exc:  # noqa: BLE001
            return f"Error removing file: {exc}"
