"""Tool to remove/delete files on disk."""

from pathlib import Path
from typing import Any, Dict, Optional

from cliv.risk import RiskAssessment, assess_delete
from cliv.security import resolve_project_path
from cliv.tools.base import BaseTool


class RemoveFileTool(BaseTool):
    name = "remove_file"
    description = "Delete/remove a file from disk. Requires approval."
    requires_approval = True
    permissions = frozenset({"read", "write"})
    risk_level = "high"
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

    def affected_paths(self, tool_input: Dict[str, Any]) -> list[Path]:
        resolved, error = resolve_project_path(
            str(tool_input.get("path", "")), for_write=True
        )
        if resolved is None or error or not resolved.exists():
            return []
        return [resolved]

    def preview(self, tool_input: Dict[str, Any]) -> Optional[str]:
        path = str(tool_input.get("path", ""))
        resolved, error = resolve_project_path(path, for_write=True)
        if error:
            return error
        assert resolved is not None
        if not resolved.exists():
            return f"Error: File not found: {path}"
        risk = assess_delete(resolved)
        return f"{risk.render()}\nWill permanently delete: {resolved}"

    def assess_risk(self, tool_input: Dict[str, Any]) -> Optional[RiskAssessment]:
        resolved, error = resolve_project_path(
            str(tool_input.get("path", "")), for_write=True
        )
        if error or resolved is None:
            return None
        return assess_delete(resolved)

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
