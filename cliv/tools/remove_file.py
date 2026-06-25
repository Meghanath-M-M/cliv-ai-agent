"""Tool to remove/delete files on disk."""

from pathlib import Path
from cliv.tools.base import BaseTool


class RemoveFileTool(BaseTool):
    name = "remove_file"
    description = "Delete/remove a file from disk. Requires approval."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative or absolute path to the file to delete",
            }
        },
        "required": ["path"],
    }

    def execute(self, path: str, **kwargs) -> str:
        try:
            file_path = Path(path).expanduser().resolve()

            # Security: prevent deleting outside home or current project
            home = Path.home().resolve()
            cwd = Path.cwd().resolve()
            allowed_roots = {home, cwd}
            if not any(str(file_path).startswith(str(r)) for r in allowed_roots):
                return f"Error: Deleting outside allowed directories is blocked: {path}"

            if not file_path.exists():
                return f"Error: File not found: {path}"
            if not file_path.is_file():
                return f"Error: Path is not a file: {path}"

            file_path.unlink()
            return f"Successfully removed {path}"

        except Exception as e:
            return f"Error removing file: {e}"
