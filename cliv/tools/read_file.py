"""Tool to read file contents."""

from cliv.security import resolve_project_path
from cliv.tools.base import BaseTool


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read the contents of a file inside the project directory"
    permissions = frozenset({"read"})
    risk_level = "low"
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file, relative to the project root",
            }
        },
        "required": ["path"],
    }

    def execute(self, path: str, **kwargs) -> str:
        file_path, error = resolve_project_path(path)
        if error:
            return error
        assert file_path is not None

        try:
            if not file_path.exists():
                return f"Error: File not found: {path}"
            if not file_path.is_file():
                return f"Error: Path is not a file: {path}"
            content = file_path.read_text(encoding="utf-8", errors="replace")
            if len(content) > 100_000:
                content = content[:100_000] + "\n\n[...truncated: file exceeds 100KB]"
            return content
        except Exception as exc:  # noqa: BLE001
            return f"Error reading file: {exc}"
