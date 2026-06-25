"""Tool to read file contents."""

from pathlib import Path
from cliv.tools.base import BaseTool


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read the contents of a file at the specified path"
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative or absolute path to the file",
            }
        },
        "required": ["path"],
    }

    def execute(self, path: str, **kwargs) -> str:
        try:
            file_path = Path(path).expanduser().resolve()
            if not file_path.exists():
                return f"Error: File not found: {path}"
            if not file_path.is_file():
                return f"Error: Path is not a file: {path}"
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            # Cap extremely large files
            if len(content) > 100_000:
                content = content[:100_000] + "\n\n[...truncated: file exceeds 100KB]"
            return content
        except Exception as e:
            return f"Error reading file: {e}"
