"""Tool to list directory contents."""

from pathlib import Path
from cliv.tools.base import BaseTool


class ListFilesTool(BaseTool):
    name = "list_files"
    description = "List the contents of a directory"
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative or absolute path to the directory (default: current)",
            }
        },
        "required": [],
    }

    def execute(self, path: str = ".", **kwargs) -> str:
        try:
            dir_path = Path(path).expanduser().resolve()
            if not dir_path.exists():
                return f"Error: Directory not found: {path}"
            if not dir_path.is_dir():
                return f"Error: Path is not a directory: {path}"

            entries = []
            for entry in sorted(dir_path.iterdir()):
                prefix = "[DIR]" if entry.is_dir() else "[FILE]"
                size = ""
                if entry.is_file():
                    size_bytes = entry.stat().st_size
                    if size_bytes < 1024:
                        size = f" ({size_bytes}B)"
                    elif size_bytes < 1024 * 1024:
                        size = f" ({size_bytes / 1024:.1f}KB)"
                    else:
                        size = f" ({size_bytes / (1024 * 1024):.1f}MB)"
                entries.append(f"{prefix} {entry.name}{size}")

            return "\n".join(entries) if entries else "(empty directory)"
        except Exception as e:
            return f"Error listing directory: {e}"
