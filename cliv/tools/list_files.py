"""Tool to list directory contents."""

from cliv.security import is_sensitive_path, resolve_project_path
from cliv.tools.base import BaseTool


class ListFilesTool(BaseTool):
    name = "list_files"
    description = "List the contents of a directory inside the project"
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the directory relative to the project root (default: current)",
            }
        },
        "required": [],
    }

    def execute(self, path: str = ".", **kwargs) -> str:
        dir_path, error = resolve_project_path(path)
        if error:
            return error
        assert dir_path is not None

        try:
            if not dir_path.exists():
                return f"Error: Directory not found: {path}"
            if not dir_path.is_dir():
                return f"Error: Path is not a directory: {path}"

            entries = []
            for entry in sorted(dir_path.iterdir()):
                # Never advertise secrets, even when read access is blocked.
                if is_sensitive_path(entry):
                    continue
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
        except Exception as exc:  # noqa: BLE001
            return f"Error listing directory: {exc}"
