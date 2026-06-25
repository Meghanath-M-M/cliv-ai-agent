"""Tool to write or patch files on disk."""

import re
from pathlib import Path
from cliv.tools.base import BaseTool


class EditFileTool(BaseTool):
    name = "edit_file"
    description = (
        "Write new content to a file, or replace a specific string within a file. "
        "If 'old_string' is provided, only that substring is replaced with 'new_string'. "
        "If 'old_string' is omitted, the entire file is overwritten with 'new_string'."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative or absolute path to the file",
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

    def execute(
        self, path: str, new_string: str, old_string: str = "", **kwargs
    ) -> str:
        try:
            file_path = Path(path).expanduser().resolve()

            # Security: prevent writing outside home or current project
            home = Path.home().resolve()
            cwd = Path.cwd().resolve()
            allowed_roots = {home, cwd}
            if not any(str(file_path).startswith(str(r)) for r in allowed_roots):
                return f"Error: Writing outside allowed directories is blocked: {path}"

            if old_string:
                # Patch mode
                if not file_path.exists():
                    return f"Error: File not found for patching: {path}"
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                if old_string not in content:
                    return f"Error: old_string not found in file. File unchanged."
                new_content = content.replace(old_string, new_string, 1)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                return f"Successfully patched {path}"
            else:
                # Overwrite mode
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_string)
                return f"Successfully wrote {path}"

        except Exception as e:
            return f"Error editing file: {e}"
