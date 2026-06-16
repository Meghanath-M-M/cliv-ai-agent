from typing import Any
from .base import BaseTool

class ReadFileTool(BaseTool):
    def __init__(self):
        self.name = "read_file"
        self.description = "Read the contents of a file at the specified path"
        self.input_schema = {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The path to the file to read"}
            },
            "required": ["path"],
        }

    def execute(self, path: str, *args: Any, **kwargs: Any) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            return f"File contents of {path}:\n{content}"
        except FileNotFoundError:
            return f"File not found: {path}"
        except Exception as e:
            return f"Error reading file: {str(e)}"
