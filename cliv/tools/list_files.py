import os
from .base import BaseTool

class ListFilesTool(BaseTool):
    def __init__(self):
        self.name = "list_files"
        self.description = "List all files and directories in the specified path"
        self.input_schema = {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The directory path to list"}
            },
            "required": [],
        }

    def execute(self, path: str = ".", **kwargs) -> str:
        try:
            if not os.path.exists(path):
                return f"Path not found: {path}"
            items = []
            for item in sorted(os.listdir(path)):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    items.append(f"[DIR]  {item}/")
                else:
                    items.append(f"[FILE] {item}")
            if not items:
                return f"Empty directory: {path}"
            return f"Contents of {path}:\n" + "\n".join(items)
        except Exception as e:
            return f"Error listing files: {str(e)}"
