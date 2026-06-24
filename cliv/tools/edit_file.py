import os
from typing import Any
from .base import BaseTool
from rich.console import Console

console = Console()

class EditFileTool(BaseTool):
    def __init__(self):
        self.name = "edit_file"
        self.description = "Edit a file by replacing old_text with new_text. Creates the file if it doesn't exist."
        self.input_schema = {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to edit",
                },
                "old_text": {
                    "type": "string",
                    "description": "The text to search for and replace (leave empty to create)",
                },
                "new_text": {
                    "type": "string",
                    "description": "The text to replace old_text with",
                },
            },
            "required": ["path", "new_text"],
        }

    def execute(self, path: str, new_text: str, old_text: str = "", *args: Any, **kwargs: Any) -> str:
        console.print(
            f"\n[bold yellow][SYSTEM WARNING][/bold yellow] The AI wants to modify the file: '{path}'"
        )
        confirm = input("Allow this change? [y/N]: ").strip().lower()
        if confirm != "y":
            return f"Operation blocked: User denied permission to edit {path}."

        try:
            if os.path.exists(path) and old_text:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                if old_text not in content:
                    return f"Text not found in file: {old_text}"
                content = content.replace(old_text, new_text)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                return f"Successfully edited {path}"
            else:
                dir_name = os.path.dirname(path)
                if dir_name:
                    os.makedirs(dir_name, exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(new_text)
                return f"Successfully created {path}"
        except Exception as e:
            return f"Error editing file: {str(e)}"
