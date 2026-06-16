from typing import Dict, Any

class BaseTool:
    """Base interface for all AI Tools"""
    name: str
    description: str
    input_schema: Dict[str, Any]

    def execute(self, *args: Any, **kwargs: Any ) -> str:
        raise NotImplementedError("Tools must implement the execute method")
