"""Base class for all cliv tools."""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseTool(ABC):
    """Abstract base class for dynamically-loaded tools."""

    name: str = ""
    description: str = ""
    input_schema: Dict[str, Any] = {}

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool and return a string result."""
        pass
