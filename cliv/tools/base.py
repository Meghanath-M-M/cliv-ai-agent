"""Base class for all cliv tools."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseTool(ABC):
    """Abstract base class for dynamically-loaded tools."""

    name: str = ""
    description: str = ""
    input_schema: Dict[str, Any] = {}
    requires_approval: bool = False

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool and return a string result."""

    def preview(self, tool_input: Dict[str, Any]) -> Optional[str]:
        """Return a human-readable preview shown before approval.

        Tools that mutate state should override this.  Returning ``None``
        means the default approval prompt (tool name and arguments) is
        sufficient.
        """

        return None
