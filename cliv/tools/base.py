"""Base class for all cliv tools."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional


class BaseTool(ABC):
    """Abstract base class for dynamically-loaded tools."""

    name: str = ""
    description: str = ""
    input_schema: Dict[str, Any] = {}
    requires_approval: bool = False

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool and return a string result."""

    def affected_paths(self, tool_input: Dict[str, Any]) -> List[Path]:
        """Return files that should be snapshotted before execution.

        Read-only and command-running tools leave this empty.  Mutating file
        tools override it so the agent can create a rollback point without
        knowing each tool's argument shape.
        """

        return []

    def preview(self, tool_input: Dict[str, Any]) -> Optional[str]:
        """Return a human-readable preview shown before approval.

        Tools that mutate state should override this.  Returning ``None``
        means the default approval prompt (tool name and arguments) is
        sufficient.
        """

        return None
