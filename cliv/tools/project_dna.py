"""Read-only Project DNA analysis tool."""

from __future__ import annotations

from typing import Any, Dict

from cliv.intelligence import analyse_project
from cliv.security import resolve_project_path
from cliv.tools.base import BaseTool


class ProjectDNATool(BaseTool):
    name = "analyse_project"
    description = (
        "Analyse the repository's architecture, languages, components, symbols, "
        "imports, dependency graph, entry points, and structural issues. "
        "Use this to understand an unfamiliar codebase before changing it. "
        "Read-only and safe to run without approval."
    )
    permissions = frozenset({"read"})
    risk_level = "low"
    input_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Project directory to analyse, relative to the project root. "
                    "Defaults to the project root."
                ),
            }
        },
        "required": [],
    }

    def execute(self, path: str = ".", **kwargs) -> str:
        root, error = resolve_project_path(path or ".")
        if error:
            return error
        assert root is not None
        if not root.exists():
            return f"Error: Directory not found: {path}"
        if not root.is_dir():
            return f"Error: Path is not a directory: {path}"

        try:
            return analyse_project(root)
        except Exception as exc:  # noqa: BLE001 - report a tool failure, never crash the agent
            return f"Error: Project analysis failed: {exc}"
