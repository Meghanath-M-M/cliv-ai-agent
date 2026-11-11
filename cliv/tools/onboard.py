"""Agent tool to generate comprehensive project onboarding reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from cliv.onboarding import generate_onboarding_report
from cliv.security import resolve_project_path
from cliv.tools.base import BaseTool


class ProjectOnboardTool(BaseTool):
    name = "project_onboard"
    description = (
        "Generate a comprehensive onboarding report for the repository, including "
        "architecture breakdown, key components, entry points, setup and test instructions, "
        "team conventions, git status, and code health hotspots. Use this whenever "
        "orienting yourself or onboarding new developers to the codebase."
    )
    permissions = frozenset({"read"})
    risk_level = "low"
    input_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Project directory to onboard, relative to the project root. "
                    "Defaults to the project root."
                ),
            },
            "save_to_file": {
                "type": "boolean",
                "description": "If true, saves the generated report to ONBOARDING.md in the project root.",
            },
        },
        "required": [],
    }

    def execute(self, path: str = ".", save_to_file: bool = False, **kwargs) -> str:
        root, error = resolve_project_path(path or ".")
        if error:
            return error
        assert root is not None
        if not root.exists():
            return f"Error: Directory not found: {path}"
        if not root.is_dir():
            return f"Error: Path is not a directory: {path}"

        try:
            report = generate_onboarding_report(root)
            saved_msg = ""
            if save_to_file:
                saved_path = report.save()
                saved_msg = f"\n\n[Report successfully saved to `{saved_path}`]"
            return report.render_markdown() + saved_msg
        except Exception as exc:  # noqa: BLE001
            return f"Error: Failed to generate onboarding report: {exc}"
