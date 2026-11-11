"""Agent tool to execute automated code reviews."""

from __future__ import annotations

from typing import Any, Dict

from cliv.review import run_code_review
from cliv.security import resolve_project_path
from cliv.tools.base import BaseTool


class CodeReviewTool(BaseTool):
    name = "code_review"
    description = (
        "Perform automated code review on modified files or specified paths. "
        "Checks for syntax errors, hardcoded secrets/credentials, dangerous functions, "
        "and project guideline violations. Safe and read-only."
    )
    permissions = frozenset({"read"})
    risk_level = "low"
    input_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Project directory or file to review, relative to project root. "
                    "Defaults to project root."
                ),
            },
            "base_ref": {
                "type": "string",
                "description": "Git base reference to diff against (e.g. 'main', 'origin/main', 'HEAD~1').",
            },
            "format": {
                "type": "string",
                "enum": ["standard", "github"],
                "description": "Markdown output formatting style ('standard' or 'github').",
            },
        },
        "required": [],
    }

    def execute(
        self,
        path: str = ".",
        base_ref: str = "",
        format: str = "standard",
        **kwargs,
    ) -> str:
        root, error = resolve_project_path(path or ".")
        if error:
            return error
        assert root is not None
        if not root.exists():
            return f"Error: Target path not found: {path}"

        try:
            report = run_code_review(
                root=root if root.is_dir() else root.parent,
                base_ref=base_ref.strip() or None,
                files=[root] if root.is_file() else None,
                execute_tests=False,
            )
            return report.render_markdown(format_mode=format)
        except Exception as exc:  # noqa: BLE001
            return f"Error: Code review execution failed: {exc}"
