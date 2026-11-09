"""Run pytest with approval and a project-root working directory."""

import subprocess
import sys
from typing import Any, Dict, Optional

from cliv.security import project_root, resolve_project_path
from cliv.tools.base import BaseTool


class RunTestsTool(BaseTool):
    name = "run_tests"
    description = (
        "Execute the pytest test suite and return the results, including error traces. "
        "Requires approval because pytest imports and executes project code."
    )
    requires_approval = True
    input_schema = {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "Specific test file/directory/node to run (empty runs all tests)",
            }
        },
        "required": [],
    }

    def _validate_target(self, target: str) -> Optional[str]:
        if not target:
            return None
        path_part = target.split("::", 1)[0]
        _, error = resolve_project_path(path_part)
        return error

    def preview(self, tool_input: Dict[str, Any]) -> Optional[str]:
        target = str(tool_input.get("target", "") or "")
        error = self._validate_target(target)
        if error:
            return error
        suffix = f" {target}" if target else ""
        return f"Will run: {sys.executable} -m pytest{suffix} (in {project_root()})"

    def execute(self, target: str = "", **kwargs) -> str:
        error = self._validate_target(target)
        if error:
            return error

        command = [sys.executable, "-m", "pytest"]
        if target:
            command.append(target)

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=project_root(),
                timeout=120,
            )
            output = result.stdout.strip()
            if result.stderr:
                output += f"\nErrors:\n{result.stderr.strip()}"
            return f"Pytest Results (exit {result.returncode}):\n{output}"
        except subprocess.TimeoutExpired:
            return "Error: pytest timed out after 120 seconds."
        except FileNotFoundError:
            return f"Error: pytest is not installed for {sys.executable}."
        except Exception as exc:  # noqa: BLE001
            return f"Unexpected error running pytest: {exc}"
