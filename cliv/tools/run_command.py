"""Approval-gated command runner with a conservative allowlist.

There is no shell interpolation: the tool takes an executable name and an
argument list, and invokes subprocess with ``shell=False``.  The
executable must be in the allowlist below (or added through
``CLIV_ALLOWED_COMMANDS``).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional

from cliv.security import project_root, resolve_project_path
from cliv.tools.base import BaseTool

DEFAULT_ALLOWED_COMMANDS = {
    "cat",
    "dir",
    "echo",
    "git",
    "ls",
    "pwd",
    "pytest",
    "python",
    "python3",
    "type",
    "which",
    "where",
}


def allowed_commands() -> set[str]:
    configured = os.environ.get("CLIV_ALLOWED_COMMANDS", "")
    extra = {item.strip() for item in configured.split(",") if item.strip()}
    return DEFAULT_ALLOWED_COMMANDS | extra


class RunCommandTool(BaseTool):
    name = "run_command"
    description = (
        "Run an allowlisted command without a shell. Requires approval. "
        "Use this for build/test commands, not for arbitrary shell pipelines."
    )
    requires_approval = True
    input_schema = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Executable name, e.g. python, git, pytest, ls",
            },
            "args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Arguments passed directly to the executable",
            },
            "cwd": {
                "type": "string",
                "description": "Optional working directory inside the project",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (1-300, default 120)",
            },
        },
        "required": ["command"],
    }

    def _resolve_executable(self, command: str) -> tuple[Optional[str], Optional[str]]:
        command = (command or "").strip()
        if not command:
            return None, "Error: command is required."
        if "/" in command or "\\" in command or command in {".", ".."}:
            return None, "Error: command must be a bare executable name."
        if command not in allowed_commands():
            allowed = ", ".join(sorted(allowed_commands()))
            return None, f"Error: command '{command}' is not allowlisted. Allowed: {allowed}"

        if command in {"python", "python3"}:
            return sys.executable, None
        executable = shutil.which(command)
        if not executable:
            return None, f"Error: command not found: {command}"
        return executable, None

    def _parse(self, tool_input: Dict[str, Any]):
        command = str(tool_input.get("command", ""))
        executable, error = self._resolve_executable(command)
        if error:
            return None, error

        raw_args = tool_input.get("args") or []
        if not isinstance(raw_args, list) or not all(
            isinstance(item, (str, int, float)) for item in raw_args
        ):
            return None, "Error: args must be a list of strings."
        args: List[str] = [str(item) for item in raw_args]

        try:
            timeout = int(tool_input.get("timeout", 120))
        except (TypeError, ValueError):
            return None, "Error: timeout must be an integer."
        timeout = max(1, min(timeout, 300))

        cwd_input = tool_input.get("cwd")
        if cwd_input:
            cwd, cwd_error = resolve_project_path(str(cwd_input))
            if cwd_error:
                return None, cwd_error
            assert cwd is not None
            if not cwd.is_dir():
                return None, f"Error: cwd is not a directory: {cwd_input}"
        else:
            cwd = project_root()

        return (executable, args, cwd, timeout), None

    def preview(self, tool_input: Dict[str, Any]) -> Optional[str]:
        parsed, error = self._parse(tool_input)
        if error:
            return error
        assert parsed is not None
        executable, args, cwd, timeout = parsed
        rendered = " ".join([str(executable), *args])
        return f"Will run (no shell, timeout {timeout}s):\n  {rendered}\n  cwd: {cwd}"

    def execute(self, **kwargs) -> str:
        parsed, error = self._parse(kwargs)
        if error:
            return error
        assert parsed is not None
        executable, args, cwd, timeout = parsed

        # Do not hand the model's own API key to a generated command.
        safe_env = {
            key: value
            for key, value in os.environ.items()
            if key != "GROQ_API_KEY"
        }
        safe_env["PYTHONIOENCODING"] = "utf-8"

        try:
            result = subprocess.run(
                [executable, *args],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False,
                env=safe_env,
            )
            output = result.stdout.strip()
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr.strip()}"
            if len(output) > 20_000:
                output = output[:20_000] + "\n... [output truncated]"
            return f"Exit code: {result.returncode}\n{output}".strip()
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {timeout} seconds."
        except FileNotFoundError:
            return f"Error: executable not found: {executable}"
        except Exception as exc:  # noqa: BLE001
            return f"Error running command: {exc}"
