"""Cross-workspace agent tools.

Allows the agent to read files and search code from any registered workspace
(not just the active one) without escaping the security boundary of that repo.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from cliv.tools.base import BaseTool
from cliv.workspace import WorkspaceRegistry, resolve_workspace_path

IGNORED_DIRS = {".git", ".venv", "__pycache__", "node_modules", ".pytest_cache", ".mypy_cache"}
INDEXED_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java",
                      ".cpp", ".c", ".h", ".cs", ".rb", ".md", ".txt", ".toml",
                      ".yaml", ".yml", ".json", ".sh", ".env.example"}


class ListWorkspaceTool(BaseTool):
    """List all registered workspaces and the active one."""

    name = "list_workspaces"
    description = (
        "List all registered multi-repo workspaces with their paths and status. "
        "Use this to discover available repos before reading cross-repo files."
    )
    permissions = frozenset({"read"})
    risk_level = "low"
    input_schema = {"type": "object", "properties": {}, "required": []}

    def __init__(self, registry: Optional[WorkspaceRegistry] = None) -> None:
        self._registry = registry

    @property
    def registry(self) -> WorkspaceRegistry:
        if self._registry is None:
            self._registry = WorkspaceRegistry()
        return self._registry

    def execute(self, **kwargs: Any) -> str:
        entries = self.registry.list()
        if not entries:
            return (
                "No workspaces registered.\n"
                "Register repos with: /workspace add <path> [<name>]\n"
                "or: cliv workspace add <path>"
            )
        active = self.registry.active_entry()
        lines = [f"Registered workspaces ({len(entries)} total):\n"]
        for e in entries:
            marker = " [ACTIVE]" if (active and e.name == active.name) else ""
            status = "✓" if e.exists() else "✗ MISSING"
            desc = f"  {e.description}" if e.description else ""
            tags = f"  [{', '.join(e.tags)}]" if e.tags else ""
            lines.append(f"  {status} {e.name}{marker}: {e.path}{desc}{tags}")
        return "\n".join(lines)


class ReadWorkspaceFileTool(BaseTool):
    """Read a file from a named workspace (cross-repo)."""

    name = "read_workspace_file"
    description = (
        "Read the contents of a file from a *different* registered workspace/repo. "
        "Use list_workspaces first to discover workspace names, then call this with "
        "workspace=<name> and path=<relative_path>."
    )
    permissions = frozenset({"read"})
    risk_level = "low"
    input_schema = {
        "type": "object",
        "properties": {
            "workspace": {
                "type": "string",
                "description": "Workspace name as shown by list_workspaces.",
            },
            "path": {
                "type": "string",
                "description": "Path to the file, relative to the workspace root.",
            },
            "max_bytes": {
                "type": "integer",
                "description": "Maximum bytes to return (default 50000).",
            },
        },
        "required": ["workspace", "path"],
    }

    def __init__(self, registry: Optional[WorkspaceRegistry] = None) -> None:
        self._registry = registry

    @property
    def registry(self) -> WorkspaceRegistry:
        if self._registry is None:
            self._registry = WorkspaceRegistry()
        return self._registry

    def execute(
        self,
        workspace: str,
        path: str,
        max_bytes: int = 50_000,
        **kwargs: Any,
    ) -> str:
        entry = self.registry.get(workspace)
        if entry is None:
            names = [e.name for e in self.registry.list()]
            hint = f"Available: {', '.join(names)}" if names else "No workspaces registered."
            return f"Unknown workspace '{workspace}'. {hint}"

        if not entry.exists():
            return f"Workspace path no longer exists: {entry.path}"

        file_path, err = resolve_workspace_path(entry, path)
        if err:
            return err
        assert file_path is not None

        if not file_path.exists():
            return f"File not found in workspace '{workspace}': {path}"
        if not file_path.is_file():
            return f"Path is not a file in workspace '{workspace}': {path}"

        try:
            content = file_path.read_bytes()
            decoded = content[:max_bytes].decode("utf-8", errors="replace")
            truncated = len(content) > max_bytes
            suffix = f"\n\n[...truncated: showing first {max_bytes} bytes]" if truncated else ""
            return decoded + suffix
        except Exception as exc:
            return f"Error reading file from workspace '{workspace}': {exc}"


class ListWorkspaceFilesTool(BaseTool):
    """List files in a named workspace (cross-repo)."""

    name = "list_workspace_files"
    description = (
        "List the files and directories inside a registered workspace/repo. "
        "Use to explore the layout of a different repo before reading specific files."
    )
    permissions = frozenset({"read"})
    risk_level = "low"
    input_schema = {
        "type": "object",
        "properties": {
            "workspace": {
                "type": "string",
                "description": "Workspace name as shown by list_workspaces.",
            },
            "path": {
                "type": "string",
                "description": "Subdirectory to list (relative to workspace root; default '.').",
                "default": ".",
            },
            "max_entries": {
                "type": "integer",
                "description": "Maximum number of entries to return (default 200).",
            },
        },
        "required": ["workspace"],
    }

    def __init__(self, registry: Optional[WorkspaceRegistry] = None) -> None:
        self._registry = registry

    @property
    def registry(self) -> WorkspaceRegistry:
        if self._registry is None:
            self._registry = WorkspaceRegistry()
        return self._registry

    def execute(
        self,
        workspace: str,
        path: str = ".",
        max_entries: int = 200,
        **kwargs: Any,
    ) -> str:
        entry = self.registry.get(workspace)
        if entry is None:
            names = [e.name for e in self.registry.list()]
            hint = f"Available: {', '.join(names)}" if names else "No workspaces registered."
            return f"Unknown workspace '{workspace}'. {hint}"

        if not entry.exists():
            return f"Workspace path no longer exists: {entry.path}"

        dir_path, err = resolve_workspace_path(entry, path)
        if err:
            return err
        assert dir_path is not None

        if not dir_path.is_dir():
            return f"Path is not a directory in workspace '{workspace}': {path}"

        lines: List[str] = [f"Contents of [{workspace}]/{path}:\n"]
        count = 0
        try:
            for item in sorted(dir_path.iterdir()):
                if count >= max_entries:
                    lines.append(f"\n[...truncated after {max_entries} entries]")
                    break
                rel = item.relative_to(entry.resolved_path)
                if item.is_dir():
                    if item.name in IGNORED_DIRS:
                        continue
                    lines.append(f"  [DIR]  {rel}/")
                else:
                    size_kb = item.stat().st_size / 1024
                    lines.append(f"  [FILE] {rel} ({size_kb:.1f} KB)")
                count += 1
        except PermissionError as exc:
            lines.append(f"\nPermission denied: {exc}")

        return "\n".join(lines)


class SearchWorkspaceTool(BaseTool):
    """Grep-based text search across a named workspace (cross-repo)."""

    name = "search_workspace"
    description = (
        "Search for a pattern (literal text or regex) in source files of a named "
        "workspace/repo. Returns matching lines with context. Use for cross-repo "
        "dependency tracing, shared interface discovery, and API surface analysis."
    )
    permissions = frozenset({"read"})
    risk_level = "low"
    input_schema = {
        "type": "object",
        "properties": {
            "workspace": {
                "type": "string",
                "description": "Workspace name as shown by list_workspaces.",
            },
            "pattern": {
                "type": "string",
                "description": "Text pattern or regex to search for.",
            },
            "file_extensions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "File extensions to include (e.g. ['.py', '.ts']). Default: all code files.",
            },
            "case_sensitive": {
                "type": "boolean",
                "description": "Whether the search is case-sensitive (default false).",
                "default": False,
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results (default 30).",
            },
        },
        "required": ["workspace", "pattern"],
    }

    def __init__(self, registry: Optional[WorkspaceRegistry] = None) -> None:
        self._registry = registry

    @property
    def registry(self) -> WorkspaceRegistry:
        if self._registry is None:
            self._registry = WorkspaceRegistry()
        return self._registry

    def execute(
        self,
        workspace: str,
        pattern: str,
        file_extensions: Optional[List[str]] = None,
        case_sensitive: bool = False,
        max_results: int = 30,
        **kwargs: Any,
    ) -> str:
        import re as _re

        entry = self.registry.get(workspace)
        if entry is None:
            names = [e.name for e in self.registry.list()]
            hint = f"Available: {', '.join(names)}" if names else "No workspaces registered."
            return f"Unknown workspace '{workspace}'. {hint}"

        if not entry.exists():
            return f"Workspace path no longer exists: {entry.path}"

        try:
            flags = 0 if case_sensitive else _re.IGNORECASE
            compiled = _re.compile(pattern, flags)
        except _re.error as exc:
            return f"Invalid regex pattern: {exc}"

        exts = set(file_extensions) if file_extensions else INDEXED_EXTENSIONS
        root = entry.resolved_path
        matches: List[str] = []

        for file_path in _walk_files(root, exts):
            if len(matches) >= max_results:
                break
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
                for lineno, line in enumerate(text.splitlines(), 1):
                    if compiled.search(line):
                        rel = file_path.relative_to(root)
                        matches.append(f"  [{workspace}]/{rel}:{lineno}: {line.rstrip()}")
                        if len(matches) >= max_results:
                            break
            except (PermissionError, OSError):
                continue

        if not matches:
            return f"No matches for '{pattern}' in workspace '{workspace}'."

        header = f"Found {len(matches)} match(es) for '{pattern}' in [{workspace}]:\n"
        suffix = "\n[...results truncated]" if len(matches) >= max_results else ""
        return header + "\n".join(matches) + suffix


# ---------------------------------------------------------------------------
# File walker
# ---------------------------------------------------------------------------

def _walk_files(root: Path, extensions: set) -> List[Path]:
    results: List[Path] = []
    try:
        for item in root.rglob("*"):
            if any(part in IGNORED_DIRS for part in item.parts):
                continue
            if item.is_file() and item.suffix.lower() in extensions:
                results.append(item)
    except (PermissionError, OSError):
        pass
    return sorted(results)
