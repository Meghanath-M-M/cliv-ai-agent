"""Multi-repo workspace registry for cliv.

A *workspace* is a named, persistent pointer to a local repository directory.
Workspaces allow the agent to reason across multiple repos, monorepo packages,
and microservice boundaries within a single cliv session.

Storage layout (all under ``~/.config/cliv/``)::

    workspaces.json        – registry of named workspace entries
    <name>/history.json    – per-workspace chat history (optional override)

The *active* workspace is the repo cliv was started in (``project_root()``).
Any extra workspaces can be queried with the ``read_workspace_file`` and
``search_workspace`` agent tools, or browsed via ``/workspace`` slash commands.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cliv.config import CONFIG_DIR, WORKSPACES_FILE
from cliv.security import is_sensitive_path, project_root


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class WorkspaceEntry:
    name: str
    path: str                       # absolute, resolved
    description: str = ""
    added_at: str = ""
    last_active: str = ""
    tags: List[str] = None          # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.tags is None:
            self.tags = []
        if not self.added_at:
            self.added_at = _now()
        if not self.last_active:
            self.last_active = self.added_at

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optional["WorkspaceEntry"]:
        if not isinstance(data, dict):
            return None
        name = str(data.get("name", "")).strip()
        path = str(data.get("path", "")).strip()
        if not name or not path:
            return None
        return cls(
            name=name,
            path=path,
            description=str(data.get("description", "")),
            added_at=str(data.get("added_at", "")),
            last_active=str(data.get("last_active", "")),
            tags=list(data.get("tags") or []),
        )

    @property
    def resolved_path(self) -> Path:
        return Path(self.path).expanduser().resolve()

    def exists(self) -> bool:
        return self.resolved_path.is_dir()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slug(name: str) -> str:
    """Lowercase, dash-separated identifier with no special chars."""
    import re
    return re.sub(r"[^a-z0-9]+", "-", name.lower().strip()).strip("-") or "workspace"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class WorkspaceRegistry:
    """Persistent multi-repo workspace registry backed by JSON."""

    def __init__(self, registry_file: Optional[Path] = None) -> None:
        self._file = registry_file or WORKSPACES_FILE

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _read(self) -> List[WorkspaceEntry]:
        if not self._file.exists():
            return []
        try:
            with open(self._file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, list):
                return []
            entries = []
            for item in data:
                e = WorkspaceEntry.from_dict(item)
                if e:
                    entries.append(e)
            return entries
        except Exception as exc:
            logging.error("Failed to read workspace registry %s: %s", self._file, exc)
            return []

    def _write(self, entries: List[WorkspaceEntry]) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._file.with_suffix(".json.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump([e.to_dict() for e in entries], fh, indent=2, ensure_ascii=False)
            os.replace(tmp, self._file)
        except Exception as exc:
            logging.error("Failed to write workspace registry: %s", exc)
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list(self) -> List[WorkspaceEntry]:
        """Return all registered workspaces, sorted alphabetically by name."""
        return sorted(self._read(), key=lambda e: e.name.lower())

    def get(self, name: str) -> Optional[WorkspaceEntry]:
        """Return the workspace with the given name (case-insensitive), or None."""
        needle = name.strip().lower()
        for e in self._read():
            if e.name.lower() == needle:
                return e
        return None

    def add(
        self,
        path: str,
        name: Optional[str] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> Tuple[WorkspaceEntry, bool]:
        """Add or update a workspace. Returns (entry, is_new)."""
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError(f"Path does not exist or is not a directory: {path}")

        auto_name = name.strip() if name and name.strip() else resolved.name
        safe_name = _slug(auto_name)

        entries = self._read()
        for i, e in enumerate(entries):
            if e.name.lower() == safe_name.lower():
                # Update existing
                entries[i].path = str(resolved)
                entries[i].description = description or e.description
                entries[i].last_active = _now()
                if tags:
                    entries[i].tags = tags
                self._write(entries)
                return entries[i], False

        new_entry = WorkspaceEntry(
            name=safe_name,
            path=str(resolved),
            description=description,
            tags=tags or [],
        )
        entries.append(new_entry)
        self._write(entries)
        return new_entry, True

    def remove(self, name: str) -> bool:
        """Remove workspace by name. Returns True if removed."""
        needle = name.strip().lower()
        entries = self._read()
        filtered = [e for e in entries if e.name.lower() != needle]
        if len(filtered) == len(entries):
            return False
        self._write(filtered)
        return True

    def touch(self, name: str) -> None:
        """Update last_active timestamp for workspace."""
        entries = self._read()
        for e in entries:
            if e.name.lower() == name.strip().lower():
                e.last_active = _now()
        self._write(entries)

    def active_entry(self) -> Optional[WorkspaceEntry]:
        """Return the workspace whose path matches the current project_root(), if any."""
        current = str(project_root())
        for e in self._read():
            if str(e.resolved_path) == current:
                return e
        return None

    def auto_register_current(self) -> WorkspaceEntry:
        """Ensure the current project_root() is in the registry, adding it if missing."""
        root = project_root()
        active = self.active_entry()
        if active:
            return active
        entry, _ = self.add(str(root))
        return entry


# ---------------------------------------------------------------------------
# Path safety helpers for cross-workspace tool calls
# ---------------------------------------------------------------------------

def resolve_workspace_path(
    workspace: "WorkspaceEntry",
    rel_path: str,
    *,
    for_write: bool = False,
) -> Tuple[Optional[Path], Optional[str]]:
    """Resolve a path relative to a specific workspace, enforcing the same
    security rules as ``cliv.security.resolve_project_path``."""
    root = workspace.resolved_path
    if not root.is_dir():
        return None, f"Workspace path no longer exists: {root}"

    raw = Path(rel_path).expanduser()
    if not raw.is_absolute():
        raw = root / raw

    try:
        resolved = raw.resolve()
    except (OSError, RuntimeError) as exc:
        return None, f"Error resolving path '{rel_path}': {exc}"

    try:
        resolved.relative_to(root)
    except ValueError:
        return None, f"Error: Path is outside workspace '{workspace.name}': {rel_path}"

    if is_sensitive_path(resolved):
        return None, f"Error: Sensitive file blocked: {rel_path}"

    if for_write:
        from cliv.security import PROTECTED_WRITE_PARTS
        rel_parts = set(resolved.relative_to(root).parts)
        if rel_parts & PROTECTED_WRITE_PARTS:
            return None, f"Error: Writing inside protected directory is blocked: {rel_path}"

    return resolved, None
