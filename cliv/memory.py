"""Agent long-term memory store for persistent user preferences and project conventions."""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from cliv.config import MEMORY_FILE
from cliv.security import project_root


@dataclass
class MemoryEntry:
    id: str
    content: str
    scope: str  # "global" or "project"
    created_at: str
    updated_at: str
    category: str = "general"  # "preference", "convention", "architecture", "tooling", "general"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], default_scope: str = "project") -> Optional["MemoryEntry"]:
        if not isinstance(data, dict):
            return None
        content = str(data.get("content") or "").strip()
        if not content:
            return None
        entry_id = str(data.get("id") or "").strip()
        if not entry_id:
            entry_id = _generate_slug(content)
        scope = str(data.get("scope") or default_scope).lower()
        if scope not in {"global", "project"}:
            scope = default_scope
        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return cls(
            id=entry_id,
            content=content,
            scope=scope,
            created_at=str(data.get("created_at") or now_iso),
            updated_at=str(data.get("updated_at") or now_iso),
            category=str(data.get("category") or "general"),
        )


def _generate_slug(content: str) -> str:
    words = [w.lower() for w in re.findall(r"[a-zA-Z0-9]+", content)[:4]]
    base = "-".join(words)
    suffix = uuid.uuid4().hex[:4]
    return f"{base}-{suffix}" if base else f"mem-{suffix}"


class MemoryStore:
    """Manages persistent global and project-scoped memories."""

    def __init__(
        self,
        project_path: Optional[Path] = None,
        global_file: Optional[Path] = None,
    ):
        self._project_path = project_path
        self._global_file = global_file or MEMORY_FILE

    @property
    def project_dir(self) -> Path:
        return (self._project_path or project_root()).resolve()

    @property
    def project_file(self) -> Path:
        return self.project_dir / ".cliv" / "memory.json"

    @property
    def project_cliv_md(self) -> Path:
        return self.project_dir / "CLIV.md"

    def _read_file(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict)]
            if isinstance(data, dict) and "memories" in data and isinstance(data["memories"], list):
                return [item for item in data["memories"] if isinstance(item, dict)]
            return []
        except Exception as exc:
            logging.error("Failed to read memory file %s: %s", path, exc)
            return []

    def _write_file(self, path: Path, entries: List[MemoryEntry]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(".json.tmp")
        try:
            payload = [entry.to_dict() for entry in entries]
            with open(temp_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
            os.replace(temp_path, path)
        except Exception as exc:
            logging.error("Failed to write memory file %s: %s", path, exc)
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    def list_entries(self, scope: Optional[str] = None) -> List[MemoryEntry]:
        """List all remembered entries, optionally filtered by scope ('global' or 'project')."""
        results: List[MemoryEntry] = []
        target_scope = scope.lower().strip() if scope else None

        # Load global memories
        if target_scope in {None, "global"}:
            for raw in self._read_file(self._global_file):
                entry = MemoryEntry.from_dict(raw, default_scope="global")
                if entry:
                    entry.scope = "global"
                    results.append(entry)

        # Load project memories
        if target_scope in {None, "project"}:
            for raw in self._read_file(self.project_file):
                entry = MemoryEntry.from_dict(raw, default_scope="project")
                if entry:
                    entry.scope = "project"
                    results.append(entry)

        return results

    def add(
        self,
        content: str,
        scope: str = "project",
        key: Optional[str] = None,
        category: str = "general",
    ) -> MemoryEntry:
        """Add or update a memory entry."""
        target_scope = "global" if scope.lower().strip() == "global" else "project"
        clean_content = content.strip()
        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        entry_id = key.strip() if key and key.strip() else _generate_slug(clean_content)

        file_path = self._global_file if target_scope == "global" else self.project_file
        existing = [
            e for raw in self._read_file(file_path)
            if (e := MemoryEntry.from_dict(raw, default_scope=target_scope)) is not None
        ]

        updated = False
        new_entry = MemoryEntry(
            id=entry_id,
            content=clean_content,
            scope=target_scope,
            created_at=now_iso,
            updated_at=now_iso,
            category=category,
        )

        for i, item in enumerate(existing):
            if item.id == entry_id:
                new_entry.created_at = item.created_at
                existing[i] = new_entry
                updated = True
                break

        if not updated:
            existing.append(new_entry)

        self._write_file(file_path, existing)
        return new_entry

    def remove(self, key_or_id: str, scope: Optional[str] = None) -> bool:
        """Remove a memory by its ID or content substring."""
        target_id = key_or_id.strip().lower()
        target_scope = scope.lower().strip() if scope else None
        removed_any = False

        scopes_to_check = []
        if target_scope in {None, "project"}:
            scopes_to_check.append(("project", self.project_file))
        if target_scope in {None, "global"}:
            scopes_to_check.append(("global", self._global_file))

        for sc_name, file_path in scopes_to_check:
            entries = [
                e for raw in self._read_file(file_path)
                if (e := MemoryEntry.from_dict(raw, default_scope=sc_name)) is not None
            ]
            original_len = len(entries)
            filtered = [
                e for e in entries
                if e.id.lower() != target_id and target_id not in e.content.lower()
            ]
            if len(filtered) < original_len:
                self._write_file(file_path, filtered)
                removed_any = True

        return removed_any

    def clear(self, scope: Optional[str] = None) -> int:
        """Clear memories for a given scope (or both if None)."""
        target_scope = scope.lower().strip() if scope else None
        count = 0

        if target_scope in {None, "project"} and self.project_file.exists():
            entries = self._read_file(self.project_file)
            count += len(entries)
            self._write_file(self.project_file, [])

        if target_scope in {None, "global"} and self._global_file.exists():
            entries = self._read_file(self._global_file)
            count += len(entries)
            self._write_file(self._global_file, [])

        return count

    def get_cliv_md_content(self) -> Optional[str]:
        """Read optional CLIV.md conventions file from project root."""
        if self.project_cliv_md.exists() and self.project_cliv_md.is_file():
            try:
                content = self.project_cliv_md.read_text(encoding="utf-8", errors="replace").strip()
                if content:
                    return content
            except Exception as exc:
                logging.error("Failed to read CLIV.md: %s", exc)
        return None

    def format_prompt_context(self, max_items: int = 30) -> str:
        """Generate markdown section for injection into the agent system prompt."""
        entries = self.list_entries()
        cliv_md = self.get_cliv_md_content()

        if not entries and not cliv_md:
            return ""

        lines: List[str] = [
            "## Long-Term Memory & Conventions",
            "Follow these remembered user preferences and project conventions strictly:",
        ]

        if cliv_md:
            lines.append("\n### Project Guidelines (CLIV.md):")
            lines.append(cliv_md)

        if entries:
            global_entries = [e for e in entries if e.scope == "global"][:max_items]
            project_entries = [e for e in entries if e.scope == "project"][:max_items]

            if project_entries:
                lines.append("\n### Project Conventions:")
                for e in project_entries:
                    cat_prefix = f"[{e.category}] " if e.category != "general" else ""
                    lines.append(f"- {cat_prefix}{e.content} (id: {e.id})")

            if global_entries:
                lines.append("\n### User Preferences:")
                for e in global_entries:
                    cat_prefix = f"[{e.category}] " if e.category != "general" else ""
                    lines.append(f"- {cat_prefix}{e.content} (id: {e.id})")

        return "\n".join(lines).strip()
