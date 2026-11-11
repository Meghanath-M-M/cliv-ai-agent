"""Tools for reading and writing agent long-term memory."""

from __future__ import annotations

from typing import Any, Dict, Optional

from cliv.memory import MemoryStore
from cliv.tools.base import BaseTool


class RememberTool(BaseTool):
    """Tool to store persistent preferences and conventions."""

    name = "remember"
    description = (
        "Store a persistent user preference, coding convention, or project "
        "architectural fact in long-term memory across sessions."
    )
    permissions = frozenset({"write"})
    risk_level = "low"
    requires_approval = False
    input_schema = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The fact, preference, or convention to remember.",
            },
            "scope": {
                "type": "string",
                "enum": ["project", "global"],
                "default": "project",
                "description": (
                    "Scope: 'project' saves to this repo's memory; "
                    "'global' saves across all projects."
                ),
            },
            "key": {
                "type": "string",
                "description": "Optional short identifier/slug for this memory (e.g. 'pytest-fixtures').",
            },
            "category": {
                "type": "string",
                "enum": ["preference", "convention", "architecture", "tooling", "general"],
                "default": "general",
                "description": "Category for organizing memories.",
            },
        },
        "required": ["content"],
    }

    def __init__(self, store: Optional[MemoryStore] = None):
        super().__init__()
        self._store = store

    @property
    def store(self) -> MemoryStore:
        if self._store is None:
            self._store = MemoryStore()
        return self._store

    def execute(
        self,
        content: str,
        scope: str = "project",
        key: Optional[str] = None,
        category: str = "general",
        **kwargs: Any,
    ) -> str:
        clean_content = content.strip()
        if not clean_content:
            return "Error: Memory content cannot be empty."

        target_scope = "global" if scope.lower().strip() == "global" else "project"
        entry = self.store.add(
            content=clean_content,
            scope=target_scope,
            key=key,
            category=category,
        )
        return (
            f"Remembered in {target_scope} memory (id: {entry.id}):\n"
            f"\"{entry.content}\""
        )


class RecallTool(BaseTool):
    """Tool to recall stored preferences and conventions."""

    name = "recall"
    description = (
        "Query or list remembered user preferences, coding conventions, and "
        "project architectural rules from long-term memory."
    )
    permissions = frozenset({"read"})
    risk_level = "low"
    requires_approval = False
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Optional keyword or search term to filter memories.",
            },
            "scope": {
                "type": "string",
                "enum": ["all", "project", "global"],
                "default": "all",
                "description": "Filter by scope ('all', 'project', or 'global').",
            },
        },
    }

    def __init__(self, store: Optional[MemoryStore] = None):
        super().__init__()
        self._store = store

    @property
    def store(self) -> MemoryStore:
        if self._store is None:
            self._store = MemoryStore()
        return self._store

    def execute(
        self,
        query: str = "",
        scope: str = "all",
        **kwargs: Any,
    ) -> str:
        filter_scope = None if scope.lower().strip() == "all" else scope.lower().strip()
        entries = self.store.list_entries(scope=filter_scope)

        clean_query = query.strip().lower()
        if clean_query:
            entries = [
                e for e in entries
                if clean_query in e.content.lower() or clean_query in e.id.lower() or clean_query in e.category.lower()
            ]

        cliv_md = self.store.get_cliv_md_content()

        if not entries and not cliv_md:
            filter_msg = f" for '{query}'" if clean_query else ""
            return f"No memories found{filter_msg}."

        lines = ["Remembered items:"]
        if cliv_md and (not clean_query or clean_query in cliv_md.lower()):
            lines.append("\n[CLIV.md Guidelines]")
            lines.append(cliv_md)

        if entries:
            lines.append("\n[Stored Memories]")
            for e in entries:
                cat = f" [{e.category}]" if e.category != "general" else ""
                lines.append(f"- ({e.scope}) {e.id}{cat}: {e.content}")

        return "\n".join(lines)
