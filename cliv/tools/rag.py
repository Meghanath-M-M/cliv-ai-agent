"""RAG Query Tool for the agent.

Allows the agent to search the project memory using semantic search.
"""

from __future__ import annotations

from typing import Any, Dict, List

from cliv.intelligence.rag_indexer import RAGIndexer, create_rag_indexer
from cliv.tools.base import BaseTool


class RAGQueryTool(BaseTool):
    name = "rag_query"
    description = (
        "Search the project's semantic memory (RAG index) for relevant code. "
        "Use natural language queries like 'authentication logic', 'retry mechanism', "
        "'database connection pooling', etc. Returns relevant code chunks with file paths."
    )
    permissions = frozenset({"read"})
    risk_level = "low"
    input_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query",
            },
            "k": {
                "type": "integer",
                "description": "Number of results to return (default: 5)",
                "default": 5,
                "minimum": 1,
                "maximum": 20,
            },
        },
        "required": ["query"],
    }

    def __init__(self):
        super().__init__()
        self._indexer: RAGIndexer | None = None

    def _get_indexer(self) -> RAGIndexer:
        if self._indexer is None:
            self._indexer = create_rag_indexer()
        return self._indexer

    def execute(self, query: str, k: int = 5, auto_index: bool = True, **kwargs) -> str:
        try:
            indexer = self._get_indexer()

            # If the index is empty, try building it automatically once
            try:
                store = indexer._load_or_create_store()
                if getattr(getattr(store, "index", None), "ntotal", 0) == 0 and auto_index:
                    indexer.index(force_full=False)
            except Exception:
                pass

            results = indexer.search(query, k=k)

            if not results:
                return (
                    f"No relevant code found matching '{query}'. "
                    "Try broader conceptual terms or use `search_codebase` for literal string/regex matches."
                )

            lines = [f"RAG Search Results for: '{query}'", ""]
            for i, (chunk, score) in enumerate(results, 1):
                lines.append(f"[{i}] {chunk.file_path} (score: {score:.3f})")
                if chunk.symbol_name:
                    lines.append(f"    Symbol: {chunk.symbol_name} ({chunk.symbol_kind})")
                lines.append(f"    Lines {chunk.start_line}-{chunk.end_line}:")
                # Truncate content for readability
                content_lines = chunk.content.splitlines()
                for line in content_lines[:10]:
                    lines.append(f"    {line}")
                if len(content_lines) > 10:
                    lines.append(f"    ... ({len(content_lines) - 10} more lines)")
                lines.append("")

            return "\n".join(lines)

        except Exception as e:
            return (
                f"RAG query unavailable: {e}. "
                "Use `search_codebase` or `read_file` to inspect code directly."
            )


class RAGIndexTool(BaseTool):
    name = "rag_index"
    description = (
        "Build or update the project's semantic memory (RAG index). "
        "Run this after significant code changes to update the search index. "
        "Use force=true for a full rebuild."
    )
    permissions = frozenset({"read", "write"})
    risk_level = "low"
    input_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "force": {
                "type": "boolean",
                "description": "Force full rebuild instead of incremental update",
                "default": False,
            },
        },
        "required": [],
    }

    def __init__(self):
        super().__init__()
        self._indexer: RAGIndexer | None = None

    def _get_indexer(self) -> RAGIndexer:
        if self._indexer is None:
            self._indexer = create_rag_indexer()
        return self._indexer

    def execute(self, force: bool = False, **kwargs) -> str:
        try:
            indexer = self._get_indexer()
            stats = indexer.index(force_full=force)

            lines = ["RAG Indexing Complete", ""]
            lines.append(f"Files scanned: {stats.files_scanned}")
            lines.append(f"Files changed: {stats.files_changed}")
            lines.append(f"Files removed: {stats.files_removed}")
            lines.append(f"Chunks added: {stats.chunks_added}")
            lines.append(f"Chunks removed: {stats.chunks_removed}")
            lines.append(f"Duration: {stats.duration_seconds:.2f}s")

            if stats.errors:
                lines.append("")
                lines.append("Errors:")
                for err in stats.errors:
                    lines.append(f"  - {err}")

            store_stats = indexer.get_stats()
            lines.append("")
            lines.append("Index Stats:")
            lines.append(f"  Total chunks: {store_stats['total_chunks']}")
            lines.append(f"  Unique files: {store_stats['unique_files']}")
            lines.append(f"  Dimension: {store_stats['dimension']}")

            return "\n".join(lines)

        except Exception as e:
            return f"RAG indexing failed: {e}"


class RAGStatsTool(BaseTool):
    name = "rag_stats"
    description = "Get statistics about the current RAG index."
    permissions = frozenset({"read"})
    risk_level = "low"
    input_schema: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}

    def __init__(self):
        super().__init__()
        self._indexer: RAGIndexer | None = None

    def _get_indexer(self) -> RAGIndexer:
        if self._indexer is None:
            self._indexer = create_rag_indexer()
        return self._indexer

    def execute(self, **kwargs) -> str:
        try:
            indexer = self._get_indexer()
            stats = indexer.get_stats()

            lines = ["RAG Index Statistics", ""]
            lines.append(f"Total chunks: {stats['total_chunks']}")
            lines.append(f"Unique files: {stats['unique_files']}")
            lines.append(f"Vector dimension: {stats['dimension']}")
            lines.append(f"Index size: {stats['index_size']}")
            lines.append(f"Created: {stats['created_at']}")
            lines.append(f"Updated: {stats['updated_at']}")

            return "\n".join(lines)

        except Exception as e:
            return f"RAG stats failed: {e}"