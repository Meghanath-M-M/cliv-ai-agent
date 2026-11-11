"""Tests for RAG query and index tools."""

from types import SimpleNamespace
import pytest

from cliv.tools.rag import RAGQueryTool, RAGIndexTool, RAGStatsTool
from cliv.intelligence.embeddings import CodeChunk


class FakeVectorStore:
    def __init__(self, ntotal=0):
        self.index = SimpleNamespace(ntotal=ntotal)

    def stats(self):
        return {
            "total_chunks": self.index.ntotal,
            "unique_files": 2 if self.index.ntotal > 0 else 0,
            "dimension": 128,
            "index_size": "10 KB",
            "created_at": 1000.0,
            "updated_at": 1001.0,
        }


class FakeIndexer:
    def __init__(self, initial_total=0):
        self.store = FakeVectorStore(ntotal=initial_total)
        self.indexed = False

    def _load_or_create_store(self):
        return self.store

    def index(self, force_full=False):
        self.indexed = True
        self.store.index.ntotal = 5
        stats = SimpleNamespace(
            files_scanned=2,
            files_changed=2,
            files_removed=0,
            chunks_added=5,
            chunks_removed=0,
            duration_seconds=0.1,
            errors=[],
        )
        return stats

    def search(self, query: str, k: int = 5):
        if self.store.index.ntotal == 0:
            return []
        chunk = CodeChunk(
            id="cliv/auth.py::login",
            file_path="cliv/auth.py",
            language="Python",
            content="def login(user, password):\n    return True",
            start_line=1,
            end_line=2,
            symbol_name="login",
            symbol_kind="function",
        )
        return [(chunk, 0.92)]

    def get_stats(self):
        return self.store.stats()


def test_rag_query_tool_properties():
    tool = RAGQueryTool()
    assert tool.name == "rag_query"
    assert "query" in tool.input_schema["required"]
    assert "read" in tool.permissions
    assert tool.risk_level == "low"


def test_rag_query_executes_and_auto_indexes():
    tool = RAGQueryTool()
    fake_indexer = FakeIndexer(initial_total=0)
    tool._indexer = fake_indexer

    result = tool.execute(query="login function")
    assert fake_indexer.indexed is True
    assert "cliv/auth.py" in result
    assert "score: 0.920" in result
    assert "def login" in result


def test_rag_query_handles_empty_search():
    tool = RAGQueryTool()
    fake_indexer = FakeIndexer(initial_total=5)
    fake_indexer.search = lambda query, k=5: []
    tool._indexer = fake_indexer

    result = tool.execute(query="nonexistent")
    assert "No relevant code found" in result
    assert "search_codebase" in result


def test_rag_index_tool():
    tool = RAGIndexTool()
    fake_indexer = FakeIndexer(initial_total=0)
    tool._indexer = fake_indexer

    result = tool.execute(force=True)
    assert fake_indexer.indexed is True
    assert "RAG Indexing Complete" in result
    assert "Files scanned: 2" in result


def test_rag_stats_tool():
    tool = RAGStatsTool()
    fake_indexer = FakeIndexer(initial_total=10)
    tool._indexer = fake_indexer

    result = tool.execute()
    assert "Total chunks: 10" in result
    assert "Vector dimension: 128" in result
