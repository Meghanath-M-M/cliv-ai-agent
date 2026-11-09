"""Tests for the cached, vectorized RAG implementation."""

import json

import pytest

from cliv.tools.search_codebase import SearchCodebaseTool


class EmbedCounter:
    def __init__(self):
        self.calls = []

    def __call__(self, text: str):
        self.calls.append(text)
        return [float(len(text)), 1.0]


def test_search_returns_relevant_chunk_and_skips_secrets(project_dir, tmp_path):
    (project_dir / "alpha.py").write_text("def alpha():\n    return 'alpha'\n")
    (project_dir / "beta.md").write_text("beta documentation\n")
    (project_dir / ".env").write_text("SUPER_SECRET=do-not-index\n")

    tool = SearchCodebaseTool(
        cache_path=tmp_path / "cache.json", root=project_dir
    )
    counter = EmbedCounter()
    tool._embed = counter

    result = tool.execute("alpha function")

    assert "alpha.py" in result
    assert "SUPER_SECRET" not in result
    assert (tmp_path / "cache.json").exists()


def test_second_run_reuses_cached_chunk_embeddings(project_dir, tmp_path):
    (project_dir / "alpha.py").write_text("alpha content\n")
    cache = tmp_path / "cache.json"

    first = SearchCodebaseTool(cache_path=cache, root=project_dir)
    first_counter = EmbedCounter()
    first._embed = first_counter
    first.execute("alpha")

    second = SearchCodebaseTool(cache_path=cache, root=project_dir)
    second_counter = EmbedCounter()
    second._embed = second_counter
    second.execute("alpha")

    # Only the query needs a new embedding on the second run.
    assert len(second_counter.calls) == 1
    assert first_counter.calls


def test_changed_file_is_reembedded(project_dir, tmp_path):
    path = project_dir / "alpha.py"
    path.write_text("alpha one\n")
    cache = tmp_path / "cache.json"

    first = SearchCodebaseTool(cache_path=cache, root=project_dir)
    first._embed = EmbedCounter()
    first.execute("alpha")

    path.write_text("alpha two\n")
    second = SearchCodebaseTool(cache_path=cache, root=project_dir)
    counter = EmbedCounter()
    second._embed = counter
    second.execute("alpha")

    # One new chunk plus the query; the unchanged chunks are cached.
    assert len(counter.calls) == 2


def test_cache_is_scoped_to_project_root(project_dir, tmp_path):
    (project_dir / "alpha.py").write_text("alpha\n")
    cache = tmp_path / "cache.json"
    tool = SearchCodebaseTool(cache_path=cache, root=project_dir)
    tool._embed = EmbedCounter()
    tool.execute("alpha")

    other = tmp_path / "other"
    other.mkdir()
    (other / "beta.py").write_text("beta\n")
    tool2 = SearchCodebaseTool(cache_path=cache, root=other)
    counter = EmbedCounter()
    tool2._embed = counter
    tool2.execute("beta")

    assert any("beta" in call for call in counter.calls)


def test_missing_numpy_returns_error(project_dir, tmp_path, monkeypatch):
    import cliv.tools.search_codebase as module

    monkeypatch.setattr(module, "np", None)
    tool = SearchCodebaseTool(cache_path=tmp_path / "cache.json", root=project_dir)
    assert "numpy" in tool.execute("anything")
