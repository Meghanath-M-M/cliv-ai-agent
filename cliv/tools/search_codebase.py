"""Local RAG search with cached, incremental embeddings.

The first search embeds each chunk with Ollama.  Embeddings are cached on
disk keyed by the chunk's SHA-256, so later sessions only embed new or
changed chunks.  Similarity is a single vectorized NumPy operation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None  # type: ignore[assignment]

from cliv.config import CONFIG_DIR
from cliv.security import is_sensitive_path, project_root, resolve_project_path
from cliv.tools.base import BaseTool

IGNORED_DIRS = {".git", ".venv", "__pycache__", "node_modules", ".pytest_cache"}
INDEXED_SUFFIXES = {".py", ".md", ".txt", ".json"}
MAX_FILE_BYTES = 1_000_000
CHUNK_SIZE = 1000
CACHE_VERSION = 1


class SearchCodebaseTool(BaseTool):
    name = "search_codebase"
    description = (
        "Search the project for code snippets relevant to a semantic query "
        "using local Ollama embeddings (RAG)."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query, e.g., 'where is the database connection?'",
            }
        },
        "required": ["query"],
    }

    def __init__(
        self,
        cache_path: Optional[Path] = None,
        root: Optional[Path] = None,
        model: str = "nomic-embed-text",
    ):
        self.cache_path = Path(cache_path) if cache_path else CONFIG_DIR / "embeddings.json"
        self.root = Path(root).resolve() if root else None
        self.model = model
        self.collection: List[Dict[str, Any]] = []
        self.is_indexed = False

    def invalidate(self) -> None:
        self.collection = []
        self.is_indexed = False

    def _get_ollama(self):
        try:
            import ollama
        except ImportError:
            return None
        return ollama

    def _embed(self, text: str) -> List[float]:
        ollama = self._get_ollama()
        if ollama is None:
            raise RuntimeError(
                "ollama is required for embeddings. Run `uv sync --extra offline`."
            )
        response = ollama.embeddings(model=self.model, prompt=text)
        embedding = response["embedding"] if isinstance(response, dict) else response.embedding
        return list(embedding)

    def _cache_key(self, relative_path: str, chunk: str) -> str:
        digest = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
        return f"{relative_path}::{digest}"

    def _load_cache(self, root: Path) -> Dict[str, Any]:
        if not self.cache_path.exists():
            return {}
        try:
            with open(self.cache_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload.get("version") != CACHE_VERSION:
                return {}
            if payload.get("root") != str(root):
                return {}
            entries = payload.get("entries")
            return entries if isinstance(entries, dict) else {}
        except Exception as exc:  # noqa: BLE001
            logging.warning("Could not read embedding cache: %s", exc)
            return {}

    def _save_cache(self, root: Path, entries: Dict[str, Any]) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": CACHE_VERSION,
                "root": str(root),
                "entries": entries,
            }
            temp_path = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
            with open(temp_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            os.replace(temp_path, self.cache_path)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Could not save embedding cache: %s", exc)

    def _iter_files(self, root: Path):
        try:
            cache_path = self.cache_path.resolve()
        except OSError:
            cache_path = self.cache_path
        for current_root, dirs, files in os.walk(root):
            dirs[:] = [
                directory
                for directory in dirs
                if directory not in IGNORED_DIRS
                and not is_sensitive_path(Path(current_root) / directory)
            ]
            for filename in files:
                path = Path(current_root) / filename
                if path.suffix.lower() not in INDEXED_SUFFIXES:
                    continue
                if is_sensitive_path(path):
                    continue
                # Follow symlinks only when they stay inside the project.
                _, path_error = resolve_project_path(str(path), root=root)
                if path_error:
                    continue
                # Never index the cache file itself when a caller put it in
                # the project directory (tests do this deliberately).
                try:
                    if path.resolve() == cache_path:
                        continue
                except OSError:
                    pass
                yield path

    def _build_index(self) -> Optional[str]:
        if np is None:
            return "Error: numpy is required for semantic search. Run `uv sync --extra offline`."
        if self._get_ollama() is None:
            return "Error: ollama is required for semantic search. Run `uv sync --extra offline`."

        root = (self.root or project_root()).resolve()
        cache = self._load_cache(root)
        fresh_cache: Dict[str, Any] = {}
        collection: List[Dict[str, Any]] = []
        embedding_failed = False

        for path in self._iter_files(root):
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
                content = path.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:  # noqa: BLE001
                logging.warning("Failed to read %s: %s", path, exc)
                continue

            relative_path = os.path.relpath(path, root)
            chunks = [
                content[start : start + CHUNK_SIZE]
                for start in range(0, len(content), CHUNK_SIZE)
            ]
            for chunk in chunks:
                key = self._cache_key(relative_path, chunk)
                cached = cache.get(key)
                if isinstance(cached, dict) and isinstance(cached.get("embedding"), list):
                    embedding = cached["embedding"]
                else:
                    try:
                        embedding = self._embed(chunk)
                    except Exception as exc:  # noqa: BLE001
                        logging.warning("Failed to embed %s: %s", relative_path, exc)
                        embedding_failed = True
                        continue
                fresh_cache[key] = {"embedding": embedding}
                collection.append(
                    {
                        "filepath": relative_path,
                        "content": chunk,
                        "embedding": embedding,
                    }
                )

        if not collection and embedding_failed:
            return (
                "Error: Ollama embeddings failed. Make sure Ollama is running "
                "and `nomic-embed-text` is pulled."
            )

        self.collection = collection
        self.is_indexed = True
        self._save_cache(root, fresh_cache)
        return None

    def execute(self, query: str, *args: Any, **kwargs: Any) -> str:
        if np is None:
            return "Error: numpy is required for vector math."

        if not self.is_indexed:
            error = self._build_index()
            if error:
                return error

        if not self.collection:
            return "No code files found to index."

        try:
            query_vec = np.asarray(self._embed(query), dtype=float)
            matrix = np.asarray(
                [item["embedding"] for item in self.collection], dtype=float
            )
        except Exception as exc:  # noqa: BLE001
            return f"Error generating query embedding: {exc}"

        if matrix.ndim != 2 or query_vec.ndim != 1:
            return "Error: embedding cache contains malformed vectors."

        norms = np.linalg.norm(matrix, axis=1)
        query_norm = np.linalg.norm(query_vec)
        denominator = norms * query_norm
        denominator[denominator == 0] = 1e-12
        scores = (matrix @ query_vec) / denominator

        top_indices = np.argsort(scores)[::-1][:3]
        output = f"Top 3 codebase matches for '{query}':\n\n"
        for index in top_indices:
            item = self.collection[int(index)]
            output += (
                f"--- File: {item['filepath']} "
                f"(Relevance Score: {scores[int(index)]:.2f}) ---\n"
                f"{item['content']}\n\n"
            )
        return output
