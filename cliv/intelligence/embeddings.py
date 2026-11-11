"""Embedding generation for RAG Project Memory.

Supports sentence-transformers (local CPU/GPU) and Ollama (local server).
"""

from __future__ import annotations

import hashlib
import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from cliv.intelligence.code_parser import ModuleRecord
from cliv.security import project_root


@dataclass
class CodeChunk:
    """A chunk of code with metadata for embedding."""

    id: str
    file_path: str
    language: str
    content: str
    start_line: int
    end_line: int
    symbol_name: Optional[str] = None
    symbol_kind: Optional[str] = None

    def to_text(self) -> str:
        """Convert chunk to text for embedding."""
        parts = [f"File: {self.file_path}"]
        if self.symbol_name:
            parts.append(f"Symbol: {self.symbol_name} ({self.symbol_kind})")
        parts.append(f"Lines {self.start_line}-{self.end_line}:")
        parts.append(self.content)
        return "\n".join(parts)


class EmbeddingBackend(ABC):
    """Abstract base for embedding backends."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension."""
        pass

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Generate embeddings for texts. Returns (n, dim) array."""
        pass

    @abstractmethod
    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for single text. Returns (dim,) array."""
        pass


class SentenceTransformersBackend(EmbeddingBackend):
    """sentence-transformers backend (local, CPU/GPU)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name)
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )

    @property
    def dimension(self) -> int:
        self._load_model()
        return self._model.get_sentence_embedding_dimension()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self._load_model()
        return self._model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)

    def embed_single(self, text: str) -> np.ndarray:
        self._load_model()
        return self._model.encode(text, convert_to_numpy=True, show_progress_bar=False)


class OllamaEmbeddingBackend(EmbeddingBackend):
    """Ollama backend for local embedding models."""

    def __init__(self, model: str = "nomic-embed-text", host: Optional[str] = None):
        self._model = model
        self._host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._dimension = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            # Probe dimension with a test embedding
            test_vec = self.embed_single("test")
            self._dimension = len(test_vec)
        return self._dimension

    def _request_embedding(self, text: str) -> np.ndarray:
        import json
        import urllib.request

        url = f"{self._host}/api/embeddings"
        data = json.dumps({"model": self._model, "prompt": text}).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
        return np.array(result["embedding"], dtype=np.float32)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = [self._request_embedding(t) for t in texts]
        return np.stack(vectors)

    def embed_single(self, text: str) -> np.ndarray:
        return self._request_embedding(text)


def get_embedding_backend(
    prefer_offline: bool = False,
    model: Optional[str] = None,
) -> EmbeddingBackend:
    """Get the best available embedding backend.

    Args:
        prefer_offline: If True, prefer Ollama over sentence-transformers
        model: Specific model name to use

    Returns:
        EmbeddingBackend instance
    """
    if prefer_offline:
        try:
            backend = OllamaEmbeddingBackend(model=model or "nomic-embed-text")
            # Test it works
            _ = backend.embed_single("test")
            return backend
        except Exception:
            pass

    try:
        backend = SentenceTransformersBackend(model_name=model or "sentence-transformers/all-MiniLM-L6-v2")
        _ = backend.embed_single("test")
        return backend
    except Exception:
        pass

    # Fallback to Ollama
    try:
        backend = OllamaEmbeddingBackend(model=model or "nomic-embed-text")
        _ = backend.embed_single("test")
        return backend
    except Exception as e:
        raise RuntimeError(
            "No embedding backend available. "
            "Install sentence-transformers or run Ollama with nomic-embed-text"
        ) from e


def chunk_module(module: ModuleRecord, max_chunk_lines: int = 100) -> List[CodeChunk]:
    """Split a module into semantic chunks for embedding."""
    chunks: List[CodeChunk] = []

    try:
        root = project_root()
        file_path = root / module.relative
        full_text = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        full_text = ""

    lines = full_text.splitlines()
    if not lines:
        return chunks

    # Create chunks for each symbol
    for symbol in module.symbols:
        start = max(0, symbol.lineno - 1)
        end = min(len(lines), start + max_chunk_lines)
        content = "\n".join(lines[start:end])

        chunk = CodeChunk(
            id=f"{module.relative}::{symbol.name}",
            file_path=module.relative,
            language=module.language,
            content=content,
            start_line=symbol.lineno,
            end_line=end,
            symbol_name=symbol.name,
            symbol_kind=symbol.kind,
        )
        chunks.append(chunk)

    # If no symbols, chunk the whole file
    if not chunks:
        for i in range(0, len(lines), max_chunk_lines):
            end = min(i + max_chunk_lines, len(lines))
            content = "\n".join(lines[i:end])
            chunk = CodeChunk(
                id=f"{module.relative}:{i//max_chunk_lines}",
                file_path=module.relative,
                language=module.language,
                content=content,
                start_line=i + 1,
                end_line=end,
            )
            chunks.append(chunk)

    return chunks


def chunk_modules(modules: Sequence[ModuleRecord], max_chunk_lines: int = 100) -> List[CodeChunk]:
    """Chunk multiple modules."""
    all_chunks: List[CodeChunk] = []
    for module in modules:
        if module.language in {"Python", "JavaScript", "TypeScript"}:
            all_chunks.extend(chunk_module(module, max_chunk_lines))
    return all_chunks