"""Vector store for RAG Project Memory.

Uses FAISS for efficient similarity search with metadata persistence.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from cliv.intelligence.embeddings import CodeChunk, EmbeddingBackend
from cliv.security import project_root


@dataclass
class VectorStore:
    """FAISS-based vector store with metadata."""

    dimension: int
    index: "faiss.Index" = field(init=False, repr=False)
    chunks: List[CodeChunk] = field(default_factory=list)
    chunk_ids: List[str] = field(default_factory=list)
    file_hashes: Dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self):
        import faiss
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine with normalized vectors)

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2 normalize vectors for cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vectors / norms

    def add(self, chunks: List[CodeChunk], vectors: np.ndarray) -> None:
        """Add chunks and their vectors to the store."""
        vectors = self._normalize(vectors.astype(np.float32))
        self.index.add(vectors)
        for chunk in chunks:
            self.chunks.append(chunk)
            self.chunk_ids.append(chunk.id)
        self.updated_at = time.time()

    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[CodeChunk, float]]:
        """Search for top-k similar chunks."""
        if self.index.ntotal == 0:
            return []

        query_vector = self._normalize(query_vector.reshape(1, -1).astype(np.float32))
        scores, indices = self.index.search(query_vector, min(k, self.index.ntotal))

        results: List[Tuple[CodeChunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        return results

    def remove_by_file(self, file_path: str) -> int:
        """Remove all chunks for a file. Returns count removed."""
        # FAISS doesn't support deletion, so we rebuild
        keep_indices = [i for i, cid in enumerate(self.chunk_ids) if not cid.startswith(file_path + "::")]
        removed = len(self.chunks) - len(keep_indices)

        if removed > 0:
            self.chunks = [self.chunks[i] for i in keep_indices]
            self.chunk_ids = [self.chunk_ids[i] for i in keep_indices]
            self._rebuild_index()
        return removed

    def _rebuild_index(self) -> None:
        """Rebuild FAISS index from current chunks (requires re-embedding)."""
        import faiss
        self.index = faiss.IndexFlatIP(self.dimension)
        # Note: vectors not stored, would need re-embedding
        # This is a limitation - for full rebuild use RAGIndexer

    def get_file_chunks(self, file_path: str) -> List[CodeChunk]:
        """Get all chunks for a file."""
        return [c for c in self.chunks if c.file_path == file_path]

    def stats(self) -> dict:
        """Get store statistics."""
        return {
            "total_chunks": len(self.chunks),
            "unique_files": len(set(c.file_path for c in self.chunks)),
            "index_size": self.index.ntotal,
            "dimension": self.dimension,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class VectorStorePersistence:
    """Handle saving/loading vector stores to disk."""

    def __init__(self, store_dir: Optional[Path] = None):
        root = project_root()
        self.store_dir = store_dir or (root / ".cliv" / "rag")
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.store_dir / "index.faiss"
        self.meta_path = self.store_dir / "metadata.json"
        self.chunks_path = self.store_dir / "chunks.pkl"
        self.hashes_path = self.store_dir / "file_hashes.json"

    def save(self, store: VectorStore) -> None:
        """Save vector store to disk."""
        import faiss

        # Save FAISS index
        faiss.write_index(store.index, str(self.index_path))

        # Save metadata
        meta = {
            "dimension": store.dimension,
            "created_at": store.created_at,
            "updated_at": store.updated_at,
            "chunk_ids": store.chunk_ids,
        }
        self.meta_path.write_text(json.dumps(meta))

        # Save chunks (pickle for complex objects)
        with open(self.chunks_path, "wb") as f:
            pickle.dump(store.chunks, f)

        # Save file hashes
        self.hashes_path.write_text(json.dumps(store.file_hashes))

    def load(self, dimension: int) -> Optional[VectorStore]:
        """Load vector store from disk."""
        import faiss

        if not self.index_path.exists() or not self.meta_path.exists():
            return None

        try:
            meta = json.loads(self.meta_path.read_text())
            if meta.get("dimension") != dimension:
                return None  # Dimension mismatch

            index = faiss.read_index(str(self.index_path))

            with open(self.chunks_path, "rb") as f:
                chunks = pickle.load(f)

            file_hashes = {}
            if self.hashes_path.exists():
                file_hashes = json.loads(self.hashes_path.read_text())

            store = VectorStore(dimension=dimension)
            store.index = index
            store.chunks = chunks
            store.chunk_ids = meta.get("chunk_ids", [])
            store.created_at = meta.get("created_at", time.time())
            store.updated_at = meta.get("updated_at", time.time())
            store.file_hashes = file_hashes

            return store
        except Exception:
            return None

    def exists(self) -> bool:
        """Check if store exists."""
        return self.index_path.exists() and self.meta_path.exists()


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of file content."""
    try:
        content = file_path.read_bytes()
        return hashlib.sha256(content).hexdigest()[:16]
    except Exception:
        return ""


import hashlib
import time