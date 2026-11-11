"""RAG Indexer: builds and maintains the project memory index.

Handles incremental updates by tracking file hashes and re-indexing only changed files.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Set

from cliv.intelligence import build_project_intelligence, ModuleRecord
from cliv.intelligence.embeddings import CodeChunk, EmbeddingBackend, chunk_modules, get_embedding_backend
from cliv.intelligence.vector_store import VectorStore, VectorStorePersistence, compute_file_hash
from cliv.security import project_root


@dataclass
class IndexStats:
    """Statistics from an indexing operation."""

    files_scanned: int = 0
    files_changed: int = 0
    files_removed: int = 0
    chunks_added: int = 0
    chunks_removed: int = 0
    duration_seconds: float = 0.0
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class RAGIndexer:
    """Builds and maintains the RAG index for a project."""

    def __init__(
        self,
        backend: Optional[EmbeddingBackend] = None,
        persist_dir: Optional[Path] = None,
        max_chunk_lines: int = 100,
    ):
        self.backend = backend or get_embedding_backend()
        self.persistence = VectorStorePersistence(persist_dir)
        self.max_chunk_lines = max_chunk_lines
        self.store: Optional[VectorStore] = None

    def _load_or_create_store(self) -> VectorStore:
        """Load existing store or create new one."""
        if self.store is not None:
            return self.store

        store = self.persistence.load(self.backend.dimension)
        if store is None:
            store = VectorStore(dimension=self.backend.dimension)
        self.store = store
        return store

    def _get_project_modules(self) -> List[ModuleRecord]:
        """Get all modules from Project DNA."""
        intelligence = build_project_intelligence(project_root())
        return intelligence.modules

    def _get_current_file_hashes(self, modules: Sequence[ModuleRecord]) -> dict:
        """Compute hashes for all tracked files."""
        hashes = {}
        root = project_root()
        for module in modules:
            if module.language in {"Python", "JavaScript", "TypeScript"}:
                file_path = root / module.relative
                if file_path.exists():
                    hashes[module.relative] = compute_file_hash(file_path)
        return hashes

    def _find_changes(
        self,
        modules: Sequence[ModuleRecord],
        old_hashes: dict,
    ) -> tuple[Set[str], Set[str], Set[str]]:
        """Find added, modified, and removed files."""
        current_hashes = self._get_current_file_hashes(modules)
        current_files = set(current_hashes.keys())
        old_files = set(old_hashes.keys())

        added = current_files - old_files
        removed = old_files - current_files
        modified = {f for f in current_files & old_files if current_hashes[f] != old_hashes[f]}

        return added, modified, removed

    def index(self, force_full: bool = False) -> IndexStats:
        """Build or update the index.

        Args:
            force_full: If True, rebuild entire index from scratch

        Returns:
            IndexStats with results
        """
        start_time = time.time()
        stats = IndexStats()

        try:
            store = self._load_or_create_store()
            modules = self._get_project_modules()
            stats.files_scanned = len(modules)

            if force_full or not self.persistence.exists():
                # Full rebuild
                return self._full_rebuild(modules, store, stats, start_time)

            # Incremental update
            added, modified, removed = self._find_changes(modules, store.file_hashes)

            # Handle removed files
            for file_path in removed:
                store.remove_by_file(file_path)
                stats.chunks_removed += 1
            stats.files_removed = len(removed)

            # Handle added/modified files
            changed_files = added | modified
            stats.files_changed = len(changed_files)

            if changed_files:
                # Get modules for changed files
                changed_modules = [m for m in modules if m.relative in changed_files]
                chunks = chunk_modules(changed_modules, self.max_chunk_lines)

                if chunks:
                    texts = [c.to_text() for c in chunks]
                    vectors = self.backend.embed(texts)
                    store.add(chunks, vectors)
                    stats.chunks_added = len(chunks)

            # Update file hashes
            store.file_hashes = self._get_current_file_hashes(modules)
            store.updated_at = time.time()

            # Persist
            self.persistence.save(store)

        except Exception as e:
            stats.errors.append(str(e))

        stats.duration_seconds = time.time() - start_time
        return stats

    def _full_rebuild(
        self,
        modules: Sequence[ModuleRecord],
        store: VectorStore,
        stats: IndexStats,
        start_time: float,
    ) -> IndexStats:
        """Perform full index rebuild."""
        # Chunk all modules
        chunks = chunk_modules(modules, self.max_chunk_lines)

        if chunks:
            # Generate embeddings in batches
            batch_size = 32
            all_vectors = []

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                texts = [c.to_text() for c in batch]
                vectors = self.backend.embed(texts)
                all_vectors.append(vectors)

            if all_vectors:
                import numpy as np
                all_vectors = np.vstack(all_vectors)
                store.add(chunks, all_vectors)
                stats.chunks_added = len(chunks)

        # Update file hashes
        store.file_hashes = self._get_current_file_hashes(modules)
        store.updated_at = time.time()
        store.created_at = time.time()

        # Persist
        self.persistence.save(store)

        stats.duration_seconds = time.time() - start_time
        return stats

    def search(self, query: str, k: int = 10) -> List[tuple[CodeChunk, float]]:
        """Search the index for relevant chunks."""
        store = self._load_or_create_store()
        if store.index.ntotal == 0:
            return []

        query_vector = self.backend.embed_single(query)
        return store.search(query_vector, k)

    def get_stats(self) -> dict:
        """Get index statistics."""
        store = self._load_or_create_store()
        return store.stats()

    def clear(self) -> None:
        """Clear the index."""
        self.store = VectorStore(dimension=self.backend.dimension)
        self.persistence.save(self.store)


def create_rag_indexer(
    prefer_offline: bool = False,
    model: Optional[str] = None,
    persist_dir: Optional[Path] = None,
) -> RAGIndexer:
    """Factory function to create RAGIndexer with appropriate backend."""
    backend = get_embedding_backend(prefer_offline=prefer_offline, model=model)
    return RAGIndexer(backend=backend, persist_dir=persist_dir)