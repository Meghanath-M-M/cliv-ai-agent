"""Project DNA: repository scanning, symbol extraction, and dependency graphs."""

from cliv.intelligence.code_parser import (
    ImportRecord,
    ModuleRecord,
    SymbolRecord,
    parse_project,
)
from cliv.intelligence.dependency_graph import DependencyGraph, build_dependency_graph
from cliv.intelligence.project_scanner import ScanResult, SourceFile, scan_project
from cliv.intelligence.report import (
    ProjectIntelligence,
    analyse_project,
    build_project_intelligence,
)
from cliv.intelligence.graph_explorer import GraphExplorer, run_graph_explorer
from cliv.intelligence.embeddings import CodeChunk, EmbeddingBackend, get_embedding_backend
from cliv.intelligence.vector_store import VectorStore, VectorStorePersistence
from cliv.intelligence.rag_indexer import RAGIndexer, create_rag_indexer, IndexStats

__all__ = [
    "CodeChunk",
    "DependencyGraph",
    "EmbeddingBackend",
    "GraphExplorer",
    "ImportRecord",
    "IndexStats",
    "ModuleRecord",
    "ProjectIntelligence",
    "RAGIndexer",
    "ScanResult",
    "SourceFile",
    "SymbolRecord",
    "VectorStore",
    "VectorStorePersistence",
    "analyse_project",
    "build_dependency_graph",
    "build_project_intelligence",
    "chunk_module",
    "chunk_modules",
    "create_rag_indexer",
    "get_embedding_backend",
    "parse_project",
    "run_graph_explorer",
    "scan_project",
]
