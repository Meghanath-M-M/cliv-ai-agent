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

__all__ = [
    "DependencyGraph",
    "ImportRecord",
    "ModuleRecord",
    "ProjectIntelligence",
    "ScanResult",
    "SourceFile",
    "SymbolRecord",
    "analyse_project",
    "build_dependency_graph",
    "build_project_intelligence",
    "parse_project",
    "scan_project",
]
