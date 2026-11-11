"""Turn scans, parsed modules, and dependency edges into Project DNA."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Sequence, Set, Tuple

from cliv.intelligence.code_parser import ModuleRecord, parse_project
from cliv.intelligence.dependency_graph import DependencyGraph, build_dependency_graph
from cliv.intelligence.project_scanner import (
    IMPORT_ALIASES,
    TOOLING_DEPENDENCIES,
    ScanResult,
    normalise_dependency_name,
    scan_project,
)

MAX_COMPONENTS = 12
MAX_ISSUES = 20
MAX_EDGES = 12
MAX_ENTRY_POINTS = 10

DUPLICATE_FUNCTION_IGNORE = {
    "main",
    "run",
    "setup",
    "configure",
    "create_app",
    "handler",
    "index",
}

CONFIG_FILE_NAMES = {
    "setup.py",
    "conftest.py",
    "__main__.py",
    "manage.py",
    "settings.py",
    "wsgi.py",
    "asgi.py",
}

ROLE_ORDER = [
    "Frontend",
    "Backend",
    "AI Integration",
    "Database",
    "Authentication",
    "Security",
    "Tooling",
    "Testing",
    "Application",
]

ROLE_KEYWORDS: List[Tuple[str, Tuple[str, ...]]] = [
    ("Authentication", ("auth", "login", "permission")),
    ("Security", ("security", "secret", "credential", "audit")),
    (
        "AI Integration",
        (
            "llm",
            "ai",
            "ml",
            "intelligence",
            "inference",
            "embedding",
            "rag",
            "nlp",
            "vision",
            "agent",
        ),
    ),
    (
        "Database",
        (
            "database",
            "db",
            "schema",
            "migration",
            "repository",
            "storage",
            "persistence",
            "sql",
        ),
    ),
    (
        "Frontend",
        (
            "frontend",
            "front-end",
            "ui",
            "web",
            "client",
            "component",
            "page",
            "view",
            "static",
            "template",
            "style",
        ),
    ),
    (
        "Backend",
        (
            "backend",
            "back-end",
            "server",
            "api",
            "route",
            "controller",
            "service",
            "handler",
            "endpoint",
        ),
    ),
    ("Testing", ("test", "tests", "spec")),
    ("Tooling", ("tool", "tools", "script", "scripts", "cli", "command")),
]

ACRONYMS = {
    "ai": "AI",
    "api": "API",
    "cli": "CLI",
    "db": "DB",
    "llm": "LLM",
    "ml": "ML",
    "rag": "RAG",
    "ui": "UI",
}

GENERIC_COMPONENT_KEYS = {
    "(root)",
    "src",
    "source",
    "sources",
    "app",
    "apps",
    "lib",
    "libs",
    "pkg",
    "packages",
    "main",
}


@dataclass(frozen=True)
class Issue:
    severity: str
    category: str
    message: str


@dataclass
class Component:
    key: str
    name: str
    role: str
    files: List[str]
    symbols: int
    internal_edges: int


@dataclass
class ProjectIntelligence:
    scan: ScanResult
    modules: List[ModuleRecord]
    graph: DependencyGraph
    components: List[Component]
    entry_points: List[str]
    issues: List[Issue]
    class_count: int = 0
    function_count: int = 0
    method_count: int = 0

    @property
    def source_files(self) -> int:
        return len(self.scan.files)

    @property
    def symbol_count(self) -> int:
        return sum(len(module.symbols) for module in self.modules)

    @property
    def external_packages(self) -> List[str]:
        return sorted(self.graph.imported_external())

    def to_dict(self) -> Dict[str, Any]:
        """Return structured dictionary representation for machine-readable output."""
        return {
            "project_name": self.scan.project_name,
            "root": str(self.scan.root),
            "languages": dict(self.scan.languages),
            "source_files": self.source_files,
            "symbol_count": self.symbol_count,
            "class_count": self.class_count,
            "function_count": self.function_count,
            "method_count": self.method_count,
            "entry_points": list(self.entry_points),
            "external_packages": self.external_packages,
            "components": [
                {
                    "key": c.key,
                    "name": c.name,
                    "role": c.role,
                    "files": list(c.files),
                    "symbols": c.symbols,
                    "internal_edges": c.internal_edges,
                }
                for c in self.components
            ],
            "issues": [
                {
                    "severity": i.severity,
                    "category": i.category,
                    "message": i.message,
                }
                for i in self.issues
            ],
        }

    def render(self) -> str:
        scan = self.scan
        lines: List[str] = [
            "CLIV PROJECT INTELLIGENCE",
            "",
            f"Project: {scan.project_name}",
            f"Root: {scan.root}",
            "",
            "Languages:",
        ]

        if scan.languages:
            for language, count in sorted(
                scan.languages.items(), key=lambda item: (-item[1], item[0])
            ):
                suffix = "file" if count == 1 else "files"
                lines.append(f"  {language}: {count} {suffix}")
        else:
            lines.append("  (none detected)")

        lines.extend(["", "Architecture:"])
        roles = self.architecture
        for role in roles:
            lines.append(f"  {role}")

        lines.extend(
            [
                "",
                "Project Statistics:",
                f"  Files analysed: {scan.all_file_count}",
                f"  Source files: {self.source_files}",
                f"  Python files: {scan.languages.get('Python', 0)}",
                f"  Declared dependencies: {len(scan.declared_dependencies)}",
                f"  Entry points: {len(self.entry_points)}",
                f"  Symbols: {self.symbol_count} "
                f"(classes: {self.class_count}, functions: {self.function_count}, "
                f"methods: {self.method_count})",
                f"  Internal dependency edges: {self.graph.edge_count}",
            ]
        )

        lines.extend(["", "Detected Components:"])
        if self.components:
            for index, component in enumerate(
                self.components[:MAX_COMPONENTS], start=1
            ):
                suffix = "file" if len(component.files) == 1 else "files"
                symbol_suffix = "symbol" if component.symbols == 1 else "symbols"
                lines.append(
                    f"  [{index}] {component.name} - {len(component.files)} {suffix}, "
                    f"{component.symbols} {symbol_suffix}"
                )
                lines.append(
                    f"      Path: {component.key} | Role: {component.role}"
                )
                samples = component.files[:4]
                more = len(component.files) - len(samples)
                file_text = ", ".join(samples)
                if more > 0:
                    file_text += f" (+{more} more)"
                lines.append(f"      Files: {file_text}")
            remaining = len(self.components) - MAX_COMPONENTS
            if remaining > 0:
                lines.append(f"  ... and {remaining} more components")
        else:
            lines.append("  (no source components detected)")

        lines.extend(["", "Entry Points:"])
        if self.entry_points:
            for path in self.entry_points[:MAX_ENTRY_POINTS]:
                lines.append(f"  - {path}")
            remaining = len(self.entry_points) - MAX_ENTRY_POINTS
            if remaining > 0:
                lines.append(f"  ... and {remaining} more")
        else:
            lines.append("  (none detected)")

        lines.extend(["", "Dependencies:"])
        lines.append(f"  Declared: {len(scan.declared_dependencies)}")
        lines.append(f"  External imports: {len(self.external_packages)}")
        if self.external_packages:
            visible = self.external_packages[:20]
            suffix = "" if len(visible) == len(self.external_packages) else ", ..."
            lines.append(f"  External packages: {', '.join(visible)}{suffix}")
        unresolved = sum(len(values) for values in self.graph.unresolved.values())
        lines.append(f"  Unresolved internal imports: {unresolved}")

        lines.extend(["", "Dependency Graph:"])
        lines.append(f"  Modules: {len(self.modules)}")
        lines.append(f"  Internal edges: {self.graph.edge_count}")
        if self.graph.cycles:
            lines.append(f"  Cycles: {len(self.graph.cycles)}")
            for cycle in self.graph.cycles[:5]:
                path = " -> ".join([*cycle, cycle[0]])
                lines.append(f"    - {path}")
        else:
            lines.append("  Cycles: none")

        edge_lines = self._top_edges()
        if edge_lines:
            lines.append("  Top internal dependencies:")
            for edge in edge_lines:
                lines.append(f"    - {edge}")

        lines.extend(["", "Potential Issues:"])
        if self.issues:
            for issue in self.issues[:MAX_ISSUES]:
                lines.append(
                    f"  - [{issue.severity}] {issue.message}"
                )
            remaining = len(self.issues) - MAX_ISSUES
            if remaining > 0:
                lines.append(f"  ... and {remaining} more")
        else:
            lines.append("  No obvious structural issues detected.")

        lines.extend(
            [
                "",
                f"Architecture understanding: {self.understanding}",
            ]
        )
        return "\n".join(lines)

    @property
    def architecture(self) -> List[str]:
        found = {component.role for component in self.components}
        ordered = [role for role in ROLE_ORDER if role in found]
        return ordered or ["Application"]

    @property
    def understanding(self) -> str:
        problems: List[str] = []
        parse_errors = sum(1 for module in self.modules if module.parse_error)
        if parse_errors:
            problems.append(f"{parse_errors} parse error(s)")
        if self.scan.truncated:
            problems.append("scan truncated")
        if self.scan.manifest_errors:
            problems.append("manifest warning(s)")
        if not self.source_files:
            return "No supported source files detected"
        if problems:
            return "Partial (" + ", ".join(problems) + ")"
        return "Complete"

    def _top_edges(self) -> List[str]:
        edges = [
            (source, target)
            for source, targets in self.graph.edges.items()
            for target in targets
            if source != target
        ]
        edges.sort()
        return [
            f"{source} -> {target}" for source, target in edges[:MAX_EDGES]
        ]


def _has_keyword(text: str, keyword: str) -> bool:
    # Accept the common plural form while still avoiding substring matches
    # such as "ai" inside "email" or "test" inside "testing".
    return bool(
        re.search(
            rf"(?<![a-z0-9]){re.escape(keyword)}(?:s|es)?(?![a-z0-9])",
            text,
        )
    )


def _classify_role(key: str, files: Sequence[str]) -> str:
    # A component's own name is the strongest signal.  For generic
    # containers such as ``src`` or root files, inspect the file paths.
    text = key
    if key.lower() in GENERIC_COMPONENT_KEYS:
        text = " ".join([key, *files])
    text = text.lower()
    for role, keywords in ROLE_KEYWORDS:
        if any(_has_keyword(text, keyword) for keyword in keywords):
            return role
    return "Application"


def _component_name(key: str) -> str:
    if key == "(root)":
        return "Root files"
    parts = [part for part in re.split(r"[/_-]+", key) if part]
    titled = [
        ACRONYMS.get(part.lower(), part[:1].upper() + part[1:])
        for part in parts
    ]
    return " / ".join(titled) if titled else key


def _component_key(relative: str, depth: int) -> str:
    parts = PurePosixPath(relative).parts
    if not parts:
        return "(root)"
    if len(parts) == 1:
        return "(root)"
    if depth == 1 or len(parts) == 2:
        return parts[0]
    return f"{parts[0]}/{parts[1]}"


def _choose_depth(modules: Sequence[ModuleRecord]) -> int:
    grouped = defaultdict(int)
    for module in modules:
        parts = PurePosixPath(module.relative).parts
        if len(parts) > 1:
            grouped[parts[0]] += 1
    if not grouped:
        return 1
    largest = max(grouped.values())
    total = sum(grouped.values())
    if largest >= 4 and largest / total >= 0.6:
        return 2
    return 1


def _build_components(
    modules: Sequence[ModuleRecord], graph: DependencyGraph
) -> List[Component]:
    depth = _choose_depth(modules)
    grouped: Dict[str, List[ModuleRecord]] = defaultdict(list)
    for module in modules:
        grouped[_component_key(module.relative, depth)].append(module)

    incoming: Dict[str, int] = defaultdict(int)
    for targets in graph.edges.values():
        for target in targets:
            incoming[target] += 1

    components: List[Component] = []
    for key, grouped_modules in grouped.items():
        files = sorted(module.relative for module in grouped_modules)
        symbol_count = sum(
            1
            for module in grouped_modules
            for symbol in module.symbols
            if symbol.parent is None
        )
        edges = sum(
            len(graph.edges.get(module.relative, ()))
            for module in grouped_modules
        )
        components.append(
            Component(
                key=key,
                name=_component_name(key),
                role=_classify_role(key, files),
                files=files,
                symbols=symbol_count,
                internal_edges=edges,
            )
        )

    components.sort(key=lambda component: (-len(component.files), component.key))
    return components


def _duplicate_functions(modules: Sequence[ModuleRecord]) -> List[Tuple[str, List[str]]]:
    locations: Dict[str, Set[str]] = defaultdict(set)
    for module in modules:
        if module.is_test:
            continue
        for symbol in module.top_level_symbols:
            if (
                symbol.kind == "function"
                and symbol.public
                and symbol.name not in DUPLICATE_FUNCTION_IGNORE
            ):
                locations[symbol.name].add(module.relative)
    return sorted(
        (name, sorted(paths))
        for name, paths in locations.items()
        if len(paths) > 1
    )


def _is_config_module(relative: str) -> bool:
    path = PurePosixPath(relative)
    name = path.name.lower()
    if name in CONFIG_FILE_NAMES:
        return True
    return name.endswith(
        (
            ".config.js",
            ".config.ts",
            ".config.mjs",
            ".config.cjs",
        )
    )


def _build_issues(
    scan: ScanResult,
    modules: Sequence[ModuleRecord],
    graph: DependencyGraph,
) -> List[Issue]:
    issues: List[Issue] = []

    for module in modules:
        if module.parse_error:
            issues.append(
                Issue(
                    "high",
                    "parse-error",
                    f"Could not parse {module.relative}: {module.parse_error}",
                )
            )

    for cycle in graph.cycles:
        cycle_path = " -> ".join([*cycle, cycle[0]])
        issues.append(Issue("high", "import-cycle", f"Import cycle: {cycle_path}"))

    duplicates = _duplicate_functions(modules)
    for name, paths in duplicates[:5]:
        preview = ", ".join(paths[:4])
        more = len(paths) - 4
        if more > 0:
            preview += f" (+{more} more)"
        issues.append(
            Issue(
                "low",
                "duplicate-function",
                f"Public function '{name}' is defined in multiple modules: {preview}",
            )
        )
    if len(duplicates) > 5:
        issues.append(
            Issue(
                "low",
                "duplicate-function",
                f"{len(duplicates) - 5} additional duplicate public function(s) detected",
            )
        )

    if scan.declared_dependencies:
        imported = {
            normalise_dependency_name(name) for name in graph.imported_external()
        }
        dev = {
            normalise_dependency_name(name) for name in scan.dev_dependencies
        }
        tooling = {
            normalise_dependency_name(name) for name in TOOLING_DEPENDENCIES
        }
        unused = []
        for dependency in sorted(scan.declared_dependencies):
            normalised = normalise_dependency_name(dependency)
            if normalised in dev or normalised in tooling:
                continue
            alias = IMPORT_ALIASES.get(normalised)
            if alias and normalise_dependency_name(alias) in imported:
                continue
            if normalised not in imported:
                unused.append(dependency)
        if unused:
            preview = ", ".join(unused[:8])
            more = len(unused) - 8
            if more > 0:
                preview += f" (+{more} more)"
            issues.append(
                Issue(
                    "low",
                    "unused-dependency",
                    f"Declared dependencies not imported in source: {preview}",
                )
            )

    incoming: Dict[str, int] = defaultdict(int)
    for targets in graph.edges.values():
        for target in targets:
            incoming[target] += 1
    isolated: List[str] = []
    for module in modules:
        if module.language not in {"Python", "JavaScript", "TypeScript"}:
            continue
        if module.is_test or module.is_entry_point or module.is_package_init:
            continue
        if _is_config_module(module.relative):
            continue
        if not graph.edges.get(module.relative) and not incoming.get(module.relative):
            isolated.append(module.relative)
    if isolated:
        preview = ", ".join(isolated[:6])
        more = len(isolated) - 6
        if more > 0:
            preview += f" (+{more} more)"
        issues.append(
            Issue(
                "low",
                "isolated-module",
                f"{len(isolated)} module(s) have no internal imports or dependents: {preview}",
            )
        )

    if not any(module.is_test for module in modules):
        issues.append(
            Issue("medium", "no-tests", "No test files detected in the project.")
        )

    if scan.truncated:
        issues.append(
            Issue(
                "medium",
                "scan-truncated",
                "Repository scan was truncated; the report covers only the first source files.",
            )
        )
    if scan.skipped_large_files:
        issues.append(
            Issue(
                "low",
                "large-file",
                f"{len(scan.skipped_large_files)} source file(s) exceeded the size limit and were skipped.",
            )
        )
    for error in scan.manifest_errors:
        issues.append(Issue("medium", "manifest", f"Could not parse manifest: {error}"))

    unresolved = sum(len(values) for values in graph.unresolved.values())
    if unresolved:
        examples = [
            f"{source} -> {name}"
            for source, names in sorted(graph.unresolved.items())
            for name in sorted(names)
        ]
        preview = ", ".join(examples[:5])
        more = unresolved - 5
        if more > 0:
            preview += f" (+{more} more)"
        issues.append(
            Issue(
                "low",
                "unresolved-import",
                f"{unresolved} internal import(s) could not be resolved: {preview}",
            )
        )

    if not scan.files:
        issues.append(
            Issue(
                "high",
                "no-source",
                "No supported source files were found; architecture analysis is empty.",
            )
        )

    return issues


def build_project_intelligence(root: Optional[Path] = None) -> ProjectIntelligence:
    """Run the complete Project DNA pipeline for *root*."""

    scan = scan_project(root)
    modules = parse_project(scan.root, scan)
    graph = build_dependency_graph(modules)

    entry_points = sorted(
        {
            module.relative
            for module in modules
            if module.is_entry_point
        }
        | set(scan.entry_point_candidates)
    )

    class_count = sum(
        1
        for module in modules
        for symbol in module.symbols
        if symbol.kind == "class"
    )
    function_count = sum(
        1
        for module in modules
        for symbol in module.symbols
        if symbol.kind == "function"
    )
    method_count = sum(
        1
        for module in modules
        for symbol in module.symbols
        if symbol.kind == "method"
    )

    return ProjectIntelligence(
        scan=scan,
        modules=modules,
        graph=graph,
        components=_build_components(modules, graph),
        entry_points=entry_points,
        issues=_build_issues(scan, modules, graph),
        class_count=class_count,
        function_count=function_count,
        method_count=method_count,
    )


def analyse_project(root: Optional[Path] = None) -> str:
    """Convenience wrapper returning the rendered report."""

    return build_project_intelligence(root).render()
