"""Build an internal dependency graph from parsed modules.

Edges point from a module to the modules it imports.  Python imports
are resolved against the project's own module names; JavaScript and
TypeScript relative imports are resolved against source paths.  External
imports are retained separately for dependency-usage checks.
"""

from __future__ import annotations

import posixpath
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

from cliv.intelligence.code_parser import ImportRecord, ModuleRecord

JS_EXTENSIONS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")
JS_BUILTINS = {
    "assert",
    "buffer",
    "child_process",
    "cluster",
    "console",
    "constants",
    "crypto",
    "dgram",
    "diagnostics_channel",
    "dns",
    "domain",
    "events",
    "fs",
    "http",
    "http2",
    "https",
    "inspector",
    "module",
    "net",
    "os",
    "path",
    "perf_hooks",
    "process",
    "punycode",
    "querystring",
    "readline",
    "repl",
    "stream",
    "string_decoder",
    "timers",
    "tls",
    "trace_events",
    "tty",
    "url",
    "util",
    "v8",
    "vm",
    "wasi",
    "worker_threads",
    "zlib",
}


@dataclass
class DependencyGraph:
    """Internal edges plus imports that leave the repository."""

    edges: Dict[str, Set[str]] = field(default_factory=dict)
    external: Dict[str, Set[str]] = field(default_factory=dict)
    unresolved: Dict[str, Set[str]] = field(default_factory=dict)
    cycles: List[List[str]] = field(default_factory=list)

    @property
    def edge_count(self) -> int:
        return sum(len(targets) for targets in self.edges.values())

    def imported_external(self) -> Set[str]:
        return {
            name
            for names in self.external.values()
            for name in names
        }


def _relative_package_parts(module: ModuleRecord) -> List[str]:
    name = module.python_name
    if not name:
        return []
    parts = name.split(".")
    return parts if module.is_package_init else parts[:-1]


def _absolute_parts(import_record: ImportRecord) -> List[str]:
    return import_record.module.split(".") if import_record.module else []


def _python_base_parts(module: ModuleRecord, import_record: ImportRecord) -> List[str]:
    if not import_record.is_relative:
        return _absolute_parts(import_record)

    package_parts = _relative_package_parts(module)
    parent_levels = max(import_record.level - 1, 0)
    if parent_levels >= len(package_parts):
        base: List[str] = []
    else:
        base = package_parts[: len(package_parts) - parent_levels]
    if import_record.module:
        base = base + import_record.module.split(".")
    return base


def _resolve_python(
    module: ModuleRecord,
    import_record: ImportRecord,
    by_python_name: Dict[str, str],
) -> Set[str]:
    """Resolve one Python import to project-relative paths."""

    base_parts = _python_base_parts(module, import_record)
    if not base_parts and not import_record.is_relative:
        return set()

    # Prefer a named submodule over its package.  This turns
    # ``from pkg import mod`` into an edge to ``pkg/mod.py`` instead of
    # only to ``pkg/__init__.py``.
    named_candidates = []
    for name in import_record.names:
        if name in {"*", ""}:
            continue
        named_candidates.append(".".join([*base_parts, name]))
    resolved_named = {
        by_python_name[candidate]
        for candidate in named_candidates
        if candidate in by_python_name
    }
    if resolved_named:
        return resolved_named

    base_name = ".".join(base_parts)
    if base_name in by_python_name:
        return {by_python_name[base_name]}
    return set()


def _is_python_stdlib(name: str) -> bool:
    top_level = name.split(".", 1)[0]
    return top_level in sys.stdlib_module_names or top_level in {
        "__future__",
        "builtins",
    }


def _external_python_name(import_record: ImportRecord) -> Optional[str]:
    if import_record.is_relative or not import_record.module:
        return None
    name = import_record.module
    if _is_python_stdlib(name):
        return None
    return name.split(".", 1)[0]


def _resolve_js(
    module: ModuleRecord,
    import_record: ImportRecord,
    by_path: Set[str],
) -> Set[str]:
    specifier = import_record.module
    if not specifier.startswith("."):
        return set()

    base = posixpath.normpath(
        posixpath.join(posixpath.dirname(module.relative), specifier)
    )
    if base.startswith("../") or base == "..":
        return set()

    candidates = [base]
    if not any(base.endswith(ext) for ext in JS_EXTENSIONS):
        candidates.extend(base + ext for ext in JS_EXTENSIONS)
        candidates.extend(
            posixpath.join(base, "index" + ext) for ext in JS_EXTENSIONS
        )
    return {candidate for candidate in candidates if candidate in by_path}


def _js_external_name(specifier: str) -> Optional[str]:
    if specifier.startswith((".", "/", "node:")):
        return None
    if specifier in JS_BUILTINS:
        return None
    if specifier.startswith("@"):
        parts = specifier.split("/")
        return "/".join(parts[:2]) if len(parts) >= 2 else specifier
    return specifier.split("/", 1)[0]


def _python_import_is_internal(
    import_record: ImportRecord, internal_roots: Set[str]
) -> bool:
    if import_record.is_relative:
        return True
    if not import_record.module:
        return False
    return import_record.module.split(".", 1)[0] in internal_roots


def _strongly_connected_components(graph: Dict[str, Set[str]]) -> List[List[str]]:
    """Tarjan's algorithm, returning cycles in deterministic order."""

    index = 0
    indices: Dict[str, int] = {}
    lowlinks: Dict[str, int] = {}
    stack: List[str] = []
    on_stack: Set[str] = set()
    components: List[List[str]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for target in sorted(graph.get(node, ())):
            if target not in indices:
                visit(target)
                lowlinks[node] = min(lowlinks[node], lowlinks[target])
            elif target in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[target])

        if lowlinks[node] == indices[node]:
            component: List[str] = []
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.append(member)
                if member == node:
                    break
            if len(component) > 1 or node in graph.get(node, set()):
                components.append(sorted(component))

    for node in sorted(graph):
        if node not in indices:
            visit(node)
    return sorted(components)


PYTHON_SOURCE_ROOTS = {"src", "lib", "source"}


def _python_import_names(module: ModuleRecord) -> Set[str]:
    """Return acceptable import names for a module.

    A module under ``src/pkg/mod.py`` is importable both as
    ``src.pkg.mod`` and, in a src-layout project, as ``pkg.mod``.
    """

    name = module.python_name
    if not name:
        return set()
    names = {name}
    parts = name.split(".")
    if len(parts) > 1 and parts[0] in PYTHON_SOURCE_ROOTS:
        names.add(".".join(parts[1:]))
    return names


def build_dependency_graph(modules: Sequence[ModuleRecord]) -> DependencyGraph:
    """Resolve imports and return the project's internal dependency graph."""

    graph = DependencyGraph()
    by_python_name: Dict[str, str] = {}
    for module in modules:
        for name in _python_import_names(module):
            by_python_name[name] = module.relative
    by_path = {module.relative for module in modules}
    internal_roots = {
        name.split(".", 1)[0] for name in by_python_name if name
    }

    for module in modules:
        source = module.relative
        graph.edges.setdefault(source, set())
        graph.external.setdefault(source, set())
        graph.unresolved.setdefault(source, set())

        for import_record in module.imports:
            if module.language == "Python":
                internal = _python_import_is_internal(
                    import_record, internal_roots
                )
                if import_record.is_type_checking:
                    # Type-only imports are real dependencies, but they do
                    # not participate in runtime module cycles.
                    if not internal:
                        external = _external_python_name(import_record)
                        if external:
                            graph.external[source].add(external)
                    continue
                resolved = _resolve_python(module, import_record, by_python_name)
                graph.edges[source].update(resolved)
                if resolved:
                    continue
                if internal:
                    label = import_record.module or "."
                    graph.unresolved[source].add(label)
                else:
                    external = _external_python_name(import_record)
                    if external:
                        graph.external[source].add(external)
            elif module.language in {"JavaScript", "TypeScript"}:
                resolved = _resolve_js(module, import_record, by_path)
                graph.edges[source].update(resolved)
                if resolved:
                    continue
                if import_record.module.startswith("."):
                    graph.unresolved[source].add(import_record.module)
                else:
                    external = _js_external_name(import_record.module)
                    if external:
                        graph.external[source].add(external)

    graph.cycles = _strongly_connected_components(graph.edges)
    return graph
