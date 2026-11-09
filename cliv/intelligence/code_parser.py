"""Extract symbols and imports from source files.

Python uses the standard-library AST.  JavaScript and TypeScript use a
small set of conservative regular expressions; the goal is an
architecture-level map, not a perfect parser.  Other languages are
counted by the scanner but do not contribute graph edges yet.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import List, Optional, Tuple

from cliv.intelligence.project_scanner import ScanResult, SourceFile

PYTHON_LANGUAGES = {"Python"}
JS_LANGUAGES = {"JavaScript", "TypeScript"}

_JS_IMPORT = re.compile(
    r"(?:\bfrom\s*|\brequire\s*\(\s*|\bimport\s*\(\s*|\bimport\s*)"
    r"['\"]([^'\"]+)['\"]"
)
_JS_FUNCTION = re.compile(
    r"(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s+"
    r"([A-Za-z_$][\w$]*)"
)
_JS_CLASS = re.compile(
    r"(?:export\s+(?:default\s+)?)?class\s+([A-Za-z_$][\w$]*)"
)
_JS_ARROW = re.compile(
    r"(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*"
    r"(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>"
)
_JS_MAIN_GUARD = re.compile(
    r"(?:require\.main\s*===\s*module|import\.meta\.url\s*===\s*"
    r"(?:pathToFileURL\s*\(\s*)?process\.argv\[1\])"
)

_TEST_DIRS = {"test", "tests", "__tests__", "spec", "specs", "testing"}


@dataclass(frozen=True)
class ImportRecord:
    """An import statement discovered in a module."""

    module: str
    names: Tuple[str, ...] = ()
    lineno: int = 0
    is_relative: bool = False
    level: int = 0
    is_type_checking: bool = False


@dataclass(frozen=True)
class SymbolRecord:
    """A class, function, or method discovered in a module."""

    name: str
    kind: str
    lineno: int
    public: bool
    parent: Optional[str] = None


@dataclass
class ModuleRecord:
    """Everything Project DNA knows about one source file."""

    relative: str
    language: str
    imports: List[ImportRecord] = field(default_factory=list)
    symbols: List[SymbolRecord] = field(default_factory=list)
    is_entry_point: bool = False
    is_test: bool = False
    is_package_init: bool = False
    parse_error: Optional[str] = None

    @property
    def python_name(self) -> Optional[str]:
        return python_module_name(self.relative)

    @property
    def top_level_symbols(self) -> List[SymbolRecord]:
        return [symbol for symbol in self.symbols if symbol.parent is None]


def python_module_name(relative: str) -> Optional[str]:
    """Convert ``pkg/mod.py`` or ``pkg/__init__.py`` to a dotted name."""

    path = PurePosixPath(relative)
    if path.suffix.lower() != ".py":
        return None
    parts = list(path.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) if parts else None


def _is_test_file(relative: str) -> bool:
    path = PurePosixPath(relative)
    if any(part.lower() in _TEST_DIRS for part in path.parts[:-1]):
        return True
    name = path.name.lower()
    return (
        name == "conftest.py"
        or name.startswith("test_")
        or name.endswith(("_test.py", ".test.js", ".test.ts", ".test.jsx", ".test.tsx"))
        or name.endswith((".spec.js", ".spec.ts", ".spec.jsx", ".spec.tsx"))
    )


def _is_main_guard(test: ast.expr) -> bool:
    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        return False
    if not isinstance(test.ops[0], ast.Eq):
        return False
    values = [test.left, *test.comparators]
    names = {value.id for value in values if isinstance(value, ast.Name)}
    constants = {
        value.value
        for value in values
        if isinstance(value, ast.Constant) and isinstance(value.value, str)
    }
    return "__name__" in names and "__main__" in constants


def _has_guard(tree: ast.AST) -> bool:
    if not isinstance(tree, ast.Module):
        return False
    for node in tree.body:
        if isinstance(node, ast.If) and _is_main_guard(node.test):
            return True
    return False


def _is_type_checking_test(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


class _ImportCollector(ast.NodeVisitor):
    """Collect imports while distinguishing type-only imports.

    Imports guarded by ``if TYPE_CHECKING:`` are recorded (so their
    external packages still count as dependencies) but skipped when the
    runtime dependency graph is built.
    """

    def __init__(self) -> None:
        self.records: List[ImportRecord] = []
        self._type_checking_depth = 0

    def visit_If(self, node: ast.If) -> None:
        if _is_type_checking_test(node.test):
            self._type_checking_depth += 1
            for child in node.body:
                self.visit(child)
            self._type_checking_depth -= 1
            for child in node.orelse:
                self.visit(child)
            return
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.records.append(
                ImportRecord(
                    module=alias.name,
                    lineno=node.lineno,
                    is_type_checking=self._type_checking_depth > 0,
                )
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.records.append(
            ImportRecord(
                module=node.module or "",
                names=tuple(alias.name for alias in node.names),
                lineno=node.lineno,
                is_relative=node.level > 0,
                level=node.level,
                is_type_checking=self._type_checking_depth > 0,
            )
        )


def _python_imports(tree: ast.AST) -> List[ImportRecord]:
    collector = _ImportCollector()
    collector.visit(tree)
    return collector.records


def _python_symbols(tree: ast.AST) -> List[SymbolRecord]:
    symbols: List[SymbolRecord] = []
    if not isinstance(tree, ast.Module):
        return symbols

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append(
                SymbolRecord(
                    name=node.name,
                    kind="function",
                    lineno=node.lineno,
                    public=not node.name.startswith("_"),
                )
            )
        elif isinstance(node, ast.ClassDef):
            symbols.append(
                SymbolRecord(
                    name=node.name,
                    kind="class",
                    lineno=node.lineno,
                    public=not node.name.startswith("_"),
                )
            )
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.append(
                        SymbolRecord(
                            name=child.name,
                            kind="method",
                            lineno=child.lineno,
                            public=not child.name.startswith("_"),
                            parent=node.name,
                        )
                    )
    return symbols


def _parse_python(
    relative: str,
    text: str,
    is_entry_candidate: bool,
) -> ModuleRecord:
    module = ModuleRecord(
        relative=relative,
        language="Python",
        is_entry_point=is_entry_candidate,
        is_test=_is_test_file(relative),
        is_package_init=PurePosixPath(relative).name == "__init__.py",
    )
    try:
        tree = ast.parse(text, filename=relative)
    except SyntaxError as exc:
        line = f" (line {exc.lineno})" if exc.lineno else ""
        module.parse_error = f"{exc.msg}{line}"
        return module

    module.imports = _python_imports(tree)
    module.symbols = _python_symbols(tree)
    module.is_entry_point = module.is_entry_point or _has_guard(tree)
    return module


def _line_number(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def _js_imports(text: str) -> List[ImportRecord]:
    records: List[ImportRecord] = []
    seen = set()
    for match in _JS_IMPORT.finditer(text):
        module = match.group(1)
        line = _line_number(text, match.start())
        key = (module, line)
        if key in seen:
            continue
        seen.add(key)
        records.append(
            ImportRecord(
                module=module,
                lineno=line,
                is_relative=module.startswith((".", "/")),
            )
        )
    return records


def _js_symbols(text: str) -> List[SymbolRecord]:
    symbols: List[SymbolRecord] = []
    seen = set()
    for kind, pattern in (
        ("function", _JS_FUNCTION),
        ("class", _JS_CLASS),
        ("function", _JS_ARROW),
    ):
        for match in pattern.finditer(text):
            name = match.group(1)
            key = (kind, name)
            if key in seen:
                continue
            seen.add(key)
            symbols.append(
                SymbolRecord(
                    name=name,
                    kind=kind,
                    lineno=_line_number(text, match.start()),
                    public=not name.startswith("_"),
                )
            )
    return sorted(symbols, key=lambda symbol: (symbol.lineno, symbol.name))


def _parse_javascript(
    relative: str,
    text: str,
    language: str,
    is_entry_candidate: bool,
) -> ModuleRecord:
    module = ModuleRecord(
        relative=relative,
        language=language,
        is_entry_point=is_entry_candidate or bool(_JS_MAIN_GUARD.search(text)),
        is_test=_is_test_file(relative),
    )
    module.imports = _js_imports(text)
    module.symbols = _js_symbols(text)
    return module


def _parse_source(
    root: Path,
    source: SourceFile,
    entry_candidates: set,
) -> ModuleRecord:
    if source.language not in PYTHON_LANGUAGES | JS_LANGUAGES:
        return ModuleRecord(
            relative=source.relative,
            language=source.language,
            is_entry_point=source.relative in entry_candidates,
            is_test=_is_test_file(source.relative),
        )

    try:
        text = (root / source.relative).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return ModuleRecord(
            relative=source.relative,
            language=source.language,
            parse_error=str(exc),
        )

    is_entry_candidate = source.relative in entry_candidates
    if source.language in PYTHON_LANGUAGES:
        return _parse_python(source.relative, text, is_entry_candidate)
    return _parse_javascript(
        source.relative, text, source.language, is_entry_candidate
    )


def parse_project(root: Path, scan: ScanResult) -> List[ModuleRecord]:
    """Parse every source file discovered by the scanner."""

    return [
        _parse_source(root, source, scan.entry_point_candidates)
        for source in scan.files
    ]
