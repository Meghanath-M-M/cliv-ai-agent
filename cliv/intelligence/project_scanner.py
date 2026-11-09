"""Scan a repository and extract language and manifest metadata.

The scanner is intentionally conservative: it respects the same ignored
directories and sensitive-file rules as the rest of the agent, never
follows symlinks out of the project, and caps the number of files it
parses so a huge repository cannot stall the CLI.
"""

from __future__ import annotations

import json
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from cliv.security import is_sensitive_path, resolve_project_path

IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "node_modules",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    "build",
    "dist",
    "target",
    "coverage",
    "htmlcov",
    ".next",
    ".turbo",
}

LANGUAGE_BY_SUFFIX = {
    ".py": "Python",
    ".pyi": "Python",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".mjs": "JavaScript",
    ".cjs": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".java": "Java",
    ".kt": "Kotlin",
    ".go": "Go",
    ".rs": "Rust",
    ".c": "C",
    ".h": "C",
    ".cpp": "C++",
    ".cc": "C++",
    ".hpp": "C++",
    ".cs": "C#",
    ".rb": "Ruby",
    ".php": "PHP",
    ".sql": "SQL",
    ".html": "HTML",
    ".css": "CSS",
    ".scss": "SCSS",
    ".sh": "Shell",
    ".ps1": "PowerShell",
}

ENTRY_POINT_NAMES = {
    "main.py",
    "app.py",
    "cli.py",
    "server.py",
    "manage.py",
    "__main__.py",
    "index.js",
    "index.ts",
    "main.js",
    "main.ts",
    "server.js",
    "server.ts",
}

MAX_FILES = 3_000
MAX_SOURCE_BYTES = 1_000_000

_REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")

# Distribution names that differ from their import names.
IMPORT_ALIASES = {
    "python-dotenv": "dotenv",
    "pyyaml": "yaml",
    "pillow": "PIL",
    "beautifulsoup4": "bs4",
    "scikit-learn": "sklearn",
    "python-dateutil": "dateutil",
    "opencv-python": "cv2",
    "pyjwt": "jwt",
    "pyopenssl": "OpenSSL",
}

# Dependencies that are build/test tooling and are not expected in imports.
TOOLING_DEPENDENCIES = {
    "pytest",
    "pytest-cov",
    "coverage",
    "ruff",
    "black",
    "isort",
    "flake8",
    "mypy",
    "tox",
    "nox",
    "pre-commit",
    "build",
    "hatchling",
    "setuptools",
    "wheel",
    "pip",
    "twine",
    "sphinx",
    "mkdocs",
}


@dataclass(frozen=True)
class SourceFile:
    """A source file discovered by the scanner."""

    relative: str
    language: str
    size: int


@dataclass
class ScanResult:
    """Repository metadata before symbol and import extraction."""

    root: Path
    project_name: str
    files: List[SourceFile] = field(default_factory=list)
    all_file_count: int = 0
    languages: Dict[str, int] = field(default_factory=dict)
    declared_dependencies: Set[str] = field(default_factory=set)
    dev_dependencies: Set[str] = field(default_factory=set)
    entry_point_candidates: Set[str] = field(default_factory=set)
    manifest_errors: List[str] = field(default_factory=list)
    skipped_large_files: List[str] = field(default_factory=list)
    truncated: bool = False


def normalise_dependency_name(name: str) -> str:
    """Normalise a Python/JS package name for comparisons."""

    return re.sub(r"[-_.]+", "-", name.strip().lower())


def _parse_requirement(spec: str) -> Optional[str]:
    spec = spec.strip()
    if not spec or spec.startswith(("#", "-r", "-e", "git+", "http:", "https:")):
        return None
    match = _REQUIREMENT_NAME.match(spec)
    if not match:
        return None
    return match.group(1)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _parse_pyproject(
    path: Path,
    result: ScanResult,
) -> Optional[str]:
    """Read project name and dependencies from pyproject.toml."""

    try:
        data = tomllib.loads(_read_text(path))
    except Exception as exc:  # noqa: BLE001 - malformed manifests are reported
        result.manifest_errors.append(f"{path.name}: {exc}")
        return None

    project = data.get("project") or {}
    poetry = (data.get("tool") or {}).get("poetry") or {}
    name = project.get("name") or poetry.get("name") or None

    for dep in project.get("dependencies") or []:
        parsed = _parse_requirement(str(dep))
        if parsed:
            result.declared_dependencies.add(parsed)

    optional = project.get("optional-dependencies") or {}
    for group, deps in optional.items():
        target = (
            result.dev_dependencies
            if str(group).lower() in {"dev", "test", "tests", "lint", "docs"}
            else result.declared_dependencies
        )
        for dep in deps or []:
            parsed = _parse_requirement(str(dep))
            if parsed:
                target.add(parsed)

    for dep in poetry.get("dependencies") or {}:
        if str(dep).lower() != "python":
            result.declared_dependencies.add(str(dep))

    for group in (poetry.get("group") or {}).values():
        for dep in (group or {}).get("dependencies") or {}:
            result.dev_dependencies.add(str(dep))

    return str(name) if name else None


def _parse_requirements(path: Path, result: ScanResult) -> None:
    target = (
        result.dev_dependencies
        if any(token in path.name.lower() for token in ("dev", "test", "lint"))
        else result.declared_dependencies
    )
    try:
        for line in _read_text(path).splitlines():
            parsed = _parse_requirement(line)
            if parsed:
                target.add(parsed)
    except Exception as exc:  # noqa: BLE001
        result.manifest_errors.append(f"{path.name}: {exc}")


def _parse_package_json(path: Path, result: ScanResult) -> Optional[str]:
    try:
        data = json.loads(_read_text(path))
    except Exception as exc:  # noqa: BLE001
        result.manifest_errors.append(f"{path.name}: {exc}")
        return None

    for key, target in (
        ("dependencies", result.declared_dependencies),
        ("peerDependencies", result.declared_dependencies),
        ("devDependencies", result.dev_dependencies),
    ):
        for dep in (data.get(key) or {}):
            target.add(str(dep))

    return str(data.get("name")) if data.get("name") else None


def _load_manifests(root: Path, result: ScanResult) -> Optional[str]:
    pyproject_name: Optional[str] = None
    package_name: Optional[str] = None

    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        pyproject_name = _parse_pyproject(pyproject, result)

    package_json = root / "package.json"
    if package_json.is_file():
        package_name = _parse_package_json(package_json, result)

    for path in root.glob("requirements*.txt"):
        if path.is_file():
            _parse_requirements(path, result)

    # A Python project manifest is the stronger identity signal when both
    # exist (monorepos often have a frontend package.json at the root too).
    return pyproject_name or package_name


def _should_skip_dir(path: Path) -> bool:
    name = path.name
    if name in IGNORED_DIRS:
        return True
    return is_sensitive_path(path)


def scan_project(root: Optional[Path] = None) -> ScanResult:
    """Walk *root* and collect source files plus declared dependencies."""

    from cliv.security import project_root

    resolved_root = (root or project_root()).resolve()
    result = ScanResult(root=resolved_root, project_name=resolved_root.name)

    manifest_name = _load_manifests(resolved_root, result)
    if manifest_name:
        result.project_name = manifest_name

    for current_root, dirs, files in os.walk(resolved_root):
        current_path = Path(current_root)

        dirs[:] = sorted(
            directory
            for directory in dirs
            if not _should_skip_dir(current_path / directory)
        )

        for filename in sorted(files):
            path = current_path / filename
            if not path.is_file():
                continue
            if is_sensitive_path(path):
                continue
            _, path_error = resolve_project_path(str(path), root=resolved_root)
            if path_error:
                continue

            result.all_file_count += 1
            if len(result.files) >= MAX_FILES:
                result.truncated = True
                continue

            if path.name in ENTRY_POINT_NAMES:
                relative = path.relative_to(resolved_root).as_posix()
                result.entry_point_candidates.add(relative)

            language = LANGUAGE_BY_SUFFIX.get(path.suffix.lower())
            if not language:
                continue

            try:
                size = path.stat().st_size
            except OSError:
                continue
            relative = path.relative_to(resolved_root).as_posix()
            if size > MAX_SOURCE_BYTES:
                result.skipped_large_files.append(relative)
                continue

            result.files.append(
                SourceFile(relative=relative, language=language, size=size)
            )
            result.languages[language] = result.languages.get(language, 0) + 1

    return result
