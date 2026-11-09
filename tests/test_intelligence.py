"""Project DNA scanner, parser, dependency graph, and tool tests."""

from __future__ import annotations

from pathlib import Path

from cliv.intelligence import build_project_intelligence, parse_project, scan_project
from cliv.intelligence.dependency_graph import build_dependency_graph
from cliv.tools.loader import load_tools
from cliv.tools.project_dna import ProjectDNATool


def _write(root: Path, name: str, content: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_scanner_builds_component_report_and_skips_secrets(project_dir):
    _write(
        project_dir,
        "pyproject.toml",
        '[project]\nname = "sample-app"\ndependencies = ["requests>=2"]\n',
    )
    _write(
        project_dir,
        "package.json",
        '{"name": "sample-app", "dependencies": {"react": "^18.0.0"}}\n',
    )
    _write(project_dir, "README.md", "# Sample\n")
    _write(project_dir, "sample/__init__.py", "")
    _write(
        project_dir,
        "sample/api.py",
        "import requests\n\nclass API:\n    def handle(self):\n        return requests\n",
    )
    _write(
        project_dir,
        "src/components/App.jsx",
        'import React from "react";\n\nexport function App() {\n  return null;\n}\n',
    )
    _write(project_dir, ".env", "GROQ_API_KEY=super-secret\n")
    _write(project_dir, "node_modules/hidden.py", "SECRET = 1\n")
    _write(project_dir, ".venv/hidden.py", "SECRET = 2\n")

    report = build_project_intelligence(project_dir)
    rendered = report.render()

    assert report.scan.project_name == "sample-app"
    assert report.scan.languages == {"Python": 2, "JavaScript": 1}
    assert {component.key for component in report.components} == {"sample", "src"}
    assert "Frontend" in report.architecture
    assert {"requests", "react"}.issubset(set(report.external_packages))
    assert "CLIV PROJECT INTELLIGENCE" in rendered
    assert "sample/api.py" in rendered
    assert "src/components/App.jsx" in rendered
    assert "super-secret" not in rendered
    assert "node_modules" not in rendered
    assert ".venv" not in rendered


def test_parser_and_graph_resolve_relative_imports_and_skip_type_only_cycles(project_dir):
    _write(project_dir, "pkg/__init__.py", "")
    _write(
        project_dir,
        "pkg/core.py",
        "from .helpers import format_name\n"
        "from typing import TYPE_CHECKING\n\n"
        "if TYPE_CHECKING:\n"
        "    from pkg.types import User\n\n"
        "class Engine:\n"
        "    def start(self):\n"
        "        pass\n\n"
        "def run():\n"
        "    pass\n",
    )
    _write(
        project_dir,
        "pkg/helpers.py",
        "from pkg.core import Engine\n\n"
        "def format_name(name):\n"
        "    return name.strip()\n",
    )

    scan = scan_project(project_dir)
    modules = parse_project(project_dir, scan)
    by_path = {module.relative: module for module in modules}
    core = by_path["pkg/core.py"]

    assert [symbol.name for symbol in core.symbols] == ["Engine", "start", "run"]
    assert any(
        record.is_relative and record.module == "helpers"
        for record in core.imports
    )
    assert any(record.is_type_checking for record in core.imports)

    graph = build_dependency_graph(modules)
    assert "pkg/helpers.py" in graph.edges["pkg/core.py"]
    assert "pkg/core.py" in graph.edges["pkg/helpers.py"]
    assert len(graph.cycles) == 1
    assert "pkg/types" not in graph.unresolved["pkg/core.py"]


def test_src_layout_imports_resolve_to_internal_modules(project_dir):
    _write(project_dir, "src/sample/__init__.py", "")
    _write(
        project_dir,
        "src/sample/app.py",
        "from sample.helpers import helper\n\nclass App:\n    pass\n",
    )
    _write(project_dir, "src/sample/helpers.py", "def helper():\n    return 1\n")

    scan = scan_project(project_dir)
    modules = parse_project(project_dir, scan)
    graph = build_dependency_graph(modules)

    assert "src/sample/helpers.py" in graph.edges["src/sample/app.py"]
    assert "sample" not in graph.external["src/sample/app.py"]


def test_javascript_symbols_and_relative_imports_are_mapped(project_dir):
    _write(
        project_dir,
        "web/components/Button.jsx",
        "import React from 'react';\n\nexport function Button() {\n  return null;\n}\n",
    )
    _write(
        project_dir,
        "web/App.jsx",
        "import { Button } from './components/Button';\n\n"
        "export default function App() {\n  return null;\n}\n",
    )

    scan = scan_project(project_dir)
    modules = parse_project(project_dir, scan)
    by_path = {module.relative: module for module in modules}

    assert [symbol.name for symbol in by_path["web/App.jsx"].symbols] == ["App"]
    assert by_path["web/App.jsx"].imports[0].module == "./components/Button"
    graph = build_dependency_graph(modules)
    assert "web/components/Button.jsx" in graph.edges["web/App.jsx"]
    assert "react" in graph.external["web/components/Button.jsx"]


def test_report_flags_duplicate_functions_unused_dependencies_and_no_tests(project_dir):
    _write(
        project_dir,
        "pyproject.toml",
        '[project]\nname = "issues-demo"\n'
        'dependencies = ["requests", "rich", "unused-package"]\n',
    )
    _write(
        project_dir,
        "a.py",
        "import requests\n\n\ndef helper():\n    return requests\n",
    )
    _write(project_dir, "b.py", "def helper():\n    return 2\n")

    report = build_project_intelligence(project_dir)
    categories = {issue.category for issue in report.issues}
    messages = "\n".join(issue.message for issue in report.issues)

    assert "duplicate-function" in categories
    assert "unused-dependency" in categories
    assert "no-tests" in categories
    assert "unused-package" in messages
    assert "helper" in messages
    assert "External packages: requests" in report.render()


def test_project_dna_tool_is_read_only_and_confined(project_dir):
    _write(
        project_dir,
        "app.py",
        "def main():\n    pass\n\nif __name__ == \"__main__\":\n    main()\n",
    )
    _write(project_dir, "utils/helpers.py", "def helper():\n    return 1\n")

    tool = ProjectDNATool()
    result = tool.execute(".")

    assert tool.requires_approval is False
    assert "CLIV PROJECT INTELLIGENCE" in result
    assert "app.py" in result
    assert "utils/helpers.py" in result
    assert "blocked" in tool.execute("..").lower()
    assert "analyse_project" in load_tools()
