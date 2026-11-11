"""Tests for Rank 10: Project Onboarding Reports (cliv onboard, /onboard, and agent tool)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cliv.onboarding import (
    OnboardingReport,
    detect_environment,
    generate_onboarding_report,
)
from cliv.tools.onboard import ProjectOnboardTool
from cliv.slash import SlashContext, handle_onboard
from cliv.cli import cmd_onboard


# ============================================================================
# Unit Tests for Environment Detection
# ============================================================================

class TestEnvironmentDetection:
    """Verify build system, virtual environment, and test runner detection."""

    def test_detect_python_pyproject(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname='test-pkg'", encoding="utf-8")
        (tmp_path / ".venv").mkdir()

        env = detect_environment(tmp_path)
        assert any("pyproject.toml" in pm for pm in env["package_managers"])
        assert env["virtualenv"] == ".venv"
        assert env["test_command"] == "pytest"
        assert env["setup_command"] == "pip install -e ."

    def test_detect_node_project(self, tmp_path):
        (tmp_path / "package.json").write_text('{"name": "test-app"}', encoding="utf-8")
        (tmp_path / "pnpm-lock.yaml").write_text("lockfileVersion: 5.4", encoding="utf-8")

        env = detect_environment(tmp_path)
        assert "pnpm" in env["package_managers"]
        assert env["test_command"] == "pnpm test"
        assert env["setup_command"] == "pnpm install"

    def test_detect_rust_project(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text('[package]\nname="crate"', encoding="utf-8")

        env = detect_environment(tmp_path)
        assert "cargo" in env["package_managers"]
        assert env["test_command"] == "cargo test"

    def test_detect_go_project(self, tmp_path):
        (tmp_path / "go.mod").write_text("module example.com/mod", encoding="utf-8")

        env = detect_environment(tmp_path)
        assert "go modules" in env["package_managers"]
        assert env["test_command"] == "go test ./..."


# ============================================================================
# Unit Tests for OnboardingReport
# ============================================================================

class TestOnboardingReport:
    """Verify report markdown rendering, serialization, and saving."""

    @pytest.fixture
    def sample_report(self, tmp_path):
        return OnboardingReport(
            project_name="sample-project",
            root_path=str(tmp_path),
            generated_at="2026-01-01 12:00:00Z",
            environment={
                "package_managers": ["pip / pyproject.toml"],
                "virtualenv": ".venv",
                "test_command": "pytest",
                "setup_command": "pip install -e .",
            },
            dna={
                "languages": {"Python": 5000},
                "source_files": 12,
                "symbol_count": 80,
                "class_count": 8,
                "function_count": 45,
                "method_count": 27,
                "entry_points": ["sample/cli.py"],
                "external_packages": ["rich", "requests"],
                "components": [
                    {
                        "name": "Core",
                        "key": "core",
                        "role": "Core Engine",
                        "files": ["sample/core.py"],
                        "symbols": 30,
                    }
                ],
            },
            conventions=[
                {"category": "testing", "content": "Always run pytest with -v", "scope": "project"}
            ],
            cliv_md="# Guidelines\n- Never break existing tests",
            git_info={
                "is_repo": True,
                "branch": "main",
                "dirty_files_count": 0,
                "recent_commits": ["abc1234 Initial commit"],
            },
            workspaces=[
                {"name": "docs", "path": str(tmp_path / "docs"), "is_active": False, "description": "Docs"}
            ],
            hotspots=[
                {"severity": "medium", "message": "High cyclomatic complexity in parser"}
            ],
        )

    def test_to_dict_structure(self, sample_report):
        d = sample_report.to_dict()
        assert d["project_name"] == "sample-project"
        assert d["environment"]["test_command"] == "pytest"
        assert d["dna"]["source_files"] == 12
        assert len(d["conventions"]) == 1
        assert d["git_info"]["branch"] == "main"

    def test_render_markdown_contains_sections(self, sample_report):
        md = sample_report.render_markdown()
        assert "# 🚀 Project Onboarding: sample-project" in md
        assert "## 1. ⚡ Quick Start & Environment" in md
        assert "## 2. 🏗️ Architecture & Topography" in md
        assert "## 3. 📋 Team Conventions & Guidelines" in md
        assert "## 4. 🌿 Git Context & Recent Activity" in md
        assert "## 5. 🗂️ Registered Multi-Repo Workspaces" in md
        assert "## 6. 🔍 Codebase Health & Hotspots" in md
        assert "## 7. 🧭 Recommended First Steps for New Contributors" in md

        # Key data presence
        assert "sample/cli.py" in md
        assert "Always run pytest with -v" in md
        assert "Never break existing tests" in md
        assert "abc1234 Initial commit" in md

    def test_save_to_file(self, sample_report, tmp_path):
        out = sample_report.save()
        assert out.exists()
        assert out.name == "ONBOARDING.md"
        content = out.read_text(encoding="utf-8")
        assert "Project Onboarding: sample-project" in content

    def test_save_to_custom_path(self, sample_report, tmp_path):
        custom = tmp_path / "docs" / "guide.md"
        out = sample_report.save(dest=custom)
        assert out == custom
        assert out.exists()


# ============================================================================
# Tests for generate_onboarding_report Generator
# ============================================================================

class TestGenerateOnboardingReport:
    """Verify end-to-end report generation from repository root."""

    def test_generate_report_fallback_when_empty_dir(self, tmp_path):
        report = generate_onboarding_report(tmp_path)
        assert report.project_name == tmp_path.name
        assert isinstance(report.environment, dict)
        assert isinstance(report.dna, dict)
        assert isinstance(report.git_info, dict)


# ============================================================================
# Tests for ProjectOnboardTool (Agent Tool)
# ============================================================================

class TestProjectOnboardTool:
    """Verify tool discovery, schema, and execution."""

    def test_tool_metadata(self):
        tool = ProjectOnboardTool()
        assert tool.name == "project_onboard"
        assert "read" in tool.permissions
        assert tool.risk_level == "low"
        assert "save_to_file" in tool.input_schema["properties"]

    def test_tool_execute_markdown(self, tmp_path):
        tool = ProjectOnboardTool()
        with patch("cliv.tools.onboard.resolve_project_path", return_value=(tmp_path, None)):
            res = tool.execute()
            assert "Project Onboarding" in res

    def test_tool_execute_with_save(self, tmp_path):
        tool = ProjectOnboardTool()
        with patch("cliv.tools.onboard.resolve_project_path", return_value=(tmp_path, None)):
            res = tool.execute(save_to_file=True)
            assert "successfully saved" in res
            assert (tmp_path / "ONBOARDING.md").exists()


# ============================================================================
# Tests for CLI & Slash Command
# ============================================================================

class TestOnboardCliAndSlash:
    """Verify cliv onboard CLI subcommand and /onboard slash command."""

    def test_cmd_onboard_json(self, capsys, tmp_path):
        args = MagicMock()
        args.path = str(tmp_path)
        args.json = True
        args.save = False
        args.output = None

        cmd_onboard(args)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["project_name"] == tmp_path.name
        assert "environment" in parsed
        assert "dna" in parsed

    def test_cmd_onboard_save(self, capsys, tmp_path):
        args = MagicMock()
        args.path = str(tmp_path)
        args.json = False
        args.save = True
        args.output = None

        cmd_onboard(args)
        assert (tmp_path / "ONBOARDING.md").exists()

    def test_handle_onboard_slash(self, tmp_path):
        agent = MagicMock()
        console = MagicMock()
        ctx = SlashContext(agent=agent, console=console)

        with patch("cliv.onboarding.project_root", return_value=tmp_path):
            # Normal render
            ret = handle_onboard(ctx, "")
            assert ret is True
            console.print.assert_called()

            # Save
            ret = handle_onboard(ctx, "save")
            assert ret is True
            assert (tmp_path / "ONBOARDING.md").exists()
