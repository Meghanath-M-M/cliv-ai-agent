"""Tests for Rank 11: CI/CD Integration & Automated Code Review (cliv review, /review, and tool)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cliv.review import (
    ReviewFinding,
    ReviewReport,
    check_conventions,
    check_security,
    check_syntax,
    generate_github_workflow,
    run_code_review,
)
from cliv.tools.review import CodeReviewTool
from cliv.slash import SlashContext, handle_review
from cliv.cli import cmd_review


# ============================================================================
# Unit Tests for Security & Secret Checks
# ============================================================================

class TestSecurityChecks:
    """Verify security scanner catches secrets, dangerous functions, and sensitive paths."""

    def test_detect_sensitive_filename(self, tmp_path):
        env_file = tmp_path / ".env"
        findings = check_security(env_file, ".env", "PORT=8000")
        assert len(findings) >= 1
        assert any(f.rule == "sensitive-filename" for f in findings)
        assert findings[0].severity == "error"

    def test_detect_hardcoded_secrets(self, tmp_path):
        py_file = tmp_path / "app.py"
        content = (
            "import os\n"
            "OPENAI_KEY = 'sk-1234567890abcdef1234567890abcdef'\n"
            "GROQ_KEY = 'gsk_1234567890abcdef1234567890abcdef'\n"
        )
        findings = check_security(py_file, "app.py", content)
        secret_findings = [f for f in findings if f.rule == "hardcoded-secret"]
        assert len(secret_findings) == 2
        assert all(f.severity == "error" for f in secret_findings)

    def test_detect_dangerous_calls(self, tmp_path):
        py_file = tmp_path / "evaluator.py"
        content = "def run(code):\n    return eval(code)\n"
        findings = check_security(py_file, "evaluator.py", content)
        danger = [f for f in findings if f.rule == "dangerous-call"]
        assert len(danger) == 1
        assert danger[0].severity == "warning"

    def test_clean_file_no_findings(self, tmp_path):
        py_file = tmp_path / "clean.py"
        content = "def add(a: int, b: int) -> int:\n    return a + b\n"
        findings = check_security(py_file, "clean.py", content)
        assert len(findings) == 0


# ============================================================================
# Unit Tests for Syntax Checks
# ============================================================================

class TestSyntaxChecks:
    """Verify AST parser catches syntax errors before execution."""

    def test_python_syntax_error(self, tmp_path):
        py_file = tmp_path / "broken.py"
        content = "def func(\n    print('missing paren')\n"
        findings = check_syntax(py_file, "broken.py", content)
        assert len(findings) == 1
        assert findings[0].category == "syntax"
        assert findings[0].severity == "error"
        assert findings[0].line is not None

    def test_json_syntax_error(self, tmp_path):
        json_file = tmp_path / "config.json"
        content = '{"key": "value",}'  # trailing comma
        findings = check_syntax(json_file, "config.json", content)
        assert len(findings) == 1
        assert findings[0].category == "syntax"
        assert findings[0].severity == "error"

    def test_valid_syntax(self, tmp_path):
        py_file = tmp_path / "valid.py"
        content = "class A:\n    pass\n"
        assert len(check_syntax(py_file, "valid.py", content)) == 0


# ============================================================================
# Unit Tests for ReviewReport
# ============================================================================

class TestReviewReport:
    """Verify report formatting for standard and GitHub Actions PR comment modes."""

    @pytest.fixture
    def mock_report(self):
        findings = [
            ReviewFinding(
                file="src/auth.py",
                line=14,
                severity="error",
                category="security",
                rule="hardcoded-secret",
                message="Hardcoded Groq API key",
                suggestion="Use os.environ",
            ),
            ReviewFinding(
                file="src/util.py",
                line=5,
                severity="warning",
                category="security",
                rule="dangerous-call",
                message="eval() detected",
            ),
        ]
        return ReviewReport(
            project_name="test-project",
            root_path="C:/test/repo",
            generated_at="2026-01-01 12:00:00Z",
            base_ref="origin/main",
            files_reviewed=["src/auth.py", "src/util.py"],
            findings=findings,
            passed=False,
            tests_run=True,
            tests_passed=True,
            tests_summary="24 passed in 1.2s",
        )

    def test_report_stats(self, mock_report):
        assert mock_report.error_count == 1
        assert mock_report.warning_count == 1
        assert mock_report.passed is False

    def test_to_dict_structure(self, mock_report):
        d = mock_report.to_dict()
        assert d["project_name"] == "test-project"
        assert d["passed"] is False
        assert d["stats"]["errors"] == 1
        assert d["stats"]["warnings"] == 1
        assert len(d["findings"]) == 2

    def test_render_markdown_standard(self, mock_report):
        md = mock_report.render_markdown(format_mode="standard")
        assert "❌ cliv Code Review: FAILED" in md
        assert "origin/main" in md
        assert "src/auth.py" in md
        assert "Hardcoded Groq API key" in md

    def test_render_markdown_github_pr(self, mock_report):
        md = mock_report.render_markdown(format_mode="github")
        assert "## ❌ cliv Code Review: FAILED" in md
        assert "<details><summary><b>Detailed Remediation Guidance</b></summary>" in md
        assert "src/auth.py" in md

    def test_render_markdown_clean(self):
        clean_report = ReviewReport(
            project_name="clean-project",
            root_path="C:/clean",
            generated_at="2026-01-01 12:00:00Z",
            base_ref=None,
            files_reviewed=["main.py"],
            findings=[],
            passed=True,
        )
        md = clean_report.render_markdown()
        assert "✅ cliv Code Review: PASSED" in md
        assert "No issues detected" in md


# ============================================================================
# Tests for CI Workflow Scaffolder
# ============================================================================

class TestWorkflowGenerator:
    """Verify GitHub Actions workflow template generation."""

    def test_generate_github_workflow(self, tmp_path):
        wf = generate_github_workflow(tmp_path)
        assert wf.exists()
        assert wf.name == "cliv-review.yml"
        content = wf.read_text(encoding="utf-8")
        assert "cliv review --base" in content
        assert "actions/checkout@v4" in content
        assert "pull_request" in content


# ============================================================================
# Tests for Agent Tool & CLI & Slash Command
# ============================================================================

class TestCodeReviewToolAndCli:
    """Verify tool discovery, CLI execution, and slash command integration."""

    def test_tool_metadata(self):
        tool = CodeReviewTool()
        assert tool.name == "code_review"
        assert "read" in tool.permissions
        assert tool.risk_level == "low"

    def test_tool_execute_clean_project(self, tmp_path):
        (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
        tool = CodeReviewTool()
        with patch("cliv.tools.review.resolve_project_path", return_value=(tmp_path, None)):
            res = tool.execute()
            assert "cliv Code Review" in res

    def test_cmd_review_json(self, capsys, tmp_path):
        args = MagicMock()
        args.path = str(tmp_path)
        args.json = True
        args.base = None
        args.format = "standard"
        args.run_tests = False
        args.no_fail = True
        args.init_ci = False

        cmd_review(args)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "passed" in parsed
        assert "stats" in parsed

    def test_cmd_review_init_ci(self, capsys, tmp_path):
        args = MagicMock()
        args.init_ci = True

        with patch("cliv.review.project_root", return_value=tmp_path):
            cmd_review(args)
            assert (tmp_path / ".github" / "workflows" / "cliv-review.yml").exists()

    def test_cmd_review_exits_on_error(self, tmp_path):
        # Create a file with a hardcoded secret
        (tmp_path / "secret.py").write_text("KEY = 'sk-1234567890abcdef1234567890abcdef'\n", encoding="utf-8")

        args = MagicMock()
        args.path = str(tmp_path)
        args.json = False
        args.base = None
        args.format = "standard"
        args.run_tests = False
        args.no_fail = False
        args.init_ci = False

        with patch("cliv.review.get_changed_files", return_value=[tmp_path / "secret.py"]):
            with pytest.raises(SystemExit) as exc_info:
                cmd_review(args)
            assert exc_info.value.code == 1

    def test_handle_review_slash(self, tmp_path):
        agent = MagicMock()
        console = MagicMock()
        ctx = SlashContext(agent=agent, console=console)

        with patch("cliv.review.project_root", return_value=tmp_path):
            # Normal review
            ret = handle_review(ctx, "")
            assert ret is True
            console.print.assert_called()

            # Init CI
            ret = handle_review(ctx, "init")
            assert ret is True
            assert (tmp_path / ".github" / "workflows" / "cliv-review.yml").exists()
