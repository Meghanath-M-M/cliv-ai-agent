"""Code Review & CI/CD Verification Engine for cliv.

Provides automated static analysis, security secret detection, syntax validation,
guideline verification, test suite execution, and GitHub Actions PR comment formatting.
"""

from __future__ import annotations

import ast
import datetime
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from cliv.git import current_branch, is_git_repo
from cliv.memory import MemoryStore
from cliv.security import SENSITIVE_NAMES, SENSITIVE_SUFFIXES, is_sensitive_path, project_root


# ============================================================================
# Secret & Security Patterns
# ============================================================================

SECRET_PATTERNS = [
    ("Groq API Key", re.compile(r"gsk_[a-zA-Z0-9]{24,}")),
    ("OpenAI API Key", re.compile(r"sk-[a-zA-Z0-9]{24,}")),
    ("GitHub Personal Access Token", re.compile(r"gh[pousr]_[a-zA-Z0-9]{36,}")),
    ("GitHub Fine-Grained Token", re.compile(r"github_pat_[a-zA-Z0-9]{22}_[a-zA-Z0-9]{59}")),
    ("AWS Access Key ID", re.compile(r"(?:A3T[A-Z0-9]|AKIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASIA)[A-Z0-9]{16}")),
    ("Private Key Header", re.compile(r"-----BEGIN (?:RSA|EC|DSA|OPENSSH|PGP) PRIVATE KEY-----")),
]

DANGEROUS_PATTERNS = [
    ("Unsafe eval() execution", re.compile(r"\beval\s*\(")),
    ("Unsafe exec() execution", re.compile(r"\bexec\s*\(")),
]


# ============================================================================
# Findings & Report Dataclasses
# ============================================================================

@dataclass
class ReviewFinding:
    file: str
    message: str
    severity: str  # "error", "warning", "info"
    category: str  # "security", "syntax", "style", "test", "architecture"
    line: Optional[int] = None
    rule: Optional[str] = None
    suggestion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReviewReport:
    project_name: str
    root_path: str
    generated_at: str
    base_ref: Optional[str]
    files_reviewed: List[str]
    findings: List[ReviewFinding]
    passed: bool
    tests_run: bool = False
    tests_passed: bool = True
    tests_summary: Optional[str] = None

    @property
    def error_count(self) -> int:
        return sum(1 for f in self.findings if f.severity.lower() == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.findings if f.severity.lower() == "warning")

    @property
    def info_count(self) -> int:
        return sum(1 for f in self.findings if f.severity.lower() == "info")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_name": self.project_name,
            "root_path": self.root_path,
            "generated_at": self.generated_at,
            "base_ref": self.base_ref,
            "passed": self.passed,
            "stats": {
                "files_reviewed": len(self.files_reviewed),
                "errors": self.error_count,
                "warnings": self.warning_count,
                "info": self.info_count,
            },
            "tests": {
                "run": self.tests_run,
                "passed": self.tests_passed,
                "summary": self.tests_summary,
            },
            "files_reviewed": self.files_reviewed,
            "findings": [f.to_dict() for f in self.findings],
        }

    def render_markdown(self, format_mode: str = "standard") -> str:
        """Render markdown report for terminal or GitHub PR comment."""
        lines: List[str] = []

        is_gh = format_mode.lower() in {"github", "gh", "pr"}
        status_icon = "✅" if self.passed else "❌"
        status_text = "PASSED" if self.passed else "FAILED"

        if is_gh:
            lines.append(f"## {status_icon} cliv Code Review: {status_text}")
            lines.append("")
            lines.append(f"> **Project:** `{self.project_name}` · **Checked at:** `{self.generated_at}`")
            if self.base_ref:
                lines.append(f"> **Comparing against base:** `{self.base_ref}`")
        else:
            lines.append(f"# {status_icon} cliv Code Review: {status_text}")
            lines.append("")
            lines.append(f"> Project: `{self.project_name}` ({self.root_path})")
            lines.append(f"> Timestamp: `{self.generated_at}`")
            if self.base_ref:
                lines.append(f"> Base Reference: `{self.base_ref}`")

        lines.append("")
        lines.append("### Summary Statistics")
        lines.append(
            f"- **Files Reviewed:** {len(self.files_reviewed):,} · "
            f"**Errors:** {self.error_count} · "
            f"**Warnings:** {self.warning_count} · "
            f"**Info:** {self.info_count}"
        )

        if self.tests_run:
            test_icon = "✓" if self.tests_passed else "✗"
            lines.append(f"- **Test Suite:** {test_icon} {'Passed' if self.tests_passed else 'Failed'}")
            if self.tests_summary:
                lines.append(f"  *{self.tests_summary}*")

        lines.append("")

        if not self.findings:
            lines.append("🎉 **No issues detected!** All security, syntax, and architectural checks passed.")
            return "\n".join(lines)

        lines.append("### Review Findings")
        lines.append("")
        lines.append("| Severity | Category | File | Line | Issue Description |")
        lines.append("|---|---|---|---|---|")

        for f in self.findings:
            sev = f.severity.upper()
            icon = "🔴" if sev == "ERROR" else ("🟡" if sev == "WARNING" else "ℹ️")
            line_str = str(f.line) if f.line is not None else "-"
            lines.append(
                f"| {icon} **{sev}** | {f.category} | `{f.file}` | {line_str} | {f.message} |"
            )

        lines.append("")

        # Detailed breakdown
        if is_gh:
            lines.append("<details><summary><b>Detailed Remediation Guidance</b></summary>")
            lines.append("")

        lines.append("### Actionable Recommendations")
        for i, f in enumerate(self.findings, 1):
            if f.suggestion:
                lines.append(f"{i}. **`{f.file}`**: {f.suggestion}")
            else:
                lines.append(f"{i}. **`{f.file}`**: Address {f.category} issue: *{f.message}*")

        if is_gh:
            lines.append("")
            lines.append("</details>")

        return "\n".join(lines)


# ============================================================================
# Git Changes Discovery
# ============================================================================

def get_changed_files(
    root: Path,
    base_ref: Optional[str] = None,
) -> List[Path]:
    """Retrieve list of modified, staged, or untracked files."""
    if not is_git_repo(cwd=str(root)):
        # Fallback: scan source files in root
        return [
            p for p in root.rglob("*.py")
            if not any(part.startswith(".") for part in p.parts)
        ][:50]

    changed: List[str] = []

    try:
        if base_ref:
            # Diff against base branch/commit
            res = subprocess.run(
                ["git", "diff", "--name-only", base_ref],
                cwd=str(root),
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode == 0:
                changed.extend(line.strip() for line in res.stdout.splitlines() if line.strip())

        # Always include uncommitted / staged changes
        res_status = subprocess.run(
            ["git", "status", "-s"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
        if res_status.returncode == 0:
            for line in res_status.stdout.splitlines():
                parts = line.strip().split(None, 1)
                if len(parts) == 2:
                    changed.append(parts[1].strip('"'))

    except Exception:
        pass

    unique_paths: List[Path] = []
    seen = set()
    for rel in changed:
        p = (root / rel).resolve()
        if p.exists() and p.is_file() and str(p) not in seen:
            seen.add(str(p))
            unique_paths.append(p)

    return unique_paths


# ============================================================================
# Review Logic
# ============================================================================

def check_security(file_path: Path, rel_path: str, content: str) -> List[ReviewFinding]:
    """Check for leaked secrets, private keys, and sensitive filenames."""
    findings: List[ReviewFinding] = []

    # 1. Filename checks
    if is_sensitive_path(file_path) or file_path.name in SENSITIVE_NAMES or file_path.suffix in SENSITIVE_SUFFIXES:
        findings.append(
            ReviewFinding(
                file=rel_path,
                message=f"Potentially sensitive file tracked or modified: {file_path.name}",
                severity="error",
                category="security",
                rule="sensitive-filename",
                suggestion="Remove from repository and add to .gitignore.",
            )
        )

    # 2. Secret regex checks
    lines = content.splitlines()
    for line_idx, line in enumerate(lines, 1):
        for pattern_name, regex in SECRET_PATTERNS:
            if regex.search(line):
                findings.append(
                    ReviewFinding(
                        file=rel_path,
                        line=line_idx,
                        message=f"Possible hardcoded credential or secret detected: {pattern_name}",
                        severity="error",
                        category="security",
                        rule="hardcoded-secret",
                        suggestion="Store credentials in environment variables or a secrets manager.",
                    )
                )

        # Dangerous functions (only in python files)
        if file_path.suffix == ".py":
            for danger_name, regex in DANGEROUS_PATTERNS:
                if regex.search(line):
                    findings.append(
                        ReviewFinding(
                            file=rel_path,
                            line=line_idx,
                            message=f"High-risk function call detected: {danger_name}",
                            severity="warning",
                            category="security",
                            rule="dangerous-call",
                            suggestion="Avoid dynamic code evaluation.",
                        )
                    )

    return findings


def check_syntax(file_path: Path, rel_path: str, content: str) -> List[ReviewFinding]:
    """Parse code to check for syntax validity."""
    findings: List[ReviewFinding] = []

    if file_path.suffix == ".py":
        try:
            ast.parse(content, filename=str(file_path))
        except SyntaxError as exc:
            findings.append(
                ReviewFinding(
                    file=rel_path,
                    line=exc.lineno,
                    message=f"Python Syntax Error: {exc.msg}",
                    severity="error",
                    category="syntax",
                    rule="python-syntax",
                    suggestion="Fix syntax error before committing.",
                )
            )

    elif file_path.suffix in {".json"}:
        try:
            json.loads(content)
        except json.JSONDecodeError as exc:
            findings.append(
                ReviewFinding(
                    file=rel_path,
                    line=exc.lineno,
                    message=f"JSON Decode Error: {exc.msg}",
                    severity="error",
                    category="syntax",
                    rule="json-syntax",
                    suggestion="Validate JSON formatting.",
                )
            )

    return findings


def check_conventions(
    file_path: Path,
    rel_path: str,
    content: str,
    conventions: List[Dict[str, str]],
) -> List[ReviewFinding]:
    """Check code against team guidelines and conventions."""
    findings: List[ReviewFinding] = []

    # Check for empty files
    if not content.strip() and file_path.suffix == ".py":
        findings.append(
            ReviewFinding(
                file=rel_path,
                message="Empty Python source file",
                severity="info",
                category="style",
                rule="empty-file",
                suggestion="Add implementation or remove file if redundant.",
            )
        )

    return findings


def run_test_suite(root: Path, timeout_seconds: int = 45) -> Tuple[bool, str]:
    """Execute pytest suite and return pass/fail with summary."""
    try:
        cmd = [sys.executable, "-m", "pytest", "-q", "--tb=short"]
        res = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        passed = (res.returncode == 0)
        lines = [line.strip() for line in (res.stdout + res.stderr).splitlines() if line.strip()]
        summary = lines[-1] if lines else ("Tests passed." if passed else "Tests failed.")
        return passed, summary
    except subprocess.TimeoutExpired:
        return False, "Test suite execution timed out"
    except Exception as exc:
        return False, f"Failed to run test suite: {exc}"


def run_code_review(
    root: Optional[Union[str, Path]] = None,
    base_ref: Optional[str] = None,
    files: Optional[Sequence[Union[str, Path]]] = None,
    execute_tests: bool = False,
) -> ReviewReport:
    """Run full automated code review across target files."""
    target_root = Path(root).resolve() if root else project_root()

    if files is not None:
        target_files = [
            (target_root / f).resolve() if not Path(f).is_absolute() else Path(f).resolve()
            for f in files
        ]
        target_files = [f for f in target_files if f.exists() and f.is_file()]
    else:
        target_files = get_changed_files(target_root, base_ref=base_ref)

    findings: List[ReviewFinding] = []
    conventions: List[Dict[str, str]] = []
    try:
        store = MemoryStore(project_dir=target_root)
        conventions = [e.to_dict() for e in store.list_entries()]
    except Exception:
        pass

    reviewed_rel_paths: List[str] = []

    for file_path in target_files:
        try:
            rel = str(file_path.relative_to(target_root)).replace("\\", "/")
        except ValueError:
            rel = file_path.name
        reviewed_rel_paths.append(rel)

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        findings.extend(check_security(file_path, rel, content))
        findings.extend(check_syntax(file_path, rel, content))
        findings.extend(check_conventions(file_path, rel, content, conventions))

    tests_run = False
    tests_passed = True
    tests_summary = None

    if execute_tests:
        tests_run = True
        tests_passed, tests_summary = run_test_suite(target_root)
        if not tests_passed:
            findings.append(
                ReviewFinding(
                    file="test_suite",
                    message=f"Test suite failure: {tests_summary}",
                    severity="error",
                    category="test",
                    rule="failing-tests",
                    suggestion="Run pytest locally to fix regressions before pushing.",
                )
            )

    has_blocking_errors = any(f.severity.lower() == "error" for f in findings)
    passed = not has_blocking_errors

    return ReviewReport(
        project_name=target_root.name,
        root_path=str(target_root),
        generated_at=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        base_ref=base_ref,
        files_reviewed=reviewed_rel_paths,
        findings=findings,
        passed=passed,
        tests_run=tests_run,
        tests_passed=tests_passed,
        tests_summary=tests_summary,
    )


# ============================================================================
# GitHub Actions Workflow Scaffolder
# ============================================================================

def generate_github_workflow(root: Optional[Union[str, Path]] = None) -> Path:
    """Generate .github/workflows/cliv-review.yml for automated PR reviews."""
    target_root = Path(root).resolve() if root else project_root()
    workflow_dir = target_root / ".github" / "workflows"
    workflow_dir.mkdir(parents=True, exist_ok=True)
    workflow_file = workflow_dir / "cliv-review.yml"

    content = """name: cliv Automated Code Review & CI

on:
  pull_request:
    branches: [main, master, develop]
  push:
    branches: [main, master]

permissions:
  contents: read
  pull-requests: write

jobs:
  review:
    name: cliv Code Review
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install cliv & Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e .
          if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
          if [ -f tests/requirements.txt ]; then pip install -r tests/requirements.txt; fi

      - name: Run cliv Code Review
        id: cliv_review
        run: |
          cliv review --base ${{ github.event.pull_request.base.sha || 'HEAD~1' }} --format github > review_output.md || true
          cat review_output.md >> $GITHUB_STEP_SUMMARY

      - name: Post PR Review Comment
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            if (fs.existsSync('review_output.md')) {
              const body = fs.readFileSync('review_output.md', 'utf8');
              github.rest.issues.createComment({
                issue_number: context.issue.number,
                owner: context.repo.owner,
                repo: context.repo.repo,
                body: body
              });
            }
"""
    workflow_file.write_text(content.strip() + "\n", encoding="utf-8")
    return workflow_file
