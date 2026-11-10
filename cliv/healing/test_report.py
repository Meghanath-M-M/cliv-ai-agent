"""Parse pytest output into structured failures for the repair loop."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from cliv.security import is_sensitive_path

MAX_CONTEXT_FILES = 6

_IGNORED_PATH_PARTS = {
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
}

_FAILURE_LINE = re.compile(
    r"(?m)^(?:FAILED|ERROR)\s+([^\s]+)(?:\s+-\s+(.*))?$"
)
_EXIT_HEADER = re.compile(r"Pytest Results \(exit (-?\d+)\):")
_COUNT = re.compile(r"(\d+)\s+(passed|failed|error|errors|skipped)", re.IGNORECASE)
_TRACEBACK_PATH = re.compile(
    r"(?m)^([A-Za-z]:[\\/][^\n]+?\.py|(?!<)[^\s:][^\n:]*?\.py):(\d+)(?::|$)"
)
_FILE_PATH = re.compile(r'File "([^"]+\.py)", line (\d+)')


@dataclass(frozen=True)
class TestFailure:
    name: str
    summary: str = ""


@dataclass
class TestReport:
    exit_code: Optional[int] = None
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    failures: List[TestFailure] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    output: str = ""
    no_tests: bool = False
    cancelled: bool = False
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return (
            self.exit_code == 0
            and self.error is None
            and not self.cancelled
            and not self.no_tests
        )

    def summary(self) -> str:
        if self.cancelled:
            return "cancelled"
        if self.error:
            return self.error
        if self.no_tests:
            return "no tests ran"
        parts = []
        if self.passed:
            parts.append(f"{self.passed} passed")
        if self.failed:
            parts.append(f"{self.failed} failed")
        if self.errors:
            parts.append(f"{self.errors} error(s)")
        if self.skipped:
            parts.append(f"{self.skipped} skipped")
        if not parts:
            if self.exit_code is not None:
                return f"exit code {self.exit_code}"
            return "unknown result"
        return ", ".join(parts)


def _normalise_path(candidate: str, root: Path) -> Optional[str]:
    candidate = candidate.strip().strip('"').replace("\\", "/")
    if not candidate or candidate.startswith("<"):
        return None

    path = Path(candidate)
    if not path.is_absolute():
        path = root / path
    try:
        path = path.resolve()
        relative = path.relative_to(root.resolve())
    except (OSError, RuntimeError, ValueError):
        return None

    if any(part.lower() in _IGNORED_PATH_PARTS for part in relative.parts):
        return None
    if is_sensitive_path(path) or not path.is_file():
        return None
    return relative.as_posix()


def _collect_files(
    output: str, failures: List[TestFailure], root: Path
) -> List[str]:
    candidates: List[str] = []

    # Source files from traceback headers usually tell the model where the
    # bug is; test files from the summary tell it what behaviour failed.
    for match in _TRACEBACK_PATH.finditer(output):
        candidates.append(match.group(1))
    for match in _FILE_PATH.finditer(output):
        candidates.append(match.group(1))
    for failure in failures:
        candidates.append(failure.name.split("::", 1)[0])

    files: List[str] = []
    seen = set()
    for candidate in candidates:
        relative = _normalise_path(candidate, root)
        key = relative.lower() if relative else None
        if not relative or key in seen:
            continue
        seen.add(key)
        files.append(relative)
        if len(files) >= MAX_CONTEXT_FILES:
            break
    return files


def parse_pytest_output(raw: str, root: Path) -> TestReport:
    """Convert the text returned by ``run_tests`` into a structured report."""

    report = TestReport(output=raw)
    lowered = raw.lower()

    if "cancelled by user" in lowered:
        report.cancelled = True
        return report
    if (
        "pytest timed out" in lowered
        or "pytest is not installed" in lowered
        or "unexpected error running pytest" in lowered
        or raw.strip().startswith("[DRY RUN]")
    ):
        report.error = raw.strip()
        return report

    match = _EXIT_HEADER.search(raw)
    if match:
        report.exit_code = int(match.group(1))
    body = raw[match.end() :] if match else raw

    for amount, label in _COUNT.findall(body):
        label = label.lower()
        if label == "passed":
            report.passed = max(report.passed, int(amount))
        elif label == "failed":
            report.failed = max(report.failed, int(amount))
        elif label in {"error", "errors"}:
            report.errors = max(report.errors, int(amount))
        elif label == "skipped":
            report.skipped = max(report.skipped, int(amount))

    for failure_match in _FAILURE_LINE.finditer(body):
        report.failures.append(
            TestFailure(
                name=failure_match.group(1),
                summary=(failure_match.group(2) or "").strip(),
            )
        )

    report.no_tests = "no tests ran" in lowered
    report.files = _collect_files(body, report.failures, root)
    return report
