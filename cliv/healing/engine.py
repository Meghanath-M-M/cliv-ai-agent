"""Test-driven self-healing loop.

The engine runs pytest, asks the selected model for a small JSON repair
plan, applies it through the normal approval-gated edit tool, and reruns
tests.  It never edits files directly and never executes arbitrary
commands.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

from cliv.healing.repair_plan import RepairEdit, parse_repair_plan
from cliv.healing.test_report import TestReport, parse_pytest_output
from cliv.security import project_root, resolve_project_path

if TYPE_CHECKING:  # avoid a runtime import cycle
    from cliv.agent import AIAgent


MAX_OUTPUT_CHARS = 8_000
MAX_FILE_CHARS = 5_000
MAX_TOTAL_FILE_CHARS = 15_000
MAX_CONTEXT_FILES = 4
MAX_REPAIR_ATTEMPTS = 2

_IMPORT_LINE = re.compile(
    r"(?m)^\s*(?:from\s+([A-Za-z_][\w\.]*)\s+import|import\s+([A-Za-z_][\w\.]*))"
)


def _looks_like_test(path: str) -> bool:
    name = Path(path).name.lower()
    return name.startswith("test_") or name.endswith("_test.py")


def _module_files(module: str) -> List[str]:
    base = module.replace(".", "/")
    return [f"{base}.py", f"{base}/__init__.py"]


@dataclass
class RepairAttempt:
    number: int
    edited_files: List[str] = field(default_factory=list)
    plan_error: str = ""
    apply_error: str = ""
    generation_error: str = ""
    test_report: Optional[TestReport] = None


@dataclass
class HealingResult:
    status: str
    initial: Optional[TestReport] = None
    final: Optional[TestReport] = None
    attempts: List[RepairAttempt] = field(default_factory=list)
    modified_files: List[str] = field(default_factory=list)
    message: str = ""

    def render(self) -> str:
        lines = ["CLIV SELF-HEALING REPORT", ""]
        lines.append(f"Initial tests: {self.initial.summary() if self.initial else 'not run'}")
        lines.append(f"Repair attempts: {len(self.attempts)}")
        if self.modified_files:
            lines.append(f"Modified files: {', '.join(self.modified_files)}")
        else:
            lines.append("Modified files: none")

        if self.attempts:
            lines.append("")
            lines.append("Attempts:")
            for attempt in self.attempts:
                if attempt.plan_error:
                    detail = f"invalid repair plan: {attempt.plan_error}"
                elif attempt.apply_error:
                    detail = f"edit failed: {attempt.apply_error}"
                elif attempt.generation_error:
                    detail = f"generation failed: {attempt.generation_error}"
                elif attempt.test_report is not None:
                    edited = (
                        ", ".join(attempt.edited_files)
                        if attempt.edited_files
                        else "no files"
                    )
                    detail = f"edited {edited}; tests: {attempt.test_report.summary()}"
                else:
                    detail = "no result"
                lines.append(f"  {attempt.number}. {detail}")

        final = self.final
        if final is not None:
            lines.append("")
            lines.append(f"Final tests: {final.summary()}")
            if final.failures:
                lines.append("Remaining failures:")
                for failure in final.failures[:8]:
                    suffix = f" - {failure.summary}" if failure.summary else ""
                    lines.append(f"  - {failure.name}{suffix}")

        lines.append("")
        if self.status == "passed":
            lines.append("Result: PASSED")
        elif self.status == "failed":
            lines.append(f"Result: FAILED after {len(self.attempts)} repair attempt(s).")
        elif self.status == "cancelled":
            lines.append("Result: CANCELLED by user")
        elif self.status == "no_tests":
            lines.append("Result: NO TESTS RAN")
        elif self.status == "skipped":
            lines.append(f"Result: SKIPPED - {self.message}")
        else:
            lines.append(f"Result: ERROR - {self.message}")
        return "\n".join(lines)


class SelfHealingEngine:
    """Run a bounded test/repair/test loop for the current project."""

    def __init__(
        self,
        agent: "AIAgent",
        *,
        max_attempts: int = MAX_REPAIR_ATTEMPTS,
    ):
        self.agent = agent
        self.max_attempts = max(0, max_attempts)
        self.root = project_root()

    def heal(self, target: str = "") -> HealingResult:
        if self.agent.dry_run:
            return HealingResult(
                status="skipped",
                message="self-healing needs to run tests and apply edits",
            )
        if self.agent.backend is None:
            return HealingResult(
                status="error",
                message=(
                    "no LLM backend is available; install cliv[online] or "
                    "cliv[offline] to generate repairs"
                ),
            )

        initial = self._run_tests(target)
        if initial.cancelled:
            return HealingResult(status="cancelled", initial=initial)
        if initial.error:
            return HealingResult(
                status="error", initial=initial, message=initial.error
            )
        if initial.no_tests:
            return HealingResult(status="no_tests", initial=initial)
        if initial.ok:
            return HealingResult(status="passed", initial=initial, final=initial)

        current = initial
        attempts: List[RepairAttempt] = []
        modified: List[str] = []

        for number in range(1, self.max_attempts + 1):
            prompt = self._build_prompt(current, attempts)
            response = self.agent._generate(prompt, history=self.agent.messages)
            if response.startswith("Error:"):
                attempts.append(
                    RepairAttempt(number=number, generation_error=response)
                )
                return HealingResult(
                    status="error",
                    initial=initial,
                    final=current,
                    attempts=attempts,
                    modified_files=modified,
                    message=response,
                )

            edits, plan_error = parse_repair_plan(response, self.root)
            if plan_error or not edits:
                attempts.append(
                    RepairAttempt(
                        number=number,
                        plan_error=plan_error or "repair plan contained no edits",
                    )
                )
                continue

            applied, apply_error = self._apply_edits(edits)
            if apply_error == "cancelled":
                attempts.append(
                    RepairAttempt(number=number, apply_error="cancelled by user")
                )
                return HealingResult(
                    status="cancelled",
                    initial=initial,
                    final=current,
                    attempts=attempts,
                    modified_files=modified,
                )
            if apply_error:
                attempts.append(
                    RepairAttempt(number=number, apply_error=apply_error)
                )
                continue

            for path in applied:
                if path not in modified:
                    modified.append(path)

            report = self._run_tests(target)
            attempts.append(
                RepairAttempt(
                    number=number,
                    edited_files=applied,
                    test_report=report,
                )
            )
            if report.cancelled:
                return HealingResult(
                    status="cancelled",
                    initial=initial,
                    final=report,
                    attempts=attempts,
                    modified_files=modified,
                )
            if report.error:
                return HealingResult(
                    status="error",
                    initial=initial,
                    final=report,
                    attempts=attempts,
                    modified_files=modified,
                    message=report.error,
                )
            current = report
            if report.ok:
                return HealingResult(
                    status="passed",
                    initial=initial,
                    final=report,
                    attempts=attempts,
                    modified_files=modified,
                )

        return HealingResult(
            status="failed",
            initial=initial,
            final=current,
            attempts=attempts,
            modified_files=modified,
        )

    def _run_tests(self, target: str) -> TestReport:
        raw = self.agent._execute_tool("run_tests", {"target": target})
        report = parse_pytest_output(raw, self.root)
        if (
            report.exit_code is None
            and not report.error
            and not report.cancelled
            and not report.no_tests
        ):
            report.error = "could not parse pytest results"
        return report

    def _apply_edits(
        self, edits: Sequence[RepairEdit]
    ) -> Tuple[List[str], Optional[str]]:
        payload = {
            "edits": [
                {
                    "path": edit.path,
                    "new_string": edit.new_string,
                    **(
                        {"old_string": edit.old_string}
                        if edit.old_string
                        else {}
                    ),
                }
                for edit in edits
            ]
        }
        result = self.agent._execute_tool("multi_edit", payload)
        lowered = result.lower()
        if "cancelled by user" in lowered:
            return [], "cancelled"
        if not result.startswith("Successfully applied"):
            return [], result
        return [edit.path for edit in edits], None

    def _build_prompt(
        self, report: TestReport, attempts: Sequence[RepairAttempt]
    ) -> str:
        failure_lines = "\n".join(
            f"- {failure.name}"
            + (f": {failure.summary}" if failure.summary else "")
            for failure in report.failures[:10]
        ) or "(pytest did not report individual failure names)"

        history_lines = []
        for attempt in attempts:
            if attempt.plan_error:
                history_lines.append(
                    f"- Attempt {attempt.number}: invalid repair plan "
                    f"({attempt.plan_error}). Return valid JSON."
                )
            elif attempt.apply_error:
                history_lines.append(
                    f"- Attempt {attempt.number}: edit failed "
                    f"({attempt.apply_error}). Make sure old_string exactly "
                    "matches the current file."
                )
            elif attempt.test_report is not None:
                history_lines.append(
                    f"- Attempt {attempt.number}: edited "
                    f"{', '.join(attempt.edited_files)}; tests still failed "
                    f"({attempt.test_report.summary()})."
                )
        history = "\n".join(history_lines) or "(this is the first repair attempt)"

        file_context = self._file_context(report, attempts)
        output = report.output[:MAX_OUTPUT_CHARS]
        if len(report.output) > MAX_OUTPUT_CHARS:
            output += "\n[...pytest output truncated...]"

        return (
            "You are repairing a Python project after pytest failures.\n\n"
            "Return ONLY valid JSON in this exact shape:\n"
            '{"edits":[{"path":"relative/path.py",'
            '"old_string":"exact text to replace",'
            '"new_string":"replacement"}]}\n\n'
            "Rules:\n"
            "- Use old_string for a precise patch. It must match the current "
            "file exactly.\n"
            "- Omit old_string only when replacing a whole file.\n"
            "- Keep changes minimal and fix the source, not the test, unless "
            "the test is clearly wrong.\n"
            "- Do not delete tests or weaken assertions to make tests pass.\n"
            "- Return no markdown, no explanation, and at most 5 edits.\n\n"
            f"Previous repair attempts:\n{history}\n\n"
            f"Failing tests:\n{failure_lines}\n\n"
            f"Pytest output:\n{output}\n\n"
            f"Relevant files:\n{file_context}\n"
        )

    def _file_context(
        self, report: TestReport, attempts: Sequence[RepairAttempt]
    ) -> str:
        candidates: List[str] = []
        for path in report.files:
            if path not in candidates:
                candidates.append(path)
        for attempt in reversed(attempts):
            for path in attempt.edited_files:
                if path not in candidates:
                    candidates.insert(0, path)

        # A traceback often points only at the failing test.  Read test
        # imports so the model can also see the implementation under test.
        for path in list(candidates):
            if not _looks_like_test(path):
                continue
            content = self.agent._execute_tool("read_file", {"path": path})
            if content.startswith("Error:"):
                continue
            for match in _IMPORT_LINE.finditer(content):
                module = match.group(1) or match.group(2)
                if not module:
                    continue
                for possible in _module_files(module):
                    resolved, error = resolve_project_path(
                        possible, root=self.root
                    )
                    if error or resolved is None or not resolved.is_file():
                        continue
                    relative = resolved.relative_to(self.root).as_posix()
                    if relative not in candidates:
                        candidates.append(relative)

        chunks: List[str] = []
        total = 0
        for path in candidates[:MAX_CONTEXT_FILES]:
            content = self.agent._execute_tool("read_file", {"path": path})
            if content.startswith("Error:"):
                continue
            if len(content) > MAX_FILE_CHARS:
                content = content[:MAX_FILE_CHARS] + "\n[...file truncated...]"
            if total + len(content) > MAX_TOTAL_FILE_CHARS:
                break
            total += len(content)
            chunks.append(f"--- {path} ---\n{content}")

        return "\n\n".join(chunks) if chunks else "(no readable failure files)"
