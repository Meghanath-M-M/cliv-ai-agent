"""Tests for the bounded self-healing test repair loop."""

from pathlib import Path

from cliv.agent import AIAgent
from cliv.healing import SelfHealingEngine
from cliv.healing.repair_plan import parse_repair_plan
from cliv.healing.test_report import parse_pytest_output
from cliv.history import HistoryStore
from cliv.llm import LLMBackend
from cliv.models import SessionStats
from cliv.ui import NullUI


class RepairBackend(LLMBackend):
    name = "repair"
    model = "repair-model"

    def __init__(self, responses):
        super().__init__("repair-model", stats=SessionStats())
        self.responses = list(responses)
        self.prompts = []

    @classmethod
    def is_available(cls):
        return True

    def chat(self, messages, tools=None):
        raise AssertionError("self-healing should use non-tool generation")

    def generate(self, messages):
        self.prompts.append(messages[-1]["content"])
        if not self.responses:
            return "fallback response"
        return self.responses.pop(0)


def make_healing_agent(project_dir, backend, *, dry_run=False):
    if not isinstance(backend, RepairBackend):
        backend = RepairBackend(backend)
    return AIAgent(
        api_key=None,
        ui=NullUI(approve=True),
        history_store=HistoryStore(project_dir / "history.json"),
        backend=backend,
        mode="offline",
        use_offline_tools=False,
        auto_approve=True,
        dry_run=dry_run,
        internet_check=lambda: False,
    )


def _write_broken_project(root: Path) -> None:
    (root / "maths.py").write_text("def add(a, b):\n    return a - b\n")
    (root / "test_maths.py").write_text(
        "from maths import add\n\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n"
    )


def test_parse_pytest_output_extracts_counts_failures_and_files(tmp_path):
    (tmp_path / "test_maths.py").write_text(
        "from maths import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    (tmp_path / "maths.py").write_text("def add(a, b):\n    return a - b\n")
    output = """Pytest Results (exit 1):
============================= test session starts =============================
collected 1 item

=================================== FAILURES ===================================
__________________________________ test_add ___________________________________

    def test_add():
>       assert add(2, 3) == 5
E       assert -1 == 5

test_maths.py:4: AssertionError
=========================== short test summary info ============================
FAILED test_maths.py::test_add - assert -1 == 5
========================= 1 failed in 0.02s ==========================
"""

    report = parse_pytest_output(output, tmp_path)

    assert report.exit_code == 1
    assert report.failed == 1
    assert report.passed == 0
    assert report.failures[0].name == "test_maths.py::test_add"
    assert report.failures[0].summary == "assert -1 == 5"
    assert "test_maths.py" in report.files
    assert report.ok is False


def test_parse_pytest_output_cancelled_and_no_tests(tmp_path):
    cancelled = parse_pytest_output(
        "Tool 'run_tests' was cancelled by user.", tmp_path
    )
    assert cancelled.cancelled is True

    empty = parse_pytest_output(
        "Pytest Results (exit 5):\nno tests ran in 0.01s", tmp_path
    )
    assert empty.no_tests is True
    assert empty.ok is False


def test_parse_repair_plan_accepts_fenced_patch_and_full_file(tmp_path):
    (tmp_path / "app.py").write_text("def answer():\n    return 1\n")

    fenced = """```json
{"edits": [{"path": "app.py", "old_string": "return 1", "new_string": "return 2"}]}
```"""
    edits, error = parse_repair_plan(fenced, tmp_path)
    assert error is None
    assert edits[0].path == "app.py"
    assert edits[0].old_string == "return 1"
    assert edits[0].new_string == "return 2"

    full_file = '{"files": [{"path": "app.py", "content": "value = 3\\n"}]}'
    edits, error = parse_repair_plan(full_file, tmp_path)
    assert error is None
    assert edits[0].old_string == ""
    assert edits[0].new_string == "value = 3\n"


def test_parse_repair_plan_rejects_outside_and_sensitive_paths(tmp_path):
    edits, error = parse_repair_plan(
        '{"edits":[{"path":"../evil.py","new_string":"x"}]}', tmp_path
    )
    assert edits is None
    assert "outside" in error.lower()

    edits, error = parse_repair_plan(
        '{"edits":[{"path":".env","new_string":"x"}]}', tmp_path
    )
    assert edits is None
    assert "sensitive" in error.lower()


def test_chat_self_heal_repairs_failing_test(project_dir):
    _write_broken_project(project_dir)
    backend = RepairBackend(
        [
            '{"edits":[{"path":"maths.py","new_string":'
            '"def add(a, b):\\n    return a + b\\n"}]}'
        ]
    )
    agent = AIAgent(
        api_key=None,
        ui=NullUI(approve=True),
        history_store=HistoryStore(project_dir / "history.json"),
        backend=backend,
        mode="offline",
        use_offline_tools=False,
        auto_approve=True,
        internet_check=lambda: False,
    )

    result = agent.chat("Fix all failing tests")

    assert "Result: PASSED" in result
    assert "Initial tests: 1 failed" in result
    assert "Modified files: maths.py" in result
    assert "test_maths.py" in backend.prompts[0]
    assert "maths.py" in backend.prompts[0]
    assert "return a + b" in (project_dir / "maths.py").read_text()
    assert agent.messages[-1]["content"] == result


def test_self_healing_is_skipped_in_dry_run(project_dir):
    _write_broken_project(project_dir)
    backend = RepairBackend(
        ['{"edits":[{"path":"maths.py","new_string":"fixed"}]}']
    )
    agent = make_healing_agent(project_dir, backend, dry_run=True)

    result = agent.chat("make the tests pass")

    assert "Result: SKIPPED" in result
    assert backend.prompts == []
    assert "return a - b" in (project_dir / "maths.py").read_text()


def test_engine_steps_are_bounded_and_require_approval(project_dir, monkeypatch):
    _write_broken_project(project_dir)
    monkeypatch.chdir(project_dir)
    backend = RepairBackend(
        [
            '{"edits":[{"path":"maths.py","new_string":"def add(a, b):\\n    return 0\\n"}]}',
            '{"edits":[{"path":"maths.py","new_string":"def add(a, b):\\n    return 0\\n"}]}',
        ]
    )
    agent = AIAgent(
        api_key=None,
        ui=NullUI(approve=True),
        history_store=HistoryStore(project_dir / "history.json"),
        backend=backend,
        mode="offline",
        use_offline_tools=False,
        auto_approve=True,
        internet_check=lambda: False,
    )

    result = SelfHealingEngine(agent, max_attempts=2).heal()

    assert result.status == "failed"
    assert len(result.attempts) == 2
    assert "Result: FAILED after 2 repair attempt(s)." in result.render()
