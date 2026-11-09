"""Integration tests for the deterministic offline fallback."""

from pathlib import Path

from cliv.agent import AIAgent
from cliv.history import HistoryStore
from cliv.llm import LLMBackend
from cliv.models import SessionStats
from cliv.ui import NullUI


class ScriptedBackend(LLMBackend):
    name = "scripted"
    model = "scripted-model"

    def __init__(self, responses):
        super().__init__("scripted-model", stats=SessionStats())
        self.responses = list(responses)
        self.prompts = []

    @classmethod
    def is_available(cls):
        return True

    def chat(self, messages, tools=None):
        raise AssertionError("offline fallback should not use tool calling")

    def generate(self, messages):
        self.prompts.append(messages[-1]["content"])
        if not self.responses:
            return "fallback response"
        return self.responses.pop(0)


def make_offline_agent(tmp_path, responses, approve=True):
    return AIAgent(
        api_key=None,
        ui=NullUI(approve=approve),
        history_store=HistoryStore(tmp_path / "history.json"),
        backend=ScriptedBackend(responses),
        mode="offline",
        use_offline_tools=False,
        internet_check=lambda: False,
    )


def test_offline_read_includes_file_content_in_generation_prompt(project_dir):
    (project_dir / "notes.txt").write_text("The code is ORANGE-42.\n")
    agent = make_offline_agent(project_dir, ["The code is ORANGE-42."])
    result = agent.chat("read notes.txt")
    assert result == "The code is ORANGE-42."
    assert "ORANGE-42" in agent.backend.prompts[-1]


def test_offline_create_writes_file(project_dir):
    agent = make_offline_agent(
        project_dir, ["print('hello')\n", "Created hello.py"]
    )
    result = agent.chat("create hello.py that prints hello")
    assert (project_dir / "hello.py").read_text() == "print('hello')"
    assert "Successfully wrote" in result


def test_offline_fix_syntax_writes_corrected_file(project_dir):
    (project_dir / "broken.py").write_text("def add(a, b)\n    return a + b\n")
    fixed = "def add(a, b):\n    return a + b\n"
    agent = make_offline_agent(project_dir, [fixed, "Fixed the syntax error."])
    result = agent.chat("fix the syntax error in broken.py")
    assert (project_dir / "broken.py").read_text() == fixed.strip()
    assert "Fixed the syntax error." in result


def test_offline_remove_deletes_file(project_dir):
    (project_dir / "old.txt").write_text("bye")
    agent = make_offline_agent(project_dir, [])
    result = agent.chat("delete old.txt")
    assert not (project_dir / "old.txt").exists()
    assert "Removed" in result


def test_offline_unknown_uses_generation(project_dir):
    agent = make_offline_agent(project_dir, ["Paris is the capital of France."])
    result = agent.chat("what is the capital of France?")
    assert result == "Paris is the capital of France."


def test_offline_project_analysis_returns_project_dna_without_generation(project_dir):
    (project_dir / "app.py").write_text(
        "def main():\n    pass\n\nif __name__ == \"__main__\":\n    main()\n"
    )
    (project_dir / "utils").mkdir()
    (project_dir / "utils" / "helpers.py").write_text(
        "def helper():\n    return 1\n"
    )
    agent = make_offline_agent(project_dir, [])

    result = agent.chat("Analyse this project.")

    assert "CLIV PROJECT INTELLIGENCE" in result
    assert "app.py" in result
    assert "utils/helpers.py" in result
    assert agent.backend.prompts == []


def test_project_analysis_works_without_any_llm_backend(project_dir):
    (project_dir / "app.py").write_text("VALUE = 1\n")
    agent = make_offline_agent(project_dir, [])
    agent.backend = None

    result = agent.chat("Analyse this project.")

    assert "CLIV PROJECT INTELLIGENCE" in result
    assert "app.py" in result
