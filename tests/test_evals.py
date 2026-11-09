"""Sanity tests for the fixed evaluation set."""

from pathlib import Path

from cliv.intelligence import analyse_project
from evals.run import run_task
from evals.tasks import EvalTask, available_tasks, dependencies_available


class FakeAgent:
    def __init__(self, response="ok"):
        self.response = response
        self.prompts = []

    def chat(self, prompt):
        self.prompts.append(prompt)
        return self.response


def test_eval_set_is_substantial_and_unique():
    tasks = available_tasks()
    assert len(tasks) >= 10
    assert len({task.name for task in tasks}) == len(tasks)


def test_eval_dependencies_are_described():
    for task in available_tasks():
        available, reason = dependencies_available(task)
        assert isinstance(available, bool)
        assert isinstance(reason, str)


def test_project_dna_eval_task_matches_the_real_report(tmp_path):
    task = next(task for task in available_tasks() if task.name == "project_dna")
    task.setup(tmp_path)
    report = analyse_project(tmp_path)
    ok, detail = task.check(tmp_path, report)
    assert ok is True, detail


def test_run_task_executes_setup_and_check(tmp_path):
    task = EvalTask(
        "synthetic",
        "do the thing",
        lambda root: (root / "result.txt").write_text("done"),
        lambda root, response: (
            (root / "result.txt").exists() and response == "ok",
            "checked",
        ),
    )
    agent = FakeAgent("ok")
    ok, detail = run_task(task, agent, tmp_path)
    assert ok is True
    assert detail == "checked"
    assert agent.prompts == ["do the thing"]
