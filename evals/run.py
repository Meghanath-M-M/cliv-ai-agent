"""Run the fixed evaluation set.

Usage:
    python -m evals.run --offline
    python -m evals.run --api-key ... --task create_file
    python -m evals.run --list

Each task runs in a fresh temporary project and is scored pass/fail.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List, Tuple

from cliv.agent import AIAgent
from cliv.config import get_api_key
from cliv.history import HistoryStore
from cliv.ui import NullUI

from evals.tasks import EvalTask, available_tasks, dependencies_available


def run_task(task: EvalTask, agent: AIAgent, root: Path) -> Tuple[bool, str]:
    """Set up *root*, run one prompt, and apply the task's checker."""

    task.setup(root)
    response = agent.chat(task.prompt)
    return task.check(root, response)


def _status_label(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run cliv's evaluation set.")
    parser.add_argument("--api-key", help="Groq API key (otherwise env/config)")
    parser.add_argument("--offline", action="store_true", help="Force Ollama")
    parser.add_argument(
        "--legacy-offline",
        action="store_true",
        help="Use regex intent routing instead of Ollama tool calling",
    )
    parser.add_argument(
        "--model", default="llama3.1:8b", help="Local Ollama model for offline runs"
    )
    parser.add_argument("--task", action="append", help="Run only this task (repeatable)")
    parser.add_argument("--list", action="store_true", help="List task names and exit")
    parser.add_argument("--keep", action="store_true", help="Keep temporary project dirs")
    args = parser.parse_args(argv)

    tasks = available_tasks()
    if args.list:
        for task in tasks:
            print(task.name)
        return 0

    if args.task:
        selected_names = set(args.task)
        tasks = [task for task in tasks if task.name in selected_names]
        missing = selected_names - {task.name for task in tasks}
        if missing:
            print(f"Unknown task(s): {', '.join(sorted(missing))}")
            return 2

    api_key = args.api_key or get_api_key()
    results = []
    failures = 0

    for task in tasks:
        available, reason = dependencies_available(task)
        if not available:
            print(f"SKIP {task.name}: {reason}")
            results.append((task.name, "SKIP", reason))
            continue

        project_dir = Path(tempfile.mkdtemp(prefix=f"cliv-eval-{task.name}-"))
        history_dir = Path(tempfile.mkdtemp(prefix="cliv-eval-history-"))
        old_cwd = Path.cwd()
        old_root = os.environ.get("CLIV_PROJECT_ROOT")
        try:
            os.environ["CLIV_PROJECT_ROOT"] = str(project_dir)
            os.chdir(project_dir)
            agent = AIAgent(
                api_key=api_key,
                local_model=args.model,
                force_offline=args.offline,
                auto_approve=True,
                use_offline_tools=not args.legacy_offline,
                ui=NullUI(approve=True),
                history_store=HistoryStore(
                    history_dir / f"{uuid.uuid4().hex}.json", max_messages=50
                ),
            )
            ok, detail = run_task(task, agent, project_dir)
        except Exception as exc:  # noqa: BLE001 - one task must not abort the suite
            ok, detail = False, f"exception: {exc}"
        finally:
            os.chdir(old_cwd)
            if old_root is None:
                os.environ.pop("CLIV_PROJECT_ROOT", None)
            else:
                os.environ["CLIV_PROJECT_ROOT"] = old_root
            if not args.keep:
                shutil.rmtree(project_dir, ignore_errors=True)
                shutil.rmtree(history_dir, ignore_errors=True)
            else:
                detail = f"{detail} (kept: {project_dir})"

        if not ok:
            failures += 1
        results.append((task.name, _status_label(ok), detail))
        print(f"{_status_label(ok):4} {task.name}: {detail}")

    passed = sum(1 for _, status, _ in results if status == "PASS")
    skipped = sum(1 for _, status, _ in results if status == "SKIP")
    total = len(results)
    print(f"\nScore: {passed}/{total} passed, {failures} failed, {skipped} skipped")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
