"""Unit tests for agent long-term memory system."""

from pathlib import Path
from types import SimpleNamespace
import pytest
from rich.console import Console

from cliv.memory import MemoryEntry, MemoryStore
from cliv.tools.memory import RememberTool, RecallTool
from cliv.agent import AIAgent
from cliv.slash import SlashContext, dispatch
import cliv.cli as cli


def test_memory_entry_serialization():
    entry = MemoryEntry(
        id="test-pref",
        content="Always write docstrings",
        scope="global",
        created_at="2026-11-11T10:00:00",
        updated_at="2026-11-11T10:00:00",
        category="convention",
    )
    d = entry.to_dict()
    assert d["id"] == "test-pref"
    assert d["content"] == "Always write docstrings"
    assert d["scope"] == "global"

    restored = MemoryEntry.from_dict(d)
    assert restored.id == entry.id
    assert restored.content == entry.content
    assert restored.scope == "global"
    assert restored.category == "convention"


def test_add_and_list_memories_scoped(tmp_path):
    global_file = tmp_path / "global_memory.json"
    proj_dir = tmp_path / "my_project"
    store = MemoryStore(project_path=proj_dir, global_file=global_file)

    # Add global preference
    e_global = store.add(
        content="I prefer pytest fixtures over setup/teardown methods.",
        scope="global",
        key="pytest-fixtures",
        category="preference",
    )
    assert e_global.scope == "global"
    assert e_global.id == "pytest-fixtures"

    # Add project convention
    e_proj = store.add(
        content="Run alembic migrations with alembic upgrade head.",
        scope="project",
        key="alembic-migrations",
        category="tooling",
    )
    assert e_proj.scope == "project"

    # List all
    all_memories = store.list_entries()
    assert len(all_memories) == 2

    # Filter global
    globals_only = store.list_entries(scope="global")
    assert len(globals_only) == 1
    assert globals_only[0].id == "pytest-fixtures"

    # Filter project
    projects_only = store.list_entries(scope="project")
    assert len(projects_only) == 1
    assert projects_only[0].id == "alembic-migrations"


def test_update_existing_memory(tmp_path):
    store = MemoryStore(project_path=tmp_path, global_file=tmp_path / "global.json")

    store.add(content="Initial port is 5000", key="port-cfg", scope="project")
    first_list = store.list_entries()
    created_at = first_list[0].created_at

    # Update with same key
    updated = store.add(content="Updated port is 8080", key="port-cfg", scope="project")
    assert updated.content == "Updated port is 8080"
    assert updated.created_at == created_at

    entries = store.list_entries(scope="project")
    assert len(entries) == 1
    assert entries[0].content == "Updated port is 8080"


def test_remove_and_clear_memories(tmp_path):
    store = MemoryStore(project_path=tmp_path, global_file=tmp_path / "global.json")

    store.add(content="Global rule A", scope="global", key="rule-a")
    store.add(content="Project rule B", scope="project", key="rule-b")
    store.add(content="Project rule C", scope="project", key="rule-c")

    # Remove by ID
    assert store.remove("rule-b") is True
    assert len(store.list_entries(scope="project")) == 1

    # Clear project only
    assert store.clear(scope="project") == 1
    assert len(store.list_entries(scope="project")) == 0
    assert len(store.list_entries(scope="global")) == 1

    # Clear global
    assert store.clear(scope="global") == 1
    assert len(store.list_entries()) == 0


def test_format_prompt_context_with_cliv_md(tmp_path):
    store = MemoryStore(project_path=tmp_path, global_file=tmp_path / "global.json")

    # Empty store produces empty prompt context
    assert store.format_prompt_context() == ""

    # Add CLIV.md
    (tmp_path / "CLIV.md").write_text("# Team Guidelines\nAlways use black formatter.", encoding="utf-8")

    # Add remembered item
    store.add(content="Prefer uv for package management", scope="global", category="tooling")
    store.add(content="FastAPI endpoints return pydantic models", scope="project", category="architecture")

    context = store.format_prompt_context()
    assert "## Long-Term Memory & Conventions" in context
    assert "Always use black formatter." in context
    assert "Prefer uv for package management" in context
    assert "FastAPI endpoints return pydantic models" in context


def test_remember_and_recall_tools(tmp_path):
    store = MemoryStore(project_path=tmp_path, global_file=tmp_path / "global.json")
    remember_tool = RememberTool(store=store)
    recall_tool = RecallTool(store=store)

    # Remember
    res_rem = remember_tool.execute(
        content="Do not use eval() under any circumstances",
        scope="project",
        key="no-eval",
        category="convention",
    )
    assert "Remembered in project memory" in res_rem
    assert "no-eval" in res_rem

    # Recall all
    res_rec = recall_tool.execute()
    assert "no-eval" in res_rec
    assert "Do not use eval()" in res_rec

    # Recall with search filter matching
    assert "no-eval" in recall_tool.execute(query="eval")

    # Recall with search filter not matching
    assert "No memories found" in recall_tool.execute(query="nonexistent_concept")


def test_agent_injects_memory_into_system_prompt(tmp_path):
    store = MemoryStore(project_path=tmp_path, global_file=tmp_path / "global.json")
    store.add(content="User prefers strict typing with mypy", scope="global")

    agent = AIAgent(force_offline=True, memory_store=store)
    sys_msg = agent._build_system_message()

    assert "## Long-Term Memory & Conventions" in sys_msg["content"]
    assert "User prefers strict typing with mypy" in sys_msg["content"]
    assert "LONG-TERM MEMORY PROTOCOL" in sys_msg["content"]


def test_slash_dispatch_memory(tmp_path):
    store = MemoryStore(project_path=tmp_path, global_file=tmp_path / "global.json")
    agent = SimpleNamespace(memory=store, tools={})
    console = Console(record=True, width=120)
    ctx = SlashContext(agent=agent, console=console)

    # /memory list (empty)
    assert dispatch("/memory", ctx) is True
    assert "No long-term memories" in console.export_text()

    # /memory add
    console = Console(record=True, width=120)
    ctx = SlashContext(agent=agent, console=console)
    assert dispatch("/memory add Always run pytest with -v", ctx) is True
    out_add = console.export_text()
    assert "Saved project memory" in out_add

    # /memory list (populated)
    console = Console(record=True, width=120)
    ctx = SlashContext(agent=agent, console=console)
    assert dispatch("/memory list", ctx) is True
    out_list = console.export_text()
    assert "Always run pytest with -v" in out_list

    # /memory remove
    console = Console(record=True, width=120)
    ctx = SlashContext(agent=agent, console=console)
    assert dispatch("/memory remove pytest", ctx) is True
    assert "Removed memory matching" in console.export_text()


def test_cmd_memory_cli(tmp_path, monkeypatch):
    store = MemoryStore(project_path=tmp_path, global_file=tmp_path / "global.json")
    monkeypatch.setattr("cliv.memory.MemoryStore", lambda **kwargs: store)

    test_console = Console(record=True, width=120)
    monkeypatch.setattr(cli, "console", test_console)

    # cliv memory add
    cli.cmd_memory(SimpleNamespace(memory_command="add", content="Use type annotations", is_global=False, category="general"))
    assert "Saved project memory" in test_console.export_text()

    # cliv memory list
    test_console = Console(record=True, width=120)
    monkeypatch.setattr(cli, "console", test_console)
    cli.cmd_memory(SimpleNamespace(memory_command="list", scope=None))
    assert "Use type annotations" in test_console.export_text()

    # cliv memory clear
    test_console = Console(record=True, width=120)
    monkeypatch.setattr(cli, "console", test_console)
    cli.cmd_memory(SimpleNamespace(memory_command="clear", scope="project"))
    assert "Cleared 1 project memories" in test_console.export_text()
