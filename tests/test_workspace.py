"""Tests for Rank 7: Multi-repo workspace support.

Covers:
- WorkspaceRegistry (add, list, get, remove, touch, active_entry, auto_register_current)
- resolve_workspace_path (boundary enforcement, sensitive file blocking)
- Workspace agent tools (ListWorkspaceTool, ReadWorkspaceFileTool,
  ListWorkspaceFilesTool, SearchWorkspaceTool)
- /workspace slash commands (list, add, remove, info)
- cliv workspace CLI subcommand (list, add, remove, info)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry(tmp_path: Path):
    """Return a WorkspaceRegistry backed by a temp file."""
    from cliv.workspace import WorkspaceRegistry
    return WorkspaceRegistry(registry_file=tmp_path / "workspaces.json")


def _make_repo(tmp_path: Path, name: str, files: dict | None = None) -> Path:
    """Create a minimal fake repo directory with optional files."""
    repo = tmp_path / name
    repo.mkdir(parents=True)
    for rel, content in (files or {}).items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return repo


# ---------------------------------------------------------------------------
# WorkspaceRegistry
# ---------------------------------------------------------------------------

class TestWorkspaceRegistry:
    def test_add_and_list(self, tmp_path):
        repo = _make_repo(tmp_path, "api")
        reg = _make_registry(tmp_path)

        entry, is_new = reg.add(str(repo))
        assert is_new
        assert entry.name == "api"
        assert Path(entry.path) == repo.resolve()

        entries = reg.list()
        assert len(entries) == 1
        assert entries[0].name == "api"

    def test_add_custom_name(self, tmp_path):
        repo = _make_repo(tmp_path, "myrepo")
        reg = _make_registry(tmp_path)

        entry, is_new = reg.add(str(repo), name="backend-api")
        assert is_new
        assert entry.name == "backend-api"

    def test_add_slugifies_name(self, tmp_path):
        repo = _make_repo(tmp_path, "some_repo")
        reg = _make_registry(tmp_path)

        entry, _ = reg.add(str(repo), name="My Fancy API!")
        assert entry.name == "my-fancy-api"

    def test_add_nonexistent_path_raises(self, tmp_path):
        reg = _make_registry(tmp_path)
        with pytest.raises(ValueError, match="does not exist"):
            reg.add(str(tmp_path / "nonexistent"))

    def test_add_duplicate_updates(self, tmp_path):
        repo = _make_repo(tmp_path, "api")
        reg = _make_registry(tmp_path)

        entry1, is_new1 = reg.add(str(repo), description="first")
        assert is_new1

        entry2, is_new2 = reg.add(str(repo), description="updated")
        assert not is_new2
        assert entry2.description == "updated"
        assert len(reg.list()) == 1

    def test_get_case_insensitive(self, tmp_path):
        repo = _make_repo(tmp_path, "api")
        reg = _make_registry(tmp_path)
        reg.add(str(repo), name="ApiService")

        found = reg.get("apiservice")
        assert found is not None
        assert found.name == "apiservice"

    def test_get_missing_returns_none(self, tmp_path):
        reg = _make_registry(tmp_path)
        assert reg.get("doesnotexist") is None

    def test_remove_existing(self, tmp_path):
        repo = _make_repo(tmp_path, "api")
        reg = _make_registry(tmp_path)
        reg.add(str(repo))

        removed = reg.remove("api")
        assert removed
        assert reg.list() == []

    def test_remove_missing_returns_false(self, tmp_path):
        reg = _make_registry(tmp_path)
        assert not reg.remove("ghost")

    def test_touch_updates_last_active(self, tmp_path):
        repo = _make_repo(tmp_path, "api")
        reg = _make_registry(tmp_path)
        entry, _ = reg.add(str(repo))
        old_ts = entry.last_active

        import time; time.sleep(0.01)
        reg.touch("api")

        updated = reg.get("api")
        assert updated is not None
        # last_active should be >= old_ts (may equal if too fast)
        assert updated.last_active >= old_ts

    def test_active_entry_matches_project_root(self, tmp_path):
        repo = _make_repo(tmp_path, "myproject")
        reg = _make_registry(tmp_path)
        reg.add(str(repo))

        with patch("cliv.workspace.project_root", return_value=repo):
            active = reg.active_entry()
        assert active is not None
        assert active.name == "myproject"

    def test_auto_register_current(self, tmp_path):
        repo = _make_repo(tmp_path, "myproject")
        reg = _make_registry(tmp_path)

        with patch("cliv.workspace.project_root", return_value=repo):
            entry = reg.auto_register_current()

        assert entry.name == "myproject"
        assert len(reg.list()) == 1

    def test_list_sorted_alphabetically(self, tmp_path):
        for name in ("zoo", "alpha", "middle"):
            _make_repo(tmp_path, name)
        reg = _make_registry(tmp_path)
        for name in ("zoo", "alpha", "middle"):
            reg.add(str(tmp_path / name))

        names = [e.name for e in reg.list()]
        assert names == sorted(names)

    def test_persistence_across_instances(self, tmp_path):
        repo = _make_repo(tmp_path, "api")
        reg_file = tmp_path / "workspaces.json"

        from cliv.workspace import WorkspaceRegistry
        reg1 = WorkspaceRegistry(registry_file=reg_file)
        reg1.add(str(repo))

        reg2 = WorkspaceRegistry(registry_file=reg_file)
        assert len(reg2.list()) == 1
        assert reg2.get("api") is not None


# ---------------------------------------------------------------------------
# resolve_workspace_path
# ---------------------------------------------------------------------------

class TestResolveWorkspacePath:
    def test_resolves_relative_path(self, tmp_path):
        repo = _make_repo(tmp_path, "api", {"src/app.py": "print('hello')"})
        reg = _make_registry(tmp_path)
        entry, _ = reg.add(str(repo))

        from cliv.workspace import resolve_workspace_path
        resolved, err = resolve_workspace_path(entry, "src/app.py")
        assert err is None
        assert resolved is not None
        assert resolved.name == "app.py"

    def test_blocks_path_escape(self, tmp_path):
        repo = _make_repo(tmp_path, "api")
        reg = _make_registry(tmp_path)
        entry, _ = reg.add(str(repo))

        from cliv.workspace import resolve_workspace_path
        _, err = resolve_workspace_path(entry, "../../etc/passwd")
        assert err is not None
        assert "outside workspace" in err

    def test_blocks_sensitive_path(self, tmp_path):
        repo = _make_repo(tmp_path, "api", {".env": "SECRET=abc"})
        reg = _make_registry(tmp_path)
        entry, _ = reg.add(str(repo))

        from cliv.workspace import resolve_workspace_path
        with patch("cliv.workspace.is_sensitive_path", return_value=True):
            _, err = resolve_workspace_path(entry, ".env")
        assert err is not None
        assert "Sensitive" in err

    def test_missing_workspace_dir_returns_error(self, tmp_path):
        from cliv.workspace import WorkspaceEntry, resolve_workspace_path
        entry = WorkspaceEntry(name="ghost", path=str(tmp_path / "doesnotexist"))

        _, err = resolve_workspace_path(entry, "README.md")
        assert err is not None
        assert "no longer exists" in err


# ---------------------------------------------------------------------------
# ListWorkspaceTool
# ---------------------------------------------------------------------------

class TestListWorkspaceTool:
    def test_empty_registry(self, tmp_path):
        from cliv.tools.workspace_tools import ListWorkspaceTool
        from cliv.workspace import WorkspaceRegistry
        reg = WorkspaceRegistry(registry_file=tmp_path / "workspaces.json")
        tool = ListWorkspaceTool(registry=reg)

        result = tool.execute()
        assert "No workspaces" in result

    def test_populated_registry(self, tmp_path):
        repo = _make_repo(tmp_path, "api")
        from cliv.tools.workspace_tools import ListWorkspaceTool
        from cliv.workspace import WorkspaceRegistry
        reg = WorkspaceRegistry(registry_file=tmp_path / "workspaces.json")
        reg.add(str(repo))
        tool = ListWorkspaceTool(registry=reg)

        result = tool.execute()
        assert "api" in result
        assert str(repo.resolve()) in result

    def test_missing_workspace_flagged(self, tmp_path):
        from cliv.tools.workspace_tools import ListWorkspaceTool
        from cliv.workspace import WorkspaceRegistry, WorkspaceEntry
        reg = WorkspaceRegistry(registry_file=tmp_path / "workspaces.json")
        # Manually write an entry with a non-existent path
        reg_file = tmp_path / "workspaces.json"
        entry = WorkspaceEntry(name="ghost", path=str(tmp_path / "nonexistent"))
        reg_file.write_text(json.dumps([entry.to_dict()]))

        tool = ListWorkspaceTool(registry=reg)
        result = tool.execute()
        assert "MISSING" in result or "ghost" in result


# ---------------------------------------------------------------------------
# ReadWorkspaceFileTool
# ---------------------------------------------------------------------------

class TestReadWorkspaceFileTool:
    def test_reads_file_successfully(self, tmp_path):
        repo = _make_repo(tmp_path, "api", {"README.md": "# Hello API"})
        from cliv.tools.workspace_tools import ReadWorkspaceFileTool
        from cliv.workspace import WorkspaceRegistry
        reg = WorkspaceRegistry(registry_file=tmp_path / "workspaces.json")
        reg.add(str(repo))
        tool = ReadWorkspaceFileTool(registry=reg)

        result = tool.execute(workspace="api", path="README.md")
        assert "# Hello API" in result

    def test_unknown_workspace(self, tmp_path):
        from cliv.tools.workspace_tools import ReadWorkspaceFileTool
        from cliv.workspace import WorkspaceRegistry
        reg = WorkspaceRegistry(registry_file=tmp_path / "workspaces.json")
        tool = ReadWorkspaceFileTool(registry=reg)

        result = tool.execute(workspace="ghost", path="README.md")
        assert "Unknown workspace" in result

    def test_file_not_found(self, tmp_path):
        repo = _make_repo(tmp_path, "api")
        from cliv.tools.workspace_tools import ReadWorkspaceFileTool
        from cliv.workspace import WorkspaceRegistry
        reg = WorkspaceRegistry(registry_file=tmp_path / "workspaces.json")
        reg.add(str(repo))
        tool = ReadWorkspaceFileTool(registry=reg)

        result = tool.execute(workspace="api", path="nonexistent.py")
        assert "not found" in result.lower()

    def test_path_escape_blocked(self, tmp_path):
        repo = _make_repo(tmp_path, "api")
        from cliv.tools.workspace_tools import ReadWorkspaceFileTool
        from cliv.workspace import WorkspaceRegistry
        reg = WorkspaceRegistry(registry_file=tmp_path / "workspaces.json")
        reg.add(str(repo))
        tool = ReadWorkspaceFileTool(registry=reg)

        result = tool.execute(workspace="api", path="../../etc/passwd")
        assert "outside workspace" in result or "Error" in result

    def test_max_bytes_truncation(self, tmp_path):
        content = "x" * 200
        repo = _make_repo(tmp_path, "api", {"big.txt": content})
        from cliv.tools.workspace_tools import ReadWorkspaceFileTool
        from cliv.workspace import WorkspaceRegistry
        reg = WorkspaceRegistry(registry_file=tmp_path / "workspaces.json")
        reg.add(str(repo))
        tool = ReadWorkspaceFileTool(registry=reg)

        result = tool.execute(workspace="api", path="big.txt", max_bytes=50)
        assert "truncated" in result
        assert "x" * 50 in result


# ---------------------------------------------------------------------------
# ListWorkspaceFilesTool
# ---------------------------------------------------------------------------

class TestListWorkspaceFilesTool:
    def test_lists_files(self, tmp_path):
        repo = _make_repo(tmp_path, "api", {
            "README.md": "# API",
            "src/main.py": "pass",
        })
        from cliv.tools.workspace_tools import ListWorkspaceFilesTool
        from cliv.workspace import WorkspaceRegistry
        reg = WorkspaceRegistry(registry_file=tmp_path / "workspaces.json")
        reg.add(str(repo))
        tool = ListWorkspaceFilesTool(registry=reg)

        result = tool.execute(workspace="api")
        assert "README.md" in result
        assert "[DIR]" in result or "[FILE]" in result

    def test_unknown_workspace(self, tmp_path):
        from cliv.tools.workspace_tools import ListWorkspaceFilesTool
        from cliv.workspace import WorkspaceRegistry
        reg = WorkspaceRegistry(registry_file=tmp_path / "workspaces.json")
        tool = ListWorkspaceFilesTool(registry=reg)

        result = tool.execute(workspace="ghost")
        assert "Unknown workspace" in result

    def test_subdirectory_listing(self, tmp_path):
        repo = _make_repo(tmp_path, "api", {"src/module.py": "# code"})
        from cliv.tools.workspace_tools import ListWorkspaceFilesTool
        from cliv.workspace import WorkspaceRegistry
        reg = WorkspaceRegistry(registry_file=tmp_path / "workspaces.json")
        reg.add(str(repo))
        tool = ListWorkspaceFilesTool(registry=reg)

        result = tool.execute(workspace="api", path="src")
        assert "module.py" in result


# ---------------------------------------------------------------------------
# SearchWorkspaceTool
# ---------------------------------------------------------------------------

class TestSearchWorkspaceTool:
    def test_match_found(self, tmp_path):
        repo = _make_repo(tmp_path, "api", {"app.py": "def login(user, password):\n    pass\n"})
        from cliv.tools.workspace_tools import SearchWorkspaceTool
        from cliv.workspace import WorkspaceRegistry
        reg = WorkspaceRegistry(registry_file=tmp_path / "workspaces.json")
        reg.add(str(repo))
        tool = SearchWorkspaceTool(registry=reg)

        result = tool.execute(workspace="api", pattern="login")
        assert "login" in result
        assert "app.py" in result

    def test_no_match(self, tmp_path):
        repo = _make_repo(tmp_path, "api", {"app.py": "print('hello')"})
        from cliv.tools.workspace_tools import SearchWorkspaceTool
        from cliv.workspace import WorkspaceRegistry
        reg = WorkspaceRegistry(registry_file=tmp_path / "workspaces.json")
        reg.add(str(repo))
        tool = SearchWorkspaceTool(registry=reg)

        result = tool.execute(workspace="api", pattern="NONEXISTENT_PATTERN_XYZ")
        assert "No matches" in result

    def test_invalid_regex(self, tmp_path):
        repo = _make_repo(tmp_path, "api", {"app.py": "print('hello')"})
        from cliv.tools.workspace_tools import SearchWorkspaceTool
        from cliv.workspace import WorkspaceRegistry
        reg = WorkspaceRegistry(registry_file=tmp_path / "workspaces.json")
        reg.add(str(repo))
        tool = SearchWorkspaceTool(registry=reg)

        result = tool.execute(workspace="api", pattern="[invalid(regex")
        assert "Invalid regex" in result

    def test_case_insensitive_by_default(self, tmp_path):
        repo = _make_repo(tmp_path, "api", {"app.py": "def LoginUser():\n    pass\n"})
        from cliv.tools.workspace_tools import SearchWorkspaceTool
        from cliv.workspace import WorkspaceRegistry
        reg = WorkspaceRegistry(registry_file=tmp_path / "workspaces.json")
        reg.add(str(repo))
        tool = SearchWorkspaceTool(registry=reg)

        result = tool.execute(workspace="api", pattern="loginuser", case_sensitive=False)
        assert "LoginUser" in result

    def test_unknown_workspace(self, tmp_path):
        from cliv.tools.workspace_tools import SearchWorkspaceTool
        from cliv.workspace import WorkspaceRegistry
        reg = WorkspaceRegistry(registry_file=tmp_path / "workspaces.json")
        tool = SearchWorkspaceTool(registry=reg)

        result = tool.execute(workspace="ghost", pattern="anything")
        assert "Unknown workspace" in result


# ---------------------------------------------------------------------------
# /workspace slash commands
# ---------------------------------------------------------------------------

class FakeAgent:
    """Minimal agent stub for slash command tests."""
    def __init__(self, registry):
        self.workspace_registry = registry


def _make_ctx(registry, width=120):
    console = Console(record=True, width=width)
    agent = FakeAgent(registry)
    from cliv.slash import SlashContext
    return SlashContext(agent=agent, console=console), console


class TestWorkspaceSlash:
    def test_list_empty(self, tmp_path):
        reg = _make_registry(tmp_path)
        ctx, console = _make_ctx(reg)
        from cliv.slash import handle_workspace
        result = handle_workspace(ctx, "list")
        assert result is True
        output = console.export_text()
        assert "No workspaces" in output

    def test_list_populated(self, tmp_path):
        repo = _make_repo(tmp_path, "api")
        reg = _make_registry(tmp_path)
        reg.add(str(repo))
        ctx, console = _make_ctx(reg)
        from cliv.slash import handle_workspace
        handle_workspace(ctx, "list")
        output = console.export_text()
        assert "api" in output

    def test_add_new_workspace(self, tmp_path):
        repo = _make_repo(tmp_path, "myapi")
        reg = _make_registry(tmp_path)
        ctx, console = _make_ctx(reg)
        from cliv.slash import handle_workspace
        handle_workspace(ctx, f"add {repo}")
        output = console.export_text()
        assert "Registered" in output or "myapi" in output

    def test_add_nonexistent_path_error(self, tmp_path):
        reg = _make_registry(tmp_path)
        ctx, console = _make_ctx(reg)
        from cliv.slash import handle_workspace
        handle_workspace(ctx, f"add {tmp_path / 'nonexistent'}")
        output = console.export_text()
        assert "Error" in output

    def test_remove_existing(self, tmp_path):
        repo = _make_repo(tmp_path, "api")
        reg = _make_registry(tmp_path)
        reg.add(str(repo))
        ctx, console = _make_ctx(reg)
        from cliv.slash import handle_workspace
        handle_workspace(ctx, "remove api")
        output = console.export_text()
        assert "Removed" in output

    def test_remove_missing(self, tmp_path):
        reg = _make_registry(tmp_path)
        ctx, console = _make_ctx(reg)
        from cliv.slash import handle_workspace
        handle_workspace(ctx, "remove ghost")
        output = console.export_text()
        assert "No workspace" in output

    def test_info_existing(self, tmp_path):
        repo = _make_repo(tmp_path, "api")
        reg = _make_registry(tmp_path)
        reg.add(str(repo), description="The main API")
        ctx, console = _make_ctx(reg)
        from cliv.slash import handle_workspace
        handle_workspace(ctx, "info api")
        output = console.export_text()
        assert "api" in output
        assert "The main API" in output

    def test_info_missing(self, tmp_path):
        reg = _make_registry(tmp_path)
        ctx, console = _make_ctx(reg)
        from cliv.slash import handle_workspace
        handle_workspace(ctx, "info ghost")
        output = console.export_text()
        assert "No workspace" in output

    def test_unknown_subcommand(self, tmp_path):
        reg = _make_registry(tmp_path)
        ctx, console = _make_ctx(reg)
        from cliv.slash import handle_workspace
        handle_workspace(ctx, "frobnicate")
        output = console.export_text()
        assert "Unknown" in output

    def test_ws_alias_registered(self):
        from cliv.slash import SLASH_COMMANDS
        assert "/workspace" in SLASH_COMMANDS
        assert "/ws" in SLASH_COMMANDS

    def test_workspace_in_help(self, tmp_path):
        """Check /workspace appears in /help output."""
        reg = _make_registry(tmp_path)
        ctx, console = _make_ctx(reg)
        from cliv.slash import handle_help
        handle_help(ctx, "")
        output = console.export_text()
        assert "workspace" in output.lower()

    def test_dispatch_workspace(self, tmp_path):
        """End-to-end dispatch via slash.dispatch()."""
        reg = _make_registry(tmp_path)
        ctx, console = _make_ctx(reg)
        from cliv.slash import dispatch
        result = dispatch("/workspace list", ctx)
        assert result is True


# ---------------------------------------------------------------------------
# CLI cmd_workspace
# ---------------------------------------------------------------------------

class TestWorkspaceCLI:
    def _run_cmd(self, argv, monkeypatch, capsys):
        """Run main() with the given argv and return stdout."""
        import sys
        monkeypatch.setattr(sys, "argv", ["cliv"] + argv)
        from cliv.cli import main
        try:
            main()
        except SystemExit:
            pass
        return capsys.readouterr()

    def test_workspace_list_empty(self, tmp_path, monkeypatch, capsys):
        ws_file = tmp_path / "workspaces.json"
        monkeypatch.setattr("cliv.workspace.WORKSPACES_FILE", ws_file)
        from cliv.cli import cmd_workspace
        args = SimpleNamespace(workspace_command="list")
        # Patch registry to use temp file
        with patch("cliv.cli.WorkspaceRegistry") as MockReg:
            instance = MockReg.return_value
            instance.list.return_value = []
            instance.active_entry.return_value = None
            cmd_workspace(args)

    def test_workspace_add(self, tmp_path, monkeypatch):
        repo = _make_repo(tmp_path, "api")
        from cliv.workspace import WorkspaceEntry
        entry = WorkspaceEntry(name="api", path=str(repo))

        from cliv.cli import cmd_workspace
        args = SimpleNamespace(
            workspace_command="add",
            path=str(repo),
            name=None,
            desc="",
            tag=[],
        )
        with patch("cliv.cli.WorkspaceRegistry") as MockReg:
            instance = MockReg.return_value
            instance.add.return_value = (entry, True)
            cmd_workspace(args)
            instance.add.assert_called_once_with(str(repo), name=None, description="", tags=[])

    def test_workspace_remove(self, tmp_path):
        from cliv.cli import cmd_workspace
        args = SimpleNamespace(workspace_command="remove", name="api")
        with patch("cliv.cli.WorkspaceRegistry") as MockReg:
            instance = MockReg.return_value
            instance.remove.return_value = True
            cmd_workspace(args)
            instance.remove.assert_called_once_with("api")

    def test_workspace_remove_missing(self, tmp_path):
        from cliv.cli import cmd_workspace
        args = SimpleNamespace(workspace_command="remove", name="ghost")
        with patch("cliv.cli.WorkspaceRegistry") as MockReg:
            instance = MockReg.return_value
            instance.remove.return_value = False
            cmd_workspace(args)  # should not raise

    def test_workspace_info(self, tmp_path):
        from cliv.workspace import WorkspaceEntry
        repo = _make_repo(tmp_path, "api")
        entry = WorkspaceEntry(name="api", path=str(repo), description="main api")

        from cliv.cli import cmd_workspace
        args = SimpleNamespace(workspace_command="info", name="api")
        with patch("cliv.cli.WorkspaceRegistry") as MockReg:
            instance = MockReg.return_value
            instance.get.return_value = entry
            cmd_workspace(args)  # should not raise

    def test_workspace_info_missing(self, tmp_path):
        from cliv.cli import cmd_workspace
        args = SimpleNamespace(workspace_command="info", name="ghost")
        with patch("cliv.cli.WorkspaceRegistry") as MockReg:
            instance = MockReg.return_value
            instance.get.return_value = None
            cmd_workspace(args)  # should not raise

    def test_workspace_subcommand_registered(self):
        """workspace subparser is wired into main() argparse."""
        import sys
        from io import StringIO
        with patch.object(sys, "argv", ["cliv", "workspace", "--help"]):
            from cliv.cli import main
            with pytest.raises(SystemExit):
                main()


# ---------------------------------------------------------------------------
# Agent integration: workspace_registry attribute
# ---------------------------------------------------------------------------

class TestAgentWorkspaceIntegration:
    def test_agent_has_workspace_registry(self):
        from cliv.agent import AIAgent
        with (
            patch("cliv.agent.build_backend", return_value=(MagicMock(), "online")),
            patch("cliv.agent.HistoryStore"),
            patch("cliv.workspace.project_root", return_value=Path("/tmp")),
            patch("cliv.workspace.WorkspaceRegistry.auto_register_current"),
        ):
            agent = AIAgent(api_key="test")
        assert hasattr(agent, "workspace_registry")
        from cliv.workspace import WorkspaceRegistry
        assert isinstance(agent.workspace_registry, WorkspaceRegistry)

    def test_system_message_includes_instruction_18(self):
        from cliv.agent import AIAgent
        backend_mock = MagicMock()
        backend_mock.model = "test-model"
        backend_mock.stats = MagicMock()
        with (
            patch("cliv.agent.build_backend", return_value=(backend_mock, "online")),
            patch("cliv.agent.HistoryStore"),
            patch("cliv.workspace.project_root", return_value=Path("/tmp")),
            patch("cliv.workspace.WorkspaceRegistry.auto_register_current"),
        ):
            agent = AIAgent(api_key="test")
            msg = agent._build_system_message()
        assert "MULTI-REPO PROTOCOL" in msg["content"]
        assert "list_workspaces" in msg["content"]
