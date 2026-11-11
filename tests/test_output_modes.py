"""Tests for Rank 9: Structured Output Modes (--json, --quiet, /json, subcommands)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from cliv.agent import AIAgent
from cliv.output import format_agent_run_json, print_json
from cliv.slash import SlashContext, handle_json
from cliv.cli import (
    process_prompt,
    build_agent,
    cmd_cost,
    cmd_models,
    cmd_dna,
    cmd_workspace,
    cmd_memory,
    cmd_tools,
    cmd_timeline,
    cmd_snapshots,
)


# ============================================================================
# Unit Tests for cliv.output
# ============================================================================

class TestOutputModule:
    """Test JSON formatting and printing helpers."""

    def test_print_json_outputs_valid_json(self, capsys):
        data = {"hello": "world", "count": 42, "items": [1, 2, 3]}
        print_json(data)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed == data

    def test_format_agent_run_json_success(self):
        agent = MagicMock()
        agent.mode = "online"
        agent.backend = MagicMock(model="openai/gpt-oss-120b")
        agent.stats = MagicMock(
            input_tokens=150,
            output_tokens=50,
            total_tokens=200,
            cost_usd=0.00012,
        )

        out = format_agent_run_json(agent, "list files", "Here are the files: ...")
        assert out["status"] == "success"
        assert out["prompt"] == "list files"
        assert out["response"] == "Here are the files: ..."
        assert out["model"] == "openai/gpt-oss-120b"
        assert out["mode"] == "online"
        assert out["usage"]["prompt_tokens"] == 150
        assert out["usage"]["completion_tokens"] == 50
        assert out["usage"]["total_tokens"] == 200
        assert out["usage"]["cost_usd"] == 0.00012
        assert out["error"] is None

    def test_format_agent_run_json_error(self):
        agent = MagicMock()
        agent.mode = "offline"
        agent.backend = MagicMock(model="llama3.1:8b")
        agent.stats = MagicMock(input_tokens=0, output_tokens=0, total_tokens=0, cost_usd=0.0)

        out = format_agent_run_json(agent, "failing prompt", "", error="Connection refused")
        assert out["status"] == "error"
        assert out["prompt"] == "failing prompt"
        assert out["response"] == ""
        assert out["error"] == "Connection refused"


# ============================================================================
# Unit Tests for process_prompt
# ============================================================================

class TestProcessPromptOutputModes:
    """Test process_prompt with --json and --quiet."""

    def test_process_prompt_json_success(self, capsys):
        agent = MagicMock()
        agent.chat.return_value = "Everything looks clean."
        agent.mode = "online"
        agent.backend = MagicMock(model="test-model")
        agent.stats = MagicMock(input_tokens=10, output_tokens=20, total_tokens=30, cost_usd=0.001)

        res = process_prompt(agent, "check code", is_json=True)
        assert res == "Everything looks clean."

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "success"
        assert data["prompt"] == "check code"
        assert data["response"] == "Everything looks clean."
        assert data["model"] == "test-model"

    def test_process_prompt_json_catches_error(self, capsys):
        agent = MagicMock()
        agent.chat.side_effect = RuntimeError("LLM API 500 internal error")
        agent.mode = "online"
        agent.backend = MagicMock(model="test-model")
        agent.stats = MagicMock(input_tokens=0, output_tokens=0, total_tokens=0, cost_usd=0.0)

        res = process_prompt(agent, "fail prompt", is_json=True)
        assert res is None

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "error"
        assert data["error"] == "LLM API 500 internal error"
        assert data["response"] == ""

    def test_process_prompt_quiet(self, capsys):
        agent = MagicMock()
        agent.chat.return_value = "Clean response line"

        res = process_prompt(agent, "say something", is_quiet=True)
        assert res == "Clean response line"

        captured = capsys.readouterr()
        # In quiet mode, should ONLY print the response without Rich banners, cliv: or cost dashboard
        assert captured.out.strip() == "Clean response line"


# ============================================================================
# Tests for Subcommand --json output
# ============================================================================

class TestSubcommandsJsonOutput:
    """Verify that CLI subcommands output valid JSON with --json flag."""

    def test_cmd_cost_json(self, capsys):
        args = MagicMock()
        args.json = True

        mock_cost = {
            "project_root": "C:/fake/repo",
            "total_sessions": 4,
            "input_tokens": 1000,
            "output_tokens": 500,
            "total_tokens": 1500,
            "total_cost_usd": 0.0015,
            "models": [],
        }

        with patch("cliv.timeline.TimelineStore") as mock_ts:
            mock_ts.return_value.get_project_cost.return_value = mock_cost
            cmd_cost(args)

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["total_tokens"] == 1500
        assert parsed["total_cost_usd"] == 0.0015

    def test_cmd_models_json(self, capsys):
        args = MagicMock()
        args.json = True

        mock_health = {
            "active_mode": "online",
            "active_model": "test-online-model",
            "ollama": {
                "running": True,
                "models": [{"name": "llama3.1:8b", "size_gb": 4.58, "has_tools": True}],
                "active": False,
            },
            "groq": {
                "has_key": True,
                "error": None,
                "models": [{"id": "llama-3.3-70b-versatile", "usable_for_chat": True}],
                "active": True,
            },
        }

        with patch("cliv.llm.discovery.check_model_health", return_value=mock_health):
            cmd_models(args)

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["active_mode"] == "online"
        assert parsed["ollama"]["running"] is True
        assert len(parsed["ollama"]["models"]) == 1

    def test_cmd_dna_json(self, capsys):
        args = MagicMock()
        args.json = True
        args.refresh = False

        mock_intel = MagicMock()
        mock_intel.to_dict.return_value = {
            "project_name": "cliv-ai-agent",
            "root": "C:/fake/repo",
            "languages": {"Python": 12000},
            "source_files": 45,
            "symbol_count": 300,
            "class_count": 25,
            "function_count": 120,
            "method_count": 80,
            "entry_points": ["cliv/cli.py"],
            "external_packages": ["rich", "requests"],
            "components": [],
            "issues": [],
        }

        with patch("cliv.intelligence.report.build_project_intelligence", return_value=mock_intel):
            cmd_dna(args)

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["project_name"] == "cliv-ai-agent"
        assert parsed["languages"]["Python"] == 12000
        assert parsed["source_files"] == 45
        assert "cliv/cli.py" in parsed["entry_points"]

    def test_cmd_workspace_list_json(self, capsys):
        args = MagicMock()
        args.workspace_command = "list"
        args.json = True

        mock_reg = MagicMock()
        entry1 = MagicMock()
        entry1.to_dict.return_value = {
            "name": "web",
            "path": "C:/fake/web",
            "description": "Frontend",
            "tags": ["ui"],
            "added_at": "2026-01-01",
            "last_active": "2026-01-02",
            "exists": True,
        }

        mock_reg.list.return_value = [entry1]

        with patch("cliv.cli.WorkspaceRegistry", return_value=mock_reg):
            cmd_workspace(args)

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert parsed[0]["name"] == "web"
        assert parsed[0]["exists"] is True

    def test_cmd_memory_list_json(self, capsys):
        args = MagicMock()
        args.memory_command = "list"
        args.json = True
        args.scope = None

        mock_entry = MagicMock()
        mock_entry.to_dict.return_value = {
            "id": "mem_1",
            "scope": "project",
            "category": "testing",
            "content": "use pytest",
            "created_at": "2026-01-01",
        }

        mock_store = MagicMock()
        mock_store.list_entries.return_value = [mock_entry]

        with patch("cliv.memory.MemoryStore", return_value=mock_store):
            cmd_memory(args)

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert parsed[0]["id"] == "mem_1"
        assert parsed[0]["content"] == "use pytest"

    def test_cmd_timeline_json(self, capsys):
        args = MagicMock()
        args.limit = 5
        args.json = True

        mock_event = MagicMock()
        mock_event.id = 1
        mock_event.timestamp = "2026-01-01T00:00:00Z"
        mock_event.event_type = "tool_call"
        mock_event.summary = "read_file"
        mock_event.metadata = {}

        mock_store = MagicMock()
        mock_store.recent.return_value = [mock_event]

        with patch("cliv.timeline.TimelineStore", return_value=mock_store):
            cmd_timeline(args)

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert parsed[0]["event_type"] == "tool_call"
        assert parsed[0]["summary"] == "read_file"

    def test_cmd_snapshots_list_json(self, capsys):
        args = MagicMock()
        args.snapshots_command = "list"
        args.json = True
        args.limit = 10

        mock_snap = MagicMock()
        mock_snap.id = "snap-123"
        mock_snap.timestamp = "2026-01-01T12:00:00Z"
        file_entry = MagicMock(path="main.py")
        mock_snap.files = [file_entry]

        mock_store = MagicMock()
        mock_store.recent.return_value = [mock_snap]

        with patch("cliv.snapshots.SnapshotStore", return_value=mock_store):
            cmd_snapshots(args)

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert parsed[0]["id"] == "snap-123"
        assert parsed[0]["files"] == ["main.py"]


# ============================================================================
# Tests for /json Slash Command
# ============================================================================

class TestJsonSlashCommand:
    """Test interactive toggle for JSON output mode."""

    def test_handle_json_toggle(self):
        agent = MagicMock()
        agent.json_mode = False
        console = MagicMock()
        ctx = SlashContext(agent=agent, console=console)

        # Toggle on
        ret = handle_json(ctx, "")
        assert ret is True
        assert agent.json_mode is True

        # Toggle off
        ret = handle_json(ctx, "")
        assert ret is True
        assert agent.json_mode is False

    def test_handle_json_explicit_on_off(self):
        agent = MagicMock()
        agent.json_mode = False
        console = MagicMock()
        ctx = SlashContext(agent=agent, console=console)

        handle_json(ctx, "on")
        assert agent.json_mode is True

        handle_json(ctx, "off")
        assert agent.json_mode is False

        handle_json(ctx, "enable")
        assert agent.json_mode is True

        handle_json(ctx, "disable")
        assert agent.json_mode is False

    def test_handle_json_invalid_argument(self):
        agent = MagicMock()
        agent.json_mode = False
        console = MagicMock()
        ctx = SlashContext(agent=agent, console=console)

        ret = handle_json(ctx, "maybe")
        assert ret is True
        # json_mode should remain unchanged
        assert agent.json_mode is False
        console.print.assert_called_once()
        assert "Usage" in console.print.call_args[0][0]
