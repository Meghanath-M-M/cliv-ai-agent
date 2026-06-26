"""Tests for the AIAgent core logic."""

import json
import pytest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path

from cliv.agent import AIAgent, AgentError, ToolCall
from cliv.tools.base import BaseTool


class DummyTool(BaseTool):
    name = "dummy"
    description = "A dummy tool for testing"
    input_schema = {
        "type": "object",
        "properties": {"x": {"type": "string"}},
        "required": ["x"],
    }

    def execute(self, x: str, **kwargs) -> str:
        return f"dummy_result: {x}"


class WriteTool(BaseTool):
    name = "write_file"
    description = "Writes a file"
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
    }

    def execute(self, path: str, content: str, **kwargs) -> str:
        return f"wrote {path}"


# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------
class TestInit:
    @patch("cliv.agent.Groq")
    @patch("cliv.agent.AIAgent._check_internet", return_value=True)
    def test_online_mode_with_key(self, mock_net, mock_groq):
        agent = AIAgent(api_key="test_key")
        assert agent.mode == "online"
        assert agent.model_name == "meta-llama/llama-4-scout-17b-16e-instruct"

    @patch("cliv.agent.AIAgent._check_internet", return_value=False)
    def test_offline_mode_no_internet(self, mock_net):
        agent = AIAgent(api_key="test_key")
        assert agent.mode == "offline"

    def test_offline_mode_no_key(self):
        agent = AIAgent(api_key=None)
        assert agent.mode == "offline"

    def test_auto_approve_flag(self):
        agent = AIAgent(api_key=None, auto_approve=True)
        assert agent.auto_approve is True

    def test_dry_run_flag(self):
        agent = AIAgent(api_key=None, dry_run=True)
        assert agent.dry_run is True


# ------------------------------------------------------------------
# History
# ------------------------------------------------------------------
class TestHistory:
    def test_load_history_normalizes_args(self, tmp_path):
        hist_path = tmp_path / "history.json"
        hist_path.write_text(
            json.dumps(
                [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "1",
                                "type": "function",
                                "function": {
                                    "name": "dummy",
                                    "arguments": '{"x": "hello"}',
                                },
                            }
                        ],
                    }
                ]
            )
        )

        # Patch the module-level HISTORY_FILE constant with the real tmp_path
        # Path object itself. The production code calls the builtin
        # open(HISTORY_FILE, ...) and HISTORY_FILE.exists() -- both work
        # natively on a real Path, so there's no need to hand-fake a mock
        # with .open()/.exists() attributes. The previous version of this
        # test stubbed mock_hist_file.open(...) as a *method*, but the real
        # code calls the global open(HISTORY_FILE, ...) instead -- that
        # mismatch meant the mock was never actually exercised, and the
        # agent ended up reading empty/garbage content instead of the
        # JSON this test wrote to tmp_path.
        with patch("cliv.agent.HISTORY_FILE", hist_path):
            agent = AIAgent(api_key=None)
            assert len(agent.messages) == 1
            args = agent.messages[0]["tool_calls"][0]["function"]["arguments"]
            assert isinstance(args, dict)
            assert args["x"] == "hello"

    def test_clear_history(self, tmp_path):
        with patch("cliv.agent.HISTORY_FILE", tmp_path / "history.json"):
            agent = AIAgent(api_key=None)
            agent.messages = [{"role": "user", "content": "hi"}]
            agent._save_history()
            assert (tmp_path / "history.json").exists()
            agent.clear_history()
            assert agent.messages == []
            assert not (tmp_path / "history.json").exists()

    def test_history_capped(self, tmp_path):
        with patch("cliv.agent.HISTORY_FILE", tmp_path / "history.json"):
            agent = AIAgent(api_key=None)
            agent.messages = [
                {"role": "user", "content": f"msg_{i}"} for i in range(60)
            ]
            agent._save_history()
            loaded = agent._load_history()
            assert len(loaded) == AIAgent.MAX_HISTORY_MESSAGES


# ------------------------------------------------------------------
# Tool Execution
# ------------------------------------------------------------------
class TestToolExecution:
    def test_unknown_tool(self):
        agent = AIAgent(api_key=None)
        result = agent._execute_tool("nonexistent", {})
        assert "Unknown tool" in result

    @patch("builtins.input", return_value="y")
    def test_write_tool_approved(self, mock_input):
        agent = AIAgent(api_key=None)
        agent.tools["write_file"] = WriteTool()
        result = agent._execute_tool(
            "write_file", {"path": "test.txt", "content": "hello"}
        )
        assert "wrote" in result

    @patch("builtins.input", return_value="n")
    def test_write_tool_denied(self, mock_input):
        agent = AIAgent(api_key=None)
        agent.tools["write_file"] = WriteTool()
        result = agent._execute_tool(
            "write_file", {"path": "test.txt", "content": "hello"}
        )
        assert "cancelled" in result

    def test_auto_approve_bypasses_prompt(self):
        agent = AIAgent(api_key=None, auto_approve=True)
        agent.tools["write_file"] = WriteTool()
        result = agent._execute_tool(
            "write_file", {"path": "test.txt", "content": "hello"}
        )
        assert "wrote" in result

    def test_dry_run_returns_preview(self):
        agent = AIAgent(api_key=None, dry_run=True)
        agent.tools["write_file"] = WriteTool()
        result = agent._execute_tool(
            "write_file", {"path": "test.txt", "content": "hello"}
        )
        assert "DRY RUN" in result
        assert "Would execute" in result

    def test_malformed_tool_call_non_dict_args(self):
        agent = AIAgent(api_key=None)
        result = agent._execute_tool("dummy", "not_a_dict")
        assert "Malformed tool call" in result

    def test_malformed_tool_call_missing_required_args(self):
        agent = AIAgent(api_key=None)
        agent.tools["dummy"] = DummyTool()
        result = agent._execute_tool("dummy", {})
        assert "Missing required arguments" in result


# ------------------------------------------------------------------
# Defensive Shield
# ------------------------------------------------------------------
class TestShield:
    def test_no_leak_returns_content(self):
        agent = AIAgent(api_key=None)
        content, tc = agent._sanitize_response("This is a normal response.")
        assert content == "This is a normal response."
        assert tc is None

    def test_intercepts_known_tool(self):
        agent = AIAgent(api_key=None)
        agent.tools["dummy"] = DummyTool()
        content, tc = agent._sanitize_response(
            'Here is the result: {"name": "dummy", "arguments": {"x": "test"}}'
        )
        assert content is None
        assert tc is not None
        assert tc.name == "dummy"
        assert tc.arguments == {"x": "test"}

    def test_blocks_unknown_tool(self):
        agent = AIAgent(api_key=None)
        content, tc = agent._sanitize_response(
            '{"name": "rm_rf", "arguments": {"path": "/"}}'
        )
        assert tc is None
        assert "rephrase" in content

    def test_blocks_write_tool_from_shield(self):
        agent = AIAgent(api_key=None)
        agent.tools["write_file"] = WriteTool()
        content, tc = agent._sanitize_response(
            '{"name": "write_file", "arguments": {"path": "x", "content": "y"}}'
        )
        assert tc is None
        assert "write operations require an explicit tool call" in content

    def test_repairs_bare_identifiers(self):
        agent = AIAgent(api_key=None)
        agent.tools["dummy"] = DummyTool()
        content, tc = agent._sanitize_response(
            '{"name": dummy, "arguments": {"x": "test"}}'
        )
        assert tc is not None
        assert tc.name == "dummy"


# ------------------------------------------------------------------
# Chat Loop
# ------------------------------------------------------------------
class TestChatLoop:
    @patch("cliv.agent.AIAgent._call_llm")
    def test_simple_response_no_tools(self, mock_call):
        agent = AIAgent(api_key=None)
        mock_call.return_value = ("Hello, user!", [])
        result = agent.chat("hi")
        assert result == "Hello, user!"
        assert agent.messages[-1]["role"] == "assistant"

    @patch("cliv.agent.AIAgent._call_llm")
    def test_tool_call_loop(self, mock_call):
        agent = AIAgent(api_key=None)
        agent.tools["dummy"] = DummyTool()
        mock_call.side_effect = [
            ("", [ToolCall(id="c1", name="dummy", arguments={"x": "hello"})]),
            ("Done with dummy tool.", []),
        ]
        result = agent.chat("use dummy")
        assert result == "Done with dummy tool."
        assert len(agent.messages) >= 4

    @patch("cliv.agent.AIAgent._call_llm")
    def test_max_iterations_guard(self, mock_call):
        agent = AIAgent(api_key=None)
        agent.tools["dummy"] = DummyTool()
        mock_call.return_value = (
            "",
            [ToolCall(id="c1", name="dummy", arguments={"x": "hello"})],
        )
        result = agent.chat("loop me")
        assert "maximum number of tool calls" in result

    @patch("cliv.agent.AIAgent._call_llm")
    def test_shield_intercept_triggers_loop(self, mock_call):
        agent = AIAgent(api_key=None)
        agent.tools["dummy"] = DummyTool()
        mock_call.side_effect = [
            ('{"name": "dummy", "arguments": {"x": "shield"}}', []),
            ("Shield result summarized.", []),
        ]
        result = agent.chat("trigger shield")
        assert result == "Shield result summarized."


# ------------------------------------------------------------------
# Cost Tracking
# ------------------------------------------------------------------
class TestCostTracking:
    def test_session_stats(self):
        from cliv.agent import SessionStats

        stats = SessionStats()
        stats.add_usage(1000, 500, "meta-llama/llama-4-scout-17b-16e-instruct")
        assert stats.total_tokens == 1500
        assert stats.input_tokens == 1000
        assert stats.output_tokens == 500
        assert abs(stats.cost_usd - 0.00028) < 1e-9


# ------------------------------------------------------------------
# Syntax Checking
# ------------------------------------------------------------------
class TestSyntaxChecking:
    def test_valid_python(self):
        agent = AIAgent(api_key=None)
        result = agent._check_syntax_python("def foo():\n    return 1")
        assert result is None

    def test_invalid_python(self):
        agent = AIAgent(api_key=None)
        result = agent._check_syntax_python("def foo():\n    returnbmm 1")
        assert "Syntax error" in result
        assert "returnbmm" in result
