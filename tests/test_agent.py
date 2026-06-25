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
    input_schema = {"type": "object", "properties": {"x": {"type": "string"}}}

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


# ------------------------------------------------------------------
# History
# ------------------------------------------------------------------
class TestHistory:
    @patch("cliv.agent.HISTORY_FILE")
    def test_load_history_normalizes_args(self, mock_hist_file, tmp_path):
        mock_hist_file.__str__ = lambda self: str(tmp_path / "history.json")
        mock_hist_file.exists = lambda self: True
        mock_hist_file.open = lambda *a, **k: mock_open(
            read_data=json.dumps(
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
        )()

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
