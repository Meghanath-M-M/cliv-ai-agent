"""Tests for the AIAgent core logic."""

import json

import pytest
from unittest.mock import patch

from cliv.agent import AIAgent, AgentError, SessionStats, ToolCall
from cliv.history import HistoryStore, trim_messages
from cliv.llm import BackendError, LLMBackend, ToolCallingUnsupported
from cliv.tools.base import BaseTool
from cliv.ui import NullUI


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
    requires_approval = True
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
    }

    def execute(self, path: str, content: str, **kwargs) -> str:
        return f"wrote {path}"


class FakeBackend(LLMBackend):
    name = "fake"
    model = "fake-model"

    def __init__(self, responses=None, generated="generated answer"):
        super().__init__("fake-model", stats=SessionStats())
        self.responses = list(responses or [])
        self.generated = generated
        self.calls = []

    @classmethod
    def is_available(cls) -> bool:
        return True

    def chat(self, messages, tools=None):
        self.calls.append(("chat", messages, tools))
        if not self.responses:
            return ("default response", [])
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def generate(self, messages):
        self.calls.append(("generate", messages))
        return self.generated


def make_agent(tmp_path, backend=None, mode="online", approve=False, **kwargs):
    backend = backend or FakeBackend()
    history_store = kwargs.pop(
        "history_store", HistoryStore(tmp_path / "history.json")
    )
    return AIAgent(
        api_key=None,
        ui=NullUI(approve=approve),
        history_store=history_store,
        backend=backend,
        mode=mode,
        internet_check=lambda: True,
        **kwargs,
    )


# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------
class TestInit:
    def test_online_mode_with_key(self, tmp_path):
        backend = FakeBackend()
        with patch("cliv.agent.build_backend", return_value=(backend, "online")):
            agent = AIAgent(
                api_key="test_key",
                history_store=HistoryStore(tmp_path / "history.json"),
            )
        assert agent.mode == "online"
        assert agent.backend is backend

    def test_offline_mode_no_internet(self, tmp_path):
        backend = FakeBackend()
        with patch("cliv.agent.build_backend", return_value=(backend, "offline")):
            agent = AIAgent(
                api_key="test_key",
                history_store=HistoryStore(tmp_path / "history.json"),
            )
        assert agent.mode == "offline"

    def test_offline_mode_no_key(self, tmp_path):
        backend = FakeBackend()
        with patch("cliv.agent.build_backend", return_value=(backend, "offline")):
            agent = AIAgent(
                api_key=None,
                history_store=HistoryStore(tmp_path / "history.json"),
            )
        assert agent.mode == "offline"

    def test_backend_selection_receives_tools_and_notice_sink(self, tmp_path):
        backend = FakeBackend()
        with patch(
            "cliv.agent.build_backend", return_value=(backend, "online")
        ) as build:
            agent = AIAgent(
                api_key="key",
                history_store=HistoryStore(tmp_path / "history.json"),
                ui=NullUI(),
            )

        kwargs = build.call_args.kwargs
        assert kwargs["tool_schemas"]
        assert kwargs["notices"] is agent.model_notices

    def test_auto_check_mode_tells_user_about_model_switch(self, tmp_path):
        class RecordingUI(NullUI):
            def __init__(self):
                super().__init__()
                self.messages = []

            def print(self, message="", **kwargs):
                self.messages.append(message)

        class FakeOllama:
            name = "ollama"

            @classmethod
            def is_available(cls):
                return True

            def __init__(self, model, *, stats):
                self.model = model
                self.stats = stats

        agent = make_agent(tmp_path, mode="online")
        agent.ui = RecordingUI()
        agent._internet_check = lambda: False

        with patch("cliv.agent.OllamaBackend", FakeOllama):
            agent._auto_check_mode()

        assert agent.mode == "offline"
        assert agent.backend.model == agent.local_model
        assert any("Network lost" in message for message in agent.ui.messages)
        assert any(agent.local_model in message for message in agent.ui.messages)

    def test_auto_approve_flag(self, tmp_path):
        agent = make_agent(tmp_path, auto_approve=True)
        assert agent.auto_approve is True

    def test_dry_run_flag(self, tmp_path):
        agent = make_agent(tmp_path, dry_run=True)
        assert agent.dry_run is True

    def test_injected_backend_sets_mode(self, tmp_path):
        agent = make_agent(tmp_path, mode="offline")
        assert agent.mode == "offline"


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

        store = HistoryStore(hist_path)
        agent = make_agent(tmp_path, backend=FakeBackend(), history_store=store)
        # The agent's history store is the one we injected.
        loaded = store.load()
        assert len(loaded) == 1
        args = loaded[0]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args, dict)
        assert args["x"] == "hello"

    def test_clear_history(self, tmp_path):
        store = HistoryStore(tmp_path / "history.json")
        store.save([{"role": "user", "content": "hi"}])
        assert (tmp_path / "history.json").exists()
        store.clear()
        assert not (tmp_path / "history.json").exists()

    def test_history_capped(self, tmp_path):
        store = HistoryStore(tmp_path / "history.json", max_messages=200)
        messages = [
            {"role": "user", "content": f"msg_{i}"} for i in range(250)
        ]
        store.save(messages)
        loaded = store.load()
        assert len(loaded) == 200

    def test_trim_preserves_tool_exchange(self):
        messages = [
            {"role": "user", "content": "old request"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "1",
                        "type": "function",
                        "function": {"name": "dummy", "arguments": {"x": "a"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "1", "content": "result"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "new request"},
            {"role": "assistant", "content": "new answer"},
        ]
        trimmed = trim_messages(messages, max_tokens=1)
        assert trimmed[0]["role"] == "user"
        assert trimmed[0]["content"] == "new request"
        assert not any(message["role"] == "tool" for message in trimmed)


# ------------------------------------------------------------------
# Tool Execution
# ------------------------------------------------------------------
class TestToolExecution:
    def test_unknown_tool(self, tmp_path):
        agent = make_agent(tmp_path)
        result = agent._execute_tool("nonexistent", {})
        assert "Unknown tool" in result

    def test_write_tool_approved(self, tmp_path):
        agent = make_agent(tmp_path, approve=True)
        agent.tools["write_file"] = WriteTool()
        result = agent._execute_tool(
            "write_file", {"path": "test.txt", "content": "hello"}
        )
        assert "wrote" in result

    def test_write_tool_denied(self, tmp_path):
        agent = make_agent(tmp_path, approve=False)
        agent.tools["write_file"] = WriteTool()
        result = agent._execute_tool(
            "write_file", {"path": "test.txt", "content": "hello"}
        )
        assert "cancelled" in result

    def test_auto_approve_bypasses_prompt(self, tmp_path):
        agent = make_agent(tmp_path, auto_approve=True, approve=False)
        agent.tools["write_file"] = WriteTool()
        result = agent._execute_tool(
            "write_file", {"path": "test.txt", "content": "hello"}
        )
        assert "wrote" in result

    def test_dry_run_returns_preview(self, tmp_path):
        agent = make_agent(tmp_path, dry_run=True)
        agent.tools["write_file"] = WriteTool()
        result = agent._execute_tool(
            "write_file", {"path": "test.txt", "content": "hello"}
        )
        assert "DRY RUN" in result
        assert "Would execute" in result

    def test_malformed_tool_call_non_dict_args(self, tmp_path):
        agent = make_agent(tmp_path)
        result = agent._execute_tool("dummy", "not_a_dict")
        assert "Malformed tool call" in result

    def test_malformed_tool_call_missing_required_args(self, tmp_path):
        agent = make_agent(tmp_path)
        agent.tools["dummy"] = DummyTool()
        result = agent._execute_tool("dummy", {})
        assert "Missing required arguments" in result

    def test_run_tests_requires_approval(self, tmp_path):
        agent = make_agent(tmp_path, approve=False)
        assert agent._needs_approval("run_tests") is True
        result = agent._execute_tool("run_tests", {})
        assert "cancelled" in result


# ------------------------------------------------------------------
# Defensive Shield
# ------------------------------------------------------------------
class TestShield:
    def test_no_leak_returns_content(self, tmp_path):
        agent = make_agent(tmp_path)
        content, tool_call = agent._sanitize_response("This is a normal response.")
        assert content == "This is a normal response."
        assert tool_call is None

    def test_intercepts_known_tool(self, tmp_path):
        agent = make_agent(tmp_path)
        agent.tools["dummy"] = DummyTool()
        content, tool_call = agent._sanitize_response(
            'Here is the result: {"name": "dummy", "arguments": {"x": "test"}}'
        )
        assert content is None
        assert tool_call is not None
        assert tool_call.name == "dummy"
        assert tool_call.arguments == {"x": "test"}

    def test_blocks_unknown_tool(self, tmp_path):
        agent = make_agent(tmp_path)
        content, tool_call = agent._sanitize_response(
            '{"name": "rm_rf", "arguments": {"path": "/"}}'
        )
        assert tool_call is None
        assert "rephrase" in content

    def test_blocks_write_tool_from_shield(self, tmp_path):
        agent = make_agent(tmp_path)
        content, tool_call = agent._sanitize_response(
            '{"name": "edit_file", "arguments": {"path": "x", "new_string": "y"}}'
        )
        assert tool_call is None
        assert "write operations require an explicit tool call" in content

    def test_repairs_bare_identifiers(self, tmp_path):
        agent = make_agent(tmp_path)
        agent.tools["dummy"] = DummyTool()
        content, tool_call = agent._sanitize_response(
            '{"name": dummy, "arguments": {"x": "test"}}'
        )
        assert tool_call is not None
        assert tool_call.name == "dummy"


# ------------------------------------------------------------------
# Chat Loop
# ------------------------------------------------------------------
class TestChatLoop:
    def test_simple_response_no_tools(self, tmp_path):
        backend = FakeBackend([("Hello, user!", [])])
        agent = make_agent(tmp_path, backend=backend)
        result = agent.chat("hi")
        assert result == "Hello, user!"
        assert agent.messages[-1]["role"] == "assistant"

    def test_tool_call_loop(self, tmp_path):
        backend = FakeBackend(
            [
                ("", [ToolCall(id="c1", name="dummy", arguments={"x": "hello"})]),
                ("Done with dummy tool.", []),
            ]
        )
        agent = make_agent(tmp_path, backend=backend)
        agent.tools["dummy"] = DummyTool()
        result = agent.chat("use dummy")
        assert result == "Done with dummy tool."
        assert len(agent.messages) >= 4

    def test_max_iterations_guard(self, tmp_path):
        backend = FakeBackend(
            [
                ("", [ToolCall(id="c1", name="dummy", arguments={"x": "hello"})])
                for _ in range(20)
            ]
        )
        agent = make_agent(tmp_path, backend=backend)
        agent.tools["dummy"] = DummyTool()
        result = agent.chat("loop me")
        assert "maximum number of tool calls" in result

    def test_shield_intercept_triggers_loop(self, tmp_path):
        backend = FakeBackend(
            [
                ('{"name": "dummy", "arguments": {"x": "shield"}}', []),
                ("Shield result summarized.", []),
            ]
        )
        agent = make_agent(tmp_path, backend=backend)
        agent.tools["dummy"] = DummyTool()
        result = agent.chat("trigger shield")
        assert result == "Shield result summarized."

    def test_online_backend_error_is_reported(self, tmp_path):
        backend = FakeBackend([BackendError("boom")])
        agent = make_agent(tmp_path, backend=backend)
        result = agent.chat("hi")
        assert "Error [ONLINE Mode]" in result
        assert "boom" in result

    def test_offline_falls_back_when_tools_unsupported(self, tmp_path):
        backend = FakeBackend(
            [ToolCallingUnsupported("no tools")], generated="offline answer"
        )
        agent = make_agent(
            tmp_path, backend=backend, mode="offline", use_offline_tools=True
        )
        result = agent.chat("hello")
        assert result == "offline answer"

    def test_no_backend_returns_error_without_writing_files(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        with patch("cliv.agent.build_backend", return_value=(None, "offline")):
            agent = AIAgent(
                history_store=HistoryStore(tmp_path / "history.json"),
                ui=NullUI(approve=True),
            )

        result = agent.chat("create a file called demo.py that prints hello")

        assert "No LLM backend" in result
        assert not (tmp_path / "demo.py").exists()


# ------------------------------------------------------------------
# Cost Tracking
# ------------------------------------------------------------------
class TestCostTracking:
    def test_session_stats(self):
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
    def test_valid_python(self, tmp_path):
        agent = make_agent(tmp_path)
        result = agent._check_syntax_python("def foo():\n    return 1")
        assert result is None

    def test_invalid_python(self, tmp_path):
        agent = make_agent(tmp_path)
        result = agent._check_syntax_python("def foo():\n    returnbmm 1")
        assert "Syntax error" in result
        assert "returnbmm" in result
