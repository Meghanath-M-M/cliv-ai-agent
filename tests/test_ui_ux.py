"""Tests for UI/UX Enhancements: Interactive Model Selector, Live Thinking Animations, and REPL Improvements."""

from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from cliv.slash import SlashContext, handle_model
from cliv.tui import ModelSelector, get_key
from cliv.cli import process_prompt


# ============================================================================
# Unit Tests for ModelSelector TUI
# ============================================================================

class TestModelSelector:
    """Verify ModelSelector TUI data building, rendering, and navigation."""

    @pytest.fixture
    def mock_health(self):
        return {
            "active_mode": "online",
            "active_model": "openai/gpt-oss-120b",
            "ollama": {
                "running": True,
                "models": [
                    {
                        "name": "llama3.1:8b",
                        "size_gb": 4.58,
                        "parameter_size": "8.0B",
                        "supports_tools": True,
                        "desc": "Meta LLaMA 3.1 8B",
                    },
                    {
                        "name": "deepseek-coder:6.7b",
                        "size_gb": 3.56,
                        "parameter_size": "7B",
                        "supports_tools": False,
                        "desc": "DeepSeek Coder",
                    },
                ],
            },
            "groq": {
                "has_key": True,
                "models": [
                    {
                        "id": "openai/gpt-oss-120b",
                        "usable_for_chat": True,
                        "desc": "OpenAI GPT OSS 120B on Groq",
                        "owned_by": "OpenAI",
                        "category": "Conversational Coding LLM",
                    },
                    {
                        "id": "qwen/qwen3.8-27b",
                        "usable_for_chat": True,
                        "desc": "Alibaba Qwen 3.8 27B",
                        "owned_by": "Alibaba",
                        "category": "Conversational Coding LLM",
                    },
                    {
                        "id": "whisper-large-v3",
                        "usable_for_chat": False,
                        "desc": "Audio Speech Recognition",
                        "owned_by": "OpenAI",
                        "category": "Audio/Speech",
                    },
                ],
            },
        }

    def test_model_list_building(self, mock_health):
        selector = ModelSelector(
            health_data=mock_health,
            current_model="openai/gpt-oss-120b",
            current_mode="online",
        )
        assert len(selector.models) == 5
        # Active model should be selected by default
        assert selector.models[selector.selected_index]["id"] == "openai/gpt-oss-120b"
        assert selector.models[selector.selected_index]["is_active"] is True

    def test_render_table_and_details(self, mock_health):
        selector = ModelSelector(
            health_data=mock_health,
            current_model="openai/gpt-oss-120b",
            current_mode="online",
        )
        table = selector._render_table()
        assert table.row_count == 5

        current = selector.models[selector.selected_index]
        details = selector._render_details(current)
        assert "openai/gpt-oss-120b" in str(details.renderable)
        assert "Direct tool calling" in str(details.renderable)

    def test_keyboard_navigation(self, mock_health):
        selector = ModelSelector(
            health_data=mock_health,
            current_model="llama3.1:8b",
            current_mode="offline",
        )
        assert selector.selected_index == 0

        # Simulate Down key ('j')
        selector.selected_index = min(selector.selected_index + 1, len(selector.models) - 1)
        assert selector.selected_index == 1

        # Simulate Down key ('j') again
        selector.selected_index = min(selector.selected_index + 1, len(selector.models) - 1)
        assert selector.selected_index == 2

        # Simulate Up key ('k')
        selector.selected_index = max(selector.selected_index - 1, 0)
        assert selector.selected_index == 1

    def test_run_interactive_selection(self, mock_health):
        selector = ModelSelector(
            health_data=mock_health,
            current_model="llama3.1:8b",
            current_mode="offline",
        )
        # Mock sys.stdin.isatty and get_key sequence: 'j' (down), '\n' (enter)
        key_sequence = ["j", "\n"]

        with patch("sys.stdin.isatty", return_value=True), \
             patch("cliv.tui.get_key", side_effect=key_sequence), \
             patch("rich.live.Live.__enter__"), \
             patch("rich.live.Live.__exit__"):
            chosen = selector.run()
            assert chosen is not None
            assert chosen["id"] == "deepseek-coder:6.7b"

    def test_run_interactive_cancel(self, mock_health):
        selector = ModelSelector(
            health_data=mock_health,
            current_model="llama3.1:8b",
            current_mode="offline",
        )
        with patch("sys.stdin.isatty", return_value=True), \
             patch("cliv.tui.get_key", return_value="q"), \
             patch("rich.live.Live.__enter__"), \
             patch("rich.live.Live.__exit__"):
            chosen = selector.run()
            assert chosen is None

    def test_run_non_interactive_fallback(self, mock_health):
        selector = ModelSelector(
            health_data=mock_health,
            current_model="llama3.1:8b",
            current_mode="offline",
        )
        with patch("sys.stdin.isatty", return_value=False):
            chosen = selector.run()
            assert chosen is None


# ============================================================================
# Unit Tests for /model with TUI Selector
# ============================================================================

class TestHandleModelTUI:
    """Verify handle_model invokes ModelSelector when interactive and switches model on Enter."""

    def test_handle_model_interactive_selection(self):
        agent = MagicMock()
        agent.mode = "offline"
        agent.backend = MagicMock(model="llama3.1:8b")
        agent.local_model = "llama3.1:8b"
        agent.stats = MagicMock()
        console = MagicMock()
        ctx = SlashContext(agent=agent, console=console)

        mock_selected = {
            "id": "qwen2.5-coder:7b",
            "provider": "Ollama",
            "provider_type": "local",
        }

        mock_health = {
            "ollama": {
                "running": True,
                "models": [{"name": "qwen2.5-coder:7b", "supports_tools": True}],
            },
            "groq": {"has_key": False, "models": []},
        }

        with patch("sys.stdin.isatty", return_value=True), \
             patch("cliv.slash.check_model_health", return_value=mock_health), \
             patch("cliv.tui.ModelSelector.run", return_value=mock_selected), \
             patch("cliv.llm.OllamaBackend.is_available", return_value=True), \
             patch("cliv.llm.OllamaBackend") as mock_ob:
            ret = handle_model(ctx, "")
            assert ret is True
            # Should have switched to qwen2.5-coder:7b
            assert agent.local_model == "qwen2.5-coder:7b"
            assert agent.mode == "offline"

    def test_handle_model_interactive_cancel(self):
        agent = MagicMock()
        agent.mode = "offline"
        agent.backend = MagicMock(model="llama3.1:8b")
        agent.local_model = "llama3.1:8b"
        console = MagicMock()
        ctx = SlashContext(agent=agent, console=console)

        mock_health = {
            "ollama": {"running": True, "models": [{"name": "llama3.1:8b"}]},
            "groq": {"has_key": False, "models": []},
        }

        with patch("sys.stdin.isatty", return_value=True), \
             patch("cliv.slash.check_model_health", return_value=mock_health), \
             patch("cliv.tui.ModelSelector.run", return_value=None):
            ret = handle_model(ctx, "")
            assert ret is True
            # Model should remain unchanged
            assert agent.local_model == "llama3.1:8b"
            console.print.assert_called()
            assert "cancelled" in str(console.print.call_args[0][0])


# ============================================================================
# Unit Tests for Live Thinking & Tool Animation
# ============================================================================

class TestLiveRunningAnimation:
    """Verify live status spinners during thinking, tool running, and offline processing."""

    def test_agent_tool_loop_shows_thinking_and_executing_status(self):
        from cliv.agent import AIAgent
        from cliv.models import ToolCall

        ui = MagicMock()
        agent = AIAgent(api_key="test-key", ui=ui)
        agent.mode = "online"
        agent.backend = MagicMock()
        agent.backend.model = "test-model"

        # Mock tool call then content reply
        tc = ToolCall(id="tc-1", name="list_files", arguments={"path": "."})
        agent.backend.chat.side_effect = [
            ("I will list the files.", [tc]),
            ("Here are the files: main.py", []),
        ]

        with patch.object(agent, "_execute_tool", return_value="main.py"):
            resp = agent.chat("show files")
            assert "Here are the files: main.py" in resp

        # Verify ui.status was called for thinking
        assert ui.status.called
        status_calls = [c[0][0] for c in ui.status.call_args_list]
        assert any("Thinking" in s for s in status_calls)

        # Verify ui.print logged the execution pill
        assert ui.print.called
        print_calls = [c[0][0] for c in ui.print.call_args_list]
        assert any("⚡ Tool" in p and "list_files" in p for p in print_calls)

    def test_console_ui_confirm_stops_active_status_and_prompts(self):
        from cliv.ui import ConsoleUI

        mock_console = MagicMock()
        ui = ConsoleUI(console=mock_console)

        # Enter a status
        with ui.status("Running something..."):
            assert ui._current_status is not None
            mock_status = ui._current_status

            # Call confirm while status was active
            with patch("builtins.input", return_value="y"):
                approved = ui.confirm("edit_file", {"path": "hi.py", "new_string": "print('hi')"})
                assert approved is True
                # Status should have been stopped before prompt
                mock_status.stop.assert_called_once()
                # Prompt must be printed with newline so it doesn't collide with table
                calls = [str(c[0][0]) for c in mock_console.print.call_args_list]
                assert any("Proceed?" in c and "y/N" in c for c in calls)

    def test_process_prompt_modern_header(self, capsys):
        agent = MagicMock()
        agent.mode = "online"
        agent.backend = MagicMock(model="openai/gpt-oss-120b")
        agent.chat.return_value = "Hello from cliv!"
        agent.stats = MagicMock(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            cost_usd=0.00005,
        )

        with patch("cliv.cli.console.print") as mock_print:
            process_prompt(agent, "hi")
            assert mock_print.called
            # Verify modern cliv header with model name and ONLINE tag was rendered
            calls_str = " ".join(str(c[0]) for c in mock_print.call_args_list)
            assert "◆ cliv" in calls_str
            assert "openai/gpt-oss-120b" in calls_str
            assert "ONLINE" in calls_str

    def test_edit_file_approval_and_execution_end_to_end(self, tmp_path):
        from cliv.agent import AIAgent
        from cliv.ui import ConsoleUI
        from cliv.models import ToolCall

        target_file = tmp_path / "hi.py"
        assert not target_file.exists()

        console = MagicMock(width=80)
        ui = ConsoleUI(console=console)
        agent = AIAgent(api_key="test-key", ui=ui)
        agent.mode = "online"
        agent.backend = MagicMock(model="qwen/qwen3.8-27b")

        tc = ToolCall(
            id="tc-edit-1",
            name="edit_file",
            arguments={"path": str(target_file), "new_string": "print('hi')"},
        )
        agent.backend.chat.side_effect = [
            ("I will create hi.py", [tc]),
            ("Done! File created.", []),
        ]

        with patch("cliv.security.project_root", return_value=tmp_path), \
             patch("builtins.input", return_value="y"):
            resp = agent.chat("create hi.py")
            assert "Done! File created." in resp

        # Verify that the file was created and contains the expected content
        assert target_file.exists()
        assert target_file.read_text(encoding="utf-8") == "print('hi')"
