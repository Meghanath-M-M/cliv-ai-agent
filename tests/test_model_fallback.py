"""Tests for Rank 8: Improved offline model fallback UX.

Covers:
- cliv.llm.discovery (is_ollama_running, list_ollama_models, list_groq_models, check_model_health)
- cliv models CLI subcommand (rich model dashboard, daemon-down guidance)
- /model slash command (list, status, switch with tool capability report, daemon-down handling)
- AIAgent graceful degradation (online to offline mid-session fallback, intent routing notices)
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from cliv.llm.base import BackendError, ToolCallingUnsupported
from cliv.llm.discovery import (
    check_model_health,
    is_ollama_running,
    list_groq_models,
    list_ollama_models,
)
from cliv.models import SessionStats


# ---------------------------------------------------------------------------
# Discovery tests
# ---------------------------------------------------------------------------

class TestModelDiscovery:
    def test_is_ollama_running_false_on_connection_error(self):
        with patch("socket.create_connection", side_effect=OSError("refused")):
            assert is_ollama_running() is False

    def test_is_ollama_running_true_when_connected(self):
        mock_sock = MagicMock()
        mock_sock.__enter__.return_value = mock_sock
        with patch("socket.create_connection", return_value=mock_sock):
            assert is_ollama_running() is True

    def test_list_ollama_models_not_running(self):
        with patch("cliv.llm.discovery.is_ollama_running", return_value=False):
            models = list_ollama_models()
            assert models == []

    def test_list_ollama_models_parses_response(self):
        fake_payload = {
            "models": [
                {
                    "name": "llama3.1:8b",
                    "size": 4920753328,
                    "details": {
                        "parameter_size": "8.0B",
                        "quantization_level": "Q4_K_M",
                        "family": "llama",
                    },
                    "capabilities": ["completion", "tools"],
                    "modified_at": "2026-06-26T13:26:18Z",
                },
                {
                    "name": "deepseek-coder:6.7b",
                    "size": 3827834503,
                    "details": {
                        "parameter_size": "7B",
                        "quantization_level": "Q4_0",
                        "family": "llama",
                    },
                    "capabilities": ["completion"],
                    "modified_at": "2026-06-25T22:50:44Z",
                },
            ]
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(fake_payload).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with (
            patch("cliv.llm.discovery.is_ollama_running", return_value=True),
            patch("urllib.request.urlopen", return_value=mock_resp),
        ):
            models = list_ollama_models()

        assert len(models) == 2
        assert models[0]["name"] == "llama3.1:8b"
        assert models[0]["supports_tools"] is True
        assert models[0]["parameter_size"] == "8.0B"
        assert models[0]["size_gb"] > 4.0
        assert models[1]["name"] == "deepseek-coder:6.7b"
        assert models[1]["supports_tools"] is False

    def test_list_groq_models_no_key_returns_empty(self):
        with patch("cliv.llm.discovery.get_api_key", return_value=None):
            assert list_groq_models() == []

    def test_list_groq_models_with_key(self):
        mock_model = SimpleNamespace(id="llama-3.3-70b-versatile", owned_by="groq", active=True)
        mock_client = MagicMock()
        mock_client.models.list.return_value = SimpleNamespace(data=[mock_model])

        models = list_groq_models("gsk_test", client=mock_client)

        assert len(models) == 1
        assert models[0]["id"] == "llama-3.3-70b-versatile"
        assert models[0]["active"] is True

    def test_check_model_health_compiles_summary(self):
        with (
            patch("cliv.llm.discovery.is_ollama_running", return_value=False),
            patch("cliv.llm.discovery.get_api_key", return_value=None),
            patch("cliv.llm.discovery.GroqBackend.is_available", return_value=True),
            patch("cliv.llm.discovery.OllamaBackend.is_available", return_value=True),
        ):
            health = check_model_health(current_model="llama3.1:8b", current_mode="offline")

        assert health["current_model"] == "llama3.1:8b"
        assert health["current_mode"] == "offline"
        assert health["ollama"]["running"] is False
        assert "not running" in health["ollama"]["error"]
        assert health["groq"]["has_key"] is False


# ---------------------------------------------------------------------------
# CLI cmd_models tests
# ---------------------------------------------------------------------------

class TestCLIModels:
    def test_cmd_models_with_downloaded_models(self):
        fake_health = {
            "current_model": "llama3.1:8b",
            "current_mode": "offline",
            "groq": {"installed": True, "has_key": False, "models": [], "error": None},
            "ollama": {
                "installed": True,
                "running": True,
                "models": [
                    {
                        "name": "llama3.1:8b",
                        "size_gb": 4.58,
                        "parameter_size": "8.0B",
                        "supports_tools": True,
                    }
                ],
                "error": None,
            },
            "recommendations": {},
        }
        with (
            patch("cliv.llm.discovery.check_model_health", return_value=fake_health),
            patch("cliv.cli.console.print") as mock_print,
        ):
            from cliv.cli import cmd_models
            cmd_models(SimpleNamespace())

        assert mock_print.called

    def test_cmd_models_when_daemon_down(self):
        fake_health = {
            "current_model": "none",
            "current_mode": "offline",
            "groq": {"installed": True, "has_key": False, "models": [], "error": None},
            "ollama": {
                "installed": True,
                "running": False,
                "models": [],
                "error": "not running",
            },
            "recommendations": {},
        }
        printed = []
        def _capture(*args, **kwargs):
            if args:
                printed.append(str(args[0]))
            return None

        with (
            patch("cliv.llm.discovery.check_model_health", return_value=fake_health),
            patch("cliv.cli.console.print", side_effect=_capture),
        ):
            from cliv.cli import cmd_models
            cmd_models(SimpleNamespace())

        output = " ".join(printed)
        assert "ollama serve" in output or "not reachable" in output.lower()


# ---------------------------------------------------------------------------
# Slash command /model tests
# ---------------------------------------------------------------------------

class FakeAgent:
    def __init__(self, model="llama3.1:8b", mode="offline"):
        self.local_model = model
        self.mode = mode
        self.backend = SimpleNamespace(model=model)
        self.stats = SessionStats()
        self.use_offline_tools = True
        self.api_key = "gsk_test"


def _make_slash_ctx(agent=None):
    from cliv.slash import SlashContext
    console = Console(record=True, width=120)
    agent = agent or FakeAgent()
    return SlashContext(agent=agent, console=console), console


class TestSlashModel:
    def test_model_status_no_args(self):
        ctx, console = _make_slash_ctx()
        fake_health = {
            "current_model": "llama3.1:8b",
            "current_mode": "offline",
            "groq": {"installed": True, "has_key": False, "models": [], "error": None},
            "ollama": {
                "installed": True,
                "running": True,
                "models": [{"name": "llama3.1:8b", "supports_tools": True, "parameter_size": "8B", "size_gb": 4.5}],
                "error": None,
            },
        }
        with patch("cliv.llm.discovery.check_model_health", return_value=fake_health):
            from cliv.slash import handle_model
            handle_model(ctx, "")

        out = console.export_text()
        assert "llama3.1:8b" in out
        assert "OFFLINE" in out

    def test_model_list_arg(self):
        ctx, console = _make_slash_ctx()
        fake_health = {
            "current_model": "llama3.1:8b",
            "current_mode": "offline",
            "groq": {"installed": True, "has_key": False, "models": [], "error": None},
            "ollama": {
                "installed": True,
                "running": True,
                "models": [{"name": "qwen2.5-coder:7b", "supports_tools": True, "parameter_size": "7B", "size_gb": 4.2}],
                "error": None,
            },
        }
        with patch("cliv.slash.check_model_health", return_value=fake_health):
            from cliv.slash import handle_model
            handle_model(ctx, "list")

        out = console.export_text()
        assert "qwen2.5-coder:7b" in out

    def test_model_switch_ollama_daemon_down(self):
        ctx, console = _make_slash_ctx()
        with (
            patch("cliv.llm.OllamaBackend.is_available", return_value=True),
            patch("cliv.slash.is_ollama_running", return_value=False),
        ):
            from cliv.slash import handle_model
            handle_model(ctx, "qwen2.5-coder:7b")

        out = console.export_text()
        assert "ollama serve" in out.lower() or "not running" in out.lower()

    def test_model_switch_ollama_success(self):
        ctx, console = _make_slash_ctx()
        fake_models = [{"name": "qwen2.5-coder:7b", "supports_tools": True}]
        fake_health = {
            "current_model": "llama3.1:8b",
            "current_mode": "offline",
            "groq": {"installed": True, "has_key": False, "models": [], "error": None},
            "ollama": {"installed": True, "running": True, "models": fake_models, "error": None},
        }
        with (
            patch("cliv.llm.OllamaBackend.is_available", return_value=True),
            patch("cliv.slash.is_ollama_running", return_value=True),
            patch("cliv.slash.check_model_health", return_value=fake_health),
            patch("cliv.llm.OllamaBackend.__init__", return_value=None),
        ):
            from cliv.slash import handle_model
            handle_model(ctx, "qwen2.5-coder:7b")

        assert ctx.agent.local_model == "qwen2.5-coder:7b"
        assert ctx.agent.mode == "offline"
        out = console.export_text()
        assert "Switched to offline model" in out
        assert "tool calling supported" in out.lower()

    def test_model_switch_groq_no_key(self):
        agent = FakeAgent()
        agent.api_key = None
        ctx, console = _make_slash_ctx(agent)
        with patch("cliv.config.get_api_key", return_value=None):
            from cliv.slash import handle_model
            handle_model(ctx, "groq:llama-3.3-70b-versatile")

        out = console.export_text()
        assert "No API key" in out

    def test_model_switch_groq_detected_without_prefix(self):
        agent = FakeAgent()
        agent.api_key = "gsk_test"
        ctx, console = _make_slash_ctx(agent)
        fake_health = {
            "current_model": "llama3.1:8b",
            "current_mode": "online",
            "groq": {
                "installed": True,
                "has_key": True,
                "models": [{"id": "meta-llama/llama-prompt-guard-2-22m", "active": True}],
                "error": None,
            },
            "ollama": {"installed": True, "running": True, "models": [], "error": None},
        }
        with (
            patch("cliv.slash.check_model_health", return_value=fake_health),
            patch("cliv.llm.GroqBackend.__init__", return_value=None),
        ):
            from cliv.slash import handle_model
            handle_model(ctx, "meta-llama/llama-prompt-guard-2-22m")

        assert ctx.agent.mode == "online"
        out = console.export_text()
        assert "Switched to Groq cloud model" in out
        assert "meta-llama/llama-prompt-guard-2-22m" in out

    def test_model_switch_blocks_missing_ollama_model(self):
        ctx, console = _make_slash_ctx()
        fake_models = [{"name": "llama3.1:8b", "supports_tools": True}]
        fake_health = {
            "current_model": "llama3.1:8b",
            "current_mode": "offline",
            "groq": {"installed": True, "has_key": False, "models": [], "error": None},
            "ollama": {"installed": True, "running": True, "models": fake_models, "error": None},
        }
        with (
            patch("cliv.slash.check_model_health", return_value=fake_health),
            patch("cliv.slash.is_ollama_running", return_value=True),
        ):
            from cliv.slash import handle_model
            handle_model(ctx, "ollama:nonexistent-model")

        out = console.export_text()
        assert "Cannot switch" in out
        assert "not downloaded" in out.lower()
        # Verify agent was NOT switched to broken model
        assert ctx.agent.local_model == "llama3.1:8b"


# ---------------------------------------------------------------------------
# Agent Fallback UX tests
# ---------------------------------------------------------------------------

class TestAgentFallbackUX:
    def test_user_notified_on_tool_calling_unsupported(self, tmp_path):
        from cliv.agent import AIAgent

        ui = MagicMock()
        backend = MagicMock()
        backend.chat.side_effect = ToolCallingUnsupported("no tools")
        backend.stats = SessionStats()

        mock_history = MagicMock()
        mock_history.max_messages = 50
        mock_history.load.return_value = []

        with (
            patch("cliv.agent.build_backend", return_value=(backend, "offline")),
            patch("cliv.agent.HistoryStore", return_value=mock_history),
            patch("cliv.agent.OfflineAssistant.chat", return_value="Intent fallback answer"),
        ):
            agent = AIAgent(mode="offline", force_offline=True, ui=ui)
            agent.backend = backend
            result = agent.chat("write code")

        assert result == "Intent fallback answer"
        # Check UI print was called with user-friendly note
        assert any(
            "does not support direct tool calling" in str(call)
            for call in ui.print.call_args_list
        )

    def test_can_fallback_to_offline_helper(self, tmp_path):
        from cliv.agent import AIAgent

        with (
            patch("cliv.agent.build_backend", return_value=(MagicMock(), "online")),
            patch("cliv.agent.HistoryStore"),
            patch("cliv.llm.OllamaBackend.is_available", return_value=True),
        ):
            agent = AIAgent(api_key="test")
            assert agent._can_fallback_to_offline() is True

    def test_graceful_runtime_fallback_on_network_loss(self, tmp_path):
        from cliv.agent import AIAgent

        ui = MagicMock()
        timeline = MagicMock()
        stats = SessionStats()

        backend = MagicMock()
        backend.chat.side_effect = BackendError("Connection timed out to Groq")
        backend.stats = stats

        mock_history = MagicMock()
        mock_history.max_messages = 50
        mock_history.load.return_value = []

        with (
            patch("cliv.agent.build_backend", return_value=(backend, "online")),
            patch("cliv.agent.HistoryStore", return_value=mock_history),
            patch("cliv.agent.OllamaBackend") as MockOllama,
            patch("cliv.agent.OfflineAssistant.chat", return_value="Answered locally"),
        ):
            MockOllama.is_available.return_value = True
            mock_local = MagicMock()
            mock_local.chat.return_value = ("Success from local", [])
            MockOllama.return_value = mock_local

            # Returns True on initial _auto_check_mode, False when tool loop fails mid-turn
            net_check = MagicMock(side_effect=[True, False, False])

            agent = AIAgent(
                api_key="test",
                ui=ui,
                timeline_store=timeline,
                internet_check=net_check,
            )
            agent.backend = backend
            agent.mode = "online"

            result = agent.chat("hello world")

        assert agent.mode == "offline"
        # Check user-facing UI notice
        assert any(
            "falling back" in str(call).lower() or "switched to local" in str(call).lower()
            for call in ui.print.call_args_list
        )
        # Check timeline event recorded
        assert any(call[0][0] == "model_fallback" for call in timeline.record_event.call_args_list)
