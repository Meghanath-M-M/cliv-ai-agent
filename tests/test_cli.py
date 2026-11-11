"""CLI routing and flag tests that avoid real terminal, network, and LLM I/O."""

from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console

from cliv import cli
from cliv.models import SessionStats
from cliv.timeline import TimelineStore
from cliv.ui import render_cost_dashboard


class _Stats:
    total_tokens = 0
    input_tokens = 0
    output_tokens = 0
    cost_usd = 0.0


def _install_fake_agent(monkeypatch, notices=None):
    """Replace ``cli.AIAgent`` and return the list of created instances."""

    created = []

    class FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.mode = "offline"
            self.stats = _Stats()
            self.chats = []
            self.cleared = False
            self.backend = SimpleNamespace(model="fake-model")
            self.model_notices = list(notices or [])
            created.append(self)

        def chat(self, user_input):
            self.chats.append(user_input)
            return "ok"

        def clear_history(self):
            self.cleared = True

        def load_session(self, session_data):
            self.loaded_session = session_data

    monkeypatch.setattr(cli, "AIAgent", FakeAgent)
    return created


def test_process_prompt_calls_agent_chat():
    agent = SimpleNamespace(
        mode="offline",
        stats=_Stats(),
        chat=lambda prompt: f"reply to {prompt}",
    )

    cli.process_prompt(agent, "hello")


def test_main_project_dna_prints_report_without_starting_agent(
    monkeypatch, project_dir, capsys
):
    (project_dir / "app.py").write_text(
        "def main():\n    pass\n\nif __name__ == \"__main__\":\n    main()\n"
    )
    created = _install_fake_agent(monkeypatch)
    monkeypatch.setattr(cli.sys, "argv", ["cliv", "--project-dna"])
    monkeypatch.setattr(
        cli,
        "getpass",
        lambda *args, **kwargs: pytest.fail("Project DNA must not prompt for a key"),
    )

    cli.main()

    output = capsys.readouterr().out
    assert created == []
    assert "CLIV PROJECT INTELLIGENCE" in output
    assert "app.py" in output


def test_main_timeline_prints_events_without_starting_agent(
    monkeypatch, project_dir, capsys
):
    store = TimelineStore()
    store.record(
        "request", "timeline smoke test", {"intent": "project_analysis"},
        project=project_dir,
    )
    created = _install_fake_agent(monkeypatch)
    monkeypatch.setattr(cli.sys, "argv", ["cliv", "--timeline"])
    monkeypatch.setattr(
        cli,
        "getpass",
        lambda *args, **kwargs: pytest.fail("Timeline must not prompt for a key"),
    )

    cli.main()

    output = capsys.readouterr().out
    assert created == []
    assert "CLIV DEVELOPMENT TIMELINE" in output
    assert "timeline smoke test" in output


def test_main_offline_single_shot_skips_key_prompt(monkeypatch):
    created = _install_fake_agent(monkeypatch)
    monkeypatch.setattr(cli.sys, "argv", ["cliv", "--offline", "hello"])
    monkeypatch.setattr(
        cli,
        "getpass",
        lambda *args, **kwargs: pytest.fail("offline mode must not prompt for a key"),
    )
    monkeypatch.setattr(
        cli,
        "check_internet",
        lambda *args, **kwargs: pytest.fail("offline mode must not check the network"),
    )

    cli.main()

    agent = created[0]
    assert agent.kwargs["force_offline"] is True
    assert agent.kwargs["use_offline_tools"] is True
    assert agent.chats == ["hello"]


def test_main_prints_selected_model_and_notices(monkeypatch, capsys):
    _install_fake_agent(
        monkeypatch,
        notices=["Switched to Groq fallback model 'backup-model'."],
    )
    monkeypatch.setattr(cli.sys, "argv", ["cliv", "--offline", "hello"])

    cli.main()

    output = capsys.readouterr().out
    assert "fake-model" in output
    assert "Model notice" in output
    assert "backup-model" in output


def test_main_legacy_offline_disables_ollama_tool_loop(monkeypatch):
    created = _install_fake_agent(monkeypatch)
    monkeypatch.setattr(
        cli.sys, "argv", ["cliv", "--offline", "--legacy-offline", "hello"]
    )

    cli.main()

    assert created[0].kwargs["use_offline_tools"] is False


def test_main_rejects_yes_and_dry_run_together(monkeypatch):
    created = _install_fake_agent(monkeypatch)
    monkeypatch.setattr(cli.sys, "argv", ["cliv", "--yes", "--dry-run", "hello"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    assert created == []


def test_interactive_clear_then_exit(monkeypatch):
    created = _install_fake_agent(monkeypatch)
    monkeypatch.setattr(cli.sys, "argv", ["cliv", "--offline"])
    inputs = iter(["clear", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    cli.main()

    assert created[0].cleared is True


def test_interactive_slash_commands(monkeypatch, capsys):
    created = _install_fake_agent(monkeypatch)
    monkeypatch.setattr(cli.sys, "argv", ["cliv", "--offline"])
    inputs = iter(["/help", "/clear", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    cli.main()

    assert created[0].cleared is True
    out = capsys.readouterr().out
    assert "/undo" in out
    assert "Session terminated" in out


def test_main_passes_security_permissions_to_agent(monkeypatch):
    created = _install_fake_agent(monkeypatch)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["cliv", "--offline", "--permissions", "read,execute", "hello"],
    )

    cli.main()

    guardian = created[0].kwargs["guardian"]
    assert guardian.policy.permissions == {"read", "execute"}


def test_main_rejects_unknown_permission(monkeypatch):
    created = _install_fake_agent(monkeypatch)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["cliv", "--offline", "--permissions", "read,teleport", "hello"],
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    assert created == []


def test_render_cost_dashboard_shows_tokens_and_cost():
    stats = SessionStats(total_tokens=1250, input_tokens=800, output_tokens=450, cost_usd=0.012345)
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=80)
    render_cost_dashboard(console, stats, model_name="llama-3.1-8b-instant", mode="offline")
    out = buf.getvalue()
    assert "Mode: OFFLINE" in out
    assert "Model: llama-3.1-8b-instant" in out
    assert "1,250" in out
    assert "800" in out
    assert "450" in out
    assert "$0.012345" in out


def test_cmd_cost_prints_project_summary(monkeypatch, tmp_path, capsys):
    from cliv.timeline import TimelineStore

    db_path = tmp_path / "timeline.db"
    monkeypatch.setenv("CLIV_TIMELINE_DB", str(db_path))

    store = TimelineStore(db_path)
    store.record_usage(
        "sess-123",
        model="openai/gpt-oss-120b",
        mode="online",
        input_tokens=1000,
        output_tokens=500,
        total_tokens=1500,
        cost_usd=0.00045,
    )

    monkeypatch.setattr(cli.sys, "argv", ["cliv", "cost"])
    cli.main()

    output = capsys.readouterr().out
    assert "Project Cost Summary" in output
    assert "Total Sessions:   1" in output
    assert "1,500" in output
    assert "0.000450" in output


def test_cmd_session_list_and_clear(monkeypatch, capsys):
    from cliv.session_io import SessionData, archive_session

    fake_session = SessionData(
        session_id="test-session-123",
        project_root="/test",
        created_at="2026-11-10T12:00:00",
        updated_at="2026-11-10T12:00:00",
        model="gpt-oss-120b",
        mode="online",
        stats={"total_tokens": 100},
        messages=[{"role": "user", "content": "hi"}],
    )
    monkeypatch.setattr("cliv.session_io.list_archived_sessions", lambda **kwargs: [
        {"id": "test-session-123", "created": "2026-11-10 12:00:00", "turns": 1, "model": "gpt-oss-120b"}
    ])

    monkeypatch.setattr(cli.sys, "argv", ["cliv", "session", "list"])
    cli.main()
    out = capsys.readouterr().out
    assert "test-session-123" in out
    assert "1 turns" in out

    # Clear
    monkeypatch.setattr(cli.sys, "argv", ["cliv", "session", "clear"])
    cli.main()
    out = capsys.readouterr().out
    assert "History cleared" in out


def test_cli_resume_flag(monkeypatch, tmp_path, capsys):
    from cliv.session_io import SessionData, export_session_json

    session_file = tmp_path / "saved_session.json"
    session_data = SessionData(
        session_id="resumed-session-999",
        project_root="/test",
        created_at="2026-11-10T12:00:00",
        updated_at="2026-11-10T12:00:00",
        model="fake-model",
        mode="offline",
        stats={"total_tokens": 100},
        messages=[{"role": "user", "content": "initial request"}],
    )
    export_session_json(session_data, session_file)

    created = _install_fake_agent(monkeypatch)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["cliv", "--offline", "--resume", str(session_file), "continue working"],
    )

    cli.main()

    assert created[0].loaded_session.session_id == "resumed-session-999"
    out = capsys.readouterr().out
    assert "Resumed session resumed-session-999" in out


def test_cmd_tools_list_and_new(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from rich.console import Console

    test_console = Console(record=True, width=120)
    monkeypatch.setattr(cli, "console", test_console)
    monkeypatch.setattr("cliv.security.project_root", lambda: tmp_path)

    # cliv tools list
    cli.cmd_tools(SimpleNamespace(tools_command="list"))
    list_out = test_console.export_text()
    assert "Installed Tools & Plugins" in list_out
    assert "read_file" in list_out

    # cliv tools new
    test_console = Console(record=True, width=120)
    monkeypatch.setattr(cli, "console", test_console)
    cli.cmd_tools(SimpleNamespace(tools_command="new", name="unit_test_tool", is_global=False))
    new_out = test_console.export_text()
    assert "Created tool plugin scaffold" in new_out
    assert (tmp_path / ".cliv" / "plugins" / "unit_test_tool.py").exists()

    # cliv tools info
    test_console = Console(record=True, width=120)
    monkeypatch.setattr(cli, "console", test_console)
    cli.cmd_tools(SimpleNamespace(tools_command="info", name="read_file"))
    info_out = test_console.export_text()
    assert "Tool: read_file" in info_out
    assert "Input Schema:" in info_out
