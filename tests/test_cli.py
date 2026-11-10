"""CLI routing and flag tests that avoid real terminal, network, and LLM I/O."""

from types import SimpleNamespace

import pytest

from cliv import cli
from cliv.timeline import TimelineStore


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
