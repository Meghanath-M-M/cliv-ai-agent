"""Tests for REPL slash-commands."""

from types import SimpleNamespace
from rich.console import Console

from cliv.slash import SlashContext, dispatch, SLASH_COMMANDS
from cliv.guardian import SecurityGuardian, SecurityPolicy


class FakeAgent:
    def __init__(self):
        self.cleared = False
        self.messages = []
        self.snapshots = None
        self.timeline = None
        self.auto_approve = False
        self.backend = SimpleNamespace(model="fake-model")
        self.local_model = "fake-model"
        self.mode = "offline"
        self.guardian = SecurityGuardian()
        self.touched_files = set()
        self.git_branch = "main"
        self.session_id = "test-session-123"
        self.tools = {}
        self.stats = SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost_usd=0.00012,
        )

    def clear_history(self):
        self.cleared = True
        self.messages.clear()

    def _finish_undo(self, snapshot_id=None):
        return f"Undone snapshot {snapshot_id}"

    def _execute_tool(self, tool_name, kwargs):
        return f"Executed {tool_name}"

    def to_session_data(self):
        from cliv.session_io import SessionData
        return SessionData(
            session_id=self.session_id,
            project_root="/test",
            created_at="2026-11-10T12:00:00",
            updated_at="2026-11-10T12:00:00",
            model="fake-model",
            mode="offline",
            stats={"total_tokens": 150, "cost_usd": 0.00012},
            messages=list(self.messages),
        )

    def load_session(self, session_data):
        self.messages = list(session_data.messages)

    def reload_tools(self):
        return self.tools


def test_dispatch_non_slash_returns_none():
    agent = FakeAgent()
    console = Console(record=True, width=120)
    ctx = SlashContext(agent=agent, console=console)

    assert dispatch("hello world", ctx) is None
    assert dispatch("", ctx) is None


def test_dispatch_unknown_slash_command():
    agent = FakeAgent()
    console = Console(record=True, width=120)
    ctx = SlashContext(agent=agent, console=console)

    res = dispatch("/foobar", ctx)
    assert res is True
    output = console.export_text()
    assert "Unknown slash-command" in output


def test_dispatch_exit_and_quit():
    agent = FakeAgent()
    console = Console(record=True, width=120)
    ctx = SlashContext(agent=agent, console=console)

    assert dispatch("/exit", ctx) is False
    assert dispatch("/quit", ctx) is False


def test_dispatch_help():
    agent = FakeAgent()
    console = Console(record=True, width=120)
    ctx = SlashContext(agent=agent, console=console)

    assert dispatch("/help", ctx) is True
    output = console.export_text()
    assert "/undo" in output
    assert "/diff" in output
    assert "/dna" in output
    assert "/snapshots" in output


def test_dispatch_clear():
    agent = FakeAgent()
    console = Console(record=True, width=120)
    ctx = SlashContext(agent=agent, console=console)

    assert dispatch("/clear", ctx) is True
    assert agent.cleared is True


def test_dispatch_history():
    agent = FakeAgent()
    agent.messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]
    console = Console(record=True, width=120)
    ctx = SlashContext(agent=agent, console=console)

    assert dispatch("/history", ctx) is True
    output = console.export_text()
    assert "What is 2+2?" in output
    assert "4" in output


def test_dispatch_undo():
    agent = FakeAgent()
    agent.snapshots = object()  # non-None
    console = Console(record=True, width=120)
    ctx = SlashContext(agent=agent, console=console)

    assert dispatch("/undo", ctx) is True
    output = console.export_text()
    assert "Undone snapshot None" in output

    assert dispatch("/undo 123", ctx) is True
    output = console.export_text()
    assert "Undone snapshot 123" in output


def test_dispatch_undo_no_snapshots():
    agent = FakeAgent()
    agent.snapshots = None
    console = Console(record=True, width=120)
    ctx = SlashContext(agent=agent, console=console)

    assert dispatch("/undo", ctx) is True
    output = console.export_text()
    assert "not enabled" in output


def test_dispatch_dna():
    agent = FakeAgent()
    console = Console(record=True, width=120)
    ctx = SlashContext(agent=agent, console=console)

    assert dispatch("/dna", ctx) is True
    output = console.export_text()
    assert "Executed analyse_project" in output


def test_dispatch_model():
    agent = FakeAgent()
    console = Console(record=True, width=120)
    ctx = SlashContext(agent=agent, console=console)

    # Show model
    assert dispatch("/model", ctx) is True
    output = console.export_text()
    assert "fake-model" in output


def test_dispatch_permissions():
    agent = FakeAgent()
    console = Console(record=True, width=120)
    ctx = SlashContext(agent=agent, console=console)

    # Show current permissions
    assert dispatch("/permissions", ctx) is True
    output = console.export_text()
    assert "Current permissions" in output

    # Set valid permissions
    assert dispatch("/permissions read,write", ctx) is True
    assert "read" in agent.guardian.policy.permissions
    assert "execute" not in agent.guardian.policy.permissions

    # Set invalid permissions
    assert dispatch("/permissions bogus", ctx) is True
    output = console.export_text()
    assert "Unknown permission" in output


def test_dispatch_rag_help_and_stats(monkeypatch):
    agent = FakeAgent()
    console = Console(record=True, width=120)
    ctx = SlashContext(agent=agent, console=console)

    # Help
    assert dispatch("/rag", ctx) is True
    output = console.export_text()
    assert "RAG Semantic Memory" in output

    # Mock indexer for stats
    class FakeIndexer:
        def get_stats(self):
            return {"total_chunks": 42, "unique_files": 5, "dimension": 384, "index_size": "12 KB"}
        def search(self, query, k=5):
            return []

    monkeypatch.setattr("cliv.intelligence.rag_indexer.create_rag_indexer", lambda **kwargs: FakeIndexer())

    console = Console(record=True, width=120)
    ctx = SlashContext(agent=agent, console=console)
    assert dispatch("/rag stats", ctx) is True
    output = console.export_text()
    assert "42" in output
    assert "12 KB" in output

    # Search empty results
    console = Console(record=True, width=120)
    ctx = SlashContext(agent=agent, console=console)
    assert dispatch("/rag query login", ctx) is True
    output = console.export_text()
    assert "No semantic matches found" in output


def test_dispatch_cost():
    agent = FakeAgent()
    console = Console(record=True, width=120)
    ctx = SlashContext(agent=agent, console=console)

    assert dispatch("/cost", ctx) is True
    output = console.export_text()
    assert "Active Session Cost" in output
    assert "150" in output
    assert "0.000120" in output


def test_dispatch_export_and_import(tmp_path):
    agent = FakeAgent()
    agent.messages = [{"role": "user", "content": "export test"}]
    console = Console(record=True, width=120)
    ctx = SlashContext(agent=agent, console=console)

    out_md = tmp_path / "my_session.md"
    assert dispatch(f"/export {out_md}", ctx) is True
    assert out_md.exists()
    assert "export test" in out_md.read_text(encoding="utf-8")

    out_json = tmp_path / "my_session.json"
    assert dispatch(f"/export {out_json} --json", ctx) is True
    assert out_json.exists()

    # Import into fresh agent
    new_agent = FakeAgent()
    new_console = Console(record=True)
    new_ctx = SlashContext(agent=new_agent, console=new_console)
    assert dispatch(f"/import {out_json}", new_ctx) is True
    assert len(new_agent.messages) == 1
    assert new_agent.messages[0]["content"] == "export test"


def test_dispatch_sessions(monkeypatch):
    agent = FakeAgent()
    console = Console(record=True, width=120)
    ctx = SlashContext(agent=agent, console=console)

    fake_sessions = [
        {
            "id": "sess-alpha-1234",
            "created": "2026-11-10 12:00:00",
            "model": "gpt-oss-120b",
            "turns": 3,
            "tokens": 1200,
            "cost": 0.00034,
        }
    ]
    monkeypatch.setattr("cliv.session_io.list_archived_sessions", lambda **kwargs: fake_sessions)

    assert dispatch("/sessions", ctx) is True
    output = console.export_text()
    assert "sess-alpha-12" in output
    assert "gpt-oss-120b" in output


def test_dispatch_tools_list_and_info():
    agent = FakeAgent()
    console = Console(record=True, width=120)
    ctx = SlashContext(agent=agent, console=console)

    fake_tool = SimpleNamespace(
        name="custom_scanner",
        plugin_source="project",
        plugin_path="/proj/.cliv/plugins/scanner.py",
        risk_level="low",
        permissions=frozenset({"read"}),
        description="Scan dependencies",
        input_schema={"type": "object", "properties": {"target": {"type": "string"}}},
    )
    agent.tools = {"custom_scanner": fake_tool}

    # /tools list
    assert dispatch("/tools", ctx) is True
    out_list = console.export_text()
    assert "custom_scanner" in out_list
    assert "project" in out_list
    assert "Scan dependencies" in out_list

    # /tools info
    console = Console(record=True, width=120)
    ctx = SlashContext(agent=agent, console=console)
    assert dispatch("/tools info custom_scanner", ctx) is True
    out_info = console.export_text()
    assert "Tool: custom_scanner" in out_info
    assert "scanner.py" in out_info
    assert "target" in out_info


def test_dispatch_tools_new_and_reload(tmp_path, monkeypatch):
    agent = FakeAgent()
    console = Console(record=True, width=120)
    ctx = SlashContext(agent=agent, console=console)

    monkeypatch.setattr("cliv.security.project_root", lambda: tmp_path)

    # /tools new
    assert dispatch("/tools new sample_tool", ctx) is True
    output = console.export_text()
    assert "Created tool plugin scaffold" in output
    assert (tmp_path / ".cliv" / "plugins" / "sample_tool.py").exists()

    # /tools reload
    assert dispatch("/tools reload", ctx) is True
    out_reload = console.export_text()
    assert "Reloaded tools" in out_reload
