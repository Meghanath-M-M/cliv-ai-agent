"""Unit tests for cliv custom tool plugin system."""

import os
from pathlib import Path
import pytest

from cliv.tools.base import BaseTool
from cliv.tools.loader import (
    load_tools,
    load_tool_from_file,
    create_plugin_template,
    list_plugins,
)


def test_builtin_tools_loaded():
    tools = load_tools(include_user_plugins=False, include_project_plugins=False)
    assert "read_file" in tools
    assert "edit_file" in tools
    assert "run_command" in tools

    read_tool = tools["read_file"]
    assert read_tool.plugin_source == "builtin"
    assert read_tool.plugin_path is not None
    assert "read_file.py" in read_tool.plugin_path


def test_user_plugin_discovery(tmp_path):
    user_plugins_dir = tmp_path / "user_plugins"
    user_plugins_dir.mkdir(parents=True)

    plugin_code = '''
from cliv.tools.base import BaseTool

class GreetingTool(BaseTool):
    name = "greet_user"
    description = "Greet a user by name."
    permissions = frozenset({"read"})
    risk_level = "low"
    input_schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }

    def execute(self, name: str = "World", **kwargs) -> str:
        return f"Hello, {name}!"
'''
    (user_plugins_dir / "greet.py").write_text(plugin_code, encoding="utf-8")

    tools = load_tools(
        user_plugins_dir=user_plugins_dir,
        include_user_plugins=True,
        include_project_plugins=False,
    )
    assert "greet_user" in tools
    tool = tools["greet_user"]
    assert tool.plugin_source == "user"
    assert tool.execute(name="Alice") == "Hello, Alice!"


def test_project_plugin_discovery(tmp_path):
    proj_dir = tmp_path / "my_project"
    proj_plugins_dir = proj_dir / ".cliv" / "plugins"
    proj_plugins_dir.mkdir(parents=True)

    plugin_code = '''
from cliv.tools.base import BaseTool

class ProjectDeployTool(BaseTool):
    name = "project_deploy"
    description = "Deploy the project."
    permissions = frozenset({"execute", "network"})
    risk_level = "high"

    def execute(self, **kwargs) -> str:
        return "Deployed to staging!"
'''
    (proj_plugins_dir / "deploy.py").write_text(plugin_code, encoding="utf-8")

    tools = load_tools(
        project=proj_dir,
        include_user_plugins=False,
        include_project_plugins=True,
    )
    assert "project_deploy" in tools
    tool = tools["project_deploy"]
    assert tool.plugin_source == "project"
    assert tool.risk_level == "high"
    assert "network" in tool.permissions
    assert tool.execute() == "Deployed to staging!"


def test_project_plugin_overrides_builtin(tmp_path):
    proj_dir = tmp_path / "override_project"
    proj_plugins_dir = proj_dir / ".cliv" / "plugins"
    proj_plugins_dir.mkdir(parents=True)

    override_code = '''
from cliv.tools.base import BaseTool

class CustomReadFile(BaseTool):
    name = "read_file"
    description = "Custom read file with watermarking."

    def execute(self, path: str = "", **kwargs) -> str:
        return f"[WATERMARKED] Reading {path}"
'''
    (proj_plugins_dir / "custom_read.py").write_text(override_code, encoding="utf-8")

    tools = load_tools(
        project=proj_dir,
        include_user_plugins=False,
        include_project_plugins=True,
    )
    assert "read_file" in tools
    tool = tools["read_file"]
    assert tool.plugin_source == "project"
    assert tool.execute(path="secret.txt") == "[WATERMARKED] Reading secret.txt"


def test_plugin_resilience_to_syntax_and_runtime_errors(tmp_path):
    bad_dir = tmp_path / "bad_plugins"
    bad_dir.mkdir()

    # Syntax error file
    (bad_dir / "syntax_error.py").write_text("def broken_code( missing_paren:", encoding="utf-8")

    # Runtime error during __init__
    runtime_err_code = '''
from cliv.tools.base import BaseTool

class ExplodingTool(BaseTool):
    name = "exploding_tool"

    def __init__(self):
        super().__init__()
        raise RuntimeError("Boom on load!")

    def execute(self, **kwargs):
        return "never"
'''
    (bad_dir / "exploding.py").write_text(runtime_err_code, encoding="utf-8")

    # Unnamed tool
    unnamed_code = '''
from cliv.tools.base import BaseTool

class NoNameTool(BaseTool):
    name = ""

    def execute(self, **kwargs):
        return "none"
'''
    (bad_dir / "no_name.py").write_text(unnamed_code, encoding="utf-8")

    # Good tool in the same directory
    good_code = '''
from cliv.tools.base import BaseTool

class SafeTool(BaseTool):
    name = "safe_tool"
    description = "A reliable tool."

    def execute(self, **kwargs):
        return "all good"
'''
    (bad_dir / "good.py").write_text(good_code, encoding="utf-8")

    # Loading bad_dir must not raise, and must load safe_tool
    tools = load_tools(
        user_plugins_dir=bad_dir,
        include_user_plugins=True,
        include_project_plugins=False,
    )
    assert "safe_tool" in tools
    assert "exploding_tool" not in tools
    assert "" not in tools


def test_create_plugin_template_and_execute(tmp_path):
    target_dir = tmp_path / "plugins"
    scaffold = create_plugin_template(
        name="code_analyzer",
        target_dir=target_dir,
        description="Analyzes code complexity",
    )

    assert scaffold.exists()
    assert scaffold.name == "code_analyzer.py"

    loaded = load_tool_from_file(scaffold, source="user")
    assert len(loaded) == 1
    tool = loaded[0]
    assert tool.name == "code_analyzer"
    assert "complexity" in tool.description
    assert tool.plugin_source == "user"
    assert "Executed code_analyzer with query: test" in tool.execute(query="test")


def test_list_plugins(tmp_path):
    user_dir = tmp_path / "user"
    proj_dir = tmp_path / "project"
    proj_plugins_dir = proj_dir / ".cliv" / "plugins"

    user_dir.mkdir(parents=True)
    proj_plugins_dir.mkdir(parents=True)

    (user_dir / "user_one.py").write_text("# tool", encoding="utf-8")
    (proj_plugins_dir / "proj_one.py").write_text("# tool", encoding="utf-8")
    (proj_plugins_dir / "__init__.py").write_text("", encoding="utf-8")

    found = list_plugins(
        project=proj_dir,
        user_plugins_dir=user_dir,
        project_plugins_dir=proj_plugins_dir,
    )
    assert len(found["user"]) == 1
    assert found["user"][0].name == "user_one.py"
    assert len(found["project"]) == 1
    assert found["project"][0].name == "proj_one.py"
