"""Auto-discovery and dynamic plugin system for cliv tools."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import os
import re
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

from cliv.config import PLUGINS_DIR
from cliv.security import project_root
from cliv.tools.base import BaseTool


def load_tool_from_file(file_path: Path, source: str = "user") -> List[BaseTool]:
    """Dynamically load BaseTool subclasses from an external .py file.

    Parameters
    ----------
    file_path:
        Absolute or relative path to a Python source file.
    source:
        Provenance tag ('user', 'project', etc.) attached to loaded tools.

    Returns
    -------
    List[BaseTool]
        Instantiated tool instances found in the file.
    """
    path = Path(file_path).resolve()
    if not path.exists() or not path.is_file() or path.suffix != ".py":
        return []

    tools: List[BaseTool] = []
    clean_stem = re.sub(r"[^a-zA-Z0-9_]", "_", path.stem)
    module_name = f"cliv_plugin_{source}_{clean_stem}_{abs(hash(str(path)))}"

    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            logging.warning("Could not create module spec for plugin: %s", path)
            return []

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, BaseTool)
                and obj is not BaseTool
                and getattr(obj, "__module__", "") == module_name
            ):
                try:
                    instance = obj()
                    if not getattr(instance, "name", ""):
                        logging.warning("Skipping unnamed tool in %s", path)
                        continue
                    instance.plugin_source = source
                    instance.plugin_path = str(path)
                    tools.append(instance)
                    logging.info(
                        "Loaded custom tool plugin '%s' from %s (%s)",
                        instance.name,
                        path,
                        source,
                    )
                except Exception as exc:
                    logging.error(
                        "Failed to instantiate tool %s from %s: %s",
                        getattr(obj, "__name__", "unknown"),
                        path,
                        exc,
                    )
    except Exception as exc:
        logging.error("Failed to load plugin file %s: %s", path, exc)

    return tools


def create_plugin_template(
    name: str,
    target_dir: Path,
    description: str = "",
) -> Path:
    """Scaffold a starter tool plugin file in target_dir."""
    target_dir.mkdir(parents=True, exist_ok=True)
    clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", name.lower().strip()).strip("_")
    if not clean_name:
        clean_name = "custom_tool"

    file_path = target_dir / f"{clean_name}.py"
    class_name = "".join(part.capitalize() for part in clean_name.split("_")) + "Tool"
    desc = description.strip() or f"Custom {clean_name} tool."

    content = f'''"""Custom tool plugin: {clean_name}."""

from typing import Any, Dict
from cliv.tools.base import BaseTool


class {class_name}(BaseTool):
    """{desc}"""

    name = "{clean_name}"
    description = "{desc}"
    permissions = frozenset({{"read"}})
    risk_level = "low"
    input_schema: Dict[str, Any] = {{
        "type": "object",
        "properties": {{
            "query": {{
                "type": "string",
                "description": "Input parameter for {clean_name}",
            }}
        }},
        "required": ["query"],
    }}

    def execute(self, query: str = "", **kwargs: Any) -> str:
        """Execute the tool logic and return a string result."""
        return f"Executed {clean_name} with query: {{query}}"
'''
    file_path.write_text(content, encoding="utf-8")
    return file_path


def list_plugins(
    project: Optional[Path] = None,
    user_plugins_dir: Optional[Path] = None,
    project_plugins_dir: Optional[Path] = None,
) -> Dict[str, List[Path]]:
    """Return discovered plugin file paths grouped by source ('user', 'project')."""
    user_dir = user_plugins_dir or PLUGINS_DIR
    proj_dir = project_plugins_dir or ((project or project_root()).resolve() / ".cliv" / "plugins")

    user_files = [f for f in user_dir.glob("*.py") if not f.name.startswith("__")] if user_dir.exists() else []
    proj_files = [f for f in proj_dir.glob("*.py") if not f.name.startswith("__")] if proj_dir.exists() else []

    return {
        "user": sorted(user_files),
        "project": sorted(proj_files),
    }


def load_tools(
    project: Optional[Path] = None,
    user_plugins_dir: Optional[Path] = None,
    project_plugins_dir: Optional[Path] = None,
    include_user_plugins: bool = True,
    include_project_plugins: bool = True,
) -> Dict[str, BaseTool]:
    """Import every ``BaseTool`` subclass dropped into builtin, user, and project plugin dirs.

    Priority order (later overrides earlier):
    1. Builtin tools (cliv/tools/*.py)
    2. Global user plugins (~/.config/cliv/plugins/*.py)
    3. Project-local plugins (<project>/.cliv/plugins/*.py)
    """
    tools: Dict[str, BaseTool] = {}

    # 1. Built-in tools
    tools_dir = Path(__file__).parent
    if tools_dir.exists():
        for file in sorted(tools_dir.glob("*.py")):
            if file.name.startswith("__") or file.name in {"base.py", "loader.py"}:
                continue
            module_name = f"cliv.tools.{file.stem}"
            try:
                module = importlib.import_module(module_name)
                for _, obj in inspect.getmembers(module, inspect.isclass):
                    if (
                        issubclass(obj, BaseTool)
                        and obj is not BaseTool
                        and obj.__module__ == module_name
                    ):
                        try:
                            instance = obj()
                            if not instance.name:
                                logging.warning("Skipping unnamed tool in %s", module_name)
                                continue
                            instance.plugin_source = "builtin"
                            instance.plugin_path = str(file.resolve())
                            tools[instance.name] = instance
                            logging.info("Dynamically loaded tool: %s", instance.name)
                        except Exception as exc:
                            logging.error(
                                "Failed to instantiate builtin tool in %s: %s",
                                module_name,
                                exc,
                            )
            except Exception as exc:
                logging.error("Failed to load tool module %s: %s", module_name, exc)

    # 2. Global user plugins
    if include_user_plugins:
        u_dir = user_plugins_dir or PLUGINS_DIR
        try:
            if u_dir.exists() and u_dir.is_dir():
                for file in sorted(u_dir.glob("*.py")):
                    if file.name.startswith("__"):
                        continue
                    custom_tools = load_tool_from_file(file, source="user")
                    for t in custom_tools:
                        if t.name in tools:
                            logging.info(
                                "User plugin '%s' from %s overrides existing tool",
                                t.name,
                                file,
                            )
                        tools[t.name] = t
        except Exception as exc:
            logging.error("Error scanning user plugins directory %s: %s", u_dir, exc)

    # 3. Project-local plugins
    if include_project_plugins:
        try:
            root = (project or project_root()).resolve()
            p_dir = project_plugins_dir or (root / ".cliv" / "plugins")
            if p_dir.exists() and p_dir.is_dir():
                for file in sorted(p_dir.glob("*.py")):
                    if file.name.startswith("__"):
                        continue
                    custom_tools = load_tool_from_file(file, source="project")
                    for t in custom_tools:
                        if t.name in tools:
                            logging.info(
                                "Project plugin '%s' from %s overrides existing tool",
                                t.name,
                                file,
                            )
                        tools[t.name] = t
        except Exception as exc:
            logging.error("Error scanning project plugins directory: %s", exc)

    return tools
