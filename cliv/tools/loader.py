"""Auto-discovery for tool plugins."""

from __future__ import annotations

import importlib
import inspect
import logging
from pathlib import Path
from typing import Dict

from cliv.tools.base import BaseTool


def load_tools() -> Dict[str, BaseTool]:
    """Import every ``BaseTool`` subclass dropped into ``cliv/tools``."""

    tools: Dict[str, BaseTool] = {}
    tools_dir = Path(__file__).parent
    if not tools_dir.exists():
        logging.warning("Tools directory not found: %s", tools_dir)
        return tools

    for file in tools_dir.glob("*.py"):
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
                    instance = obj()
                    if not instance.name:
                        logging.warning("Skipping unnamed tool in %s", module_name)
                        continue
                    tools[instance.name] = instance
                    logging.info("Dynamically loaded tool: %s", instance.name)
        except Exception as exc:  # noqa: BLE001 - one broken plugin must not kill startup
            logging.error("Failed to load tool module %s: %s", module_name, exc)
    return tools
