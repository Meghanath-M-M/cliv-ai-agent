"""cliv - Terminal-native intelligence for your local codebase."""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("cliv-ai")
except Exception:
    try:
        __version__ = importlib.metadata.version("cliv")
    except Exception:
        __version__ = "1.0.0"

__all__ = ["__version__"]
