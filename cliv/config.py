"""Configuration and API-key storage.

The key is never printed.  On POSIX the config file is created with mode
0600 and the directory with 0700.  Windows ignores most POSIX mode bits,
so the same calls are best-effort there.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

CONFIG_DIR = Path.home() / ".config" / "cliv"
CONFIG_FILE = CONFIG_DIR / "config.json"
HISTORY_FILE = CONFIG_DIR / "history.json"
LOG_FILE = CONFIG_DIR / "agent.log"
TIMELINE_FILE = CONFIG_DIR / "timeline.db"
SNAPSHOT_DIR = CONFIG_DIR / "snapshots"

# The primary online model plus ordered fallbacks.  Availability and tool
# compatibility are checked at startup, so a retired model does not break
# the CLI.  Override with GROQ_MODEL / GROQ_FALLBACK_MODELS or the saved
# config keys ``groq_model`` / ``groq_fallback_models``.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_GROQ_FALLBACK_MODELS = (
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
)



def _ensure_config_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(CONFIG_DIR, 0o700)
    except OSError:
        pass


def _read_config() -> Dict[str, Any]:
    if not CONFIG_FILE.exists():
        return {}
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        return config if isinstance(config, dict) else {}
    except Exception:
        return {}


def _secure_write_config(config: Dict[str, Any]) -> None:
    """Write JSON atomically with restrictive permissions."""

    _ensure_config_dir()
    temp_path = CONFIG_FILE.with_suffix(CONFIG_FILE.suffix + ".tmp")
    try:
        # 0600 on POSIX; the mode argument is harmless on Windows.
        fd = os.open(
            temp_path,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o600,
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
        os.replace(temp_path, CONFIG_FILE)
        try:
            os.chmod(CONFIG_FILE, 0o600)
        except OSError:
            pass
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass


def get_api_key() -> Optional[str]:
    """Retrieve API key from env var, then config file."""

    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        return api_key
    return _read_config().get("api_key")


def save_api_key(api_key: str) -> None:
    """Save the Groq API key with private file permissions."""

    config = _read_config()
    config["api_key"] = api_key
    _secure_write_config(config)


def get_config(key: str, default: Any = None) -> Any:
    """Get an arbitrary config value."""

    return _read_config().get(key, default)


def set_config(key: str, value: Any) -> None:
    """Set an arbitrary config value."""

    config = _read_config()
    config[key] = value
    _secure_write_config(config)


def get_groq_model() -> str:
    """Return the configured primary Groq model."""

    return (
        os.environ.get("GROQ_MODEL", "").strip()
        or str(get_config("groq_model", DEFAULT_GROQ_MODEL)).strip()
        or DEFAULT_GROQ_MODEL
    )


def get_groq_fallback_models() -> List[str]:
    """Return configured fallback models in priority order."""

    raw_env = os.environ.get("GROQ_FALLBACK_MODELS")
    if raw_env is not None:
        return [item.strip() for item in raw_env.split(",") if item.strip()]

    configured = get_config("groq_fallback_models", list(DEFAULT_GROQ_FALLBACK_MODELS))
    if isinstance(configured, str):
        configured = configured.split(",")
    if not isinstance(configured, list):
        configured = list(DEFAULT_GROQ_FALLBACK_MODELS)

    return [str(item).strip() for item in configured if str(item).strip()]
