import os
import json
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "cliv"
CONFIG_FILE = CONFIG_DIR / "config.json"
HISTORY_FILE = CONFIG_DIR / "history.json"
LOG_FILE = CONFIG_DIR / "agent.log"


def get_api_key():
    """Retrieve API key from env var, then config file."""
    # 1. Try environment variable
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        return api_key

    # 2. Try config file
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)
                return config.get("api_key")
        except Exception:
            pass

    return None


def save_api_key(api_key):
    """Save API key to config file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config = {}
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            pass
    config["api_key"] = api_key
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def get_config(key, default=None):
    """Get arbitrary config value."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)
                return config.get(key, default)
        except Exception:
            pass
    return default


def set_config(key, value):
    """Set arbitrary config value."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config = {}
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            pass
    config[key] = value
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
