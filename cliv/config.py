import os
import json
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "cliv"
CONFIG_FILE = CONFIG_DIR / "config.json"


def get_api_key():
    # 1. Try environment variable
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        return api_key

    # 2. Try config file
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
                return config.get("api_key")
        except Exception:
            pass

    return None


def save_api_key(api_key):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump({"api_key": api_key}, f)
