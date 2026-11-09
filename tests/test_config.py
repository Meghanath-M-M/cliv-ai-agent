"""Tests for API-key storage and permissions."""

import os
import stat

import pytest

import cliv.config as config


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    config_dir = tmp_path / "cliv"
    monkeypatch.setattr(config, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(config, "CONFIG_FILE", config_dir / "config.json")
    return config_dir


def test_save_and_get_api_key(isolated_config, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    config.save_api_key("secret-key")
    assert config.get_api_key() == "secret-key"
    assert config.get_config("api_key") == "secret-key"


def test_environment_wins(isolated_config, monkeypatch):
    config.save_api_key("file-key")
    monkeypatch.setenv("GROQ_API_KEY", "env-key")
    assert config.get_api_key() == "env-key"


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are not enforced on Windows")
def test_api_key_file_is_private(isolated_config, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    config.save_api_key("secret-key")
    mode = stat.S_IMODE(config.CONFIG_FILE.stat().st_mode)
    assert mode == 0o600


def test_get_and_set_config(isolated_config):
    config.set_config("theme", "dark")
    assert config.get_config("theme") == "dark"
    assert config.get_config("missing", "fallback") == "fallback"


def test_groq_model_priority(isolated_config, monkeypatch):
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    assert config.get_groq_model() == config.DEFAULT_GROQ_MODEL

    config.set_config("groq_model", "saved-model")
    assert config.get_groq_model() == "saved-model"

    monkeypatch.setenv("GROQ_MODEL", "env-model")
    assert config.get_groq_model() == "env-model"


def test_groq_fallback_models_from_config_and_env(isolated_config, monkeypatch):
    monkeypatch.delenv("GROQ_FALLBACK_MODELS", raising=False)
    assert config.get_groq_fallback_models() == list(
        config.DEFAULT_GROQ_FALLBACK_MODELS
    )

    config.set_config("groq_fallback_models", ["first", "second"])
    assert config.get_groq_fallback_models() == ["first", "second"]

    monkeypatch.setenv("GROQ_FALLBACK_MODELS", "third, fourth,,")
    assert config.get_groq_fallback_models() == ["third", "fourth"]
