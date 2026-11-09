"""Shared test fixtures."""

import pytest
from unittest.mock import MagicMock


@pytest.fixture
def project_dir(tmp_path, monkeypatch):
    """Run a tool test inside an isolated project root."""

    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def mock_groq_client():
    """Returns a mocked Groq client."""

    client = MagicMock()
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message = MagicMock()
    response.choices[0].message.content = "Hello from Groq"
    response.choices[0].message.tool_calls = None
    response.usage = MagicMock()
    response.usage.total_tokens = 100
    response.usage.prompt_tokens = 80
    response.usage.completion_tokens = 20
    client.chat.completions.create.return_value = response
    return client
