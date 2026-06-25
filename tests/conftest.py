"""Shared test fixtures."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


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


@pytest.fixture
def temp_history(tmp_path):
    """Provides a temporary history file path."""
    return tmp_path / "history.json"
