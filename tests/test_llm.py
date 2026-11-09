"""Backend tests that do not require the optional provider SDKs."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cliv.llm.base import BackendError, normalise_messages
from cliv.llm.groq_backend import GroqBackend
from cliv.llm.ollama_backend import OllamaBackend
from cliv.llm.router import build_backend
from cliv.models import SessionStats


def _tool_messages():
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": {"path": "a.py"}},
                }
            ],
        }
    ]


def test_normalise_stringifies_for_groq():
    messages = normalise_messages(_tool_messages(), stringify_arguments=True)
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == '{"path": "a.py"}'
    # The input must not be mutated.
    assert isinstance(
        _tool_messages()[0]["tool_calls"][0]["function"]["arguments"], dict
    )


def test_normalise_parses_for_ollama():
    messages = _tool_messages()
    messages[0]["tool_calls"][0]["function"]["arguments"] = '{"path": "a.py"}'
    normalised = normalise_messages(messages, stringify_arguments=False)
    assert normalised[0]["tool_calls"][0]["function"]["arguments"] == {"path": "a.py"}


def test_groq_backend_parses_tool_calls_and_usage():
    backend = object.__new__(GroqBackend)
    backend.model = "test-model"
    backend.stats = SessionStats()
    backend.client = MagicMock()

    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="",
                    tool_calls=[
                        SimpleNamespace(
                            id="call_1",
                            function=SimpleNamespace(
                                name="read_file", arguments='{"path": "a.py"}'
                            ),
                        )
                    ],
                )
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )
    backend.client.chat.completions.create.return_value = response

    content, tool_calls = backend.chat(
        [{"role": "user", "content": "read a.py"}], []
    )

    assert content == ""
    assert tool_calls[0].name == "read_file"
    assert tool_calls[0].arguments == {"path": "a.py"}
    assert backend.stats.total_tokens == 15


def test_ollama_backend_extracts_dict_tool_calls():
    backend = object.__new__(OllamaBackend)
    response = {
        "message": {
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "list_files",
                        "arguments": {"path": "."},
                    }
                }
            ],
        }
    }
    content, tool_calls = backend._extract_message(response)
    assert content == ""
    assert tool_calls[0].name == "list_files"
    assert tool_calls[0].arguments == {"path": "."}


def _tool_schemas():
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ]


class _FakeGroq:
    available_models = {"groq-model", "fallback-model"}
    incompatible_models = set()
    list_error = None
    instances = []

    @classmethod
    def is_available(cls):
        return True

    def __init__(self, api_key, model=None, *, stats):
        self.name = "groq"
        self.model = model or "groq-model"
        self.stats = stats
        self.probe_calls = []
        type(self).instances.append(self)

    def list_model_ids(self):
        if type(self).list_error is not None:
            raise type(self).list_error
        return set(type(self).available_models)

    def probe_tool_calling(self, schemas):
        self.probe_calls.append(schemas)
        if self.model in type(self).incompatible_models:
            raise BackendError(f"'{self.model}' does not support tools")


class _FakeOllama:
    @classmethod
    def is_available(cls):
        return True

    def __init__(self, model, *, stats):
        self.name = "ollama"
        self.model = model
        self.stats = stats


def test_build_backend_prefers_online_with_key_and_network():
    with patch("cliv.llm.router.GroqBackend", _FakeGroq), patch(
        "cliv.llm.router.OllamaBackend", _FakeOllama
    ):
        backend, mode = build_backend(
            api_key="key",
            local_model="local",
            stats=SessionStats(),
            internet_check=lambda: True,
            model="groq-model",
            fallback_models=[],
        )
    assert mode == "online"
    assert backend.name == "groq"
    assert backend.model == "groq-model"


def test_build_backend_falls_back_offline_without_network():
    with patch("cliv.llm.router.GroqBackend", _FakeGroq), patch(
        "cliv.llm.router.OllamaBackend", _FakeOllama
    ):
        backend, mode = build_backend(
            api_key="key",
            local_model="local",
            stats=SessionStats(),
            internet_check=lambda: False,
        )
    assert mode == "offline"
    assert backend.name == "ollama"


def test_build_backend_force_offline():
    with patch("cliv.llm.router.GroqBackend", _FakeGroq), patch(
        "cliv.llm.router.OllamaBackend", _FakeOllama
    ):
        backend, mode = build_backend(
            api_key="key",
            local_model="local",
            stats=SessionStats(),
            force_offline=True,
            internet_check=lambda: True,
        )
    assert mode == "offline"
    assert backend.name == "ollama"


def test_build_backend_probes_real_tool_schemas(monkeypatch):
    monkeypatch.setattr(_FakeGroq, "available_models", {"groq-model"})
    monkeypatch.setattr(_FakeGroq, "incompatible_models", set())
    schemas = _tool_schemas()
    notices = []

    with patch("cliv.llm.router.GroqBackend", _FakeGroq), patch(
        "cliv.llm.router.OllamaBackend", _FakeOllama
    ):
        backend, mode = build_backend(
            api_key="key",
            local_model="local",
            stats=SessionStats(),
            internet_check=lambda: True,
            tool_schemas=schemas,
            model="groq-model",
            fallback_models=[],
            notices=notices,
        )

    assert mode == "online"
    assert backend.probe_calls == [schemas]
    assert notices == []


def test_build_backend_uses_fallback_when_primary_is_retired(monkeypatch, caplog):
    monkeypatch.setattr(_FakeGroq, "available_models", {"fallback-model"})
    monkeypatch.setattr(_FakeGroq, "incompatible_models", set())
    notices = []

    with patch("cliv.llm.router.GroqBackend", _FakeGroq), patch(
        "cliv.llm.router.OllamaBackend", _FakeOllama
    ):
        backend, mode = build_backend(
            api_key="key",
            local_model="local",
            stats=SessionStats(),
            internet_check=lambda: True,
            tool_schemas=_tool_schemas(),
            model="groq-model",
            fallback_models=["fallback-model"],
            notices=notices,
        )

    assert mode == "online"
    assert backend.model == "fallback-model"
    assert any("'groq-model' is unavailable" in notice for notice in notices)
    assert any(
        "Switched to Groq fallback model 'fallback-model'" in notice
        for notice in notices
    )
    assert any(
        "Switched to Groq fallback model 'fallback-model'" in record.message
        for record in caplog.records
    )


def test_build_backend_skips_model_without_tool_support(monkeypatch):
    monkeypatch.setattr(
        _FakeGroq, "available_models", {"groq-model", "fallback-model"}
    )
    monkeypatch.setattr(_FakeGroq, "incompatible_models", {"groq-model"})
    notices = []

    with patch("cliv.llm.router.GroqBackend", _FakeGroq), patch(
        "cliv.llm.router.OllamaBackend", _FakeOllama
    ):
        backend, mode = build_backend(
            api_key="key",
            local_model="local",
            stats=SessionStats(),
            internet_check=lambda: True,
            tool_schemas=_tool_schemas(),
            model="groq-model",
            fallback_models=["fallback-model"],
            notices=notices,
        )

    assert mode == "online"
    assert backend.model == "fallback-model"
    assert len(backend.probe_calls) == 2
    assert any("cannot use CLIV's tools" in notice for notice in notices)


def test_build_backend_uses_ollama_when_all_groq_models_fail(monkeypatch):
    monkeypatch.setattr(_FakeGroq, "available_models", {"groq-model"})
    monkeypatch.setattr(_FakeGroq, "incompatible_models", {"groq-model"})
    notices = []

    with patch("cliv.llm.router.GroqBackend", _FakeGroq), patch(
        "cliv.llm.router.OllamaBackend", _FakeOllama
    ):
        backend, mode = build_backend(
            api_key="key",
            local_model="local",
            stats=SessionStats(),
            internet_check=lambda: True,
            tool_schemas=_tool_schemas(),
            model="groq-model",
            fallback_models=[],
            notices=notices,
        )

    assert mode == "offline"
    assert backend.name == "ollama"
    assert backend.model == "local"
    assert any("No configured Groq model passed" in notice for notice in notices)
    assert any("Using local Ollama model 'local'" in notice for notice in notices)


def test_build_backend_uses_ollama_when_model_listing_fails(monkeypatch):
    monkeypatch.setattr(
        _FakeGroq, "list_error", BackendError("models endpoint down")
    )
    notices = []

    with patch("cliv.llm.router.GroqBackend", _FakeGroq), patch(
        "cliv.llm.router.OllamaBackend", _FakeOllama
    ):
        backend, mode = build_backend(
            api_key="key",
            local_model="local",
            stats=SessionStats(),
            internet_check=lambda: True,
            model="groq-model",
            fallback_models=[],
            notices=notices,
        )

    assert mode == "offline"
    assert backend.name == "ollama"
    assert any(
        "Could not verify Groq model availability" in notice for notice in notices
    )


def _uninitialised_groq():
    backend = object.__new__(GroqBackend)
    backend.model = "model-a"
    backend.stats = SessionStats()
    backend.client = MagicMock()
    return backend


def test_groq_backend_lists_models_from_objects_and_dicts():
    backend = _uninitialised_groq()
    backend.client.models.list.return_value = SimpleNamespace(
        data=[SimpleNamespace(id="model-a"), {"id": "model-b"}]
    )

    assert backend.list_model_ids() == {"model-a", "model-b"}


def test_groq_backend_wraps_model_listing_errors():
    backend = _uninitialised_groq()
    backend.client.models.list.side_effect = RuntimeError("denied")

    with pytest.raises(BackendError, match="Could not list Groq models"):
        backend.list_model_ids()


def test_groq_probe_sends_actual_tool_schemas():
    backend = _uninitialised_groq()
    schemas = _tool_schemas()

    backend.probe_tool_calling(schemas)

    kwargs = backend.client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "model-a"
    assert kwargs["tools"] == schemas
    assert kwargs["tool_choice"] == "auto"


def test_groq_probe_reports_tool_incompatibility():
    backend = _uninitialised_groq()
    backend.client.chat.completions.create.side_effect = RuntimeError(
        "tools are not supported"
    )

    with pytest.raises(BackendError, match="rejected CLIV's tool schemas"):
        backend.probe_tool_calling(_tool_schemas())


def test_groq_probe_skips_when_no_tools_are_given():
    backend = _uninitialised_groq()

    backend.probe_tool_calling([])

    backend.client.chat.completions.create.assert_not_called()
