"""Tests for history persistence and context budgeting."""

import json

from cliv.history import HistoryStore, _normalise_content, estimate_tokens, trim_messages


def test_estimate_tokens_is_roughly_characters_over_four():
    assert estimate_tokens("abcdefgh") == 2
    assert estimate_tokens("") == 0


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "history.json"
    store = HistoryStore(path)
    messages = [{"role": "user", "content": "hello"}]
    store.save(messages)
    assert store.load() == messages


def test_save_is_atomic_and_creates_parent(tmp_path):
    path = tmp_path / "nested" / "history.json"
    store = HistoryStore(path)
    store.save([{"role": "user", "content": "hi"}])
    assert path.exists()
    assert not path.with_suffix(".json.tmp").exists()


def test_corrupt_history_does_not_crash(tmp_path):
    path = tmp_path / "history.json"
    path.write_text("{not json")
    store = HistoryStore(path)
    assert store.load() == []


def test_trim_keeps_newest_turn_under_budget():
    messages = [
        {"role": "user", "content": "old " * 500},
        {"role": "assistant", "content": "old answer " * 500},
        {"role": "user", "content": "new request"},
        {"role": "assistant", "content": "new answer"},
    ]
    trimmed = trim_messages(messages, max_tokens=20)
    assert trimmed[0]["role"] == "user"
    assert trimmed[0]["content"] == "new request"
    assert trimmed[-1]["content"] == "new answer"


def test_trim_never_starts_with_orphan_tool_message():
    messages = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "", "tool_calls": []},
        {"role": "tool", "content": "result"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "two"},
        {"role": "assistant", "content": "", "tool_calls": []},
        {"role": "tool", "content": "result two"},
    ]
    trimmed = trim_messages(messages, max_tokens=1000)
    assert trimmed[0]["role"] == "user"
    assert trimmed[0]["content"] == "one"


def test_history_file_stores_normalised_arguments(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(
        json.dumps(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path": "a.py"}',
                            },
                        }
                    ],
                }
            ]
        )
    )
    loaded = HistoryStore(path).load()
    assert loaded[0]["tool_calls"][0]["function"]["arguments"] == {"path": "a.py"}


# ---------------------------------------------------------------------------
# Regression: list-typed content in history (the "undo" corruption bug)
# ---------------------------------------------------------------------------

def test_normalise_content_converts_list_to_string():
    """List content (e.g. ``["undo"]``) must be coerced to a plain string."""
    messages = [
        {"role": "user", "content": ["undo"]},
        {"role": "assistant", "content": "ok"},
    ]
    result = _normalise_content(messages)
    assert result[0]["content"] == "undo"
    assert result[1]["content"] == "ok"


def test_normalise_content_handles_multi_item_list():
    messages = [{"role": "user", "content": ["hello", "world"]}]
    result = _normalise_content(messages)
    assert result[0]["content"] == "hello world"


def test_normalise_content_leaves_strings_unchanged():
    messages = [{"role": "user", "content": "already a string"}]
    result = _normalise_content(messages)
    assert result[0]["content"] == "already a string"


def test_load_normalises_list_content_on_disk(tmp_path):
    """Corrupt history with list content is healed transparently on load."""
    path = tmp_path / "history.json"
    path.write_text(
        json.dumps([{"role": "user", "content": ["undo"]}])
    )
    loaded = HistoryStore(path).load()
    assert loaded[0]["content"] == "undo"


def test_save_normalises_list_content_before_writing(tmp_path):
    """Saving a message with list content must not persist the malformed form."""
    path = tmp_path / "history.json"
    store = HistoryStore(path)
    store.save([{"role": "user", "content": ["undo"]}])
    raw = json.loads(path.read_text())
    assert raw[0]["content"] == "undo"
