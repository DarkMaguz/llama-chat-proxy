import pytest

from llama_chat_proxy.crypto import Vault
from llama_chat_proxy.db import Storage
from llama_chat_proxy.recorder import Recorder

KEY = "ab" * 64


@pytest.fixture
def recorder(tmp_path):
    storage = Storage(str(tmp_path / "recorder.db"))
    vault = Vault(KEY)
    rec = Recorder(storage, vault)
    yield rec
    storage.close()


def test_new_conversation_and_continue(recorder):
    conv_id, appended = recorder.resolve_conversation(
        [{"role": "user", "content": "hello"}], None, True
    )
    assert appended == 1
    recorder.end_response(conv_id, "hi there")

    conv_id2, appended2 = recorder.resolve_conversation(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "more"},
        ],
        None,
        True,
    )
    assert conv_id2 == conv_id
    assert appended2 == 1
    recorder.end_response(conv_id2, "sure")

    seed = recorder.build_seed()
    assert seed["count"] == 1
    item = seed["conversations"][0]
    assert len(item["messages"]) == 5  # root + 4 turns
    roles = [m["role"] for m in item["messages"][1:]]
    assert roles == ["user", "assistant", "user", "assistant"]


def test_unrelated_request_starts_new_conversation(recorder):
    c1, _ = recorder.resolve_conversation([{"role": "user", "content": "alpha"}], None, True)
    recorder.end_response(c1, "A")
    c2, _ = recorder.resolve_conversation([{"role": "user", "content": "beta"}], None, True)
    recorder.end_response(c2, "B")
    assert c1 != c2
    assert recorder.build_seed()["count"] == 2


def test_no_header_does_not_fabricate_conversation(recorder):
    # A brand-new single user prompt with create_if_missing=False must not create a conv.
    conv_id, appended = recorder.resolve_conversation(
        [{"role": "user", "content": "generate a short title"}], None, False
    )
    assert conv_id is None
    assert appended == 0
    assert recorder.build_seed()["count"] == 0


def test_extends_existing_conversation(recorder):
    conv_id, _ = recorder.resolve_conversation([{"role": "user", "content": "hello"}], None, True)
    recorder.end_response(conv_id, "hi there")

    assert recorder.extends_existing_conversation(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "next"},
        ]
    ) is True
    # A lone, unrelated prompt matches nothing.
    assert (
        recorder.extends_existing_conversation([{"role": "user", "content": "something else"}])
        is False
    )


def test_retry_of_same_prefix_appends_only_assistant(recorder):
    conv_id, _ = recorder.resolve_conversation([{"role": "user", "content": "hi"}], None, True)
    # Simulate a failed response (no end_response), then an identical retry.
    conv_id2, appended = recorder.resolve_conversation([{"role": "user", "content": "hi"}], None, True)
    assert conv_id2 == conv_id
    assert appended == 0
    recorder.end_response(conv_id2, "yo")

    item = recorder.build_seed()["conversations"][0]
    assert len(item["messages"]) == 3  # root + user + assistant


def test_empty_response_is_not_stored(recorder):
    conv_id, _ = recorder.resolve_conversation([{"role": "user", "content": "hi"}], None, True)
    recorder.end_response(conv_id, "")  # empty content -> ignored
    item = recorder.build_seed()["conversations"][0]
    assert len(item["messages"]) == 2  # root + user only


def test_tool_calls_and_reasoning_are_captured(recorder):
    conv_id, _ = recorder.resolve_conversation([{"role": "user", "content": "hi"}], None, True)
    recorder.end_response(conv_id, "", reasoning="r", tool_calls='[{"id":"1","function":{"name":"f","arguments":"{}"}}]')
    item = recorder.build_seed()["conversations"][0]
    asst = item["messages"][-1]
    assert asst["role"] == "assistant"
    assert asst["reasoningContent"] == "r"
    assert asst["toolCalls"].startswith("[")


def test_persistence_across_reload(tmp_path):
    vault = Vault(KEY)
    s1 = Storage(str(tmp_path / "db.sqlite"))
    rec = Recorder(s1, vault)
    conv_id, _ = rec.resolve_conversation([{"role": "user", "content": "persist me"}], None, True)
    rec.end_response(conv_id, "done")
    s1.close()

    s2 = Storage(str(tmp_path / "db.sqlite"))
    rec2 = Recorder(s2, vault)
    s2.close()
    assert rec2.build_seed()["count"] == 1
    item = rec2.build_seed()["conversations"][0]
    assert item["messages"][1]["content"] == "persist me"
    assert item["messages"][-1]["content"] == "done"


def test_conversation_name_derived_from_first_user(recorder):
    long = "a " * 80
    conv_id, _ = recorder.resolve_conversation([{"role": "user", "content": long}], None, True)
    conv = recorder.build_seed()["conversations"][0]["conv"]
    assert len(conv["name"]) <= 61
    assert conv["name"].endswith("\u2026")


def test_stats(recorder):
    conv_id, _ = recorder.resolve_conversation([{"role": "user", "content": "x"}], None, True)
    recorder.end_response(conv_id, "y")
    stats = recorder.stats()
    assert stats["conversations"] == 1
    assert stats["turns"] == 2
    assert stats["storage"]["messages"] == 2
