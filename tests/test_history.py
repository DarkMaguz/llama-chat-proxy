from llama_chat_proxy.history import build_seed_payload


def _turn(role, content, **kw):
    base = {
        "role": role,
        "content": content,
        "reasoning": "",
        "tool_calls": "",
        "model": "model-x",
        "completion_id": "",
        "tool_call_id": "",
        "ts": 1000.0,
    }
    base.update(kw)
    return base


def test_seed_has_root_chain_and_currnode():
    convs = [
        {
            "id": "c1",
            "name": "hello",
            "created_at": 1000.0,
            "updated_at": 1003.0,
            "first_user": "hello",
            "turns": [
                _turn("user", "hello"),
                _turn("assistant", "hi!", reasoning="thinking..."),
            ],
        }
    ]
    seed = build_seed_payload(convs)
    assert seed["count"] == 1
    item = seed["conversations"][0]
    conv, msgs = item["conv"], item["messages"]

    root, user, asst = msgs[0], msgs[1], msgs[2]

    assert root["type"] == "root"
    assert root["parent"] is None
    assert root["children"] == [user["id"]]

    assert user["parent"] == root["id"]
    assert user["role"] == "user"
    assert user["type"] == "text"
    assert user["children"] == [asst["id"]]

    assert asst["parent"] == user["id"]
    assert asst["role"] == "assistant"
    assert asst["reasoningContent"] == "thinking..."
    assert asst["children"] == []

    assert conv["currNode"] == asst["id"]
    assert conv["id"] == user["convId"] == asst["convId"] == root["convId"]


def test_seed_timestamps_strictly_increasing():
    convs = [
        {
            "id": "c1",
            "name": "x",
            "created_at": 0,
            "updated_at": 0,
            "first_user": "",
            "turns": [_turn("user", "a"), _turn("assistant", "b"), _turn("user", "c")],
        }
    ]
    msgs = build_seed_payload(convs)["conversations"][0]["messages"]
    stamps = [m["timestamp"] for m in msgs]
    assert stamps == sorted(stamps)
    assert len(set(stamps)) == len(stamps)


def test_seed_orders_conversations_newest_first():
    convs = [
        {"id": "old", "name": "old", "created_at": 1, "updated_at": 10.0, "first_user": "", "turns": []},
        {"id": "new", "name": "new", "created_at": 2, "updated_at": 99.0, "first_user": "", "turns": []},
    ]
    # Note: build_seed_payload mints a fresh uuid for each conversation id
    # (so `conv["id"]` is not stable across runs) — assert ordering on the
    # preserved `name` instead.
    ordered = [c["conv"]["name"] for c in build_seed_payload(convs)["conversations"]]
    assert ordered == ["new", "old"]


def test_empty_conversation_is_valid():
    seed = build_seed_payload(
        [{"id": "c", "name": "empty", "created_at": 1, "updated_at": 2, "first_user": "", "turns": []}]
    )
    item = seed["conversations"][0]
    assert item["conv"]["currNode"] == ""
    assert len(item["messages"]) == 1  # just the root
    assert item["messages"][0]["children"] == []
