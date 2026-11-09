"""Build a seed payload shaped exactly like the llama.cpp WebUI's IndexedDB.

The WebUI (a SvelteKit app) persists conversations with Dexie in an IndexedDB
database named ``LlamaUi`` with two object stores::

    conversations: 'id, lastModified, currNode, name'
    messages:      'id, convId, type, role, timestamp, parent, children'

Each conversation is a tree.  ``currNode`` points at the *leaf* (latest)
message; the UI walks ``parent`` links up to the (excluded) ``root`` message
and renders the collected path.  :func:`build_seed_payload` turns the proxy's
linear conversation log into that exact shape so the browser can write it
straight into IndexedDB and immediately show the resumed chats.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List


def _new_id() -> str:
    return str(uuid.uuid4())


def _build_conversation(conv: Dict[str, Any]) -> Dict[str, Any]:
    """Shape one stored conversation into ``{conv, messages}`` for the WebUI."""
    turns = conv.get("turns", [])
    conv_uuid = _new_id()
    base_ms = int(turns[0]["ts"] * 1000) if turns else int(time.time() * 1000)

    turn_uuids = [_new_id() for _ in turns]
    root_id = _new_id()

    messages: List[Dict[str, Any]] = []
    # Root: parent=null, type='root'; not rendered but required to stop the walk.
    messages.append(
        {
            "id": root_id,
            "convId": conv_uuid,
            "type": "root",
            "role": "system",
            "content": "",
            "parent": None,
            "children": [turn_uuids[0]] if turn_uuids else [],
            "timestamp": base_ms - 1,
            "toolCalls": "",
        }
    )

    for i, turn in enumerate(turns):
        msg: Dict[str, Any] = {
            "id": turn_uuids[i],
            "convId": conv_uuid,
            "type": "system" if turn["role"] == "system" else "text",
            "role": turn["role"],
            "content": turn.get("content", ""),
            "parent": root_id if i == 0 else turn_uuids[i - 1],
            "children": [turn_uuids[i + 1]] if i + 1 < len(turns) else [],
            "timestamp": base_ms + i,
            "toolCalls": turn.get("tool_calls", "") or "",
        }
        if turn.get("reasoning"):
            msg["reasoningContent"] = turn["reasoning"]
        if turn.get("model"):
            msg["model"] = turn["model"]
        if turn["role"] == "tool" and turn.get("tool_call_id"):
            msg["toolCallId"] = turn["tool_call_id"]
        messages.append(msg)

    conv_row: Dict[str, Any] = {
        "id": conv_uuid,
        "name": conv.get("name") or "New chat",
        "lastModified": base_ms + len(turns),
        "currNode": turn_uuids[-1] if turn_uuids else "",
        "pinned": False,
    }
    return {"conv": conv_row, "messages": messages}


def build_seed_payload(convs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build the full seed payload for :data:`convs`.

    Each item of :data:`convs` is a plain dict with keys ``id, name,
    created_at, updated_at, first_user`` and ``turns`` (a list of turn dicts,
    see :mod:`llama_chat_proxy.recorder`).
    """
    ordered = sorted(convs, key=lambda c: c.get("updated_at", 0.0), reverse=True)
    return {
        "conversations": [_build_conversation(c) for c in ordered],
        "generated_at": time.time(),
        "count": len(ordered),
    }
