"""Record chat turns from OpenAI-compatible traffic into conversations.

The llama.cpp WebUI is fully client-side: it keeps its own per-browser
IndexedDB and talks to the model with stateless ``POST /v1/chat/completions``
calls (the whole message array is re-sent every turn).  The proxy cannot see
the browser's conversation UUIDs, so it reconstructs conversations from the
wire by *prefix matching*: a request belongs to an existing conversation when
that conversation's stored turns are an exact prefix of the request's
messages.  A brand-new request that matches nothing starts a new
conversation (only when ``create_if_missing`` is set -- see
:ref:`should-record` so that internal calls such as title generation never
fabricate a conversation).

``should-record`` is decided by the caller (see :mod:`llama_chat_proxy.app`):
real streaming turns carry an ``x-conversation-id`` header, internal calls do
not.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .crypto import Vault, VaultError
from .db import Storage
from .history import build_seed_payload

log = logging.getLogger(__name__)

_ROLES = ("system", "user", "assistant", "tool")


@dataclass
class Turn:
    """One normalised chat turn (a request message or a streamed response)."""

    role: str
    content: str
    ts: float
    reasoning: str = ""
    tool_calls: str = ""
    model: str = ""
    completion_id: str = ""
    tool_call_id: str = ""

    def as_payload(self) -> dict:
        return {
            "c": self.content,
            "r": self.reasoning,
            "t": self.tool_calls,
            "m": self.model,
            "i": self.completion_id,
        }


@dataclass
class Conversation:
    id: str
    name: str
    created_at: float
    updated_at: float
    first_user: str
    turns: List[Turn] = field(default_factory=list)
    # (role, content) list used for cheap prefix matching.
    norm: List[Tuple[str, str]] = field(default_factory=list)


class Recorder:
    """In-memory conversation store persisted to :class:`Storage`/Vault."""

    def __init__(self, storage: Storage, vault: Vault, max_conversations: int = 500) -> None:
        self._storage = storage
        self._vault = vault
        self._max = max(1, max_conversations)
        self._convs: Dict[str, Conversation] = {}
        self._lock = threading.Lock()
        self._load()

    # -- lifecycle -----------------------------------------------------------
    def _load(self) -> None:
        with self._lock:
            for row in self._storage.list_conversations():
                conv = Conversation(
                    id=row.id,
                    name=row.name,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                    first_user=row.first_user,
                )
                for mrow in self._storage.messages_for(row.id):
                    try:
                        data = self._vault.decrypt_obj(mrow.payload)
                    except (VaultError, ValueError) as exc:
                        log.warning("skipping unreadable message %s#%d: %s", row.id, mrow.seq, exc)
                        continue
                    turn = Turn(
                        role=mrow.role,
                        content=data.get("c", ""),
                        ts=mrow.ts,
                        reasoning=data.get("r", ""),
                        tool_calls=data.get("t", ""),
                        model=data.get("m", mrow.model or "") or "",
                        completion_id=data.get("i", ""),
                    )
                    conv.turns.append(turn)
                    conv.norm.append((turn.role, turn.content))
                self._convs[conv.id] = conv
        log.info("loaded %d conversations from storage", len(self._convs))

    # -- public API (thread-safe) -------------------------------------------
    def resolve_conversation(
        self, messages: list, model: Optional[str], create_if_missing: bool
    ) -> Tuple[Optional[str], int]:
        """Attach a request to a conversation; return ``(conv_id, appended)``.

        ``appended`` is the number of *new* user turns stored (0 for a
        retry/re-send of an already-stored prefix).
        """
        with self._lock:
            turns = _normalise(messages)
            if not turns:
                return (None, 0)
            conv = self._find_match_locked(turns)
            if conv is not None:
                new = turns[len(conv.norm):]
                for turn in new:
                    self._append_locked(conv, turn)
                conv.updated_at = time.time()
                self._persist_conv_locked(conv)
                return (conv.id, len(new))
            if create_if_missing:
                conv = self._new_conversation_locked(turns, model)
                return (conv.id, len(turns))
            return (None, 0)

    def end_response(
        self,
        conv_id: Optional[str],
        content: str,
        reasoning: str = "",
        tool_calls: str = "",
        model: str = "",
        completion_id: str = "",
    ) -> None:
        """Persist the assistant turn for a request begun by
        :meth:`resolve_conversation`."""
        if not conv_id:
            return
        with self._lock:
            conv = self._convs.get(conv_id)
            if conv is None:
                return
            if not content and not reasoning and not tool_calls:
                return  # empty response: nothing to remember
            turn = Turn(
                role="assistant",
                content=content,
                reasoning=reasoning,
                tool_calls=tool_calls,
                model=model,
                completion_id=completion_id,
                ts=time.time(),
            )
            self._append_locked(conv, turn)
            conv.updated_at = turn.ts
            self._persist_conv_locked(conv)

    def extends_existing_conversation(self, messages: list) -> bool:
        """True if ``messages`` strictly extends one stored conversation.

        Used to record *headerless* requests that are genuine continuations
        while ignoring internal single-prompt calls (title generation), which
        match nothing.
        """
        with self._lock:
            turns = _normalise(messages)
            if not turns:
                return False
            pairs = [(t.role, t.content) for t in turns]
            for conv in self._convs.values():
                n = len(conv.norm)
                if 0 < n < len(pairs) and conv.norm == pairs[:n]:
                    return True
        return False

    def build_seed(self) -> dict:
        """Return the WebUI seed payload for every stored conversation."""
        with self._lock:
            convs = [
                {
                    "id": c.id,
                    "name": c.name,
                    "created_at": c.created_at,
                    "updated_at": c.updated_at,
                    "first_user": c.first_user,
                    "turns": [
                        {
                            "role": t.role,
                            "content": t.content,
                            "reasoning": t.reasoning,
                            "tool_calls": t.tool_calls,
                            "model": t.model,
                            "completion_id": t.completion_id,
                            "tool_call_id": t.tool_call_id,
                            "ts": t.ts,
                        }
                        for t in c.turns
                    ],
                }
                for c in self._convs.values()
            ]
        return build_seed_payload(convs)

    def stats(self) -> dict:
        with self._lock:
            total_turns = sum(len(c.turns) for c in self._convs.values())
            return {
                "conversations": len(self._convs),
                "turns": total_turns,
                "storage": self._storage.counts(),
            }

    # -- internals (caller holds the lock) ----------------------------------
    def _find_match_locked(self, turns: List[Turn]) -> Optional[Conversation]:
        pairs = [(t.role, t.content) for t in turns]
        convs = self._convs.values()
        if len(self._convs) > self._max:
            convs = sorted(self._convs.values(), key=lambda c: -c.updated_at)[: self._max]
        best: Optional[Conversation] = None
        for conv in convs:
            n = len(conv.norm)
            if n == 0 or n > len(pairs):
                continue
            if conv.norm == pairs[:n]:
                if best is None or conv.updated_at > best.updated_at:
                    best = conv
        return best

    def _new_conversation_locked(self, turns: List[Turn], model: Optional[str]) -> Conversation:
        first_user = next((t.content for t in turns if t.role == "user"), "")
        conv = Conversation(
            id=str(uuid.uuid4()),
            name=_derive_name(first_user),
            created_at=time.time(),
            updated_at=time.time(),
            first_user=first_user,
        )
        self._convs[conv.id] = conv
        for turn in turns:
            self._append_locked(conv, turn)
        self._persist_conv_locked(conv)
        return conv

    def _append_locked(self, conv: Conversation, turn: Turn) -> None:
        conv.turns.append(turn)
        conv.norm.append((turn.role, turn.content))
        seq = len(conv.turns) - 1
        payload = self._vault.encrypt_obj(turn.as_payload())
        self._storage.upsert_message(
            conv.id, seq, turn.role, turn.ts, turn.model or None, payload
        )

    def _persist_conv_locked(self, conv: Conversation) -> None:
        self._storage.upsert_conversation(
            conv.id, conv.name, conv.created_at, conv.updated_at, conv.first_user
        )


# -- normalisation helpers ---------------------------------------------------
def _content_text(content) -> str:
    """Reduce an OpenAI content field (string or part list) to display text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def _tool_calls_json(tool_calls) -> str:
    if not tool_calls:
        return ""
    try:
        return json.dumps(tool_calls, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError):
        return ""


def _normalise(messages) -> List[Turn]:
    turns: List[Turn] = []
    if not isinstance(messages, list):
        return turns
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in _ROLES:
            continue
        turns.append(
            Turn(
                role=role,
                content=_content_text(msg.get("content")),
                ts=time.time(),
                reasoning=msg.get("reasoning_content") or "",
                tool_calls=_tool_calls_json(msg.get("tool_calls")),
                tool_call_id=msg.get("tool_call_id") or "",
                model=msg.get("model") or "",
            )
        )
    return turns


def _derive_name(first_user: str) -> str:
    for line in (first_user or "").splitlines():
        line = line.strip()
        if line:
            name = line
            if len(name) > 60:
                name = name[:60].rstrip() + "\u2026"
            return name
    return "New chat"
