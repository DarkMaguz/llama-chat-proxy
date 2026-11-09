"""SQLite persistence for the (already-encrypted) chat history.

The database stores *structure* in the clear -- conversation ids, roles,
sequence numbers, timestamps and models -- but the actual message text lives
in an encrypted ``payload`` BLOB produced by :class:`~.crypto.Vault`.  This
gives "encrypted at rest" for the sensitive content while keeping the index
and ordering queryable without ever decrypting at the SQL layer.

All methods are safe to call from any thread: a single :class:`threading.Lock`
serialises access and the connection uses ``check_same_thread=False``.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ConversationRow:
    id: str
    name: str
    created_at: float
    updated_at: float
    first_user: str


@dataclass
class MessageRow:
    conv_id: str
    seq: int
    role: str
    ts: float
    model: Optional[str]
    payload: bytes


_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL DEFAULT '',
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL,
    first_user   TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_conversations_updated
    ON conversations (updated_at DESC);

CREATE TABLE IF NOT EXISTS messages (
    conv_id  TEXT    NOT NULL,
    seq      INTEGER NOT NULL,
    role     TEXT    NOT NULL,
    ts       REAL    NOT NULL,
    model    TEXT,
    payload  BLOB    NOT NULL,
    PRIMARY KEY (conv_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_messages_ts ON messages (conv_id, seq);
"""


class Storage:
    """A tiny, locked, single-file SQLite wrapper."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # WAL: readers don't block the writer and the writer doesn't block reads.
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self.init()

    def init(self) -> None:
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # -- conversations -------------------------------------------------------
    def upsert_conversation(
        self,
        conv_id: str,
        name: str,
        created_at: float,
        updated_at: float,
        first_user: str,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO conversations (id, name, created_at, updated_at, first_user)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    updated_at = excluded.updated_at,
                    first_user = excluded.first_user
                """,
                (conv_id, name, created_at, updated_at, first_user),
            )
            self._conn.commit()

    def list_conversations(self) -> List[ConversationRow]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, name, created_at, updated_at, first_user "
                "FROM conversations ORDER BY updated_at DESC"
            )
            return [
                ConversationRow(
                    id=r["id"],
                    name=r["name"],
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                    first_user=r["first_user"],
                )
                for r in cur.fetchall()
            ]

    def conversation(self, conv_id: str) -> Optional[ConversationRow]:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, name, created_at, updated_at, first_user "
                "FROM conversations WHERE id = ?",
                (conv_id,),
            ).fetchone()
        if row is None:
            return None
        return ConversationRow(
            id=row["id"],
            name=row["name"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            first_user=row["first_user"],
        )

    # -- messages ------------------------------------------------------------
    def upsert_message(
        self,
        conv_id: str,
        seq: int,
        role: str,
        ts: float,
        model: Optional[str],
        payload: bytes,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO messages (conv_id, seq, role, ts, model, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(conv_id, seq) DO UPDATE SET
                    role = excluded.role,
                    ts = excluded.ts,
                    model = excluded.model,
                    payload = excluded.payload
                """,
                (conv_id, seq, role, ts, model, payload),
            )
            self._conn.commit()

    def messages_for(self, conv_id: str) -> List[MessageRow]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT conv_id, seq, role, ts, model, payload "
                "FROM messages WHERE conv_id = ? ORDER BY seq",
                (conv_id,),
            )
            return [
                MessageRow(
                    conv_id=r["conv_id"],
                    seq=r["seq"],
                    role=r["role"],
                    ts=r["ts"],
                    model=r["model"],
                    payload=r["payload"],
                )
                for r in cur.fetchall()
            ]

    # -- housekeeping --------------------------------------------------------
    def counts(self) -> dict:
        with self._lock:
            convs = self._conn.execute(
                "SELECT COUNT(*) AS n FROM conversations"
            ).fetchone()["n"]
            msgs = self._conn.execute("SELECT COUNT(*) AS n FROM messages").fetchone()["n"]
        return {"conversations": convs, "messages": msgs}

    def close(self) -> None:
        with self._lock:
            self._conn.close()
