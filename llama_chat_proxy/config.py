"""Runtime configuration for llama-chat-proxy.

All settings are sourced from environment variables so the process is
stateless on startup and can be launched from a shell, systemd unit, Docker
image or an init script without an on-disk config file.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Mapping, Optional

_HEX128 = re.compile(r"^[0-9a-fA-F]{128}$")


class ConfigError(RuntimeError):
    """Raised when the environment does not describe a usable configuration."""


@dataclass
class Config:
    """Validated, immutable-ish view over the proxy's environment.

    Attributes
    ----------
    key_hex:
        The 128 character hexadecimal database key (64 raw bytes).
    upstream:
        Base URL of the running llama.cpp server, e.g. ``http://127.0.0.1:9931``.
    host/port:
        Where the proxy itself listens.
    db_path:
        Path of the SQLite file that holds the encrypted chat history.
    log_level:
        Python :mod:`logging` level name.
    history_token:
        Optional shared secret that must be supplied (``?token=``) to read the
        history endpoint.  When ``None`` the endpoint is protected only by
        same-origin (no CORS) plus network reachability.
    seed_cookie:
        Name of the cookie the landing page sets to mark that IndexedDB has
        been hydrated on the current origin.
    max_conversations:
        Upper bound on the number of conversations kept in memory for
        prefix matching / seeding.
    """

    key_hex: str
    upstream: str = "http://127.0.0.1:9931"
    host: str = "127.0.0.1"
    port: int = 9932
    db_path: str = "data/llama-chat-proxy.db"
    log_level: str = "INFO"
    history_token: Optional[str] = None
    seed_cookie: str = "lcx-seeded"
    max_conversations: int = 500
    # hop-by-hop / transport headers we must never forward verbatim.
    hop_by_hop = frozenset(
        {
            "connection",
            "keep-alive",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailers",
            "transfer-encoding",
            "upgrade",
        }
    )

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "Config":
        source = os.environ if env is None else env

        key = source.get("LLAMA_PROXY_KEY") or source.get("LLAMA_CHAT_PROXY_KEY") or ""
        cfg = cls(
            key_hex=key,
            upstream=_clean(source.get("LLAMA_UPSTREAM", "http://127.0.0.1:9931")),
            host=source.get("LLAMA_PROXY_HOST", "127.0.0.1"),
            port=_int(source.get("LLAMA_PROXY_PORT", "9932"), "LLAMA_PROXY_PORT"),
            db_path=source.get("LLAMA_PROXY_DB", "data/llama-chat-proxy.db"),
            log_level=source.get("LLAMA_PROXY_LOG_LEVEL", "INFO"),
            history_token=(source.get("LLAMA_PROXY_HISTORY_TOKEN") or None),
            seed_cookie=source.get("LLAMA_PROXY_SEED_COOKIE", "lcx-seeded"),
            max_conversations=_int(
                source.get("LLAMA_PROXY_MAX_CONVERSATIONS", "500"),
                "LLAMA_PROXY_MAX_CONVERSATIONS",
            ),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not self.key_hex:
            raise ConfigError(
                "LLAMA_PROXY_KEY is required: set a 128 character hexadecimal "
                "(64 byte) database key."
            )
        if not _HEX128.match(self.key_hex):
            raise ConfigError(
                "LLAMA_PROXY_KEY must be exactly 128 hexadecimal characters "
                "(64 bytes). Got %d characters." % len(self.key_hex)
            )
        if not self.upstream.startswith(("http://", "https://")):
            raise ConfigError(
                "LLAMA_UPSTREAM must be a full URL, e.g. http://127.0.0.1:9931"
            )
        if not 0 < self.port < 65536:
            raise ConfigError("LLAMA_PROXY_PORT must be in range 1..65535")
        if self.max_conversations < 1:
            raise ConfigError("LLAMA_PROXY_MAX_CONVERSATIONS must be >= 1")


def _clean(value: str) -> str:
    return value.strip().rstrip("/")


def _int(value: str, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError("%s must be an integer, got %r" % (name, value)) from exc
