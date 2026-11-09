"""Process entry point: build the app and serve it with Uvicorn."""

from __future__ import annotations

import logging
import sys

import uvicorn

from .app import create_app
from .config import Config, ConfigError
from .crypto import Vault
from .db import Storage
from .recorder import Recorder

log = logging.getLogger("llama_chat_proxy")


def build_app(config: Config):
    vault = Vault(config.key_hex)
    storage = Storage(config.db_path)
    recorder = Recorder(storage, vault, config.max_conversations)
    app = create_app(config, recorder)
    # Keep a reference so the DB is closed on interpreter shutdown.
    app.state.storage = storage
    return app


def main(argv=None) -> int:  # pragma: no cover - thin wrapper
    try:
        config = Config.from_env()
    except ConfigError as exc:
        print("llama-chat-proxy: %s" % exc, file=sys.stderr)
        return 2

    level = getattr(logging, config.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s",
        stream=sys.stderr,
    )

    app = build_app(config)
    try:
        uvicorn.run(
            app,
            host=config.host,
            port=config.port,
            log_level=config.log_level.lower(),
            lifespan="auto",
        )
    except KeyboardInterrupt:  # pragma: no cover
        pass
    finally:
        app.state.storage.close()
    return 0
