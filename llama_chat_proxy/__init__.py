"""llama-chat-proxy.

A lightweight, production-ready reverse proxy for llama.cpp that passes
OpenAI-compatible chat traffic (including streaming SSE) straight through to
the browser while persisting every chat -- encrypted at rest -- to a local
database.  A fresh browser can then hydrate the WebUI's IndexedDB from the
proxy and resume previous conversations, which is the main feature.
"""

__version__ = "0.1.0"
