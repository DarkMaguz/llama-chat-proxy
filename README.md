# llama-chat-proxy

A lightweight, production-ready **reverse proxy for llama.cpp** that sits in
front of a running llama.cpp server and:

* passes OpenAI-compatible chat traffic **straight through** to the browser
  (including streaming SSE), byte-for-byte, so the built-in WebUI is unchanged;
* **persists every chat encrypted at rest** to a local SQLite database —
  giving you durable chat history without modifying the UI;
* lets a **fresh browser on another device resume your old chats** by
  hydrating the WebUI's own `IndexedDB` and handing the page back to
  llama.cpp.

The main feature is **cross-device resume**: start a chat in an incognito
window, close it, open a new incognito window on another machine, and pick up
where you left off.

```
                 ┌─────────────────────────────────────────────────────────┐
 browser ──▶    │  llama-chat-proxy  (this)                                 │
 (WebUI)        │   • reverse proxy  (SSE passthrough, unchanged)          │──▶ llama.cpp
                │   • chat capture   (prefix-match → conversation)         │    127.0.0.1:9931
                │   • encrypt at rest (AES-256-GCM → SQLite)               │
                │   • landing page  (seeds IndexedDB, then reload)         │
                └─────────────────────────────────────────────────────────┘
```

---

## Why a proxy?

The llama.cpp WebUI is a **client-side SvelteKit app**. It stores its chat
history in the *browser* (`IndexedDB` via Dexie, a little in `LocalStorage`)
and talks to the model with **stateless** `POST /v1/chat/completions` calls
(the whole message array is re-sent every turn). There is no server-side
history and no way to "export" a conversation back into a clean IndexedDB.

This proxy fixes exactly that: it observes the same chat traffic the WebUI
already sends, reconstructs the conversations, stores them encrypted on disk,
and — when a new browser shows up — writes them into the WebUI's `IndexedDB`
so the UI "remembers" everything with zero UI changes.

---

## How it works

### 1. Faithful reverse proxy
Every request (static WebUI assets, `/v1/models`, `/props`, `/health`, SSE
replays, etc.) is forwarded to the upstream llama.cpp server and the response
streamed back unmodified. Streaming is done with unbuffered async iteration so
`text/event-stream` tokens reach the browser as they are generated.

### 2. Chat capture (no conversation id required)
Real chat turns are the only messages that matter. The WebUI tags its
streaming turns with an `x-conversation-id` header; internal calls (title
generation, KV pre-encode with `n_predict: 0`) do **not**. The proxy:

* records a turn if it carries `x-conversation-id` (a real generation), or if it
  *extends* an already-stored conversation (a headerless continuation);
* groups turns into conversations by **prefix matching** — a request belongs
  to a conversation when that conversation's stored turns are an exact prefix
  of the request's messages. A request that matches nothing starts a new
  conversation; a retry of the same prefix appends only the assistant reply
  (self-healing).
* accumulates the streamed `content`, `reasoning_content` and `tool_calls`
  from the SSE and stores the assistant turn when the stream ends.

### 3. Encrypted at rest
Each message's text/reasoning/tool-calls are encrypted with **AES-256-GCM**
into a self-contained blob (`magic ‖ nonce ‖ ciphertext ‖ tag`) before being
written to SQLite. The 32-byte AES key is derived from your 128-hex-char
database key via HKDF-SHA256. A wrong key fails authentication with a clear
error instead of silently corrupting data.

### 4. Resume on a fresh device (the landing page)
A browser that has never seen the proxy has no seed cookie, so the proxy
serves a tiny self-contained landing page instead of the WebUI. It:

1. fetches `/proxy/api/history` (decrypted on the server);
2. writes every conversation + message into the `LlamaUi` `IndexedDB` using
   the exact Dexie schema and parent/child tree the WebUI expects;
3. drops a cookie and reloads — the very next request is proxied straight to
   llama.cpp;
4. the WebUI boots, reads `IndexedDB`, and shows your restored conversations
   in the sidebar.

Because seeding is cookie-gated, a returning browser is never re-seeded (so
its own in-progress local chats are never clobbered).

---

## Requirements

* Python **3.9+**
* A running llama.cpp server (default `http://127.0.0.1:9931`)
* ~10 KB of disk per conversation (encrypted)

## Setup

```bash
cd llama-chat-proxy

# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Provide your 128-hex-char database key
export LLAMA_PROXY_KEY="…128 hex chars (64 bytes)…"

# 4. Run the proxy
python -m llama_chat_proxy
```

The proxy listens on `http://127.0.0.1:9932` by default. Point your browser at
that address — **not** directly at the llama.cpp port.

> **Tip:** `uvicorn[standard]` pulls native `uvloop`/`httptools`. If you're on
> a platform without prebuilt wheels, `pip install -r requirements.txt` will
> build them, or swap the line in `requirements.txt` to plain `uvicorn`.

### Run the tests (optional)

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

---

## Configuration (environment variables)

| Variable | Default | Description |
| --- | --- | --- |
| `LLAMA_PROXY_KEY` | — (required) | **128-character hexadecimal** database key (64 bytes). Never printed, never stored. |
| `LLAMA_UPSTREAM` | `http://127.0.0.1:9931` | Base URL of the running llama.cpp server. |
| `LLAMA_PROXY_HOST` | `127.0.0.1` | Bind address. Use `0.0.0.0` to reach it from other machines. |
| `LLAMA_PROXY_PORT` | `9932` | Bind port. |
| `LLAMA_PROXY_DB` | `data/llama-chat-proxy.db` | SQLite file holding the encrypted history. |
| `LLAMA_PROXY_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, … |
| `LLAMA_PROXY_HISTORY_TOKEN` | *(unset)* | If set, `/proxy/api/history` requires `?token=<value>`. Injected into the landing page automatically. |
| `LLAMA_PROXY_SEED_COOKIE` | `lcx-seeded` | Cookie name that marks "IndexedDB hydrated on this origin". |
| `LLAMA_PROXY_MAX_CONVERSATIONS` | `500` | Bounding window for prefix matching (seeding always uses all). |

Example — expose it to a second machine on a trusted network, protected by a
token:

```bash
export LLAMA_PROXY_KEY="<128 hex chars>"
export LLAMA_UPSTREAM="http://127.0.0.1:9931"
export LLAMA_PROXY_HOST="0.0.0.0"
export LLAMA_PROXY_PORT="9932"
export LLAMA_PROXY_HISTORY_TOKEN="<a long random string>"
python -m llama_chat_proxy
```

Then on the other machine, open `http://<this-host>:9932/`.

---

## Using it (the resume flow)

1. **Device A:** open `http://127.0.0.1:9932/`. Because it's a fresh browser,
   you get the landing page. Chat as normal — every turn is captured and
   encrypted.
2. **Close the browser / incognito.**
3. **Device B (or a new incognito window):** open `http://<proxy>:9932/`.
   The landing page fetches your history, writes it into that browser's
   `IndexedDB`, and reloads into the WebUI. Your old conversations appear in
   the sidebar — click one to continue.

> The proxy does **not** modify or replace the WebUI; it only (a) proxies and
> (b) serves the one-time landing page. Everything else is llama.cpp's own UI.

---

## Endpoints

| Method & path | Purpose |
| --- | --- |
| `GET /` (no seed cookie) | Landing page that seeds `IndexedDB` and reloads. |
| `GET /` (with seed cookie) | Proxied to llama.cpp (the WebUI). |
| `*  <anything else>` | Reverse-proxied to llama.cpp (SSE included). |
| `POST /v1/chat/completions` | Proxied **and** recorded (streaming or not). |
| `GET /proxy/health` | Liveness probe (`{"status":"ok"}`). |
| `GET /proxy/api/history` | The decrypted seed payload (same-origin; token if configured). |
| `GET /proxy/api/stats` | Conversation/turn counts. |

`/proxy/api/history` returns:

```jsonc
{
  "count": 2,
  "generated_at": 1733333333.0,
  "conversations": [
    {
      "conv": { "id": "…", "name": "hello", "lastModified": 0, "currNode": "…", "pinned": false },
      "messages": [
        { "id": "…", "convId": "…", "type": "root", "role": "system", "content": "",
          "parent": null, "children": ["…"], "timestamp": 0, "toolCalls": "" },
        { "id": "…", "convId": "…", "type": "text", "role": "user", "content": "hello",
          "parent": "…", "children": ["…"], "timestamp": 1 }
        // …assistant, user, assistant…
      ]
    }
  ]
}
```

— shaped exactly like the WebUI's `LlamaUi` `IndexedDB` stores, ready to be
`put()` into `conversations` / `messages`.

---

## Security notes

* **Encryption at rest** protects the `.db` file on disk. The database key is
  required to read any message; a wrong key yields an authentication error.
* The proxy binds to **loopback by default**. Only expose `0.0.0.0` on a
  network you trust.
* `/proxy/api/history` is **same-origin only** (no CORS headers), so a
  cross-origin website can't read it. If you expose the proxy to other
  machines, set `LLAMA_PROXY_HISTORY_TOKEN` so only the proxy's own landing
  page (and you) can pull decrypted history.
* The database key and history token are secrets — keep them out of shell
  history and version control.

---

## Storage layout

SQLite file (WAL mode). Message **content is encrypted**; structure is
plaintext for ordering/indexing.

```
conversations(id PK, name, created_at, updated_at, first_user)
messages(conv_id, seq, role, ts, model, payload BLOB, PRIMARY KEY (conv_id, seq))
```

`payload` = `AES-256-GCM( magic ‖ nonce ‖ {content, reasoning, tool_calls,
model, completion_id} ‖ tag )`.

---

## Design notes & limitations

* **Linear conversations.** Editing/regenerating a message in the WebUI
  branches it; the proxy records the *new* branch as a fresh conversation
  (it stores what it sees on the wire). The primary use case — start,
  resume, continue — is fully supported.
* **Attachments.** Text content is stored verbatim. Inline images/audio/PDFs
  sent as content parts are reduced to their text for v1 (the text of a
  multimodal turn is preserved; the binary part is not re-attached).
* **Deterministic ids.** Seeded conversations get fresh UUIDs each landing,
  matching how the WebUI itself generates ids. A device that already has local
  chats is never re-seeded (cookie-gated), so nothing is clobbered.
* **No auth on the proxy** by design (it fronts a local model). Rely on
  network trust + the optional history token; put a TLS-terminating reverse
  proxy in front if you expose it beyond your LAN.

## Roadmap ideas

* Multi-modal attachment passthrough (store base64 parts as `extra`).
* Branch-aware capture (record edits/regenerations as forks).
* A `llama-chat-proxy export|import` CLI and a whole-DB SQLCipher option.
