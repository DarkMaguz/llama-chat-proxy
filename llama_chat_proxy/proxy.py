"""Transport helpers: header filtering, upstream URL building and the
incremental SSE parser used to capture streamed assistant output while the
raw bytes are passed through to the browser byte-for-byte.
"""

from __future__ import annotations

import json
from typing import Iterable, List, Tuple

from .config import Config

# Request headers we rewrite or drop before forwarding to llama.cpp.
_DROP_REQUEST = {
    "host",
    "content-length",
    "accept-encoding",
    "cookie",
    # hop-by-hop
    "connection",
    "keep-alive",
    "transfer-encoding",
    "upgrade",
    "te",
    "trailers",
    "proxy-authorization",
    "proxy-authenticate",
}

# Response headers that are meaningless (or wrong) when we re-stream a body.
_DROP_RESPONSE = {
    "content-length",
    "content-encoding",
    "transfer-encoding",
    "connection",
    "keep-alive",
}


def upstream_host(config: Config) -> str:
    from urllib.parse import urlparse

    netloc = urlparse(config.upstream).netloc
    return netloc or config.upstream


def upstream_url(config: Config, path: str, query: str) -> str:
    url = config.upstream.rstrip("/") + path
    if query:
        url += "?" + query
    return url


def filter_request_headers(headers: Iterable[Tuple[str, str]], host: str) -> List[Tuple[str, str]]:
    """Return the (name, value) pairs to forward for a client request."""
    out: List[Tuple[str, str]] = []
    for name, value in headers:
        if name.lower() in _DROP_REQUEST:
            continue
        out.append((name, value))
    # llama.cpp ships its WebUI only as a gzip page: unless the request's
    # Accept-Encoding advertises "gzip" it answers 415 with "Error: gzip is
    # not supported by this browser". So we request gzip (the client's own
    # value is dropped above). This is safe for streaming: httpx decodes the
    # response transparently (SSE is uncompressed to begin with) and
    # content-encoding is stripped on the way back, so the browser always
    # receives plain bytes.
    out.append(("Host", host))
    out.append(("Accept-Encoding", "gzip"))
    return out


def filter_response_headers(headers: Iterable[Tuple[str, str]], streaming: bool = True) -> dict:
    """Return a dict of response headers to send to the client."""
    out: dict = {}
    for name, value in headers:
        if name.lower() in _DROP_RESPONSE:
            continue
        out[name] = value
    if streaming:
        out.setdefault("Cache-Control", "no-cache, no-transform")
        out.setdefault("X-Accel-Buffering", "no")
    return out


# -- SSE capture -------------------------------------------------------------
def merge_tool_calls(existing: List[dict], deltas: List[dict]) -> List[dict]:
    """Merge OpenAI-style ``tool_calls`` deltas (by index, appending args)."""
    result: List[dict] = []
    for entry in existing:
        if not isinstance(entry, dict):
            continue
        clone = dict(entry)
        fn = entry.get("function")
        clone["function"] = dict(fn) if isinstance(fn, dict) else {}
        result.append(clone)

    for delta in deltas:
        if not isinstance(delta, dict):
            continue
        index = delta.get("index")
        index = index if isinstance(index, int) and index >= 0 else len(result)
        while len(result) <= index:
            result.append({"function": {}})
        target = result[index]
        if delta.get("id"):
            target["id"] = delta["id"]
        if delta.get("type"):
            target["type"] = delta["type"]
        fn = delta.get("function")
        if isinstance(fn, dict):
            target_fn = target.setdefault("function", {})
            if fn.get("name"):
                target_fn["name"] = fn["name"]
            if fn.get("arguments"):
                target_fn["arguments"] = (target_fn.get("arguments") or "") + fn["arguments"]
    return result


class SseCapture:
    """Incrementally parses ``data:`` SSE lines while bytes pass through.

    llama.cpp emits one JSON object per ``data:`` line under
    ``choices[0].delta``; we accumulate ``content`` / ``reasoning_content`` /
    ``tool_calls`` and grab the completion ``id`` and model.  Lines can arrive
    split across TCP chunks, so we buffer until each newline.
    """

    def __init__(self) -> None:
        self._buf = b""
        self.content = ""
        self.reasoning = ""
        self._tool_calls: List[dict] = []
        self.model = ""
        self.completion_id = ""

    def feed(self, data: bytes) -> None:
        self._buf += data
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            self._process_line(line)

    def _process_line(self, raw: bytes) -> None:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            return
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            return
        try:
            obj = json.loads(payload)
        except ValueError:
            return
        if not isinstance(obj, dict):
            return

        choices = obj.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else {}
        delta = choice.get("delta") if isinstance(choice, dict) else None
        if not isinstance(delta, dict):
            delta = {}

        content = delta.get("content")
        if isinstance(content, str):
            self.content += content
        reasoning = delta.get("reasoning_content")
        if isinstance(reasoning, str):
            self.reasoning += reasoning
        tool_calls = delta.get("tool_calls")
        if tool_calls:
            self._tool_calls = merge_tool_calls(self._tool_calls, tool_calls)

        if not self.model:
            model = obj.get("model") or delta.get("model")
            if isinstance(model, str) and model.strip():
                self.model = model
        if not self.completion_id:
            completion_id = obj.get("id")
            if isinstance(completion_id, str):
                self.completion_id = completion_id

    def result(
        self,
    ) -> Tuple[str, str, str, str, str]:
        tool_calls = (
            json.dumps(self._tool_calls, separators=(",", ":"), ensure_ascii=False)
            if self._tool_calls
            else ""
        )
        return self.content, self.reasoning, tool_calls, self.model, self.completion_id


def parse_nonstream_assistant(body: bytes) -> Tuple[str, str, str, str, str]:
    """Extract the assistant message from a non-streaming chat response."""
    try:
        obj = json.loads(body)
    except ValueError:
        return "", "", "", "", ""
    if not isinstance(obj, dict):
        return "", "", "", "", ""
    choices = obj.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    message = choice.get("message") if isinstance(choice, dict) else None
    if not isinstance(message, dict):
        message = {}
    content = message.get("content")
    content = content if isinstance(content, str) else ""
    reasoning = message.get("reasoning_content")
    reasoning = reasoning if isinstance(reasoning, str) else ""
    tool_calls = message.get("tool_calls")
    tool_calls_json = (
        json.dumps(tool_calls, separators=(",", ":"), ensure_ascii=False) if tool_calls else ""
    )
    model = obj.get("model") or message.get("model") or ""
    model = model if isinstance(model, str) else ""
    completion_id = obj.get("id")
    completion_id = completion_id if isinstance(completion_id, str) else ""
    return content, reasoning, tool_calls_json, model, completion_id
