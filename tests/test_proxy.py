import json

from llama_chat_proxy.proxy import (
    SseCapture,
    filter_request_headers,
    filter_response_headers,
    merge_tool_calls,
    parse_nonstream_assistant,
    upstream_url,
)
from llama_chat_proxy.config import Config


def _cfg():
    return Config(key_hex="ab" * 64, upstream="http://127.0.0.1:9931/")


def test_upstream_url_strips_and_keeps_query():
    cfg = _cfg()
    assert upstream_url(cfg, "/v1/chat/completions", "") == "http://127.0.0.1:9931/v1/chat/completions"
    assert (
        upstream_url(cfg, "/stream", "conv_id=abc&from=10")
        == "http://127.0.0.1:9931/stream?conv_id=abc&from=10"
    )


def test_sse_capture_accumulates_across_chunk_boundaries():
    cap = SseCapture()
    events = [
        '{"id":"cmpl-1","model":"llama","choices":[{"delta":{"role":"assistant"}}]}',
        '{"id":"cmpl-1","choices":[{"delta":{"content":"Hel"}}]}',
        '{"id":"cmpl-1","choices":[{"delta":{"content":"lo"}}]}',
        '{"id":"cmpl-1","choices":[{"delta":{"reasoning_content":"thinking"}}]}',
        '{"id":"cmpl-1","choices":[{"delta":{"finish_reason":"stop"}}]}',
        "[DONE]",
    ]
    stream = "".join("data: %s\n\n" % e for e in events).encode()
    # Feed byte-by-byte to force awkward splits mid-line and mid-UTF8.
    for i in range(len(stream)):
        cap.feed(stream[i : i + 1])
    content, reasoning, tool_calls, model, completion_id = cap.result()
    assert content == "Hello"
    assert reasoning == "thinking"
    assert model == "llama"
    assert completion_id == "cmpl-1"
    assert tool_calls == ""


def test_sse_capture_merges_tool_calls():
    cap = SseCapture()
    # The tool-call `arguments` arrive as raw text fragments that only form
    # valid JSON once every fragment has been appended in order. Build each
    # payload with json.dumps so the escaping is explicit and the JSON is
    # guaranteed to be valid for the parser under test.
    payloads = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "tc1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": ""},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": '{"city":'},
                            }
                        ]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": '"Paris"}'},
                            }
                        ]
                    }
                }
            ]
        },
    ]
    for obj in payloads:
        cap.feed(("data: %s\n" % json.dumps(obj)).encode())
    cap.feed(b"data: [DONE]\n")
    _, _, tool_calls, _, _ = cap.result()
    merged = json.loads(tool_calls)
    assert len(merged) == 1
    assert merged[0]["id"] == "tc1"
    assert merged[0]["function"]["name"] == "get_weather"
    assert merged[0]["function"]["arguments"] == '{"city":"Paris"}'


def test_merge_tool_calls_multiple_indexes():
    existing = []
    existing = merge_tool_calls(existing, [{"index": 0, "id": "a", "function": {"name": "f"}}])
    existing = merge_tool_calls(existing, [{"index": 1, "id": "b", "function": {"name": "g"}}])
    assert len(existing) == 2
    assert existing[0]["function"]["name"] == "f"
    assert existing[1]["function"]["name"] == "g"


def test_parse_nonstream_assistant():
    body = json.dumps(
        {
            "id": "cmpl-9",
            "model": "llama",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "full answer",
                        "reasoning_content": "thoughts",
                    },
                }
            ],
        }
    ).encode()
    content, reasoning, tool_calls, model, completion_id = parse_nonstream_assistant(body)
    assert content == "full answer"
    assert reasoning == "thoughts"
    assert model == "llama"
    assert completion_id == "cmpl-9"
    assert tool_calls == ""


def test_parse_nonstream_bad_body():
    assert parse_nonstream_assistant(b"not json") == ("", "", "", "", "")


def test_filter_request_headers():
    headers = [
        ("Host", "localhost:9932"),
        ("Accept-Encoding", "gzip"),
        ("Content-Length", "12"),
        ("Cookie", "lcx-seeded=1"),
        ("Connection", "keep-alive"),
        ("x-conversation-id", "conv-abc"),
        ("Authorization", "Bearer secret"),
        ("User-Agent", "Mozilla"),
    ]
    out = dict(filter_request_headers(headers, "127.0.0.1:9931"))
    assert out["Host"] == "127.0.0.1:9931"
    assert out["Accept-Encoding"] == "gzip"
    assert "Cookie" not in out
    assert "Content-Length" not in out
    assert "Connection" not in out
    # preserved:
    assert out["x-conversation-id"] == "conv-abc"
    assert out["Authorization"] == "Bearer secret"
    assert out["User-Agent"] == "Mozilla"


def test_filter_response_headers_streaming_adds_cache_headers():
    out = filter_response_headers([("Content-Type", "text/event-stream"), ("Content-Length", "99")], streaming=True)
    assert out["Content-Type"] == "text/event-stream"
    assert "Content-Length" not in out
    assert out["Cache-Control"] == "no-cache, no-transform"
    assert out["X-Accel-Buffering"] == "no"
