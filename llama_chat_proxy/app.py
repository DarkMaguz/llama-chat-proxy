"""The ASGI application: a faithful streaming reverse proxy plus the two
proxy-only routes (landing page and history endpoint)."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional
from urllib.parse import quote

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from . import landing
from .config import Config
from .proxy import (
    SseCapture,
    filter_request_headers,
    filter_response_headers,
    parse_nonstream_assistant,
    upstream_host,
    upstream_url,
)
from .recorder import Recorder

log = logging.getLogger(__name__)

# The OpenAI-compatible chat endpoint the WebUI uses for every turn.
CHAT_COMPLETIONS = "/v1/chat/completions"


def _wants_html(headers) -> bool:
    """True only for a real browser navigation (Accept: text/html).

    ``fetch`` from JS sends ``Accept: */*`` which is deliberately *not*
    treated as an HTML request, so API calls never trip the landing page.
    """
    accept = headers.get("accept", "")
    for part in accept.split(","):
        if part.split(";")[0].strip().lower() == "text/html":
            return True
    return False


def _has_seed_cookie(headers, name: str) -> bool:
    cookie = headers.get("cookie", "")
    for part in cookie.split(";"):
        if part.strip().startswith(name + "="):
            return True
    return False


def _should_record(parsed: dict, has_conv_header: bool, recorder: Recorder) -> bool:
    """Decide whether a chat request represents a real, user-facing turn.

    * pre-encode warm-ups (``n_predict: 0``) never record;
    * streaming turns tagged with ``x-conversation-id`` always record;
    * anything else records only if it *extends* an existing conversation,
      which is how internal calls (e.g. title generation, a single user
      prompt with no history) are excluded.
    """
    if not isinstance(parsed, dict):
        return False
    if parsed.get("n_predict") == 0:
        return False
    if has_conv_header:
        return True
    messages = parsed.get("messages")
    if not isinstance(messages, list) or not messages:
        return False
    return recorder.extends_existing_conversation(messages)


def create_app(config: Config, recorder: Recorder) -> Starlette:
    host = upstream_host(config)

    @asynccontextmanager
    async def lifespan(app: Starlette):
        client = httpx.AsyncClient(
            follow_redirects=False,
            http1=True,
            timeout=httpx.Timeout(None, connect=10.0, read=None, write=60.0, pool=30.0),
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=40),
        )
        app.state.client = client
        log.info(
            "llama-chat-proxy ready on http://%s:%d (upstream %s)",
            config.host,
            config.port,
            config.upstream,
        )
        try:
            yield
        finally:
            await client.aclose()

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "service": "llama-chat-proxy"})

    async def history_endpoint(request: Request) -> Response:
        token = request.query_params.get("token")
        if config.history_token:
            provided = token or ""
            if not hmac.compare_digest(provided.encode(), config.history_token.encode()):
                return JSONResponse(
                    {"error": "unauthorized"},
                    status_code=401,
                    headers={"Cache-Control": "no-store"},
                )
        seed = await asyncio.to_thread(recorder.build_seed)
        return JSONResponse(seed, headers={"Cache-Control": "no-store"})

    async def stats_endpoint(request: Request) -> JSONResponse:
        stats = await asyncio.to_thread(recorder.stats)
        return JSONResponse(stats, headers={"Cache-Control": "no-store"})

    def history_url_for() -> str:
        base = "/proxy/api/history"
        if config.history_token:
            return base + "?token=" + quote(config.history_token, safe="")
        return base

    async def _landing_page(request: Request) -> HTMLResponse:
        url = history_url_for()
        return HTMLResponse(landing.landing_html(url, config.seed_cookie))

    async def _handle_chat(request: Request, client: httpx.AsyncClient) -> Response:
        path = request.url.path
        query = request.url.query
        headers = request.headers
        body = await request.body()
        try:
            parsed = json.loads(body) if body else {}
        except ValueError:
            parsed = {}
        if not isinstance(parsed, dict):
            parsed = {}

        model = parsed.get("model")
        is_stream = bool(parsed.get("stream", False))
        pre_encode = parsed.get("n_predict") == 0
        has_conv_header = headers.get("x-conversation-id") is not None
        messages = parsed.get("messages")
        # Runs in a worker thread: the decision may take the recorder's lock.
        record = await asyncio.to_thread(_should_record, parsed, has_conv_header, recorder)

        up_headers = filter_request_headers(headers.items(), host)
        up_req = httpx.Request(
            request.method, upstream_url(config, path, query), content=body, headers=up_headers
        )
        up_resp = await client.send(up_req, stream=True)

        if up_resp.status_code >= 400:
            data = await up_resp.aread()
            await up_resp.aclose()
            return Response(
                content=data,
                status_code=up_resp.status_code,
                headers=filter_response_headers(up_resp.headers.items(), streaming=False),
            )

        conv_id: Optional[str] = None
        if record and not pre_encode:
            conv_id, _appended = await asyncio.to_thread(
                recorder.resolve_conversation, messages, model, has_conv_header
            )

        if is_stream:
            capture = SseCapture()

            async def gen():
                try:
                    async for chunk in up_resp.aiter_bytes():
                        capture.feed(chunk)
                        yield chunk
                    if conv_id:
                        content, reasoning, tool_calls, model_out, completion_id = capture.result()
                        await asyncio.to_thread(
                            recorder.end_response,
                            conv_id,
                            content,
                            reasoning,
                            tool_calls,
                            model_out or (model or ""),
                            completion_id,
                        )
                finally:
                    await up_resp.aclose()

            return StreamingResponse(
                gen(),
                status_code=up_resp.status_code,
                headers=filter_response_headers(up_resp.headers.items(), streaming=True),
            )

        # Non-streaming chat completion.
        data = await up_resp.aread()
        await up_resp.aclose()
        if conv_id:
            content, reasoning, tool_calls, model_out, completion_id = parse_nonstream_assistant(data)
            await asyncio.to_thread(
                recorder.end_response,
                conv_id,
                content,
                reasoning,
                tool_calls,
                model_out or (model or ""),
                completion_id,
            )
        return Response(
            content=data,
            status_code=up_resp.status_code,
            headers=filter_response_headers(up_resp.headers.items(), streaming=False),
        )

    async def _passthrough(request: Request, client: httpx.AsyncClient) -> StreamingResponse:
        path = request.url.path
        query = request.url.query
        headers = request.headers
        body = request.stream() if request.method in ("POST", "PUT", "PATCH") else None
        up_headers = filter_request_headers(headers.items(), host)
        up_req = httpx.Request(
            request.method,
            upstream_url(config, path, query),
            content=body,
            headers=up_headers,
        )
        up_resp = await client.send(up_req, stream=True)

        async def gen():
            try:
                async for chunk in up_resp.aiter_bytes():
                    yield chunk
            finally:
                await up_resp.aclose()

        return StreamingResponse(
            gen(),
            status_code=up_resp.status_code,
            headers=filter_response_headers(up_resp.headers.items(), streaming=True),
        )

    async def catch_all(request: Request) -> Response:
        client: httpx.AsyncClient = request.app.state.client
        method = request.method

        if method == "GET" and _wants_html(request.headers) and not _has_seed_cookie(
            request.headers, config.seed_cookie
        ):
            return await _landing_page(request)

        # The WebUI calls a *relative* URL (./v1/chat/completions), so the real
        # path depends on the current page: "/v1/chat/completions" from "/" but
        # "/chat/v1/chat/completions" from a /chat/[id] route. Match the suffix.
        if method == "POST" and request.url.path.endswith(CHAT_COMPLETIONS):
            return await _handle_chat(request, client)

        return await _passthrough(request, client)

    routes = [
        Route("/proxy/health", health, methods=["GET"]),
        Route("/proxy/api/history", history_endpoint, methods=["GET"]),
        Route("/proxy/api/stats", stats_endpoint, methods=["GET"]),
        Route(
            "/{path:path}",
            catch_all,
            methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
        ),
    ]

    app = Starlette(lifespan=lifespan, routes=routes)
    return app
