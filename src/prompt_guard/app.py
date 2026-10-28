"""FastAPI application.

/health does not require credentials and returns no configuration.
/v1 requires Authorization: Bearer. Submitted text is never logged.
"""

import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import Message, Receive, Scope, Send

from prompt_guard import CONTRACT_VERSION, SERVICE_NAME
from prompt_guard.auth import credentials_match
from prompt_guard.config import Settings
from prompt_guard.engine import Engine
from prompt_guard.errors import (
    ScanError,
    error_payload,
    internal_error,
    invalid_request,
    not_ready,
    request_too_large,
    unauthorized,
)
from prompt_guard.runtime import load_scanner
from prompt_guard.schemas import ScanRequest, ScanResponse

log = logging.getLogger(__name__)


def create_app(settings: Settings, engine: Engine | None = None) -> FastAPI:
    if engine is None:
        engine = Engine(queue_size=settings.queue_size, loader=lambda: load_scanner(settings))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine.start()
        try:
            yield
        finally:
            app.state.engine.stop()

    app = FastAPI(
        title="Prompt Guard",
        version=CONTRACT_VERSION,
        debug=False,
        lifespan=lifespan,
        description=(
            "Classifies supplied text for prompt injection. "
            "Callers decide what to allow or block. "
            "This service does not fetch URLs, read caller paths, or execute commands."
        ),
    )
    app.state.settings = settings
    app.state.engine = engine
    app.add_middleware(GuardMiddleware, settings=settings)

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"service": SERVICE_NAME, "contract_version": CONTRACT_VERSION}

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "live"}

    @app.get("/health/ready")
    async def ready() -> JSONResponse:
        if app.state.engine.ready:
            return JSONResponse({"status": "ready"})
        return JSONResponse({"status": "not_ready"}, status_code=503)

    @app.get("/v1/model")
    async def model_info(request: Request) -> JSONResponse:
        info = app.state.engine.info
        if info is None:
            raise not_ready()
        body = info.public_dict()
        body["request_id"] = request.state.request_id
        return JSONResponse(body)

    @app.post("/v1/scan", response_model=ScanResponse)
    async def scan(payload: ScanRequest, request: Request) -> ScanResponse:
        started = time.perf_counter()
        deadline = time.monotonic() + (settings.deadline_ms / 1000)
        outcome = await app.state.engine.submit(payload.text, deadline)
        if (
            not outcome.complete
            or outcome.truncated
            or outcome.windows_scanned != outcome.windows_total
        ):
            raise internal_error()
        info = app.state.engine.info
        if info is None:
            raise not_ready()
        threshold = settings.default_threshold if payload.threshold is None else payload.threshold
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return ScanResponse(
            request_id=request.state.request_id,
            status="ok",
            injection_score=outcome.injection_score,
            threshold=threshold,
            flagged=outcome.injection_score >= threshold,
            model={"id": info.model_id, "revision": info.revision},
            coverage={
                "complete": True,
                "truncated": False,
                "windows_scanned": outcome.windows_scanned,
                "windows_total": outcome.windows_total,
            },
            elapsed_ms=elapsed_ms,
        )

    @app.exception_handler(ScanError)
    async def handle_scan_error(request: Request, exc: ScanError) -> JSONResponse:
        log.info(
            "request failed request_id=%s code=%s status=%s",
            _request_id(request),
            exc.code,
            exc.status_code,
        )
        headers = {}
        if exc.status_code == 429:
            headers["Retry-After"] = str(settings.retry_after_seconds)
        return JSONResponse(
            error_payload(_request_id(request), exc),
            status_code=exc.status_code,
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        fields = sorted(
            {
                ".".join(str(part) for part in error.get("loc", ()) if part != "body") or "body"
                for error in exc.errors()
            }
        )
        log.info(
            "invalid request request_id=%s fields=%s",
            _request_id(request),
            ",".join(fields),
        )
        message = "Invalid request fields: " + ", ".join(fields) + "."
        if fields == ["threshold"]:
            message = "threshold must be a finite number between 0 and 1."
        if fields == ["text"]:
            message = "text must be a nonempty string."
        return JSONResponse(
            error_payload(_request_id(request), invalid_request(message)),
            status_code=422,
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        if exc.status_code == 404:
            code = "not_found"
            message = "Not found."
        elif exc.status_code == 405:
            code = "method_not_allowed"
            message = "Method not allowed."
        else:
            code = "http_error"
            message = "Request failed."
        error = ScanError(code, message, exc.status_code, False)
        return JSONResponse(error_payload(_request_id(request), error), status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error request_id=%s", _request_id(request))
        return JSONResponse(
            error_payload(_request_id(request), internal_error()),
            status_code=500,
        )

    return app


class GuardMiddleware:
    """Assign a request id, authenticate /v1, and cap request bodies."""

    def __init__(self, app: Callable[[Scope, Receive, Send], Awaitable[None]], settings: Settings) -> None:
        self.app = app
        self.settings = settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = str(uuid.uuid4())
        state = scope.setdefault("state", {})
        if isinstance(state, dict):
            state["request_id"] = request_id
        path = scope.get("path", "")
        method = scope.get("method", "")

        if path.startswith("/v1/") and not credentials_match(
            _header(scope, "authorization"),
            self.settings.revealed_api_key(),
        ):
            await _send_error(send, request_id, unauthorized())
            return

        if method == "POST" and path == "/v1/scan":
            too_big, body = await _read_limited(receive, self.settings.max_request_bytes, scope)
            if too_big:
                await _send_error(send, request_id, request_too_large())
                return
            receive = _replay(body)

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                names = {key.lower() for key, _value in headers}
                if b"x-request-id" not in names:
                    headers.append((b"x-request-id", request_id.encode("ascii")))
                if b"cache-control" not in names:
                    headers.append((b"cache-control", b"no-store"))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_id)


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", None) or uuid.uuid4())


def _header(scope: Scope, name: str) -> str | None:
    target = name.lower().encode("latin-1")
    for key, value in scope.get("headers", []):
        if key.lower() == target:
            return value.decode("latin-1")
    return None


async def _read_limited(receive: Receive, max_bytes: int, scope: Scope) -> tuple[bool, bytes]:
    declared = _header(scope, "content-length")
    if declared is not None:
        try:
            if int(declared) > max_bytes:
                return True, b""
        except ValueError:
            return True, b""
    chunks: list[bytes] = []
    total = 0
    while True:
        message = await receive()
        if message["type"] != "http.request":
            continue
        chunk = message.get("body", b"")
        total += len(chunk)
        if total > max_bytes:
            return True, b""
        chunks.append(chunk)
        if not message.get("more_body", False):
            return False, b"".join(chunks)


def _replay(body: bytes) -> Receive:
    sent = False

    async def receive() -> Message:
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return receive


async def _send_error(send: Send, request_id: str, error: ScanError) -> None:
    payload = json.dumps(error_payload(request_id, error)).encode("utf-8")
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(payload)).encode("ascii")),
        (b"x-request-id", request_id.encode("ascii")),
        (b"cache-control", b"no-store"),
    ]
    await send({"type": "http.response.start", "status": error.status_code, "headers": headers})
    await send({"type": "http.response.body", "body": payload})
