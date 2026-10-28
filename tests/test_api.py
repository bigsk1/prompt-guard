import asyncio
import math

import httpx
import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport

from tests.conftest import API_KEY, FixedBackend, GateBackend, auth_header, make_app

CANARY = "CANARY-SECRET-PROMPT-do-not-log-9f3a"


def test_health_is_public_and_minimal() -> None:
    app, _backend = make_app()
    with TestClient(app) as client:
        live = client.get("/health/live", headers={"Authorization": "Bearer wrong"})
        ready = client.get("/health/ready")
    assert live.status_code == 200
    assert live.json() == {"status": "live"}
    assert ready.status_code == 200
    assert ready.json() == {"status": "ready"}


def test_not_ready_scan_is_503_without_a_score() -> None:
    app, _backend = make_app()
    with TestClient(app) as client:
        app.state.engine._ready.clear()
        denied = client.get("/health/ready")
        response = client.post("/v1/scan", headers=auth_header(), json={"text": "hello"})
    assert denied.status_code == 503
    assert denied.json() == {"status": "not_ready"}
    assert response.status_code == 503
    assert "flagged" not in response.json()
    assert "injection_score" not in response.json()
    assert response.json()["error"]["code"] == "not_ready"


def test_auth_required_for_v1() -> None:
    app, _backend = make_app()
    with TestClient(app) as client:
        missing = client.post("/v1/scan", json={"text": "hello"})
        wrong = client.get("/v1/model", headers={"Authorization": "Bearer nope"})
        basic = client.post(
            "/v1/scan",
            headers={"Authorization": f"Basic {API_KEY}"},
            json={"text": "hello"},
        )
        ok = client.get("/v1/model", headers=auth_header())
    for response in (missing, wrong, basic):
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "unauthorized"
        assert "flagged" not in response.json()
    assert ok.status_code == 200
    body = ok.json()
    assert body["model"]["revision"] == app.state.settings.model_revision
    assert body["runtime"]["quantization"] == "dynamic-int8"
    assert body["window"]["aggregation"] == "max"
    assert body["labels"] == [{"index": 0, "name": "SAFE"}, {"index": 1, "name": "INJECTION"}]
    assert "api_key" not in body
    assert API_KEY not in ok.text


def test_validation_and_threshold_boundaries() -> None:
    backend = FixedBackend((0.0, math.log(4.0)))
    app, _backend = make_app(backend)
    with TestClient(app) as client:
        empty = client.post("/v1/scan", headers=auth_header(), json={"text": ""})
        bad_threshold = client.post(
            "/v1/scan",
            headers=auth_header(),
            json={"text": "hello", "threshold": 1.5},
        )
        boolean = client.post(
            "/v1/scan",
            headers=auth_header(),
            json={"text": "hello", "threshold": True},
        )
        extra = client.post(
            "/v1/scan",
            headers=auth_header(),
            json={"text": "hello", "url": "http://example.invalid"},
        )
        at = client.post(
            "/v1/scan",
            headers=auth_header(),
            json={"text": "hello", "threshold": 0.8},
        )
        above = client.post(
            "/v1/scan",
            headers=auth_header(),
            json={"text": "hello", "threshold": 0.81},
        )
        zero = client.post(
            "/v1/scan",
            headers=auth_header(),
            json={"text": "hello", "threshold": 0},
        )
        one = client.post(
            "/v1/scan",
            headers=auth_header(),
            json={"text": "hello", "threshold": 1},
        )
    assert empty.status_code == 422
    assert bad_threshold.status_code == 422
    assert boolean.status_code == 422
    assert extra.status_code == 422
    for response in (empty, bad_threshold, boolean, extra):
        assert "flagged" not in response.json()
        assert CANARY not in response.text
    assert at.status_code == 200
    assert at.json()["injection_score"] == pytest.approx(0.8)
    assert at.json()["flagged"] is True
    assert above.json()["flagged"] is False
    assert zero.json()["flagged"] is True
    assert one.json()["flagged"] is False


def test_source_does_not_change_the_score() -> None:
    backend = FixedBackend((0.0, math.log(4.0)))
    app, _backend = make_app(backend)
    with TestClient(app) as client:
        plain = client.post("/v1/scan", headers=auth_header(), json={"text": "hello"})
        sourced = client.post(
            "/v1/scan",
            headers=auth_header(),
            json={
                "text": "hello",
                "source": {"type": "tool_result", "name": "web_crawl", "url": "http://example.invalid"},
            },
        )
    assert plain.json()["injection_score"] == sourced.json()["injection_score"]
    assert backend.calls == 2


def test_long_input_is_covered_and_oversized_input_is_rejected() -> None:
    backend = FixedBackend((math.log(4.0), 0.0))
    app, _backend = make_app(backend, window_size=10, window_overlap=2, max_windows=4, max_input_tokens=100)
    with TestClient(app) as client:
        covered = client.post("/v1/scan", headers=auth_header(), json={"text": "a" * 20})
        too_many = client.post("/v1/scan", headers=auth_header(), json={"text": "b" * 40})
    assert covered.status_code == 200
    body = covered.json()
    assert body["coverage"] == {
        "complete": True,
        "truncated": False,
        "windows_scanned": body["coverage"]["windows_scanned"],
        "windows_total": body["coverage"]["windows_total"],
    }
    assert body["coverage"]["windows_scanned"] == body["coverage"]["windows_total"]
    assert body["coverage"]["windows_total"] > 1
    assert body["flagged"] is False
    assert too_many.status_code == 413
    assert too_many.json()["error"]["code"] == "too_many_windows"
    assert "flagged" not in too_many.json()


def test_request_bytes_are_rejected() -> None:
    backend = FixedBackend()
    app, _backend = make_app(backend, max_request_bytes=80)
    with TestClient(app) as client:
        response = client.post(
            "/v1/scan",
            headers=auth_header(),
            json={"text": "x" * 200},
        )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"
    assert "flagged" not in response.json()
    assert backend.calls == 0


def test_deadline_does_not_return_a_partial_score() -> None:
    backend = FixedBackend(delay=0.3)
    app, _backend = make_app(
        backend,
        window_size=10,
        window_overlap=2,
        deadline_ms=150,
        max_windows=8,
    )
    with TestClient(app) as client:
        response = client.post("/v1/scan", headers=auth_header(), json={"text": "abcdefghijKLMNOPQRST"})
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "scan_timeout"
    assert response.json()["error"]["retryable"] is True
    assert "flagged" not in response.json()
    assert "injection_score" not in response.json()
    assert backend.calls == 1


def test_internal_failure_hides_the_exception() -> None:
    class Boom:
        def predict_logits(self, input_ids, attention_mask, should_stop):
            raise RuntimeError("boom-secret-traceback")

    app, _backend = make_app(Boom())
    with TestClient(app) as client:
        response = client.post("/v1/scan", headers=auth_header(), json={"text": "hello"})
    assert response.status_code == 500
    assert "boom-secret-traceback" not in response.text
    assert "Traceback" not in response.text
    assert "flagged" not in response.json()


def test_submitted_text_is_not_logged(caplog) -> None:
    app, _backend = make_app()
    with TestClient(app) as client:
        client.post(
            "/v1/scan",
            headers=auth_header(),
            json={"text": CANARY, "threshold": 5, "source": {"name": CANARY}},
        )
        client.post("/v1/scan", headers=auth_header(), json={"text": CANARY})
    assert CANARY not in caplog.text


async def test_queue_full_sets_retry_after() -> None:
    backend = GateBackend()
    app, _backend = make_app(backend, queue_size=1, deadline_ms=5000)

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:

            async def post(text: str) -> httpx.Response:
                return await client.post("/v1/scan", headers=auth_header(), json={"text": text})

            first = asyncio.create_task(post("one"))
            assert await asyncio.to_thread(backend.started.wait, 2)
            second = asyncio.create_task(post("two"))
            await asyncio.sleep(0.05)
            rejected = await post("three")
            backend.release.set()
            first_response = await first
            second_response = await second

    assert rejected.status_code == 429
    assert rejected.json()["error"]["code"] == "queue_full"
    assert rejected.headers["retry-after"] == "1"
    assert "flagged" not in rejected.json()
    assert first_response.status_code == 200
    assert second_response.status_code == 200
