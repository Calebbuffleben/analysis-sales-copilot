"""GET /health must not crash websockets 13+ handshake (reject encodes str)."""

from __future__ import annotations

from http import HTTPStatus
from types import SimpleNamespace

from src.ws_gateway.health_handshake import HEALTH_BODY_TEXT, health_http_result


def test_health_websockets13_respond_gets_str() -> None:
    seen: list[tuple[object, object]] = []

    class Conn:
        def respond(self, status, text):
            seen.append((status, text))
            if not isinstance(text, str):
                raise AttributeError("'bytes' object has no attribute 'encode'")
            return text.encode()

    result = health_http_result(Conn(), SimpleNamespace(path='/health'))
    assert seen == [(HTTPStatus.OK, HEALTH_BODY_TEXT)]
    assert result == HEALTH_BODY_TEXT.encode()


def test_health_websockets12_tuple_still_bytes() -> None:
    status, headers, body = health_http_result('/health', {})
    assert status is HTTPStatus.OK
    assert headers == []
    assert body == HEALTH_BODY_TEXT.encode()


def test_healthz_and_non_health() -> None:
    status, _, _ = health_http_result('/healthz', {})
    assert status is HTTPStatus.OK
    assert health_http_result('/ws', {}) is None
