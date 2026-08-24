"""HTTP GET /health during the websockets handshake (same PORT as WSS).

websockets 12: process_request is ``(path, headers)`` and the body is bytes.
websockets 13+: process_request is ``(connection, request)`` and
``connection.respond(status, text)`` encodes ``text`` — it must be str.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import Any

HEALTH_BODY_TEXT = '{"status":"ok"}\n'
_HEALTH_PATHS = ('/health', '/healthz')


def _path_of(*args: Any) -> str:
    if len(args) == 2 and not hasattr(args[0], 'respond'):
        return str(args[0] or '')
    request = args[1]
    return str(getattr(request, 'path', '') or '')


def is_health_path(path: str) -> bool:
    return path.split('?', 1)[0] in _HEALTH_PATHS


def health_http_result(*args: Any) -> Any:
    """Return a handshake response for /health, or None to continue the WS upgrade."""
    path = _path_of(*args)
    if not is_health_path(path):
        return None
    if len(args) == 2 and not hasattr(args[0], 'respond'):
        return HTTPStatus.OK, [], HEALTH_BODY_TEXT.encode()
    connection = args[0]
    return connection.respond(HTTPStatus.OK, HEALTH_BODY_TEXT)
