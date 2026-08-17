"""Minimal HTTP /health when the desktop WS gateway is not bound to PORT."""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

logger = logging.getLogger(__name__)


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split('?', 1)[0]
        if path in ('/health', '/healthz'):
            body = json.dumps({'status': 'ok'}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


def start_health_http(host: str, port: int) -> Optional[HTTPServer]:
    """Start a daemon-thread HTTP server for Cloud Run probes. Returns the server."""
    try:
        server = HTTPServer((host, port), _HealthHandler)
    except OSError as exc:
        logger.warning('Health HTTP bind failed | %s:%s | %s', host, port, exc)
        return None

    thread = threading.Thread(
        target=server.serve_forever,
        name='health-http',
        daemon=True,
    )
    thread.start()
    logger.info('Health HTTP listening | http://%s:%s/health', host, port)
    return server
