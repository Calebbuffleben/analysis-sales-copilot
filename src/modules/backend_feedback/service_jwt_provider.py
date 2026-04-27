"""Mint and cache short-lived SERVICE JWTs via the backend HTTP bootstrap endpoint."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

# Refresh before wall-clock expiry so calls rarely hit "jwt expired" on the wire.
_DEFAULT_REFRESH_SKEW_SECONDS = 120


class ServiceJwtProvider:
    """POST /auth/service-token; thread-safe cache with proactive refresh."""

    def __init__(
        self,
        http_base_url: str,
        bootstrap_key: str,
        tenant_slug: str,
        *,
        ttl_seconds: int = 3600,
        refresh_skew_seconds: int = _DEFAULT_REFRESH_SKEW_SECONDS,
        session: Optional[requests.Session] = None,
    ) -> None:
        self._http_base = http_base_url.rstrip('/')
        self._bootstrap_key = bootstrap_key.strip()
        self._tenant_slug = tenant_slug.strip().lower()
        self._ttl_seconds = ttl_seconds
        self._skew = refresh_skew_seconds
        self._session = session or requests.Session()
        self._lock = threading.Lock()
        self._token: Optional[str] = None
        self._refresh_not_after: float = 0.0

    def invalidate(self) -> None:
        with self._lock:
            self._token = None
            self._refresh_not_after = 0.0

    def prewarm(self) -> None:
        """Mint immediately so misconfiguration fails at startup, not on first publish."""
        self.get_token()

    def get_token(self) -> str:
        """Return a valid JWT, minting or refreshing when near expiry."""
        with self._lock:
            now = time.time()
            if self._token and now < self._refresh_not_after:
                return self._token
            self._mint_locked()
            assert self._token is not None
            return self._token

    def _mint_locked(self) -> None:
        url = f'{self._http_base}/auth/service-token'
        try:
            resp = self._session.post(
                url,
                headers={
                    'Content-Type': 'application/json',
                    'x-service-bootstrap-key': self._bootstrap_key,
                },
                json={
                    'tenantSlug': self._tenant_slug,
                    'label': 'python-audio-pipeline',
                    'ttlSeconds': self._ttl_seconds,
                },
                timeout=15.0,
            )
        except requests.RequestException as exc:
            logger.error('service-token mint HTTP failed | url=%s | %s', url, exc)
            raise

        if not resp.ok:
            body_preview = (resp.text or '')[:500]
            logger.error(
                'service-token mint rejected | status=%s | body=%s',
                resp.status_code,
                body_preview,
            )
            resp.raise_for_status()

        data: dict[str, Any] = resp.json()
        token = data.get('token')
        expires_at_ms = data.get('expiresAt')
        if not token or not isinstance(token, str):
            raise RuntimeError('service-token response missing string "token"')
        if not isinstance(expires_at_ms, (int, float)):
            raise RuntimeError('service-token response missing numeric "expiresAt"')

        self._token = token
        exp_s = float(expires_at_ms) / 1000.0
        now = time.time()
        ttl_remaining = max(0.0, exp_s - now)
        # Cap skew so short TTLs (e.g. 60s) still cache for a useful window.
        skew_eff = min(float(self._skew), max(10.0, ttl_remaining - 5.0))
        self._refresh_not_after = exp_s - skew_eff
        logger.info(
            'Minted backend SERVICE JWT via bootstrap | tenant_slug=%s | ttl_requested=%ss',
            self._tenant_slug,
            self._ttl_seconds,
        )
