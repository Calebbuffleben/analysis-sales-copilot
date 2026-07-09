"""JWT validation for the direct desktop WebSocket gateway.

Validates the SAME access tokens issued by the NestJS backend
(`AuthJwtService`): RS256 with the shared public key in production, or
HS256 with the shared secret in development. Claims contract mirrors
`backend/src/auth/jwt.service.ts`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


class WsAuthError(Exception):
    """Raised when a desktop WS connection fails authentication."""


@dataclass(frozen=True)
class WsAuthContext:
    """Verified identity attached to an accepted WS connection."""

    user_id: str
    tenant_id: str
    role: str


class DesktopWsAuthenticator:
    """Verifies backend-issued access JWTs (RS256 prod / HS256 dev)."""

    def __init__(
        self,
        *,
        jwt_public_key: Optional[str] = None,
        jwt_secret: Optional[str] = None,
        issuer: str = 'meet-backend',
        audience: str = 'meet-platform',
        require_auth: bool = True,
    ) -> None:
        self._issuer = issuer
        self._audience = audience
        self._require_auth = require_auth
        self._algorithm: Optional[str] = None
        self._verify_key: Optional[str] = None

        public_key = self._normalize_pem(jwt_public_key)
        secret = (jwt_secret or '').strip() or None
        if public_key:
            self._algorithm = 'RS256'
            self._verify_key = public_key
        elif secret:
            self._algorithm = 'HS256'
            self._verify_key = secret

        if require_auth and self._verify_key is None:
            raise ValueError(
                'Desktop WS auth requires JWT_PUBLIC_KEY (RS256) or '
                'JWT_SECRET (HS256, dev only). Set DESKTOP_WS_REQUIRE_AUTH=false '
                'to explicitly disable validation (trusted network only).',
            )

    @staticmethod
    def _normalize_pem(raw: Optional[str]) -> Optional[str]:
        if not raw:
            return None
        # Support escaped newlines (\n) for env-based PEM delivery.
        normalized = raw.replace('\\n', '\n').strip()
        return normalized or None

    def authenticate(
        self,
        token: Optional[str],
        expected_tenant_id: Optional[str],
    ) -> WsAuthContext:
        """Validate the token and cross-check the tenant hint from the URL."""
        if not self._require_auth and self._verify_key is None:
            return WsAuthContext(
                user_id='anonymous',
                tenant_id=expected_tenant_id or '',
                role='UNVERIFIED',
            )

        if not token:
            raise WsAuthError('missing token')

        try:
            import jwt as pyjwt
        except ImportError as exc:  # pragma: no cover
            raise WsAuthError('PyJWT is not installed') from exc

        assert self._verify_key is not None and self._algorithm is not None
        try:
            claims = pyjwt.decode(
                token,
                self._verify_key,
                algorithms=[self._algorithm],
                issuer=self._issuer,
                audience=self._audience,
            )
        except Exception as exc:
            raise WsAuthError(f'JWT verification failed: {exc}') from exc

        token_type = str(claims.get('type') or '')
        if token_type != 'access':
            raise WsAuthError(f'unexpected token type: {token_type!r}')

        subject = str(claims.get('sub') or '')
        tenant_id = str(claims.get('tid') or '')
        if not subject or not tenant_id:
            raise WsAuthError('token missing sub/tid claims')

        if expected_tenant_id and expected_tenant_id != tenant_id:
            raise WsAuthError('tenantId mismatch between URL and token')

        return WsAuthContext(
            user_id=subject,
            tenant_id=tenant_id,
            role=str(claims.get('role') or ''),
        )
