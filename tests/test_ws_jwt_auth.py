"""Tests for JWT algorithm selection in the direct desktop WS gateway."""

from __future__ import annotations

import time

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from src.ws_gateway.jwt_auth import DesktopWsAuthenticator, WsAuthError


def _claims() -> dict[str, object]:
    now = int(time.time())
    return {
        'sub': 'user-1',
        'tid': 'tenant-1',
        'role': 'MEMBER',
        'type': 'access',
        'iss': 'meet-backend',
        'aud': 'meet-platform',
        'iat': now,
        'exp': now + 300,
    }


def test_uses_hs256_key_when_both_algorithms_are_configured() -> None:
    token = jwt.encode(_claims(), 'shared-secret', algorithm='HS256')
    authenticator = DesktopWsAuthenticator(
        jwt_public_key='unused-for-hs256',
        jwt_secret='shared-secret',
    )

    context = authenticator.authenticate(token, 'tenant-1')

    assert context.user_id == 'user-1'
    assert context.tenant_id == 'tenant-1'


def test_uses_rs256_key_from_token_header() -> None:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    token = jwt.encode(_claims(), private_pem, algorithm='RS256')
    authenticator = DesktopWsAuthenticator(
        jwt_public_key=public_pem.decode(),
        jwt_secret='unused-for-rs256',
    )

    context = authenticator.authenticate(token, 'tenant-1')

    assert context.user_id == 'user-1'


def test_rejects_algorithm_without_a_matching_configured_key() -> None:
    token = jwt.encode(_claims(), 'other-secret', algorithm='HS384')
    authenticator = DesktopWsAuthenticator(jwt_secret='shared-secret')

    with pytest.raises(WsAuthError, match='HS384.*not configured'):
        authenticator.authenticate(token, 'tenant-1')
