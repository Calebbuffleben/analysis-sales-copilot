"""Tests for HTTP bootstrap minting of SERVICE JWTs."""

from __future__ import annotations

import os
import sys
import time

import pytest
import requests

SERVICE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SERVICE_ROOT not in sys.path:
    sys.path.insert(0, SERVICE_ROOT)

from src.modules.backend_feedback.service_jwt_provider import ServiceJwtProvider  # noqa: E402


def test_mint_once_then_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    posts = {'n': 0}

    def fake_post(
        self: requests.Session,
        url: str,
        headers: dict | None = None,
        json: dict | None = None,
        timeout: float | None = None,
    ) -> object:
        posts['n'] += 1
        exp_ms = (time.time() + 3600.0) * 1000.0

        class Resp:
            ok = True
            status_code = 200
            text = ''

            def json(self_inner: object) -> dict:
                return {'token': 'jwt-one', 'expiresAt': exp_ms}

            def raise_for_status(self_inner: object) -> None:
                return None

        return Resp()

    monkeypatch.setattr(requests.Session, 'post', fake_post)
    p = ServiceJwtProvider(
        'http://api.example',
        'bootstrap-secret',
        'acme',
        ttl_seconds=3600,
    )
    assert p.get_token() == 'jwt-one'
    assert p.get_token() == 'jwt-one'
    assert posts['n'] == 1


def test_invalidate_forces_remint(monkeypatch: pytest.MonkeyPatch) -> None:
    posts = {'n': 0}

    def fake_post(
        self: requests.Session,
        url: str,
        headers: dict | None = None,
        json: dict | None = None,
        timeout: float | None = None,
    ) -> object:
        posts['n'] += 1
        exp_ms = (time.time() + 3600.0) * 1000.0
        tok = f'jwt-{posts["n"]}'

        class Resp:
            ok = True
            status_code = 200
            text = ''

            def json(self_inner: object) -> dict:
                return {'token': tok, 'expiresAt': exp_ms}

            def raise_for_status(self_inner: object) -> None:
                return None

        return Resp()

    monkeypatch.setattr(requests.Session, 'post', fake_post)
    p = ServiceJwtProvider('http://api.example', 'boot', 'slug', ttl_seconds=3600)
    assert p.get_token() == 'jwt-1'
    p.invalidate()
    assert p.get_token() == 'jwt-2'
    assert posts['n'] == 2
