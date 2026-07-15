"""Unit tests for playbook resolve + catalog cache (Live path)."""

from __future__ import annotations

import json

from src.modules.playbooks.catalog_cache import PlaybookCatalogCache
from src.modules.playbooks.resolve import (
    format_catalog_for_prompt,
    resolve_playbook_metadata,
)


def test_resolve_interpolates_variables() -> None:
    templates = {
        'preco_vs_concorrente': {
            'key': 'preco_vs_concorrente',
            'title': 'Objeção {{competidor}}',
            'steps': [
                {
                    'id': 's1',
                    'label': 'Comparar',
                    'action': {
                        'type': 'copy_text',
                        'payload': 'Compare com {{competidor}}',
                    },
                },
            ],
        },
    }
    hint = json.dumps(
        {
            'playbook_template_key': 'preco_vs_concorrente',
            'playbook_variables': {'competidor': 'Salesforce'},
        },
    )
    meta = resolve_playbook_metadata(
        templates_by_key=templates,
        playbook_hint_json=hint,
        url_allowlist=set(),
    )
    assert meta is not None
    assert meta['title'] == 'Objeção Salesforce'
    assert meta['steps'][0]['action']['payload'] == 'Compare com Salesforce'


def test_resolve_missing_key_returns_none() -> None:
    assert (
        resolve_playbook_metadata(
            templates_by_key={},
            playbook_hint_json='{"playbook_template_key":"nope"}',
            url_allowlist=set(),
        )
        is None
    )


def test_resolve_drops_open_url_without_allowlist() -> None:
    templates = {
        'link': {
            'key': 'link',
            'title': 'Link',
            'steps': [
                {
                    'id': 'u1',
                    'label': 'Abrir',
                    'action': {
                        'type': 'open_url',
                        'payload': 'https://docs.example.com/x',
                    },
                },
            ],
        },
    }
    hint = '{"playbook_template_key":"link"}'
    assert (
        resolve_playbook_metadata(
            templates_by_key=templates,
            playbook_hint_json=hint,
            url_allowlist=set(),
        )
        is None
    )
    meta = resolve_playbook_metadata(
        templates_by_key=templates,
        playbook_hint_json=hint,
        url_allowlist={'docs.example.com'},
    )
    assert meta is not None
    assert meta['steps'][0]['action']['type'] == 'open_url'


def test_catalog_cache_ttl_and_hot_path() -> None:
    calls = {'n': 0}

    def fetch(_tid: str) -> list[dict]:
        calls['n'] += 1
        return [{'key': 'k1', 'title': 'T', 'steps': []}]

    cache = PlaybookCatalogCache(ttl_sec=60.0, fetch_fn=fetch)
    assert len(cache.get('t1')) == 1
    assert calls['n'] == 1
    assert len(cache.get('t1')) == 1
    assert calls['n'] == 1  # cached
    assert cache.get_by_key('t1', hot_path=True)['k1']['key'] == 'k1'
    assert calls['n'] == 1  # hot_path never fetches


def test_catalog_cache_hot_path_empty_before_warm() -> None:
    cache = PlaybookCatalogCache(
        ttl_sec=60.0,
        fetch_fn=lambda _t: [{'key': 'k', 'title': 'x', 'steps': []}],
    )
    assert cache.get_by_key('t1', hot_path=True) == {}


def test_format_catalog_for_prompt() -> None:
    text = format_catalog_for_prompt(
        [{'key': 'preco_vs_concorrente', 'title': 'Preço'}],
    )
    assert 'preco_vs_concorrente' in text
    assert 'Playbooks do tenant' in text


def test_ws_metadata_playbook_shape_matches_overlay_contract() -> None:
    """Same shape FeedbackHub puts on metadata.playbook for the overlay."""
    templates = {
        'preco_vs_concorrente': {
            'key': 'preco_vs_concorrente',
            'title': 'Preço',
            'steps': [
                {
                    'id': 's1',
                    'label': 'Validar',
                    'action': {'type': 'noop', 'payload': ''},
                },
            ],
        },
    }
    playbook = resolve_playbook_metadata(
        templates_by_key=templates,
        playbook_hint_json=json.dumps(
            {'playbook_template_key': 'preco_vs_concorrente'},
        ),
        url_allowlist=set(),
    )
    assert playbook is not None
    assert playbook['templateKey'] == 'preco_vs_concorrente'
    assert isinstance(playbook['steps'], list)
    assert playbook['steps'][0]['action']['type'] == 'noop'
