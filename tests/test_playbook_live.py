"""Unit tests for playbook resolve + catalog cache (Live path)."""

from __future__ import annotations

import json

from src.modules.playbooks.catalog_cache import PlaybookCatalogCache
from src.modules.playbooks.resolve import (
    format_catalog_for_prompt,
    resolve_playbook_metadata,
)
from src.modules.playbooks.retrieve import (
    CATALOG_PROMPT_MAX,
    RETRIEVE_MIN_TEMPLATES,
    PlaybookIndex,
    build_retrieve_query,
    format_retrieve_nudge,
    hint_from_emit_feedback_args,
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


def _base_templates() -> list[dict]:
    return [
        {
            'key': 'preco_vs_concorrente',
            'title': 'Objeção preço vs concorrente',
            'description': 'Cliente compara Salesforce preço barato',
            'steps': [{'id': '1', 'label': 'Isolar preço', 'action': {'type': 'noop'}}],
        },
        {
            'key': 'objecao_tempo',
            'title': 'Preciso pensar',
            'description': 'Adia decisão prazo',
            'steps': [{'id': '1', 'label': 'Agendar follow-up', 'action': {'type': 'noop'}}],
        },
        {
            'key': 'spin_situacao',
            'title': 'Descoberta SPIN',
            'description': 'Perguntas de situação',
            'steps': [{'id': '1', 'label': 'Perguntar contexto', 'action': {'type': 'noop'}}],
        },
    ]


def test_retrieve_salesforce_preco_ranks_preco_playbook() -> None:
    index = PlaybookIndex.from_templates(_base_templates())
    hits = index.retrieve('Salesforce preço barato concorrente', k=3)
    assert hits
    assert hits[0].key == 'preco_vs_concorrente'


def test_retrieve_empty_query_returns_empty() -> None:
    index = PlaybookIndex.from_templates(_base_templates())
    assert index.retrieve('', k=3) == []
    assert format_retrieve_nudge([]) == ''


def test_retrieve_nudge_format() -> None:
    index = PlaybookIndex.from_templates(_base_templates())
    hits = index.retrieve('preço Salesforce', k=2)
    text = format_retrieve_nudge(hits)
    assert 'Candidatos de playbook' in text
    assert 'preco_vs_concorrente' in text


def test_top_templates_for_prompt_stable_without_query() -> None:
    many = []
    for i in range(25):
        many.append(
            {
                'key': f'pb_{i:02d}',
                'title': f'Title {i}',
                'description': 'generic',
                'steps': [],
            },
        )
    # Insert distinctive playbook that should rank when queried
    many.append(
        {
            'key': 'preco_vs_concorrente',
            'title': 'Preço',
            'description': 'Salesforce barato orçamento',
            'steps': [{'id': '1', 'label': 'Valor', 'action': {'type': 'noop'}}],
        },
    )
    index = PlaybookIndex.from_templates(many)
    assert index.size == 26
    assert index.size > RETRIEVE_MIN_TEMPLATES
    ranked = index.top_templates_for_prompt('Salesforce preço orçamento', max_items=20)
    assert len(ranked) <= CATALOG_PROMPT_MAX
    keys = [t['key'] for t in ranked]
    assert 'preco_vs_concorrente' in keys
    # Empty query → stable key sort, first 20
    stable = index.top_templates_for_prompt('', max_items=20)
    assert len(stable) == 20
    stable_keys = [t['key'] for t in stable]
    assert stable_keys == sorted(stable_keys)


def test_retrieve_perf_50_templates_under_20ms() -> None:
    import time

    templates = []
    for i in range(50):
        templates.append(
            {
                'key': f'tpl_{i}',
                'title': f'Playbook número {i} vendas objeção',
                'description': f'tema {i} cliente produto',
                'steps': [
                    {
                        'id': 's',
                        'label': f'passo {i}',
                        'action': {'type': 'noop'},
                    },
                ],
            },
        )
    templates[7] = {
        'key': 'preco_vs_concorrente',
        'title': 'Preço vs Salesforce',
        'description': 'concorrente barato orçamento',
        'steps': [{'id': 's', 'label': 'valor', 'action': {'type': 'noop'}}],
    }
    index = PlaybookIndex.from_templates(templates)
    started = time.perf_counter()
    hits = index.retrieve('Salesforce preço barato', k=3)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    assert hits[0].key == 'preco_vs_concorrente'
    assert elapsed_ms < 20.0, f'retrieve took {elapsed_ms:.1f}ms'


def test_retrieve_uses_source_text_excerpt() -> None:
    templates = [
        {
            'key': 'outro',
            'title': 'Outro',
            'description': 'genérico',
            'steps': [],
        },
        {
            'key': 'preco_vs_concorrente',
            'title': 'Preço',
            'description': '',
            'sourceTextExcerpt': (
                'Battle card Salesforce preço barato orçamento concorrente'
            ),
            'steps': [],
        },
    ]
    index = PlaybookIndex.from_templates(templates)
    hits = index.retrieve('Salesforce preço barato', k=2)
    assert hits[0].key == 'preco_vs_concorrente'


def test_format_catalog_includes_excerpt_cap() -> None:
    from src.modules.playbooks.resolve import format_catalog_for_prompt

    text = format_catalog_for_prompt(
        [
            {
                'key': 'preco_vs_concorrente',
                'title': 'Preço',
                'sourceTextExcerpt': 'x' * 500,
            },
        ],
    )
    assert 'preco_vs_concorrente' in text
    assert '|' in text
    # Cap keeps prompt small (latency / token budget).
    assert len(text) < 400


def test_hint_from_emit_feedback_and_query_builder() -> None:
    hint = hint_from_emit_feedback_args(
        {
            'feedback_type': 'objection',
            'evidence_text': 'está caro',
            'estado': {'objecoes_detectadas': ['preco']},
        },
    )
    assert 'objection' in hint
    assert 'caro' in hint
    q = build_retrieve_query(
        context_summary='host falou de pricing',
        retrieve_query_hint=hint,
    )
    assert 'pricing' in q and 'objection' in q
