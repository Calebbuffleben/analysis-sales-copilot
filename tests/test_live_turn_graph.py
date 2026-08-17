"""Tests for the checkpoint-free LangGraph wrapper around Gemini Live."""

from __future__ import annotations

import asyncio

from src.modules.text_analysis.live_turn_graph import LiveTurnGraphs


def test_pre_graph_routes_host_to_observe() -> None:
    graphs = LiveTurnGraphs()
    called = False

    def retrieve() -> str:
        nonlocal called
        called = True
        return 'unused'

    result = asyncio.run(
        graphs.run_pre_tool(
            {
                'participant_role': 'host',
                'signal_valid': True,
                'base_nudge': 'base',
                'retrieve_fn': retrieve,
            },
        ),
    )

    assert result['route'] == 'observe'
    assert called is False


def test_pre_graph_retrieves_and_builds_customer_nudge() -> None:
    graphs = LiveTurnGraphs()
    result = asyncio.run(
        graphs.run_pre_tool(
            {
                'participant_role': 'client',
                'signal_valid': True,
                'base_nudge': 'base',
                'retrieve_fn': lambda: 'playbook',
                'prosody_nudge': 'prosody',
            },
        ),
    )

    assert result['route'] == 'customer'
    assert result['nudge'] == 'base\nplaybook\nprosody'
    assert graphs.pre_tool.checkpointer is None
    assert graphs.post_tool.checkpointer is None


def test_pre_graph_suppresses_invalid_signal() -> None:
    graphs = LiveTurnGraphs()
    result = asyncio.run(
        graphs.run_pre_tool(
            {
                'participant_role': 'client',
                'signal_valid': False,
                'base_nudge': 'base',
            },
        ),
    )

    assert result['route'] == 'suppress'
    assert 'nudge' not in result


def test_post_graph_awaits_prosody_and_publishes_once() -> None:
    graphs = LiveTurnGraphs()
    published = []

    async def prosody():
        return {'energy': 'high'}

    def publish(value) -> bool:
        published.append(value)
        return True

    result = asyncio.run(
        graphs.run_post_tool(
            {
                'turn_id': 'turn-1',
                'await_prosody_fn': prosody,
                'publish_fn': publish,
            },
        ),
    )

    assert result['published'] is True
    assert published == [{'energy': 'high'}]
