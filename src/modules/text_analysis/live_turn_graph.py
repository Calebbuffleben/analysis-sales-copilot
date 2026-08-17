"""Thin LangGraph orchestration around the existing Gemini Live turn."""

from __future__ import annotations

import asyncio
import inspect
import time
from typing import Any, Awaitable, Callable, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from ...metrics.realtime_metrics import LANGGRAPH_NODE_MS


class LiveTurnState(TypedDict, total=False):
    """Ephemeral per-turn state; intentionally has no checkpointer."""

    participant_role: str
    signal_valid: bool
    base_nudge: str
    retrieve_fn: Callable[[], str]
    playbook_nudge: str
    prosody_nudge: str
    nudge: str
    route: Literal['customer', 'observe', 'suppress']
    turn_id: str
    args: dict[str, Any]
    await_prosody_fn: Callable[[], Awaitable[Any]]
    publish_fn: Callable[[Any], bool]
    prosody: Any
    published: bool


def _timed(node: str, fn: Callable[[LiveTurnState], dict[str, Any]]):
    def wrapped(state: LiveTurnState) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            return fn(state)
        finally:
            LANGGRAPH_NODE_MS.labels(node=node).observe(
                (time.perf_counter() - started) * 1000.0,
            )

    return wrapped


class LiveTurnGraphs:
    """Pre-tool and post-tool graphs compiled once for the process."""

    def __init__(self) -> None:
        pre = StateGraph(LiveTurnState)
        pre.add_node('gate_role', _timed('gate_role', self._gate_role))
        pre.add_node('gate_signal', _timed('gate_signal', self._gate_signal))
        pre.add_node('retrieve_playbook', _timed('retrieve_playbook', self._retrieve))
        pre.add_node('prepare_nudge', _timed('prepare_nudge', self._prepare_nudge))
        pre.add_edge(START, 'gate_role')
        pre.add_conditional_edges(
            'gate_role',
            self._route_role,
            {'customer': 'gate_signal', 'observe': END},
        )
        pre.add_conditional_edges(
            'gate_signal',
            self._route_signal,
            {'customer': 'retrieve_playbook', 'suppress': END},
        )
        pre.add_edge('retrieve_playbook', 'prepare_nudge')
        pre.add_edge('prepare_nudge', END)
        self.pre_tool = pre.compile()

        post = StateGraph(LiveTurnState)
        post.add_node('merge_prosody', self._merge_prosody)
        post.add_node('publish_primary', self._publish_primary)
        post.add_edge(START, 'merge_prosody')
        post.add_edge('merge_prosody', 'publish_primary')
        post.add_edge('publish_primary', END)
        self.post_tool = post.compile()

    @staticmethod
    def _gate_role(state: LiveTurnState) -> dict[str, Any]:
        role = str(state.get('participant_role') or '').strip().lower()
        return {'route': 'observe' if role == 'host' else 'customer'}

    @staticmethod
    def _route_role(state: LiveTurnState) -> Literal['customer', 'observe']:
        return 'observe' if state.get('route') == 'observe' else 'customer'

    @staticmethod
    def _gate_signal(state: LiveTurnState) -> dict[str, Any]:
        return {'route': 'customer' if state.get('signal_valid', True) else 'suppress'}

    @staticmethod
    def _route_signal(state: LiveTurnState) -> Literal['customer', 'suppress']:
        return 'suppress' if state.get('route') == 'suppress' else 'customer'

    @staticmethod
    def _retrieve(state: LiveTurnState) -> dict[str, Any]:
        retrieve_fn = state.get('retrieve_fn')
        return {'playbook_nudge': retrieve_fn() if retrieve_fn else ''}

    @staticmethod
    def _prepare_nudge(state: LiveTurnState) -> dict[str, Any]:
        parts = [
            str(state.get('base_nudge') or '').strip(),
            str(state.get('playbook_nudge') or '').strip(),
            str(state.get('prosody_nudge') or '').strip(),
        ]
        return {'nudge': '\n'.join(part for part in parts if part)}

    @staticmethod
    async def _merge_prosody(state: LiveTurnState) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            await_fn = state.get('await_prosody_fn')
            if await_fn is None:
                return {'prosody': None}
            result = await_fn()
            if inspect.isawaitable(result):
                result = await result
            return {'prosody': result}
        finally:
            LANGGRAPH_NODE_MS.labels(node='merge_prosody').observe(
                (time.perf_counter() - started) * 1000.0,
            )

    @staticmethod
    async def _publish_primary(state: LiveTurnState) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            publish_fn = state.get('publish_fn')
            if publish_fn is None:
                return {'published': False}
            published = await asyncio.to_thread(publish_fn, state.get('prosody'))
            return {'published': bool(published)}
        finally:
            LANGGRAPH_NODE_MS.labels(node='publish_primary').observe(
                (time.perf_counter() - started) * 1000.0,
            )

    async def run_pre_tool(self, state: LiveTurnState) -> LiveTurnState:
        return await self.pre_tool.ainvoke(state)

    async def run_post_tool(self, state: LiveTurnState) -> LiveTurnState:
        return await self.post_tool.ainvoke(state)
