"""Realtime coaching provider Protocol — swap Live vendors without touching VAD/graph."""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import Any, AsyncIterator, Literal, Protocol

from .types import SessionContext

ActivityKind = Literal['start', 'end']


class CoachSession(Protocol):
    """One Live connection for a meeting. VAD/LangGraph stay outside."""

    async def send_audio(self, pcm: bytes) -> None: ...

    async def send_activity(self, kind: ActivityKind) -> None: ...

    async def send_text(self, text: str) -> None: ...

    def receive(self) -> AsyncIterator[Any]: ...

    async def ack_tools(self, function_calls: list[Any]) -> None: ...

    def parse_tool_calls(self, response: Any) -> list[tuple[str, dict[str, Any], Any]]: ...


class RealtimeCoachProvider(Protocol):
    name: str

    def open_session(
        self,
        ctx: SessionContext,
    ) -> AbstractAsyncContextManager[CoachSession]: ...
