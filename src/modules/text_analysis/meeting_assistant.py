"""Personal meeting assistant.

English speech is transcribed upstream. This module publishes Portuguese
captions, a short Portuguese sense of each finished turn, and one recap
when the meeting ends.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from typing import Any, Callable, Optional

from .types import TranscriptionChunk

logger = logging.getLogger(__name__)

UNAVAILABLE_CAPTION = 'Tradução indisponível'
UNAVAILABLE_SUMMARY = ''
UNAVAILABLE_RECAP = 'Não foi possível gerar o fechamento da reunião.'

TranslateFn = Callable[[str], str]
SummarizeFn = Callable[[str], str]
RecapFn = Callable[[list[dict[str, Any]]], dict[str, Any]]
BroadcastFn = Callable[[str, str, dict[str, Any]], bool]
PersistTurnFn = Callable[[dict[str, Any]], None]
PersistRecapFn = Callable[[dict[str, Any]], None]


def speaker_for(participant_role: str, track: str) -> str:
    role = (participant_role or '').strip().lower()
    track_name = (track or '').strip().lower()
    if role == 'host' or track_name == 'microphone':
        return 'you'
    return 'meeting'


def _parse_recap_payload(raw: str) -> dict[str, Any]:
    text = (raw or '').strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*```$', '', text).strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError('recap payload is not an object')

    def _strings(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    items: list[dict[str, str]] = []
    raw_items = data.get('action_items')
    if isinstance(raw_items, list):
        for item in raw_items:
            if isinstance(item, str) and item.strip():
                items.append({'task': item.strip(), 'owner': '', 'due': ''})
                continue
            if not isinstance(item, dict):
                continue
            task = str(item.get('task') or '').strip()
            if not task:
                continue
            items.append({
                'task': task,
                'owner': str(item.get('owner') or '').strip(),
                'due': str(item.get('due') or '').strip(),
            })

    return {
        'summary': str(data.get('summary') or '').strip(),
        'decisions': _strings(data.get('decisions')),
        'highlights': _strings(data.get('highlights')),
        'actionItems': items,
        'unavailable': False,
    }


class MeetingAssistant:
    """Turn English transcript events into the assistant UI contract."""

    def __init__(
        self,
        *,
        broadcast: BroadcastFn,
        translate: TranslateFn,
        summarize: SummarizeFn,
        recap: RecapFn,
        persist_turn: Optional[PersistTurnFn] = None,
        persist_recap: Optional[PersistRecapFn] = None,
        stable_ms: int = 400,
        recap_quiet_s: float = 1.0,
    ) -> None:
        self._broadcast = broadcast
        self._translate = translate
        self._summarize = summarize
        self._recap = recap
        self._persist_turn = persist_turn
        self._persist_recap = persist_recap
        self._stable_ms = max(0, int(stable_ms))
        self._recap_quiet_s = max(0.0, float(recap_quiet_s))
        self._lock = threading.Lock()
        self._partials: dict[str, dict[str, Any]] = {}
        self._turns: dict[str, list[dict[str, Any]]] = {}
        self._seq: dict[str, int] = {}
        self._open_streams: dict[str, set[str]] = {}
        self._ending: set[str] = set()
        self._recap_done: set[str] = set()
        self._recap_timers: dict[str, threading.Timer] = {}

    def on_partial(
        self,
        stream_key: str,
        chunk: TranscriptionChunk,
        meta: Optional[dict[str, Any]] = None,
    ) -> None:
        text = (chunk.text or '').strip()
        if not text:
            return
        meta = meta or {}
        turn_id = self._turn_id(stream_key, chunk, meta)
        with self._lock:
            state = self._partials.setdefault(stream_key, {
                'text': '',
                'published_source': '',
                'published_pt': '',
                'timer': None,
                'chunk': chunk,
                'meta': meta,
                'turn_id': turn_id,
            })
            if state['text'] == text:
                return
            state['text'] = text
            state['chunk'] = chunk
            state['meta'] = meta
            state['turn_id'] = turn_id
            timer = state.get('timer')
            if isinstance(timer, threading.Timer):
                timer.cancel()
            if self._stable_ms <= 0:
                state['timer'] = None
                self._publish_stable_partial_locked(stream_key)
                return
            timer = threading.Timer(
                self._stable_ms / 1000.0,
                self._flush_partial,
                args=(stream_key,),
            )
            timer.daemon = True
            state['timer'] = timer
            timer.start()

    def on_final(
        self,
        stream_key: str,
        chunk: TranscriptionChunk,
        meta: Optional[dict[str, Any]] = None,
    ) -> None:
        text = (chunk.text or '').strip()
        if not text:
            return
        meta = dict(meta or {})
        with self._lock:
            state = self._partials.get(stream_key) or {}
            timer = state.get('timer')
            if isinstance(timer, threading.Timer):
                timer.cancel()
            published_source = str(state.get('published_source') or '')
            published_pt = str(state.get('published_pt') or '')
            turn_id = self._turn_id(stream_key, chunk, meta or state.get('meta') or {})
            self._partials.pop(stream_key, None)

        if text == published_source and published_pt:
            text_pt = published_pt
        else:
            text_pt = self._safe_translate(text)
        caption_pt = text_pt or UNAVAILABLE_CAPTION
        speaker = speaker_for(chunk.participant_role, chunk.track)
        seq = self._next_seq(chunk.meeting_id)
        self._emit(chunk.tenant_id, chunk.meeting_id, {
            'type': 'caption',
            'payload': {
                'id': f'cap-{uuid.uuid4()}',
                'speaker': speaker,
                'textPt': caption_pt,
                'textEn': text,
                'isPartial': False,
                'turnId': turn_id,
                'seq': seq,
            },
        })

        summary_pt = self._safe_summarize(text)
        self._emit(chunk.tenant_id, chunk.meeting_id, {
            'type': 'turn_summary',
            'payload': {
                'id': f'sum-{uuid.uuid4()}',
                'turnId': turn_id,
                'speaker': speaker,
                'summaryPt': summary_pt,
                'unavailable': not bool(summary_pt),
            },
        })

        record = {
            'tenantId': chunk.tenant_id,
            'meetingId': chunk.meeting_id,
            'turnId': turn_id,
            'speaker': speaker,
            'textEn': text,
            'textPt': text_pt,
            'summaryPt': summary_pt,
            'seq': seq,
            'startedAtMs': int(chunk.window_start_ms or chunk.timestamp_ms or 0),
            'endedAtMs': int(chunk.window_end_ms or chunk.timestamp_ms or 0),
        }
        with self._lock:
            self._turns.setdefault(chunk.meeting_id, []).append(record)
            ending = chunk.meeting_id in self._ending
        if self._persist_turn is not None:
            try:
                self._persist_turn(record)
            except Exception:
                logger.exception(
                    'assistant turn persist failed | meeting=%s',
                    chunk.meeting_id,
                )
        if ending:
            self._schedule_recap(chunk.meeting_id)

    def note_stream_started(self, meeting_id: str, stream_key: str) -> None:
        if not meeting_id or not stream_key:
            return
        with self._lock:
            self._open_streams.setdefault(meeting_id, set()).add(stream_key)

    def note_stream_ended(self, meeting_id: str, stream_key: str) -> None:
        if not meeting_id:
            return
        should_schedule = False
        with self._lock:
            streams = self._open_streams.setdefault(meeting_id, set())
            streams.discard(stream_key)
            if not streams:
                self._ending.add(meeting_id)
                should_schedule = True
        if should_schedule:
            self._schedule_recap(meeting_id)

    def request_end(self, meeting_id: str) -> None:
        if not meeting_id:
            return
        with self._lock:
            self._ending.add(meeting_id)
        self._schedule_recap(meeting_id)

    def _flush_partial(self, stream_key: str) -> None:
        with self._lock:
            self._publish_stable_partial_locked(stream_key)

    def _publish_stable_partial_locked(self, stream_key: str) -> None:
        state = self._partials.get(stream_key)
        if not state:
            return
        text = str(state.get('text') or '').strip()
        if not text or text == state.get('published_source'):
            return
        chunk = state.get('chunk')
        if not isinstance(chunk, TranscriptionChunk):
            return
        turn_id = str(state.get('turn_id') or self._turn_id(stream_key, chunk, state.get('meta')))
        state['published_source'] = text
        # Release the lock before the model call.
        tenant_id = chunk.tenant_id
        meeting_id = chunk.meeting_id
        speaker = speaker_for(chunk.participant_role, chunk.track)
        self._lock.release()
        try:
            text_pt = self._safe_translate(text)
        finally:
            self._lock.acquire()
        current = self._partials.get(stream_key)
        if current is None or current.get('published_source') != text:
            return
        current['published_pt'] = text_pt
        seq = self._next_seq_locked(meeting_id)
        caption_pt = text_pt or UNAVAILABLE_CAPTION
        self._lock.release()
        try:
            self._emit(tenant_id, meeting_id, {
                'type': 'caption',
                'payload': {
                    'id': f'cap-{uuid.uuid4()}',
                    'speaker': speaker,
                    'textPt': caption_pt,
                    'textEn': text,
                    'isPartial': True,
                    'turnId': turn_id,
                    'seq': seq,
                },
            })
        finally:
            self._lock.acquire()

    def _schedule_recap(self, meeting_id: str) -> None:
        with self._lock:
            if meeting_id in self._recap_done:
                return
            existing = self._recap_timers.get(meeting_id)
            if isinstance(existing, threading.Timer):
                existing.cancel()
            if self._recap_quiet_s <= 0:
                self._recap_timers.pop(meeting_id, None)
                self._run_recap_locked(meeting_id)
                return
            timer = threading.Timer(
                self._recap_quiet_s,
                self._run_recap,
                args=(meeting_id,),
            )
            timer.daemon = True
            self._recap_timers[meeting_id] = timer
            timer.start()

    def _run_recap(self, meeting_id: str) -> None:
        with self._lock:
            self._recap_timers.pop(meeting_id, None)
            self._run_recap_locked(meeting_id)

    def _run_recap_locked(self, meeting_id: str) -> None:
        if meeting_id in self._recap_done:
            return
        if meeting_id not in self._ending:
            return
        turns = list(self._turns.get(meeting_id) or [])
        if not turns:
            return
        self._recap_done.add(meeting_id)
        tenant_id = str(turns[-1].get('tenantId') or '')
        self._lock.release()
        try:
            try:
                payload = self._recap([
                    {
                        'speaker': turn['speaker'],
                        'textEn': turn['textEn'],
                    }
                    for turn in turns
                ])
            except Exception:
                logger.exception('assistant recap failed | meeting=%s', meeting_id)
                payload = {
                    'summary': UNAVAILABLE_RECAP,
                    'decisions': [],
                    'highlights': [],
                    'actionItems': [],
                    'unavailable': True,
                }
            payload = {
                'summary': str(payload.get('summary') or UNAVAILABLE_RECAP),
                'decisions': list(payload.get('decisions') or []),
                'highlights': list(payload.get('highlights') or []),
                'actionItems': list(payload.get('actionItems') or payload.get('action_items') or []),
                'unavailable': bool(payload.get('unavailable')),
            }
            self._emit(tenant_id, meeting_id, {
                'type': 'meeting_recap',
                'payload': {
                    'id': f'recap-{uuid.uuid4()}',
                    **payload,
                },
            })
            if self._persist_recap is not None:
                try:
                    self._persist_recap({
                        'tenantId': tenant_id,
                        'meetingId': meeting_id,
                        **payload,
                    })
                except Exception:
                    logger.exception(
                        'assistant recap persist failed | meeting=%s',
                        meeting_id,
                    )
        finally:
            self._lock.acquire()

    def _safe_translate(self, text: str) -> str:
        try:
            return (self._translate(text) or '').strip()
        except Exception:
            logger.exception('assistant translation failed')
            return ''

    def _safe_summarize(self, text: str) -> str:
        try:
            return (self._summarize(text) or '').strip()
        except Exception:
            logger.exception('assistant turn summary failed')
            return UNAVAILABLE_SUMMARY

    def _emit(self, tenant_id: str, meeting_id: str, message: dict[str, Any]) -> None:
        if not tenant_id or not meeting_id:
            return
        try:
            self._broadcast(tenant_id, meeting_id, message)
        except Exception:
            logger.exception(
                'assistant broadcast failed | meeting=%s | type=%s',
                meeting_id,
                message.get('type'),
            )

    def _next_seq(self, meeting_id: str) -> int:
        with self._lock:
            return self._next_seq_locked(meeting_id)

    def _next_seq_locked(self, meeting_id: str) -> int:
        seq = int(self._seq.get(meeting_id, 0)) + 1
        self._seq[meeting_id] = seq
        return seq

    @staticmethod
    def _turn_id(
        stream_key: str,
        chunk: TranscriptionChunk,
        meta: Optional[dict[str, Any]],
    ) -> str:
        meta = meta or {}
        start = meta.get('turn_start_ms') or chunk.window_start_ms or chunk.timestamp_ms
        return f'{stream_key}:{int(start or 0)}'


def build_gemini_assistant_fns(
    *,
    api_key: str,
    model_name: str,
) -> tuple[TranslateFn, SummarizeFn, RecapFn]:
    """Portuguese translation, turn summary, and meeting recap via Gemini."""

    def _generate(prompt: str, *, json_mode: bool) -> str:
        from .gemini_transport import generate_with_transport_chain

        text, _mode = generate_with_transport_chain(
            api_key=api_key,
            model_name=model_name,
            prompt=prompt,
            json_mode=json_mode,
        )
        return text or ''

    def translate(text: str) -> str:
        if not api_key:
            return ''
        return _generate(
            'Traduza o texto a seguir para português do Brasil. '
            'Mantenha o sentido, os nomes e os números. '
            'Responda só com a tradução, sem aspas e sem comentários.\n\n'
            f'{text}',
            json_mode=False,
        )

    def summarize(text: str) -> str:
        if not api_key:
            return ''
        return _generate(
            'Explique em português do Brasil, em 1 a 3 frases, o sentido do que '
            'foi dito. Não traduza palavra por palavra e não dê conselhos. '
            'Não invente fatos. Responda só com o resumo.\n\n'
            f'{text}',
            json_mode=False,
        )

    def recap(turns: list[dict[str, Any]]) -> dict[str, Any]:
        if not api_key:
            return {
                'summary': UNAVAILABLE_RECAP,
                'decisions': [],
                'highlights': [],
                'actionItems': [],
                'unavailable': True,
            }
        lines = []
        for turn in turns:
            speaker = 'Você' if turn.get('speaker') == 'you' else 'Reunião'
            lines.append(f"{speaker}: {turn.get('textEn') or ''}")
        transcript = '\n'.join(lines)
        raw = _generate(
            'Você recebe a transcrição em inglês de uma reunião. '
            'Responda em JSON com as chaves summary (string em português), '
            'decisions (lista de strings em português: o que foi definido), '
            'highlights (lista de strings em português: informações importantes) '
            'e action_items (lista de objetos com task, owner e due, em português; '
            'owner e due vazios quando a reunião não disser). '
            'Não invente decisões nem tarefas.\n\n'
            f'{transcript}',
            json_mode=True,
        )
        try:
            return _parse_recap_payload(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            logger.exception('assistant recap JSON parse failed')
            return {
                'summary': UNAVAILABLE_RECAP,
                'decisions': [],
                'highlights': [],
                'actionItems': [],
                'unavailable': True,
            }

    return translate, summarize, recap


class HttpAssistantStore:
    """Persist turns and the meeting recap through the Nest internal API."""

    def __init__(
        self,
        *,
        base_url: str,
        bootstrap_key: str = '',
        get_token: Optional[Callable[[], str]] = None,
        timeout_seconds: float = 5.0,
    ) -> None:
        self._base_url = base_url.rstrip('/')
        self._bootstrap_key = bootstrap_key.strip()
        self._get_token = get_token
        self._timeout_seconds = timeout_seconds

    def save_turn(self, record: dict[str, Any]) -> None:
        self._post('/internal/assistant/turns', record)

    def save_recap(self, record: dict[str, Any]) -> None:
        self._post('/internal/assistant/recap', record)

    def _post(self, path: str, payload: dict[str, Any]) -> None:
        import requests

        headers = {'Content-Type': 'application/json'}
        if self._bootstrap_key:
            headers['x-service-bootstrap-key'] = self._bootstrap_key
        elif self._get_token is not None:
            token = self._get_token()
            if token:
                headers['Authorization'] = f'Bearer {token}'
        response = requests.post(
            f'{self._base_url}{path}',
            json=payload,
            headers=headers,
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()


def build_meeting_assistant(
    *,
    api_key: str,
    model_name: str,
    broadcast: BroadcastFn,
    base_url: str = '',
    bootstrap_key: str = '',
    get_token: Optional[Callable[[], str]] = None,
    stable_ms: int = 400,
) -> MeetingAssistant:
    translate, summarize, recap = build_gemini_assistant_fns(
        api_key=api_key,
        model_name=model_name,
    )
    store: Optional[HttpAssistantStore] = None
    if base_url and (bootstrap_key or get_token is not None):
        store = HttpAssistantStore(
            base_url=base_url,
            bootstrap_key=bootstrap_key,
            get_token=get_token,
        )
    return MeetingAssistant(
        broadcast=broadcast,
        translate=translate,
        summarize=summarize,
        recap=recap,
        persist_turn=store.save_turn if store else None,
        persist_recap=store.save_recap if store else None,
        stable_ms=stable_ms,
    )
