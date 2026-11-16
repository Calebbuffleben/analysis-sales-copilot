"""Personal meeting assistant: captions, turn summaries, and one recap."""

from __future__ import annotations

from typing import Any

from src.modules.text_analysis.meeting_assistant import (
    MeetingAssistant,
    _parse_recap_payload,
    speaker_for,
)
from src.modules.text_analysis.types import TranscriptionChunk


def _chunk(text: str, *, role: str = 'participant', track: str = 'tab-audio') -> TranscriptionChunk:
    return TranscriptionChunk(
        meeting_id='meet-1',
        participant_id='p1',
        track=track,
        text=text,
        confidence=0.9,
        timestamp_ms=2_000,
        window_start_ms=1_000,
        window_end_ms=2_000,
        tenant_id='tenant-1',
        participant_role=role,
    )


def _assistant() -> tuple[MeetingAssistant, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    turns: list[dict[str, Any]] = []
    recaps: list[dict[str, Any]] = []

    def broadcast(tenant_id: str, meeting_id: str, message: dict[str, Any]) -> bool:
        events.append({'tenantId': tenant_id, 'meetingId': meeting_id, **message})
        return True

    assistant = MeetingAssistant(
        broadcast=broadcast,
        translate=lambda text: f'PT:{text}',
        summarize=lambda text: f'RES:{text}',
        recap=lambda rows: {
            'summary': 'Reunião sobre o lançamento.',
            'decisions': ['Lançar na sexta'],
            'highlights': ['Orçamento aprovado'],
            'actionItems': [{'task': 'Enviar a pauta', 'owner': 'Ana', 'due': 'sexta'}],
            'unavailable': False,
        },
        persist_turn=turns.append,
        persist_recap=recaps.append,
        stable_ms=0,
        recap_quiet_s=0,
    )
    return assistant, events, turns, recaps


def test_speaker_labels_host_mic_as_you() -> None:
    assert speaker_for('host', 'microphone') == 'you'
    assert speaker_for('participant', 'tab-audio') == 'meeting'


def test_stable_partial_is_translated_and_final_adds_summary() -> None:
    assistant, events, turns, recaps = _assistant()
    chunk = _chunk('We should ship on Friday')
    assistant.on_partial('meet-1:p1:tab-audio', chunk, {'turn_start_ms': 1000})
    assistant.on_final('meet-1:p1:tab-audio', chunk, {'turn_start_ms': 1000})

    captions = [event for event in events if event['type'] == 'caption']
    summaries = [event for event in events if event['type'] == 'turn_summary']
    assert captions[0]['payload']['isPartial'] is True
    assert captions[0]['payload']['textPt'] == 'PT:We should ship on Friday'
    assert captions[-1]['payload']['isPartial'] is False
    assert captions[-1]['payload']['textEn'] == 'We should ship on Friday'
    assert summaries[0]['payload']['summaryPt'] == 'RES:We should ship on Friday'
    assert summaries[0]['payload']['unavailable'] is False
    assert turns[0]['textPt'] == 'PT:We should ship on Friday'
    assert turns[0]['summaryPt'].startswith('RES:')
    assert recaps == []


def test_end_meeting_emits_one_recap() -> None:
    assistant, events, _turns, recaps = _assistant()
    chunk = _chunk('Ship Friday', role='host', track='microphone')
    assistant.on_final('meet-1:p1:microphone', chunk, {'turn_start_ms': 1000})
    assistant.request_end('meet-1')
    assistant.request_end('meet-1')

    recap_events = [event for event in events if event['type'] == 'meeting_recap']
    assert len(recap_events) == 1
    payload = recap_events[0]['payload']
    assert payload['summary'] == 'Reunião sobre o lançamento.'
    assert payload['decisions'] == ['Lançar na sexta']
    assert payload['actionItems'][0]['owner'] == 'Ana'
    assert len(recaps) == 1


def test_summary_failure_keeps_caption() -> None:
    events: list[dict[str, Any]] = []

    def boom(_text: str) -> str:
        raise RuntimeError('down')

    assistant = MeetingAssistant(
        broadcast=lambda _t, _m, message: events.append(message) or True,
        translate=lambda text: f'PT:{text}',
        summarize=boom,
        recap=lambda _rows: {},
        stable_ms=0,
        recap_quiet_s=0,
    )
    assistant.on_final('s', _chunk('Hello team'), {'turn_start_ms': 1})
    caption = next(event for event in events if event['type'] == 'caption')
    summary = next(event for event in events if event['type'] == 'turn_summary')
    assert caption['payload']['textPt'] == 'PT:Hello team'
    assert summary['payload']['unavailable'] is True
    assert summary['payload']['summaryPt'] == ''


def test_parse_recap_payload_normalizes_items() -> None:
    parsed = _parse_recap_payload(
        '{"summary":"Oi","decisions":["A"],"highlights":[],'
        '"action_items":[{"task":"Fazer","owner":"","due":""}]}'
    )
    assert parsed['summary'] == 'Oi'
    assert parsed['actionItems'][0]['task'] == 'Fazer'
    assert parsed['unavailable'] is False
