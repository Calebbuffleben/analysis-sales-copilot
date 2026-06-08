"""Tests for pipeline latency milestone logging."""

from __future__ import annotations

import logging

from src.pipeline_latency import (
    LatencyTraceContext,
    log_assemblyai_partial_stable,
    log_assemblyai_transcript_received,
    log_assemblyai_turn_open_ms,
    log_gemini_prompt_sent,
    log_gemini_response_received,
    log_speech_to_publish_ms,
)


def test_latency_milestones_emit_summary(caplog) -> None:
    caplog.set_level(logging.INFO)
    logger = logging.getLogger('test.pipeline_latency')
    ctx = LatencyTraceContext(
        trace_id='trace123456',
        meeting_id='meet-1',
        participant_id='part-1',
        window_end_ms=1_700_000_000_000,
    )

    log_assemblyai_transcript_received(
        logger,
        ctx,
        stream_key='meet-1:part-1:microphone',
        since_last_audio_ms=420,
        turn_bytes=32_000,
        audio_chunks_sent=50,
        transcript_chars=42,
        last_audio_sent_wall_ms=1_000,
    )
    prompt_sent_ms = log_gemini_prompt_sent(
        logger,
        ctx,
        prompt_chars=1200,
        speaker_role='client',
    )
    log_gemini_response_received(
        logger,
        ctx,
        prompt_sent_wall_ms=prompt_sent_ms,
        response_chars=800,
        llm_round_trip_ms=950,
        has_feedback=True,
        confidence=0.82,
    )

    messages = [record.message for record in caplog.records]
    assert any('⏱️ LATENCY' in message and '② assemblyai.transcript_received' in message for message in messages)
    assert any('③ gemini.prompt_sent' in message for message in messages)
    assert any('④ gemini.response_received' in message for message in messages)
    assert any('RESUMO' in message and 'traceId=trace123456' in message for message in messages)
    assert any('"stage":"python.latency.gemini.response_received"' in message for message in messages)


def test_partial_stable_and_speech_to_publish(caplog) -> None:
    caplog.set_level(logging.INFO)
    logger = logging.getLogger('test.pipeline_latency.partial')
    ctx = LatencyTraceContext(
        trace_id='partialtrace1',
        meeting_id='meet-1',
        participant_id='meet-remote',
        window_end_ms=1_700_000_000_000,
    )

    log_assemblyai_partial_stable(
        logger,
        ctx,
        stream_key='meet-1:meet-remote:tab-audio',
        transcript_chars=42,
        completeness_reason='terminal_punctuation',
        turn_open_ms=3500,
    )
    log_speech_to_publish_ms(
        logger,
        ctx,
        partial_stable_wall_ms=ctx.window_end_ms,
    )

    messages = [record.message for record in caplog.records]
    assert any('② assemblyai.partial_stable' in message for message in messages)
    assert any('speech_to_publish' in message for message in messages)


def test_turn_open_ms_logged(caplog) -> None:
    caplog.set_level(logging.INFO)
    logger = logging.getLogger('test.pipeline_latency.turn_open')
    log_assemblyai_turn_open_ms(
        logger,
        meeting_id='meet-1',
        participant_id='meet-remote',
        stream_key='meet-1:meet-remote:tab-audio',
        turn_open_ms=37_500,
        turn_chunks=75,
        turn_audio_ms=37_500,
    )
    assert any(
        'turn_open' in record.message and 'turnOpenMs=37500' in record.message
        for record in caplog.records
    )
