import json
import struct

from src.modules.audio_buffer.sliding_worker import SlidingWindowWorker
from src.modules.text_analysis.gemini_analyzer import GeminiAnalyzer
from src.modules.text_analysis.types import TextAnalysisResult
from src.modules.transcription.transcription_pipeline_service import (
    TranscriptionPipelineService,
)


def test_incremental_window_sends_only_new_audio_plus_overlap() -> None:
    state = {
        'pcm': b'\x01\x00' * 160_000,
        'meta': {
            'sample_rate': 16_000,
            'channels': 1,
            'window_start_ms': 0,
            'window_end_ms': 10_000,
            'participant_role': 'participant',
        },
    }
    emitted = []
    worker = SlidingWindowWorker(
        lambda _: (state['pcm'], state['meta']),
        min_window_seconds=1,
        min_interval_ms=2_000,
        incremental=True,
        overlap_ms=500,
    )
    worker.register_window_callback(lambda _, pcm, meta: emitted.append((pcm, meta)))

    assert worker.on_chunk_appended('m:p:t', 10_000)
    state['meta'] = {**state['meta'], 'window_start_ms': 2_000, 'window_end_ms': 12_000}
    assert worker.on_chunk_appended('m:p:t', 12_000)

    assert len(emitted[0][0]) == 320_000
    assert len(emitted[1][0]) == 80_000  # 2s new audio + 500ms overlap
    assert emitted[1][1]['window_start_ms'] == 9_500


def test_host_uses_slower_window_interval() -> None:
    meta = {
        'sample_rate': 16_000,
        'channels': 1,
        'window_start_ms': 0,
        'window_end_ms': 4_000,
        'participant_role': 'host',
    }
    worker = SlidingWindowWorker(
        lambda _: (b'\x01\x00' * 64_000, meta),
        min_window_seconds=1,
        min_interval_ms=7_000,
        host_min_interval_ms=20_000,
    )
    worker.register_window_callback(lambda *_: None)

    assert worker.on_chunk_appended('m:p:t', 4_000)
    meta['window_end_ms'] = 11_000
    assert not worker.on_chunk_appended('m:p:t', 11_000)
    meta['window_end_ms'] = 24_000
    assert worker.on_chunk_appended('m:p:t', 24_000)


class _FakeModels:
    def __init__(self):
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        payload = {
            'feedback': 'Explore o impacto antes de apresentar a solução.',
            'confidence': 0.9,
            'feedback_type': 'risk',
            'evidence_text': 'Isso está consumindo muito tempo.',
            'estado': {'fase_spin': 'problema'},
            'playbook_template_key': None,
            'playbook_variables': {},
        }
        return type('Response', (), {'text': json.dumps(payload)})()


def test_gemini_analyzer_accepts_wav_and_returns_evidence() -> None:
    models = _FakeModels()
    client = type('Client', (), {'models': models})()
    analyzer = GeminiAnalyzer(api_key='test-key', client=client)

    result = analyzer.analyze_audio(b'RIFFfake-wav', {}, speaker_role='client')

    assert result['direct_feedback']
    assert result['evidence_text'] == 'Isso está consumindo muito tempo.'
    assert models.calls[0]['contents'][1].inline_data.mime_type == 'audio/wav'


class _FakeAudioAnalysis:
    def __init__(self):
        self.analyzed = []
        self.observed = []

    def analyze_audio(self, chunk, wav_bytes):
        self.analyzed.append((chunk, wav_bytes))
        return TextAnalysisResult(direct_feedback='Insight', confidence=0.8), 'Evidência'

    def observe_audio_context(self, chunk, wav_bytes):
        self.observed.append((chunk, wav_bytes))
        return TextAnalysisResult(direct_feedback='', confidence=0.0), 'Contexto'


class _FakeDispatcher:
    def __init__(self):
        self.events = []

    def enqueue(self, event):
        self.events.append(event)
        return True


def _pcm() -> bytes:
    return b''.join(struct.pack('<h', 5_000) for _ in range(16_000))


def _meta(role: str) -> dict:
    return {
        'meeting_id': 'meeting-1',
        'participant_id': f'user-{role}',
        'track': 'tab-audio',
        'sample_rate': 16_000,
        'channels': 1,
        'window_start_ms': 1_000,
        'window_end_ms': 2_000,
        'tenant_id': 'tenant-1',
        'participant_role': role,
    }


def test_multimodal_pipeline_publishes_customer_evidence() -> None:
    analysis = _FakeAudioAnalysis()
    dispatcher = _FakeDispatcher()
    pipeline = TranscriptionPipelineService(
        transcription_service=None,
        text_analysis_service=analysis,
        publish_dispatcher=dispatcher,
        multimodal_audio_enabled=True,
    )

    pipeline.process_window('meeting-1:user-participant:tab-audio', _pcm(), _meta('participant'))

    assert analysis.analyzed[0][1].startswith(b'RIFF')
    assert dispatcher.events[0].transcript_text == 'Evidência'
    assert dispatcher.events[0].analysis.direct_feedback == 'Insight'


def test_multimodal_pipeline_observes_host_without_publish() -> None:
    analysis = _FakeAudioAnalysis()
    dispatcher = _FakeDispatcher()
    pipeline = TranscriptionPipelineService(
        transcription_service=None,
        text_analysis_service=analysis,
        publish_dispatcher=dispatcher,
        multimodal_audio_enabled=True,
    )

    pipeline.process_window('meeting-1:user-host:microphone', _pcm(), _meta('host'))

    assert analysis.observed
    assert dispatcher.events == []
