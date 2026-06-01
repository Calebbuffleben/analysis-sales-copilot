from src.config.settings import Settings
from src.modules.text_analysis.gemini_key_pool import GeminiKeyPool


class FakeAnalyzer:
    def __init__(self, key: str, model: str) -> None:
        self.key = key
        self.model = model

    def analyze(self, text: str, state: dict) -> dict:
        return {
            'direct_feedback': '',
            'confidence': 0.0,
            'feedback_type': None,
            'conversation_state': state,
        }


def _fake_factory(key: str, model: str) -> FakeAnalyzer:
    return FakeAnalyzer(key, model)


def test_routes_same_tenant_to_same_slot() -> None:
    settings = Settings(
        gemini_api_keys=('key-a', 'key-b', 'key-c'),
        gemini_model='gemini-test',
    )
    pool = GeminiKeyPool.from_settings(settings, analyzer_factory=_fake_factory)

    first = pool.resolve_slot('tenant-alpha')
    second = pool.resolve_slot('tenant-alpha')

    assert first is second


def test_single_gemini_api_key_is_backwards_compatible() -> None:
    settings = Settings(
        gemini_api_key='single-key',
        gemini_api_keys=(),
        gemini_model='gemini-test',
    )
    pool = GeminiKeyPool.from_settings(settings, analyzer_factory=_fake_factory)

    assert len(pool.slots) == 1
    assert pool.slots[0].analyzer.key == 'single-key'


def test_rpm_limit_is_enforced_per_slot() -> None:
    settings = Settings(
        gemini_api_keys=('key-a', 'key-b'),
        gemini_model='gemini-test',
        gemini_rpm_limit=2,
        gemini_rpm_window_sec=60.0,
    )
    pool = GeminiKeyPool.from_settings(settings, analyzer_factory=_fake_factory)

    for slot in pool.slots:
        assert slot.try_acquire(now=1000.0)
        assert slot.try_acquire(now=1001.0)
        assert not slot.try_acquire(now=1002.0)

    assert pool.slots[0].try_acquire(now=1061.1)
