from src.config.settings import Settings


def test_static_token_disables_auto_jwt_mint() -> None:
    settings = Settings(
        grpc_feedback_enabled=True,
        grpc_feedback_service_token='static-token',
        backend_http_base_url='http://backend:3001',
        service_bootstrap_key='bootstrap-key',
        llm_provider='gemini',
        gemini_api_key='test-key',
    )

    assert settings.grpc_feedback_wants_auto_jwt() is False
    settings.validate()


def test_auto_jwt_enabled_when_no_static_token() -> None:
    settings = Settings(
        grpc_feedback_enabled=True,
        grpc_feedback_service_token=None,
        backend_http_base_url='http://backend:3001',
        service_bootstrap_key='bootstrap-key',
        llm_provider='gemini',
        gemini_api_key='test-key',
    )

    assert settings.grpc_feedback_wants_auto_jwt() is True
    settings.validate()


def test_live_mode_is_default() -> None:
    settings = Settings(
        grpc_feedback_enabled=False,
        llm_provider='gemini',
        gemini_api_key='test-key',
    )
    assert settings.audio_analysis_mode == 'live'
    settings.validate()


def test_multimodal_mode_bypasses_assemblyai_key_requirement() -> None:
    settings = Settings(
        grpc_feedback_enabled=False,
        audio_analysis_mode='multimodal',
        stt_provider='assemblyai',
        assemblyai_api_key=None,
        llm_provider='gemini',
        gemini_api_key='test-key',
    )

    settings.validate()


def test_multimodal_mode_requires_gemini() -> None:
    settings = Settings(
        grpc_feedback_enabled=False,
        audio_analysis_mode='multimodal',
        llm_provider='ollama',
    )

    try:
        settings.validate()
    except ValueError as exc:
        assert 'requires LLM_PROVIDER=gemini' in str(exc)
    else:
        raise AssertionError('Expected multimodal mode with Ollama to fail validation')


def test_gemini_provider_requires_at_least_one_api_key() -> None:
    settings = Settings(
        grpc_feedback_enabled=False,
        stt_provider='local',
        llm_provider='gemini',
        gemini_api_key=None,
        gemini_api_keys=(),
    )

    try:
        settings.validate()
    except ValueError as exc:
        assert 'GEMINI_API_KEYS or GEMINI_API_KEY' in str(exc)
    else:
        raise AssertionError('Expected missing Gemini API keys to fail validation')


def test_gemini_provider_accepts_multi_key_pool() -> None:
    settings = Settings(
        grpc_feedback_enabled=False,
        stt_provider='local',
        llm_provider='gemini',
        gemini_api_keys=(
            'AIzaSyTestKey00000000000000000001',
            'AIzaSyTestKey00000000000000000002',
        ),
        gemini_rpm_limit=12,
        gemini_rpm_window_sec=60.0,
        gemini_key_routing='tenant',
    )

    settings.validate()


def test_effective_gemini_api_keys_splits_comma_in_single_env_var() -> None:
    settings = Settings(
        gemini_api_key='AIzaSyOne,AIzaSyTwo',
        gemini_api_keys=(),
    )
    assert settings.effective_gemini_api_keys() == ('AIzaSyOne', 'AIzaSyTwo')


def test_gemini_validate_accepts_keys_without_format_prefix_check() -> None:
    """Startup does not guess Google key shape; invalid keys fail at API call time."""
    settings = Settings(
        grpc_feedback_enabled=False,
        stt_provider='local',
        llm_provider='gemini',
        gemini_api_keys=('AQ.custom-format-key', 'AIzaSyAlsoValid'),
        gemini_rpm_limit=12,
        gemini_rpm_window_sec=60.0,
        gemini_key_routing='tenant',
    )

    settings.validate()


def test_gemini_api_keys_strips_wrapping_quotes(monkeypatch) -> None:
    monkeypatch.setenv('GEMINI_API_KEYS', '"key-a,key-b"')
    settings = Settings.from_env()
    assert settings.gemini_api_keys == ('key-a', 'key-b')


def test_gemini_api_keys_rejects_empty_entries(monkeypatch) -> None:
    monkeypatch.setenv('GEMINI_API_KEYS', 'key-a,,key-b')

    try:
        Settings.from_env()
    except ValueError as exc:
        assert 'Gemini API key list contains empty entries' in str(exc)
    else:
        raise AssertionError('Expected malformed GEMINI_API_KEYS to fail parsing')
