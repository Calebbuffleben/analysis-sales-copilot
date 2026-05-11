from src.config.settings import Settings


def test_static_token_disables_auto_jwt_mint() -> None:
    settings = Settings(
        grpc_feedback_enabled=True,
        grpc_feedback_service_token='static-token',
        backend_http_base_url='http://backend:3001',
        service_bootstrap_key='bootstrap-key',
    )

    assert settings.grpc_feedback_wants_auto_jwt() is False
    settings.validate()


def test_auto_jwt_enabled_when_no_static_token() -> None:
    settings = Settings(
        grpc_feedback_enabled=True,
        grpc_feedback_service_token=None,
        backend_http_base_url='http://backend:3001',
        service_bootstrap_key='bootstrap-key',
    )

    assert settings.grpc_feedback_wants_auto_jwt() is True
    settings.validate()
