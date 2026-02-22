import hashlib
import hmac
from unittest.mock import MagicMock, patch

import grpc
import pytest

from gen.stt.python.v1 import stt_pb2
from stt_server.backend.application.session_manager import (
    CreateSessionConfig,
    CreateSessionHandler,
    SessionFacade,
    SessionRegistry,
)
from stt_server.backend.component import vad_gate
from stt_server.backend.runtime.metrics import Metrics


@pytest.fixture
def mock_servicer_context():
    """Fixture for mock servicer context."""
    context = MagicMock()
    context.abort.side_effect = grpc.RpcError("Aborted")
    context.peer.return_value = ""
    return context


@pytest.fixture
def mock_session_registry():
    """Fixture for mock session registry."""
    return MagicMock(spec=SessionRegistry)


@pytest.fixture
def create_session_handler(mock_session_registry):
    """Fixture for create session handler."""
    mock_model_registry = MagicMock()
    mock_model_registry.get_next_model_id.return_value = "default"
    mock_model_registry.load_model.return_value = None

    supported_languages = MagicMock()
    supported_languages.is_supported.return_value = True

    config = CreateSessionConfig(
        decode_profiles={"default": {}},
        default_decode_profile="default",
        default_language="en",
        language_fix=False,
        default_task="transcribe",
        supported_languages=supported_languages,
        default_vad_silence=0.5,
        default_vad_threshold=0.5,
        require_api_key=False,
        create_session_rps=0.0,
        create_session_burst=0.0,
        max_sessions_per_ip=0,
        max_sessions_per_api_key=0,
    )
    return CreateSessionHandler(
        session_registry=mock_session_registry,
        model_registry=mock_model_registry,
        config=config,
    )


@pytest.fixture
def session_facade(mock_session_registry):
    """Fixture for session facade."""
    return SessionFacade(mock_session_registry)


def test_err1001_missing_session_id(create_session_handler, mock_servicer_context):
    """Test err1001 missing session id."""
    request = stt_pb2.SessionRequest(session_id="")
    with pytest.raises(grpc.RpcError):
        create_session_handler.handle(request, mock_servicer_context)
    mock_servicer_context.abort.assert_called_with(
        grpc.StatusCode.INVALID_ARGUMENT, "ERR1001 session_id is required"
    )


def test_err1013_rejects_create_session_when_shutting_down(
    mock_session_registry, mock_servicer_context
):
    """Test err1013 rejects CreateSession during shutdown."""
    mock_model_registry = MagicMock()
    mock_model_registry.get_next_model_id.return_value = "default"
    supported_languages = MagicMock()
    supported_languages.is_supported.return_value = True

    config = CreateSessionConfig(
        decode_profiles={"default": {}},
        default_decode_profile="default",
        default_language="en",
        language_fix=False,
        default_task="transcribe",
        supported_languages=supported_languages,
        default_vad_silence=0.5,
        default_vad_threshold=0.5,
        require_api_key=False,
        create_session_rps=0.0,
        create_session_burst=0.0,
        max_sessions_per_ip=0,
        max_sessions_per_api_key=0,
        allow_new_sessions=lambda: False,
    )
    handler = CreateSessionHandler(
        session_registry=mock_session_registry,
        model_registry=mock_model_registry,
        config=config,
    )

    request = stt_pb2.SessionRequest(session_id="ok")
    with pytest.raises(grpc.RpcError):
        handler.handle(request, mock_servicer_context)
    mock_servicer_context.abort.assert_called_with(
        grpc.StatusCode.UNAVAILABLE, "ERR1013 server shutting down"
    )


def test_err1002_duplicate_session_id(
    create_session_handler, mock_session_registry, mock_servicer_context
):
    """Test err1002 duplicate session id."""
    mock_session_registry.create_session.side_effect = ValueError("Duplicate")
    request = stt_pb2.SessionRequest(session_id="dup")
    with pytest.raises(grpc.RpcError):
        create_session_handler.handle(request, mock_servicer_context)
    mock_servicer_context.abort.assert_called_with(
        grpc.StatusCode.ALREADY_EXISTS, "ERR1002 session_id already active"
    )


def test_err1003_negative_vad_threshold(create_session_handler, mock_servicer_context):
    """Test err1003 negative vad threshold."""
    request = stt_pb2.SessionRequest(session_id="ok", vad_threshold=-0.1)
    with pytest.raises(grpc.RpcError):
        create_session_handler.handle(request, mock_servicer_context)
    mock_servicer_context.abort.assert_called_with(
        grpc.StatusCode.INVALID_ARGUMENT, "ERR1003 vad_threshold must be non-negative"
    )


def test_err1009_missing_api_key_when_required(
    create_session_handler, mock_servicer_context
):
    """Test err1009 missing api key when required."""
    request = stt_pb2.SessionRequest(
        session_id="ok", attributes={"api_key_required": "true"}
    )
    with pytest.raises(grpc.RpcError):
        create_session_handler.handle(request, mock_servicer_context)
    mock_servicer_context.abort.assert_called_with(
        grpc.StatusCode.UNAUTHENTICATED, "ERR1009 API key is required"
    )


def test_err1009_missing_api_key_when_config_requires(
    mock_session_registry, mock_servicer_context
):
    """Test err1009 missing api key when config requires."""
    mock_model_registry = MagicMock()
    mock_model_registry.get_next_model_id.return_value = "default"
    supported_languages = MagicMock()
    supported_languages.is_supported.return_value = True

    config = CreateSessionConfig(
        decode_profiles={"default": {}},
        default_decode_profile="default",
        default_language="en",
        language_fix=False,
        default_task="transcribe",
        supported_languages=supported_languages,
        default_vad_silence=0.5,
        default_vad_threshold=0.5,
        require_api_key=True,
        create_session_rps=0.0,
        create_session_burst=0.0,
        max_sessions_per_ip=0,
        max_sessions_per_api_key=0,
    )
    handler = CreateSessionHandler(
        session_registry=mock_session_registry,
        model_registry=mock_model_registry,
        config=config,
    )

    request = stt_pb2.SessionRequest(session_id="ok")
    with pytest.raises(grpc.RpcError):
        handler.handle(request, mock_servicer_context)
    mock_servicer_context.abort.assert_called_with(
        grpc.StatusCode.UNAUTHENTICATED, "ERR1009 API key is required"
    )


def test_err1010_invalid_decode_options(mock_session_registry, mock_servicer_context):
    """Test err1010 invalid decode options."""
    mock_model_registry = MagicMock()
    mock_model_registry.get_next_model_id.return_value = "default"

    supported_languages = MagicMock()
    supported_languages.get_codes.return_value = None

    config = CreateSessionConfig(
        decode_profiles={"default": {"bogus": True}},
        default_decode_profile="default",
        default_language="en",
        language_fix=False,
        default_task="transcribe",
        supported_languages=supported_languages,
        default_vad_silence=0.5,
        default_vad_threshold=0.5,
        require_api_key=False,
        create_session_rps=0.0,
        create_session_burst=0.0,
        max_sessions_per_ip=0,
        max_sessions_per_api_key=0,
    )
    handler = CreateSessionHandler(
        session_registry=mock_session_registry,
        model_registry=mock_model_registry,
        config=config,
    )

    request = stt_pb2.SessionRequest(session_id="ok", vad_threshold_override=0.0)
    with pytest.raises(grpc.RpcError):
        handler.handle(request, mock_servicer_context)
    mock_servicer_context.abort.assert_called_with(
        grpc.StatusCode.INVALID_ARGUMENT,
        "ERR1010 invalid decode option(s): bogus",
    )


def test_create_session_uses_override_threshold_even_when_zero(
    create_session_handler, mock_session_registry, mock_servicer_context
):
    """Test create session uses override threshold even when zero."""
    request = stt_pb2.SessionRequest(session_id="ok", vad_threshold_override=0.0)
    create_session_handler.handle(request, mock_servicer_context)
    args, _kwargs = mock_session_registry.create_session.call_args
    session_info = args[1]
    assert session_info.vad_threshold == 0.0


def _signed_metadata(
    session_id: str, secret: str, timestamp: str, bearer_with_timestamp: bool = False
):
    payload = f"{session_id}:{timestamp}".encode("utf-8")
    signature = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    if bearer_with_timestamp:
        return [("authorization", f"Bearer {timestamp}:{signature}")]
    return [("authorization", f"Bearer {signature}"), ("x-stt-auth-ts", timestamp)]


def test_create_session_signed_token_metadata_valid(
    mock_session_registry, mock_servicer_context
):
    """Accept signed_token metadata when authorization + timestamp provided."""
    mock_model_registry = MagicMock()
    mock_model_registry.get_next_model_id.return_value = "default"
    supported_languages = MagicMock()
    supported_languages.is_supported.return_value = True
    config = CreateSessionConfig(
        decode_profiles={"default": {}},
        default_decode_profile="default",
        default_language="en",
        language_fix=False,
        default_task="transcribe",
        supported_languages=supported_languages,
        default_vad_silence=0.5,
        default_vad_threshold=0.5,
        require_api_key=False,
        create_session_auth_profile="signed_token",
        create_session_auth_secret="secret",
        create_session_auth_ttl_sec=0.0,
        create_session_rps=0.0,
        create_session_burst=0.0,
        max_sessions_per_ip=0,
        max_sessions_per_api_key=0,
    )
    handler = CreateSessionHandler(
        session_registry=mock_session_registry,
        model_registry=mock_model_registry,
        config=config,
    )
    mock_servicer_context.invocation_metadata.return_value = _signed_metadata(
        "ok", "secret", "12345"
    )
    request = stt_pb2.SessionRequest(session_id="ok")
    handler.handle(request, mock_servicer_context)
    mock_session_registry.create_session.assert_called_once()


def test_create_session_signed_token_metadata_accepts_bearer_with_timestamp(
    mock_session_registry, mock_servicer_context
):
    """Accept signed_token metadata when timestamp is embedded in authorization."""
    mock_model_registry = MagicMock()
    mock_model_registry.get_next_model_id.return_value = "default"
    supported_languages = MagicMock()
    supported_languages.is_supported.return_value = True
    config = CreateSessionConfig(
        decode_profiles={"default": {}},
        default_decode_profile="default",
        default_language="en",
        language_fix=False,
        default_task="transcribe",
        supported_languages=supported_languages,
        default_vad_silence=0.5,
        default_vad_threshold=0.5,
        require_api_key=False,
        create_session_auth_profile="signed_token",
        create_session_auth_secret="secret",
        create_session_auth_ttl_sec=0.0,
        create_session_rps=0.0,
        create_session_burst=0.0,
        max_sessions_per_ip=0,
        max_sessions_per_api_key=0,
    )
    handler = CreateSessionHandler(
        session_registry=mock_session_registry,
        model_registry=mock_model_registry,
        config=config,
    )
    mock_servicer_context.invocation_metadata.return_value = _signed_metadata(
        "ok", "secret", "12345", bearer_with_timestamp=True
    )
    request = stt_pb2.SessionRequest(session_id="ok")
    handler.handle(request, mock_servicer_context)
    mock_session_registry.create_session.assert_called_once()


def test_create_session_signed_token_metadata_invalid(
    mock_session_registry, mock_servicer_context
):
    """Reject invalid signed_token metadata."""
    mock_model_registry = MagicMock()
    mock_model_registry.get_next_model_id.return_value = "default"
    supported_languages = MagicMock()
    supported_languages.is_supported.return_value = True
    config = CreateSessionConfig(
        decode_profiles={"default": {}},
        default_decode_profile="default",
        default_language="en",
        language_fix=False,
        default_task="transcribe",
        supported_languages=supported_languages,
        default_vad_silence=0.5,
        default_vad_threshold=0.5,
        require_api_key=False,
        create_session_auth_profile="signed_token",
        create_session_auth_secret="secret",
        create_session_auth_ttl_sec=0.0,
        create_session_rps=0.0,
        create_session_burst=0.0,
        max_sessions_per_ip=0,
        max_sessions_per_api_key=0,
    )
    handler = CreateSessionHandler(
        session_registry=mock_session_registry,
        model_registry=mock_model_registry,
        config=config,
    )
    mock_servicer_context.invocation_metadata.return_value = [
        ("authorization", "Bearer deadbeef"),
        ("x-stt-auth-ts", "12345"),
    ]
    request = stt_pb2.SessionRequest(session_id="ok")
    with pytest.raises(grpc.RpcError):
        handler.handle(request, mock_servicer_context)
    mock_servicer_context.abort.assert_called_with(
        grpc.StatusCode.UNAUTHENTICATED,
        "ERR1014 CreateSession authentication failed",
    )


def test_create_session_skips_vad_reserve_when_token_required(
    mock_session_registry, mock_servicer_context
):
    """Token-required sessions should defer VAD reservation until streaming."""
    mock_model_registry = MagicMock()
    mock_model_registry.get_next_model_id.return_value = "default"
    supported_languages = MagicMock()
    supported_languages.is_supported.return_value = True
    config = CreateSessionConfig(
        decode_profiles={"default": {}},
        default_decode_profile="default",
        default_language="en",
        language_fix=False,
        default_task="transcribe",
        supported_languages=supported_languages,
        default_vad_silence=0.5,
        default_vad_threshold=0.5,
        require_api_key=False,
        create_session_rps=0.0,
        create_session_burst=0.0,
        max_sessions_per_ip=0,
        max_sessions_per_api_key=0,
    )
    handler = CreateSessionHandler(
        session_registry=mock_session_registry,
        model_registry=mock_model_registry,
        config=config,
    )
    request = stt_pb2.SessionRequest(
        session_id="ok", require_token=True, vad_threshold=0.5
    )
    with patch.object(handler._vad_model_pool, "reserve_slot") as reserve_mock:
        handler.handle(request, mock_servicer_context)
    reserve_mock.assert_not_called()
    args, _kwargs = mock_session_registry.create_session.call_args
    session_info = args[1]
    assert session_info.vad_reserved is False


def test_create_session_falls_back_to_default_when_override_unset(
    create_session_handler, mock_session_registry, mock_servicer_context
):
    """Test create session falls back to default when override unset."""
    request = stt_pb2.SessionRequest(session_id="ok", vad_threshold=0.0)
    create_session_handler.handle(request, mock_servicer_context)
    args, _kwargs = mock_session_registry.create_session.call_args
    session_info = args[1]
    assert session_info.vad_threshold == 0.5


def test_vad_pool_expands_when_capacity_exceeded(
    create_session_handler, mock_session_registry, mock_servicer_context
):
    """Test vad pool expands when capacity exceeded."""
    vad_gate.configure_vad_model_pool(
        max_size=2, prewarm=0, max_capacity=4, growth_factor=1.5
    )
    try:
        for idx in range(3):
            request = stt_pb2.SessionRequest(
                session_id=f"sess-{idx}", vad_threshold=0.5
            )
            create_session_handler.handle(request, mock_servicer_context)
    finally:
        vad_gate.configure_vad_model_pool(max_size=0, prewarm=0, max_capacity=0)


def test_vad_pool_exhausted_rejects_session(
    create_session_handler, mock_session_registry, mock_servicer_context
):
    """Test vad pool exhausted rejects session."""
    vad_gate.configure_vad_model_pool(
        max_size=2, prewarm=0, max_capacity=3, growth_factor=1.5
    )
    with pytest.raises(grpc.RpcError):
        for idx in range(4):
            request = stt_pb2.SessionRequest(
                session_id=f"sess-x-{idx}", vad_threshold=0.5
            )
            create_session_handler.handle(request, mock_servicer_context)
    mock_servicer_context.abort.assert_called_with(
        grpc.StatusCode.RESOURCE_EXHAUSTED, "ERR1008 VAD capacity exhausted"
    )
    vad_gate.configure_vad_model_pool(max_size=0, prewarm=0, max_capacity=0)


def test_err1011_session_limit_per_ip():
    """Test err1011 session limit per client IP."""
    session_registry = SessionRegistry()
    mock_model_registry = MagicMock()
    mock_model_registry.get_next_model_id.return_value = "default"
    supported_languages = MagicMock()
    supported_languages.is_supported.return_value = True
    config = CreateSessionConfig(
        decode_profiles={"default": {}},
        default_decode_profile="default",
        default_language="en",
        language_fix=False,
        default_task="transcribe",
        supported_languages=supported_languages,
        default_vad_silence=0.5,
        default_vad_threshold=0.5,
        require_api_key=False,
        create_session_rps=0.0,
        create_session_burst=0.0,
        max_sessions_per_ip=1,
        max_sessions_per_api_key=0,
    )
    handler = CreateSessionHandler(
        session_registry=session_registry,
        model_registry=mock_model_registry,
        config=config,
    )
    context = MagicMock()
    context.abort.side_effect = grpc.RpcError("Aborted")
    context.peer.return_value = "ipv4:127.0.0.1:12345"

    handler.handle(
        stt_pb2.SessionRequest(session_id="s1", vad_threshold_override=0.0),
        context,
    )

    with pytest.raises(grpc.RpcError):
        handler.handle(
            stt_pb2.SessionRequest(session_id="s2", vad_threshold_override=0.0),
            context,
        )
    context.abort.assert_called_with(
        grpc.StatusCode.RESOURCE_EXHAUSTED, "ERR1011 session limit exceeded"
    )


def test_err1012_create_session_rate_limited():
    """Test err1012 create session rate limited."""
    session_registry = SessionRegistry()
    mock_model_registry = MagicMock()
    mock_model_registry.get_next_model_id.return_value = "default"
    supported_languages = MagicMock()
    supported_languages.is_supported.return_value = True
    config = CreateSessionConfig(
        decode_profiles={"default": {}},
        default_decode_profile="default",
        default_language="en",
        language_fix=False,
        default_task="transcribe",
        supported_languages=supported_languages,
        default_vad_silence=0.5,
        default_vad_threshold=0.5,
        require_api_key=False,
        create_session_rps=1.0,
        create_session_burst=1.0,
        max_sessions_per_ip=0,
        max_sessions_per_api_key=0,
    )
    handler = CreateSessionHandler(
        session_registry=session_registry,
        model_registry=mock_model_registry,
        config=config,
    )
    context = MagicMock()
    context.abort.side_effect = grpc.RpcError("Aborted")
    context.peer.return_value = "ipv4:127.0.0.1:2222"

    handler.handle(
        stt_pb2.SessionRequest(session_id="s1", vad_threshold_override=0.0),
        context,
    )

    with pytest.raises(grpc.RpcError):
        handler.handle(
            stt_pb2.SessionRequest(session_id="s2", vad_threshold_override=0.0),
            context,
        )
    context.abort.assert_called_with(
        grpc.StatusCode.RESOURCE_EXHAUSTED, "ERR1012 create session rate limited"
    )


def test_metrics_api_key_sessions_hidden_by_default():
    """Test metrics api key sessions hidden by default."""
    metrics = Metrics()
    metrics.increase_active_sessions("key-1")
    payload = metrics.render()
    assert "active_sessions_by_api" not in payload


def test_metrics_api_key_sessions_exposed_when_enabled():
    """Test metrics api key sessions exposed when enabled."""
    metrics = Metrics()
    metrics.set_expose_api_key_metrics(True)
    metrics.increase_active_sessions("key-1")
    payload = metrics.render()
    assert payload.get("active_sessions_by_api") == {"key-1": 1}


def test_err1004_unknown_session_id(
    session_facade, mock_session_registry, mock_servicer_context
):
    """Test err1004 unknown session id."""
    mock_session_registry.get_session.return_value = None
    with pytest.raises(grpc.RpcError):
        session_facade.resolve_from_metadata(
            {"session-id": "unknown"}, mock_servicer_context
        )
    mock_servicer_context.abort.assert_called_with(
        grpc.StatusCode.UNAUTHENTICATED, "ERR1004 Unknown or missing session_id"
    )


def test_err1005_invalid_token(session_facade, mock_servicer_context):
    """Test err1005 invalid token."""
    session_info = MagicMock()
    session_info.token_required = True
    session_info.token = "secret"

    state = MagicMock()
    state.session_info = session_info
    state.session_id = "sess"

    chunk = stt_pb2.AudioChunk(session_id="sess", session_token="wrong")

    with pytest.raises(grpc.RpcError):
        session_facade.validate_token(state, chunk, mock_servicer_context)
    mock_servicer_context.abort.assert_called_with(
        grpc.StatusCode.PERMISSION_DENIED, "ERR1005 Invalid session token"
    )
