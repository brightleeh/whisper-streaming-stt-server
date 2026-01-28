from unittest.mock import MagicMock

import grpc
import pytest

from gen.stt.python.v1 import stt_pb2
from stt_server.backend.application.session_manager import (
    CreateSessionHandler,
    SessionFacade,
    SessionRegistry,
)


@pytest.fixture
def mock_servicer_context():
    context = MagicMock()
    context.abort.side_effect = grpc.RpcError("Aborted")
    return context


@pytest.fixture
def mock_session_registry():
    return MagicMock(spec=SessionRegistry)


@pytest.fixture
def create_session_handler(mock_session_registry):
    mock_model_registry = MagicMock()
    mock_model_registry.get_next_model_id.return_value = "default"
    mock_model_registry.load_model.return_value = None

    supported_languages = MagicMock()
    supported_languages.is_supported.return_value = True

    return CreateSessionHandler(
        session_registry=mock_session_registry,
        model_registry=mock_model_registry,
        decode_profiles={"default": {}},
        default_decode_profile="default",
        default_language="en",
        language_fix=False,
        default_task="transcribe",
        supported_languages=supported_languages,
        default_vad_silence=0.5,
        default_vad_threshold=0.5,
    )


@pytest.fixture
def session_facade(mock_session_registry):
    return SessionFacade(mock_session_registry)


def test_err1001_missing_session_id(create_session_handler, mock_servicer_context):
    request = stt_pb2.SessionRequest(session_id="")
    with pytest.raises(grpc.RpcError):
        create_session_handler.handle(request, mock_servicer_context)
    mock_servicer_context.abort.assert_called_with(
        grpc.StatusCode.INVALID_ARGUMENT, "ERR1001 session_id is required"
    )


def test_err1002_duplicate_session_id(
    create_session_handler, mock_session_registry, mock_servicer_context
):
    mock_session_registry.create_session.side_effect = ValueError("Duplicate")
    request = stt_pb2.SessionRequest(session_id="dup")
    with pytest.raises(grpc.RpcError):
        create_session_handler.handle(request, mock_servicer_context)
    mock_servicer_context.abort.assert_called_with(
        grpc.StatusCode.ALREADY_EXISTS, "ERR1002 session_id already active"
    )


def test_err1003_negative_vad_threshold(create_session_handler, mock_servicer_context):
    request = stt_pb2.SessionRequest(session_id="ok", vad_threshold=-0.1)
    with pytest.raises(grpc.RpcError):
        create_session_handler.handle(request, mock_servicer_context)
    mock_servicer_context.abort.assert_called_with(
        grpc.StatusCode.INVALID_ARGUMENT, "ERR1003 vad_threshold must be non-negative"
    )


def test_err1004_unknown_session_id(
    session_facade, mock_session_registry, mock_servicer_context
):
    mock_session_registry.get_session.return_value = None
    with pytest.raises(grpc.RpcError):
        session_facade.resolve_from_metadata(
            {"session-id": "unknown"}, mock_servicer_context
        )
    mock_servicer_context.abort.assert_called_with(
        grpc.StatusCode.UNAUTHENTICATED, "ERR1004 Unknown or missing session_id"
    )


def test_err1005_invalid_token(session_facade, mock_servicer_context):
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
