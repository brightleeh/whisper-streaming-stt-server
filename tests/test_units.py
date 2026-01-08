import os
import sys
from concurrent import futures
from unittest.mock import MagicMock, patch

import grpc
import pytest

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gen.stt.python.v1 import stt_pb2
from stt_server.backend.application.session_registry import (
    CreateSessionHandler,
    SessionFacade,
    SessionRegistry,
)
from stt_server.backend.component.decode_scheduler import DecodeScheduler, DecodeStream
from stt_server.backend.transport.grpc_servicer import STTGrpcServicer


@pytest.fixture
def mock_servicer_context():
    context = MagicMock()
    context.abort.side_effect = grpc.RpcError("Aborted")
    return context


@pytest.fixture
def servicer():
    # Mock the ApplicationRuntime to avoid real initialization
    with patch(
        "stt_server.backend.transport.grpc_servicer.ApplicationRuntime"
    ) as MockRuntime:
        mock_runtime = MockRuntime.return_value
        # Setup default mocks
        mock_runtime.metrics.record_error = MagicMock()
        srv = STTGrpcServicer(MagicMock())
        return srv


@pytest.fixture
def mock_session_registry():
    return MagicMock(spec=SessionRegistry)


@pytest.fixture
def create_session_handler(mock_session_registry):
    supported_languages = MagicMock()
    supported_languages.is_supported.return_value = True
    return CreateSessionHandler(
        session_registry=mock_session_registry,
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
    """Unit test for ERR1001: Missing session_id."""
    request = stt_pb2.SessionRequest(session_id="")
    with pytest.raises(grpc.RpcError):
        create_session_handler.handle(request, mock_servicer_context)
    mock_servicer_context.abort.assert_called_with(
        grpc.StatusCode.INVALID_ARGUMENT, "ERR1001 session_id is required"
    )


def test_err1002_duplicate_session_id(
    create_session_handler, mock_session_registry, mock_servicer_context
):
    """Unit test for ERR1002: Duplicate session_id."""
    mock_session_registry.create_session.side_effect = ValueError("Duplicate")
    request = stt_pb2.SessionRequest(session_id="dup")
    with pytest.raises(grpc.RpcError):
        create_session_handler.handle(request, mock_servicer_context)
    mock_servicer_context.abort.assert_called_with(
        grpc.StatusCode.ALREADY_EXISTS, "ERR1002 session_id already active"
    )


def test_err1003_negative_vad_threshold(create_session_handler, mock_servicer_context):
    """Unit test for ERR1003: Negative vad_threshold."""
    request = stt_pb2.SessionRequest(session_id="ok", vad_threshold=-0.1)
    with pytest.raises(grpc.RpcError):
        create_session_handler.handle(request, mock_servicer_context)
    mock_servicer_context.abort.assert_called_with(
        grpc.StatusCode.INVALID_ARGUMENT, "ERR1003 vad_threshold must be non-negative"
    )


def test_err1004_unknown_session_id(
    session_facade, mock_session_registry, mock_servicer_context
):
    """Unit test for ERR1004: Unknown session_id."""
    mock_session_registry.get_session.return_value = None
    with pytest.raises(grpc.RpcError):
        session_facade.resolve_from_metadata(
            {"session-id": "unknown"}, mock_servicer_context
        )
    mock_servicer_context.abort.assert_called_with(
        grpc.StatusCode.UNAUTHENTICATED, "ERR1004 Unknown or missing session_id"
    )


def test_err1005_invalid_token(session_facade, mock_servicer_context):
    """Unit test for ERR1005: Invalid session token."""
    # Setup state with token requirement
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


def test_err2001_decode_timeout(servicer, mock_servicer_context):
    """Unit test for ERR2001: Decode timeout."""
    # Simulate TimeoutError from stream_orchestrator
    servicer.stream_orchestrator.run.side_effect = TimeoutError("Simulated timeout")

    with pytest.raises(grpc.RpcError):
        list(servicer.StreamingRecognize(iter([]), mock_servicer_context))

    mock_servicer_context.abort.assert_called_with(
        grpc.StatusCode.INTERNAL,
        "ERR2001 (INTERNAL): decode timeout waiting for pending tasks",
    )


def test_decode_stream_logic_err2001_timeout():
    """Test that DecodeStream actually raises ERR2001 on timeout."""
    scheduler = MagicMock(spec=DecodeScheduler)
    scheduler.decode_timeout_sec = 0.01
    stream = DecodeStream(scheduler)

    # Add a fake pending future
    mock_future = MagicMock(spec=futures.Future)
    mock_future.done.return_value = False
    stream.pending_results.append((mock_future, False, 0.0, False))

    # Mock futures.wait to return (done={}, not_done={mock_future}) to simulate timeout
    with patch(
        "stt_server.backend.component.decode_scheduler.futures.wait"
    ) as mock_wait:
        mock_wait.return_value = (set(), {mock_future})

        with pytest.raises(TimeoutError) as exc:
            # Must consume generator to trigger logic
            list(stream.emit_ready(block=True))

    assert "ERR2001" in str(exc.value)


def test_err2002_decode_failed(servicer, mock_servicer_context):
    """Unit test for ERR2002: Decode task failed."""
    # Simulate generic Exception with specific message from stream_orchestrator
    servicer.stream_orchestrator.run.side_effect = RuntimeError(
        "Something went wrong: Decode task failed"
    )

    with pytest.raises(grpc.RpcError):
        list(servicer.StreamingRecognize(iter([]), mock_servicer_context))

    mock_servicer_context.abort.assert_called_with(
        grpc.StatusCode.INTERNAL, "ERR2002 (INTERNAL): decode task failed"
    )


def test_decode_stream_logic_err2002_task_failed():
    """Test that DecodeStream actually raises ERR2002 on task failure."""
    scheduler = MagicMock(spec=DecodeScheduler)
    stream = DecodeStream(scheduler)

    # Add a fake completed future that raises an exception
    mock_future = MagicMock(spec=futures.Future)
    mock_future.done.return_value = True
    mock_future.result.side_effect = ValueError("Model crash")
    stream.pending_results.append((mock_future, False, 0.0, False))

    with pytest.raises(RuntimeError) as exc:
        list(stream.emit_ready(block=False))

    assert "ERR2002" in str(exc.value)
    assert "Model crash" in str(exc.value)


def test_err3001_unexpected_create_session(servicer, mock_servicer_context, caplog):
    """Unit test for ERR3001: Unexpected CreateSession error."""
    # Simulate unexpected exception from create_session_handler
    servicer.create_session_handler.handle.side_effect = RuntimeError("Unexpected boom")
    request = stt_pb2.SessionRequest(session_id="test")

    with pytest.raises(
        RuntimeError
    ):  # The servicer re-raises unexpected exceptions after logging
        servicer.CreateSession(request, mock_servicer_context)

    # Verify it recorded the error as UNKNOWN
    servicer._error_recorder.assert_called_with(grpc.StatusCode.UNKNOWN)
    # Verify it logged the error code
    assert "ERR3001" in caplog.text


def test_err3002_unexpected_streaming_error(servicer, mock_servicer_context, caplog):
    """Unit test for ERR3002: Unexpected StreamingRecognize error."""
    # Simulate generic unexpected exception
    servicer.stream_orchestrator.run.side_effect = RuntimeError(
        "Unexpected streaming boom"
    )

    with pytest.raises(RuntimeError):  # The servicer re-raises unexpected exceptions
        list(servicer.StreamingRecognize(iter([]), mock_servicer_context))

    # Verify it recorded the error as UNKNOWN
    servicer._error_recorder.assert_called_with(grpc.StatusCode.UNKNOWN)
    # Verify it logged the error code
    assert "ERR3002" in caplog.text
