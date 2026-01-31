from unittest.mock import MagicMock, patch

import grpc
import pytest

from gen.stt.python.v1 import stt_pb2
from stt_server.backend.transport.grpc_servicer import STTGrpcServicer
from stt_server.errors import ErrorCode, STTError, format_error, status_for


@pytest.fixture
def mock_servicer_context():
    """Fixture for mock servicer context."""
    context = MagicMock()
    context.abort.side_effect = grpc.RpcError("Aborted")
    return context


@pytest.fixture
def servicer():
    """Fixture for servicer."""
    with patch(
        "stt_server.backend.transport.grpc_servicer.ApplicationRuntime"
    ) as MockRuntime:
        mock_runtime = MockRuntime.return_value
        mock_runtime.metrics.record_error = MagicMock()
        srv = STTGrpcServicer(MagicMock())
        return srv


def test_err2001_decode_timeout(servicer, mock_servicer_context):
    """Test err2001 decode timeout."""
    servicer.runtime.stream_orchestrator.run.side_effect = STTError(
        ErrorCode.DECODE_TIMEOUT
    )

    with pytest.raises(grpc.RpcError):
        list(servicer.StreamingRecognize(iter([]), mock_servicer_context))

    mock_servicer_context.abort.assert_called_with(
        status_for(ErrorCode.DECODE_TIMEOUT),
        format_error(ErrorCode.DECODE_TIMEOUT),
    )


def test_err2002_decode_failed(servicer, mock_servicer_context):
    """Test err2002 decode failed."""
    detail = "decode task failed: model crashed"
    servicer.runtime.stream_orchestrator.run.side_effect = STTError(
        ErrorCode.DECODE_TASK_FAILED, detail
    )

    with pytest.raises(grpc.RpcError):
        list(servicer.StreamingRecognize(iter([]), mock_servicer_context))

    mock_servicer_context.abort.assert_called_with(
        status_for(ErrorCode.DECODE_TASK_FAILED),
        format_error(ErrorCode.DECODE_TASK_FAILED, detail),
    )


def test_err3001_unexpected_create_session(servicer, mock_servicer_context, caplog):
    """Test err3001 unexpected create session."""
    servicer.runtime.create_session_handler.handle.side_effect = RuntimeError(
        "Unexpected boom"
    )
    request = stt_pb2.SessionRequest(session_id="test")

    with pytest.raises(RuntimeError):
        servicer.CreateSession(request, mock_servicer_context)

    servicer.runtime.metrics.record_error.assert_called_with(
        status_for(ErrorCode.CREATE_SESSION_UNEXPECTED)
    )
    assert "ERR3001" in caplog.text


def test_err3002_unexpected_streaming_error(servicer, mock_servicer_context, caplog):
    """Test err3002 unexpected streaming error."""
    servicer.runtime.stream_orchestrator.run.side_effect = RuntimeError(
        "Unexpected streaming boom"
    )

    with pytest.raises(RuntimeError):
        list(servicer.StreamingRecognize(iter([]), mock_servicer_context))

    servicer.runtime.metrics.record_error.assert_called_with(
        status_for(ErrorCode.STREAM_UNEXPECTED)
    )
    assert "ERR3002" in caplog.text
