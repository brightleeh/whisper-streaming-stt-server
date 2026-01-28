from unittest.mock import MagicMock, patch

import grpc
import pytest

from gen.stt.python.v1 import stt_pb2
from stt_server.backend.transport.grpc_servicer import STTGrpcServicer


@pytest.fixture
def mock_servicer_context():
    context = MagicMock()
    context.abort.side_effect = grpc.RpcError("Aborted")
    return context


@pytest.fixture
def servicer():
    with patch(
        "stt_server.backend.transport.grpc_servicer.ApplicationRuntime"
    ) as MockRuntime:
        mock_runtime = MockRuntime.return_value
        mock_runtime.metrics.record_error = MagicMock()
        srv = STTGrpcServicer(MagicMock())
        return srv


def test_err2001_decode_timeout(servicer, mock_servicer_context):
    servicer.runtime.stream_orchestrator.run.side_effect = TimeoutError(
        "Simulated timeout"
    )

    with pytest.raises(grpc.RpcError):
        list(servicer.StreamingRecognize(iter([]), mock_servicer_context))

    mock_servicer_context.abort.assert_called_with(
        grpc.StatusCode.INTERNAL,
        "ERR2001 (INTERNAL): decode timeout waiting for pending tasks",
    )


def test_err2002_decode_failed(servicer, mock_servicer_context):
    servicer.runtime.stream_orchestrator.run.side_effect = RuntimeError(
        "Something went wrong: Decode task failed"
    )

    with pytest.raises(grpc.RpcError):
        list(servicer.StreamingRecognize(iter([]), mock_servicer_context))

    mock_servicer_context.abort.assert_called_with(
        grpc.StatusCode.INTERNAL, "ERR2002 (INTERNAL): decode task failed"
    )


def test_err3001_unexpected_create_session(servicer, mock_servicer_context, caplog):
    servicer.runtime.create_session_handler.handle.side_effect = RuntimeError(
        "Unexpected boom"
    )
    request = stt_pb2.SessionRequest(session_id="test")

    with pytest.raises(RuntimeError):
        servicer.CreateSession(request, mock_servicer_context)

    servicer.runtime.metrics.record_error.assert_called_with(grpc.StatusCode.UNKNOWN)
    assert "ERR3001" in caplog.text


def test_err3002_unexpected_streaming_error(servicer, mock_servicer_context, caplog):
    servicer.runtime.stream_orchestrator.run.side_effect = RuntimeError(
        "Unexpected streaming boom"
    )

    with pytest.raises(RuntimeError):
        list(servicer.StreamingRecognize(iter([]), mock_servicer_context))

    servicer.runtime.metrics.record_error.assert_called_with(grpc.StatusCode.UNKNOWN)
    assert "ERR3002" in caplog.text
