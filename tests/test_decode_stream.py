from concurrent import futures
from unittest.mock import MagicMock, patch

import grpc
import pytest

from stt_server.backend.component.decode_scheduler import (
    DecodeScheduler,
    DecodeSchedulerHooks,
    DecodeStream,
)


def test_decode_stream_logic_err2001_timeout():
    hooks = DecodeSchedulerHooks(on_error=MagicMock())
    scheduler = DecodeScheduler(MagicMock(), 0.01, MagicMock(), hooks=hooks)
    stream = DecodeStream(scheduler)

    mock_future = MagicMock(spec=futures.Future)
    mock_future.done.return_value = False
    scheduler._increment_pending()
    stream.pending_partials = 1
    stream.pending_results.append((mock_future, False, 0.0, False))

    with patch(
        "stt_server.backend.component.decode_scheduler.futures.wait"
    ) as mock_wait:
        mock_wait.return_value = (set(), {mock_future})

        with pytest.raises(TimeoutError) as exc:
            list(stream.emit_ready(block=True))

    assert "ERR2001" in str(exc.value)
    hooks.on_error.assert_called_once_with(grpc.StatusCode.INTERNAL)
    assert scheduler.pending_decodes() == 0
    assert stream.pending_partials == 0
    assert stream.pending_results == []


def test_decode_stream_logic_err2002_task_failed():
    hooks = DecodeSchedulerHooks(on_error=MagicMock())
    scheduler = DecodeScheduler(MagicMock(), 0.0, MagicMock(), hooks=hooks)
    stream = DecodeStream(scheduler)

    mock_future = MagicMock(spec=futures.Future)
    mock_future.done.return_value = True
    mock_future.result.side_effect = ValueError("Model crash")
    scheduler._increment_pending()
    stream.pending_partials = 1
    stream.pending_results.append((mock_future, False, 0.0, False))

    with pytest.raises(RuntimeError) as exc:
        list(stream.emit_ready(block=False))

    assert "ERR2002" in str(exc.value)
    assert "Model crash" in str(exc.value)
    hooks.on_error.assert_called_once_with(grpc.StatusCode.INTERNAL)
    assert scheduler.pending_decodes() == 0
    assert stream.pending_partials == 0
    assert stream.pending_results == []
