from concurrent import futures
from unittest.mock import MagicMock, patch

import grpc
import pytest

from stt_server.backend.component.decode_scheduler import (
    DecodeScheduler,
    DecodeSchedulerHooks,
    DecodeStream,
)
from stt_server.errors import ErrorCode, STTError


def test_decode_stream_logic_err2001_timeout():
    hooks = DecodeSchedulerHooks(on_error=MagicMock())
    scheduler = DecodeScheduler(MagicMock(), 0.01, MagicMock(), hooks=hooks)
    stream = DecodeStream(scheduler)

    mock_future = MagicMock(spec=futures.Future)
    mock_future.done.return_value = False
    scheduler._increment_pending()
    stream.pending_partials = 1
    stream.pending_results.append((mock_future, False, 0.0, False, 0.0))

    with patch(
        "stt_server.backend.component.decode_scheduler.futures.wait"
    ) as mock_wait:
        mock_wait.return_value = (set(), {mock_future})

        with pytest.raises(STTError) as exc:
            list(stream.emit_ready(block=True))

    assert exc.value.code == ErrorCode.DECODE_TIMEOUT
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
    stream.pending_results.append((mock_future, False, 0.0, False, 0.0))

    with pytest.raises(STTError) as exc:
        list(stream.emit_ready(block=False))

    assert exc.value.code == ErrorCode.DECODE_TASK_FAILED
    assert "Model crash" in str(exc.value)
    hooks.on_error.assert_called_once_with(grpc.StatusCode.INTERNAL)
    assert scheduler.pending_decodes() == 0
    assert stream.pending_partials == 0
    assert stream.pending_results == []


def test_decode_stream_timing_summary_after_emit_ready():
    hooks = DecodeSchedulerHooks(on_decode_result=MagicMock())
    language_lookup = MagicMock()
    language_lookup.get_name.return_value = "English"
    scheduler = DecodeScheduler(MagicMock(), 0.0, language_lookup, hooks=hooks)
    stream = DecodeStream(scheduler)

    class FakeSegment:
        def __init__(self, text: str, start: float, end: float) -> None:
            self.text = text
            self.start = start
            self.end = end

    class FakeResult:
        def __init__(self) -> None:
            self.segments = [FakeSegment("hello", 0.0, 0.1)]
            self.language_code = "en"
            self.language_probability = 0.9
            self.latency_sec = 0.25
            self.rtf = 0.5
            self.queue_wait_sec = 0.3
            self.audio_duration = 0.1

    fake_result = FakeResult()
    mock_future = MagicMock(spec=futures.Future)
    mock_future.done.return_value = True
    mock_future.result.return_value = fake_result

    buffer_wait_sec = 0.12
    scheduler._increment_pending()
    stream.pending_results.append((mock_future, True, 0.0, False, buffer_wait_sec))

    with patch(
        "stt_server.backend.component.decode_scheduler.time.perf_counter",
        side_effect=[10.0, 10.05],
    ):
        list(stream.emit_ready(block=False))

    (
        buffer_wait_total,
        queue_wait_total,
        inference_total,
        response_emit_total,
        decode_count,
    ) = stream.timing_summary()

    assert buffer_wait_total == pytest.approx(buffer_wait_sec)
    assert queue_wait_total == pytest.approx(fake_result.queue_wait_sec)
    assert inference_total == pytest.approx(fake_result.latency_sec)
    assert response_emit_total == pytest.approx(0.05)
    assert decode_count == 1


def test_decode_stream_drop_pending_partials_updates_counts():
    scheduler = DecodeScheduler(MagicMock(), 0.0, MagicMock())
    stream = DecodeStream(scheduler)

    futures_list = [MagicMock(spec=futures.Future) for _ in range(3)]
    futures_list[0].cancel.return_value = True
    futures_list[1].cancel.return_value = False
    futures_list[2].cancel.return_value = True

    for _ in futures_list:
        scheduler._increment_pending()
    stream.pending_partials = 2
    stream.pending_results.extend(
        [
            (futures_list[0], False, 0.0, False, 0.0),
            (futures_list[1], True, 0.0, False, 0.0),
            (futures_list[2], False, 0.0, False, 0.0),
        ]
    )

    cancelled, orphaned = stream.drop_pending_partials(max_drop=1)

    assert cancelled + orphaned == 1
    assert scheduler.pending_decodes() == 2
    assert stream.pending_partials == 1
    assert len(stream.pending_results) == 2
